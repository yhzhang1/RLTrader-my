import pandas as pd
import os

from typing import Tuple
from lib.data.providers.dates import ProviderDateFormat
from lib.data.providers import BaseDataProvider


class VbDataProvider(BaseDataProvider):
    _current_index = 0

    def __init__(self,
                 date_format: ProviderDateFormat = ProviderDateFormat.TIMESTAMP_MS,
                 data_frame: pd.DataFrame = None,
                 csv_data_path: str = None,
                 skip_prepare_data: bool = False, 
                 variety: str = 'RB', **kwargs):
        BaseDataProvider.__init__(self, date_format, **kwargs)

        self.kwargs = kwargs

        if data_frame is not None:
            self.data_frame = data_frame
        elif variety is not None:
            self.data_frame = self.process_data(variety)
        elif csv_data_path is not None:
            if not os.path.isfile(csv_data_path):
                raise ValueError(
                    'Invalid "csv_data_path" argument passed to StaticDataProvider, file could not be found.')

            self.data_frame = pd.read_csv(csv_data_path)
        else:
            raise ValueError(
                'StaticDataProvider requires either a "data_frame" or "csv_data_path argument".')

        if not skip_prepare_data:
            self.data_frame = self.prepare_data(self.data_frame)


    @staticmethod
    def from_prepared(data_frame: pd.DataFrame, date_format: ProviderDateFormat, **kwargs):
        return VbDataProvider(date_format=date_format, data_frame=data_frame, skip_prepare_data=True, **kwargs)

    def split_data_train_test(self, train_split_percentage: float = 0.8) -> Tuple[BaseDataProvider, BaseDataProvider]:
        train_len = int(train_split_percentage * len(self.data_frame))

        train_df = self.data_frame[:train_len].copy()
        test_df = self.data_frame[train_len:].copy()

        train_provider = VbDataProvider.from_prepared(
            data_frame=train_df, date_format=self.date_format, **self.kwargs)
        test_provider = VbDataProvider.from_prepared(
            data_frame=test_df, date_format=self.date_format, **self.kwargs)

        return train_provider, test_provider

    def historical_ohlcv(self) -> pd.DataFrame:
        return self.data_frame

    def has_next_ohlcv(self) -> bool:
        return self._current_index < len(self.data_frame)

    def reset_ohlcv_index(self) -> int:
        self._current_index = 0

    def next_ohlcv(self) -> pd.DataFrame:
        frame = self.data_frame[self.columns].values[self._current_index]
        frame = pd.DataFrame([frame], columns=self.columns)

        self._current_index += 1

        return frame
    # edited
    def process_data(self, variety: str) -> pd.DataFrame:
        factor_store = pd.HDFStore('/home/data/vb/training_x_150.h5', mode='r')
        factor_df = factor_store.get(variety)
        helper_df = pd.read_parquet('/home/data/vb/training_helper_150_{}.parquet'.format(variety))
        df = helper_df.join(factor_df, how='inner')
        df.insert(loc=0, column='Date', value=df.index)
        pd.options.mode.use_inf_as_na = True
        df.dropna(inplace=True)
        
        factor_store.close()
        
        return df