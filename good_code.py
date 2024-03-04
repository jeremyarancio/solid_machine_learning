from abc import ABC, abstractmethod
from typing import List

import pandas as pd
import numpy as np
from sklearn.base import TransformerMixin


class FeatureProcessor(ABC):
    @abstractmethod
    def process(feature: pd.Series) -> pd.Series:
        pass


class Normalizer(FeatureProcessor):
    def process(feature: pd.Series) -> pd.Series:
        std = np.std(feature)
        mean = np.mean(feature)
        return (feature - mean) / std


class Standardizer(FeatureProcessor):
    def process(feature: pd.Series) -> pd.Series:
        minimum = feature.min()
        maximum = feature.max()
        return (feature - minimum) / (maximum - minimum)


class Encoder(FeatureProcessor):
    def __init__(self, encoder: TransformerMixin) -> None:
        self.encoder = encoder

    def process(self, feature: pd.Series) -> pd.Series:
        array = self.encoder.fit_transform(feature)
        return pd.Series(array, name=feature.name)
    

class NaFiller(FeatureProcessor):
    def process(self, feature: pd.Series, value: int = -1) -> pd.Series:
        return feature.fillna(value=value)


class DataLoader(ABC):
    @abstractmethod
    def load_data(path: str) -> pd.DataFrame:
        pass


class ParquetDataLoader(DataLoader):
    def load_data(path: str) -> pd.DataFrame:
        return pd.read_parquet(path)
    

class DataSaver(ABC):
    @abstractmethod
    def save(df: pd.DataFrame, path: str) -> None:
        pass


class ParquetDataSaver(DataSaver):
    def save(df: pd.DataFrame, path: str) -> None:
        df.to_parquet(path)


class DataProcessor(ABC):
    @abstractmethod
    def process(path: str, output_path: str) -> None:
        pass


class ExampleDataProcessor(DataProcessor):

    def __init__(
        self,
        numerical_processor: FeatureProcessor,
        categorical_processor: FeatureProcessor,
        na_filler: NaFiller,
        data_loader: DataLoader, #TODO
        data_saver: DataSaver #TODO
    ) -> None:
        self.numerical_processor = numerical_processor
        self.categorical_processor = categorical_processor
        self.na_filler = na_filler
        self.data_loader = data_loader
        self.data_saver = data_saver     

    def process(self, path: str, output_path: str) -> None:
        df = self.data_loader.load_data(path) # TODO
        normalized_feature = self.numerical_processor.process(df["feature_a"])
        encoded_feature = self.categorical_processor.process(df["feature_b"])
        filled_feature = self.na_filler.process(df["feature_c"])
        processed_df = compose_df(
            normalized_feature, 
            encoded_feature, 
            filled_feature,
            column_names=df.columns
        )
        self.data_saver.save(df=processed_df, path=output_path) #TODO


def compose_df(*args, column_names: List[str]) -> pd.DataFrame:
    data = {column_name: series for column_name, series in zip(column_names, args) }
    return pd.DataFrame(data)

if __name__ == "__main__":
    processor