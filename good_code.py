from abc import ABC, abstractmethod
from typing import List
import logging

import pandas as pd
import numpy as np
from sklearn.base import TransformerMixin
from sklearn.preprocessing import LabelEncoder

logging.basicConfig(level=logging.INFO)


class FeatureProcessor(ABC):
    def __init__(self, feature_names: List[str]) -> None:
        self.feature_names = feature_names

    @abstractmethod
    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        pass


class Normalizer(FeatureProcessor):   
    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        features = df[self.feature_names]
        std = np.std(features)
        mean = np.mean(features)
        return (features - mean) / std


class Standardizer(FeatureProcessor):
    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        features = df[self.feature_names]
        minimum = features.min()
        maximum = features.max()
        return (features - minimum) / (maximum - minimum)


class Encoder(FeatureProcessor):

    def __init__(self, encoder: TransformerMixin, feature_names: List[str]) -> None:
        self.encoder = encoder
        super().__init__(feature_names)

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        features = df[self.feature_names]
        array = self.encoder.fit_transform(features)
        array = np.atleast_2d(array) # Transform array into 2D from 1D or 2D arrays
        processed_df = pd.DataFrame({name: data for name, data in zip(features.columns, array)})
        return processed_df
    

class NaFiller(FeatureProcessor):

    def __init__(self, feature_names: List[str], value: int = -1) -> None:
        self.value = value
        super().__init__(feature_names)

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        features = df[self.feature_names]
        return features.fillna(value=self.value)


class DataLoader(ABC):
    @abstractmethod   
    def load_data(self, path: str) -> pd.DataFrame:
        pass


class ParquetDataLoader(DataLoader):
    def load_data(self, path: str) -> pd.DataFrame:
        return pd.read_parquet(path)
    

class DataSaver(ABC):
    @abstractmethod
    def save_data(self, df: pd.DataFrame, path: str) -> None:
        pass


class ParquetDataSaver(DataSaver):
    def save_data(self, df: pd.DataFrame, path: str) -> None:
        df.to_parquet(path)


class DataProcessor(ABC):
    @abstractmethod
    def process(self, path: str, output_path: str) -> None:
        pass


class ExampleDataProcessor(DataProcessor):
    def __init__(
        self,
        feature_processors: List[FeatureProcessor],
        data_loader: DataLoader,
        data_saver: DataSaver
    ) -> None:
        self.feature_processors = feature_processors
        self.data_loader = data_loader
        self.data_saver = data_saver     

    def process(self, path: str, output_path: str) -> None:
        df = self.data_loader.load_data(path)
        processed_df = pd.concat(
            [feature_processor.process(df) for feature_processor in self.feature_processors],
            axis=1
        )
        self.data_saver.save_data(df=processed_df, path=output_path)
        logging.info(f"Processed df: {processed_df}")


if __name__ == "__main__":
    processor = ExampleDataProcessor(
        feature_processors=[
            Normalizer(feature_names=["feature_a"]),
            Encoder(encoder=LabelEncoder(), feature_names=["feature_b"]),
            NaFiller(feature_names=["feature_c"], value=5)
        ],
        data_loader=ParquetDataLoader(),
        data_saver=ParquetDataSaver()
    )
    processor.process(
        path="data/data.parquet", 
        output_path="data/preprocessed_data.parquet"
    )