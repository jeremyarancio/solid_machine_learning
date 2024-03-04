from abc import ABC, abstractmethod

import pandas as pd
import numpy as np
from sklearn.base import TransformerMixin


class Processor(ABC):
    @abstractmethod
    def process(feature: pd.Series, *args, **kwargs) -> pd.Series:
        pass


class Normalizer(Processor):
    def process(feature: pd.Series) -> pd.Series:
        std = np.std(feature)
        mean = np.mean(feature)
        return (feature - mean) / std


class Standardizer(Processor):
    def process(feature: pd.Series) -> pd.Series:
        minimum = feature.min()
        maximum = feature.max()
        return (feature - minimum) / (maximum - minimum)


class Encoder(Processor):

    def __init__(self, encoder: TransformerMixin) -> None:
        self.encoder = encoder

    def process(self, feature: pd.Series) -> pd.Series:
        array = self.encoder.fit_transform(feature)
        return pd.Series(array, name=feature.name)
    

class NaFiller(Processor):

    def process(self, feature: pd.Series, value: int = -1) -> pd.Series:
        return feature.fillna(value=value)
