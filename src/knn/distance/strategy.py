import pandas as pd
from abc import ABC, abstractmethod

class DistanceStrategy(ABC):
    @abstractmethod
    def computeDistance(self,a: pd.DataFrame,b: pd.Series) -> pd.DataFrame:
        raise NotImplementedError()