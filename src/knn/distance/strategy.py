import pandas as pd
from abc import ABC, abstractmethod

class DistanceStrategy(ABC):
    @abstractmethod
    def computeDistance(self,df: pd.DataFrame,s: pd.Series) -> pd.Series:
        raise NotImplementedError()