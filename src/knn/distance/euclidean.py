from pandas import DataFrame,Series
from .strategy import DistanceStrategy

class EuclideanDistance(DistanceStrategy):
    def computeDistance(self, df: DataFrame, s: Series) -> Series:
        return ((df - s) ** 2).sum(axis=1)