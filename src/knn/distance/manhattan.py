from pandas import DataFrame, Series
from .strategy import DistanceStrategy

class ManhattanDistance(DistanceStrategy):
    def computeDistance(self, df: DataFrame, s: Series) -> Series:
        return (df - s).sum(axis=1)