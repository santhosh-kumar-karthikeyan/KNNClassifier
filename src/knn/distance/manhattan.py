from pandas import DataFrame, Series
from .strategy import DistanceStrategy
from numpy import abs


class ManhattanDistance(DistanceStrategy):
    def computeDistance(self, df: DataFrame, s: Series) -> Series:
        return (abs(df - s)).sum(axis=1)
