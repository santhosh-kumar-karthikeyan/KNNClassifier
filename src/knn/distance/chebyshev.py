from pandas import DataFrame, Series
from .strategy import DistanceStrategy
from numpy import max, abs

class ChebyshevDistance(DistanceStrategy):
    def computeDistance(self, df: DataFrame, s: Series) -> Series:
        return max(abs(df - s))