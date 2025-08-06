from pandas import DataFrame,Series
import pandas as pd
from numpy import abs
from .strategy import DistanceStrategy

class EuclideanDistance(DistanceStrategy):
    def computeDistance(self, df: DataFrame, s: Series) -> Series:
        return (((abs(df - s)) ** 2).sum(axis=1))**(1/2)