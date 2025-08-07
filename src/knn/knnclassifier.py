from typing import Any
from .distance import DistanceStrategy
from .voiting import VotingStrategy
import pandas as pd
from typing import Self
from sklearn.metrics import confusion_matrix
from numpy import ndarray

class KNNClassifer:
    def __init__(self):
        self.confusion_matrix: ndarray
        pass
    def set_k(self,k: int) -> Self:
        self.k = k
        return self
    def set_distance_strategy(self, dis_strat: DistanceStrategy) -> Self:
        self.distance_strategy = dis_strat
        return self
    def set_voter(self, voter : VotingStrategy) -> Self:
        self.voter = voter
        return self
    def set_dataframe(self,df: pd.DataFrame) -> Self:
        self.df = df
        return self
    def set_labels(self,labels: pd.Series) -> Self:
        self.labels = labels
        return self
    def classify(self, test: pd.DataFrame):
        if self.df is None:
            return
        distance : pd.DataFrame = test.apply(lambda test_row: self.distance_strategy.computeDistance(self.df,test_row), axis = 1)
        label_frame: pd.Series = distance.apply(lambda distance_row: self.voter.getLabel(distance_row,self.labels,self.k))
        print(label_frame.value_counts())
        