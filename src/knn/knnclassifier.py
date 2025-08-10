from typing import Any
from .distance import DistanceStrategy
from .voting import VotingStrategy
import pandas as pd
from typing import Self
from sklearn.metrics import confusion_matrix, classification_report
from numpy import ndarray

class KNNClassifer:
    def __init__(self):
        self.k: int = 3
    def set_k(self,k: int) -> Self:
        self.k = k
        return self
    def set_distance_strategy(self, dis_strat: DistanceStrategy) -> Self:
        self.distance_strategy = dis_strat
        return self
    def set_voter(self, voter : VotingStrategy) -> Self:
        self.voter = voter
        return self
    def set_X_train(self,df: pd.DataFrame) -> Self:
        self.X_train = df
        return self
    def set_y_train(self,labels: pd.Series) -> Self:
        self.y_train = labels
        return self
    def set_target_name(self,name: str) -> Self:
        self.target_name = name
        return self
    def classify(self, X_test: pd.DataFrame, y_test: pd.DataFrame ):
        if self.X_train is None:
            return
        distance : pd.DataFrame = X_test.apply(lambda test_row: self.distance_strategy.computeDistance(self.X_train,test_row), axis = 1)
        y_pred: pd.Series = distance.apply(lambda distance_row: self.voter.getLabel(distance_row,self.y_train,self.k), axis = 1)
        y_pred.name = "Predicted"
        X_test = pd.concat([X_test,y_test, y_pred], axis = 1)
        self.cm =  confusion_matrix(y_test,y_pred)
        print(self.cm)
        print(classification_report(y_true=y_test, y_pred=y_pred))