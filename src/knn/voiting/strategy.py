from abc import ABC, abstractmethod
from pandas import DataFrame, Series


class VotingStrategy(ABC):
    """Class description:
    Strategy class that is abstract and expects a concrete strategy, either
    weighted or unweighted voting to be implemented in the following getRank()
    method
    """

    @abstractmethod
    def getLabel(self, distance: Series, labels: Series, k: int) -> str | int:
        """
        Function description:
        Method that based on the approach, will provide the label of a target point from a dataframe given the distance between the class of points and the target point.
        """
        raise NotImplementedError()
