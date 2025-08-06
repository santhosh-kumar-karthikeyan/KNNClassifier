from abc import ABC, abstractmethod
from pandas import DataFrame, Series


class VotingStrategy(ABC):
    """Class description:
    Strategy class that is abstract and expects a concrete strategy, either
    weighted or unweighted voting to be implemented in the following getRank()
    method
    """

    @abstractmethod
    def getRank(self, distance: Series) -> Series:
        """
        Function description:
        Method that based on the approach, will calculate the rank for a series of distance provided.
        """
        raise NotImplementedError()
