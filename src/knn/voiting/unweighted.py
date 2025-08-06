from pandas import Series
from .strategy import VotingStrategy


class UnweightedVoting(VotingStrategy):
    """
    Class description:
    UnweightedVoting approach that doesn't consider the magnitude of distance, just the order of distance
    """

    def getRank(self, distance: Series) -> Series:
        """Method Description

        Args:
            distance (Series): The distance between a target point and the whole dataframe, represented as series with the index being the row numbers

        Returns:
            Series: A series with the index being the row number and the rank being ranked using the min method for tie resolution. i.e if the distances are 15,16,16,17 then the rankss would be 1,2,2,3
        """
        return distance.rank(method="min")
