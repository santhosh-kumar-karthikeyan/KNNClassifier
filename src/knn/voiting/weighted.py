from pandas import Series
from .strategy import VotingStrategy

class WeightedVoting(VotingStrategy):
    """Class description

    Args:
        VotingStrategy (VotingStrategy): Concrete class of VotinGStrategy class that provides the impelementation of getRank() method.
    """
    def getLabel(self, distance: Series, labels: Series, k: int) -> str:
        """Method description

        Args:
            distance (Series): The distance between a datafram of points and a single target point
            labels (Series): The target lables of every poitn in the dataframe
            k (int) : Number of points to be considered for voting

        Returns:
            str: Returms tje ;ane; of the target datapoint
        """