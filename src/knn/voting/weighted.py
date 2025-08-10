from pandas import Series
from numpy import sum
from .strategy import VotingStrategy


class WeightedVoting(VotingStrategy):
    """Class description

    Args:
        VotingStrategy (VotingStrategy): Concrete class of VotinGStrategy class that provides the impelementation of getRank() method.
    """

    def getLabel(self, distance: Series, labels: Series, k: int) -> str | int:
        """Method description

        Args:
            distance (Series): The distance between a datafram of points and a single target point
            labels (Series): The target lables of every poitn in the dataframe
            k (int) : Number of points to be considered for voting

        Returns:
            str | int: Returms the label of the target datapoint
        """
        top_k_indices = distance.nsmallest(k).index.to_numpy()
        weight: Series = distance[top_k_indices].pow(-2)
        cumulative_weigths: Series = weight.groupby(labels).apply(sum)
        target_label: str | int = cumulative_weigths.idxmax()
        return target_label
