from pandas import Series
from numpy import ndarray,array
from .strategy import VotingStrategy


class UnweightedVoting(VotingStrategy):
    """
    Class description:
    UnweightedVoting approach that doesn't consider the magnitude of distance, just the order of distance
    """

    def getLabel(self, distance: Series, labels: Series, k: int) -> str | int:
        """Method description

        Args:
            distance (Series): The distance between a dataframe and the target point, any distance metric followed
            labels (Series): The labels of every point in the dataframe, respectively matched with distance through index
            k (int): The number of entries to be considered for label estimation

        Returns:
            str | int: A label based on the voting strategy
        """
        ranks: Series = distance.rank(method = "min")
        top_k_indices: ndarray = ranks[:k].index.to_numpy()
        top_k_labels : Series = labels[top_k_indices]
        target_label: int = top_k_labels.value_counts()[0]
        return target_label
