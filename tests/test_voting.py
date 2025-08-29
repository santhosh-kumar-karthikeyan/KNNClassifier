from knn.voting import VotingStrategy, UnweightedVoting, WeightedVoting
from pandas import Series

distance: Series = Series([0.2, 0.4, 0.5, 0.8, 0.9, 1, 0.95, 1.92])

labels: Series = Series([1, 0, 0, 0, 1, 0, 0, 1])
k: int = 5


def test_unweighted():
    voter: VotingStrategy = UnweightedVoting()
    expected_label: int = 0
    actual_label = voter.getLabel(distance, labels, k)
    try:
        assert expected_label == actual_label
        print("Unweighted voting intact.")
    except AssertionError as e:
        print(f"Expected {expected_label}, got {actual_label}")
        raise e


def test_weighted():
    voter: VotingStrategy = WeightedVoting()
    expected_label: int = 1
    actual_label = voter.getLabel(distance, labels, k)
    try:
        assert expected_label == actual_label
        print("Weighted voting intact.")
    except AssertionError as e:
        print(f"Expected {expected_label}, got {actual_label}")
        raise e
