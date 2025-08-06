import pandas as pd
from src.knn.distance import (
    DistanceStrategy,
    EuclideanDistance,
    ManhattanDistance,
    ChebyshevDistance,
)
from pandas import DataFrame, Series

df: DataFrame = DataFrame(
    {"Column1": [1, 4, 7], "Column2": [2, 5, 8], "Column3": [3, 6, 9]}
)
s: Series = Series({"Column1": 4, "Column2": 5, "Column3": 6})


def test_euclidean():
    euclidean: DistanceStrategy = EuclideanDistance()
    expected_s: Series = Series({0: 5.1961, 1: 0, 2: 5.1961})
    actual_s: Series = euclidean.computeDistance(df, s)
    try:
        pd.testing.assert_series_equal(expected_s, actual_s, atol=1e-2)
    except AssertionError as e:
        print(f"Expected output:\n{expected_s}")
        print(f"Actual output:\n{actual_s}")
        raise e


def test_manhattan():
    manhattan: DistanceStrategy = ManhattanDistance()
    expected_s: Series = Series({0: 9, 1: 0, 2: 9})
    actual_s: Series = manhattan.computeDistance(df, s)
    try:
        pd.testing.assert_series_equal(expected_s, actual_s)
    except AssertionError as e:
        print(f"Expected output:\n{expected_s}")
        print(f"Actual output:\n{actual_s}")
        raise e


def test_shebyshev():
    chebyshev: DistanceStrategy = ChebyshevDistance()
    expected_s: Series = Series({0: 3, 1: 0, 2: 3})
    actual_s: Series = chebyshev.computeDistance(df, s)
    try:
        pd.testing.assert_series_equal(expected_s, actual_s)
        print("Shebysev distance intact.")
    except AssertionError as e:
        print(f"Expected output:\n{expected_s}")
        print(f"Actual output:\n{actual_s}")
        raise e
