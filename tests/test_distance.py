import pandas as pd
from src.knn.distance import DistanceStrategy, EuclideanDistance, ManhattanDistance, ChebyshevDistance
from pandas import DataFrame, read_csv, Series
from numpy import sqrt,sum

df: DataFrame = DataFrame(
    {
        "Column1" : [1,4,7],
        "Column2" : [2,5,8],
        "Column3" : [3,6,9]
    }
)
s: Series = Series({
    "Column1" : 4,
    "Column2" : 5,
    "Column3" : 6
})


def test_euclidean():
    eu: DistanceStrategy = EuclideanDistance() 
    expected_s : Series = Series({
        0 : 5.1961,
        1 : 0,
        2 : 5.1961
    })
    actual_s: Series = eu.computeDistance(df,s)
    pd.testing.assert_series_equal(expected_s,actual_s,atol=1e-2)