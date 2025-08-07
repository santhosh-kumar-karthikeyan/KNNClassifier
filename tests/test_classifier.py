from knn.knnclassifier import KNNClassifer
from knn.distance import DistanceStrategy,EuclideanDistance, ManhattanDistance, ChebyshevDistance
from knn.voiting import VotingStrategy, UnweightedVoting, WeightedVoting
import pandas as pd
from sklearn.model_selection import train_test_split

df: pd.DataFrame = pd.read_csv("tests/diabetes.csv");
labels: pd.Series = df["Outcome"]
df = df.loc[:, df.columns != "Outcome"]
X_train, X_test, y_train, y_test = train_test_split(df,labels,test_size=0.33,random_state=42)

k:int = 3
def test_classifier():
    knn: KNNClassifer = KNNClassifer()
    knn.set_distance_strategy(ChebyshevDistance()).set_k(50).set_labels(y_train).set_voter(UnweightedVoting()).set_dataframe(df)
    knn.classify(X_train)