import pandas as pd
import sys
from sklearn.model_selection import train_test_split

class DatasetHandler:
    def __init__(self):
        pass
    def read_from_stdin(self) -> pd.DataFrame:
        df = pd.read_csv(sys.stdin)
        return df

    def read_from_path(self, path: str) -> pd.DataFrame:
        df = pd.read_csv(path)
        return df

    def set_target_label(self, label: str) -> None:
        self.target_label = label

    def split_df(self, df: pd.DataFrame, features: list[str],target_label: str, test_size: float):
        y = df[target_label]
        X = df[df.columns.difference([target_label])]
        if len(features) != 0:
            X = X[features]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X,y,test_size = test_size)