import pandas as pd
import sys
from sklearn.model_selection import train_test_split

class DatasetHandler:
    INLINE = 0
    FILE = 1
    def __init__(self):
        self.type: int = DatasetHandler.FILE
    def read_from_stdin(self) -> pd.DataFrame:
        df = pd.read_csv(sys.stdin)
        self.type = DatasetHandler.INLINE
        return df

    def read_from_path(self, path: str) -> pd.DataFrame:
        df = pd.read_csv(path)
        self.type = DatasetHandler.FILE
        return df
    
    def read_test_data_from_stdin(self, features: list[str]) -> pd.DataFrame:
        """Read test data through stdin by prompting for each feature"""
        print("Enter test data:")
        test_data = {}
        for feature in features:
            while True:
                try:
                    value = input(f"Enter value for {feature}: ").strip()
                    # Try to convert to numeric if possible
                    try:
                        test_data[feature] = float(value)
                    except ValueError:
                        # Keep as string if conversion fails
                        test_data[feature] = value
                    break
                except (EOFError, KeyboardInterrupt):
                    print("\nInput cancelled.")
                    return pd.DataFrame()
                except Exception as e:
                    print(f"Invalid input for {feature}. Please try again.")
        
        # Create DataFrame with single row
        return pd.DataFrame([test_data])

    def set_target_label(self, label: str) -> None:
        self.target_label = label

    def split_df(self, df: pd.DataFrame, features: list[str],target_label: str, test_size: float):
        y = df[target_label]
        X = df[df.columns.difference([target_label])]
        if len(features) != 0:
            X = X[features]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X,y,test_size = test_size)