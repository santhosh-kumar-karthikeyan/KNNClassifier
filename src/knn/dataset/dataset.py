import pandas as pd
import sys
from sklearn.model_selection import train_test_split

class DatasetHandler:
    INLINE = 0
    FILE = 1
    def __init__(self):
        self.type: int = DatasetHandler.FILE
    def read_from_stdin(self) -> pd.DataFrame:
        """Read CSV training data from stdin with interactive prompts"""
        print("\n=== Training Data Input ===")
        print("Please enter your training data in CSV format.")
        print("Instructions:")
        print("1. First line should contain column headers (comma-separated)")
        print("2. Following lines should contain data rows (comma-separated)")
        print("3. Press Enter after each line")
        print("4. Type 'END' on a new line when finished")
        print("\nExample:")
        print("feature1,feature2,target")
        print("1.2,3.4,class_a")
        print("2.1,4.3,class_b")
        print("END")
        print("\nEnter your CSV data now:")
        
        lines = []
        while True:
            try:
                line = input().strip()
                if line.upper() == 'END':
                    break
                if line:  # Only add non-empty lines
                    lines.append(line)
            except (EOFError, KeyboardInterrupt):
                print("\nInput cancelled.")
                return pd.DataFrame()
        
        if not lines:
            print("No data provided.")
            return pd.DataFrame()
        
        try:
            # Create CSV content from input lines
            csv_content = '\n'.join(lines)
            from io import StringIO
            df = pd.read_csv(StringIO(csv_content))
            self.type = DatasetHandler.INLINE
            print(f"Successfully loaded {len(df)} rows with columns: {list(df.columns)}")
            return df
        except Exception as e:
            print(f"Error parsing CSV data: {e}")
            print("Please check your CSV format and try again.")
            return pd.DataFrame()

    def read_from_path(self, path: str) -> pd.DataFrame:
        df = pd.read_csv(path)
        self.type = DatasetHandler.FILE
        return df
    
    def read_test_data_from_stdin(self, features: list[str]) -> pd.DataFrame:
        """Read test data through stdin by prompting for each feature"""
        print("\n=== Test Data Input ===")
        print(f"Please enter values for the following features: {', '.join(features)}")
        print("Note: Enter numeric values for numeric features, text for categorical features")
        
        test_data = {}
        for feature in features:
            while True:
                try:
                    value = input(f"Enter value for '{feature}': ").strip()
                    if not value:
                        print("Value cannot be empty. Please try again.")
                        continue
                        
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
                    print(f"Invalid input for {feature}: {e}. Please try again.")
        
        # Create DataFrame with single row
        result_df = pd.DataFrame([test_data])
        print(f"Test data entered: {dict(test_data)}")
        return result_df

    def set_target_label(self, label: str) -> None:
        self.target_label = label

    def split_df(self, df: pd.DataFrame, features: list[str],target_label: str, test_size: float):
        y = df[target_label]
        X = df[df.columns.difference([target_label])]
        if len(features) != 0:
            X = X[features]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X,y,test_size = test_size)