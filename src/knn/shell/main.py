from cmd2 import Cmd, Cmd2ArgumentParser,with_argparser,Settable
import sys
from knn.knnclassifier import KNNClassifer
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import PathCompleter
from knn.distance import EuclideanDistance, ManhattanDistance, ChebyshevDistance, DistanceStrategy
from knn.voting import UnweightedVoting, WeightedVoting, VotingStrategy
import os
from knn.dataset import DatasetHandler
from argcomplete.completers import FilesCompleter
import questionary

prompt_session = PromptSession(completer=PathCompleter())
class App(Cmd):
    def __init__(self):
        super().__init__()
        self.prompt: str = "(KNN) >> "
        self.knn: KNNClassifer = KNNClassifer()
        self.k: int | None = None
        self.distance: str | None = None
        self.voter : str | None = None
        self.test_rate: float | None = None
        self.dataset: str | None = None
        self.reader: str | None = "file"
        self.data_handler: DatasetHandler = DatasetHandler()
        self.add_settable(
            Settable("k",int,"The number of neighbours to be considered for classification",self)
        )
        self.add_settable(
            Settable("distance",str,"Choice for a distance metric", choices=["euclidean", "manhattan", "chebyshev"],settable_object= self, onchange_cb=self._onchange_distance)
        )
        self.add_settable(
            Settable("voter", str, "Choice for voting strategy to be followed", self, choices= ["unweighted", "weighted"], onchange_cb=self._onchange_voter)
        )
        self.add_settable(
            Settable("test_rate", self._test_rate_type, "The rate of test data to be splitted from the dataframe. Ranges from 0 to 1.",self)
        )
        self.add_settable(
            Settable("dataset", self._path_type, "Path to the dataset to be used for classification", self)
        )
        self.add_settable(
            Settable("reader", str, "Type of reading to be done for the dataset", self, choices=["stdin", "file"], onchange_cb=self._onchange_reader)
        )
    
    def do_help_stdin(self, args):
        """Show detailed help for stdin mode usage"""
        self.poutput("=== STDIN Mode Usage Guide ===\n")
        self.poutput("STDIN mode allows you to input training data interactively and get predictions.")
        self.poutput("No need to prepare CSV files beforehand!\n")
        
        self.poutput("📋 Step-by-step process:")
        self.poutput("1. Set reader to stdin: set reader stdin")
        self.poutput("2. Configure parameters (optional):")
        self.poutput("   - set k <number>          # Number of neighbors (default: 3)")
        self.poutput("   - set distance <metric>   # euclidean, manhattan, or chebyshev")
        self.poutput("   - set voter <strategy>    # weighted or unweighted")
        self.poutput("3. Run classify command: classify")
        self.poutput("4. Enter training data in CSV format when prompted")
        self.poutput("5. Select target column and features")
        self.poutput("6. Enter test data for prediction")
        self.poutput("7. Use 'predict' command for additional predictions\n")
        
        self.poutput("💡 CSV Input Format:")
        self.poutput("Enter data line by line, starting with headers:")
        self.poutput("feature1,feature2,target")
        self.poutput("1.2,3.4,class_a")
        self.poutput("2.1,4.3,class_b")
        self.poutput("END")
        self.poutput("\n✨ Type 'END' when finished entering data")
        
    def do_help_modes(self, args):
        """Show comparison between file and stdin modes"""
        self.poutput("=== KNN Classifier Modes ===\n")
        self.poutput("📁 FILE MODE (reader=file):")
        self.poutput("- Load data from CSV files")
        self.poutput("- Automatically split into train/test sets")
        self.poutput("- Shows confusion matrix and classification report")
        self.poutput("- Best for: Model evaluation and testing\n")
        
        self.poutput("⌨️  STDIN MODE (reader=stdin):")
        self.poutput("- Enter training data interactively")
        self.poutput("- Uses all data for training")
        self.poutput("- Prompts for test data input")
        self.poutput("- Shows single predictions")
        self.poutput("- Best for: Real-time predictions and quick testing")
        self.poutput("- Use 'predict' command for additional predictions")
    def _test_rate_type(self, val: float):
        if not 0 <= val <= 1:
            raise ValueError("Value must be a number between 0 and 1")
        return val

    def _path_type(self, path: str):
        path = os.path.expanduser(path.strip())
        if not os.path.isfile(path):
            raise ValueError(f"File {path} doesn't exist")
        return path
    
    def voter_map(self,voter: str) -> VotingStrategy:
        if(voter.lower() == "unweighted"):
            return UnweightedVoting()
        return WeightedVoting()
    
    def distance_map(self, distance: str ) -> DistanceStrategy:
        if(distance.lower() == "euclidean"):
            return EuclideanDistance()

        if(distance.lower() == "manhattan"):
            return ManhattanDistance()

        return ChebyshevDistance()

    def _onchange_voter(self,_param_name, _old, new) -> None:
        self.knn.set_voter(self.voter_map(new))
    
    def _onchange_distance(self,_param_name,old,new) -> None:
        self.knn.set_distance_strategy(self.distance_map(new))
    
    def _onchange_reader(self, _param_name, _old, new) -> None:
        """Callback when reader type changes"""
        self.reader = new
        if new == "stdin":
            self.poutput("📝 Reader set to STDIN mode.")
            self.poutput("   You will be prompted to enter training data during classification.")
            self.poutput("   Type 'help_stdin' for detailed usage instructions.")
        else:
            self.poutput("📁 Reader set to FILE mode.")
            self.poutput("   Configure 'dataset' parameter to specify the CSV file path.")
        
    def do_classify(self,args):
        "Classifies the configured dataset with the configured meta-parameters"
        if self.distance is None:
            self.distance = "euclidean"
            self.pwarning("Distance metric not configured. Defaulting to Euclidean distance")
        if self.voter is None:
            self.voter = "unweighted"
            self.pwarning("Voting mechanism not configured. Defaulting to unweighted voting.")
        if self.k is None:
            self.k = 3
            self.pwarning("Number of neighbours not configured. Defaulting to 3")
        
        # Handle reader mode configuration
        if self.reader == "stdin":
            self.poutput("=== STDIN Mode Active ===")
            self.poutput("You will be prompted to enter training data in CSV format,")
            self.poutput("then select target and features, and finally provide test data for prediction.")
        else:
            if self.test_rate is None:
                self.test_rate = 0.3
                self.pwarning("Rate of dataset to be splitted for testing is not configured. Defaulting to 0.3")
            if self.dataset is None :
                self.dataset = "./diabetes.csv"
                self.pwarning("Dataset not configured. Defaulting to diabetes dataset.")
        
        # Read training data
        if self.reader == "file":
            df = self.data_handler.read_from_path(self.dataset)
        else:
            df = self.data_handler.read_from_stdin()
            
        # Check if data was successfully loaded
        if df.empty:
            self.poutput("No data loaded. Classification cannot proceed.")
            return
        
        features = list(df.columns)
        target = self.select(opts = features,prompt= "Choose a target column: ")
        self.poutput(f"'{target}' selected as target column")
        features.remove(target)
        features = questionary.checkbox("Select the features needed to be computed", choices = features).ask()
        
        if not features:
            self.poutput("No features selected. Classification cannot proceed.")
            return
            
        self.poutput(f"Features selected for training: {features}")
        
        # Set up the classifier
        self.knn.set_distance_strategy(self.distance_map(self.distance))
        self.knn.set_voter(self.voter_map(self.voter))
        
        # Handle different reading modes
        if self.reader == "stdin":
            # For stdin mode, train on all data and predict on user input
            X = df[features]
            y = df[target]
            self.knn.set_X_train(X)
            self.knn.set_y_train(y)
            
            self.poutput(f"Model trained with {len(X)} samples using {len(features)} features.")
            
            # Get test data from user input
            test_data = self.data_handler.read_test_data_from_stdin(features)
            
            if not test_data.empty:
                # Predict using the trained model
                prediction = self.knn.predict(test_data)
                self.poutput(f"\n🎯 Predicted label: {prediction.iloc[0]}")
                self.poutput("\nYou can use the 'predict' command to make additional predictions.")
            else:
                self.poutput("No test data provided.")
        else:
            # For file mode, split data and show classification report
            self.data_handler.split_df(df = df, features=features, target_label=target, test_size= self.test_rate)
            self.knn.set_X_train(self.data_handler.X_train)
            self.knn.set_y_train(self.data_handler.y_train)
            self.knn.classify(self.data_handler.X_test,self.data_handler.y_test)
            self.poutput(self.knn.cm)
            self.poutput(self.knn.report)
    
    def do_predict(self, args):
        "Predict a single instance using trained model (for stdin input mode)"
        if self.reader == "file":
            self.poutput("❌ This command is only available when reader is set to 'stdin'")
            self.poutput("   Current reader mode: file")
            self.poutput("   To use prediction mode: set reader stdin")
            return
            
        if not hasattr(self.knn, 'X_train') or self.knn.X_train is None:
            self.poutput("❌ Model not trained. Please run 'classify' first to train the model.")
            return
            
        # Get the features from the trained model
        features = list(self.knn.X_train.columns)
        self.poutput("=== Additional Prediction ===")
        
        # Get test data from user input
        test_data = self.data_handler.read_test_data_from_stdin(features)
        
        if not test_data.empty:
            # Predict using the trained model
            prediction = self.knn.predict(test_data)
            self.poutput(f"\n🎯 Predicted label: {prediction.iloc[0]}")
        else:
            self.poutput("No test data provided.")
def main():
    app = App()
    sys.exit(app.cmdloop())
    
if __name__ == "__main__":
    main()