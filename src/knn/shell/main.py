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
            Settable("reader", str, "Type of reading to be done for the dataset", self, choices=["inline", "file"])
        )
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
        
    def do_classify(self,args):
        if self.distance is None:
            self.distance = "euclidean"
            self.pwarning("Distance metric not configured. Defaulting to Euclidean distance")
        if self.voter is None:
            self.voter = "unweighted"
            self.pwarning("Voting mechanism not configured. Defaulting to unweighted voting.")
        if self.k is None:
            self.k = 3
            self.pwarning("Number of neighbours not configured. Defaulting to 3")
        if self.test_rate is None:
            self.test_rate = 0.3
            self.pwarning("Rate of dataset to be splitted for testing is not configured. Defaulting to 0.3")
        if self.dataset is None :
            self.dataset = "./diabetes.csv"
            self.pwarning("Dataset not configured. Defaulting to diabetes dataset.")
        if self.reader == "file":
            df = self.data_handler.read_from_path(self.dataset)
        else:
            df = self.data_handler.read_from_stdin()
        features = list(df.columns)
        target = self.select(opts = features,prompt= "Choose a target column: ")
        self.poutput(f"{target} selected as target column")
        features.remove(target)
        features = questionary.checkbox("Select the features needed to be computed", choices = features).ask()
        self.poutput(f"Features selected for training: {features}")
        self.knn.set_distance_strategy(self.distance_map(self.distance))
        self.knn.set_voter(self.voter_map(self.voter))
        self.data_handler.split_df(df = df, features=features, target_label=target, test_size= self.test_rate)
        self.knn.set_X_train(self.data_handler.X_train)
        self.knn.set_y_train(self.data_handler.y_train)
        self.knn.classify(self.data_handler.X_test,self.data_handler.y_test)
        self.poutput(self.knn.cm)
        self.poutput(self.knn.report)
def main():
    app = App()
    sys.exit(app.cmdloop())
    
if __name__ == "__main__":
    main()