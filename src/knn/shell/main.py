from cmd2 import Cmd, Cmd2ArgumentParser,with_argparser,Settable
import sys
from knn.knnclassifier import KNNClassifer
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import PathCompleter
from knn.distance import EuclideanDistance, ManhattanDistance, ChebyshevDistance, DistanceStrategy
from knn.voting import UnweightedVoting, WeightedVoting, VotingStrategy

prompt_session = PromptSession(completer=PathCompleter())
class App(Cmd):
    def __init__(self):
        super().__init__()
        self.knn: KNNClassifer = KNNClassifer()
        self.k: int | None = None
        self.distance: str | None = None
        self.voter : str | None = None
        self.test_rate: float | None = None
        self.add_settable(
            Settable("k",int,"The number of neighbours to be considered for classification",self)
        )
        self.add_settable(
            Settable("distance",str,"Choice for a distance metric", choices=["euclidean", "manhattan", "chebyshev"],settable_object= self)
        )
        self.add_settable(
            Settable("voter", str, "Choice for voting strategy to be followed", self, choices= ["unweighted", "weighted"])
        )
        self.add_settable(
            Settable("test_rate", self._test_rate_type, "The rate of test data to be splitted from the dataframe. Ranges from 0 to 1.",self)
        )

    def _test_rate_type(self, val: float):
        if not 0 <= val <= 1:
            raise ValueError("Value must be a number between 0 and 1")
        return val

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

    def do_classify(self,args):
        if self.distance is None:
            self.distance = "euclidean"
            self.poutput("Distance metric not configured. Defaulting to Euclidean distance")
        if self.voter is None:
            self.voter = "unweighted"
            self.poutput("Voting mechanism not configured. Defaulting to unweighted voting.")
        if self.k is None:
            self.k = 3
            self.poutput("Number of neighbours not configured. Defaulting to 3")
        if self.test_rate is None:
            self.test_rate = 0.3
            self.poutput("Rate of dataset to be splitted for testing is not configured. Defaulting to 0.3")
        self.knn.set_distance_strategy(self.distance_map(self.distance))
        self.knn.set_voter(self.voter_map(self.voter))
    
if __name__ == "__main__":
    app = App()
    sys.exit(app.cmdloop())