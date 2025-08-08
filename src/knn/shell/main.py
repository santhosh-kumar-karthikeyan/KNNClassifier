from knn.knnclassifier import KNNClassifer
from knn.voiting import UnweightedVoting, WeightedVoting
from pandas import DataFrame, read_csv
import cmd
import sys
class main(cmd.Cmd):
    intro = "Welcome to the KNN Classification shell. Type help or ? to list commands\n"
    prompt = "( KNN ) "
    def __init__(self):
        self.knn = KNNClassifer()
        
    def do_set_dataset(self,args):
        if args in None:
            self.knn.set_X_train(read_csv(sys.stdin))
            print("Dataframe set successfully")
            return
        filename: str = args.split()[0]
        self.knn.set_X_train(read_csv(filename))
        print("Dataframe set successfully")
    
    def do_set_k(self, args):
        if args is None:
            print("Usage: set_k k")
            return
        k: int = 0
        try:
            k: int = int(args.split()[0])
        except Exception as e:
            print("Provide an integer argument")
        self.knn.set_k(k)
    
    def do_set_voter(self,args):
        if args is None:
            print("Usage: set_voter [voter] ( 0 for weighted voting, 1 for unweighted voting )")
            return
        choice: int = 0
        try:
            choice = int(args.split()[0])
        except:
            print("Provide an integer argument")
            return
        if(choice == 0):
            self.knn.set_voter(UnweightedVoting())
        else:
            self.knn.set_voter(WeightedVoting())
        print(f"Voting mechanism set {"Weighted" if choice else "Unweighted"}")
        
    
if __name__ == "__main__":
    main().cmdloop()