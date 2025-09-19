from typing import Any
from .distance import DistanceStrategy
from .voting import VotingStrategy
import pandas as pd
from typing import Self
from sklearn.metrics import confusion_matrix, classification_report
from numpy import ndarray
from sklearn.model_selection import train_test_split
class KNNClassifer:
    def __init__(self):
        self.k: int = 3
        self.verbose: bool = False
        self.max_display_samples: int = 10  # For file mode
        
    def set_verbose(self, verbose: bool) -> Self:
        """Enable/disable verbose output for intermediate results"""
        self.verbose = verbose
        return self
        
    def set_max_display_samples(self, max_samples: int) -> Self:
        """Set maximum number of samples to display in verbose mode for file input"""
        self.max_display_samples = max_samples
        return self
        
    def calculate_auto_k(self, training_size: int) -> int:
        """Calculate k automatically as closest odd number to 10% of training data length"""
        auto_k = int(training_size * 0.1)
        # Ensure k is odd
        if auto_k % 2 == 0:
            auto_k += 1
        # Ensure k is at least 1
        auto_k = max(1, auto_k)
        # Ensure k doesn't exceed training size
        auto_k = min(auto_k, training_size)
        return auto_k
    def set_k(self,k: int) -> Self:
        self.k = k
        return self
    def set_distance_strategy(self, dis_strat: DistanceStrategy) -> Self:
        self.distance_strategy = dis_strat
        return self
    def set_voter(self, voter : VotingStrategy) -> Self:
        self.voter = voter
        return self
    def set_X_train(self,df: pd.DataFrame) -> Self:
        self.X_train = df
        return self
    def set_y_train(self,labels: pd.Series) -> Self:
        self.y_train = labels
        return self
    def set_target_name(self,name: str) -> Self:
        self.target_name = name
        return self
    def split_df(self, dataframe: pd.DataFrame, target_label: str):
        return train_test_split()
    def classify(self, X_test: pd.DataFrame, y_test: pd.DataFrame, verbose: bool | None = None, is_stdin_mode: bool = False):
        if self.X_train is None:
            return
            
        if verbose is None:
            verbose = self.verbose
            
        if verbose:
            print(f"\nStarting KNN Classification with k={self.k}")
            print(f"Training data: {len(self.X_train)} samples")
            print(f"Test data: {len(X_test)} samples")
            
            # Use verbose prediction
            y_pred = self.predict_verbose(X_test, is_stdin_mode)
        else:
            # Original classification
            distance : pd.DataFrame = X_test.apply(lambda test_row: self.distance_strategy.computeDistance(self.X_train,test_row), axis = 1)
            y_pred: pd.Series = distance.apply(lambda distance_row: self.voter.getLabel(distance_row,self.y_train,self.k), axis = 1)
        
        y_pred.name = "Predicted"
        X_test = pd.concat([X_test,y_test, y_pred], axis = 1)
        self.cm =  confusion_matrix(y_test,y_pred)
        self.report = classification_report(y_true=y_test, y_pred=y_pred)
        return y_pred
    
    def predict_verbose(self, X_test: pd.DataFrame, is_stdin_mode: bool = False) -> pd.Series:
        """Predict with detailed intermediate output"""
        if self.X_train is None:
            raise ValueError("Model must be trained first (X_train not set)")
            
        print(f"\nStarting KNN Prediction with k={self.k}")
        print(f"Training data: {len(self.X_train)} samples")
        print(f"Test data: {len(X_test)} samples")
        print(f"Distance metric: {type(self.distance_strategy).__name__}")
        print(f"Voting strategy: {type(self.voter).__name__}")
        
        # Determine how many samples to show details for
        if is_stdin_mode:
            samples_to_show = len(X_test)  # Show all for stdin
            print(f"Showing detailed results for all {samples_to_show} test samples (STDIN mode)")
        else:
            samples_to_show = min(self.max_display_samples, len(X_test))
            print(f"Showing detailed results for first {samples_to_show} test samples (FILE mode)")
        
        y_pred_list = []
        
        for i, (test_idx, test_row) in enumerate(X_test.iterrows()):
            print(f"\n{'='*60}")
            print(f"Processing Test Sample {i+1}/{len(X_test)} (Index: {test_idx})")
            print(f"Test point: {dict(test_row)}")
            
            # Calculate distances
            distances = self.distance_strategy.computeDistance(self.X_train, test_row)
            
            if i < samples_to_show:
                print(f"\nDistance Calculations:")
                print(f"{'Index':<8} {'Distance':<12} {'Features':<30} {'Label'}")
                print("-" * 70)
                
                # Show distances for all training points (or first 10 for large datasets)
                display_count = min(10, len(distances)) if not is_stdin_mode else len(distances)
                sorted_distances = distances.sort_values()
                
                for j, (train_idx, dist) in enumerate(sorted_distances.head(display_count).items()):
                    features_str = str(dict(self.X_train.loc[train_idx]))[:25] + "..." if len(str(dict(self.X_train.loc[train_idx]))) > 25 else str(dict(self.X_train.loc[train_idx]))
                    print(f"{train_idx:<8} {dist:<12.4f} {features_str:<30} {self.y_train.loc[train_idx]}")
                
                if display_count < len(distances):
                    print(f"... and {len(distances) - display_count} more training samples")
            
            # Get top k neighbors and calculate prediction
            prediction, top_k_details = self._get_prediction_with_details(distances, test_row, i < samples_to_show)
            y_pred_list.append(prediction)
            
        return pd.Series(y_pred_list, index=X_test.index, name="Predicted")
    
    def _get_prediction_with_details(self, distances: pd.Series, test_row: pd.Series, show_details: bool):
        """Get prediction with detailed breakdown of voting process"""
        # Get top k neighbors
        top_k_indices = distances.nsmallest(self.k).index
        top_k_distances = distances[top_k_indices]
        top_k_labels = self.y_train[top_k_indices]
        
        if show_details:
            print(f"\nTop {self.k} Nearest Neighbors:")
            print(f"{'Rank':<6} {'Index':<8} {'Distance':<12} {'Label':<15} {'Weight':<12}")
            print("-" * 65)
            
            weights = []
            for rank, (train_idx, dist) in enumerate(top_k_distances.items(), 1):
                label = self.y_train.loc[train_idx]
                
                # Calculate weight based on voting strategy
                if hasattr(self.voter, 'getLabel') and 'Weighted' in type(self.voter).__name__:
                    weight = dist ** -2 if dist > 0 else float('inf')
                    weights.append(weight)
                    print(f"{rank:<6} {train_idx:<8} {dist:<12.4f} {label:<15} {weight:<12.4f}")
                else:
                    weights.append(1.0)  # Unweighted
                    print(f"{rank:<6} {train_idx:<8} {dist:<12.4f} {label:<15} {'1.0 (equal)':<12}")
            
            # Show voting breakdown
            print(f"\nVoting Breakdown:")
            unique_labels = top_k_labels.unique()
            
            if 'Weighted' in type(self.voter).__name__:
                print("Weighted Voting Results:")
                label_weights = {}
                for i, (train_idx, dist) in enumerate(top_k_distances.items()):
                    label = self.y_train.loc[train_idx]
                    weight = weights[i]
                    if label not in label_weights:
                        label_weights[label] = 0
                    label_weights[label] += weight
                
                print(f"{'Label':<15} {'Total Weight':<15} {'Votes'}")
                print("-" * 45)
                for label in sorted(label_weights.keys()):
                    vote_count = sum(1 for l in top_k_labels if l == label)
                    print(f"{label:<15} {label_weights[label]:<15.4f} {vote_count}")
                
                winning_label = max(label_weights.items(), key=lambda x: x[1])[0]
            else:
                print("📊 Unweighted Voting Results:")
                from collections import Counter
                vote_counts = Counter(top_k_labels)
                
                print(f"{'Label':<15} {'Vote Count':<15}")
                print("-" * 30)
                for label, count in sorted(vote_counts.items()):
                    print(f"{label:<15} {count:<15}")
                
                winning_label = vote_counts.most_common(1)[0][0]
            
            print(f"\nPredicted Label: {winning_label}")
        
        # Get the actual prediction using the voter
        prediction = self.voter.getLabel(distances, self.y_train, self.k)
        
        return prediction, {
            'top_k_indices': top_k_indices,
            'top_k_distances': top_k_distances,
            'top_k_labels': top_k_labels
        }
    
    def predict(self, X_test: pd.DataFrame, verbose: bool | None = None, is_stdin_mode: bool = False) -> pd.Series:
        """Predict labels for test data without requiring ground truth labels"""
        if verbose is None:
            verbose = self.verbose
            
        if verbose:
            return self.predict_verbose(X_test, is_stdin_mode)
        else:
            # Original simple prediction
            if self.X_train is None:
                raise ValueError("Model must be trained first (X_train not set)")
            distance : pd.DataFrame = X_test.apply(lambda test_row: self.distance_strategy.computeDistance(self.X_train,test_row), axis = 1)
            y_pred: pd.Series = distance.apply(lambda distance_row: self.voter.getLabel(distance_row,self.y_train,self.k), axis = 1)
            y_pred.name = "Predicted"
            return y_pred