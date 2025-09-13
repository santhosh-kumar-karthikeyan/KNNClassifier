#!/usr/bin/env python3
"""
Demonstration of the new stdin functionality for KNN Classifier
This script shows how to use the enhanced shell with stdin input
"""

import os
import sys

def create_demo_usage_guide():
    print("=== KNN Classifier Enhanced Shell Usage Guide ===\n")
    
    print("The KNN Classifier shell has been extended with the following new features:")
    print("1. Support for 'stdin' reader type")
    print("2. Interactive test data input when using stdin mode")
    print("3. Single prediction output instead of classification report for stdin mode")
    print("4. New 'predict' command for additional predictions\n")
    
    print("=== Usage Examples ===\n")
    
    print("1. Using File Mode (Original functionality):")
    print("   (KNN) >> set reader file")
    print("   (KNN) >> set dataset path/to/your/dataset.csv")
    print("   (KNN) >> set k 5")
    print("   (KNN) >> set distance euclidean")
    print("   (KNN) >> set voter weighted")
    print("   (KNN) >> classify")
    print("   # This will show confusion matrix and classification report\n")
    
    print("2. Using Stdin Mode (New functionality):")
    print("   # First, prepare your training data in CSV format")
    print("   (KNN) >> set reader stdin")
    print("   (KNN) >> set k 3")
    print("   (KNN) >> set distance manhattan")
    print("   (KNN) >> set voter unweighted")
    print("   # Then pipe your training CSV data to the application:")
    print("   $ cat training_data.csv | python -m src.knn.shell.main")
    print("   (KNN) >> classify")
    print("   # The app will prompt you to select target and features")
    print("   # Then it will ask for test data input:")
    print("   # Enter value for feature1: 5.2")
    print("   # Enter value for feature2: 3.1")
    print("   # ... (for each feature)")
    print("   # Output: Predicted label: class_name\n")
    
    print("3. Making Additional Predictions:")
    print("   # After running classify in stdin mode, you can make more predictions")
    print("   (KNN) >> predict")
    print("   # Enter value for feature1: 4.8")
    print("   # Enter value for feature2: 2.9")
    print("   # Output: Predicted label: another_class\n")
    
    print("=== Key Differences ===")
    print("• File mode: Splits data, shows confusion matrix and classification report")
    print("• Stdin mode: Uses all data for training, prompts for test input, shows single prediction")
    print("• The 'predict' command only works in stdin mode for additional predictions")
    print("• Stdin mode is ideal for real-time prediction scenarios")
    print("• File mode is ideal for model evaluation and testing\n")
    
    print("=== Configuration Options ===")
    print("• reader: 'file' or 'stdin'")
    print("• k: Number of neighbors (default: 3)")
    print("• distance: 'euclidean', 'manhattan', or 'chebyshev' (default: euclidean)")
    print("• voter: 'weighted' or 'unweighted' (default: unweighted)")
    print("• test_rate: Only used in file mode (default: 0.3)")
    print("• dataset: Only used in file mode (default: ./diabetes.csv)")

def create_sample_training_data():
    """Create a sample CSV file for testing"""
    sample_data = """sepal_length,sepal_width,petal_length,petal_width,species
5.1,3.5,1.4,0.2,setosa
4.9,3.0,1.4,0.2,setosa
4.7,3.2,1.3,0.2,setosa
7.0,3.2,4.7,1.4,versicolor
6.4,3.2,4.5,1.5,versicolor
6.9,3.1,4.9,1.5,versicolor
6.3,3.3,6.0,2.5,virginica
5.8,2.7,5.1,1.9,virginica
7.1,3.0,5.9,2.1,virginica"""
    
    with open("/home/s4ndy/Programs/CI/KNNClassifier/sample_iris.csv", "w") as f:
        f.write(sample_data)
    
    print("Created sample_iris.csv for testing")

if __name__ == "__main__":
    create_demo_usage_guide()
    create_sample_training_data()
    
    print("\n=== Testing the Application ===")
    print("You can now test the application with:")
    print("1. File mode: python -m src.knn.shell.main")
    print("2. Stdin mode: cat sample_iris.csv | python -m src.knn.shell.main")
    print("\nSample test data for iris dataset:")
    print("sepal_length: 5.0")
    print("sepal_width: 3.0") 
    print("petal_length: 1.5")
    print("petal_width: 0.3")
    print("Expected prediction: setosa")
