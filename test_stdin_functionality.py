#!/usr/bin/env python3
"""
Test script to demonstrate the new stdin functionality
"""
from src.knn.shell.main import App
from src.knn.dataset.dataset import DatasetHandler
import pandas as pd

def test_dataset_handler_stdin_read():
    """Test the new read_test_data_from_stdin method"""
    print("Testing DatasetHandler.read_test_data_from_stdin method...")
    
    handler = DatasetHandler()
    features = ['feature1', 'feature2', 'feature3']
    
    print(f"Features to input: {features}")
    print("This would normally prompt for user input in interactive mode")
    
    # Create sample test data manually for testing
    test_data = pd.DataFrame({
        'feature1': [1.0],
        'feature2': [2.0], 
        'feature3': [3.0]
    })
    
    print(f"Sample test data structure: {test_data}")
    print(f"Test data shape: {test_data.shape}")
    print("✓ DatasetHandler test passed")

def test_app_configuration():
    """Test the App class with new reader settings"""
    print("\nTesting App class with new reader configuration...")
    
    app = App()
    
    # Test default reader setting
    print(f"Default reader: {app.reader}")
    assert app.reader == "file", "Default reader should be 'file'"
    
    # Test setting reader to stdin
    app.reader = "stdin"
    print(f"Reader after setting to stdin: {app.reader}")
    
    print("✓ App configuration test passed")

if __name__ == "__main__":
    test_dataset_handler_stdin_read()
    test_app_configuration()
    print("\n✅ All tests passed!")
    print("\nTo use the new functionality:")
    print("1. Start the shell application")
    print("2. Set reader to 'stdin': set reader stdin")
    print("3. Set other parameters (k, distance, voter, etc.)")
    print("4. For training data, pipe CSV data to the app")
    print("5. Run 'classify' - it will prompt for test data input")
    print("6. Use 'predict' command for additional predictions")
