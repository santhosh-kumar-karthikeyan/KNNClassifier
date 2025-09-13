# KNN Classifier Shell - Stdin Enhancement

This document describes the new stdin input functionality added to the KNN Classifier shell application.

## New Features

### 1. Stdin Reader Mode

- Added support for `reader` setting with options: `"file"` and `"stdin"`
- When set to `"stdin"`, the application reads training data from standard input instead of files

### 2. Interactive Test Data Input

- In stdin mode, after training, the application prompts users to enter test data manually
- Each feature is prompted individually for user input
- Supports both numeric and string inputs with automatic type conversion

### 3. Prediction-Only Output

- In stdin mode, the application outputs only the predicted label
- No confusion matrix or classification report (since no ground truth is available)
- Ideal for real-time prediction scenarios

### 4. New Predict Command

- Added `predict` command that works only in stdin mode
- Allows making additional predictions after the initial training
- Reuses the trained model without retraining

## Code Changes

### DatasetHandler (`src/knn/dataset/dataset.py`)

- Added `read_test_data_from_stdin(features)` method
- Prompts user for each feature value
- Returns a single-row DataFrame with the test instance

### KNNClassifier (`src/knn/knnclassifier.py`)

- Added `predict(X_test)` method
- Similar to `classify()` but doesn't require ground truth labels
- Returns predicted labels only

### Shell App (`src/knn/shell/main.py`)

- Modified `reader` settable to accept `"stdin"` option
- Added `_onchange_reader()` callback
- Enhanced `do_classify()` to handle both file and stdin modes
- Added `do_predict()` command for additional predictions in stdin mode

## Usage Examples

### File Mode (Original)

```bash
python -m src.knn.shell.main
(KNN) >> set reader file
(KNN) >> set dataset data.csv
(KNN) >> set k 5
(KNN) >> classify
# Shows confusion matrix and classification report
```

### Stdin Mode (New)

```bash
cat training_data.csv | python -m src.knn.shell.main
(KNN) >> set reader stdin
(KNN) >> set k 3
(KNN) >> classify
# Prompts for target/features selection
# Then prompts for test data input
# Shows: "Predicted label: class_name"

(KNN) >> predict
# Prompts for test data input again
# Shows: "Predicted label: another_class"
```

## Configuration Options

| Setting   | Options                         | Default        | Description                       |
| --------- | ------------------------------- | -------------- | --------------------------------- |
| reader    | file, stdin                     | file           | Input mode for training data      |
| k         | integer                         | 3              | Number of neighbors               |
| distance  | euclidean, manhattan, chebyshev | euclidean      | Distance metric                   |
| voter     | weighted, unweighted            | unweighted     | Voting strategy                   |
| test_rate | 0.0-1.0                         | 0.3            | Test split ratio (file mode only) |
| dataset   | path                            | ./diabetes.csv | Dataset path (file mode only)     |

## Testing

Run the test and demo scripts:

```bash
python test_stdin_functionality.py
python demo_stdin_usage.py
```

The demo script creates a sample iris dataset (`sample_iris.csv`) for testing the new functionality.
