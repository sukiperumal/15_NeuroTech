import numpy as np

# Load your feature-extracted data
X_train_features = np.load('X_train_features.npy')  # Load your feature-extracted training data
X_validation_features = np.load('X_validation_features.npy')  # Load your feature-extracted validation data
X_test_features = np.load('X_test_features.npy')  # Load your feature-extracted test data

# Check for any missing data
if np.isnan(X_train_features).any() or np.isnan(X_validation_features).any() or np.isnan(X_test_features).any():
    print("Missing data (NaN values) found in the dataset.")
else:
    print("No missing data (NaN values) found in the dataset.")

# You can perform additional checks for outliers or anomalies as needed
# For example, you can check for extreme values in the features

# Your code for additional checks here
