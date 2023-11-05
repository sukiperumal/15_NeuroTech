import h5py
from Image_Feature_Extraction import X_train_features, X_test_features, X_validation_features

# Define the file names for saving the data
hdf5_file = 'preprocessed_data.h5'

# Create an HDF5 file for storing the data
with h5py.File(hdf5_file, 'w') as hf:
    # Save your preprocessed data arrays
    hf.create_dataset('X_train_features', data=X_train_features)
    hf.create_dataset('X_validation_features', data=X_validation_features)
    hf.create_dataset('X_test_features', data=X_test_features)

print(f"Preprocessed data saved to {hdf5_file}")
