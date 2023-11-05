import numpy as np
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model
from Image_Arrays import X_train, X_test, X_validation

# Load the VGG16 model pre-trained on ImageNet data
base_model = VGG16(weights='imagenet', include_top=False)

# Extract features from the dataset
def extract_features(model, data):
    data = preprocess_input(data)  # Preprocess the data according to VGG16 requirements
    features = model.predict(data)
    return features

# Assuming X_train, X_validation, and X_test are your NumPy arrays of images
X_train_features = extract_features(base_model, X_train)
X_validation_features = extract_features(base_model, X_validation)
X_test_features = extract_features(base_model, X_test)

# Save the extracted features as separate files
np.save('X_train_features.npy', X_train_features)
np.save('X_validation_features.npy', X_validation_features)
np.save('X_test_features.npy', X_test_features)

# Print the shapes of the feature arrays
print("X_train_features shape:", X_train_features.shape)
print("X_validation_features shape:", X_validation_features.shape)
print("X_test_features shape:", X_test_features.shape)
