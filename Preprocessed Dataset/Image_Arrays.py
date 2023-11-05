import os
import numpy as np
from PIL import Image

# Define the directories where your data is split
train_directory = 'Train'
validation_directory = 'Validation'
test_directory = 'Test'

# Define a function to load and convert images to NumPy arrays
def load_images_from_directory(directory):
    image_list = []
    for filename in os.listdir(directory):
        if filename.endswith('.jpg'):
            img = Image.open(os.path.join(directory, filename))
            img_array = np.array(img)
            image_list.append(img_array)
    return np.array(image_list)

# Load and convert images in the training set
X_train = load_images_from_directory(train_directory)

# Load and convert images in the validation set
X_validation = load_images_from_directory(validation_directory)

# Load and convert images in the test set
X_test = load_images_from_directory(test_directory)

# Optionally, you can also load and convert labels (if available)
# For example, if you have a separate file or data structure containing labels
# You can load and convert them to NumPy arrays in a similar way

# Print the shape of the resulting arrays
print("X_train shape:", X_train.shape)
print("X_validation shape:", X_validation.shape)
print("X_test shape:", X_test.shape)
