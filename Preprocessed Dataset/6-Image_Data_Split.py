import os
from sklearn.model_selection import train_test_split

# Directory containing your 160 preprocessed images
input_directory = 'Normalized_Images'

# Define the proportions for splitting (e.g., 70% train, 15% validation, 15% test)
train_ratio = 0.7
validation_ratio = 0.15
test_ratio = 0.15

# Get the list of image filenames
image_filenames = [filename for filename in os.listdir(input_directory) if filename.endswith('.jpg')]

# Split the data into training, validation, and test sets
train_filenames, test_filenames = train_test_split(image_filenames, test_size=(1 - train_ratio), random_state=42)
validation_filenames, test_filenames = train_test_split(test_filenames, test_size=(test_ratio / (test_ratio + validation_ratio)), random_state=42)

# Define output directories for each split
train_directory = 'Train'
validation_directory = 'Validation'
test_directory = 'Test'

# Create the output directories if they don't exist
for directory in [train_directory, validation_directory, test_directory]:
    os.makedirs(directory, exist_ok=True)

# Move the images to their respective split directories
for filename in train_filenames:
    src = os.path.join(input_directory, filename)
    dst = os.path.join(train_directory, filename)
    os.rename(src, dst)

for filename in validation_filenames:
    src = os.path.join(input_directory, filename)
    dst = os.path.join(validation_directory, filename)
    os.rename(src, dst)

for filename in test_filenames:
    src = os.path.join(input_directory, filename)
    dst = os.path.join(test_directory, filename)
    os.rename(src, dst)

print(f"Dataset split into {len(train_filenames)} training images, {len(validation_filenames)} validation images, and {len(test_filenames)} test images.")
