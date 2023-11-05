import os
import cv2

# Directory containing your 160 normalized, resized, and augmented images
input_directory = 'Normalized_Images'

# Create a list to track missing or corrupted images
missing_images = []

# Loop through the images in the input directory
for filename in os.listdir(input_directory):
    if filename.endswith('.jpg'):
        file_path = os.path.join(input_directory, filename)
        
        # Attempt to read the image using OpenCV
        img = cv2.imread(file_path)
        
        # Check if the image is None (indicating an issue)
        if img is None:
            # Add the filename to the list of missing or corrupted images
            missing_images.append(filename)

# Output the list of missing or corrupted images
if missing_images:
    print(f"Missing or corrupted images: {missing_images}")
else:
    print("No missing or corrupted images found.")
