from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import cv2
import numpy as np

# Directory containing your 160 augmented images
input_directory = 'Resized_Images'
output_directory = 'Normalized_Images'

# Create the output directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

# Define an instance of the ImageDataGenerator for scaling pixel values (you can skip this if you only want normalization)
datagen = ImageDataGenerator(rescale=1.0 / 255)

# Loop through the augmented images in the input directory and save the normalized versions
for filename in os.listdir(input_directory):
    if filename.endswith('.jpg'):
        img = cv2.imread(os.path.join(input_directory, filename))
        
        if img is not None:
            # Normalize the image (scaling pixel values to the range [0, 1])
            normalized_img = img.astype('float32') / 255.0

            # Save the normalized image to the output directory
            normalized_filename = os.path.splitext(filename)[0] + '_normalized.jpg'
            cv2.imwrite(os.path.join(output_directory, normalized_filename), normalized_img * 255)
