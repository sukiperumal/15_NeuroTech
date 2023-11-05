from PIL import Image
import os

# Input and output directories
input_dir = "Cleaned_Images"
output_dir = "Resized_Images"

# Target dimensions
target_width = 224
target_height = 224

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# List all files in the input directory
image_files = os.listdir(input_dir)

# Loop through each image file and resize it
for image_file in image_files:
    try:
        # Open the image
        with Image.open(os.path.join(input_dir, image_file)) as img:
            # Resize the image
            img = img.resize((target_width, target_height), Image.ANTIALIAS)

            # Save the resized image to the output directory
            img.save(os.path.join(output_dir, image_file))

            print(f"Resized: {image_file}")
    except Exception as e:
        print(f"Error processing {image_file}: {e}")
