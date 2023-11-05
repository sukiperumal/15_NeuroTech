from PIL import Image
import os

# Set the input directory where your images are located
input_directory = 'Images'

# Set the output directory where cleaned images will be saved
output_directory = 'Cleaned_Images'

# Create the output directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

# A dictionary to store unique image hashes
image_hashes = {}

# Process each image in the input directory
for filename in os.listdir(input_directory):
    if filename.endswith(('.jpg', '.jpeg', '.png')):
        image_path = os.path.join(input_directory, filename)
        
        # Open the image and convert it to grayscale
        image = Image.open(image_path).convert('L')

        # Calculate a hash of the image to identify duplicates
        image_hash = image.tobytes()

        # Check if this image is a duplicate
        if image_hash not in image_hashes:
            # Save the unique image to the output directory
            output_path = os.path.join(output_directory, filename)
            image.save(output_path)
            
            # Store the image hash for future comparisons
            image_hashes[image_hash] = filename

print(f"Cleaned {len(image_hashes)} unique images.")

