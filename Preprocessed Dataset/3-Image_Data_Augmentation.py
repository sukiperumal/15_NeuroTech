import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, array_to_img, load_img

# Directory containing your 160 MRI scan images in a standard image format (e.g., PNG or JPG)
image_directory = 'Resized_Images'

# Directory to save the augmented images
output_directory = 'Augmented_Images'

# Create the output directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

# Define an instance of the ImageDataGenerator with reduced augmentation settings
datagen = ImageDataGenerator(
    rotation_range=20,      # Reduced rotation angle in degrees
    width_shift_range=0.1,  # Reduced shift the width by a fraction
    height_shift_range=0.1, # Reduced shift the height by a fraction
    shear_range=0.1,        # Reduced shear angle in radians
    zoom_range=0.1,         # Reduced zoom range
    horizontal_flip=True,   # Randomly flip images horizontally
    fill_mode='nearest'     # Fill mode for points outside the input boundaries
)

# Loop through the MRI scan images in the directory and apply data augmentation
for filename in os.listdir(image_directory):
    if filename.endswith('.jpg'):  # Assuming your images are in JPG format
        img = load_img(os.path.join(image_directory, filename))
        x = img_to_array(img)
        x = x.reshape((1,) + x.shape)

        i = 0
        for batch in datagen.flow(x, batch_size=1):
            img_aug = array_to_img(batch[0])
            augmented_filename = os.path.splitext(filename)[0] + f'_aug_{i}.jpg'
            img_aug.save(os.path.join(output_directory, augmented_filename))
            i += 1
            if i >= 5:  # Generate 5 augmented versions for each image (reduced from 20)
                break
