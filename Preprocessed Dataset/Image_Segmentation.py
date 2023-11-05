import cv2
import numpy as np

# Load the image
image = cv2.imread('Cleaned_Images\image_95.jpg')  # Load your image

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply color thresholding to segment objects of interest
lower_bound = np.array([50, 50, 50])  # Define your lower color threshold
upper_bound = np.array([255, 255, 255])  # Define your upper color threshold
mask = cv2.inRange(image, lower_bound, upper_bound)

# Apply edge detection to find object contours
edges = cv2.Canny(mask, threshold1=30, threshold2=100)

# Find object contours
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Create a mask for the segmented objects
segmented_mask = np.zeros_like(image)

# Draw the contours on the mask
cv2.drawContours(segmented_mask, contours, -1, (0, 255, 0), 2)

# Display the segmented objects
cv2.imshow('Segmented Objects', segmented_mask)
cv2.waitKey(0)
cv2.destroyAllWindows()
