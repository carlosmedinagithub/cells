import cv2
import numpy as np

# Load and preprocess image
image = cv2.imread('./images/sample1.png')
cv2.imshow('Original Image', image)
cv2.waitKey(0)

# Convert to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow('Grayscale Image', gray_image)
cv2.waitKey(0)

# Noise reduction using Gaussian Blur
preprocessed_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
cv2.imshow('Blurred Image', preprocessed_image)
cv2.waitKey(0)

# Edge detection using Canny
canny_edges = cv2.Canny(preprocessed_image, 50, 150)
cv2.imshow('Canny Edges', canny_edges)
cv2.waitKey(0)

# Morphological processing to enhance cell shapes
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
dilated_edges = cv2.dilate(canny_edges, kernel, iterations=2)
eroded_edges = cv2.erode(dilated_edges, kernel, iterations=2)
cv2.imshow('Morphological Processing', eroded_edges)
cv2.waitKey(0)

# Create masks from the morphological image
cell_mask = eroded_edges == 255  # White areas are cells
empty_space_mask = eroded_edges == 0  # Black areas are empty spaces

# Apply masks to create the respective images
cell_highlighted_image = image.copy()
cell_highlighted_image[cell_mask] = [0, 0, 255]  # Highlight cells in red on the original image

empty_space_highlighted_image = image.copy()
empty_space_highlighted_image[empty_space_mask] = [255, 0, 0]  # Highlight empty spaces in blue on the original image

# Calculate cell area
cell_area = np.sum(cell_mask)  # Total number of pixels that are red (cells)
total_pixels = image.shape[0] * image.shape[1]  # Total number of pixels in the image
cell_coverage_percentage = (cell_area / total_pixels) * 100

# Display the images
cv2.imshow('Cell Highlighted Image', cell_highlighted_image)
cv2.waitKey(0)
cv2.imshow('Empty Space Highlighted Image', empty_space_highlighted_image)
cv2.waitKey(0)

# Display cell area coverage percentage
print(f"Total Coverage by Cells: {cell_coverage_percentage:.2f}%")

cv2.destroyAllWindows()
