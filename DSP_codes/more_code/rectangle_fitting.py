import cv2
import os
import numpy as np

# Define a scale factor for visualization (e.g., 2.0 to double the size)
scale_factor = 0.5

current_path = os.getcwd()
image_path = os.path.join(current_path, 'test', 'uniform_detect2.jpg')

# Load the image (ensure it's grayscale or already binary)
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Threshold the image to ensure it's binary if not already
_, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

# Find contours of the shapes
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Convert single channel image to BGR for drawing colored lines
output = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

for cnt in contours:
    # Calculate the perimeter of the contour
    peri = cv2.arcLength(cnt, True)
    # Approximate the polygon with a certain precision factor
    approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

    # Draw the approximated polygon onto the output image
    cv2.drawContours(output, [approx], -1, (0, 255, 0), 2)

# Resize images for better visualization
def resize_image(image, scale_factor):
    if scale_factor == 1.0:
        return image
    height, width = image.shape[:2]
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

resized_original = resize_image(img, scale_factor)
resized_output = resize_image(output, scale_factor)

# Show the original mask and the approximated result
cv2.imshow("Original Mask (Scaled)", resized_original)
cv2.imshow("Approximated Polygon Shapes (Scaled)", resized_output)

cv2.waitKey(0)
cv2.destroyAllWindows()
