import cv2
import numpy as np
import os

# Set the scaling factor
SCALING_FACTOR = 0.6  # Adjust this value to resize the image

# Path setup
current_path = os.getcwd()
image_path = os.path.join(current_path, 'test', 'uniform_detect.jpg')

# Load the image
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Threshold the image to ensure it is binary
_, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

# Define a larger kernel for morphological operations
kernel_size = (40, 40)  # Wider kernel size
kernel = np.ones(kernel_size, np.uint8)

# Perform morphological closing (dilation followed by erosion)
closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

# Resize images for display using the scaling factor
binary_resized = cv2.resize(binary, (0, 0), fx=SCALING_FACTOR, fy=SCALING_FACTOR, interpolation=cv2.INTER_AREA)
closed_resized = cv2.resize(closed, (0, 0), fx=SCALING_FACTOR, fy=SCALING_FACTOR, interpolation=cv2.INTER_AREA)

# Display results using OpenCV windows
cv2.imshow("Original Image (Scaled)", binary_resized)
cv2.imshow("Gaps Filled (Closing, Scaled)", closed_resized)

# Wait and close windows
cv2.waitKey(0)
cv2.destroyAllWindows()
