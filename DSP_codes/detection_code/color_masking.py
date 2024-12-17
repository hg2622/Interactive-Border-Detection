import cv2
import os
import numpy as np


def detect_brown_mask(image_path, brown_range):
    """
    Detects the brown color mask in an image.

    Parameters:
        image_path (str): Path to the input image.
        brown_range (tuple): Lower and upper HSV bounds for the brown region.

    Returns:
        numpy.ndarray: Mask for the brown region.
    """
    original_image = cv2.imread(image_path)
    if original_image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")

    # Convert to HSV color space
    hsv_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2HSV)

    # Create a mask for the brown region
    brown_mask = cv2.inRange(hsv_image, brown_range[0], brown_range[1])

    return brown_mask


def scale_image(image, scale):
    """Scales the image by the given factor."""
    width = int(image.shape[1] * scale)
    height = int(image.shape[0] * scale)
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


# Define color range for brown detection
brown_lower = (10, 50, 50)
brown_upper = (30, 255, 255)

# File path
current_path = os.getcwd()
image_path = os.path.join(current_path, 'test', 'IMG-3645.jpg')

# Detect the brown mask
brown_mask = detect_brown_mask(
    image_path,
    brown_range=(brown_lower, brown_upper),
)

# Scale the brown mask for display
brown_scaled = scale_image(brown_mask, scale=0.2)

# Display the brown mask
cv2.imshow("Brown Mask", brown_scaled)
cv2.waitKey(0)
cv2.destroyAllWindows()
