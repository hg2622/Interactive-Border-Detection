import cv2
import os
import numpy as np

def is_uniform(image_section, threshold=50):
    mean_color = np.mean(image_section, axis=(0, 1))
    max_diff = np.max(np.abs(image_section - mean_color), axis=(0, 1))
    return np.all(max_diff < threshold)

def apply_uniformity_filter(image, block_size=20, threshold=50):
    height, width, _ = image.shape
    output_mask = np.zeros((height, width), dtype=np.uint8)

    for y in range(0, height, block_size):
        for x in range(0, width, block_size):
            block = image[y:y + block_size, x:x + block_size]
            if is_uniform(block, threshold):
                output_mask[y:y + block_size, x:x + block_size] = 255
            else:
                output_mask[y:y + block_size, x:x + block_size] = 0

    return output_mask

def detect_brown_mask(image_path, brown_range):
    original_image = cv2.imread(image_path)
    if original_image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    hsv_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2HSV)
    brown_mask = cv2.inRange(hsv_image, brown_range[0], brown_range[1])
    return brown_mask

def scale_image(image, scale):
    width = int(image.shape[1] * scale)
    height = int(image.shape[0] * scale)
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

def fit_hough_lines(binary_mask, original_image):
    """
    Applies Hough Line Transform on the binary mask and draws lines on the original image.
    Args:
        binary_mask: Binary mask result from detection.
        original_image: The original input image.
    Returns:
        Image with Hough lines drawn.
    """
    # Detect edges using Canny
    edges = cv2.Canny(binary_mask, 50, 150, apertureSize=3)

    # Apply Hough Line Transform
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180, threshold=40, minLineLength=40, maxLineGap=5)
    result_image = original_image.copy()

    # Draw detected lines on the original image
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green lines

    return result_image

def custom_gap_fill_horizontal(binary_image, kernel_height=5, kernel_width=10):
    """
    Scans horizontally with a kernel of given size.
    If the leftmost and rightmost columns in the kernel window are all white,
    the entire block is set to white.
    """
    output = binary_image.copy()
    height, width = binary_image.shape

    for y in range(0, height - kernel_height + 1):
        for x in range(0, width - kernel_width + 1):
            block = output[y:y+kernel_height, x:x+kernel_width]

            # Check if first and last columns are all white
            first_col = block[:, 0]
            last_col = block[:, -1]

            if np.all(first_col == 255) and np.all(last_col == 255):
                # Set entire block to white
                output[y:y+kernel_height, x:x+kernel_width] = 255

    return output

def custom_gap_fill_vertical(binary_image, kernel_height=10, kernel_width=5):
    """
    Scans vertically with a kernel of given size.
    If the topmost and bottommost rows in the kernel window are all white,
    the entire block is set to white.
    """
    output = binary_image.copy()
    height, width = binary_image.shape

    for y in range(0, height - kernel_height + 1):
        for x in range(0, width - kernel_width + 1):
            block = output[y:y+kernel_height, x:x+kernel_width]

            # Check if top and bottom rows are all white
            top_row = block[0, :]
            bottom_row = block[-1, :]

            if np.all(top_row == 255) and np.all(bottom_row == 255):
                # Set entire block to white
                output[y:y+kernel_height, x:x+kernel_width] = 255

    return output

# File path
current_path = os.getcwd()
image_path = os.path.join(current_path, 'test', 'IMG-3651.jpg')  # Replace with your path

# Parameters for uniformity filter and brown mask
block_size = 15
threshold = 50
brown_lower = (10, 50, 50)
brown_upper = (30, 255, 255)

# Load the image and scale it
image = cv2.imread(image_path)
if image is None:
    raise FileNotFoundError(f"Image not found at {image_path}")
image = scale_image(image, scale=0.2)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Apply uniformity filter
uniform_filter_mask = apply_uniformity_filter(image_rgb, block_size=block_size, threshold=threshold)

# Apply brown mask
brown_mask = detect_brown_mask(image_path, brown_range=(brown_lower, brown_upper))
brown_mask = scale_image(brown_mask, scale=0.2)  # Scale the brown mask to match

# Intersection of the two masks
intersection_mask = cv2.bitwise_and(uniform_filter_mask, brown_mask)

# First, do horizontal gap fill with kernel 5x10
horizontal_filled = custom_gap_fill_horizontal(intersection_mask, kernel_height=20, kernel_width=40)

# Then, do vertical gap fill with kernel 10x5
gap_filled_mask = custom_gap_fill_vertical(horizontal_filled, kernel_height=40, kernel_width=20)

# Show the scaled resulting mask after gap filling
scaled_gap_filled_mask = scale_image(gap_filled_mask, scale=1)  # adjust scale as needed
cv2.imshow("Gap Filled Intersection Mask (Scaled)", scaled_gap_filled_mask)

# Fit Hough lines to the fully gap-filled mask
hough_result_image = fit_hough_lines(gap_filled_mask, image)

# Display the results
cv2.imshow("Intersection Mask", intersection_mask)
cv2.imshow("Hough Line Fit Result", hough_result_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
