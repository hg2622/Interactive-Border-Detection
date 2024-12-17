import cv2
import os
import numpy as np


def detect_large_edges(image_path, sigma=1.2, gradient_threshold=50, kernel_size=3, display_scale=0.2):
    """
    Detect edges in an image and return the edge-detected image.

    Parameters:
        image_path (str): Path to the image.
        sigma (float): Gaussian blur standard deviation.
        gradient_threshold (int): Threshold for edge gradients.
        kernel_size (int): Kernel size for Gaussian blur.
        display_scale (float): Scale factor for resizing the output images.

    Returns:
        numpy.ndarray: Edge-detected image.
    """
    # Read the image in grayscale
    original = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if original is None:
        raise FileNotFoundError(f"Image at {image_path} could not be loaded.")

    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(original, (kernel_size, kernel_size), sigma)

    # Detect edges using Canny edge detection
    edges = cv2.Canny(blurred, gradient_threshold, gradient_threshold * 2)

    return edges


def boxes_overlap(box1, box2):
    """
    Check if two boxes overlap.

    Parameters:
        box1 (tuple): Top-left and bottom-right coordinates of the first box ((x1, y1), (x2, y2)).
        box2 (tuple): Top-left and bottom-right coordinates of the second box ((x1, y1), (x2, y2)).

    Returns:
        bool: True if the boxes overlap, False otherwise.
    """
    return not (box1[1][0] <= box2[0][0] or  # box1 is to the left of box2
                box1[0][0] >= box2[1][0] or  # box1 is to the right of box2
                box1[1][1] <= box2[0][1] or  # box1 is above box2
                box1[0][1] >= box2[1][1])    # box1 is below box2


def template_match_with_edges(template_paths, target_path, scale_factor=1, threshold_ratio=0.1, max_boxes=10):
    """
    Perform template matching on edge-detected grayscale images.

    Parameters:
        template_paths (list): List of paths to template images.
        target_path (str): Path to the target image.
        scale_factor (float): Scale factor to resize the output images.
        threshold_ratio (float): Ratio to define the threshold (e.g., 0.1 for 10%).
        max_boxes (int): Maximum number of boxes to draw per template.

    Returns:
        numpy.ndarray: Target image with matched regions highlighted.
    """
    # Apply edge detection to the target image
    edges_target = detect_large_edges(target_path)

    # Initialize the result image
    result_image = cv2.cvtColor(edges_target, cv2.COLOR_GRAY2BGR)

    # Process each template
    for template_path in template_paths:
        # Apply edge detection to the template
        edges_template = detect_large_edges(template_path)

        # Get dimensions of the edge-detected template
        h, w = edges_template.shape

        # Perform template matching using grayscale edges
        result = cv2.matchTemplate(edges_target, edges_template, cv2.TM_SQDIFF)

        # Find the location of the minimum error
        min_val, _, min_loc, _ = cv2.minMaxLoc(result)
        print(f"Minimum SSD Error Value for {template_path}: {min_val}")

        # Calculate the threshold
        threshold_value = min_val * (1 + threshold_ratio)
        print(f"Threshold Value for {template_path}: {threshold_value}")

        # Get non-overlapping boxes for this template
        boxes = []
        count = 0
        for y in range(result.shape[0]):
            for x in range(result.shape[1]):
                if result[y, x] <= threshold_value:
                    top_left = (x, y)
                    bottom_right = (x + w, y + h)
                    current_box = (top_left, bottom_right)

                    # Check if the current box overlaps with any existing box
                    if any(boxes_overlap(current_box, box) for box in boxes):
                        continue

                    boxes.append(current_box)
                    count += 1

                    if count >= max_boxes:
                        break
            if count >= max_boxes:
                break

        # Draw all boxes for this template
        for top_left, bottom_right in boxes:
            cv2.rectangle(result_image, top_left, bottom_right, (0, 0, 255), 2)

        # Highlight the best match with a green rectangle
        top_left = min_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)
        cv2.rectangle(result_image, top_left, bottom_right, (0, 255, 0), 2)

    # Resize for display
    resized_result = cv2.resize(result_image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)

    return resized_result


# Paths to images
current_path = os.getcwd()
template_paths = [
    os.path.join(current_path, 'test', 'vertical1.jpg'),
    os.path.join(current_path, 'test', 'horizontal1.jpg'),
]
target_path = os.path.join(current_path, 'test', 'IMG-3650.jpg')

# Perform template matching using edge-detected grayscale images
result_image = template_match_with_edges(template_paths, target_path, scale_factor=0.2, threshold_ratio=0.2,
                                         max_boxes=10)

# Display the result
cv2.imshow("Template Match with Edge Detection", result_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
