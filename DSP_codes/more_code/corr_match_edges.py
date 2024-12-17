import cv2
import os
import numpy as np

def detect_large_edges(image_path, sigma=1.0, gradient_threshold=100, kernel_size=9, display_scale=1.0, min_contour_length=20):
    """
    Detect large edges in an image using the gradient method and filter noise using contour filtering.

    Parameters:
        image_path (str): Path to the input image.
        sigma (float): Gaussian blur standard deviation.
        gradient_threshold (int): Threshold to filter weak gradients.
        kernel_size (int): Size of the Sobel operator kernel (must be odd).
        display_scale (float): Scale factor to resize the output for display.
        min_contour_length (int): Minimum contour length to retain.

    Returns:
        numpy.ndarray: Edge-detected image.
    """
    # Load the image in grayscale
    original_image = cv2.imread(image_path)
    if original_image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    grayscale_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian Blur to reduce noise
    blurred = cv2.GaussianBlur(grayscale_image, (0, 0), sigma)

    # Compute gradients using the Sobel operator
    grad_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=kernel_size)
    grad_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=kernel_size)

    # Compute gradient magnitude
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    gradient_magnitude = cv2.convertScaleAbs(gradient_magnitude)

    # Apply threshold to filter weak gradients
    _, large_edges = cv2.threshold(gradient_magnitude, gradient_threshold, 255, cv2.THRESH_BINARY)

    # Find contours in the edge-detected image
    contours, _ = cv2.findContours(large_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create an empty mask to draw filtered contours
    filtered_edges = np.zeros_like(large_edges)

    # Filter contours by length and draw them on the mask
    for contour in contours:
        if cv2.arcLength(contour, closed=False) > min_contour_length:
            cv2.drawContours(filtered_edges, [contour], -1, 255, thickness=cv2.FILLED)

    # Resize the edges image for display
    edges_resized = cv2.resize(filtered_edges, None, fx=display_scale, fy=display_scale, interpolation=cv2.INTER_AREA)

    return edges_resized

def template_match_multiple_templates(template_paths, target_edges, scale_factor=1, threshold_ratio=0.8, max_boxes=10):
    """
    Perform template matching for multiple templates on edge-detected images.

    Parameters:
        template_paths (list): List of paths to template images.
        target_edges (numpy.ndarray): Edge-detected target image.
        scale_factor (float): Scale factor to resize the output images.
        threshold_ratio (float): Ratio to define the threshold (e.g., 0.8 for 80% correlation).
        max_boxes (int): Maximum number of boxes to draw per template.

    Returns:
        numpy.ndarray: Target image with matched regions highlighted.
    """
    result_image = cv2.cvtColor(target_edges, cv2.COLOR_GRAY2BGR)

    for template_path in template_paths:
        template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
        if template is None:
            raise FileNotFoundError(f"Template image {template_path} could not be loaded.")

        # Perform template matching using normalized cross-correlation
        result = cv2.matchTemplate(target_edges, template, cv2.TM_CCORR_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        print(f"Maximum Correlation Value for {template_path}: {max_val}")

        threshold_value = max_val * threshold_ratio
        print(f"Threshold Value for {template_path}: {threshold_value}")

        # Extract boxes above the threshold
        h, w = template.shape
        boxes = []
        for y in range(result.shape[0]):
            for x in range(result.shape[1]):
                if result[y, x] >= threshold_value:
                    top_left = (x, y)
                    bottom_right = (x + w, y + h)
                    boxes.append((top_left, bottom_right))

                    if len(boxes) >= max_boxes:
                        break
            if len(boxes) >= max_boxes:
                break

        # Draw rectangles around the matches
        for top_left, bottom_right in boxes:
            cv2.rectangle(result_image, top_left, bottom_right, (0, 0, 255), 2)

    resized_result = cv2.resize(result_image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)
    return resized_result

# Paths to images
current_path = os.getcwd()
template_paths = [
    os.path.join(current_path, 'test', 'up_left_corner.jpg'),
    os.path.join(current_path, 'test', 'up_right_corner.jpg'),
    os.path.join(current_path, 'test', 'bottom_left_corner.jpg'),
    os.path.join(current_path, 'test', 'bottom_right_corner.jpg')
]
target_path = os.path.join(current_path, 'test', 'IMG-3651.jpg')

# Perform border detection
target_edges = detect_large_edges(target_path, sigma=1.2, gradient_threshold=50, kernel_size=3, display_scale=1.0, min_contour_length=250)

# Perform template matching on edge-detected result
result_image = template_match_multiple_templates(template_paths, target_edges, scale_factor=0.5, threshold_ratio=0.8, max_boxes=2)

# Display the result
cv2.imshow("Template Match on Edge Detection", result_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
