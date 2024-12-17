import cv2
import os
import numpy as np


def detect_large_edges(image_path, sigma=1.0, gradient_threshold=100, kernel_size=9, min_contour_length=20):
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

    return filtered_edges


def perform_hough_lines(edge_image, original_image):
    lines = cv2.HoughLinesP(edge_image, rho=1, theta=np.pi/180, threshold=80, minLineLength=100, maxLineGap=40)
    line_mask = np.zeros_like(edge_image)

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_mask, (x1, y1), (x2, y2), 255, 2)

    return line_mask


def detect_color_transition(image_path, brown_range, black_range):
    original_image = cv2.imread(image_path)
    if original_image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")

    # Convert to HSV color space
    hsv_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2HSV)

    # Create masks for brown and black regions
    brown_mask = cv2.inRange(hsv_image, brown_range[0], brown_range[1])
    black_mask = cv2.inRange(hsv_image, black_range[0], black_range[1])

    # Dilate both masks
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
    brown_dilated = cv2.dilate(brown_mask, kernel, iterations=1)
    black_dilated = cv2.dilate(black_mask, kernel, iterations=1)

    # Find overlap
    transition_region = cv2.bitwise_and(brown_dilated, black_dilated)

    return transition_region


def filter_noise(mask, min_area=500):
    """
    Filters out small noisy regions from a binary mask based on contour area.

    Parameters:
        mask (numpy.ndarray): Binary mask to filter.
        min_area (int): Minimum contour area to retain.

    Returns:
        numpy.ndarray: Filtered binary mask.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_mask = np.zeros_like(mask)

    for contour in contours:
        if cv2.contourArea(contour) >= min_area:  # Keep only large contours
            cv2.drawContours(filtered_mask, [contour], -1, 255, thickness=cv2.FILLED)

    return filtered_mask


def scale_image(image, scale):
    """Scales the image by the given factor."""
    width = int(image.shape[1] * scale)
    height = int(image.shape[0] * scale)
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def combine_and_display(image_path, brown_range, black_range, display_scale=0.2, min_area=500):
    original_image = cv2.imread(image_path)
    if original_image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")

    # Detect large edges
    filtered_edges = detect_large_edges(image_path, sigma=4, gradient_threshold=30, kernel_size=3, min_contour_length=250)

    # Detect Hough lines
    hough_lines = perform_hough_lines(filtered_edges, original_image)

    # Detect ground regions
    ground_mask = detect_color_transition(image_path, brown_range, black_range)

    # Check overlap
    overlap = cv2.bitwise_and(hough_lines, ground_mask)

    # Make the overlap dots bigger using dilation
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))  # Adjust size for larger dots
    overlap_dilated = cv2.dilate(overlap, kernel, iterations=1)

    # Filter out noise from the overlap
    filtered_overlap = filter_noise(overlap_dilated, min_area=min_area)

    # Create result images
    result_image = original_image.copy()
    result_image[filtered_overlap > 0] = [0, 255, 0]  # Highlight strong lines in green

    overlap_only_image = np.zeros_like(original_image)
    overlap_only_image[filtered_overlap > 0] = [0, 255, 0]  # Green dots on a black background

    # Scale all images for display
    original_scaled = scale_image(original_image, display_scale)
    hough_scaled = scale_image(hough_lines, display_scale)
    ground_scaled = scale_image(ground_mask, display_scale)
    result_scaled = scale_image(result_image, display_scale)
    overlap_only_scaled = scale_image(overlap_only_image, display_scale)

    # Display the images
    #cv2.imshow("Original Image", original_scaled)
    cv2.imshow("Hough Lines", hough_scaled)
    cv2.imshow("Ground Mask", ground_scaled)
    #cv2.imshow("Filtered Overlap Result", result_scaled)
    cv2.imshow("Filtered Overlap Only", overlap_only_scaled)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Define color ranges for ground detection
brown_lower = (10, 50, 50)
brown_upper = (30, 255, 255)
black_lower = (0, 0, 0)
black_upper = (180, 255, 50)

# File path
current_path = os.getcwd()
image_path = os.path.join(current_path, 'test', 'IMG-3649.jpg')

# Run the combined detection and display the result
combine_and_display(image_path, brown_range=(brown_lower, brown_upper), black_range=(black_lower, black_upper), display_scale=0.2, min_area=500)
