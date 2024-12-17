import cv2
import os
import numpy as np

def detect_large_edges(image_path, sigma=1.0, gradient_threshold=100, kernel_size=9, min_contour_length=20):
    """
    Detect large edges in an image using the gradient method and filter noise using contour filtering.
    """
    # Load the image in grayscale and color
    original_image = cv2.imread(image_path)
    grayscale_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    if original_image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    print(f"Image resolution: {original_image.shape[:2]}")

    # Apply Gaussian Blur to reduce noise
    blurred = cv2.GaussianBlur(grayscale_image, (0, 0), sigma)

    # Compute gradients using the Sobel operator with a tunable kernel size
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
        if cv2.arcLength(contour, closed=False) > min_contour_length:  # Filter by length
            cv2.drawContours(filtered_edges, [contour], -1, 255, thickness=cv2.FILLED)

    return original_image, filtered_edges

def detect_and_mark_corners_from_edges(edges_image, max_corners=100, quality_level=0.01, min_distance=10):
    """
    Detect and mark corners based on the edge-detected image using Shi-Tomasi Corner Detection.
    """
    # Detect corners using Shi-Tomasi method
    corners = cv2.goodFeaturesToTrack(edges_image, maxCorners=max_corners, qualityLevel=quality_level, minDistance=min_distance)
    corners = np.int8(corners)

    # Convert edges to BGR to visualize corners
    edges_colored = cv2.cvtColor(edges_image, cv2.COLOR_GRAY2BGR)

    # Mark corners on the image
    for corner in corners:
        x, y = corner.ravel()
        cv2.circle(edges_colored, (x, y), 5, (0, 0, 255), -1)  # Mark corners in red

    return edges_colored

# Construct the image path
current_path = os.getcwd()
image_path = os.path.join(current_path, 'test', 'IMG-3644.jpg')

# Detect edges
original_image, edges_detected = detect_large_edges(
    image_path, sigma=1.2, gradient_threshold=50, kernel_size=3, min_contour_length=250
)

# Detect and mark corners
corners_detected = detect_and_mark_corners_from_edges(edges_detected, max_corners=50, quality_level=0.1, min_distance=20)

# Resize images for consistent height
display_scale = 0.3
original_resized = cv2.resize(original_image, None, fx=display_scale, fy=display_scale, interpolation=cv2.INTER_AREA)
corners_detected_resized = cv2.resize(corners_detected, (original_resized.shape[1], original_resized.shape[0]), interpolation=cv2.INTER_AREA)

# Combine images for display
combined = np.hstack((original_resized, corners_detected_resized))

# Display the result
cv2.imshow("Original, Edges, and Corners", combined)
cv2.waitKey(0)
cv2.destroyAllWindows()
