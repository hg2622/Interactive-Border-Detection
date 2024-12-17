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


def perform_hough_lines(edge_image):
    lines = cv2.HoughLinesP(edge_image, rho=1, theta=np.pi/180, threshold=80, minLineLength=100, maxLineGap=40)
    line_mask = np.zeros_like(edge_image)

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_mask, (x1, y1), (x2, y2), 255, 2)

    return line_mask


def scale_image(image, scale):
    """Scales the image by the given factor."""
    width = int(image.shape[1] * scale)
    height = int(image.shape[0] * scale)
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


# File path
current_path = os.getcwd()
image_path = os.path.join(current_path, 'test', 'IMG-3644.jpg')

# Load original image and detect edges
filtered_edges = detect_large_edges(image_path, sigma=4, gradient_threshold=30, kernel_size=3, min_contour_length=250)

# Perform Hough Transform
hough_lines = perform_hough_lines(filtered_edges)

# Scale the Hough result for display
hough_scaled = scale_image(hough_lines, scale=0.2)

# Display the scaled Hough image
cv2.imshow("Hough Scaled Image", hough_scaled)
cv2.waitKey(0)
cv2.destroyAllWindows()
