import cv2
import os
import numpy as np


def detect_large_edges(input_image, sigma=1.0, gradient_threshold=100, kernel_size=9, display_scale=0.2, min_contour_length=20):
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
        numpy.ndarray: Edge-detected image resized for display.
    """
    # Load the image in grayscale
    #original_image = cv2.imread(image_path)
    #if original_image is None:
        #raise FileNotFoundError(f"Image not found at {image_path}")
    grayscale_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

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


# Paths to images
current_path = os.getcwd()
image_path = os.path.join(current_path, 'DSP_codes','photos', 'IMG-3644.jpg')

# Perform edge detection
#edges_resized = detect_large_edges(image_path, sigma=4, gradient_threshold=30, kernel_size=3, display_scale=0.2, min_contour_length=250)

# Display the edge-detected image
#cv2.imshow("Edge Detection with Contour Filtering", edges_resized)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
