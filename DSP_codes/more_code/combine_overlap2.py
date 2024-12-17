import cv2
import os
import numpy as np


# === Hough's Method Code ===
def detect_large_edges(image_path, sigma=1.0, gradient_threshold=100, kernel_size=9, min_contour_length=20):
    original_image = cv2.imread(image_path)
    if original_image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    grayscale_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian Blur to reduce noise
    blurred = cv2.GaussianBlur(grayscale_image, (0, 0), sigma)

    # Compute gradients
    grad_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=kernel_size)
    grad_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=kernel_size)

    # Gradient magnitude and thresholding
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    gradient_magnitude = cv2.convertScaleAbs(gradient_magnitude)
    _, large_edges = cv2.threshold(gradient_magnitude, gradient_threshold, 255, cv2.THRESH_BINARY)

    # Find contours and filter by length
    contours, _ = cv2.findContours(large_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_edges = np.zeros_like(large_edges)
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


# === Uniform Mask + Brown Mask Code ===
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
    return output_mask


def detect_brown_mask(image_path, brown_range):
    original_image = cv2.imread(image_path)
    if original_image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    hsv_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv_image, brown_range[0], brown_range[1])


def fit_hough_lines(binary_mask, original_image):
    edges = cv2.Canny(binary_mask, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180, threshold=50, minLineLength=30, maxLineGap=10)
    result_image = original_image.copy()
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green lines
    return result_image


# === Combined Code to Find Intersection ===
def main():
    # File path
    current_path = os.getcwd()
    image_path = os.path.join(current_path, 'test', 'IMG-3649.jpg')  # Replace with your path

    # === Hough's Method Border ===
    filtered_edges = detect_large_edges(image_path, sigma=4, gradient_threshold=30, kernel_size=3, min_contour_length=250)
    hough_lines = perform_hough_lines(filtered_edges)

    # === Uniform Mask + Brown Mask Border ===
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    image_scaled = cv2.resize(image, (0, 0), fx=0.2, fy=0.2)  # Scale down for faster processing

    # Uniformity filter
    uniform_filter_mask = apply_uniformity_filter(image_scaled, block_size=10, threshold=50)

    # Brown mask
    brown_lower = (10, 50, 50)
    brown_upper = (30, 255, 255)
    brown_mask = detect_brown_mask(image_path, brown_range=(brown_lower, brown_upper))
    brown_mask_scaled = cv2.resize(brown_mask, (uniform_filter_mask.shape[1], uniform_filter_mask.shape[0]))

    # Intersection of Uniformity and Brown Mask
    uniform_brown_intersection = cv2.bitwise_and(uniform_filter_mask, brown_mask_scaled)

    # Fit Hough lines to the intersection
    uniform_hough_result = fit_hough_lines(uniform_brown_intersection, image_scaled)

    # === Find Final Intersection ===
    hough_lines_scaled = cv2.resize(hough_lines, (uniform_brown_intersection.shape[1], uniform_brown_intersection.shape[0]))
    final_intersection = cv2.bitwise_and(hough_lines_scaled, uniform_brown_intersection)

    # === Display Results ===
    cv2.imshow("Hough Method Border", hough_lines)
    cv2.imshow("Uniform + Brown Mask Border", uniform_brown_intersection)
    cv2.imshow("Final Intersection", final_intersection)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
