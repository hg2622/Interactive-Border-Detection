import cv2
import os

def crop_and_display(image_path, x, y, width, height, display_scale=0.5):
    """
    Crop a rectangle from the image and display the cropped portion.

    Parameters:
        image_path (str): Path to the input image.
        x (int): X-coordinate of the top-left corner of the rectangle.
        y (int): Y-coordinate of the top-left corner of the rectangle.
        width (int): Width of the rectangle.
        height (int): Height of the rectangle.
        display_scale (float): Scale factor to resize the output for display.

    Returns:
        numpy.ndarray: Cropped image.
    """
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    print(f"Image resolution: {image.shape[:2]}")

    # Crop the rectangle
    cropped_image = image[y:y+height, x:x+width]

    # Resize the cropped image for display
    resized_cropped = cv2.resize(cropped_image, None, fx=display_scale, fy=display_scale, interpolation=cv2.INTER_AREA)

    # Display the cropped image
    cv2.imshow("Cropped Image", resized_cropped)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return cropped_image

# Example usage
current_path = os.getcwd()
path = os.path.join(current_path, 'maze_photos', '00000066.jpg')

# Crop and display a rectangle (adjust parameters as needed)
cropped = crop_and_display(path, x=100, y=200, width=1000, height=300, display_scale=0.5)
