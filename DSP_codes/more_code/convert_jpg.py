import cv2
import os
def convert_png_to_jpg(png_path, jpg_path, quality=95):
    """
    Convert a PNG image to JPG format.

    Parameters:
        png_path (str): Path to the PNG image.
        jpg_path (str): Path to save the converted JPG image.
        quality (int): Compression quality (0-100), higher is better quality.
    """
    # Read the PNG image
    png_image = cv2.imread(png_path)

    # Convert and save as JPG
    cv2.imwrite(jpg_path, png_image, [cv2.IMWRITE_JPEG_QUALITY, quality])

# Example usage
convert_png_to_jpg("image.png", "image.jpg", quality=95)

current_path = os.getcwd()
path = os.path.join(current_path, 'maze_photos', '00000190.jpg')