import os
from window_detection import *

# File path
current_path = os.getcwd()
image_path = os.path.join(current_path,'DSP_codes','images')  # Replace with your path

# Parameters for uniformity filter and brown mask
block_size = 15
threshold = 50
brown_lower = (10, 50, 50)
brown_upper = (30, 255, 255)

for element in os.listdir(image_path):
    if element.lower().endswith('jpeg'):
        # Load the image and scale it
        single_imagepath = os.path.join(image_path,element)
        image = cv2.imread(single_imagepath)
        if image is None:
            raise FileNotFoundError(f"Image not found at {image_path}")
        image = scale_image(image, scale=0.2)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply uniformity filter
        uniform_filter_mask = apply_uniformity_filter(image_rgb, block_size=block_size, threshold=threshold)

        # Apply brown mask
        brown_mask = detect_brown_mask(single_imagepath, brown_range=(brown_lower, brown_upper))
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
        save_path = os.path.join(current_path,'DSP_codes','window',element)
        cv2.imwrite(save_path,hough_result_image)

