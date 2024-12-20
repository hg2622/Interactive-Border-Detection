import os
from ground_detect import *

# File path
current_path = os.getcwd()
image_path = os.path.join(current_path,'DSP_codes','images')  # Replace with your path
# Define color ranges for ground detection
brown_lower = (10, 50, 50)
brown_upper = (30, 255, 255)
black_lower = (0, 0, 0)
black_upper = (180, 255, 50)


for element in os.listdir(image_path):
    if element.lower().endswith('jpeg'):
        single_imagepath = os.path.join(image_path,element)
        image = cv2.imread(single_imagepath)
        if image is None:
            raise FileNotFoundError(f"Image not found at {image_path}")
        
        # Run the combined detection and display the result
        res = combine_and_display(single_imagepath, brown_range=(brown_lower, brown_upper), black_range=(black_lower, black_upper), display_scale=0.2, min_area=500)
        save_path = os.path.join(current_path,'DSP_codes','ground',element)
        cv2.imwrite(save_path,res)
