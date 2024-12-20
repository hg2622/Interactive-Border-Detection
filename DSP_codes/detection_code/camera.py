import cv2
import os
import numpy as np
from boder_detection import detect_large_edges  # Import the function from the previous file
from color_masking import detect_brown_mask
from high_pass_filter import high_pass_filter

# video_capture_2.py
# Capture video and display it on the screen in real-time
# Reduce size of frame 

brown_lower = (10, 50, 50)
brown_upper = (30, 255, 255)
current_path = os.getcwd()
save_directory = os.path.join(current_path, 'DSP_codes','images')

cap = cv2.VideoCapture(0)  # Open the default webcam
#saving every 5 frames 
saving_rate = 5
count = 0

if not cap.isOpened():
    print("Error: Unable to access the camera.")
    exit()

print("Switch to video window. Then press 'q' to quit")

while cap.isOpened():
    # Capture a frame
    ok, frame = cap.read()
    if not ok or frame is None:
        print("Error: Unable to read video frame.")
        break

    try:
        # Pass the captured frame directly to detect_large_edges
        edges = detect_large_edges(frame, sigma=4, gradient_threshold=30, kernel_size=3, display_scale=1.0, min_contour_length=250)

        # Resize the result for display
        frame_resized = cv2.resize(frame, None, fx=0.5, fy=0.5)
        gray_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
        resized_edges = cv2.resize(edges, None, fx=0.5, fy=0.5)
        brown_mask = detect_brown_mask(frame,brown_range=(brown_lower, brown_upper))
        brown_mask = cv2.resize(brown_mask, None, fx=0.5, fy=0.5)
        high_pass_res = high_pass_filter(frame, 7)
        high_pass_res = cv2.convertScaleAbs(high_pass_res)
        high_pass_res = cv2.resize(high_pass_res, None, fx=0.5, fy=0.5)
        
        top_row = cv2.hconcat([gray_frame, resized_edges])  # Combine top row
        bottom_row = cv2.hconcat([brown_mask, high_pass_res])  # Combine bottom row
        # Combine rows to create a 2x2 grid
        combined_output = cv2.vconcat([top_row, bottom_row])        
        
        # Show the processed frame
        cv2.imshow('Live video - Edge Detection', combined_output)
        
        if count % saving_rate == 0:
            image_name = os.path.join(save_directory, f"frame_{count:04d}.jpg")
            cv2.imwrite(image_name, frame)
            print(f"Saved: {image_name}")
        count += 1


    except Exception as e:
        print(f"Error during edge detection: {e}")
        break

    # Quit if 'q' is pressed
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
