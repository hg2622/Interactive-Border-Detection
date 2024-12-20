import cv2
import os
import threading
import queue
from window_detection import apply_uniformity_filter, fit_hough_lines, scale_image
from ground_detection import detect_large_edges, perform_hough_lines, filter_noise
from color_masking import detect_brown_mask
from high_pass_filter import high_pass_filter

# Directories for input and output
current_path = os.getcwd()
save_directory = os.path.join(current_path, 'DSP_codes', 'images')
output_directory = os.path.join(current_path, 'DSP_codes', 'output')
os.makedirs(output_directory, exist_ok=True)

# Thread-safe queue to hold frames
frame_queue = queue.Queue(maxsize=10)  # Buffer for frames

# Save rate for frames
saving_rate = 5
brown_lower = (10, 50, 50)
brown_upper = (30, 255, 255)

# Camera thread: Captures and saves frames
def camera_thread():
    cap = cv2.VideoCapture(0)  # Open the default webcam
    count = 0

    if not cap.isOpened():
        print("Error: Unable to access the camera.")
        return

    print("Camera started. Press 'q' in the camera window to quit.")

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok or frame is None:
            print("Error: Unable to read video frame.")
            break

        try:
            # Process frame as in camera.py
            frame_resized = cv2.resize(frame, None, fx=0.5, fy=0.5)
            gray_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
            edges = detect_large_edges(frame, sigma=4, gradient_threshold=30, kernel_size=3, min_contour_length=250)
            resized_edges = cv2.resize(edges, None, fx=0.5, fy=0.5)
            brown_mask = detect_brown_mask(frame, brown_range=(brown_lower, brown_upper))
            brown_mask = cv2.resize(brown_mask, None, fx=0.5, fy=0.5)
            high_pass_res = high_pass_filter(frame, 7)
            high_pass_res = cv2.convertScaleAbs(high_pass_res)
            high_pass_res = cv2.resize(high_pass_res, None, fx=0.5, fy=0.5)

            top_row = cv2.hconcat([gray_frame, resized_edges])  # Combine top row
            bottom_row = cv2.hconcat([brown_mask, high_pass_res])  # Combine bottom row
            combined_output = cv2.vconcat([top_row, bottom_row])  # Create 2x2 grid

            # Display live feed
            cv2.imshow('Live video - Camera', combined_output)

            # Save frames periodically
            if count % saving_rate == 0:
                frame_queue.put((count, frame))  # Add frame to the queue
                image_name = os.path.join(save_directory, f"frame_{count:04d}.jpg")
                cv2.imwrite(image_name, frame)
                print(f"Saved: {image_name}")
            count += 1

        except Exception as e:
            print(f"Error in camera processing: {e}")

        # Quit if 'q' is pressed
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Worker thread: Processes frames for window detection
def window_detection_thread():
    while True:
        if not frame_queue.empty():
            count, frame = frame_queue.get()
            try:
                block_size = 15
                threshold = 50

                # Apply uniformity filter
                uniform_mask = apply_uniformity_filter(frame, block_size=block_size, threshold=threshold)

                # Scale for Hough line fitting
                scaled_frame = scale_image(frame, scale=0.5)
                result_window = fit_hough_lines(uniform_mask, scaled_frame)

                # Save the result
                output_path = os.path.join(output_directory, f"window_frame_{count:04d}.jpg")
                cv2.imwrite(output_path, result_window)
                print(f"Window Detection Saved: {output_path}")
            except Exception as e:
                print(f"Error in window detection: {e}")

# Worker thread: Processes frames for ground detection
def ground_detection_thread():
    while True:
        if not frame_queue.empty():
            count, frame = frame_queue.get()
            try:
                # Parameters for ground detection
                sigma = 4
                gradient_threshold = 30
                kernel_size = 3
                min_contour_length = 250

                # Detect large edges
                edges = detect_large_edges(frame, sigma=sigma, gradient_threshold=gradient_threshold, kernel_size=kernel_size, min_contour_length=min_contour_length)

                # Apply Hough lines
                hough_lines = perform_hough_lines(edges, frame)

                # Filter noise
                filtered_result = filter_noise(hough_lines, min_area=500)

                # Save the result
                output_path = os.path.join(output_directory, f"ground_frame_{count:04d}.jpg")
                cv2.imwrite(output_path, filtered_result)
                print(f"Ground Detection Saved: {output_path}")
            except Exception as e:
                print(f"Error in ground detection: {e}")

# Start threads
camera_thread_handle = threading.Thread(target=camera_thread)
window_thread_handle = threading.Thread(target=window_detection_thread, daemon=True)
ground_thread_handle = threading.Thread(target=ground_detection_thread, daemon=True)

camera_thread_handle.start()
window_thread_handle.start()
ground_thread_handle.start()

camera_thread_handle.join()
