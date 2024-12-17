import cv2
import os
import numpy as np

def resize_template_to_fit(template, target_shape):
    """
    Resize the template if it is larger than the target shape.

    Parameters:
        template (numpy.ndarray): Original template image.
        target_shape (tuple): Shape (height, width) of the target image.

    Returns:
        numpy.ndarray: Resized template.
    """
    template_height, template_width = template.shape[:2]
    target_height, target_width = target_shape

    if template_height > target_height or template_width > target_width:
        scale = min(target_height / template_height, target_width / template_width)
        new_size = (int(template_width * scale), int(template_height * scale))
        return cv2.resize(template, new_size, interpolation=cv2.INTER_AREA)
    return template

def expand_template(template, target_shape):
    """
    Expand the template into a larger white image.

    Parameters:
        template (numpy.ndarray): Original template image.
        target_shape (tuple): Shape (height, width) of the larger template.

    Returns:
        numpy.ndarray: Expanded template.
    """
    expanded_template = 255 * np.ones((target_shape[0], target_shape[1]), dtype=np.uint8)
    template = resize_template_to_fit(template, target_shape)

    y_offset = (target_shape[0] - template.shape[0]) // 2
    x_offset = (target_shape[1] - template.shape[1]) // 2

    expanded_template[y_offset:y_offset + template.shape[0], x_offset:x_offset + template.shape[1]] = template
    return expanded_template

def find_template(template_path, target_path, scale_factor=0.5):
    """
    Find a template image within a target image using feature-based matching with SIFT.

    Parameters:
        template_path (str): Path to the template image.
        target_path (str): Path to the target image.
        scale_factor (float): Scale factor for resizing the output images.

    Returns:
        tuple: Match visualization image, result image with matched region.
    """
    template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    target = cv2.imread(target_path, cv2.IMREAD_GRAYSCALE)

    if template is None or target is None:
        raise FileNotFoundError("One or both images could not be loaded.")

    target_height, target_width = target.shape
    expanded_template = expand_template(template, (target_height // 2, target_width // 2))

    sift = cv2.SIFT_create()
    keypoints1, descriptors1 = sift.detectAndCompute(expanded_template, None)
    keypoints2, descriptors2 = sift.detectAndCompute(target, None)

    if descriptors1 is None or descriptors2 is None:
        raise ValueError("Could not compute descriptors for one or both images.")

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)

    # Apply ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    match_image = cv2.drawMatches(expanded_template, keypoints1, target, keypoints2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    if len(good_matches) > 4:
        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        h, w = expanded_template.shape
        template_corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
        if M is not None:
            transformed_corners = cv2.perspectiveTransform(template_corners, M)
            result_image = cv2.polylines(target.copy(), [np.int32(transformed_corners)], True, 255, 3, cv2.LINE_AA)
        else:
            result_image = target.copy()
            print("No homography could be computed.")
    else:
        result_image = target.copy()
        print("Not enough good matches found to compute homography.")

    # Resize images for display
    match_image_resized = cv2.resize(match_image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)
    result_image_resized = cv2.resize(result_image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)

    return match_image_resized, result_image_resized

# Paths to images
current_path = os.getcwd()
template_path = os.path.join(current_path, 'test', 'IMG-3650.jpg')
target_path = os.path.join(current_path, 'test', 'IMG-3644.jpg')

# Find and display the template in the target image
try:
    match_image, result_image = find_template(template_path, target_path, scale_factor=0.2)
    cv2.imshow("Matches", match_image)
    cv2.imshow("Detected Template", result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
except Exception as e:
    print(f"An error occurred: {e}")
