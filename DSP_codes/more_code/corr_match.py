import cv2
import os
import numpy as np

def boxes_overlap(box1, box2):
    """
    Check if two boxes overlap.

    Parameters:
        box1 (tuple): Top-left and bottom-right coordinates of the first box ((x1, y1), (x2, y2)).
        box2 (tuple): Top-left and bottom-right coordinates of the second box ((x1, y1), (x2, y2)).

    Returns:
        bool: True if the boxes overlap, False otherwise.
    """
    return not (box1[1][0] <= box2[0][0] or  # box1 is to the left of box2
                box1[0][0] >= box2[1][0] or  # box1 is to the right of box2
                box1[1][1] <= box2[0][1] or  # box1 is above box2
                box1[0][1] >= box2[1][1])  # box1 is below box2

def template_match_with_threshold(template, target, threshold_value, w, h, max_boxes):
    """
    Perform template matching and return a list of non-overlapping boxes.

    Parameters:
        template (numpy.ndarray): Template image.
        target (numpy.ndarray): Target image.
        threshold_value (float): Minimum acceptable correlation for matches.
        w (int): Width of the template.
        h (int): Height of the template.
        max_boxes (int): Maximum number of boxes to draw.

    Returns:
        list: List of non-overlapping boxes (top-left and bottom-right coordinates).
    """
    combined_result = template
    boxes = []
    count = 0

    for y in range(combined_result.shape[0]):
        for x in range(combined_result.shape[1]):
            if combined_result[y, x] >= threshold_value:
                top_left = (x, y)
                bottom_right = (x + w, y + h)
                current_box = (top_left, bottom_right)

                if any(boxes_overlap(current_box, box) for box in boxes):
                    continue

                boxes.append(current_box)
                count += 1

                if count >= max_boxes:
                    break
        if count >= max_boxes:
            break

    return boxes

def template_match_multiple_templates(template_paths, target_path, scale_factor=1, threshold_ratio=0.8, max_boxes=10):
    """
    Perform template matching for multiple templates and draw all matching regions.

    Parameters:
        template_paths (list): List of paths to template images.
        target_path (str): Path to the target image.
        scale_factor (float): Scale factor to resize the output images.
        threshold_ratio (float): Ratio to define the threshold (e.g., 0.8 for 80% correlation).
        max_boxes (int): Maximum number of boxes to draw per template.

    Returns:
        numpy.ndarray: Target image with matched regions highlighted.
    """
    target = cv2.imread(target_path)
    if target is None:
        raise FileNotFoundError("Target image could not be loaded.")

    result_image = target.copy()

    for template_path in template_paths:
        template = cv2.imread(template_path)
        if template is None:
            raise FileNotFoundError(f"Template image {template_path} could not be loaded.")

        # Resize the template to 70% of its original size
        h, w, _ = template.shape
        resized_template = cv2.resize(template, (int(w * 0.7), int(h * 0.7)), interpolation=cv2.INTER_AREA)
        h, w, _ = resized_template.shape

        # Perform template matching using normalized cross-correlation
        result_bgr = []
        for channel in range(3):
            result = cv2.matchTemplate(target[:, :, channel], resized_template[:, :, channel], cv2.TM_CCORR_NORMED)
            result_bgr.append(result)

        combined_result = sum(result_bgr) / len(result_bgr)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(combined_result)
        print(f"Maximum Correlation Value for {template_path}: {max_val}")

        threshold_value = max_val * threshold_ratio
        print(f"Threshold Value for {template_path}: {threshold_value}")

        boxes = template_match_with_threshold(combined_result, target, threshold_value, w, h, max_boxes)

        for top_left, bottom_right in boxes:
            cv2.rectangle(result_image, top_left, bottom_right, (0, 0, 255), 2)

        top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)
        cv2.rectangle(result_image, top_left, bottom_right, (0, 255, 0), 2)

    resized_result = cv2.resize(result_image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)
    return resized_result

# Paths to images
current_path = os.getcwd()
template_paths = [
    os.path.join(current_path, 'test', 'IMG-3641_right_vertical.jpg'),
    os.path.join(current_path, 'test', 'IMG-3641_left_vertical.jpg'),
os.path.join(current_path, 'test', 'IMG-3651_top.jpg'),
os.path.join(current_path, 'test', 'IMG-3652_bottom.jpg')

]
target_path = os.path.join(current_path, 'test', 'IMG-3649.jpg')

# Perform template matching with resized templates
result_image = template_match_multiple_templates(template_paths, target_path, scale_factor=0.2, threshold_ratio=0.8,
                                                 max_boxes=2)

cv2.imshow("Template Match with Resized Templates", result_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
