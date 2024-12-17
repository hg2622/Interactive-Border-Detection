import cv2
import os
import numpy as np


def boxes_intersect(box1, box2):
    """
    Check if two boxes intersect.

    Parameters:
        box1 (tuple): Top-left and bottom-right coordinates of the first box ((x1, y1), (x2, y2)).
        box2 (tuple): Top-left and bottom-right coordinates of the second box ((x1, y1), (x2, y2)).

    Returns:
        bool: True if the boxes intersect, False otherwise.
    """
    return not (box1[1][0] <= box2[0][0] or  # box1 is to the left of box2
                box1[0][0] >= box2[1][0] or  # box1 is to the right of box2
                box1[1][1] <= box2[0][1] or  # box1 is above box2
                box1[0][1] >= box2[1][1])  # box1 is below box2


def template_match_with_intersections(template_paths, target_path, scale_factor=1, threshold_ratio=0.1, max_boxes=10):
    """
    Perform template matching for multiple templates and identify intersecting boxes.
    Draw both boxes that overlap with each other from different templates.

    Parameters:
        template_paths (list): List of paths to template images.
        target_path (str): Path to the target image.
        scale_factor (float): Scale factor to resize the output images.
        threshold_ratio (float): Ratio to define the threshold (e.g., 0.1 for 10%).
        max_boxes (int): Maximum number of boxes to draw per template.

    Returns:
        numpy.ndarray: Target image with matched regions highlighted.
        list: List of intersecting boxes (from different templates).
    """
    # Load the target image in color
    target = cv2.imread(target_path)
    if target is None:
        raise FileNotFoundError("Target image could not be loaded.")

    # Initialize variables
    result_image = target.copy()
    all_boxes = []  # Store boxes for all templates (template index and boxes)
    intersecting_boxes = []  # Store intersecting boxes

    # Process each template
    for template_index, template_path in enumerate(template_paths):
        # Load the current template
        template = cv2.imread(template_path)
        if template is None:
            raise FileNotFoundError(f"Template image {template_path} could not be loaded.")

        # Get dimensions of the template
        h, w, _ = template.shape

        # Perform template matching using SSD (Sum of Squared Differences)
        result_bgr = []
        for channel in range(3):  # Loop over B, G, R channels
            result = cv2.matchTemplate(target[:, :, channel], template[:, :, channel], cv2.TM_SQDIFF)
            result_bgr.append(result)

        # Combine the results from all channels (sum of SSD scores)
        combined_result = sum(result_bgr)

        # Calculate the threshold
        min_val, _, _, _ = cv2.minMaxLoc(combined_result)
        threshold_value = min_val * (1 + threshold_ratio)

        # Get non-overlapping boxes for this template
        boxes = []
        count = 0
        for y in range(combined_result.shape[0]):
            for x in range(combined_result.shape[1]):
                if combined_result[y, x] <= threshold_value:
                    top_left = (x, y)
                    bottom_right = (x + w, y + h)
                    current_box = (top_left, bottom_right)

                    # Check if the current box overlaps with any existing box
                    if any(boxes_intersect(current_box, box) for box in boxes):
                        continue

                    boxes.append(current_box)
                    count += 1

                    if count >= max_boxes:
                        break
            if count >= max_boxes:
                break

        # Add the boxes for this template to the list of all boxes
        all_boxes.append((template_index, boxes))

    # Find intersecting boxes between templates
    for i, (template_idx1, boxes1) in enumerate(all_boxes):
        for j, (template_idx2, boxes2) in enumerate(all_boxes):
            if i >= j:  # Avoid comparing the same template or redundant comparisons
                continue
            for box1 in boxes1:
                for box2 in boxes2:
                    if boxes_intersect(box1, box2):
                        intersecting_boxes.append((template_idx1, box1, template_idx2, box2))

    # Draw both intersecting boxes on the result image
    for template_idx1, box1, template_idx2, box2 in intersecting_boxes:
        top_left1, bottom_right1 = box1
        top_left2, bottom_right2 = box2
        # Draw box1 in blue
        cv2.rectangle(result_image, top_left1, bottom_right1, (255, 0, 0), 2)
        # Draw box2 in red
        cv2.rectangle(result_image, top_left2, bottom_right2, (0, 0, 255), 2)

    # Resize for display
    resized_result = cv2.resize(result_image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)

    return resized_result, intersecting_boxes

# Paths to images
current_path = os.getcwd()
template_paths = [
    os.path.join(current_path, 'test', 'IMG-3641_vertical.jpg'),
    os.path.join(current_path, 'test', 'IMG-3641_horizontal.jpg'),

]
target_path = os.path.join(current_path, 'test', 'IMG-3644.jpg')

# Perform template matching with multiple templates and find intersecting boxes
result_image, intersecting_boxes = template_match_with_intersections(template_paths, target_path, scale_factor=0.2,
                                                                     threshold_ratio=0.1, max_boxes=10)

# Display the result
cv2.imshow("Template Match (Intersecting Boxes)", result_image)
cv2.waitKey(0)
cv2.destroyAllWindows()



