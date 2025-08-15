import numpy as np
import cv2
from collections import Counter

def post_speckle_cleanup(image_bgr, max_area=8):
    """
    Removes extremely small speckle regions (< max_area) by replacing them with surrounding majority color.

    Args:
        image_bgr (np.ndarray): Input BGR image with flat color blocks.
        max_area (int): Maximum size of region to treat as a speckle.

    Returns:
        np.ndarray: Cleaned image.
    """
    h, w = image_bgr.shape[:2]
    flat = image_bgr.reshape(-1, 3)
    unique_colors, inverse = np.unique(flat, axis=0, return_inverse=True)
    label_map = inverse.reshape((h, w))
    cleaned = image_bgr.copy()

    for idx, color in enumerate(unique_colors):
        mask = (label_map == idx).astype(np.uint8)
        num_labels, cc_labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area > max_area:
                continue

            region_mask = (cc_labels == i)
            dilated = cv2.dilate(region_mask.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1)
            border = (dilated.astype(bool)) & (~region_mask)
            neighbors = image_bgr[border]

            if len(neighbors) > 0:
                new_color = Counter(map(tuple, neighbors)).most_common(1)[0][0]
                cleaned[region_mask] = new_color
    return cleaned