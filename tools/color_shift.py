import cv2
import numpy as np

YELLOW_HUE_RANGE = (20, 40)  # Approximate HSV hue range for yellow

def shift_yellow(image_bgr, hue_shift=60):
    """
    Detect and shift yellow hues in the image toward cooler tones.

    Args:
        image_bgr: Input image in BGR format (OpenCV)
        hue_shift: Degrees to shift hue (e.g. 60 → green/cyan)

    Returns:
        Modified image in BGR
    """
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # Create mask for yellow hue range
    mask = (h > YELLOW_HUE_RANGE[0]) & (h < YELLOW_HUE_RANGE[1])

    # Apply hue shift only to yellow areas
    h[mask] = (h[mask] + hue_shift) % 180

    shifted = cv2.merge([h, s, v])
    return cv2.cvtColor(shifted, cv2.COLOR_HSV2BGR)