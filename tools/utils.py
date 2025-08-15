import cv2
import numpy as np
import json
from pathlib import Path
from datetime import datetime

def load_image(path, color_mode=cv2.IMREAD_COLOR):
    """
    Loads an image from disk with error handling.
    """
    img = cv2.imread(str(path), color_mode)
    if img is None:
        print(f"[!] Could not load image: {path}")
    return img

def save_feedback_log(log_data, feedback_path="feedback/feedback_log.json"):
    """
    Appends feedback entry to JSON file.
    """
    path = Path(feedback_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        with open(path, "r") as f:
            logs = json.load(f)
    else:
        logs = []

    logs.append(log_data)

    with open(path, "w") as f:
        json.dump(logs, f, indent=2)

def get_dominant_hue(image_bgr):
    """
    Returns the dominant hue from the image (0–179 HSV scale).
    """
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    h, _, _ = cv2.split(hsv)
    hist = cv2.calcHist([h], [0], None, [180], [0, 180])
    dominant_hue = int(np.argmax(hist))
    return dominant_hue

def get_timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")