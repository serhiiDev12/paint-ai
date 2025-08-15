import numpy as np
import cv2
from collections import Counter
import aiohttp
from concurrent.futures import ThreadPoolExecutor

async def report_status(guid: str, message: str):
    callback_url = "http://localhost:5161/api/status-callback"
    async with aiohttp.ClientSession() as session:
        try:
            await session.post(callback_url, json={
                "id": guid,
                "status": message
            })
        except Exception as e:
            print(f"[STATUS-CALLBACK] Failed to report: {e}")

def _process_color(idx, color, label_map, image_bgr, max_region_area, guid=None):
    mask = (label_map == idx).astype(np.uint8)
    num_labels, cc_labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    changes = []

    for i in range(1, num_labels):  # skip background
        area = stats[i, cv2.CC_STAT_AREA]
        if area > max_region_area:
            continue

        region_mask = (cc_labels == i)

        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(region_mask.astype(np.uint8), kernel, iterations=1)
        border_mask = (dilated.astype(bool)) & (~region_mask)

        neighbors = image_bgr[border_mask]
        if neighbors.size > 0:
            dominant_color = Counter(map(tuple, neighbors)).most_common(1)[0][0]
            changes.append((region_mask, dominant_color))

    return changes


def final_cleanup(image_bgr: np.ndarray, max_region_area: int = 50) -> np.ndarray:
    """
    Cleans up small isolated color regions using multithreading.

    Args:
        image_bgr (np.ndarray): Input BGR image.
        max_region_area (int): Max area of a region to be considered small.

    Returns:
        np.ndarray: Cleaned image.
    """
    h, w = image_bgr.shape[:2]
    flat = image_bgr.reshape(-1, 3)
    unique_colors, inverse = np.unique(flat, axis=0, return_inverse=True)
    label_map = inverse.reshape((h, w))
    total_colors = len(unique_colors)

    print(f"[CLEANUP] Starting threaded cleanup for {total_colors} color regions...")

    cleaned = image_bgr.copy()

    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(_process_color, idx, color, label_map, image_bgr, max_region_area)
            for idx, color in enumerate(unique_colors)
        ]

        for idx, future in enumerate(futures):
            changes = future.result()
            for region_mask, new_color in changes:
                cleaned[region_mask] = new_color

            if (idx + 1) % 10 == 0 or (idx + 1) == total_colors:
                report_status(guid, f"Regions {idx + 1}/{total_colors} processed")
                print(f"[CLEANUP] {idx + 1}/{total_colors} processed")

    print("[CLEANUP] Final cleanup complete.")
    return cleaned