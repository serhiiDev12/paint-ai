import numpy as np
import cv2
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import aiohttp

def _process_color_region(idx, color, label_map, image_bgr, min_area, h, w):
    """
    Processes a single color label and returns a cleaned region mask.
    Runs in a separate process.
    """
    color = tuple(map(int, color))
    mask = (label_map == idx).astype(np.uint8)
    pixel_count = np.count_nonzero(mask)

    output = np.zeros((h, w, 3), dtype=np.uint8)

    if pixel_count > (0.9 * h * w):
        output[mask.astype(bool)] = color
        return output

    try:
        num_labels, cc_labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    except Exception as e:
        print(f"[ERROR] Component labeling failed for color {color}: {e}")
        output[mask.astype(bool)] = color
        return output

    for i in range(1, num_labels):
        region_mask = (cc_labels == i)
        area = stats[i, cv2.CC_STAT_AREA]

        if area >= min_area:
            output[region_mask] = color
        else:
            kernel = np.ones((3, 3), np.uint8)
            dilated = cv2.dilate(region_mask.astype(np.uint8), kernel, iterations=1)
            border = (dilated.astype(bool)) & (~region_mask)
            neighbors = image_bgr[border]

            if len(neighbors) > 0:
                new_color = np.median(neighbors, axis=0).astype(np.uint8)
            else:
                new_color = np.array(color, dtype=np.uint8)

            output[region_mask] = new_color

    return output

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

def remove_small_regions(image_bgr, min_area=200, max_colors=100, guid=None):
    """
    Removes small regions in parallel using multiple processes for speed.
    """
    h, w = image_bgr.shape[:2]
    flat = image_bgr.reshape(-1, 3)
    unique_colors, inverse = np.unique(flat, axis=0, return_inverse=True)

    if len(unique_colors) > max_colors:
        report_status(guid, f"Too many colors ({len(unique_colors)}), skipping cleanup.")
        return image_bgr

    label_map = inverse.reshape((h, w))
    cleaned = np.zeros_like(image_bgr)

    report_status(guid, f"Cleaning {len(unique_colors)} color regions using multiprocessing...")

    # Serialize required arrays
    label_map_serial = label_map.copy()
    image_bgr_serial = image_bgr.copy()

    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = [
            executor.submit(
                _process_color_region,
                idx, color, label_map_serial, image_bgr_serial, min_area, h, w
            )
            for idx, color in enumerate(unique_colors)
        ]

        for i, future in enumerate(as_completed(futures)):
            cleaned += future.result()
            report_status(guid, f"Completed region {i + 1}/{len(unique_colors)}")

    # FINAL GRAIN REMOVAL: remove 1-pixel blobs using morphological open
    report_status(guid, "Final pass: removing 1px grains.")
    cleaned_post = np.zeros_like(cleaned)
    for i in range(3):
        channel = cleaned[:, :, i]
        opened = cv2.morphologyEx(channel, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        cleaned_post[:, :, i] = opened

    return cleaned_post