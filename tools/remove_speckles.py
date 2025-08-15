import numpy as np
import cv2
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed

def _process_speckle_region(args):
    idx, color, label_map, image_bgr, max_area, h, w = args
    output = np.zeros((h, w, 3), dtype=np.uint8)
    color = tuple(map(int, color))
    mask = (label_map == idx).astype(np.uint8)

    try:
        num_labels, cc_labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    except Exception as e:
        print(f"[ERROR] Connected components failed for color {color}: {e}")
        return output

    for i in range(1, num_labels):  # skip background
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= max_area:
            continue

        region_mask = (cc_labels == i)
        if np.count_nonzero(region_mask) == 0:
            continue

        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(region_mask.astype(np.uint8), kernel, iterations=1)
        border_mask = (dilated.astype(bool)) & (~region_mask)
        neighbors = image_bgr[border_mask]

        if len(neighbors) > 0:
            new_color = Counter(map(tuple, neighbors)).most_common(1)[0][0]
        else:
            new_color = color

        output[region_mask] = new_color

    return output

def remove_speckles(image_bgr, max_area=50, min_global_pixel_count=100):
    h, w = image_bgr.shape[:2]
    flat = image_bgr.reshape(-1, 3)
    unique_colors, counts = np.unique(flat, axis=0, return_counts=True)
    inverse = np.zeros(flat.shape[0], dtype=np.int32)
    color_to_index = {tuple(c): i for i, c in enumerate(unique_colors)}
    for i, pix in enumerate(flat):
        inverse[i] = color_to_index[tuple(pix)]
    label_map = inverse.reshape((h, w))

    # Filter out colors with too few total pixels
    mask_colors = counts >= min_global_pixel_count
    filtered_colors = unique_colors[mask_colors]
    if len(filtered_colors) == 0:
        print("[INFO] No significant speckles found. Skipping cleanup.")
        return image_bgr.copy()

    print(f"[INFO] Speckle removal: processing {len(filtered_colors)} colors (min {min_global_pixel_count} px)...")

    cleaned = np.zeros_like(image_bgr)
    shared_args = [
        (idx, color, label_map, image_bgr, max_area, h, w)
        for idx, color in enumerate(unique_colors)
        if tuple(color) in map(tuple, filtered_colors)
    ]

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(_process_speckle_region, args) for args in shared_args]

        for i, future in enumerate(as_completed(futures), 1):
            cleaned += future.result()
            print(f"[PROGRESS] {i}/{len(shared_args)} colors processed ({(i/len(shared_args))*100:.1f}%)")

    print("[DONE] Speckle cleanup complete.")
    return cleaned