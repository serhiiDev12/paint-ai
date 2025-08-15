import cv2
import numpy as np
import aiohttp
from tools.quantizer import quantize_image
from tools.remove_small_regions import remove_small_regions
from tools.color_shift import shift_yellow
from tools.final_cleanup import final_cleanup
from tools.post_speckle_cleanup import post_speckle_cleanup
from tools.compress_preserve import compress_preserve

# ===== CONFIGURATION =====
N_COLORS = 30
FEEDBACK_ENABLED = False
DEBUG = False

async def process_image(image: np.ndarray, guid=None):
    """
    Accepts an OpenCV image and sends the result to callback after processing.
    """
    if image is None or not isinstance(image, np.ndarray):
        print(f"[ERROR] Invalid image for GUID {guid}")
        return None, "Failed to load"

    try:
        if DEBUG:
            cv2.imwrite("debug_0_original.jpg", image)

        image_smooth = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)
        if DEBUG:
            cv2.imwrite("debug_1_smoothed.jpg", image_smooth)

        quantized = quantize_image(image_smooth, use_smoothing=False)
        if DEBUG:
            cv2.imwrite("debug_2_quantized.jpg", quantized)

        cleaned = remove_small_regions(quantized, min_area=300)
        if DEBUG:
            cv2.imwrite("debug_3_cleaned.jpg", cleaned)

        color_fixed = shift_yellow(cleaned)
        if DEBUG:
            cv2.imwrite("debug_4_colorfix.jpg", color_fixed)

        final = final_cleanup(color_fixed, max_region_area=50)
        cleaned = post_speckle_cleanup(final, max_area=18)
        cleaned = post_speckle_cleanup(cleaned, max_area=30)
        cleaned = post_speckle_cleanup(cleaned, max_area=5)
        cleaned = post_speckle_cleanup(cleaned, max_area=65)
        cleaned = post_speckle_cleanup(cleaned, max_area=2)
        cleaned = post_speckle_cleanup(cleaned, max_area=12)

        if DEBUG:
            cv2.imwrite("debug_6_final.jpg", cleaned)

        # Encode image to PNG
        _, encoded = cv2.imencode(".png", cleaned)
        image_bytes = encoded.tobytes()

        # === CALLBACK ===
        callback_url = "http://localhost:5161/api/callback"

        async with aiohttp.ClientSession() as session:
            form = aiohttp.FormData()
            form.add_field("file", image_bytes, filename=f"{guid}.png", content_type="image/png")
            form.add_field("id", guid or "no-guid")

            try:
                async with session.post(callback_url, data=form) as resp:
                    print(f"[CALLBACK] GUID {guid} returned status {resp.status}")
            except Exception as e:
                print(f"[CALLBACK ERROR] GUID {guid}: {e}")

        return None, "Success"

    except Exception as e:
        print(f"[PROCESS ERROR] GUID {guid}: {str(e)}")
        return None, f"Exception: {str(e)}"