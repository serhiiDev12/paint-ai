import cv2
import numpy as np
from PIL import Image

def final_color_quantization(image_bgr, n_colors=30):
    """
    Reduces image to a fixed palette of n_colors using median cut quantization (via PIL).
    """
    # Convert BGR to RGB for Pillow
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)

    # Convert to palette image using median cut, no dithering to avoid semi-pixels
    quantized = pil_image.convert("P", palette=Image.ADAPTIVE, colors=n_colors, dither=Image.NONE)
    quant_rgb = quantized.convert("RGB")

    # Convert back to BGR
    return cv2.cvtColor(np.array(quant_rgb), cv2.COLOR_RGB2BGR)

def quantize_image(image_bgr, use_smoothing=False, spatial_radius=6, color_radius=20, n_colors=20):
    """
    Optional mean shift smoothing, then quantization to a clean, reduced color palette.

    Args:
        image_bgr: Input image (BGR)
        use_smoothing: If True, apply mean shift first
        spatial_radius, color_radius: Parameters for mean shift
        n_colors: Number of output colors

    Returns:
        Quantized BGR image with clean flat regions
    """
    if use_smoothing:
        image_bgr = cv2.pyrMeanShiftFiltering(image_bgr, sp=spatial_radius, sr=color_radius)
    return final_color_quantization(image_bgr, n_colors)