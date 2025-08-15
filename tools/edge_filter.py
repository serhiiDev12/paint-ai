def sharpen_edges(image_bgr):
    """
    Light edge enhancement to preserve smooth regions without haloing.

    Args:
        image_bgr (np.ndarray): Input image.

    Returns:
        np.ndarray: Slightly sharpened image.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_16S, ksize=3)
    laplacian = cv2.convertScaleAbs(laplacian)
    sharpened = cv2.addWeighted(image_bgr, 1.0, cv2.cvtColor(laplacian, cv2.COLOR_GRAY2BGR), 0.2, 0)
    return sharpened