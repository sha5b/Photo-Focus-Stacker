#!/usr/bin/env python3

# Context: Focus measure utilities for Photo Focus Stacker
# Purpose: Compute per-pixel focus/sharpness maps used for selecting or blending source images.
# Notes: Called by `src.core.focus_stacker.FocusStacker`.

import cv2
import numpy as np


def _as_float_gray(img_gray):
    """Convert to float32 on a 0..255 scale without quantizing high-bit-depth input."""
    if np.issubdtype(img_gray.dtype, np.integer):
        maximum = float(np.iinfo(img_gray.dtype).max)
        return img_gray.astype(np.float32) * (255.0 / maximum)
    result = np.nan_to_num(img_gray.astype(np.float32), nan=0.0, posinf=1.0, neginf=0.0)
    if result.size and float(np.max(result)) <= 1.5:
        result *= 255.0
    return result

# Only focus measure method: Laplacian Variance Map
def measure_laplacian_variance_map(img_gray, window_size=7, normalize=True): # Defaulting window size to 7
    """
    Calculates a focus measure map using the variance of the Laplacian within a local window.

    @param img_gray: Grayscale input image (float32 [0, 1] or uint8 [0, 255]).
    @param window_size: Size of the square window for variance calculation (must be odd). Default: 7.
    @param normalize: Whether to normalize the focus map to [0, 1] per image. Default: True.
    @return: Focus map (float32 NumPy array [0, 1]), same HxW as input.
    """
    print(f"Calculating Laplacian variance map (window={window_size})...")
    img_gray_uint8 = _as_float_gray(img_gray)

    # Ensure window size is odd
    window_size = window_size if window_size % 2 != 0 else window_size + 1

    # Calculate Laplacian (float32 is sufficient and faster than float64 for this use case)
    laplacian = cv2.Laplacian(img_gray_uint8, cv2.CV_32F, ksize=3) # ksize=3 is common for focus maps

    # Calculate local mean of Laplacian using a box filter (efficient)
    mean = cv2.boxFilter(laplacian, -1, (window_size, window_size), normalize=True, borderType=cv2.BORDER_REFLECT)

    # Calculate local mean of squared Laplacian
    laplacian_sq = cv2.multiply(laplacian, laplacian)
    mean_sq = cv2.boxFilter(laplacian_sq, -1, (window_size, window_size), normalize=True, borderType=cv2.BORDER_REFLECT)

    # Calculate local variance: variance = E[X^2] - (E[X])^2
    variance_map = mean_sq - mean**2

    variance_map = np.maximum(variance_map, 0)

    if not normalize:
        print("Laplacian variance map calculation complete.")
        return variance_map.astype(np.float32)

    # Normalize the variance map to [0, 1]
    min_val, max_val, _, _ = cv2.minMaxLoc(variance_map)
    if max_val > min_val:
        focus_map = (variance_map - min_val) / (max_val - min_val)
    else:
        focus_map = np.zeros_like(variance_map) # Avoid division by zero

    print("Laplacian variance map calculation complete.")
    return focus_map.astype(np.float32)


def measure_tenengrad_map(img_gray, window_size=7, normalize=True):
    print(f"Calculating Tenengrad focus map (window={window_size})...")
    img_gray_uint8 = _as_float_gray(img_gray)

    window_size = window_size if window_size % 2 != 0 else window_size + 1

    gx = cv2.Sobel(img_gray_uint8, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img_gray_uint8, cv2.CV_32F, 0, 1, ksize=3)
    grad_energy = cv2.multiply(gx, gx) + cv2.multiply(gy, gy)

    focus_map = cv2.boxFilter(grad_energy, -1, (window_size, window_size), normalize=True, borderType=cv2.BORDER_REFLECT)
    focus_map = np.maximum(focus_map, 0.0)

    if not normalize:
        print("Tenengrad focus map calculation complete.")
        return focus_map.astype(np.float32)

    min_val, max_val, _, _ = cv2.minMaxLoc(focus_map)
    if max_val > min_val:
        focus_map = (focus_map - min_val) / (max_val - min_val)
    else:
        focus_map = np.zeros_like(focus_map)

    print("Tenengrad focus map calculation complete.")
    return focus_map.astype(np.float32)


def measure_sml_map(img_gray, window_size=7, normalize=True):
    print(f"Calculating SML focus map (window={window_size})...")
    img_gray_uint8 = _as_float_gray(img_gray)

    window_size = window_size if window_size % 2 != 0 else window_size + 1

    kernel = np.array([[0, 0, 0], [1, -2, 1], [0, 0, 0]], dtype=np.float32)
    lap_x = cv2.filter2D(img_gray_uint8.astype(np.float32), ddepth=cv2.CV_32F, kernel=kernel, borderType=cv2.BORDER_REFLECT)
    lap_y = cv2.filter2D(img_gray_uint8.astype(np.float32), ddepth=cv2.CV_32F, kernel=kernel.T, borderType=cv2.BORDER_REFLECT)
    sml = np.abs(lap_x) + np.abs(lap_y)

    focus_map = cv2.boxFilter(sml, -1, (window_size, window_size), normalize=True, borderType=cv2.BORDER_REFLECT)
    focus_map = np.maximum(focus_map, 0.0)

    if not normalize:
        print("SML focus map calculation complete.")
        return focus_map.astype(np.float32)

    min_val, max_val, _, _ = cv2.minMaxLoc(focus_map)
    if max_val > min_val:
        focus_map = (focus_map - min_val) / (max_val - min_val)
    else:
        focus_map = np.zeros_like(focus_map)

    print("SML focus map calculation complete.")
    return focus_map.astype(np.float32)


def measure_multiscale_map(img_gray, window_size=7, normalize=True):
    """Noise-tolerant multi-scale focus evidence for fine macro structures."""
    if img_gray.dtype == np.uint8:
        gray = img_gray.astype(np.float32) / 255.0
    else:
        gray = np.clip(img_gray.astype(np.float32), 0.0, 1.0)
    window_size = max(3, int(window_size) | 1)
    combined = np.zeros_like(gray, dtype=np.float32)
    total_weight = 0.0

    for sigma, scale_weight in ((0.0, 0.55), (1.0, 0.30), (2.0, 0.15)):
        source = gray if sigma == 0.0 else cv2.GaussianBlur(gray, (0, 0), sigma, borderType=cv2.BORDER_REFLECT)
        gx = cv2.Sobel(source, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(source, cv2.CV_32F, 0, 1, ksize=3)
        lap = cv2.Laplacian(source, cv2.CV_32F, ksize=3)
        evidence = gx * gx + gy * gy + 0.35 * lap * lap
        local_window = max(3, int(round(window_size * (1.0 + sigma * 0.5))) | 1)
        evidence = cv2.boxFilter(evidence, -1, (local_window, local_window),
                                 normalize=True, borderType=cv2.BORDER_REFLECT)
        if normalize:
            scale = float(np.percentile(evidence, 99.5))
            if scale > 1e-12:
                evidence = np.clip(evidence / scale, 0.0, 1.0)
        combined += evidence.astype(np.float32) * scale_weight
        total_weight += scale_weight

    combined /= max(total_weight, 1e-10)
    # Remove isolated sensor-noise responses without widening real edges much.
    combined = cv2.medianBlur(combined.astype(np.float32), 3)
    if normalize:
        maximum = float(np.max(combined))
        if maximum > 0:
            combined /= maximum
    return np.maximum(combined, 0.0).astype(np.float32)
