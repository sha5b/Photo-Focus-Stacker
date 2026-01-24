#!/usr/bin/env python3

# Context: Focus measure utilities for Photo Focus Stacker
# Purpose: Compute per-pixel focus/sharpness maps used for selecting or blending source images.
# Notes: Called by `src.core.focus_stacker.FocusStacker`.

import cv2
import numpy as np

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
    if img_gray.dtype != np.uint8:
        img_gray_uint8 = (img_gray * 255).astype(np.uint8)
    else:
        img_gray_uint8 = img_gray

    # Ensure window size is odd
    window_size = window_size if window_size % 2 != 0 else window_size + 1

    # Calculate Laplacian (use CV_64F for potentially negative values)
    laplacian = cv2.Laplacian(img_gray_uint8, cv2.CV_64F, ksize=3) # ksize=3 is common for focus maps

    # Calculate local mean of Laplacian using a box filter (efficient)
    mean = cv2.boxFilter(laplacian, -1, (window_size, window_size), normalize=True, borderType=cv2.BORDER_REFLECT)

    # Calculate local mean of squared Laplacian
    laplacian_sq = laplacian**2
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
