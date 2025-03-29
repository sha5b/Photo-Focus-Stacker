#!/usr/bin/env python3

import cv2
import numpy as np

def measure_laplacian_variance(img_gray, ksize=5):
    """
    Calculates focus measure using the variance of the Laplacian.
    A simple and common focus measure.

    @param img_gray: Grayscale input image (float32 [0, 1] or uint8 [0, 255]).
    @param ksize: Kernel size for the Laplacian operator (must be odd).
    @return: Single float value representing the focus measure (higher is sharper).
    """
    # Ensure ksize is odd
    ksize = ksize if ksize % 2 != 0 else ksize + 1
    # Ensure input is float32 [0, 1] for consistency if needed, though Laplacian works on uint8 too
    if img_gray.dtype != np.uint8:
        img_gray_uint8 = (img_gray * 255).astype(np.uint8)
    else:
        img_gray_uint8 = img_gray

    # Calculate Laplacian
    laplacian = cv2.Laplacian(img_gray_uint8, cv2.CV_64F, ksize=ksize)
    # Calculate the variance
    variance = laplacian.var()
    return variance

def measure_custom(img, radius=8):
    """
    Calculate a focus measure map for a single image using a custom, multi-faceted approach.
    Combines gradient magnitude, Laplacian, high-frequency components, and local contrast.
    Operates on a color image (float32 [0, 1]).

    @param img: Input color image (float32 NumPy array [0, 1]).
    @param radius: Parameter influencing window sizes (indirectly via internal logic).
                   Original class used this, kept for potential future use, but current
                   implementation uses fixed kernel sizes. Consider removing or integrating.
    @return: Focus map (float32 NumPy array [0, 1]), same HxW as input.
    """
    print(f"Calculating custom focus measure...") # Add print statement
    if len(img.shape) == 3:
        try:
            img_gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        except cv2.error as e:
            print(f"Error converting image to grayscale in focus measure: {e}")
            # Return a zero map or raise error? Returning zero map for now.
            return np.zeros(img.shape[:2], dtype=np.float32)
    elif img.dtype == np.uint8: # Handle uint8 grayscale input
         img_gray = img.astype(np.float32) / 255.0
    elif img.dtype == np.float32: # Handle float32 grayscale input
         img_gray = img
    else:
        raise TypeError(f"Unsupported image type for focus measure: {img.dtype}")

    h, w = img_gray.shape

    # Pre-calculate derivatives (using float32 input)
    dx = cv2.Sobel(img_gray, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(img_gray, cv2.CV_32F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(dx*dx + dy*dy)
    laplacian = cv2.Laplacian(img_gray, cv2.CV_32F, ksize=3) # Use ksize=3 for consistency

    # Multi-scale analysis
    scales = [1.0, 0.5, 0.25]
    weights = [0.6, 0.3, 0.1]
    focus_map = np.zeros_like(img_gray, dtype=np.float32)

    for scale, weight in zip(scales, weights):
        # Resize inputs for current scale
        if scale != 1.0:
            # Use INTER_AREA for shrinking, might be better than default INTER_LINEAR
            scaled_img = cv2.resize(img_gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            scaled_grad = cv2.resize(gradient_magnitude, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            scaled_lap = cv2.resize(laplacian, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        else:
            scaled_img = img_gray
            scaled_grad = gradient_magnitude
            scaled_lap = laplacian

        # Calculate components for this scale
        # Use appropriate kernel sizes for potentially smaller scaled images
        ksize_blur = 5 if scale > 0.3 else 3
        ksize_contrast = 7 if scale > 0.3 else 5
        sigma_contrast = 1.5 if scale > 0.3 else 1.0

        high_freq = np.abs(scaled_img - cv2.GaussianBlur(scaled_img, (ksize_blur, ksize_blur), 0))
        edge_strength = np.abs(scaled_lap)
        local_mean = cv2.GaussianBlur(scaled_img, (ksize_contrast, ksize_contrast), sigma_contrast)
        local_contrast = np.abs(scaled_img - local_mean)

        # Combine components - ensure non-negative before potential sqrt/log
        # Using multiplication can lead to very small numbers, maybe addition is more stable?
        # Let's stick to multiplication for now as per original logic.
        # Add small epsilon to avoid issues with zero values if taking log later.
        epsilon = 1e-6
        scale_measure = (high_freq + epsilon) * (edge_strength + epsilon) * (local_contrast + epsilon) * (scaled_grad + epsilon)

        # Resize result back to original size if needed
        if scale != 1.0:
            # Use INTER_LINEAR for upscaling
            scale_measure = cv2.resize(scale_measure, (w, h), interpolation=cv2.INTER_LINEAR)

        focus_map += weight * scale_measure

    # Normalize the combined map robustly
    min_val = np.min(focus_map)
    max_val = np.max(focus_map)
    if max_val > min_val:
        focus_map = (focus_map - min_val) / (max_val - min_val)
    else:
        focus_map = np.zeros_like(focus_map) # Avoid division by zero if map is flat

    # Optional edge-aware enhancement
    lap_abs = np.abs(laplacian)
    max_lap = np.max(lap_abs)
    if max_lap > 0:
        edge_mask = np.clip(lap_abs / max_lap, 0, 1)
        focus_map = focus_map * (1.0 + 0.2 * edge_mask) # Apply enhancement

    # Final normalization and clipping
    min_val = np.min(focus_map)
    max_val = np.max(focus_map)
    if max_val > min_val:
        focus_map = np.clip((focus_map - min_val) / (max_val - min_val), 0, 1)
    else:
        focus_map = np.zeros_like(focus_map) # Ensure output is [0, 1]

    print("Custom focus measure calculation complete.")
    return focus_map.astype(np.float32)

# --- Add other focus measure functions here ---
# e.g., measure_sobel_variance, measure_fft_based, etc.
