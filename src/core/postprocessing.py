#!/usr/bin/env python3

import cv2
import numpy as np

def apply_unsharp_mask(image, strength=0.5, kernel_size=(5, 5), sigma=1.0):
    """
    Applies Unsharp Masking to enhance image sharpness.

    @param image: Input image (float32 NumPy array [0, 1]).
    @param strength: Strength of the sharpening effect (0.0 to disable, typical range 0.5-1.5).
    @param kernel_size: Gaussian kernel size for blurring (must be odd).
    @param sigma: Gaussian sigma for blurring.
    @return: Sharpened image (float32 NumPy array [0, 1]).
    """
    if strength <= 0:
        return image # No sharpening needed

    print(f"\nApplying Unsharp Mask (Strength: {strength:.2f}, Kernel: {kernel_size}, Sigma: {sigma:.2f})...")

    # Ensure kernel size is odd
    k_h = kernel_size[0] if kernel_size[0] % 2 != 0 else kernel_size[0] + 1
    k_w = kernel_size[1] if kernel_size[1] % 2 != 0 else kernel_size[1] + 1
    kernel_size = (k_h, k_w)

    # Create blurred version
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)

    # Calculate the sharpened image: Original + strength * (Original - Blurred)
    sharpened = cv2.addWeighted(image, 1.0 + strength, blurred, -strength, 0)

    # Clip result to valid range [0, 1]
    sharpened = np.clip(sharpened, 0.0, 1.0)

    print("Unsharp Mask applied.")
    return sharpened

# --- Add other post-processing functions below if needed ---
