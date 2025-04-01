#!/usr/bin/env python3

import cv2
import numpy as np

# --- Weighted Blending (Original Method Refactored) ---

def blend_weighted(aligned_images, focus_maps):
    """
    Blend aligned images using their focus maps with a custom weighted approach.
    Refined weights based on multi-scale analysis and depth gradients.

    @param aligned_images: List of aligned input images (float32 [0, 1]).
    @param focus_maps: List of corresponding focus maps (float32 [0, 1]).
    @return: Blended image (float32 [0, 1]), before post-processing.
    """
    print("\nBlending images using custom weighted method...")
    if not aligned_images or not focus_maps or len(aligned_images) != len(focus_maps):
        raise ValueError("Invalid input: aligned_images and focus_maps must be non-empty and have the same length.")

    h, w = aligned_images[0].shape[:2]
    result = np.zeros((h, w, 3), dtype=np.float32)
    weights_sum = np.zeros((h, w, 1), dtype=np.float32)
    epsilon = 1e-10 # For numerical stability

    # Process each image
    for i, (img, fm) in enumerate(zip(aligned_images, focus_maps)):
        print(f"  Processing weights for image {i+1}/{len(aligned_images)}...")
        if img is None or fm is None:
            print(f"  Warning: Skipping None image or focus map at index {i}.")
            continue
        if img.shape[:2] != (h, w) or fm.shape[:2] != (h, w):
             print(f"  Warning: Skipping image/map with mismatched dimensions at index {i}.")
             continue

        # Ensure focus map is 2D
        fm_2d = fm[..., 0] if len(fm.shape) > 2 else fm
        fm_2d = fm_2d.astype(np.float32) # Ensure float32

        # Calculate depth gradients (edges in the focus map)
        dx = cv2.Sobel(fm_2d, cv2.CV_32F, 1, 0, ksize=3)
        dy = cv2.Sobel(fm_2d, cv2.CV_32F, 0, 1, ksize=3)
        depth_gradient = np.sqrt(dx*dx + dy*dy)

        # Create depth-aware mask (smoother transitions near focus edges)
        # Using bilateral filter helps preserve edges while smoothing
        depth_mask = cv2.bilateralFilter(depth_gradient, d=9, sigmaColor=75, sigmaSpace=75)
        min_dg, max_dg = np.min(depth_mask), np.max(depth_mask)
        if max_dg > min_dg:
            depth_mask = (depth_mask - min_dg) / (max_dg - min_dg)
        else:
            depth_mask = np.zeros_like(depth_mask) # Avoid division by zero

        # Multi-scale analysis to refine focus weights
        fm_refined = np.zeros_like(fm_2d)
        # Scales adjusted slightly based on typical image sizes and performance
        scales = [150, 100, 50, 25] # Kernel sizes (radius)
        weights = [0.4, 0.3, 0.2, 0.1] # Contribution weights

        for scale, weight_factor in zip(scales, weights):
            kernel_size = (scale*2+1, scale*2+1)
            sigma = max(scale / 3.0, 1.0) # Ensure sigma is at least 1

            # Gaussian blur of the focus map
            fm_blur = cv2.GaussianBlur(fm_2d, kernel_size, sigma)

            # Edge strength (difference between original and blurred)
            edge_strength = np.abs(fm_2d - fm_blur)
            # Boost edges based on depth gradient mask
            edge_strength_boosted = edge_strength * (1.0 + depth_mask)

            # Thresholding based on local statistics (more robust threshold)
            # Use a smaller window for local stats to be more adaptive
            local_mean_edge = cv2.GaussianBlur(edge_strength_boosted, (25, 25), 5)
            local_sq_edge = cv2.GaussianBlur(edge_strength_boosted**2, (25, 25), 5)
            local_std_edge = np.sqrt(np.maximum(local_sq_edge - local_mean_edge**2, 0)) # Avoid negative variance

            # Threshold: mean + k * std_dev, modulated by depth mask
            threshold = local_mean_edge + local_std_edge * (1.5 + 0.5 * depth_mask)

            # Combine weights: Use blurred map where edge strength is high, original otherwise
            # Weight factor modulated by depth mask
            blend_weight = weight_factor * (1.0 + 0.5 * depth_mask)
            fm_refined += np.where(edge_strength_boosted > threshold, fm_blur * blend_weight, fm_2d * blend_weight)

        # Smooth the refined weights using bilateral filter for edge preservation
        smoothed_weights = cv2.bilateralFilter(fm_refined, d=11, sigmaColor=100, sigmaSpace=100)
        smoothed_weights = cv2.bilateralFilter(smoothed_weights, d=7, sigmaColor=50, sigmaSpace=50)

        # Normalize weights for this image to [0, 1]
        min_w, max_w = np.min(smoothed_weights), np.max(smoothed_weights)
        if max_w > min_w:
            weight_map = (smoothed_weights - min_w) / (max_w - min_w)
        else:
            weight_map = np.ones_like(smoothed_weights) # If flat, give full weight (shouldn't happen often)

        weight_map = weight_map.reshape(h, w, 1) # Ensure 3D for broadcasting

        # Weighted blending
        result += img * weight_map
        weights_sum += weight_map

    # Normalize the final blended image by the sum of weights
    # Add epsilon to avoid division by zero where weights_sum is zero
    result = result / (weights_sum + epsilon)
    result = np.clip(result, 0.0, 1.0) # Clip final result to [0, 1]

    print("Weighted blending complete.")
    return result

# Removed Laplacian blending and consistency filter functions
# Removed blend_direct_select as weighted is now the only option
