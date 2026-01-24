#!/usr/bin/env python3

# Context: Blending routines for Photo Focus Stacker
# Purpose: Combine aligned source images into a single output using focus maps.
# Notes: Called by `src.core.focus_stacker.FocusStacker`.

import cv2
import numpy as np

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
        depth_mask = cv2.bilateralFilter(depth_gradient, d=9, sigmaColor=75, sigmaSpace=75)
        min_dg, max_dg = np.min(depth_mask), np.max(depth_mask)
        if max_dg > min_dg:
            depth_mask = (depth_mask - min_dg) / (max_dg - min_dg)
        else:
            depth_mask = np.zeros_like(depth_mask) # Avoid division by zero

        # Multi-scale analysis to refine focus weights
        fm_refined = np.zeros_like(fm_2d)
        min_dim = int(min(h, w))

        scale_candidates = [
            min(150, max(5, int(min_dim * 0.15))),
            min(100, max(5, int(min_dim * 0.10))),
            min(50, max(5, int(min_dim * 0.05))),
            min(25, max(5, int(min_dim * 0.025))),
        ]
        scales = [s for s in scale_candidates if s > 0]
        weights = [0.4, 0.3, 0.2, 0.1] # Contribution weights

        for scale, weight_factor in zip(scales, weights):
            kernel_size = (scale*2+1, scale*2+1)
            sigma = max(scale / 3.0, 1.0) # Ensure sigma is at least 1

            fm_blur = cv2.GaussianBlur(fm_2d, kernel_size, sigma)

            edge_strength = np.abs(fm_2d - fm_blur)
            edge_strength_boosted = edge_strength * (1.0 + depth_mask)

            local_mean_edge = cv2.GaussianBlur(edge_strength_boosted, (25, 25), 5)
            local_sq_edge = cv2.GaussianBlur(edge_strength_boosted**2, (25, 25), 5)
            local_std_edge = np.sqrt(np.maximum(local_sq_edge - local_mean_edge**2, 0)) # Avoid negative variance

            threshold = local_mean_edge + local_std_edge * (1.5 + 0.5 * depth_mask)

            blend_weight = weight_factor * (1.0 + 0.5 * depth_mask)
            fm_refined += np.where(edge_strength_boosted > threshold, fm_blur * blend_weight, fm_2d * blend_weight)

        smoothed_weights = cv2.bilateralFilter(fm_refined, d=11, sigmaColor=100, sigmaSpace=100)
        smoothed_weights = cv2.bilateralFilter(smoothed_weights, d=7, sigmaColor=50, sigmaSpace=50)

        min_w, max_w = np.min(smoothed_weights), np.max(smoothed_weights)
        if max_w > min_w:
            weight_map = (smoothed_weights - min_w) / (max_w - min_w)
        else:
            weight_map = np.ones_like(smoothed_weights) # If flat, give full weight (shouldn't happen often)

        weight_map = weight_map.reshape(h, w, 1) # Ensure 3D for broadcasting

        result += img * weight_map
        weights_sum += weight_map

    result = result / (weights_sum + epsilon)
    result = np.clip(result, 0.0, 1.0) # Clip final result to [0, 1]

    print("Weighted blending complete.")
    return result


def blend_direct_map(aligned_images, sharpest_indices):
    """
    Blend aligned images by directly selecting pixels based on the sharpest index map.

    @param aligned_images: List of aligned input images (float32 [0, 1]).
    @param sharpest_indices: 2D NumPy array (uint16) indicating the index of the
                             sharpest image for each pixel.
    @return: Blended image (float32 [0, 1]).
    """
    print("\nBlending images using direct map selection...")
    if not aligned_images or sharpest_indices is None:
        raise ValueError("Invalid input: aligned_images and sharpest_indices must be provided.")
    if len(aligned_images) == 0:
         raise ValueError("aligned_images list cannot be empty.")

    h, w = aligned_images[0].shape[:2]
    num_images = len(aligned_images)

    if sharpest_indices.shape != (h, w):
        raise ValueError(f"Shape mismatch: sharpest_indices {sharpest_indices.shape} vs expected {(h, w)}")

    result = np.zeros((h, w, 3), dtype=np.float32)

    for i, img in enumerate(aligned_images):
        if img.shape[:2] != (h, w):
            raise ValueError(f"Image {i} has shape {img.shape[:2]}, expected {(h, w)}")
    image_stack = np.stack(aligned_images, axis=0)

    row_indices, col_indices = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    result = image_stack[sharpest_indices, row_indices, col_indices]
    result = np.clip(result.astype(np.float32), 0.0, 1.0)

    print("Direct map blending complete.")
    return result

