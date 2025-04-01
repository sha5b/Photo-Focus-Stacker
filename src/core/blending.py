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


# --- Laplacian Pyramid Blending ---

def _build_gaussian_pyramid(img, levels):
    """Builds a Gaussian pyramid."""
    pyramid = [img]
    for i in range(levels - 1):
        img = cv2.pyrDown(img)
        pyramid.append(img)
    return pyramid

def _build_laplacian_pyramid(img, levels):
    """Builds a Laplacian pyramid."""
    gaussian_pyramid = _build_gaussian_pyramid(img, levels)
    laplacian_pyramid = []
    # Ensure levels doesn't exceed actual pyramid size possible
    actual_levels = min(levels, len(gaussian_pyramid))
    if actual_levels != levels:
        print(f"      Warning: Requested {levels} levels, but image size only allows {actual_levels}.")

    for i in range(actual_levels - 1): # Use actual_levels
        expanded = cv2.pyrUp(gaussian_pyramid[i+1], dstsize=(gaussian_pyramid[i].shape[1], gaussian_pyramid[i].shape[0]))
        laplacian = cv2.subtract(gaussian_pyramid[i], expanded)
        laplacian_pyramid.append(laplacian)
    laplacian_pyramid.append(gaussian_pyramid[actual_levels-1]) # Use actual_levels for top level
    return laplacian_pyramid

def _reconstruct_from_laplacian_pyramid(pyramid):
    """Reconstructs an image from its Laplacian pyramid."""
    if not pyramid:
        return None # Or raise error
    num_levels = len(pyramid)
    img = pyramid[-1] # Start with the top level
    for i in range(num_levels - 2, -1, -1): # Iterate downwards
        expanded = cv2.pyrUp(img, dstsize=(pyramid[i].shape[1], pyramid[i].shape[0]))
        img = cv2.add(expanded, pyramid[i])
    return img

def blend_laplacian_pyramid(aligned_images, sharpest_indices_map, levels=5):
    """
    Blend images using Laplacian pyramid fusion based on a pre-computed index map.

    @param aligned_images: List of aligned input images (float32 [0, 1]).
    @param sharpest_indices_map: 2D NumPy array indicating the index of the
                                 sharpest source image for each pixel. This map
                                 might have been filtered for consistency.
    @param levels: Number of pyramid levels.
    @return: Blended image (float32 [0, 1]).
    """
    print("\nBlending images using Laplacian Pyramid method...")
    if not aligned_images:
        raise ValueError("aligned_images cannot be empty for Laplacian blending.")
    if sharpest_indices_map is None:
         raise ValueError("sharpest_indices_map is required for Laplacian blending.")

    num_images = len(aligned_images)
    h, w = aligned_images[0].shape[:2]

    if sharpest_indices_map.shape != (h, w):
        raise ValueError("sharpest_indices_map dimensions must match aligned_images dimensions.")

    # Use the provided sharpest_indices_map directly
    sharpest_indices = sharpest_indices_map

    # 2. Build Laplacian pyramids for all source images
    laplacian_pyramids = [] # List of pyramids, one for each image
    for i, img in enumerate(aligned_images):
        print(f"  Building Laplacian pyramid for image {i+1}/{num_images}...")
        # Ensure image has 3 channels for pyramid functions if needed, or handle grayscale
        if len(img.shape) == 2: img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        laplacian_pyramids.append(_build_laplacian_pyramid(img, levels))

    # 3. Build Gaussian pyramid for the sharpest_indices map
    # This map will guide the selection at each level.
    # We need float32 for pyrDown/pyrUp operations on the index map.
    index_pyramid = _build_gaussian_pyramid(sharpest_indices.astype(np.float32), levels)

    # 4. Fuse Laplacian pyramids using the index pyramid
    print("  Fusing pyramids...")
    fused_laplacian_pyramid = []
    for level in range(levels):
        fused_level = np.zeros_like(laplacian_pyramids[0][level], dtype=np.float32)
        level_indices = np.round(index_pyramid[level]).astype(int) # Round indices at this level
        level_indices = np.clip(level_indices, 0, num_images - 1) # Ensure indices are valid

        # Select coefficients directly based on the index map at this level using masks.
        h_level, w_level = laplacian_pyramids[0][level].shape[:2]
        num_channels = laplacian_pyramids[0][level].shape[2] if len(laplacian_pyramids[0][level].shape) == 3 else 1

        for i in range(num_images):
            # Create a mask for regions where image 'i' is selected at this level
            mask = (level_indices == i).astype(np.float32)
            # Ensure mask is broadcastable (add channel dim if needed)
            if num_channels > 1 and len(mask.shape) == 2:
                mask = mask[..., np.newaxis]

            # Apply mask to the Laplacian coefficients of image 'i'
            fused_level += laplacian_pyramids[i][level] * mask

        fused_laplacian_pyramid.append(fused_level)


    # 5. Reconstruct the final image from the fused Laplacian pyramid
    print("  Reconstructing final image...")
    result = _reconstruct_from_laplacian_pyramid(fused_laplacian_pyramid)
    result = np.clip(result, 0.0, 1.0)

    print("Laplacian Pyramid blending complete.")
    return result.astype(np.float32)


# --- Consistency Filter ---

def apply_consistency_filter(index_map, kernel_size=5):
    """
    Applies a consistency filter (e.g., median filter) to the map
    of sharpest source indices to remove isolated pixels.

    @param index_map: 2D NumPy array where each value is the index of the sharpest source image.
    @param kernel_size: Size of the filter kernel (must be odd).
    @return: Filtered index map.
    """
    print(f"\nApplying consistency filter (Median Filter k={kernel_size})...")
    if kernel_size % 2 == 0:
        kernel_size += 1 # Ensure odd kernel size
        print(f"  Adjusted kernel size to {kernel_size} (must be odd).")

    # Median filter is good for removing salt-and-pepper noise from the index map.
    # Ensure input is suitable type for medianBlur.
    if np.max(index_map) < 256:
         filtered_map = cv2.medianBlur(index_map.astype(np.uint8), kernel_size)
    else:
         # Handle cases with more than 256 images (medianBlur might be slow on float32)
         print("  Warning: More than 255 images, median filter might be slow.")
         filtered_map = cv2.medianBlur(index_map.astype(np.float32), kernel_size)
         filtered_map = np.round(filtered_map).astype(index_map.dtype) # Convert back

    print("Consistency filter applied.")
    return filtered_map
