#!/usr/bin/env python3

# Context: Blending routines for Photo Focus Stacker
# Purpose: Combine aligned source images into a single output using focus maps.
# Notes: Called by `src.core.focus_stacker.FocusStacker`.

import cv2
import numpy as np


def _build_gaussian_pyramid(img: np.ndarray, levels: int) -> list:
    pyramid = [img]
    current = img
    keep_singleton_channel = img.ndim == 3 and img.shape[2] == 1
    for _ in range(max(int(levels), 1) - 1):
        next_level = cv2.pyrDown(current)
        if keep_singleton_channel and next_level is not None and next_level.ndim == 2:
            next_level = next_level[..., np.newaxis]
        if next_level is None or next_level.shape[0] < 2 or next_level.shape[1] < 2:
            break
        pyramid.append(next_level)
        current = next_level
    return pyramid


def _build_laplacian_pyramid(img: np.ndarray, levels: int) -> list:
    gaussian = _build_gaussian_pyramid(img, levels)
    laplacian: list = []
    for i in range(len(gaussian) - 1):
        h, w = gaussian[i].shape[:2]
        expanded = cv2.pyrUp(gaussian[i + 1], dstsize=(w, h))
        if gaussian[i].ndim == 3 and gaussian[i].shape[2] == 1 and expanded is not None and expanded.ndim == 2:
            expanded = expanded[..., np.newaxis]
        laplacian.append(gaussian[i] - expanded)
    laplacian.append(gaussian[-1])
    return laplacian


def _collapse_laplacian_pyramid(laplacian: list) -> np.ndarray:
    current = laplacian[-1]
    for level in range(len(laplacian) - 2, -1, -1):
        h, w = laplacian[level].shape[:2]
        current = cv2.pyrUp(current, dstsize=(w, h))
        if laplacian[level].ndim == 3 and laplacian[level].shape[2] == 1 and current is not None and current.ndim == 2:
            current = current[..., np.newaxis]
        current = current + laplacian[level]
    return current


def _compute_weight_maps(aligned_images, focus_maps):
    h, w = aligned_images[0].shape[:2]
    weight_maps = []
    valid_images = []

    epsilon = 1e-12
    valid_pairs = []
    for img, fm in zip(aligned_images, focus_maps):
        if img is None or fm is None:
            continue
        if img.shape[:2] != (h, w) or fm.shape[:2] != (h, w):
            continue

        fm_2d = fm[..., 0] if len(fm.shape) > 2 else fm
        fm_2d = np.nan_to_num(fm_2d.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        valid_pairs.append((img, fm_2d))

    if not valid_pairs:
        return weight_maps, valid_images

    stack_scales = [float(np.percentile(fm_2d, 99.5)) for _, fm_2d in valid_pairs]
    stack_scales = [s for s in stack_scales if np.isfinite(s) and s > 0.0]
    stack_scale = float(np.median(stack_scales)) if stack_scales else 1.0

    for img, fm_2d in valid_pairs:
        fm_2d = np.clip(fm_2d / (stack_scale + epsilon), 0.0, 1.0)

        smoothed_weights = cv2.GaussianBlur(
            fm_2d,
            (0, 0),
            sigmaX=1.0,
            sigmaY=1.0,
            borderType=cv2.BORDER_REFLECT,
        )
        smoothed_weights = np.nan_to_num(smoothed_weights, nan=0.0, posinf=0.0, neginf=0.0)
        weight_map = np.maximum(smoothed_weights, 0.0).astype(np.float32)

        weight_maps.append(weight_map.reshape(h, w, 1))
        valid_images.append(img)

    return weight_maps, valid_images


def _edge_aware_smooth_weight_map(guide_gray: np.ndarray, weight_map_2d: np.ndarray, radius: int, eps: float) -> np.ndarray:
    guide = np.nan_to_num(guide_gray.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    src = np.nan_to_num(weight_map_2d.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

    if hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "guidedFilter"):
        try:
            return cv2.ximgproc.guidedFilter(guide=guide, src=src, radius=int(radius), eps=float(eps))
        except Exception:
            pass

    d = int(max(5, min(21, radius * 2 + 1)))
    return cv2.bilateralFilter(src, d=d, sigmaColor=0.15, sigmaSpace=float(max(radius, 1)))


def blend_guided_weighted(aligned_images, focus_maps):
    print("\nBlending images using guided edge-aware weighted method...")
    if aligned_images is None or focus_maps is None:
        raise ValueError("Invalid input: aligned_images and focus_maps must be provided.")
    if len(aligned_images) == 0 or len(focus_maps) == 0 or len(aligned_images) != len(focus_maps):
        raise ValueError("Invalid input: aligned_images and focus_maps must be non-empty and have the same length.")

    weight_maps, valid_images = _compute_weight_maps(aligned_images, focus_maps)
    if not weight_maps:
        raise ValueError("No valid weight maps were produced for blending.")

    h, w = valid_images[0].shape[:2]
    epsilon = 1e-10

    ref_gray = cv2.cvtColor((valid_images[0] * 255.0 + 0.5).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    radius = int(max(4, min(32, round(min(h, w) / 250.0))))
    eps = 1e-3

    weights_stack = np.stack(weight_maps, axis=0).astype(np.float32)[..., 0]
    weights_stack = np.maximum(weights_stack, 0.0) + epsilon
    gamma = 3.0
    weights_stack = np.power(weights_stack, gamma)
    weights_sum = np.sum(weights_stack, axis=0)
    weights_norm = weights_stack / (weights_sum + epsilon)

    guided_list = []
    for k in range(weights_norm.shape[0]):
        guided = _edge_aware_smooth_weight_map(ref_gray, weights_norm[k], radius=radius, eps=eps)
        guided_list.append(np.maximum(guided, 0.0))

    guided_stack = np.stack(guided_list, axis=0).astype(np.float32)
    guided_sum = np.sum(guided_stack, axis=0)
    guided_norm = guided_stack / (guided_sum + epsilon)

    image_stack = np.stack(valid_images, axis=0).astype(np.float32)
    result = np.sum(image_stack * guided_norm[..., np.newaxis], axis=0)
    result = np.clip(result.astype(np.float32), 0.0, 1.0)
    print("Guided weighted blending complete.")
    return result


def blend_laplacian_pyramid(aligned_images, focus_maps, num_levels: int = 3):
    print("\nBlending images using Laplacian pyramid fusion...")
    if aligned_images is None or focus_maps is None:
        raise ValueError("Invalid input: aligned_images and focus_maps must be provided.")
    if len(aligned_images) == 0 or len(focus_maps) == 0 or len(aligned_images) != len(focus_maps):
        raise ValueError("Invalid input: aligned_images and focus_maps must be non-empty and have the same length.")

    weight_maps, valid_images = _compute_weight_maps(aligned_images, focus_maps)
    if not weight_maps:
        raise ValueError("No valid weight maps were produced for blending.")

    desired_levels = max(int(num_levels), 3)
    epsilon = 1e-10
    gamma = 3.0

    image_laplacians = [_build_laplacian_pyramid(img.astype(np.float32), desired_levels) for img in valid_images]
    weight_gaussians = [_build_gaussian_pyramid(wm.astype(np.float32), desired_levels) for wm in weight_maps]

    actual_levels = min(min(len(p) for p in image_laplacians), min(len(p) for p in weight_gaussians))
    if actual_levels < 1:
        raise ValueError("Failed to construct pyramids for blending.")

    fused_pyramid = []
    for level in range(actual_levels):
        weights_level = [wg[level][..., np.newaxis] if wg[level].ndim == 2 else wg[level] for wg in weight_gaussians]
        weights_stack = np.stack(weights_level, axis=0)
        weights_stack = np.maximum(weights_stack, 0.0) + epsilon
        weights_stack = np.power(weights_stack, gamma)
        weights_sum = np.sum(weights_stack, axis=0)
        weights_norm = weights_stack / (weights_sum + epsilon)

        fused_level = np.zeros_like(image_laplacians[0][level], dtype=np.float32)
        for img_idx in range(len(image_laplacians)):
            w = weights_norm[img_idx]
            if w.ndim == 2:
                w = w[..., np.newaxis]
            fused_level += image_laplacians[img_idx][level] * w
        fused_pyramid.append(fused_level)

    result = _collapse_laplacian_pyramid(fused_pyramid)
    result = np.clip(result.astype(np.float32), 0.0, 1.0)
    print("Laplacian pyramid fusion complete.")
    return result


def _refine_indices_majority(indices: np.ndarray, num_labels: int, window_size: int = 5, iterations: int = 2) -> np.ndarray:
    current = indices.astype(np.int32, copy=True)
    ksize = (int(window_size), int(window_size))

    for _ in range(max(int(iterations), 1)):
        counts = []
        for label in range(int(num_labels)):
            mask = (current == label).astype(np.float32)
            count = cv2.boxFilter(mask, ddepth=-1, ksize=ksize, normalize=False, borderType=cv2.BORDER_REFLECT)
            counts.append(count)
        current = np.argmax(np.stack(counts, axis=0), axis=0).astype(np.int32)
    return current


def _remove_small_label_regions(indices: np.ndarray, num_labels: int, min_area: int = 64) -> np.ndarray:
    refined = indices.astype(np.int32, copy=True)

    for label in range(int(num_labels)):
        mask = (refined == label).astype(np.uint8)
        num_components, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        for component_id in range(1, num_components):
            area = int(stats[component_id, cv2.CC_STAT_AREA])
            if area < int(min_area):
                refined[labels == component_id] = -1

    if np.any(refined < 0):
        unknown = refined < 0
        filled = _refine_indices_majority(np.where(unknown, 0, refined).astype(np.int32), num_labels=num_labels, window_size=7, iterations=2)
        refined[unknown] = filled[unknown]

    return refined

def blend_weighted(aligned_images, focus_maps):
    """
    Blend aligned images using their focus maps with a custom weighted approach.
    Refined weights based on multi-scale analysis and depth gradients.

    @param aligned_images: List of aligned input images (float32 [0, 1]).
    @param focus_maps: List of corresponding focus maps (float32 [0, 1]).
    @return: Blended image (float32 [0, 1]), before post-processing.
    """
    print("\nBlending images using custom weighted method...")
    if aligned_images is None or focus_maps is None:
        raise ValueError("Invalid input: aligned_images and focus_maps must be provided.")
    if len(aligned_images) == 0 or len(focus_maps) == 0 or len(aligned_images) != len(focus_maps):
        raise ValueError("Invalid input: aligned_images and focus_maps must be non-empty and have the same length.")

    h, w = aligned_images[0].shape[:2]
    result = np.zeros((h, w, 3), dtype=np.float32)
    weights_sum = np.zeros((h, w, 1), dtype=np.float32)
    epsilon = 1e-10 # For numerical stability

    weight_maps, valid_images = _compute_weight_maps(aligned_images, focus_maps)

    if not weight_maps:
        raise ValueError("No valid weight maps were produced for blending.")

    weights_stack = np.stack(weight_maps, axis=0).astype(np.float32)
    weights_stack = np.maximum(weights_stack, 0.0) + epsilon

    gamma = 3.0
    weights_stack = np.power(weights_stack, gamma)

    weights_sum = np.sum(weights_stack, axis=0)
    normalized_weights = weights_stack / (weights_sum + epsilon)

    image_stack = np.stack(valid_images, axis=0).astype(np.float32)
    result = np.sum(image_stack * normalized_weights, axis=0)
    result = np.clip(result, 0.0, 1.0) # Clip final result to [0, 1]

    print("Weighted blending complete.")
    return result


def blend_direct_map(aligned_images, sharpest_indices, focus_maps=None):
    """
    Blend aligned images by directly selecting pixels based on the sharpest index map.

    @param aligned_images: List of aligned input images (float32 [0, 1]).
    @param sharpest_indices: 2D NumPy array (uint16) indicating the index of the
                             sharpest image for each pixel.
    @return: Blended image (float32 [0, 1]).
    """
    print("\nBlending images using direct map selection...")
    if aligned_images is None or sharpest_indices is None:
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

    if focus_maps is not None and len(focus_maps) == num_images:
        best_val = np.full((h, w), -np.inf, dtype=np.float32)
        best_idx = np.zeros((h, w), dtype=np.int32)
        second_val = np.full((h, w), -np.inf, dtype=np.float32)
        second_idx = np.zeros((h, w), dtype=np.int32)

        for i, fm in enumerate(focus_maps):
            fm_2d = fm[..., 0] if getattr(fm, "ndim", 0) > 2 else fm
            fm_2d = np.nan_to_num(fm_2d.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

            is_best = fm_2d > best_val
            second_val = np.where(is_best, best_val, second_val)
            second_idx = np.where(is_best, best_idx, second_idx)
            best_val = np.where(is_best, fm_2d, best_val)
            best_idx = np.where(is_best, i, best_idx)

            is_second = (~is_best) & (fm_2d > second_val)
            second_val = np.where(is_second, fm_2d, second_val)
            second_idx = np.where(is_second, i, second_idx)

        epsilon = 1e-10
        best_val = np.maximum(best_val, 0.0)
        second_val = np.maximum(second_val, 0.0)

        gamma = 4.0
        best_pow = np.power(best_val + epsilon, gamma)
        second_pow = np.power(second_val + epsilon, gamma)
        w = best_pow / (best_pow + second_pow + epsilon)

        confidence = (best_val - second_val) / (best_val + second_val + epsilon)
        ambiguous = confidence < 0.25
        if np.any(ambiguous):
            w_smooth = cv2.GaussianBlur(w.astype(np.float32), (0, 0), sigmaX=1.0, sigmaY=1.0, borderType=cv2.BORDER_REFLECT)
            w = np.where(ambiguous, w_smooth, w)

        image_stack = np.stack(aligned_images, axis=0).astype(np.float32)
        row_idx = np.arange(h)[:, np.newaxis]
        col_idx = np.arange(w)[np.newaxis, :]
        best_img = image_stack[best_idx, row_idx, col_idx]
        second_img = image_stack[second_idx, row_idx, col_idx]

        w3 = w.reshape(h, w, 1).astype(np.float32)
        result = best_img * w3 + second_img * (1.0 - w3)
        result = np.clip(result.astype(np.float32), 0.0, 1.0)
    else:
        refined_indices = sharpest_indices.astype(np.int32)
        if num_images > 1:
            refined_indices = _refine_indices_majority(refined_indices, num_labels=num_images, window_size=3, iterations=2)
            refined_indices = _refine_indices_majority(refined_indices, num_labels=num_images, window_size=3, iterations=1)
        refined_indices = np.clip(refined_indices, 0, num_images - 1).astype(np.uint16)

        image_stack = np.stack(aligned_images, axis=0).astype(np.float32)
        row_idx = np.arange(h)[:, np.newaxis]
        col_idx = np.arange(w)[np.newaxis, :]
        result = image_stack[refined_indices, row_idx, col_idx]
        result = np.clip(result.astype(np.float32), 0.0, 1.0)

    print("Direct map blending complete.")
    return result

