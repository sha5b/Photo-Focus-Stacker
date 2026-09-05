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


def _compute_weight_maps(aligned_images, focus_maps, valid_masks=None):
    h, w = aligned_images[0].shape[:2]
    weight_maps = []
    valid_images = []

    epsilon = 1e-12
    valid_pairs = []
    if valid_masks is None:
        valid_masks = [None] * len(aligned_images)

    for img, fm, valid_mask in zip(aligned_images, focus_maps, valid_masks):
        if img is None or fm is None:
            continue
        if img.shape[:2] != (h, w) or fm.shape[:2] != (h, w):
            continue

        fm_2d = fm[..., 0] if len(fm.shape) > 2 else fm
        fm_2d = np.nan_to_num(fm_2d.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        if valid_mask is None:
            mask_2d = np.ones((h, w), dtype=np.float32)
        else:
            mask_2d = np.asarray(valid_mask, dtype=np.float32)
            if mask_2d.shape != (h, w):
                continue
            mask_2d = np.clip(mask_2d, 0.0, 1.0)
        valid_pairs.append((img, fm_2d, mask_2d))

    if not valid_pairs:
        return weight_maps, valid_images

    stack_scales = [float(np.percentile(fm_2d[mask_2d > 0.5], 99.5))
                    for _, fm_2d, mask_2d in valid_pairs if np.any(mask_2d > 0.5)]
    stack_scales = [s for s in stack_scales if np.isfinite(s) and s > 0.0]
    stack_scale = float(np.median(stack_scales)) if stack_scales else 1.0

    for img, fm_2d, mask_2d in valid_pairs:
        fm_2d = np.clip(fm_2d / (stack_scale + epsilon), 0.0, 1.0)

        smoothed_weights = cv2.GaussianBlur(
            fm_2d,
            (0, 0),
            sigmaX=1.0,
            sigmaY=1.0,
            borderType=cv2.BORDER_REFLECT,
        )
        smoothed_weights = np.nan_to_num(smoothed_weights, nan=0.0, posinf=0.0, neginf=0.0)
        weight_map = np.where(mask_2d > 0.5, np.maximum(smoothed_weights, 0.0), -1.0).astype(np.float32)

        weight_maps.append(weight_map.reshape(h, w, 1))
        valid_images.append(img)

    return weight_maps, valid_images


def _normalize_weights_softmax(weights_stack: np.ndarray, beta: float = 6.0,
                               epsilon: float = 1e-10, valid_stack=None) -> np.ndarray:
    w = np.nan_to_num(weights_stack.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    inferred_valid = w >= 0.0
    w = np.maximum(w, 0.0)
    x = w * float(beta)
    x = x - np.max(x, axis=0, keepdims=True)
    exp_x = np.exp(np.clip(x, -50.0, 50.0))
    if valid_stack is None:
        valid = inferred_valid.astype(np.float32)
        exp_x *= valid
    else:
        valid = np.asarray(valid_stack, dtype=np.float32)
        while valid.ndim < exp_x.ndim:
            valid = valid[..., np.newaxis]
        exp_x *= np.clip(valid, 0.0, 1.0)
    sum_exp = np.sum(exp_x, axis=0, keepdims=True)
    normalized = exp_x / (sum_exp + float(epsilon))
    # Pixels outside every warped source are filled from the reference frame.
    empty = sum_exp <= float(epsilon)
    if np.any(empty):
        normalized[0] = np.where(empty[0], 1.0, normalized[0])
    return normalized


def build_focus_depth_map(focus_maps, valid_masks=None, window_size: int = 5,
                          min_region_area: int = 64):
    """Build a spatially regularized source-index map and confidence map."""
    if not focus_maps:
        raise ValueError("focus_maps cannot be empty")
    h, w = focus_maps[0].shape[:2]
    if valid_masks is None:
        valid_masks = [np.ones((h, w), dtype=np.float32) for _ in focus_maps]

    normalized = []
    for fm, mask in zip(focus_maps, valid_masks):
        values = np.nan_to_num(np.asarray(fm, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        mask = np.asarray(mask, dtype=np.float32)
        valid_values = values[mask > 0.5]
        scale = float(np.percentile(valid_values, 99.5)) if valid_values.size else 1.0
        score = np.clip(values / max(scale, 1e-10), 0.0, 1.0)
        score = cv2.GaussianBlur(score, (0, 0), 1.0, borderType=cv2.BORDER_REFLECT)
        normalized.append(np.where(mask > 0.5, score, -1e6))

    scores = np.stack(normalized, axis=0)
    order = np.argsort(scores, axis=0)
    best = order[-1].astype(np.int32)
    best_score = np.take_along_axis(scores, order[-1:,...], axis=0)[0]
    second_score = np.take_along_axis(scores, order[-2:-1,...], axis=0)[0] if len(focus_maps) > 1 else np.zeros_like(best_score)
    ordered_window = max(3, min(5, int(window_size) | 1))
    best = cv2.medianBlur(best.astype(np.uint16), ordered_window).astype(np.int32)
    best = _refine_indices_majority(best, len(focus_maps), window_size=max(3, int(window_size) | 1), iterations=1)
    best = _remove_small_label_regions(best, len(focus_maps), min_area=max(1, int(min_region_area)))
    confidence = np.clip((best_score - second_score) / (np.abs(best_score) + np.abs(second_score) + 1e-10), 0.0, 1.0)
    return best.astype(np.uint16), confidence.astype(np.float32)


def _apply_depth_prior(weights: np.ndarray, depth_map, confidence_map) -> np.ndarray:
    if depth_map is None or confidence_map is None:
        return weights
    result = weights.astype(np.float32, copy=True)
    squeeze_channel = result.ndim == 4 and result.shape[-1] == 1
    working = result[..., 0] if squeeze_channel else result
    alpha = (np.clip(confidence_map, 0.0, 1.0) * 0.85).astype(np.float32)
    for index in range(working.shape[0]):
        hard = (np.asarray(depth_map) == index).astype(np.float32)
        working[index] = working[index] * (1.0 - alpha) + hard * alpha
    working /= np.sum(working, axis=0, keepdims=True) + 1e-10
    return working[..., np.newaxis] if squeeze_channel else working


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


def blend_guided_weighted(aligned_images, focus_maps, valid_masks=None,
                          depth_map=None, confidence_map=None):
    print("\nBlending images using guided edge-aware weighted method...")
    if aligned_images is None or focus_maps is None:
        raise ValueError("Invalid input: aligned_images and focus_maps must be provided.")
    if len(aligned_images) == 0 or len(focus_maps) == 0 or len(aligned_images) != len(focus_maps):
        raise ValueError("Invalid input: aligned_images and focus_maps must be non-empty and have the same length.")

    weight_maps, valid_images = _compute_weight_maps(aligned_images, focus_maps, valid_masks)
    if not weight_maps:
        raise ValueError("No valid weight maps were produced for blending.")

    h, w = valid_images[0].shape[:2]
    epsilon = 1e-10

    ref_gray = cv2.cvtColor(
        np.clip(valid_images[0], 0.0, 1.0).astype(np.float32), cv2.COLOR_RGB2GRAY
    )
    radius = int(max(4, min(32, round(min(h, w) / 250.0))))
    eps = 1e-3

    weights_stack = np.stack(weight_maps, axis=0).astype(np.float32)[..., 0]
    weights_norm = _normalize_weights_softmax(weights_stack, beta=6.0, epsilon=epsilon)

    refined = []
    for k in range(weights_norm.shape[0]):
        refined_k = _edge_aware_smooth_weight_map(ref_gray, weights_norm[k], radius=radius, eps=eps)
        if valid_masks is not None:
            refined_k = np.where(np.asarray(valid_masks[k]) > 0.5, refined_k, 0.0)
        refined.append(np.maximum(refined_k, 0.0))

    refined_stack = np.stack(refined, axis=0).astype(np.float32)
    refined_sum = np.sum(refined_stack, axis=0)
    refined_norm = refined_stack / (refined_sum + epsilon)
    refined_norm = _apply_depth_prior(refined_norm, depth_map, confidence_map)

    result = np.zeros_like(valid_images[0], dtype=np.float32)
    for index, image in enumerate(valid_images):
        result += image.astype(np.float32) * refined_norm[index, ..., np.newaxis]
    result = np.clip(result.astype(np.float32), 0.0, 1.0)
    print("Guided weighted blending complete.")
    return result


def blend_luma_weighted_chroma_pick(aligned_images, focus_maps, valid_masks=None,
                                    depth_map=None, confidence_map=None):
    print("\nBlending images using luma weighted + chroma pick (MFF)...")
    if aligned_images is None or focus_maps is None:
        raise ValueError("Invalid input: aligned_images and focus_maps must be provided.")
    if len(aligned_images) == 0 or len(focus_maps) == 0 or len(aligned_images) != len(focus_maps):
        raise ValueError("Invalid input: aligned_images and focus_maps must be non-empty and have the same length.")

    h, w = aligned_images[0].shape[:2]
    for i, img in enumerate(aligned_images):
        if img is None or img.shape[:2] != (h, w):
            raise ValueError(f"Image {i} has shape {None if img is None else img.shape[:2]}, expected {(h, w)}")

    weight_maps, valid_images = _compute_weight_maps(aligned_images, focus_maps, valid_masks)
    if not weight_maps:
        raise ValueError("No valid weight maps were produced for blending.")

    epsilon = 1e-10
    weights_stack = np.stack(weight_maps, axis=0).astype(np.float32)[..., 0]
    weights_norm = _normalize_weights_softmax(weights_stack, beta=6.0, epsilon=epsilon)[..., np.newaxis]
    weights_norm = _apply_depth_prior(weights_norm, depth_map, confidence_map)

    print("  Converting images to YCrCb...")
    y_list = []
    cr_list = []
    cb_list = []
    for img in valid_images:
        ycrcb = cv2.cvtColor(np.clip(img, 0.0, 1.0).astype(np.float32), cv2.COLOR_RGB2YCrCb)
        y_list.append(ycrcb[..., 0])
        cr_list.append(ycrcb[..., 1])
        cb_list.append(ycrcb[..., 2])

    print("  Fusing luminance...")
    y_fused = np.zeros((h, w), dtype=np.float32)
    for k in range(len(y_list)):
        y_fused += y_list[k].astype(np.float32) * weights_norm[k, ..., 0].astype(np.float32)
    y_fused = np.clip(y_fused, 0.0, 1.0)

    print("  Selecting chroma sources...")
    best_val = np.full((h, w), -np.inf, dtype=np.float32)
    best_idx = np.zeros((h, w), dtype=np.int32)
    second_val = np.full((h, w), -np.inf, dtype=np.float32)
    second_idx = np.zeros((h, w), dtype=np.int32)
    for i, fm in enumerate(focus_maps):
        fm_smooth = cv2.GaussianBlur(fm.astype(np.float32), (0, 0), sigmaX=1.0, sigmaY=1.0, borderType=cv2.BORDER_REFLECT)
        if valid_masks is not None:
            fm_smooth = np.where(np.asarray(valid_masks[i]) > 0.5, fm_smooth, -1e6)
        is_best = fm_smooth > best_val
        second_val = np.where(is_best, best_val, second_val)
        second_idx = np.where(is_best, best_idx, second_idx)
        best_val = np.where(is_best, fm_smooth, best_val)
        best_idx = np.where(is_best, int(i), best_idx)

        is_second = (~is_best) & (fm_smooth > second_val)
        second_val = np.where(is_second, fm_smooth, second_val)
        second_idx = np.where(is_second, int(i), second_idx)

    best_idx = best_idx.astype(np.int32)
    second_idx = second_idx.astype(np.int32)
    if len(valid_images) > 1:
        best_idx = _refine_indices_majority(best_idx, num_labels=len(valid_images), window_size=3, iterations=2)
        second_idx = _refine_indices_majority(second_idx, num_labels=len(valid_images), window_size=3, iterations=1)
    best_idx = np.clip(best_idx, 0, len(valid_images) - 1).astype(np.int32)
    second_idx = np.clip(second_idx, 0, len(valid_images) - 1).astype(np.int32)

    cr_stack = np.stack(cr_list, axis=0)
    cb_stack = np.stack(cb_list, axis=0)
    row_idx = np.arange(h)[:, np.newaxis]
    col_idx = np.arange(w)[np.newaxis, :]
    cr_best = cr_stack[best_idx, row_idx, col_idx].astype(np.float32)
    cb_best = cb_stack[best_idx, row_idx, col_idx].astype(np.float32)
    cr_second = cr_stack[second_idx, row_idx, col_idx].astype(np.float32)
    cb_second = cb_stack[second_idx, row_idx, col_idx].astype(np.float32)

    best_val = np.maximum(best_val, 0.0)
    second_val = np.maximum(second_val, 0.0)
    gamma_chroma = 4.0
    best_pow = np.power(best_val + epsilon, gamma_chroma)
    second_pow = np.power(second_val + epsilon, gamma_chroma)
    w_focus = best_pow / (best_pow + second_pow + epsilon)
    confidence = (best_val - second_val) / (best_val + second_val + epsilon)
    ambiguous = confidence < 0.25
    if np.any(ambiguous):
        w_smooth = cv2.GaussianBlur(w_focus.astype(np.float32), (0, 0), sigmaX=1.0, sigmaY=1.0, borderType=cv2.BORDER_REFLECT)
        w_focus = np.where(ambiguous, w_smooth, w_focus)

    w_ch = w_focus.astype(np.float32)[..., np.newaxis]
    cr = (cr_best * w_ch[..., 0] + cr_second * (1.0 - w_ch[..., 0])).astype(np.float32)
    cb = (cb_best * w_ch[..., 0] + cb_second * (1.0 - w_ch[..., 0])).astype(np.float32)
    cr = np.clip(cr, 0.0, 1.0).astype(np.float32)
    cb = np.clip(cb, 0.0, 1.0).astype(np.float32)

    ycrcb_fused = np.stack([y_fused, cr, cb], axis=2).astype(np.float32)
    result = cv2.cvtColor(ycrcb_fused, cv2.COLOR_YCrCb2RGB)
    result = np.clip(result.astype(np.float32), 0.0, 1.0)
    print("Luma/chroma fusion complete.")
    return result


def blend_laplacian_pyramid(aligned_images, focus_maps, num_levels: int = 3, valid_masks=None):
    print("\nBlending images using Laplacian pyramid fusion...")
    if aligned_images is None or focus_maps is None:
        raise ValueError("Invalid input: aligned_images and focus_maps must be provided.")
    if len(aligned_images) == 0 or len(focus_maps) == 0 or len(aligned_images) != len(focus_maps):
        raise ValueError("Invalid input: aligned_images and focus_maps must be non-empty and have the same length.")

    weight_maps, valid_images = _compute_weight_maps(aligned_images, focus_maps, valid_masks)
    if not weight_maps:
        raise ValueError("No valid weight maps were produced for blending.")

    desired_levels = max(int(num_levels), 3)
    epsilon = 1e-10

    image_laplacians = [_build_laplacian_pyramid(img.astype(np.float32), desired_levels) for img in valid_images]
    weight_gaussians = [_build_gaussian_pyramid(wm.astype(np.float32), desired_levels) for wm in weight_maps]

    actual_levels = min(min(len(p) for p in image_laplacians), min(len(p) for p in weight_gaussians))
    if actual_levels < 1:
        raise ValueError("Failed to construct pyramids for blending.")

    fused_pyramid = []
    for level in range(actual_levels):
        weights_level = [wg[level][..., np.newaxis] if wg[level].ndim == 2 else wg[level] for wg in weight_gaussians]
        weights_stack = np.stack(weights_level, axis=0)
        weights_norm = _normalize_weights_softmax(weights_stack, beta=6.0, epsilon=epsilon)

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
        counts = []
        for label in range(int(num_labels)):
            mask = (refined == label).astype(np.float32)
            counts.append(cv2.boxFilter(mask, -1, (7, 7), normalize=False, borderType=cv2.BORDER_REFLECT))
        filled = np.argmax(np.stack(counts, axis=0), axis=0)
        refined[unknown] = filled[unknown]

    return refined

def blend_weighted(aligned_images, focus_maps, valid_masks=None,
                   depth_map=None, confidence_map=None):
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
    epsilon = 1e-10 # For numerical stability

    weight_maps, valid_images = _compute_weight_maps(aligned_images, focus_maps, valid_masks)

    if not weight_maps:
        raise ValueError("No valid weight maps were produced for blending.")

    weights_stack = np.stack(weight_maps, axis=0).astype(np.float32)
    normalized_weights = _normalize_weights_softmax(weights_stack, beta=6.0, epsilon=epsilon)
    normalized_weights = _apply_depth_prior(normalized_weights, depth_map, confidence_map)

    result = np.zeros_like(valid_images[0], dtype=np.float32)
    for index, image in enumerate(valid_images):
        result += image.astype(np.float32) * normalized_weights[index]
    result = np.clip(result, 0.0, 1.0) # Clip final result to [0, 1]

    print("Weighted blending complete.")
    return result


def blend_direct_map(aligned_images, sharpest_indices, focus_maps=None, valid_masks=None):
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
        best_idx = np.clip(sharpest_indices.astype(np.int32), 0, num_images - 1)
        second_val = np.full((h, w), -np.inf, dtype=np.float32)
        second_idx = np.zeros((h, w), dtype=np.int32)

        for i, fm in enumerate(focus_maps):
            fm_2d = fm[..., 0] if getattr(fm, "ndim", 0) > 2 else fm
            fm_2d = np.nan_to_num(fm_2d.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
            if valid_masks is not None:
                fm_2d = np.where(np.asarray(valid_masks[i]) > 0.5, fm_2d, -1e6)

            selected = best_idx == i
            best_val = np.where(selected, fm_2d, best_val)
            is_second = (~selected) & (fm_2d > second_val)
            second_val = np.where(is_second, fm_2d, second_val)
            second_idx = np.where(is_second, i, second_idx)

        epsilon = 1e-10
        best_val = np.maximum(best_val, 0.0)
        second_val = np.maximum(second_val, 0.0)

        gamma = 4.0
        best_pow = np.power(best_val + epsilon, gamma)
        second_pow = np.power(second_val + epsilon, gamma)
        blend_weight = best_pow / (best_pow + second_pow + epsilon)

        confidence = (best_val - second_val) / (best_val + second_val + epsilon)
        ambiguous = confidence < 0.25
        if np.any(ambiguous):
            w_smooth = cv2.GaussianBlur(blend_weight.astype(np.float32), (0, 0), sigmaX=1.0, sigmaY=1.0, borderType=cv2.BORDER_REFLECT)
            blend_weight = np.where(ambiguous, w_smooth, blend_weight)

        w3 = blend_weight.reshape(h, w, 1).astype(np.float32)
        for i, image in enumerate(aligned_images):
            best_mask = (best_idx == i)[..., np.newaxis]
            second_mask = (second_idx == i)[..., np.newaxis]
            result += image.astype(np.float32) * (best_mask * w3 + second_mask * (1.0 - w3))
        result = np.clip(result.astype(np.float32), 0.0, 1.0)
    else:
        refined_indices = sharpest_indices.astype(np.int32)
        if num_images > 1:
            refined_indices = _refine_indices_majority(refined_indices, num_labels=num_images, window_size=3, iterations=2)
            refined_indices = _refine_indices_majority(refined_indices, num_labels=num_images, window_size=3, iterations=1)
        refined_indices = np.clip(refined_indices, 0, num_images - 1).astype(np.uint16)

        for i, image in enumerate(aligned_images):
            result += image.astype(np.float32) * (refined_indices == i)[..., np.newaxis]
        result = np.clip(result.astype(np.float32), 0.0, 1.0)

    print("Direct map blending complete.")
    return result
