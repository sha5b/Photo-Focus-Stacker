#!/usr/bin/env python3
"""Robust, diagnostics-producing registration for focus stacks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import cv2
import numpy as np


@dataclass(frozen=True)
class AlignmentDiagnostic:
    source_index: int
    accepted: bool
    model: str
    correlation: float
    overlap_ratio: float
    message: str = ""
    transform: tuple[tuple[float, ...], ...] = ()


@dataclass
class AlignmentResult:
    images: list[np.ndarray]
    valid_masks: list[np.ndarray]
    source_indices: list[int]
    transforms: list[np.ndarray]
    diagnostics: list[AlignmentDiagnostic]
    reference_index: int


_MOTION_TYPES = {
    "translation": cv2.MOTION_TRANSLATION,
    "euclidean": cv2.MOTION_EUCLIDEAN,
    "affine": cv2.MOTION_AFFINE,
    "homography": cv2.MOTION_HOMOGRAPHY,
}


def _build_gaussian_pyramid(img: np.ndarray, levels: int) -> list[np.ndarray]:
    pyramid = [img]
    for _ in range(max(1, int(levels)) - 1):
        # ECC becomes unstable when the coarsest level contains too few features.
        if img.shape[0] < 64 or img.shape[1] < 64:
            break
        img = cv2.pyrDown(img, borderType=cv2.BORDER_REFLECT)
        pyramid.append(img)
    return pyramid


def _scale_warp_for_finer_level(warp: np.ndarray) -> np.ndarray:
    result = warp.copy()
    if result.shape == (3, 3):
        scale_up = np.diag([2.0, 2.0, 1.0]).astype(np.float32)
        scale_down = np.diag([0.5, 0.5, 1.0]).astype(np.float32)
        return scale_up @ result @ scale_down
    result[0, 2] *= 2.0
    result[1, 2] *= 2.0
    return result


def _as_homography(warp: np.ndarray) -> np.ndarray:
    if warp.shape == (3, 3):
        return warp.astype(np.float32)
    return np.vstack([warp, np.array([0.0, 0.0, 1.0], dtype=np.float32)])


def _gray_for_registration(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(np.clip(image, 0.0, 1.0).astype(np.float32), cv2.COLOR_RGB2GRAY)
    lo, hi = np.percentile(gray, (0.5, 99.5))
    if hi > lo:
        gray = np.clip((gray - lo) / (hi - lo), 0.0, 1.0)
    return gray.astype(np.float32)


def _gradient_mask(gray: np.ndarray, strength: int) -> np.ndarray:
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(gx, gy)
    percentile = float(np.clip(100.0 - float(strength), 50.0, 95.0))
    threshold = float(np.percentile(magnitude, percentile))
    if not np.isfinite(threshold) or threshold <= 0.0:
        return np.full(gray.shape, 255, dtype=np.uint8)
    mask = (magnitude >= threshold).astype(np.uint8) * 255
    return cv2.dilate(mask, None, iterations=2)


def _fallback_models(model: str) -> list[str]:
    order = ["homography", "affine", "euclidean", "translation"]
    return order[order.index(model):]


def _validate_images(images: list[np.ndarray]) -> tuple[int, int]:
    if not images:
        raise ValueError("No images were supplied for alignment.")
    h, w = images[0].shape[:2]
    for index, image in enumerate(images):
        if image is None or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Image {index} is not an HxWx3 RGB image.")
        if image.shape[:2] != (h, w):
            raise ValueError(
                f"All stack images must have the same dimensions; image {index} is "
                f"{image.shape[:2]}, expected {(h, w)}."
            )
    return h, w


def align_images_detailed(
    images: list[np.ndarray], num_pyramid_levels: int = 3,
    max_iterations: int = 100, epsilon: float = 1e-5,
    gradient_threshold: int = 10, motion_model: str = "affine",
    reference_index: Optional[int] = None, min_correlation: float = 0.35,
    min_overlap: float = 0.70, failure_policy: str = "exclude",
    cancel_check: Optional[Callable[[], None]] = None, release_sources: bool = False,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> AlignmentResult:
    """Align a stack and return transforms, validity masks, and quality diagnostics."""
    h, w = _validate_images(images)
    if motion_model not in _MOTION_TYPES:
        raise ValueError(f"Unsupported alignment model: {motion_model}")
    if failure_policy not in ("exclude", "error", "keep"):
        raise ValueError("failure_policy must be 'exclude', 'error', or 'keep'.")

    ref_index = len(images) // 2 if reference_index is None else int(reference_index)
    if not 0 <= ref_index < len(images):
        raise ValueError("reference_index is outside the stack.")

    reference = images[ref_index].astype(np.float32, copy=False)
    ref_gray = _gray_for_registration(reference)
    ref_pyramid = _build_gaussian_pyramid(ref_gray, num_pyramid_levels)
    mask_pyramid = _build_gaussian_pyramid(_gradient_mask(ref_gray, gradient_threshold), len(ref_pyramid))
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, int(max_iterations), float(epsilon))

    aligned: list[np.ndarray] = []
    valid_masks: list[np.ndarray] = []
    source_indices: list[int] = []
    transforms: list[np.ndarray] = []
    diagnostics: list[AlignmentDiagnostic] = []

    for source_index, image in enumerate(images):
        if cancel_check:
            cancel_check()
        if source_index == ref_index:
            transform = np.eye(3, dtype=np.float32)
            aligned.append(reference)
            valid_masks.append(np.ones((h, w), dtype=np.float32))
            source_indices.append(source_index)
            transforms.append(transform)
            diagnostics.append(AlignmentDiagnostic(
                source_index, True, "identity", 1.0, 1.0, "",
                tuple(tuple(float(value) for value in row) for row in transform),
            ))
            if release_sources:
                images[source_index] = None
            if progress_callback:
                progress_callback(source_index + 1, len(images))
            continue

        image_pyramid = _build_gaussian_pyramid(_gray_for_registration(image), len(ref_pyramid))
        last_error = "ECC did not converge"
        accepted = False
        final_model = motion_model
        final_cc = float("nan")
        final_transform = np.eye(3, dtype=np.float32)

        for candidate_model in _fallback_models(motion_model):
            warp = np.eye(3, dtype=np.float32) if candidate_model == "homography" else np.eye(2, 3, dtype=np.float32)
            try:
                coarse_ref = ref_pyramid[-1]
                coarse_image = image_pyramid[-1]
                shift, phase_response = cv2.phaseCorrelate(coarse_ref, coarse_image)
                if np.isfinite(phase_response) and phase_response >= 0.05:
                    warp[0, 2] = float(shift[0])
                    warp[1, 2] = float(shift[1])
                for level in range(len(ref_pyramid) - 1, -1, -1):
                    if cancel_check:
                        cancel_check()
                    if level < len(ref_pyramid) - 1:
                        warp = _scale_warp_for_finer_level(warp)
                    final_cc, warp = cv2.findTransformECC(
                        ref_pyramid[level], image_pyramid[level], warp,
                        _MOTION_TYPES[candidate_model], criteria,
                        inputMask=mask_pyramid[level], gaussFiltSize=5,
                    )
                final_transform = _as_homography(warp)
                final_model = candidate_model
                accepted = np.isfinite(final_cc) and final_cc >= float(min_correlation)
                if accepted:
                    break
                last_error = f"correlation {final_cc:.3f} is below {min_correlation:.3f}"
            except cv2.error as exc:
                last_error = str(exc).splitlines()[0]

        if accepted:
            warped_mask = cv2.warpPerspective(
                np.ones((h, w), dtype=np.uint8), final_transform, (w, h),
                flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
                borderMode=cv2.BORDER_CONSTANT, borderValue=0,
            ).astype(np.float32)
            overlap = float(np.mean(warped_mask > 0.5))
            accepted = overlap >= float(min_overlap)
            if not accepted:
                last_error = f"overlap {overlap:.3f} is below {min_overlap:.3f}"
        else:
            overlap = 0.0

        diagnostics.append(AlignmentDiagnostic(
            source_index, accepted, final_model,
            float(final_cc) if np.isfinite(final_cc) else 0.0, overlap,
            "" if accepted else last_error,
            tuple(tuple(float(value) for value in row) for row in final_transform),
        ))

        if not accepted:
            if failure_policy == "error":
                raise ValueError(f"Alignment rejected image {source_index}: {last_error}")
            if failure_policy == "exclude":
                if release_sources:
                    images[source_index] = None
                if progress_callback:
                    progress_callback(source_index + 1, len(images))
                continue
            final_transform = np.eye(3, dtype=np.float32)
            warped_mask = np.ones((h, w), dtype=np.float32)

        warped = cv2.warpPerspective(
            image.astype(np.float32), final_transform, (w, h),
            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        )
        aligned.append(warped.astype(np.float32))
        valid_masks.append(warped_mask)
        source_indices.append(source_index)
        transforms.append(final_transform)
        if release_sources:
            images[source_index] = None
        if progress_callback:
            progress_callback(source_index + 1, len(images))

    if len(aligned) < 2:
        raise ValueError("Fewer than two images passed alignment quality checks.")
    return AlignmentResult(aligned, valid_masks, source_indices, transforms, diagnostics, ref_index)


def align_images(images, num_pyramid_levels=3, max_iterations=100, epsilon=1e-5,
                 gradient_threshold=10, **kwargs):
    """Backward-compatible API returning only registered images."""
    return align_images_detailed(
        images, num_pyramid_levels=num_pyramid_levels, max_iterations=max_iterations,
        epsilon=epsilon, gradient_threshold=gradient_threshold, **kwargs,
    ).images
