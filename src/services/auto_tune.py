#!/usr/bin/env python3

# Context: Auto-tuning utilities for Photo Focus Stacker
# Purpose: Analyze input image stacks quickly and recommend stacking parameters for speed/quality tradeoffs.
# Notes: Used by `src.ui.main_window.MainWindow` when Auto Tune is enabled.

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np

from src.config.stacking_settings import StackerSettings


@dataclass(frozen=True)
class AutoTuneReport:
    max_dim: int
    megapixels: float
    contrast_std: float
    motion_ratio: float


def _load_gray_preview(path: str, max_dim: int = 640) -> Optional[np.ndarray]:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    h, w = img.shape[:2]
    scale = 1.0
    current_max = int(max(h, w))
    if current_max > int(max_dim) and current_max > 0:
        scale = float(max_dim) / float(current_max)

    if scale < 1.0:
        new_w = max(int(round(w * scale)), 2)
        new_h = max(int(round(h * scale)), 2)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    return img


def _estimate_motion_ratio(img_a_u8: np.ndarray, img_b_u8: np.ndarray) -> float:
    if img_a_u8 is None or img_b_u8 is None:
        return 0.0

    a = img_a_u8.astype(np.float32) / 255.0
    b = img_b_u8.astype(np.float32) / 255.0

    if a.shape != b.shape:
        h = min(a.shape[0], b.shape[0])
        w = min(a.shape[1], b.shape[1])
        a = a[:h, :w]
        b = b[:h, :w]

    try:
        shift, _response = cv2.phaseCorrelate(a, b)
        shift_mag = float(np.hypot(shift[0], shift[1]))
        denom = float(max(a.shape[0], a.shape[1], 1))
        return float(shift_mag / denom)
    except Exception:
        return 0.0


def recommend_stacker_settings(image_paths: List[str], preferred_blend_method: Optional[str] = None) -> tuple[StackerSettings, AutoTuneReport]:
    if not image_paths:
        default = StackerSettings().validated()
        return default, AutoTuneReport(max_dim=0, megapixels=0.0, contrast_std=0.0, motion_ratio=0.0)

    sample_paths: List[str] = []
    idxs = {0, len(image_paths) // 2, len(image_paths) - 1}
    for idx in sorted(idxs):
        if 0 <= idx < len(image_paths):
            sample_paths.append(image_paths[idx])

    previews = [_load_gray_preview(p) for p in sample_paths]
    previews = [p for p in previews if p is not None]

    if not previews:
        default = StackerSettings().validated()
        return default, AutoTuneReport(max_dim=0, megapixels=0.0, contrast_std=0.0, motion_ratio=0.0)

    full = cv2.imread(image_paths[0], cv2.IMREAD_GRAYSCALE)
    if full is None:
        h_full, w_full = previews[0].shape[:2]
    else:
        h_full, w_full = full.shape[:2]

    max_dim = int(max(h_full, w_full))
    megapixels = float(h_full * w_full) / 1_000_000.0

    contrast_std = float(np.mean([np.std(p.astype(np.float32) / 255.0) for p in previews]))

    motion_ratio = 0.0
    if len(previews) >= 2:
        motion_ratio = _estimate_motion_ratio(previews[0], previews[-1])

    if motion_ratio > 0.015:
        num_pyramid_levels = 5
    elif motion_ratio > 0.007:
        num_pyramid_levels = 4
    elif motion_ratio > 0.003:
        num_pyramid_levels = 3
    else:
        num_pyramid_levels = 2 if max_dim >= 2200 else 1

    if contrast_std < 0.04:
        gradient_threshold = 5
    elif contrast_std < 0.07:
        gradient_threshold = 8
    elif contrast_std < 0.11:
        gradient_threshold = 10
    else:
        gradient_threshold = 12

    if max_dim >= 3200:
        focus_window_size = 9 if contrast_std < 0.06 else 7
    elif max_dim >= 1800:
        focus_window_size = 7
    else:
        focus_window_size = 5

    blend_method = None
    if isinstance(preferred_blend_method, str) and preferred_blend_method in (
        "weighted",
        "direct_map",
        "laplacian_pyramid",
        "guided_weighted",
        "luma_weighted_chroma_pick",
    ):
        blend_method = preferred_blend_method

    if blend_method is None:
        if len(image_paths) >= 4 and (contrast_std >= 0.06 or max_dim >= 1800):
            blend_method = "laplacian_pyramid"
        else:
            blend_method = "weighted"

    if blend_method == "direct_map":
        num_pyramid_levels = max(int(num_pyramid_levels), 3)
        focus_window_size = max(int(focus_window_size), 7)

    if blend_method == "guided_weighted" and max_dim >= 3200:
        focus_window_size = max(int(focus_window_size), 7)

    if blend_method == "luma_weighted_chroma_pick" and max_dim >= 3200:
        focus_window_size = max(int(focus_window_size), 7)

    settings = StackerSettings(
        focus_window_size=focus_window_size,
        sharpen_strength=0.0,
        num_pyramid_levels=num_pyramid_levels,
        gradient_threshold=gradient_threshold,
        blend_method=blend_method,
    ).validated()

    return settings, AutoTuneReport(
        max_dim=max_dim,
        megapixels=megapixels,
        contrast_std=contrast_std,
        motion_ratio=motion_ratio,
    )
