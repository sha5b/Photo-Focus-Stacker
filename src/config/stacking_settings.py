# Context: Typed stacking settings for Photo Focus Stacker
# Purpose: Provide a validated settings model for FocusStacker parameters and UI persistence.

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Literal

BlendMethod = Literal["weighted", "direct_map", "laplacian_pyramid", "guided_weighted", "luma_weighted_chroma_pick"]
FocusMeasureMethod = Literal["laplacian_var", "tenengrad", "sml", "multiscale"]
AlignmentModel = Literal["translation", "euclidean", "affine", "homography"]


@dataclass
class StackerSettings:
    focus_window_size: int = 7
    focus_measure_method: FocusMeasureMethod = "laplacian_var"
    sharpen_strength: float = 0.0
    num_pyramid_levels: int = 3
    gradient_threshold: int = 10
    blend_method: BlendMethod = "weighted"
    alignment_model: AlignmentModel = "affine"
    crop_to_common_area: bool = True
    linear_light_blending: bool = True
    normalize_exposure: bool = True
    focus_analysis_max_dim: int = 0
    cache_memory_limit_mb: int = 512
    focus_workers: int = 2
    photogrammetry_mode: bool = False

    def validated(self) -> StackerSettings:
        focus_window_size = int(self.focus_window_size)
        focus_window_size = max(focus_window_size, 3)
        focus_window_size = min(focus_window_size, 21)
        if focus_window_size % 2 == 0:
            focus_window_size += 1

        num_pyramid_levels = int(self.num_pyramid_levels)
        num_pyramid_levels = max(num_pyramid_levels, 1)
        num_pyramid_levels = min(num_pyramid_levels, 6)

        gradient_threshold = int(self.gradient_threshold)
        gradient_threshold = max(gradient_threshold, 1)
        gradient_threshold = min(gradient_threshold, 100)

        sharpen_strength = float(self.sharpen_strength)
        sharpen_strength = max(sharpen_strength, 0.0)
        sharpen_strength = min(sharpen_strength, 3.0)

        blend_method: BlendMethod = self.blend_method if self.blend_method in ("weighted", "direct_map", "laplacian_pyramid", "guided_weighted", "luma_weighted_chroma_pick") else "weighted"

        focus_measure_method: FocusMeasureMethod = self.focus_measure_method if self.focus_measure_method in (
            "laplacian_var",
            "tenengrad",
            "sml",
            "multiscale",
        ) else "laplacian_var"

        alignment_model: AlignmentModel = self.alignment_model if self.alignment_model in (
            "translation", "euclidean", "affine", "homography"
        ) else "affine"
        focus_analysis_max_dim = max(0, min(int(self.focus_analysis_max_dim), 12_000))
        cache_memory_limit_mb = max(0, min(int(self.cache_memory_limit_mb), 16_384))
        focus_workers = max(1, min(int(self.focus_workers), 16))
        photogrammetry_mode = bool(self.photogrammetry_mode)
        if photogrammetry_mode:
            alignment_model = "euclidean"

        return StackerSettings(
            focus_window_size=focus_window_size,
            focus_measure_method=focus_measure_method,
            sharpen_strength=sharpen_strength,
            num_pyramid_levels=num_pyramid_levels,
            gradient_threshold=gradient_threshold,
            blend_method=blend_method,
            alignment_model=alignment_model,
            crop_to_common_area=bool(self.crop_to_common_area) and not photogrammetry_mode,
            linear_light_blending=bool(self.linear_light_blending),
            normalize_exposure=bool(self.normalize_exposure),
            focus_analysis_max_dim=focus_analysis_max_dim,
            cache_memory_limit_mb=cache_memory_limit_mb,
            focus_workers=focus_workers,
            photogrammetry_mode=photogrammetry_mode,
        )

    def to_focus_stacker_kwargs(self) -> Dict[str, Any]:
        validated = self.validated()
        return {
            "focus_window_size": validated.focus_window_size,
            "focus_measure_method": validated.focus_measure_method,
            "sharpen_strength": validated.sharpen_strength,
            "num_pyramid_levels": validated.num_pyramid_levels,
            "gradient_threshold": validated.gradient_threshold,
            "blend_method": validated.blend_method,
            "alignment_model": validated.alignment_model,
            "crop_to_common_area": validated.crop_to_common_area,
            "linear_light_blending": validated.linear_light_blending,
            "normalize_exposure": validated.normalize_exposure,
            "focus_analysis_max_dim": validated.focus_analysis_max_dim,
            "cache_memory_limit_mb": validated.cache_memory_limit_mb,
            "focus_workers": validated.focus_workers,
            "photogrammetry_mode": validated.photogrammetry_mode,
        }

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self.validated())

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> StackerSettings:
        return StackerSettings(
            focus_window_size=int(data.get("focus_window_size", 7)),
            focus_measure_method=data.get("focus_measure_method", "laplacian_var"),
            sharpen_strength=float(data.get("sharpen_strength", 0.0)),
            num_pyramid_levels=int(data.get("num_pyramid_levels", 3)),
            gradient_threshold=int(data.get("gradient_threshold", 10)),
            blend_method=data.get("blend_method", "weighted"),
            alignment_model=data.get("alignment_model", "affine"),
            crop_to_common_area=bool(data.get("crop_to_common_area", True)),
            linear_light_blending=bool(data.get("linear_light_blending", True)),
            normalize_exposure=bool(data.get("normalize_exposure", True)),
            focus_analysis_max_dim=int(data.get("focus_analysis_max_dim", 0)),
            cache_memory_limit_mb=int(data.get("cache_memory_limit_mb", 512)),
            focus_workers=int(data.get("focus_workers", 2)),
            photogrammetry_mode=bool(data.get("photogrammetry_mode", False)),
        ).validated()
