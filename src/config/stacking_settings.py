# Context: Typed stacking settings for Photo Focus Stacker
# Purpose: Provide a validated settings model for FocusStacker parameters and UI persistence.

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Literal

BlendMethod = Literal["weighted", "direct_map", "laplacian_pyramid", "guided_weighted", "luma_weighted_chroma_pick"]
FocusMeasureMethod = Literal["laplacian_var", "tenengrad", "sml"]


@dataclass
class StackerSettings:
    focus_window_size: int = 7
    focus_measure_method: FocusMeasureMethod = "laplacian_var"
    sharpen_strength: float = 0.0
    num_pyramid_levels: int = 3
    gradient_threshold: int = 10
    blend_method: BlendMethod = "weighted"

    def validated(self) -> "StackerSettings":
        focus_window_size = int(self.focus_window_size)
        if focus_window_size < 3:
            focus_window_size = 3
        if focus_window_size > 21:
            focus_window_size = 21
        if focus_window_size % 2 == 0:
            focus_window_size += 1

        num_pyramid_levels = int(self.num_pyramid_levels)
        if num_pyramid_levels < 1:
            num_pyramid_levels = 1
        if num_pyramid_levels > 6:
            num_pyramid_levels = 6

        gradient_threshold = int(self.gradient_threshold)
        if gradient_threshold < 1:
            gradient_threshold = 1
        if gradient_threshold > 100:
            gradient_threshold = 100

        sharpen_strength = float(self.sharpen_strength)
        if sharpen_strength < 0.0:
            sharpen_strength = 0.0
        if sharpen_strength > 3.0:
            sharpen_strength = 3.0

        blend_method: BlendMethod = self.blend_method if self.blend_method in ("weighted", "direct_map", "laplacian_pyramid", "guided_weighted", "luma_weighted_chroma_pick") else "weighted"

        focus_measure_method: FocusMeasureMethod = self.focus_measure_method if self.focus_measure_method in (
            "laplacian_var",
            "tenengrad",
            "sml",
        ) else "laplacian_var"

        return StackerSettings(
            focus_window_size=focus_window_size,
            focus_measure_method=focus_measure_method,
            sharpen_strength=sharpen_strength,
            num_pyramid_levels=num_pyramid_levels,
            gradient_threshold=gradient_threshold,
            blend_method=blend_method,
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
        }

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self.validated())

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "StackerSettings":
        return StackerSettings(
            focus_window_size=int(data.get("focus_window_size", 7)),
            focus_measure_method=data.get("focus_measure_method", "laplacian_var"),
            sharpen_strength=float(data.get("sharpen_strength", 0.0)),
            num_pyramid_levels=int(data.get("num_pyramid_levels", 3)),
            gradient_threshold=int(data.get("gradient_threshold", 10)),
            blend_method=data.get("blend_method", "weighted"),
        ).validated()
