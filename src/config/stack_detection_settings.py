# Context: Stack detection settings for Photo Focus Stacker
# Purpose: Provide a validated settings model describing how image files are grouped into stacks.

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Literal, Optional

StackDetectionMode = Literal["auto", "legacy", "common_suffix", "fixed_size", "regex"]


@dataclass
class StackDetectionSettings:
    mode: StackDetectionMode = "auto"

    fixed_stack_size: int = 0

    regex_pattern: str = ""

    def validated(self) -> "StackDetectionSettings":
        mode: StackDetectionMode = self.mode if self.mode in (
            "auto",
            "legacy",
            "common_suffix",
            "fixed_size",
            "regex",
        ) else "auto"

        fixed_stack_size = int(self.fixed_stack_size)
        if fixed_stack_size < 0:
            fixed_stack_size = 0

        regex_pattern = str(self.regex_pattern or "")

        return StackDetectionSettings(
            mode=mode,
            fixed_stack_size=fixed_stack_size,
            regex_pattern=regex_pattern,
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self.validated())

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "StackDetectionSettings":
        return StackDetectionSettings(
            mode=data.get("mode", "auto"),
            fixed_stack_size=int(data.get("fixed_stack_size", 0)),
            regex_pattern=str(data.get("regex_pattern", "")),
        ).validated()
