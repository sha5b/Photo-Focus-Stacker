# Context: Output settings for Photo Focus Stacker
# Purpose: Define user-configurable output behavior (e.g., output directory) for the UI.

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict


_DEFAULT_OUTPUT_DIR = "results"


@dataclass
class OutputSettings:
    output_dir: str = _DEFAULT_OUTPUT_DIR

    def validated(self) -> "OutputSettings":
        output_dir = str(self.output_dir or "").strip() or _DEFAULT_OUTPUT_DIR
        return OutputSettings(output_dir=output_dir)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self.validated())

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "OutputSettings":
        return OutputSettings(output_dir=str(data.get("output_dir", _DEFAULT_OUTPUT_DIR))).validated()
