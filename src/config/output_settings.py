# Context: Output settings for Photo Focus Stacker
# Purpose: Define user-configurable output behavior (e.g., output directory) for the UI.

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict

_DEFAULT_OUTPUT_DIR = "results"


@dataclass
class OutputSettings:
    output_dir: str = _DEFAULT_OUTPUT_DIR
    output_format: str = "TIFF"
    bit_depth: int = 16
    preserve_metadata: bool = True
    export_maps: bool = False

    def validated(self) -> OutputSettings:
        output_dir = str(self.output_dir or "").strip() or _DEFAULT_OUTPUT_DIR
        bit_depth = 16 if int(self.bit_depth) == 16 else 8
        output_format = str(self.output_format).upper()
        if output_format not in ("JPEG", "PNG", "TIFF"):
            output_format = "TIFF"
        if output_format == "JPEG":
            bit_depth = 8
        return OutputSettings(
            output_dir=output_dir,
            output_format=output_format,
            bit_depth=bit_depth,
            preserve_metadata=bool(self.preserve_metadata),
            export_maps=bool(self.export_maps),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self.validated())

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> OutputSettings:
        return OutputSettings(
            output_dir=str(data.get("output_dir", _DEFAULT_OUTPUT_DIR)),
            output_format=str(data.get("output_format", "TIFF")),
            bit_depth=int(data.get("bit_depth", 16)),
            preserve_metadata=bool(data.get("preserve_metadata", True)),
            export_maps=bool(data.get("export_maps", False)),
        ).validated()
