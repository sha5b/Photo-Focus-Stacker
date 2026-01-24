# Context: Settings persistence for Photo Focus Stacker
# Purpose: Load and save UI settings to a user-writable JSON file.

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

from .output_settings import OutputSettings
from .stack_detection_settings import StackDetectionSettings
from .stacking_settings import StackerSettings


_DEFAULT_SETTINGS_FILENAME = "photo_focus_stacker_settings.json"


@dataclass
class AppSettings:
    stacker: StackerSettings
    stack_detection: StackDetectionSettings
    output: OutputSettings

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stacker": self.stacker.to_dict(),
            "stack_detection": self.stack_detection.to_dict(),
            "output": self.output.to_dict(),
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "AppSettings":
        stacker_data = data.get("stacker", {}) if isinstance(data, dict) else {}
        detection_data = data.get("stack_detection", {}) if isinstance(data, dict) else {}
        output_data = data.get("output", {}) if isinstance(data, dict) else {}

        return AppSettings(
            stacker=StackerSettings.from_dict(stacker_data),
            stack_detection=StackDetectionSettings.from_dict(detection_data),
            output=OutputSettings.from_dict(output_data),
        )


def get_default_settings_path() -> str:
    # Use %APPDATA% on Windows when available, otherwise fall back to user home.
    base_dir = os.environ.get("APPDATA") or os.path.expanduser("~")
    return os.path.join(base_dir, _DEFAULT_SETTINGS_FILENAME)


def load_settings(path: Optional[str] = None) -> AppSettings:
    settings_path = path or get_default_settings_path()

    if not os.path.exists(settings_path):
        return AppSettings(
            stacker=StackerSettings(),
            stack_detection=StackDetectionSettings(),
            output=OutputSettings(),
        )

    try:
        with open(settings_path, "r", encoding="utf-8", errors="replace") as f:
            raw = json.load(f)
    except Exception:
        return AppSettings(
            stacker=StackerSettings(),
            stack_detection=StackDetectionSettings(),
            output=OutputSettings(),
        )

    try:
        return AppSettings.from_dict(raw)
    except Exception:
        return AppSettings(
            stacker=StackerSettings(),
            stack_detection=StackDetectionSettings(),
            output=OutputSettings(),
        )


def save_settings(settings: AppSettings, path: Optional[str] = None) -> None:
    settings_path = path or get_default_settings_path()
    settings_dir = os.path.dirname(settings_path)
    if settings_dir:
        os.makedirs(settings_dir, exist_ok=True)

    with open(settings_path, "w", encoding="utf-8") as f:
        json.dump(settings.to_dict(), f, indent=2, sort_keys=True)
