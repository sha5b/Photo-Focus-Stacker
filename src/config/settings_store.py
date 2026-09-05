# Context: Settings persistence for Photo Focus Stacker
# Purpose: Load and save UI settings to a user-writable JSON file.

from __future__ import annotations

import json
import os
import tempfile
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
    last_input_dir: str = ""
    auto_tune_enabled: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stacker": self.stacker.to_dict(),
            "stack_detection": self.stack_detection.to_dict(),
            "output": self.output.to_dict(),
            "last_input_dir": str(self.last_input_dir or ""),
            "auto_tune_enabled": bool(self.auto_tune_enabled),
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> AppSettings:
        stacker_data = data.get("stacker", {}) if isinstance(data, dict) else {}
        detection_data = data.get("stack_detection", {}) if isinstance(data, dict) else {}
        output_data = data.get("output", {}) if isinstance(data, dict) else {}
        last_input_dir = str(data.get("last_input_dir", "")) if isinstance(data, dict) else ""
        auto_tune_enabled = bool(data.get("auto_tune_enabled", False)) if isinstance(data, dict) else False

        return AppSettings(
            stacker=StackerSettings.from_dict(stacker_data),
            stack_detection=StackDetectionSettings.from_dict(detection_data),
            output=OutputSettings.from_dict(output_data),
            last_input_dir=last_input_dir,
            auto_tune_enabled=auto_tune_enabled,
        )


def get_default_settings_path() -> str:
    appdata = os.environ.get("APPDATA")
    if appdata:
        return os.path.join(appdata, _DEFAULT_SETTINGS_FILENAME)
    config_home = os.environ.get("XDG_CONFIG_HOME") or os.path.join(os.path.expanduser("~"), ".config")
    return os.path.join(config_home, "photo-focus-stacker", "settings.json")


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

    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=settings_dir or ".",
            prefix=".settings-", suffix=".tmp", delete=False,
        ) as handle:
            temporary_path = handle.name
            json.dump(settings.to_dict(), handle, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, settings_path)
    finally:
        if temporary_path and os.path.exists(temporary_path):
            os.unlink(temporary_path)
