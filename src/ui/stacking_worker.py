# Context: Background worker thread for Photo Focus Stacker UI
# Purpose: Run the focus stacking pipeline on a QThread to keep the PyQt UI responsive.

from __future__ import annotations

from typing import List, Optional

from PyQt5.QtCore import QThread, pyqtSignal

from src.config.stacking_settings import StackerSettings
from src.core.focus_stacker import FocusStacker, StackingCancelledException


class FocusStackingWorker(QThread):
    finished = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, stacker_settings: StackerSettings, image_paths: List[str], color_space: str):
        super().__init__()
        self._stacker_settings = stacker_settings
        self._image_paths = image_paths
        self._color_space = color_space

        self._stopped = False
        self._stacker: Optional[FocusStacker] = None

    def run(self) -> None:
        try:
            self._stacker = FocusStacker(**self._stacker_settings.to_focus_stacker_kwargs())
            result = self._stacker.process_stack(self._image_paths, self._color_space)
            if not self._stopped:
                self.finished.emit(result)
        except StackingCancelledException:
            pass
        except Exception as e:
            if not self._stopped:
                self.error.emit(str(e))

    def stop(self) -> None:
        self._stopped = True
        if self._stacker is not None:
            self._stacker.request_stop()
