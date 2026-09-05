"""Regression coverage for the OpenCV/PyQt Linux plugin collision."""

from __future__ import annotations

import os
import subprocess
import sys

from src.cli import _prepare_qt_environment


def test_qt_environment_removes_opencv_plugin_path(monkeypatch):
    monkeypatch.setenv(
        "QT_QPA_PLATFORM_PLUGIN_PATH",
        "/tmp/site-packages/cv2/qt/plugins",
    )
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")

    _prepare_qt_environment()

    assert "QT_QPA_PLATFORM_PLUGIN_PATH" not in os.environ
    assert os.environ["QT_QPA_PLATFORM"] == "offscreen"


def test_opencv_and_pyqt_can_start_in_same_process(tmp_path):
    script = """
import os
import cv2
from PyQt5.QtWidgets import QApplication, QScrollArea
from src.ui.main_window import MainWindow

assert hasattr(cv2, "ximgproc")
assert "/cv2/qt/plugins" not in os.environ.get(
    "QT_QPA_PLATFORM_PLUGIN_PATH", ""
).replace("\\\\", "/").lower()
app = QApplication([])
window = MainWindow()
window.show()
app.processEvents()
assert not window.findChildren(QScrollArea)
assert [window.settings_tabs.tabText(i) for i in range(4)] == [
    "Stacks", "Quality", "Advanced", "Output"
]
assert window.process_btn.isVisible()
window.close()
print(app.platformName())
"""
    environment = os.environ.copy()
    environment["QT_QPA_PLATFORM"] = "offscreen"
    environment["XDG_CONFIG_HOME"] = str(tmp_path)
    environment.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)

    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        env=environment,
        text=True,
        timeout=20,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "offscreen"
