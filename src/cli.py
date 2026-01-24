# Context: Command-line entry points for Photo Focus Stacker
# Purpose: Provide a stable console script target for uv (e.g., `uv run photostacker`).
# Notes: This launches the PyQt5 GUI from `src.ui.main_window`.

from __future__ import annotations

import sys

from PyQt5.QtWidgets import QApplication

from src.ui.main_window import MainWindow


def main() -> int:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return int(app.exec_())
