#!/usr/bin/env python3

# Context: Application entry point for Photo Focus Stacker
# Purpose: Launch the PyQt5 GUI main window.
# Notes: The UI is implemented in `src.ui.main_window`.

import sys

from PyQt5.QtWidgets import QApplication

from src.ui.main_window import MainWindow


def main() -> int:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return int(app.exec_())


if __name__ == "__main__":
    raise SystemExit(main())
