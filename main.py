#!/usr/bin/env python3

# Context: Application entry point for Photo Focus Stacker
# Purpose: Launch the GUI through the shared CLI entry point.
# Notes: Environment preparation must happen before importing PyQt5.

from src.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
