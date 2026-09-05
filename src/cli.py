"""Console and GUI entry point for Photo Focus Stacker."""

from __future__ import annotations

import argparse
import os
import sys

from src.config.stacking_settings import StackerSettings


def _prepare_qt_environment() -> None:
    """Select native Wayland and discard OpenCV's incompatible Qt plugin path."""
    plugin_path = os.environ.get("QT_QPA_PLATFORM_PLUGIN_PATH", "")
    normalized_path = plugin_path.replace("\\", "/").lower()
    if "/cv2/qt/plugins" in normalized_path:
        os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)

    if (
        sys.platform.startswith("linux")
        and os.environ.get("XDG_SESSION_TYPE", "").lower() == "wayland"
        and os.environ.get("WAYLAND_DISPLAY")
    ):
        os.environ.setdefault("QT_QPA_PLATFORM", "wayland")


def _preset(name: str) -> StackerSettings:
    if name == "fast":
        return StackerSettings(
            num_pyramid_levels=1, focus_analysis_max_dim=1200,
            blend_method="direct_map", linear_light_blending=False,
            cache_memory_limit_mb=0,
        ).validated()
    if name == "balanced":
        return StackerSettings(
            num_pyramid_levels=3, focus_analysis_max_dim=2000,
            blend_method="weighted", cache_memory_limit_mb=0,
        ).validated()
    if name == "photogrammetry":
        return StackerSettings(
            num_pyramid_levels=4, focus_measure_method="multiscale",
            blend_method="guided_weighted", alignment_model="euclidean",
            crop_to_common_area=False, photogrammetry_mode=True,
            cache_memory_limit_mb=0,
        ).validated()
    return StackerSettings(
        num_pyramid_levels=5, gradient_threshold=8,
        focus_window_size=5, focus_measure_method="multiscale",
        blend_method="guided_weighted", focus_analysis_max_dim=0,
        cache_memory_limit_mb=0,
    ).validated()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="High-quality macro focus stacking")
    parser.add_argument("images", nargs="*", help="Ordered source images in one focus stack")
    parser.add_argument("-o", "--output", help="Output JPEG, PNG, or TIFF path")
    parser.add_argument("--preset", choices=("fast", "balanced", "quality", "photogrammetry"), default="quality")
    parser.add_argument("--bit-depth", choices=(8, 16), type=int, default=16)
    parser.add_argument("--export-maps", action="store_true", help="Export depth, confidence, and alignment diagnostics")
    parser.add_argument("--no-linear", action="store_true", help="Disable linear-light fusion")
    parser.add_argument("--no-crop", action="store_true", help="Preserve the reference-frame canvas")
    return parser


def _run_batch(args: argparse.Namespace) -> int:
    if len(args.images) < 2:
        raise SystemExit("At least two input images are required.")
    if not args.output:
        raise SystemExit("--output is required for command-line processing.")

    settings = _preset(args.preset)
    if args.no_linear or args.no_crop:
        values = settings.to_dict()
        if args.no_linear:
            values["linear_light_blending"] = False
        if args.no_crop:
            values["crop_to_common_area"] = False
        settings = StackerSettings.from_dict(values)

    from src.core import utils
    from src.core.focus_stacker import FocusStacker

    stacker = FocusStacker(
        **settings.to_focus_stacker_kwargs(),
        progress_callback=lambda value: print(f"Progress: {value}%", end="\r", flush=True),
    )
    result = stacker.process_stack(args.images)
    extension = os.path.splitext(args.output)[1].lower()
    format_name = {".jpg": "JPEG", ".jpeg": "JPEG", ".png": "PNG", ".tif": "TIFF", ".tiff": "TIFF"}.get(extension)
    if format_name is None:
        raise SystemExit("Output extension must be .jpg, .jpeg, .png, .tif, or .tiff.")
    output_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(output_dir, exist_ok=True)
    bit_depth = 8 if format_name == "JPEG" else args.bit_depth
    stacker.save_image(result, args.output, format=format_name, bit_depth=bit_depth)
    if args.export_maps:
        utils.save_focus_maps(stacker.depth_map, stacker.confidence_map, args.output)
        utils.save_alignment_report(stacker.alignment_diagnostics, args.output, stacker.source_paths)
    print(f"\nSaved {args.output}")
    return 0


def main() -> int:
    args = _parser().parse_args()
    if args.images or args.output:
        return _run_batch(args)

    _prepare_qt_environment()

    from PyQt5.QtWidgets import QApplication

    from src.ui.main_window import MainWindow

    # Older OpenCV GUI wheels set this while importing cv2. Clear it again so
    # an existing environment cannot redirect PyQt to OpenCV's bundled Qt.
    _prepare_qt_environment()
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return int(app.exec_())
