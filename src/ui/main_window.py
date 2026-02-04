# Context: Main PyQt5 window for Photo Focus Stacker
# Purpose: Provide the user interface for loading images, detecting stacks, configuring parameters, and running processing.

from __future__ import annotations

import os
import re
from datetime import datetime
from typing import Optional

from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QProgressBar,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from src.config.output_settings import OutputSettings
from src.config.settings_store import AppSettings, load_settings, save_settings
from src.config.stack_detection_settings import StackDetectionSettings
from src.config.stacking_settings import StackerSettings
from src.core import utils
from src.services.auto_tune import recommend_stacker_settings
from src.services.stack_detection import detect_stacks
from src.ui.stacking_worker import FocusStackingWorker


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self._settings_path: Optional[str] = None
        self._app_settings: AppSettings = load_settings(self._settings_path)

        self._image_paths: list[str] = []
        self._stack_items: list[tuple[str, list[str]]] = []

        self._current_stack_index: int = 0
        self._processed_stack_count: int = 0
        self._worker: Optional[FocusStackingWorker] = None

        self.init_ui()
        self._apply_loaded_settings_to_ui()

    # ----------------------------
    # UI construction
    # ----------------------------

    def init_ui(self) -> None:
        self.setWindowTitle("Focus Stacking Tool")
        self.setMinimumSize(560, 460)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        self.load_btn = QPushButton("Load Images")
        self.load_btn.clicked.connect(self.load_images)

        self.stack_detection_group = self._create_stack_detection_group()
        self.params_group = self._create_parameter_group()
        self.output_group = self._create_output_group()
        self.action_buttons_layout = self._create_action_buttons()

        self.status_label = QLabel("Ready")
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)

        layout.addWidget(self.load_btn)
        layout.addWidget(self.stack_detection_group)
        layout.addWidget(self.params_group)
        layout.addWidget(self.output_group)
        layout.addLayout(self.action_buttons_layout)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.status_label)

    def _create_stack_detection_group(self) -> QGroupBox:
        group = QGroupBox("Stack Detection")
        grid = QGridLayout()
        grid.setVerticalSpacing(10)
        grid.setHorizontalSpacing(15)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 3)

        self.stack_mode_label = QLabel("Mode:")
        self.stack_mode_combo = QComboBox()
        self.stack_mode_combo.addItems(
            [
                "Auto (Recommended)",
                "Legacy (name_123-45)",
                "Common suffix (strip last number)",
                "Fixed stack size",
                "Regex (Advanced)",
            ]
        )
        self.stack_mode_combo.currentIndexChanged.connect(self._on_stack_detection_changed)

        self.fixed_size_label = QLabel("Fixed stack size:")
        self.fixed_size_spin = QSpinBox()
        self.fixed_size_spin.setRange(0, 10_000)
        self.fixed_size_spin.valueChanged.connect(self._on_stack_detection_changed)

        self.regex_label = QLabel("Regex pattern:")
        self.regex_edit = QLineEdit()
        self.regex_edit.setPlaceholderText("Example: ^(.*_\\d+)-(\\d+)$")
        self.regex_edit.textChanged.connect(self._on_stack_detection_changed)

        grid.addWidget(self.stack_mode_label, 0, 0)
        grid.addWidget(self.stack_mode_combo, 0, 1)
        grid.addWidget(self.fixed_size_label, 1, 0)
        grid.addWidget(self.fixed_size_spin, 1, 1)
        grid.addWidget(self.regex_label, 2, 0)
        grid.addWidget(self.regex_edit, 2, 1)

        group.setLayout(grid)
        return group

    def _create_parameter_group(self) -> QGroupBox:
        group = QGroupBox("Stacking Parameters")
        grid = QGridLayout()
        grid.setVerticalSpacing(10)
        grid.setHorizontalSpacing(15)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 3)

        row = 0

        self.preset_label = QLabel("Preset:")
        self.preset_combo = QComboBox()
        self.preset_combo.addItems(["Custom", "Fast Preview", "Balanced", "Best Quality"])
        self.preset_combo.currentIndexChanged.connect(self._on_preset_changed)
        grid.addWidget(self.preset_label, row, 0)
        grid.addWidget(self.preset_combo, row, 1)
        row += 1

        self.auto_tune_checkbox = QCheckBox("Auto Tune")
        self.auto_tune_checkbox.stateChanged.connect(self._on_auto_tune_changed)
        grid.addWidget(self.auto_tune_checkbox, row, 1)
        row += 1

        self.pyramid_label = QLabel("Alignment Pyramid Levels:")
        self.pyramid_spinbox = QSpinBox()
        self.pyramid_spinbox.setRange(1, 6)
        self.pyramid_spinbox.valueChanged.connect(self._on_stacker_changed)
        grid.addWidget(self.pyramid_label, row, 0)
        grid.addWidget(self.pyramid_spinbox, row, 1)
        row += 1

        self.gradient_label = QLabel("Alignment Mask Threshold:")
        self.gradient_spinbox = QSpinBox()
        self.gradient_spinbox.setRange(1, 100)
        self.gradient_spinbox.valueChanged.connect(self._on_stacker_changed)
        grid.addWidget(self.gradient_label, row, 0)
        grid.addWidget(self.gradient_spinbox, row, 1)
        row += 1

        self.focus_window_label = QLabel("Focus Window Size:")
        self.focus_window_spinbox = QSpinBox()
        self.focus_window_spinbox.setRange(3, 21)
        self.focus_window_spinbox.setSingleStep(2)
        self.focus_window_spinbox.valueChanged.connect(self._on_stacker_changed)
        grid.addWidget(self.focus_window_label, row, 0)
        grid.addWidget(self.focus_window_spinbox, row, 1)
        row += 1

        self.sharpen_label = QLabel("Sharpening Strength:")
        self.sharpen_spinbox = QDoubleSpinBox()
        self.sharpen_spinbox.setRange(0.0, 3.0)
        self.sharpen_spinbox.setSingleStep(0.1)
        self.sharpen_spinbox.setDecimals(2)
        self.sharpen_spinbox.valueChanged.connect(self._on_stacker_changed)
        grid.addWidget(self.sharpen_label, row, 0)
        grid.addWidget(self.sharpen_spinbox, row, 1)
        row += 1

        self.blend_label = QLabel("Blending Method:")
        self.blend_combo = QComboBox()
        self.blend_combo.addItems([
            "Weighted Blending",
            "Direct Map Selection",
            "Laplacian Pyramid Fusion",
            "Guided Weighted (Edge-Aware)",
            "Luma Weighted + Chroma Pick (MFF)",
        ])
        self.blend_combo.currentIndexChanged.connect(self._on_stacker_changed)
        grid.addWidget(self.blend_label, row, 0)
        grid.addWidget(self.blend_combo, row, 1)

        group.setLayout(grid)
        return group

    def _create_output_group(self) -> QGroupBox:
        group = QGroupBox("Output Settings")
        grid = QGridLayout()

        name_label = QLabel("Output Base Name:")
        self.output_name_edit = QLineEdit()
        self.output_name_edit.setPlaceholderText("Default: [Original Stack Name]")

        format_label = QLabel("Format:")
        self.format_combo = QComboBox()
        self.format_combo.addItems(["JPEG", "PNG", "TIFF"])
        self.format_combo.setCurrentText("JPEG")

        color_label = QLabel("Color Space:")
        self.color_combo = QComboBox()
        self.color_combo.addItems(["sRGB"])

        output_dir_label = QLabel("Output Directory:")
        self.output_dir_edit = QLineEdit()
        self.output_dir_edit.setPlaceholderText("Default: results")
        self.output_dir_edit.editingFinished.connect(self._on_output_changed)
        self.output_dir_browse_btn = QPushButton("Browse...")
        self.output_dir_browse_btn.clicked.connect(self._choose_output_dir)

        grid.addWidget(name_label, 0, 0)
        grid.addWidget(self.output_name_edit, 0, 1)
        grid.addWidget(format_label, 1, 0)
        grid.addWidget(self.format_combo, 1, 1)
        grid.addWidget(color_label, 2, 0)
        grid.addWidget(self.color_combo, 2, 1)

        grid.addWidget(output_dir_label, 3, 0)
        grid.addWidget(self.output_dir_edit, 3, 1)
        grid.addWidget(self.output_dir_browse_btn, 3, 2)
        grid.setColumnStretch(1, 1)

        group.setLayout(grid)
        return group

    def _create_action_buttons(self) -> QHBoxLayout:
        self.process_btn = QPushButton("Process Stack")
        self.process_btn.clicked.connect(self.process_stack)

        self.stop_btn = QPushButton("Stop Processing")
        self.stop_btn.clicked.connect(self.stop_processing)
        self.stop_btn.setEnabled(False)

        layout = QHBoxLayout()
        layout.addWidget(self.process_btn)
        layout.addWidget(self.stop_btn)
        return layout

    # ----------------------------
    # Settings wiring
    # ----------------------------

    def _apply_loaded_settings_to_ui(self) -> None:
        stacker = self._app_settings.stacker.validated()
        detection = self._app_settings.stack_detection.validated()
        output = self._app_settings.output.validated()

        self._set_stacker_controls_from_settings(stacker)

        self.auto_tune_checkbox.blockSignals(True)
        self.auto_tune_checkbox.setChecked(bool(getattr(self._app_settings, "auto_tune_enabled", False)))
        self.auto_tune_checkbox.blockSignals(False)

        self.stack_mode_combo.blockSignals(True)
        self.fixed_size_spin.blockSignals(True)
        self.regex_edit.blockSignals(True)
        self.stack_mode_combo.setCurrentIndex(_stack_mode_to_index(detection.mode))
        self.fixed_size_spin.setValue(detection.fixed_stack_size)
        self.regex_edit.setText(detection.regex_pattern)
        self.stack_mode_combo.blockSignals(False)
        self.fixed_size_spin.blockSignals(False)
        self.regex_edit.blockSignals(False)

        self._sync_stack_detection_control_enabled_state()

        self._set_preset_from_settings(stacker)

        self.output_dir_edit.blockSignals(True)
        self.output_dir_edit.setText(output.output_dir)
        self.output_dir_edit.blockSignals(False)

    def _sync_stack_detection_control_enabled_state(self) -> None:
        mode = self._app_settings.stack_detection.mode
        is_fixed_size = mode == "fixed_size"
        is_regex = mode == "regex"

        self.fixed_size_label.setEnabled(is_fixed_size)
        self.fixed_size_spin.setEnabled(is_fixed_size)
        self.regex_label.setEnabled(is_regex)
        self.regex_edit.setEnabled(is_regex)

    def _on_stacker_changed(self) -> None:
        blend_index = int(self.blend_combo.currentIndex())
        if blend_index == 1:
            blend_method = "direct_map"
        elif blend_index == 2:
            blend_method = "laplacian_pyramid"
        elif blend_index == 3:
            blend_method = "guided_weighted"
        elif blend_index == 4:
            blend_method = "luma_weighted_chroma_pick"
        else:
            blend_method = "weighted"

        current = StackerSettings(
            focus_window_size=self.focus_window_spinbox.value(),
            sharpen_strength=float(self.sharpen_spinbox.value()),
            num_pyramid_levels=self.pyramid_spinbox.value(),
            gradient_threshold=self.gradient_spinbox.value(),
            blend_method=blend_method,
        ).validated()

        self._app_settings = AppSettings(
            stacker=current,
            stack_detection=self._app_settings.stack_detection.validated(),
            output=self._app_settings.output.validated(),
            last_input_dir=self._app_settings.last_input_dir,
            auto_tune_enabled=self._app_settings.auto_tune_enabled,
        )
        self._set_preset_from_settings(current)
        self._save_settings_best_effort()

    def _on_auto_tune_changed(self) -> None:
        enabled = bool(self.auto_tune_checkbox.isChecked())
        self._app_settings = AppSettings(
            stacker=self._app_settings.stacker.validated(),
            stack_detection=self._app_settings.stack_detection.validated(),
            output=self._app_settings.output.validated(),
            last_input_dir=self._app_settings.last_input_dir,
            auto_tune_enabled=enabled,
        )
        self._save_settings_best_effort()

    def _on_preset_changed(self) -> None:
        preset_name = self.preset_combo.currentText()
        preset = _preset_to_settings(preset_name)
        if preset is None:
            return

        self._set_stacker_controls_from_settings(preset)
        self._on_stacker_changed()

    def _set_preset_from_settings(self, settings: StackerSettings) -> None:
        preset_name = _settings_to_preset(settings)
        self.preset_combo.blockSignals(True)
        self.preset_combo.setCurrentText(preset_name)
        self.preset_combo.blockSignals(False)

    def _set_stacker_controls_from_settings(self, settings: StackerSettings) -> None:
        self.pyramid_spinbox.blockSignals(True)
        self.gradient_spinbox.blockSignals(True)
        self.focus_window_spinbox.blockSignals(True)
        self.sharpen_spinbox.blockSignals(True)
        self.blend_combo.blockSignals(True)

        self.pyramid_spinbox.setValue(settings.num_pyramid_levels)
        self.gradient_spinbox.setValue(settings.gradient_threshold)
        self.focus_window_spinbox.setValue(settings.focus_window_size)
        self.sharpen_spinbox.setValue(settings.sharpen_strength)
        if settings.blend_method == "direct_map":
            blend_index = 1
        elif settings.blend_method == "laplacian_pyramid":
            blend_index = 2
        elif settings.blend_method == "guided_weighted":
            blend_index = 3
        elif settings.blend_method == "luma_weighted_chroma_pick":
            blend_index = 4
        else:
            blend_index = 0
        self.blend_combo.setCurrentIndex(blend_index)

        self.pyramid_spinbox.blockSignals(False)
        self.gradient_spinbox.blockSignals(False)
        self.focus_window_spinbox.blockSignals(False)
        self.sharpen_spinbox.blockSignals(False)
        self.blend_combo.blockSignals(False)

    def _on_stack_detection_changed(self) -> None:
        mode = _index_to_stack_mode(self.stack_mode_combo.currentIndex())
        detection = StackDetectionSettings(
            mode=mode,
            fixed_stack_size=self.fixed_size_spin.value(),
            regex_pattern=self.regex_edit.text(),
        ).validated()

        self._app_settings = AppSettings(
            stacker=self._app_settings.stacker.validated(),
            stack_detection=detection,
            output=self._app_settings.output.validated(),
            last_input_dir=self._app_settings.last_input_dir,
            auto_tune_enabled=self._app_settings.auto_tune_enabled,
        )

        self._sync_stack_detection_control_enabled_state()
        self._save_settings_best_effort()

    def _save_settings_best_effort(self) -> None:
        try:
            save_settings(self._app_settings, self._settings_path)
        except Exception:
            pass

    def _choose_output_dir(self) -> None:
        selected = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if not selected:
            return
        self.output_dir_edit.setText(selected)
        self._on_output_changed()

    def _on_output_changed(self) -> None:
        output = OutputSettings(output_dir=self.output_dir_edit.text()).validated()
        self._app_settings = AppSettings(
            stacker=self._app_settings.stacker.validated(),
            stack_detection=self._app_settings.stack_detection.validated(),
            output=output,
            last_input_dir=self._app_settings.last_input_dir,
            auto_tune_enabled=self._app_settings.auto_tune_enabled,
        )
        self._save_settings_best_effort()

    # ----------------------------
    # User actions
    # ----------------------------

    def load_images(self) -> None:
        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.ExistingFiles)
        file_dialog.setNameFilter("Images (*.png *.jpg *.jpeg *.tif *.tiff *.bmp *.webp)")

        last_dir = str(getattr(self._app_settings, "last_input_dir", "") or "")
        if last_dir and os.path.isdir(last_dir):
            file_dialog.setDirectory(last_dir)

        if not file_dialog.exec_():
            return

        self._image_paths = file_dialog.selectedFiles()
        if not self._image_paths:
            self.status_label.setText("No images selected.")
            return

        try:
            first_path = self._image_paths[0]
            first_dir = os.path.dirname(first_path)
            if first_dir and os.path.isdir(first_dir):
                self._app_settings.last_input_dir = first_dir
                self._save_settings_best_effort()
        except Exception:
            pass

        detection = self._app_settings.stack_detection.validated()
        try:
            self._stack_items = detect_stacks(
                self._image_paths,
                mode=detection.mode,
                fixed_stack_size=detection.fixed_stack_size,
                regex_pattern=detection.regex_pattern,
            )
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to detect stacks: {e}")
            self._stack_items = []
            return

        if not self._stack_items:
            self._stack_items = [("stack", sorted(self._image_paths))]

        if bool(getattr(self._app_settings, "auto_tune_enabled", False)):
            try:
                sample_paths = self._stack_items[0][1] if self._stack_items and self._stack_items[0][1] else self._image_paths
                preferred_blend = self._app_settings.stacker.validated().blend_method
                tuned_settings, _report = recommend_stacker_settings(sample_paths, preferred_blend_method=preferred_blend)
                self._set_stacker_controls_from_settings(tuned_settings)
                self._app_settings = AppSettings(
                    stacker=tuned_settings,
                    stack_detection=self._app_settings.stack_detection.validated(),
                    output=self._app_settings.output.validated(),
                    last_input_dir=self._app_settings.last_input_dir,
                    auto_tune_enabled=self._app_settings.auto_tune_enabled,
                )
                self._set_preset_from_settings(tuned_settings)
                self._save_settings_best_effort()
            except Exception:
                pass

        num_images_in_stacks = sum(len(item[1]) for item in self._stack_items)
        self.status_label.setText(f"Loaded {num_images_in_stacks} images in {len(self._stack_items)} stacks.")

    def process_stack(self) -> None:
        if not self._image_paths or not self._stack_items:
            QMessageBox.warning(self, "Error", "Please load images first (ensure stacks were detected).")
            return

        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self._current_stack_index = 0
        self._processed_stack_count = 0
        self.stop_btn.setEnabled(True)

        self._process_next_stack()

    def stop_processing(self) -> None:
        if self._worker is None or not self._worker.isRunning():
            return

        self._worker.stop()
        self._worker.wait()

        self.progress_bar.setVisible(False)
        self.stop_btn.setEnabled(False)
        self.status_label.setText("Processing stopped")

    def _process_next_stack(self) -> None:
        if self._current_stack_index >= len(self._stack_items):
            self._on_all_stacks_finished()
            return

        overall_progress = (self._current_stack_index * 100) // len(self._stack_items)
        self.progress_bar.setValue(overall_progress)

        stack_base_name, image_paths = self._stack_items[self._current_stack_index]

        self._worker = FocusStackingWorker(
            stacker_settings=self._app_settings.stacker.validated(),
            image_paths=image_paths,
            color_space=self.color_combo.currentText(),
        )
        self._worker.finished.connect(self._on_one_stack_finished)
        self._worker.error.connect(self._on_processing_error)
        self._worker.start()

        self.status_label.setText(
            f"Processing stack {self._current_stack_index + 1}/{len(self._stack_items)} ({len(image_paths)} images): {stack_base_name}"
        )

    def _on_one_stack_finished(self, result) -> None:
        original_stack_base_name, _ = self._stack_items[self._current_stack_index]

        user_prefix = self.output_name_edit.text().strip()
        output_format = self.format_combo.currentText().upper()
        ext = f".{output_format.lower()}"

        match = re.search(r"(\d+)$", original_stack_base_name)
        stack_number_str = match.group(1) if match else None

        if user_prefix:
            if stack_number_str:
                filename = f"{user_prefix}_{stack_number_str}{ext}"
            else:
                if len(self._stack_items) > 1:
                    filename = f"{user_prefix}_{self._current_stack_index + 1}of{len(self._stack_items)}{ext}"
                else:
                    filename = f"{user_prefix}{ext}"
        else:
            safe_base_name = re.sub(r"[\\/*?:\"<>| ]+", "_", original_stack_base_name)
            if not safe_base_name:
                safe_base_name = "stack"
            filename = f"{safe_base_name}{ext}"

        if not filename or filename == ext:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"stack_{timestamp}{ext}"

        output_dir = self._app_settings.output.validated().output_dir
        output_path = os.path.join(output_dir, filename)

        try:
            os.makedirs(output_dir, exist_ok=True)
            utils.save_image(
                result,
                output_path,
                format=output_format,
                color_space=self.color_combo.currentText(),
            )
            self._processed_stack_count += 1
            self.status_label.setText(
                f"Completed stack {self._current_stack_index + 1}/{len(self._stack_items)}: {original_stack_base_name}"
            )
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save image: {e}")

        self._current_stack_index += 1
        self._process_next_stack()

    def _on_all_stacks_finished(self) -> None:
        self.progress_bar.setVisible(False)
        self.stop_btn.setEnabled(False)
        self.status_label.setText(f"Processing complete - {self._processed_stack_count} stacks processed.")

    def _on_processing_error(self, error_msg: str) -> None:
        QMessageBox.critical(self, "Error", f"Processing failed: {error_msg}")
        self.progress_bar.setVisible(False)
        self.stop_btn.setEnabled(False)
        self.status_label.setText("Processing failed")

    # ----------------------------
    # Qt overrides
    # ----------------------------

    def closeEvent(self, event) -> None:  # noqa: N802
        self._save_settings_best_effort()
        super().closeEvent(event)


def _stack_mode_to_index(mode: str) -> int:
    mapping = {
        "auto": 0,
        "legacy": 1,
        "common_suffix": 2,
        "fixed_size": 3,
        "regex": 4,
    }
    return mapping.get(mode, 0)


def _index_to_stack_mode(index: int) -> str:
    mapping = {
        0: "auto",
        1: "legacy",
        2: "common_suffix",
        3: "fixed_size",
        4: "regex",
    }
    return mapping.get(index, "auto")


def _preset_to_settings(preset_name: str) -> Optional[StackerSettings]:
    preset_name = preset_name.strip()
    if preset_name == "Fast Preview":
        return StackerSettings(
            num_pyramid_levels=1,
            gradient_threshold=10,
            focus_window_size=7,
            sharpen_strength=0.0,
            blend_method="direct_map",
        ).validated()
    if preset_name == "Balanced":
        return StackerSettings(
            num_pyramid_levels=3,
            gradient_threshold=10,
            focus_window_size=7,
            sharpen_strength=0.0,
            blend_method="weighted",
        ).validated()
    if preset_name == "Best Quality":
        return StackerSettings(
            num_pyramid_levels=5,
            gradient_threshold=8,
            focus_window_size=5,
            sharpen_strength=0.0,
            blend_method="weighted",
        ).validated()
    return None


def _settings_to_preset(settings: StackerSettings) -> str:
    settings = settings.validated()

    fast = _preset_to_settings("Fast Preview")
    balanced = _preset_to_settings("Balanced")
    quality = _preset_to_settings("Best Quality")

    if fast is not None and settings == fast:
        return "Fast Preview"
    if balanced is not None and settings == balanced:
        return "Balanced"
    if quality is not None and settings == quality:
        return "Best Quality"
    return "Custom"
