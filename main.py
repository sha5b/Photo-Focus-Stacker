#!/usr/bin/env python3

import os
import sys
import re
import os
from datetime import datetime
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                            QHBoxLayout, QPushButton, QLabel, QFileDialog,
                            QComboBox, QProgressBar, QMessageBox, QGroupBox,
                            QGridLayout, QLineEdit, QCheckBox, QSpinBox,
                            QDoubleSpinBox) # Keep needed imports
from PyQt5.QtCore import QThread, pyqtSignal

# Import from the new structure
from src.core.focus_stacker import FocusStacker, StackingCancelledException # Import exception
from src.core import utils # Import utils for saving/splitting

class FocusStackingThread(QThread):
    """
    @class FocusStackingThread
    @brief Thread for processing focus stacking to keep UI responsive
    """
    finished = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, stacker_config, image_paths, color_space):
        """
        @param stacker_config Dictionary with parameters for FocusStacker initialization
        @param image_paths List of paths to images
        @param color_space Selected color space
        """
        super().__init__()
        # Create a new stacker instance within the thread using the config
        self.stacker = FocusStacker(**stacker_config)
        self.image_paths = image_paths
        self.color_space = color_space
        self.stopped = False

    def run(self):
        """Process the focus stacking operation"""
        try:
            result = self.stacker.process_stack(
                self.image_paths,
                self.color_space
            )
            if not self.stopped:
                self.finished.emit(result)
        except StackingCancelledException:
            print("Stacking thread caught cancellation.")
            pass
        except Exception as e:
            if not self.stopped:
                self.error.emit(str(e))

    def stop(self):
        """Signal the thread and the FocusStacker instance to stop processing"""
        self.stopped = True
        if hasattr(self, 'stacker') and self.stacker:
            self.stacker.request_stop()

class MainWindow(QMainWindow):
    """
    @class MainWindow
    @brief Main application window for the focus stacking tool
    """
    def __init__(self):
        super().__init__()
        self.image_paths = []
        self.stack_items = [] # Store (base_name, paths) tuples

        # Default Stacker Configuration (Simplified)
        self.stacker_config = {
            'focus_window_size': 7,
            'sharpen_strength': 0.0,
            'num_pyramid_levels': 3, # Default pyramid levels
            'gradient_threshold': 10 # Default gradient threshold for ECC mask
        }

        self.init_ui()

    def _create_parameter_group(self):
        """Creates the QGroupBox for stacking parameters."""
        params_group = QGroupBox("Stacking Parameters")
        params_layout = QGridLayout()
        params_layout.setVerticalSpacing(10)
        params_layout.setHorizontalSpacing(15)
        params_layout.setColumnStretch(0, 1)
        params_layout.setColumnStretch(1, 3)
        row = 0

        # --- Pyramid Levels (for ECC Alignment) ---
        self.pyramid_label = QLabel('Alignment Pyramid Levels:')
        self.pyramid_spinbox = QSpinBox()
        self.pyramid_spinbox.setRange(1, 6) # 1 = no pyramid, reasonable upper limit
        self.pyramid_spinbox.setValue(self.stacker_config.get('num_pyramid_levels', 3))
        self.pyramid_spinbox.valueChanged.connect(self.update_stacker_config)
        params_layout.addWidget(self.pyramid_label, row, 0)
        params_layout.addWidget(self.pyramid_spinbox, row, 1)
        row += 1

        # --- Gradient Threshold (for ECC Alignment Mask) ---
        self.gradient_label = QLabel('Alignment Mask Threshold:')
        self.gradient_spinbox = QSpinBox()
        self.gradient_spinbox.setRange(1, 100) # Range for gradient threshold
        self.gradient_spinbox.setValue(self.stacker_config.get('gradient_threshold', 10))
        self.gradient_spinbox.valueChanged.connect(self.update_stacker_config)
        params_layout.addWidget(self.gradient_label, row, 0)
        params_layout.addWidget(self.gradient_spinbox, row, 1)
        row += 1

        # --- Focus Window Size ---
        self.focus_window_label = QLabel('Focus Window Size:')
        self.focus_window_spinbox = QSpinBox()
        self.focus_window_spinbox.setRange(3, 21)
        self.focus_window_spinbox.setSingleStep(2)
        self.focus_window_spinbox.setValue(self.stacker_config.get('focus_window_size', 7))
        self.focus_window_spinbox.valueChanged.connect(self.update_stacker_config)
        params_layout.addWidget(self.focus_window_label, row, 0)
        params_layout.addWidget(self.focus_window_spinbox, row, 1)
        row += 1

        # --- Sharpening Strength ---
        self.sharpen_label = QLabel('Sharpening Strength:')
        self.sharpen_spinbox = QDoubleSpinBox()
        self.sharpen_spinbox.setRange(0.0, 3.0)
        self.sharpen_spinbox.setSingleStep(0.1)
        self.sharpen_spinbox.setDecimals(2)
        self.sharpen_spinbox.setValue(self.stacker_config.get('sharpen_strength', 0.0))
        self.sharpen_spinbox.valueChanged.connect(self.update_stacker_config)
        params_layout.addWidget(self.sharpen_label, row, 0)
        params_layout.addWidget(self.sharpen_spinbox, row, 1)
        row += 1

        # Add stretch to push controls to the top
        params_layout.setRowStretch(row, 1)

        params_group.setLayout(params_layout)
        return params_group

    # Removed toggle visibility functions

    def _create_output_group(self):
        """Creates the QGroupBox for output settings."""
        output_group = QGroupBox("Output Settings")
        output_layout = QGridLayout()

        # Output Base Name
        name_label = QLabel('Output Base Name:')
        self.output_name_edit = QLineEdit()
        self.output_name_edit.setPlaceholderText("Default: [Original Stack Name]") # Updated placeholder

        # Format
        format_label = QLabel('Format:')
        self.format_combo = QComboBox()
        self.format_combo.addItems(['JPEG', 'PNG', 'TIFF']) # Added more formats
        self.format_combo.setCurrentText('JPEG') # Keep JPEG default

        # Color Space
        color_label = QLabel('Color Space:')
        self.color_combo = QComboBox()
        self.color_combo.addItems(['sRGB']) # Keep sRGB for now

        # Add widgets to layout
        output_layout.addWidget(name_label, 0, 0)
        output_layout.addWidget(self.output_name_edit, 0, 1)
        output_layout.addWidget(format_label, 1, 0)
        output_layout.addWidget(self.format_combo, 1, 1)
        output_layout.addWidget(color_label, 2, 0)
        output_layout.addWidget(self.color_combo, 2, 1)

        output_group.setLayout(output_layout)
        return output_group

    def _create_action_buttons(self):
        """Creates the main action buttons and their layout."""
        process_btn = QPushButton('Process Stack')
        process_btn.clicked.connect(self.process_stack)

        self.stop_btn = QPushButton('Stop Processing')
        self.stop_btn.clicked.connect(self.stop_processing)
        self.stop_btn.setEnabled(False)

        button_layout = QHBoxLayout()
        button_layout.addWidget(process_btn)
        button_layout.addWidget(self.stop_btn)
        return button_layout

    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle('Focus Stacking Tool')
        # Adjust minimum size if needed, but keep it reasonable
        self.setMinimumSize(500, 400) # Reduced size as fewer options

        # Central Widget and Main Layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Create UI Elements using Helper Methods
        load_btn = QPushButton('Load Images')
        load_btn.clicked.connect(self.load_images)

        params_group = self._create_parameter_group()
        output_group = self._create_output_group()
        action_button_layout = self._create_action_buttons()

        # Status Label and Progress Bar
        self.status_label = QLabel('Ready')
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)

        # --- Assemble Main Layout ---
        layout.addWidget(load_btn)
        layout.addWidget(params_group)
        layout.addWidget(output_group)
        layout.addLayout(action_button_layout)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.status_label)

    def update_stacker_config(self):
        """Update the stacker configuration dictionary from UI controls."""
        # Only update the parameters that still exist in the UI
        self.stacker_config['num_pyramid_levels'] = self.pyramid_spinbox.value()
        self.stacker_config['gradient_threshold'] = self.gradient_spinbox.value() # Add gradient threshold
        self.stacker_config['focus_window_size'] = self.focus_window_spinbox.value()
        self.stacker_config['sharpen_strength'] = self.sharpen_spinbox.value()

        print("\nStacker configuration updated:")
        for key, value in self.stacker_config.items():
            print(f"  {key}: {value}")

    def detect_stack_size(self, image_paths):
        # This function is no longer needed as logic is in utils.split_into_stacks
        pass

    def load_images(self):
        """Open file dialog to select images and use utils.split_into_stacks"""
        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.ExistingFiles)
        file_dialog.setNameFilter("Images (*.png *.jpg *.jpeg *.tif *.tiff *.bmp *.webp)")

        if file_dialog.exec_():
            self.image_paths = file_dialog.selectedFiles()
            if not self.image_paths:
                self.status_label.setText("No images selected.")
                return

            print(f"\nSelected {len(self.image_paths)} image files.")

            try:
                self.stack_items = utils.split_into_stacks(self.image_paths, stack_size=0)
            except ImportError:
                 QMessageBox.warning(self, 'Error', 'Failed to import natsort for natural sorting. Stacks might be ordered incorrectly.')
                 self.stack_items = [] # Fallback
            except Exception as e:
                 QMessageBox.critical(self, 'Error', f'Failed to split images into stacks: {e}')
                 self.stack_items = []
                 return

            if not self.stack_items:
                 warning_msg = "Could not detect distinct stacks based on filenames. Treating all images as one stack."
                 print(f"Warning: {warning_msg}")
                 self.stack_items = [("stack", sorted(self.image_paths))]

            num_images_in_stacks = sum(len(item[1]) for item in self.stack_items)
            self.status_label.setText(f'Loaded {num_images_in_stacks} images in {len(self.stack_items)} stacks.')
            print(f"Successfully split into {len(self.stack_items)} stacks.")

    def process_stack(self):
        """Start the focus stacking process"""
        if not self.image_paths or not self.stack_items:
            QMessageBox.warning(self, 'Error', 'Please load images first (ensure stacks were detected).')
            return

        self.update_stacker_config() # Ensure config is up-to-date

        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.current_stack = 0
        self.processed_stack_count = 0
        self.stop_btn.setEnabled(True)

        self._process_next_stack()

    def stop_processing(self):
        """Stop the current processing operation"""
        if hasattr(self, 'thread') and self.thread.isRunning():
            print("\nStopping processing...")
            self.thread.stop()
            self.thread.wait()

            self.progress_bar.setVisible(False)
            self.stop_btn.setEnabled(False)
            self.status_label.setText('Processing stopped')
            print("Processing stopped")

    def _process_next_stack(self):
        """Process the next stack in the queue"""
        if self.current_stack >= len(self.stack_items):
            print("\nAll stacks processed!")
            self.processing_all_finished()
            return

        overall_progress = (self.current_stack * 100) // len(self.stack_items)
        self.progress_bar.setValue(overall_progress)

        current_stack_base_name, current_image_paths = self.stack_items[self.current_stack]

        print(f"\n=== Processing stack {self.current_stack + 1}/{len(self.stack_items)} ('{current_stack_base_name}') ===")
        print(f"Stack contains {len(current_image_paths)} images.")

        self.thread = FocusStackingThread(
            self.stacker_config.copy(),
            current_image_paths,
            self.color_combo.currentText()
        )
        self.thread.finished.connect(self.processing_one_finished)
        self.thread.error.connect(self.processing_error)
        self.thread.start()

    def processing_one_finished(self, result):
        """Handle completion of one stack"""
        original_stack_base_name, _ = self.stack_items[self.current_stack]

        print(f"\n=== Saving result for stack '{original_stack_base_name}' ({self.current_stack + 1}/{len(self.stack_items)}) ===")

        user_prefix = self.output_name_edit.text().strip()
        output_format = self.format_combo.currentText().upper()
        ext = f'.{output_format.lower()}'

        match = re.search(r'(\d+)$', original_stack_base_name)
        stack_number_str = match.group(1) if match else None

        if user_prefix:
            if stack_number_str:
                filename = f'{user_prefix}_{stack_number_str}{ext}'
            else:
                print(f"  Warning: Could not extract trailing number from original stack name '{original_stack_base_name}'. Using sequential numbering.")
                if len(self.stack_items) > 1:
                    filename = f'{user_prefix}_{self.current_stack + 1}of{len(self.stack_items)}{ext}'
                else:
                    filename = f'{user_prefix}{ext}'
        else:
            safe_base_name = re.sub(r'[\\/*?:"<>| ]+', '_', original_stack_base_name)
            if not safe_base_name: safe_base_name = "stack"
            filename = f'{safe_base_name}{ext}'

        if not filename or filename == ext:
             timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
             print(f"  Warning: Filename generation resulted in empty name. Using timestamp fallback.")
             filename = f'stack_{timestamp}{ext}'

        output_dir = 'results'
        output_path = os.path.join(output_dir, filename)

        try:
            os.makedirs(output_dir, exist_ok=True)
            print(f"Saving to: {output_path}")
            utils.save_image(
                result,
                output_path,
                format=output_format,
                color_space=self.color_combo.currentText()
            )
            print(f"Successfully saved stack result.")
            self.processed_stack_count += 1
            status_text = f"Completed stack '{original_stack_base_name}' ({self.current_stack + 1} of {len(self.stack_items)})"
            print(status_text)
            self.status_label.setText(status_text)
        except Exception as e:
            error_msg = f'Failed to save image: {str(e)}'
            print(f"ERROR: {error_msg}")
            QMessageBox.critical(self, 'Error', error_msg)

        self.current_stack += 1
        self._process_next_stack()

    def processing_all_finished(self):
        """Handle completion of all stacks"""
        print("\n=== All Processing Complete ===")
        print(f"Total stacks processed: {self.processed_stack_count}")
        self.progress_bar.setVisible(False)
        self.stop_btn.setEnabled(False)
        status_text = f'Processing complete - {self.processed_stack_count} stacks processed.'
        print(status_text)
        self.status_label.setText(status_text)

    def processing_error(self, error_msg):
        """Handle processing errors"""
        QMessageBox.critical(self, 'Error', f'Processing failed: {error_msg}')
        self.progress_bar.setVisible(False)
        self.stop_btn.setEnabled(False)
        self.status_label.setText('Processing failed')

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
