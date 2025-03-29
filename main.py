#!/usr/bin/env python3

import os
import sys
import re
import os
from datetime import datetime
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                            QHBoxLayout, QPushButton, QLabel, QFileDialog,
                            QComboBox, QProgressBar, QMessageBox, QGroupBox,
                            QGridLayout, QLineEdit, QCheckBox, QSpinBox) # Added QCheckBox, QSpinBox
from PyQt5.QtCore import QThread, pyqtSignal

# Import from the new structure
from src.core.focus_stacker import FocusStacker
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
        # This ensures the thread uses the settings from when it was started
        self.stacker = FocusStacker(**stacker_config)
        self.image_paths = image_paths
        self.color_space = color_space
        self.stopped = False

    def run(self):
        """Process the focus stacking operation"""
        try:
            # process_stack now uses the stacker instance created in __init__
            result = self.stacker.process_stack(
                self.image_paths,
                self.color_space
            )
            if not self.stopped:
                self.finished.emit(result)
        except Exception as e:
            if not self.stopped:
                self.error.emit(str(e))

    def stop(self):
        """Signal the thread to stop processing"""
        self.stopped = True

class MainWindow(QMainWindow):
    """
    @class MainWindow
    @brief Main application window for the focus stacking tool
    """
    def __init__(self):
        super().__init__()
        self.image_paths = []
        self.stacks = [] # Initialize stacks list

        # Default Stacker Configuration
        self.stacker_config = {
            'align_method': 'orb',
            'focus_measure_method': 'custom',
            'blend_method': 'weighted',
            'consistency_filter': False,
            'consistency_kernel': 5,
            'postprocess': True,
            'laplacian_levels': 5
        }

        # Create initial stacker instance (will be updated by UI)
        # self.stacker = FocusStacker(**self.stacker_config) # No longer needed here

        self.init_ui()

    def _create_parameter_group(self):
        """Creates the QGroupBox for stacking parameters."""
        params_group = QGroupBox("Stacking Parameters")
        params_layout = QGridLayout()
        row = 0

        # --- Alignment Method ---
        align_label = QLabel('Alignment:')
        self.align_combo = QComboBox()
        self.align_combo.addItems(['orb', 'ecc']) # Add more as implemented
        self.align_combo.setCurrentText(self.stacker_config['align_method'])
        self.align_combo.currentTextChanged.connect(self.update_stacker_config)
        params_layout.addWidget(align_label, row, 0)
        params_layout.addWidget(self.align_combo, row, 1)
        row += 1

        # --- Focus Measure Method ---
        focus_label = QLabel('Focus Measure:')
        self.focus_combo = QComboBox()
        self.focus_combo.addItems(['custom', 'laplacian_variance']) # Add more as implemented
        self.focus_combo.setCurrentText(self.stacker_config['focus_measure_method'])
        self.focus_combo.currentTextChanged.connect(self.update_stacker_config)
        params_layout.addWidget(focus_label, row, 0)
        params_layout.addWidget(self.focus_combo, row, 1)
        row += 1

        # --- Blending Method ---
        blend_label = QLabel('Blending:')
        self.blend_combo = QComboBox()
        self.blend_combo.addItems(['weighted', 'laplacian']) # Add more as implemented
        self.blend_combo.setCurrentText(self.stacker_config['blend_method'])
        self.blend_combo.currentTextChanged.connect(self.update_stacker_config)
        params_layout.addWidget(blend_label, row, 0)
        params_layout.addWidget(self.blend_combo, row, 1)
        row += 1

        # --- Laplacian Levels (Conditional) ---
        self.laplacian_label = QLabel('Laplacian Levels:')
        self.laplacian_spinbox = QSpinBox()
        self.laplacian_spinbox.setRange(2, 10)
        self.laplacian_spinbox.setValue(self.stacker_config['laplacian_levels'])
        self.laplacian_spinbox.valueChanged.connect(self.update_stacker_config)
        params_layout.addWidget(self.laplacian_label, row, 0)
        params_layout.addWidget(self.laplacian_spinbox, row, 1)
        # Show/hide based on blend method selection
        self.laplacian_label.setVisible(self.stacker_config['blend_method'] == 'laplacian')
        self.laplacian_spinbox.setVisible(self.stacker_config['blend_method'] == 'laplacian')
        self.blend_combo.currentTextChanged.connect(lambda text: self.laplacian_label.setVisible(text == 'laplacian'))
        self.blend_combo.currentTextChanged.connect(lambda text: self.laplacian_spinbox.setVisible(text == 'laplacian'))
        row += 1

        # --- Consistency Filter (Conditional) ---
        self.consistency_checkbox = QCheckBox('Apply Consistency Filter')
        self.consistency_checkbox.setChecked(self.stacker_config['consistency_filter'])
        self.consistency_checkbox.stateChanged.connect(self.update_stacker_config)
        self.consistency_kernel_label = QLabel('Filter Kernel Size:')
        self.consistency_kernel_spinbox = QSpinBox()
        self.consistency_kernel_spinbox.setRange(3, 21) # Odd numbers usually
        self.consistency_kernel_spinbox.setSingleStep(2)
        self.consistency_kernel_spinbox.setValue(self.stacker_config['consistency_kernel'])
        self.consistency_kernel_spinbox.valueChanged.connect(self.update_stacker_config)
        params_layout.addWidget(self.consistency_checkbox, row, 0, 1, 2) # Span checkbox
        row += 1
        params_layout.addWidget(self.consistency_kernel_label, row, 0)
        params_layout.addWidget(self.consistency_kernel_spinbox, row, 1)
        # Show/hide kernel size based on checkbox and blend method
        show_consistency_kernel = (self.stacker_config['consistency_filter'] and self.stacker_config['blend_method'] == 'laplacian')
        self.consistency_kernel_label.setVisible(show_consistency_kernel)
        self.consistency_kernel_spinbox.setVisible(show_consistency_kernel)
        self.consistency_checkbox.stateChanged.connect(self.toggle_consistency_kernel_visibility)
        self.blend_combo.currentTextChanged.connect(self.toggle_consistency_kernel_visibility)
        row += 1

        # --- Post-processing ---
        self.postprocess_checkbox = QCheckBox('Apply Post-processing (Contrast/Sharpen)')
        self.postprocess_checkbox.setChecked(self.stacker_config['postprocess'])
        self.postprocess_checkbox.stateChanged.connect(self.update_stacker_config)
        params_layout.addWidget(self.postprocess_checkbox, row, 0, 1, 2) # Span checkbox
        row += 1


        params_group.setLayout(params_layout)
        return params_group

    def toggle_consistency_kernel_visibility(self):
        """Shows/hides the consistency kernel size controls based on checkbox and blend method."""
        show = (self.consistency_checkbox.isChecked() and self.blend_combo.currentText() == 'laplacian')
        self.consistency_kernel_label.setVisible(show)
        self.consistency_kernel_spinbox.setVisible(show)

    def _create_output_group(self):
        """Creates the QGroupBox for output settings."""
        output_group = QGroupBox("Output Settings")
        output_layout = QGridLayout()

        # Output Base Name
        name_label = QLabel('Output Base Name:')
        self.output_name_edit = QLineEdit()
        self.output_name_edit.setPlaceholderText("Default: stack_NofM_timestamp")

        # Format
        format_label = QLabel('Format:')
        self.format_combo = QComboBox()
        self.format_combo.addItems(['JPEG']) # Only JPEG supported for now

        # Color Space
        color_label = QLabel('Color Space:')
        self.color_combo = QComboBox()
        self.color_combo.addItems(['sRGB'])

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
        self.setMinimumSize(600, 400)

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
        self.stacker_config['align_method'] = self.align_combo.currentText()
        self.stacker_config['focus_measure_method'] = self.focus_combo.currentText()
        self.stacker_config['blend_method'] = self.blend_combo.currentText()
        self.stacker_config['laplacian_levels'] = self.laplacian_spinbox.value()
        self.stacker_config['consistency_filter'] = self.consistency_checkbox.isChecked()
        self.stacker_config['consistency_kernel'] = self.consistency_kernel_spinbox.value()
        self.stacker_config['postprocess'] = self.postprocess_checkbox.isChecked()

        # Optional: Could re-instantiate self.stacker here if needed immediately,
        # but it's safer to create it fresh in the processing thread.
        print("\nStacker configuration updated:")
        for key, value in self.stacker_config.items():
            print(f"  {key}: {value}")

    def detect_stack_size(self, image_paths):
        """
        Detect stack size by finding sequences in filenames
        @param image_paths List of image paths
        @return Number of images per stack
        """
        # Group files by base name (part before the last number sequence)
        stacks = {}
        for path in image_paths:
            filename = os.path.splitext(os.path.basename(path))[0]
            match = re.search(r'^(.+?)(\d+)$', filename) # Find base name and trailing number
            if match:
                base_name = match.group(1)
                number = int(match.group(2))
                if base_name not in stacks:
                    stacks[base_name] = []
                stacks[base_name].append(number)
        # Use the refactored utility function for splitting
        # This function needs to be called *after* loading images
        # Let's adjust the load_images function instead.
        pass # Remove old logic

    def load_images(self):
        """Open file dialog to select images and use utils.split_into_stacks"""
        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.ExistingFiles)
        # Allow various image types
        file_dialog.setNameFilter("Images (*.png *.jpg *.jpeg *.tif *.tiff *.bmp *.webp)")

        if file_dialog.exec_():
            self.image_paths = file_dialog.selectedFiles() # Keep unsorted list from dialog
            if not self.image_paths:
                self.status_label.setText("No images selected.")
                return

            print(f"\nSelected {len(self.image_paths)} image files.")

            # Use the utility function to split into stacks
            # Pass 0 for stack_size to auto-detect based on names
            try:
                # We need an instance of FocusStacker to call split_into_stacks
                # Or make split_into_stacks a static method or move it entirely to utils
                # Let's assume it's moved entirely to utils (which we did)
                self.stacks = utils.split_into_stacks(self.image_paths, stack_size=0)
            except ImportError:
                 QMessageBox.warning(self, 'Error', 'Failed to import natsort for natural sorting. Stacks might be ordered incorrectly.')
                 # Fallback or handle error appropriately
                 self.stacks = [] # Or basic splitting logic here
            except Exception as e:
                 QMessageBox.critical(self, 'Error', f'Failed to split images into stacks: {e}')
                 self.stacks = []
                 return


            if not self.stacks:
                 warning_msg = "Could not detect distinct stacks based on filenames. Treating all images as one stack."
                 print(f"Warning: {warning_msg}")
                 # QMessageBox.information(self, 'Stack Detection', warning_msg) # Can be annoying
                 # Ensure the fallback stack is also sorted if needed, though split_into_stacks should handle sorting
                 self.stacks = [sorted(self.image_paths)] # Treat all as one stack, ensure sorted

            num_images_in_stacks = sum(len(s) for s in self.stacks)
            self.status_label.setText(f'Loaded {num_images_in_stacks} images in {len(self.stacks)} stacks.')
            print(f"Successfully split into {len(self.stacks)} stacks.")

    def process_stack(self):
        """Start the focus stacking process"""
        if not self.image_paths or not self.stacks: # Check self.stacks as well
            QMessageBox.warning(self, 'Error', 'Please load images first (ensure stacks were detected).')
            return

        # Ensure config is up-to-date before starting
        self.update_stacker_config()

        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.current_stack = 0
        self.processed_stack_count = 0
        self.stop_btn.setEnabled(True)

        self._process_next_stack() # Start processing the first stack

    def stop_processing(self):
        """Stop the current processing operation"""
        if hasattr(self, 'thread') and self.thread.isRunning():
            print("\nStopping processing...")
            self.thread.stop() # Signal thread to stop
            self.thread.wait() # Wait for thread to finish cleanly
            
            # Reset UI state
            self.progress_bar.setVisible(False)
            self.stop_btn.setEnabled(False)
            self.status_label.setText('Processing stopped')
            print("Processing stopped")

    def _process_next_stack(self):
        """Process the next stack in the queue"""
        if self.current_stack >= len(self.stacks):
            print("\nAll stacks processed!")
            self.processing_all_finished()
            return
            
        # Update overall progress bar
        overall_progress = (self.current_stack * 100) // len(self.stacks)
        self.progress_bar.setValue(overall_progress)
        
        current_image_stack = self.stacks[self.current_stack]
        print(f"\n=== Processing stack {self.current_stack + 1}/{len(self.stacks)} ===")
        print(f"Stack contains {len(current_image_stack)} images.")
        # print("Stack images:", current_image_stack) # Can be verbose

        # Create and start processing thread for the current stack
        # Pass the current stacker configuration dictionary
        self.thread = FocusStackingThread(
            self.stacker_config.copy(), # Pass a copy of the config
            current_image_stack,
            self.color_combo.currentText() # Still using sRGB only for now
        )
        self.thread.finished.connect(self.processing_one_finished)
        self.thread.error.connect(self.processing_error)
        self.thread.start()

    def processing_one_finished(self, result):
        """Handle completion of one stack
        @param result Processed image (NumPy array)
        """
        print(f"\n=== Saving result for stack {self.current_stack + 1} ===")
        
        # Determine filename based on user input or default
        base_name = self.output_name_edit.text().strip()
        # Get format from combo box (though only JPEG is supported currently)
        output_format = self.format_combo.currentText().upper()
        ext = f'.{output_format.lower()}'

        if base_name:
            # Use user-provided base name, add stack number if multiple stacks
            if len(self.stacks) > 1:
                 filename = f'{base_name}_{self.current_stack + 1}of{len(self.stacks)}{ext}'
            else:
                 filename = f'{base_name}{ext}'
        else:
            # Fallback to default timestamp-based name
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            if len(self.stacks) > 1:
                 filename = f'stack_{self.current_stack + 1}of{len(self.stacks)}_{timestamp}{ext}'
            else:
                 filename = f'stack_{timestamp}{ext}'

        # Use 'results' directory consistent with project structure
        output_dir = 'results'
        output_path = os.path.join(output_dir, filename)

        try:
            # Ensure output directory exists
            os.makedirs(output_dir, exist_ok=True)
            print(f"Saving to: {output_path}")

            # Save the resulting image using the utility function directly
            # Pass format and color space from UI
            utils.save_image(
                result,
                output_path,
                format=output_format,
                color_space=self.color_combo.currentText() # Assumes result is sRGB before saving
            )
            print(f"Successfully saved stack result.")
            self.processed_stack_count += 1
            status_text = f'Completed stack {self.current_stack + 1} of {len(self.stacks)}'
            print(status_text)
            self.status_label.setText(status_text)
        except Exception as e:
            error_msg = f'Failed to save image: {str(e)}'
            print(f"ERROR: {error_msg}")
            QMessageBox.critical(self, 'Error', error_msg)
            # Stop processing further stacks on save error? Or continue? Currently continues.
            
        # Process the next stack
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
        """Handle processing errors
        @param error_msg Error message to display
        """
        QMessageBox.critical(self, 'Error', f'Processing failed: {error_msg}')
        self.progress_bar.setVisible(False)
        self.stop_btn.setEnabled(False)
        self.status_label.setText('Processing failed')

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
