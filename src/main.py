#!/usr/bin/env python3

import os
import sys
import re
import os
# Removed duplicate sys import
from datetime import datetime
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QPushButton, QLabel, QFileDialog, 
                            QComboBox, QProgressBar, QMessageBox, QGroupBox,
                            QGridLayout, QLineEdit) # Added QLineEdit
from PyQt5.QtCore import QThread, pyqtSignal
from focus_stacker import FocusStacker

class FocusStackingThread(QThread):
    """
    @class FocusStackingThread
    @brief Thread for processing focus stacking to keep UI responsive
    """
    finished = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, stacker, image_paths, color_space):
        """
        @param stacker FocusStacker instance
        @param image_paths List of paths to images
        @param color_space Selected color space
        """
        super().__init__()
        self.stacker = stacker
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
        
        # Default stacking parameters
        self.radius = 2
        self.smoothing = 1
        
        # Create stacker instance
        self.stacker = FocusStacker(
            radius=self.radius,
            smoothing=self.smoothing
        )
        
        self.init_ui()

    def _create_parameter_group(self):
        """Creates the QGroupBox for stacking parameters."""
        params_group = QGroupBox("Stacking Parameters")
        params_layout = QGridLayout()

        # Radius control
        radius_label = QLabel('Radius:')
        self.radius_combo = QComboBox()
        self.radius_combo.addItems([str(i) for i in range(1, 21)])
        self.radius_combo.setCurrentText(str(self.radius))
        self.radius_combo.currentTextChanged.connect(self.update_stacker)
        radius_desc = QLabel("Use 2 for maximum sharpness in photogrammetry, 3+ only if noise becomes problematic")
        radius_desc.setWordWrap(True)

        # Smoothing control
        smoothing_label = QLabel('Smoothing:')
        self.smoothing_combo = QComboBox()
        self.smoothing_combo.addItems([str(i) for i in range(1, 11)])
        self.smoothing_combo.setCurrentText(str(self.smoothing))
        self.smoothing_combo.currentTextChanged.connect(self.update_stacker)
        smoothing_desc = QLabel("Keep at 1 for photogrammetry to preserve maximum detail, increase only if artifacts appear")
        smoothing_desc.setWordWrap(True)

        # Add parameter controls to grid layout
        row = 0
        params_layout.addWidget(radius_label, row, 0)
        params_layout.addWidget(self.radius_combo, row, 1)
        row += 1
        params_layout.addWidget(radius_desc, row, 0, 1, 2)
        row += 1
        params_layout.addWidget(smoothing_label, row, 0)
        params_layout.addWidget(self.smoothing_combo, row, 1)
        row += 1
        params_layout.addWidget(smoothing_desc, row, 0, 1, 2)

        params_group.setLayout(params_layout)
        return params_group

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

    def update_stacker(self):
        """Update stacker with current parameter values"""
        self.radius = int(self.radius_combo.currentText())
        self.smoothing = int(self.smoothing_combo.currentText())
        
        # Update stacker instance with new parameters
        self.stacker = FocusStacker(
            radius=self.radius,
            smoothing=self.smoothing
        )

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
        
        if not stacks:
            print("Warning: No numbered sequences found in filenames.")
            return len(image_paths) # Treat all images as one stack if no sequences detected
            
        # Sort numbers within each detected stack
        for numbers in stacks.values():
            numbers.sort()
            
        # Verify if sequences are continuous (optional check)
        for base_name, numbers in stacks.items():
            if numbers != list(range(min(numbers), max(numbers) + 1)):
                print(f"Warning: Non-continuous sequence for {base_name}: {numbers}")
                
        # Determine stack size based on the most common sequence length
        sizes = [len(numbers) for numbers in stacks.values()]
        if sizes:
            detected_size = max(set(sizes), key=sizes.count)
            print(f"Detected stack size: {detected_size}")
            return detected_size
            
        return len(image_paths) # Fallback if size detection fails

    def load_images(self):
        """Open file dialog to select images"""
        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.ExistingFiles)
        file_dialog.setNameFilter("Images (*.png *.jpg *.jpeg *.tif *.tiff)")
        
        if file_dialog.exec_():
            self.image_paths = sorted(file_dialog.selectedFiles()) # Sort for consistent processing order
            print("\nLoaded images:", self.image_paths)
            
            stack_size = self.detect_stack_size(self.image_paths)
            self.stacks = self.stacker.split_into_stacks(self.image_paths, stack_size)
            
            print(f"\nSplit into {len(self.stacks)} stacks of size {stack_size}")
            for i, stack in enumerate(self.stacks):
                print(f"Stack {i+1}:", stack)
                
            self.status_label.setText(f'Loaded {len(self.image_paths)} images in {len(self.stacks)} stacks')

    def process_stack(self):
        """Start the focus stacking process"""
        if not hasattr(self, 'stacks'):
            QMessageBox.warning(self, 'Error', 'Please load images first')
            return

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
        print(f"Stack contains {len(current_image_stack)} images")
        print("Stack images:", current_image_stack)
        
        # Create and start processing thread for the current stack
        self.thread = FocusStackingThread(
            self.stacker,
            current_image_stack,
            self.color_combo.currentText()
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
        ext = '.jpg' # Hardcoded for now based on format combo
        
        if base_name:
            # Use user-provided base name
            filename = f'{base_name}_{self.current_stack + 1}of{len(self.stacks)}{ext}'
        else:
            # Fallback to default timestamp-based name
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'stack_{self.current_stack + 1}of{len(self.stacks)}_{timestamp}{ext}'
            
        output_dir = 'output' # Hardcoded output directory for now
        output_path = os.path.join(output_dir, filename)
        
        try:
            # Ensure output directory exists
            os.makedirs(output_dir, exist_ok=True)
            print(f"Saving to: {output_path}")
            
            # Save the resulting image
            self.stacker.save_image(
                result,
                output_path,
                'JPEG', # Hardcoded for now
                self.color_combo.currentText()
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
