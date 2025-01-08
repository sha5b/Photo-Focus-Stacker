#!/usr/bin/env python3

import os
import sys
from datetime import datetime
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QPushButton, QLabel, QFileDialog, 
                            QComboBox, QProgressBar, QMessageBox, QGroupBox,
                            QGridLayout, QCheckBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from focus_stacker import FocusStacker

class FocusStackingThread(QThread):
    """
    @class FocusStackingThread
    @brief Thread for processing focus stacking to keep UI responsive
    """
    progress = pyqtSignal(int)
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
                self.color_space,
                progress_callback=self.progress.emit
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
        self.radius = 8    # Balanced radius for detail preservation
        self.smoothing = 4 # Moderate smoothing for natural transitions
        self.enhance_sharpness = True  # Enable sharpness enhancement by default
        
        # Default sharpening parameters - more conservative values
        self.snr_values = [100, 80, 60]  # Default to Low for more natural look
        self.focus_threshold = 0.3  # Default to Medium for selective enhancement
        
        # Create stacker with default parameters
        self.stacker = FocusStacker(
            radius=self.radius,
            smoothing=self.smoothing,
            enhance_sharpness=self.enhance_sharpness,
            sharpening_params={
                'snr_values': self.snr_values,
                'focus_threshold': self.focus_threshold
            }
        )
        
        self.init_ui()

    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle('Focus Stacking Tool')
        self.setMinimumSize(600, 400)

        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Create buttons
        load_btn = QPushButton('Load Images')
        load_btn.clicked.connect(self.load_images)
        
        process_btn = QPushButton('Process Stack')
        process_btn.clicked.connect(self.process_stack)
        
        stop_btn = QPushButton('Stop Processing')
        stop_btn.clicked.connect(self.stop_processing)
        stop_btn.setEnabled(False)
        self.stop_btn = stop_btn

        # Create parameter controls
        params_group = QGroupBox("Stacking Parameters")
        params_layout = QGridLayout()
        
        # Sharpness enhancement toggle
        enhance_label = QLabel('Enhance Sharpness:')
        self.enhance_check = QCheckBox()
        self.enhance_check.setChecked(self.enhance_sharpness)
        self.enhance_check.stateChanged.connect(self.update_sharpness_controls)
        
        enhance_desc = QLabel("Automatically enhance sharpness in poorly focused regions")
        enhance_desc.setWordWrap(True)
        
        # Sharpening parameters (initially hidden)
        self.sharpening_group = QGroupBox("Sharpening Parameters")
        sharpening_layout = QGridLayout()
        
        # SNR control
        snr_label = QLabel('Enhancement Strength:')
        self.snr_combo = QComboBox()
        self.snr_combo.addItems(['Low', 'Medium', 'High', 'Very High'])
        self.snr_combo.setCurrentText('Low')  # Default to Low for more natural look
        self.snr_combo.currentTextChanged.connect(self.update_stacker)
        snr_desc = QLabel("Higher values increase sharpening intensity")
        snr_desc.setWordWrap(True)
        
        # Focus threshold control
        threshold_label = QLabel('Region Coverage:')
        self.threshold_combo = QComboBox()
        self.threshold_combo.addItems(['Small', 'Medium', 'Large', 'Maximum'])
        self.threshold_combo.setCurrentText('Medium')  # Default to Medium
        self.threshold_combo.currentTextChanged.connect(self.update_stacker)
        threshold_desc = QLabel("Controls how much of the image gets sharpened")
        threshold_desc.setWordWrap(True)
        
        # Add sharpening controls to layout
        sharpening_layout.addWidget(snr_label, 0, 0)
        sharpening_layout.addWidget(self.snr_combo, 0, 1)
        sharpening_layout.addWidget(snr_desc, 1, 0, 1, 2)
        sharpening_layout.addWidget(threshold_label, 2, 0)
        sharpening_layout.addWidget(self.threshold_combo, 2, 1)
        sharpening_layout.addWidget(threshold_desc, 3, 0, 1, 2)
        
        self.sharpening_group.setLayout(sharpening_layout)
        self.sharpening_group.setVisible(False)
        
        # Radius control
        radius_label = QLabel('Radius:')
        self.radius_combo = QComboBox()
        self.radius_combo.addItems([str(i) for i in range(1, 21)])
        self.radius_combo.setCurrentText(str(self.radius))
        self.radius_combo.currentTextChanged.connect(self.update_stacker)
        radius_desc = QLabel("Lower values (2-4) maximize sharpness, higher values for smoother blending")
        radius_desc.setWordWrap(True)
        
        # Smoothing control
        smoothing_label = QLabel('Smoothing:')
        self.smoothing_combo = QComboBox()
        self.smoothing_combo.addItems([str(i) for i in range(1, 11)])
        self.smoothing_combo.setCurrentText(str(self.smoothing))
        self.smoothing_combo.currentTextChanged.connect(self.update_stacker)
        smoothing_desc = QLabel("Lower values (1-3) preserve edges, higher values smooth transitions")
        smoothing_desc.setWordWrap(True)
        
        # Add parameter controls to grid
        params_layout.addWidget(enhance_label, 0, 0)
        params_layout.addWidget(self.enhance_check, 0, 1)
        params_layout.addWidget(enhance_desc, 1, 0, 1, 2)
        params_layout.addWidget(self.sharpening_group, 2, 0, 1, 2)
        params_layout.addWidget(radius_label, 3, 0)
        params_layout.addWidget(self.radius_combo, 3, 1)
        params_layout.addWidget(radius_desc, 4, 0, 1, 2)
        params_layout.addWidget(smoothing_label, 5, 0)
        params_layout.addWidget(self.smoothing_combo, 5, 1)
        params_layout.addWidget(smoothing_desc, 6, 0, 1, 2)
        
        params_group.setLayout(params_layout)
        
        # Create output format controls
        output_group = QGroupBox("Output Settings")
        output_layout = QGridLayout()
        
        format_label = QLabel('Format:')
        self.format_combo = QComboBox()
        self.format_combo.addItems(['JPEG'])  # Temporarily remove PNG

        color_label = QLabel('Color Space:')
        self.color_combo = QComboBox()
        self.color_combo.addItems(['sRGB'])
        
        output_layout.addWidget(format_label, 0, 0)
        output_layout.addWidget(self.format_combo, 0, 1)
        output_layout.addWidget(color_label, 1, 0)
        output_layout.addWidget(self.color_combo, 1, 1)
        
        output_group.setLayout(output_layout)

        # Create status label and progress bar
        self.status_label = QLabel('Ready')
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)

        # Add all widgets to main layout
        layout.addWidget(load_btn)
        layout.addWidget(params_group)
        layout.addWidget(output_group)
        
        button_layout = QHBoxLayout()
        button_layout.addWidget(process_btn)
        button_layout.addWidget(stop_btn)
        layout.addLayout(button_layout)
        
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.status_label)

    def update_sharpness_controls(self):
        """Show/hide sharpening controls based on checkbox"""
        is_checked = self.enhance_check.isChecked()
        self.sharpening_group.setVisible(is_checked)
        self.update_stacker()

    def update_stacker(self):
        """Update stacker with current parameter values"""
        self.radius = int(self.radius_combo.currentText())
        self.smoothing = int(self.smoothing_combo.currentText())
        self.enhance_sharpness = self.enhance_check.isChecked()
        
        # Convert UI selections to numerical parameters
        snr_map = {
            'Low': [100, 80, 60],      # More conservative values
            'Medium': [80, 60, 40],     # Balanced enhancement
            'High': [60, 40, 20],       # Moderate aggressive
            'Very High': [40, 30, 20]   # Less aggressive than before
        }
        threshold_map = {
            'Small': 0.15,              # More selective
            'Medium': 0.3,              # Balanced coverage
            'Large': 0.45,              # Wider coverage
            'Maximum': 0.6              # Maximum coverage
        }
        
        self.snr_values = snr_map[self.snr_combo.currentText()]
        self.focus_threshold = threshold_map[self.threshold_combo.currentText()]
        
        self.stacker = FocusStacker(
            radius=self.radius,
            smoothing=self.smoothing,
            enhance_sharpness=self.enhance_sharpness,
            sharpening_params={
                'snr_values': self.snr_values,
                'focus_threshold': self.focus_threshold
            }
        )

    def detect_stack_size(self, image_paths):
        """
        Detect stack size by finding sequences in filenames
        @param image_paths List of image paths
        @return Number of images per stack
        """
        import re
        
        # Group files by their base name (everything before the last number)
        stacks = {}
        for path in image_paths:
            # Get filename without extension
            filename = os.path.splitext(os.path.basename(path))[0]
            # Find last number in filename
            match = re.search(r'^(.+?)(\d+)$', filename)
            if match:
                base_name = match.group(1)  # Everything before the last number
                number = int(match.group(2))  # The last number
                if base_name not in stacks:
                    stacks[base_name] = []
                stacks[base_name].append(number)
        
        if not stacks:
            print("Warning: No numbered sequences found in filenames")
            return len(image_paths)  # Treat all images as one stack
            
        # Sort numbers in each stack
        for numbers in stacks.values():
            numbers.sort()
            
        # Verify sequences are continuous
        for base_name, numbers in stacks.items():
            expected = list(range(min(numbers), max(numbers) + 1))
            if numbers != expected:
                print(f"Warning: Non-continuous sequence for {base_name}: {numbers}")
                
        # Find most common stack size
        sizes = [len(numbers) for numbers in stacks.values()]
        if sizes:
            size = max(set(sizes), key=sizes.count)  # Most common size
            print(f"Detected stack size: {size}")
            return size
            
        return len(image_paths)  # Default to all images as one stack

    def load_images(self):
        """Open file dialog to select images"""
        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.ExistingFiles)
        file_dialog.setNameFilter("Images (*.png *.jpg *.jpeg *.tif *.tiff)")
        
        if file_dialog.exec_():
            self.image_paths = sorted(file_dialog.selectedFiles())  # Sort for consistent order
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
        self.results = []
        self.stop_btn.setEnabled(True)
        
        # Process first stack
        self._process_next_stack()

    def stop_processing(self):
        """Stop the current processing operation"""
        if hasattr(self, 'thread') and self.thread.isRunning():
            print("\nStopping processing...")
            self.thread.stop()  # Signal thread to stop
            self.thread.wait()  # Wait for thread to finish
            
            # Reset UI
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
            
        # Calculate overall progress
        overall_progress = (self.current_stack * 100) // len(self.stacks)
        self.progress_bar.setValue(overall_progress)
        
        print(f"\n=== Processing stack {self.current_stack + 1}/{len(self.stacks)} ===")
        print(f"Stack contains {len(self.stacks[self.current_stack])} images")
        print("Stack images:", self.stacks[self.current_stack])
        
        # Create and start processing thread
        self.thread = FocusStackingThread(
            self.stacker,
            self.stacks[self.current_stack],
            self.color_combo.currentText()
        )
        self.thread.progress.connect(lambda p: self.update_stack_progress(p, overall_progress))
        self.thread.finished.connect(self.processing_one_finished)
        self.thread.error.connect(self.processing_error)
        self.thread.start()

    def update_stack_progress(self, stack_progress, overall_base):
        """Update progress bar with combined progress
        @param stack_progress Progress of current stack (0-100)
        @param overall_base Base progress from completed stacks
        """
        # Scale stack progress to portion of total progress
        stack_portion = stack_progress / len(self.stacks)
        total_progress = overall_base + stack_portion
        self.progress_bar.setValue(int(total_progress))

    def processing_one_finished(self, result):
        """Handle completion of one stack
        @param result Processed image
        """
        print(f"\n=== Saving result for stack {self.current_stack + 1} ===")
        
        # Save intermediate result
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'stack_{self.current_stack + 1}of{len(self.stacks)}_{timestamp}'
        ext = '.jpg'
            
        output_path = os.path.join('output', filename + ext)
        print(f"Creating output directory...")
        os.makedirs('output', exist_ok=True)
        print(f"Saving to: {output_path}")
        
        try:
            print(f"Saving image with format JPEG and color space {self.color_combo.currentText()}")
            self.stacker.save_image(
                result,
                output_path,
                'JPEG',
                self.color_combo.currentText()
            )
            print(f"Successfully saved stack result")
            self.results.append(result)
            status_text = f'Completed stack {self.current_stack + 1} of {len(self.stacks)}'
            print(status_text)
            self.status_label.setText(status_text)
        except Exception as e:
            error_msg = f'Failed to save image: {str(e)}'
            print(f"ERROR: {error_msg}")
            QMessageBox.critical(self, 'Error', error_msg)
            
        # Process next stack
        self.current_stack += 1
        self._process_next_stack()

    def processing_all_finished(self):
        """Handle completion of all stacks"""
        print("\n=== All Processing Complete ===")
        print(f"Total stacks processed: {len(self.results)}")
        self.progress_bar.setVisible(False)
        self.stop_btn.setEnabled(False)
        status_text = f'Processing complete - {len(self.results)} stacks processed'
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
