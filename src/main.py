#!/usr/bin/env python3

import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QPushButton, QLabel, QFileDialog, 
                            QComboBox, QProgressBar, QMessageBox)
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

    def run(self):
        """Process the focus stacking operation"""
        try:
            result = self.stacker.process_stack(
                self.image_paths, 
                self.color_space,
                progress_callback=self.progress.emit
            )
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))

class MainWindow(QMainWindow):
    """
    @class MainWindow
    @brief Main application window for the focus stacking tool
    """
    def __init__(self):
        super().__init__()
        self.image_paths = []
        self.stacker = FocusStacker()
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
        
        save_btn = QPushButton('Save Result')
        save_btn.clicked.connect(self.save_result)
        save_btn.setEnabled(False)
        self.save_btn = save_btn

        # Create comboboxes for format and color space
        format_label = QLabel('Output Format:')
        self.format_combo = QComboBox()
        self.format_combo.addItems(['TIFF (16-bit)', 'PNG (16-bit)', 'JPEG'])

        color_label = QLabel('Color Space:')
        self.color_combo = QComboBox()
        self.color_combo.addItems(['sRGB'])

        # Create status label and progress bar
        self.status_label = QLabel('Ready')
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)

        # Add widgets to layout
        layout.addWidget(load_btn)
        
        format_layout = QHBoxLayout()
        format_layout.addWidget(format_label)
        format_layout.addWidget(self.format_combo)
        layout.addLayout(format_layout)
        
        color_layout = QHBoxLayout()
        color_layout.addWidget(color_label)
        color_layout.addWidget(self.color_combo)
        layout.addLayout(color_layout)
        
        layout.addWidget(process_btn)
        layout.addWidget(save_btn)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.status_label)
        
        # Add stretch to push everything to the top
        layout.addStretch()

    def load_images(self):
        """Open file dialog to select images"""
        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.ExistingFiles)
        file_dialog.setNameFilter("Images (*.png *.jpg *.jpeg *.tif *.tiff)")
        
        if file_dialog.exec_():
            self.image_paths = file_dialog.selectedFiles()
            self.status_label.setText(f'Loaded {len(self.image_paths)} images')

    def process_stack(self):
        """Start the focus stacking process"""
        if not self.image_paths:
            QMessageBox.warning(self, 'Error', 'Please load images first')
            return

        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        
        # Create and start processing thread
        self.thread = FocusStackingThread(
            self.stacker,
            self.image_paths,
            self.color_combo.currentText()
        )
        self.thread.progress.connect(self.update_progress)
        self.thread.finished.connect(self.processing_finished)
        self.thread.error.connect(self.processing_error)
        self.thread.start()

    def update_progress(self, value):
        """Update progress bar
        @param value Progress percentage (0-100)
        """
        self.progress_bar.setValue(value)

    def processing_finished(self, result):
        """Handle completed processing
        @param result Processed image
        """
        self.result_image = result
        self.save_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.status_label.setText('Processing complete')

    def processing_error(self, error_msg):
        """Handle processing errors
        @param error_msg Error message to display
        """
        QMessageBox.critical(self, 'Error', f'Processing failed: {error_msg}')
        self.progress_bar.setVisible(False)
        self.status_label.setText('Processing failed')

    def save_result(self):
        """Save the processed image"""
        if not hasattr(self, 'result_image'):
            return

        # Create output directory if it doesn't exist
        import os
        from datetime import datetime
        output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
        os.makedirs(output_dir, exist_ok=True)

        # Generate default filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        selected_format = self.format_combo.currentText()
        if selected_format == 'TIFF (16-bit)':
            default_ext = '.tif'
            file_filter = 'TIFF (*.tif)'
        elif selected_format == 'PNG (16-bit)':
            default_ext = '.png'
            file_filter = 'PNG (*.png)'
        else:
            default_ext = '.jpg'
            file_filter = 'JPEG (*.jpg)'

        default_filename = f'focus_stacked_{timestamp}{default_ext}'
        default_path = os.path.join(output_dir, default_filename)

        file_dialog = QFileDialog()
        file_dialog.setAcceptMode(QFileDialog.AcceptSave)
        file_dialog.setDirectory(output_dir)
        file_dialog.selectFile(default_filename)
        file_dialog.setNameFilter(file_filter)
            
        if file_dialog.exec_():
            file_path = file_dialog.selectedFiles()[0]
            if not file_path.endswith(default_ext):
                file_path += default_ext
                
            try:
                self.stacker.save_image(
                    self.result_image,
                    file_path,
                    selected_format,
                    self.color_combo.currentText()
                )
                self.status_label.setText('Image saved successfully')
            except Exception as e:
                QMessageBox.critical(self, 'Error', f'Failed to save image: {str(e)}')

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
