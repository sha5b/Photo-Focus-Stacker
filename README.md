# Focus Stacking for Microscopy and Macro Photography

This focus stacking tool was developed specifically for the [OpenScan](https://openscan.eu) community to enable high-quality focus stacking for photogrammetry and 3D scanning applications. OpenScan is an open-source 3D scanner project that makes professional 3D scanning accessible to everyone.

## Results

Here are some example results from the focus stacking process:

![Focus Stack Result 2](results/stack_3of40_20250108_190134.jpg)

These images demonstrate the tool's capability to combine multiple images taken at different focus distances into a single, sharp composite image where the entire subject is in focus.

## Quick Start Guide

### Using the GUI

1. Launch the application:
```bash
python src/main.py
```

2. Using the tool:
   - Click "Select Images" to choose your focus stack images
   - Images should be taken at different focus distances of the same subject
   - Name your images with sequential numbers (e.g., scan_1.jpg, scan_2.jpg, etc.) For best results check the output filenames of your Openscanner, should be ok.
   - All images in a stack should be taken with the same camera settings
   - Recommended: 5-15 images per stack with small focus steps

3. Processing:
   - Select your output folder
   - Click "Process Stack" to start focus stacking
   - The tool will automatically align and blend your images
   - Results will be saved in your chosen output folder

### Tips for Best Results

1. Image Capture:
   - Use a stable setup (tripod or scanning rig)
   - Keep consistent lighting
   - Use manual focus and consistent camera settings
   - Take more images than you think you need (small focus steps)
   - Ensure good overlap between focus areas

2. For Photogrammetry:
   - Ensure the main subject is perfectly sharp
   - Background blur is acceptable and normal
   - Use enough images to capture all depth levels of your subject

Advanced focus stacking implementation optimized for microscopy and macro photography, with GPU acceleration using CUDA. This tool is specifically designed for handling the unique challenges of microscopy and macro photography, including high magnification, shallow depth of field, and the need for precise alignment.

Perfect for:
- Photogrammetry and 3D scanning
- Microscopy imaging (biological, metallurgical, etc.)
- Macro photography (insects, minerals, small objects)
- Product photography requiring full depth of field
- Scientific documentation and research

[!["Buy Me A Coffee"](https://www.buymeacoffee.com/assets/img/guidelines/download-assets-2.svg)](https://buymeacoffee.com/sha5b)

## System Requirements & Testing Environment

This software has been developed and tested on:
- Windows 11
- NVIDIA RTX 3080 Ti with CUDA 12.x
- Python 3.8 or higher

Requirements:
- NVIDIA GPU with CUDA support (minimum 4GB VRAM recommended)
- CUDA Toolkit 12.x (we use CUDA 12.3)
- Visual Studio Build Tools (Windows) or GCC (Linux)

CUDA Installation Guide:

For Windows:
1. Install Visual Studio Build Tools 2019 or later with C++ development tools
2. Download and install CUDA 12.3 from NVIDIA's website:
   https://developer.nvidia.com/cuda-12-3-0-download-archive
   - Select Windows
   - Select your version (10 or 11)
   - Select your architecture (x86_64)
3. Ensure your GPU drivers are up to date

For macOS:
- Note: CUDA is not supported on macOS since macOS 10.14 (Mojave)
- For Mac users, the fallback method for cpu is not implemented, sorry get a real pc.

For Linux:

Debian/Ubuntu:
```bash
# Add NVIDIA package repositories
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.3.0/local_installers/cuda-repo-ubuntu2204-12-3-local_12.3.0-545.23.06-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2204-12-3-local_12.3.0-545.23.06-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2204-12-3-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt update
sudo apt install cuda-12-3
```

Fedora:
```bash
sudo dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/fedora37/x86_64/cuda-fedora37.repo
sudo dnf clean all
sudo dnf module disable nvidia-driver
sudo dnf -y install cuda-12-3
```

Arch Linux:
```bash
# Install from official repositories
sudo pacman -S cuda
```

After installation on any system:
1. Add CUDA to your PATH (add to your .bashrc or .zshrc):
```bash
export PATH=/usr/local/cuda-12.3/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.3/lib64:$LD_LIBRARY_PATH
```
2. Verify installation:
```bash
nvcc --version
```

## Setup

1. Create and activate a virtual environment:
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

2. Install requirements:
```bash
pip install -r requirements.txt
```

## Advanced Usage (Python API)

For advanced users who want to integrate the focus stacking into their own scripts:

```python
from focus_stacker import FocusStacker

# Initialize stacker with default settings
stacker = FocusStacker()

# Process a single stack
result = stacker.process_stack(['image1.jpg', 'image2.jpg', 'image3.jpg'])
stacker.save_image(result, 'output.jpg')

# Process multiple stacks in a directory
import glob
image_paths = glob.glob('path/to/images/*.jpg')
stacks = stacker.split_into_stacks(image_paths, stack_size=3)

for i, stack in enumerate(stacks):
    result = stacker.process_stack(stack)
    stacker.save_image(result, f'output_{i+1}.jpg')
```

### Options

The `FocusStacker` class accepts several parameters to fine-tune the stacking process:

- `radius` (1-20, default: 8): Size of the focus measure window. Larger values can help with noisy images but may reduce detail.
- `smoothing` (1-10, default: 4): Amount of smoothing applied to focus maps. Higher values reduce noise but may affect edge detection.
- `scale_factor` (1-4, default: 2): Processing scale multiplier:
  - 1 = original resolution
  - 2 = 2x upscaling (recommended)
  - 3 = 3x upscaling (more detail, slower)
  - 4 = 4x upscaling (maximum detail, much slower)

Example with custom options:
```python
stacker = FocusStacker(
    radius=10,      # Larger focus window
    smoothing=3,    # Less smoothing for more detail
    scale_factor=3  # 3x upscaling for enhanced detail
)
```

## Contributing

Contributions are very welcome! Whether you're fixing bugs, adding new features, or improving documentation, your help is appreciated.

Feel free to:
- Fork the repository
- Create a feature branch
- Submit pull requests
- Open issues for bugs or feature requests
- Suggest improvements

Let's make this tool even better together!

## License

This project is licensed under a Non-Commercial Open Source License (NCOSL). This means:
- ✅ Free for personal and academic use
- ✅ Open source and can be modified
- ✅ Must maintain the same license if distributed
- ❌ Cannot be used for commercial purposes without permission
- ❌ Cannot be sold or included in commercial products

See the LICENSE file for full details.

For commercial licensing inquiries, please open an issue or contact the project maintainer.
