# Focus Stacking for Microscopy and Macro Photography

Advanced focus stacking implementation optimized for microscopy and macro photography, with GPU acceleration using CUDA. This tool is specifically designed for handling the unique challenges of microscopy and macro photography, including high magnification, shallow depth of field, and the need for precise alignment.

Perfect for:
- Microscopy imaging (biological, metallurgical, etc.)
- Macro photography (insects, minerals, small objects)
- Product photography requiring full depth of field
- Scientific documentation and research

[!["Buy Me A Coffee"](https://www.buymeacoffee.com/assets/img/custom_images/orange_img.png)](https://buymeacoffee.com/sha5b)

## Prerequisites

Before installing the project requirements, you need:

- Python 3.8 or higher
- NVIDIA GPU with CUDA support (minimum 4GB VRAM recommended)
- CUDA Toolkit 11.0 or higher
- Visual Studio Build Tools (Windows) or GCC (Linux)

For Windows users:
1. Install Visual Studio Build Tools 2019 or later with C++ development tools
2. Install CUDA Toolkit from NVIDIA's website
3. Ensure your GPU drivers are up to date

For Linux users:
1. Install GCC and required development tools:
   ```bash
   # Ubuntu/Debian
   sudo apt-get update
   sudo apt-get install build-essential

   # Fedora
   sudo dnf groupinstall "Development Tools"
   ```
2. Install CUDA Toolkit from NVIDIA's website or package manager

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

## Usage

Basic usage:
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

### Tips for Best Results

1. Image Capture:
   - Use a tripod or stable mounting
   - Ensure consistent lighting across all images
   - Take more images than you think you need (small focus steps)
   - Use manual focus and consistent camera settings

2. Memory Management:
   - For large images (>30MP), start with scale_factor=1
   - Process one stack at a time
   - Close other GPU-intensive applications

3. Troubleshooting:
   - If alignment fails, try with more overlap between focus steps
   - For noisy results, increase the smoothing parameter
   - For more detail, decrease radius and increase scale_factor
   - If you get CUDA out of memory errors, reduce scale_factor

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

### Automatic Stack Detection

The `split_into_stacks` method can automatically group images into stacks based on filename patterns:
```python
stacks = stacker.split_into_stacks(image_paths, stack_size=3)
for stack in stacks:
    result = stacker.process_stack(stack)
    # Save each result...
```

## Improvements

Current areas being worked on:

1. Memory optimization for processing very large images
2. Enhanced alignment for challenging subjects
3. Additional color space support
4. Multi-GPU support for faster processing
5. Batch processing interface
6. Progress reporting and cancellation support

## Performance and Results

### Processing Times
Typical processing times on an RTX 3060 (12GB):
- 12MP images: ~5-10 seconds per stack
- 24MP images: ~15-20 seconds per stack
- 45MP images: ~30-40 seconds per stack
- >50MP images: May require scale_factor=1 due to memory constraints

Memory usage scales with image resolution and scale_factor. As a rough guide:
- 12MP images: ~4GB VRAM
- 24MP images: ~6GB VRAM
- 45MP images: ~10GB VRAM
- >50MP images: >12GB VRAM

### Quality Expectations
- Best results with 5-15 images per stack
- Excellent detail preservation in microscopy samples
- Clean edges with minimal artifacts
- Natural-looking transitions between focus regions

## Known Issues

1. High memory usage with large images (>50MP)
2. May require manual alignment for extreme focus differences
3. Limited to NVIDIA GPUs (CUDA requirement)
4. Some color shifts in high-contrast areas
5. Occasional alignment issues with highly repetitive patterns
6. Memory errors possible with large images at high scale factors

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

This project is licensed under the MIT License - see the LICENSE file for details.
