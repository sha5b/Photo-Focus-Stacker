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

2. **Load Images:** Click "Load Images" and select the image files for your focus stack. Images should ideally be named sequentially (e.g., `img_001.jpg`, `img_002.jpg`). The tool will attempt to automatically group them into stacks.
3. **Configure Parameters:** Adjust the stacking parameters in the UI as needed:
    *   **Alignment:** Choose the image alignment method (e.g., 'orb').
    *   **Focus Measure:** Select how image sharpness is determined (e.g., 'custom').
    *   **Blending:** Choose the method for combining images (e.g., 'weighted', 'laplacian').
    *   **Laplacian Levels:** (Only for 'laplacian' blending) Number of pyramid levels.
    *   **Consistency Filter:** (Only for 'laplacian' blending) Check to apply a median filter to reduce noise in the selection map. Adjust kernel size if needed.
    *   **Post-processing:** Check to apply contrast adjustment and sharpening.
4. **Output Settings:** Optionally set a base name for the output files.
5. **Process:** Click "Process Stack". The results will be saved in the `results/` directory.

### Tips for Best Results

*   **Stability:** Use a stable setup (tripod, copy stand, scanner) to minimize movement between shots.
*   **Lighting:** Keep lighting consistent across all images in a stack.
*   **Settings:** Use manual focus and consistent camera settings (aperture, shutter speed, ISO).
*   **Overlap:** Ensure sufficient overlap in focus between consecutive images. Small focus steps are better than large ones.
*   **Sequence:** Take enough images to cover the entire depth of field of your subject.

## Installation

This tool requires Python 3 (3.8 or higher recommended) and several Python packages.

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/your-username/Photo-Focus-Stacker.git # Replace with actual URL if different
    cd Photo-Focus-Stacker
    ```

2.  **Create a Virtual Environment:** (Recommended)
    ```bash
    # Windows
    python -m venv venv
    venv\Scripts\activate

    # macOS / Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install Requirements:**
    ```bash
    pip install -r requirements.txt
    ```
    *Note: This installs packages like OpenCV, NumPy, PyQt5, etc.*

## Running the Application

Once installed, run the GUI using:
```bash
python src/main.py
```

## Advanced Usage (Python API)

You can integrate the focus stacking logic into your own Python scripts.

```python
# Ensure you are in the project's root directory or have src in your Python path
from src.focus_stacker import FocusStacker
from src.utils import save_image, split_into_stacks # Import necessary utils
import glob
import os

# --- Example 1: Process a single stack with default settings ---
print("Processing single stack...")
stacker_defaults = FocusStacker() # Uses default parameters
image_files = sorted(glob.glob('path/to/your/stack/*.jpg')) # Get your image files
if image_files:
    result_default = stacker_defaults.process_stack(image_files)
    save_image(result_default, 'results/output_default.jpg')
    print("Saved default result.")
else:
    print("No images found for single stack example.")


# --- Example 2: Process multiple stacks with custom settings ---
print("\nProcessing multiple stacks with custom settings...")
stacker_custom = FocusStacker(
    align_method='orb',
    focus_measure_method='custom',
    blend_method='laplacian',
    consistency_filter=True,
    consistency_kernel=7,
    postprocess=True,
    laplacian_levels=6
)

all_images = sorted(glob.glob('path/to/all/images/*.tif')) # Get all images
if all_images:
    # Auto-detect stacks based on filenames (e.g., stack1_001.tif, stack1_002.tif, stack2_001.tif)
    stacks = split_into_stacks(all_images, stack_size=0) # stack_size=0 for auto-detect

    if not stacks:
        print("Could not detect stacks, treating all images as one.")
        stacks = [all_images] # Fallback to single stack

    output_dir = 'results/custom_stacks'
    os.makedirs(output_dir, exist_ok=True)

    for i, stack in enumerate(stacks):
        print(f"\nProcessing custom stack {i+1}/{len(stacks)}...")
        if stack:
            result_custom = stacker_custom.process_stack(stack)
            output_filename = f'custom_stack_{i+1}.png' # Save as PNG
            save_image(result_custom, os.path.join(output_dir, output_filename), format='PNG')
            print(f"Saved custom result {i+1}.")
        else:
            print(f"Skipping empty stack {i+1}.")
else:
    print("No images found for multiple stacks example.")

```

### `FocusStacker` Options

When initializing `FocusStacker`, you can customize its behavior:

*   `align_method` (str, default='orb'): Method for aligning images. Options: 'orb'. ('ecc' is a placeholder).
*   `focus_measure_method` (str, default='custom'): Method for measuring focus. Options: 'custom', 'laplacian_variance' (uses absolute Laplacian map).
*   `blend_method` (str, default='weighted'): Method for blending images. Options: 'weighted', 'laplacian'.
*   `consistency_filter` (bool, default=False): Apply median filter to the sharpness map before Laplacian blending. Helps reduce noise.
*   `consistency_kernel` (int, default=5): Kernel size for the consistency filter (must be odd).
*   `postprocess` (bool, default=True): Apply contrast/brightness adjustment and sharpening after blending.
*   `laplacian_levels` (int, default=5): Number of pyramid levels used in Laplacian blending.

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
