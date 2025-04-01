# Focus Stacking for Microscopy and Macro Photography

This focus stacking tool was developed specifically for the [OpenScan](https://openscan.eu) community to enable high-quality focus stacking for photogrammetry and 3D scanning applications. OpenScan is an open-source 3D scanner project that makes professional 3D scanning accessible to everyone.

## Table of Contents

- [Quick Start Guide](#quick-start-guide)
  - [Using the GUI](#using-the-gui)
  - [Tips for Best Results](#tips-for-best-results)
  - [Parameter Tuning Guide](#parameter-tuning-guide)
- [Installation](#installation)
- [Running the Application](#running-the-application)
- [Advanced Usage (Python API)](#advanced-usage-python-api)
  - [`FocusStacker` Options (Python API)](#focusstacker-options-python-api)
- [Contributing](#contributing)
- [License](#license)

---

## Quick Start Guide

### Using the GUI

1. Launch the application:
```bash
python src/main.py
```

2. **Load Images:** Click "Load Images" and select the image files. The tool groups images into stacks based on filenames (e.g., `scan1_001.jpg`, `scan1_002.jpg`).
3. **Configure Parameters:** Adjust settings like Alignment Levels, Focus Window, Blending Method, etc. See the **Parameter Tuning Guide** below for details.
4. **Output Settings:** Optionally set a custom output name prefix and format.
5. **Process:** Click "Process Stack". Results are saved in the `results/` directory.

### Tips for Best Results

*   **Stability:** Use a stable setup (tripod, copy stand, scanner) to minimize movement between shots.
*   **Lighting:** Keep lighting consistent across all images in a stack.
*   **Settings:** Use manual focus and consistent camera settings (aperture, shutter speed, ISO).
*   **Overlap:** Ensure sufficient overlap in focus between consecutive images.
*   **Sequence:** Take enough images to cover the entire depth of field of your subject. Small focus steps are generally better than large ones.

### Parameter Tuning Guide

The core algorithms (Pyramid ECC Homography alignment, Laplacian Variance focus measure) are fixed for robustness. You can tune the following parameters in the GUI:

*   **Alignment Pyramid Levels:** (Default: 3) Controls the multi-resolution strategy for image alignment. The alignment starts on downscaled images (higher pyramid levels) and refines on progressively larger versions.
    *   **Increase:** **Improves alignment** for images with larger shifts/rotations. **Increases processing time** significantly. Try `4` or `5` if alignment fails or seems inaccurate.
    *   **Decrease:** **Speeds up alignment** considerably. May **fail** if images have significant movements. `1` disables the pyramid (fastest, but least robust).
*   **Alignment Mask Threshold:** (Default: 10) Sets the minimum edge strength required for a pixel to be considered during alignment. Alignment focuses on matching these edge pixels.
    *   **Increase:** Includes weaker edges (more permissive). Might **help in low-contrast areas** but can be **influenced by noise**. Try `15` or `20`.
    *   **Decrease:** Focuses only on strong edges (stricter). **Improves robustness against noise** but might **fail if strong edges are lacking**. Try `5` or `8`.
*   **Focus Window Size:** (Default: 7) The side length (in pixels) of the square window used to calculate local sharpness. Must be odd.
    *   **Increase:** **Smoother focus map** (less noise) by averaging over a larger area. Can **blur focus transitions** and lose fine detail. Try `9` or `11`.
    *   **Decrease:** **Finer detail** in focus map, potentially sharper transitions. **More sensitive to noise**. Try `5`. Using `3` is often too noisy.
*   **Sharpening Strength:** (Default: 0.0) Controls the intensity of an Unsharp Mask filter applied *after* stacking.
    *   **Increase:** **Enhances apparent sharpness** and contrast. Moderate values: `0.5` to `1.0`. Higher values (> 1.0) often cause **excessive noise and halos**.
    *   **Decrease:** Reduces sharpening. `0.0` **disables sharpening** (recommended for initial processing).
*   **Blending Method:** (Default: Weighted Blending) Determines how the final image is assembled from the sharpest parts of the aligned source images.
    *   `Weighted Blending`: **Smoothly blends** pixels based on focus scores. Generally **fewer artifacts** and handles noisy focus maps better. **Recommended starting point.**
    *   `Direct Map Selection`: Directly copies the pixel from the single sharpest source image. Can be **maximally sharp** but **prone to noise/artifacts** if the focus map is noisy. Can be slightly faster.

**Good Starting Point:**

For most stacks, the default settings are a reasonable starting point:
*   Pyramid Levels: `3`
*   Mask Threshold: `10`
*   Focus Window: `7`
*   Sharpening: `0.0`
*   Blending: `Weighted Blending`

**Example: Settings for Best Quality (Potentially Slower):**

If quality is paramount and processing time is less critical, you might try:
*   Pyramid Levels: `4` or `5` (Handles larger misalignments more accurately)
*   Mask Threshold: `8` (Focuses alignment on stronger edges)
*   Focus Window: `5` (Captures finer focus detail, but check for noise)
*   Sharpening: `0.0` (Apply sharpening later if needed, starting clean)
*   Blending: `Weighted Blending` (Generally smoother results)
*Note: These settings increase processing time, especially the pyramid levels.*

**Example: Settings for Fastest Speed (Potentially Lower Quality):**

If speed is the main goal (e.g., for quick previews):
*   Pyramid Levels: `1` (Fastest alignment, assumes minimal image shift)
*   Mask Threshold: `10` (Default is usually fine for speed)
*   Focus Window: `7` (Default is a balance)
*   Sharpening: `0.0` (Sharpening adds time)
*   Blending: `Direct Map Selection` (May be slightly faster than weighted)
*Note: Using Pyramid Level 1 may fail if images are not well-aligned initially. Direct Map blending might introduce artifacts.*

**Combinations to Watch Out For (Potentially "Bad" Settings):**

While results depend heavily on the source images, some combinations tend to cause issues:
*   **Small Focus Window + Direct Map Blending + High Sharpening:** Using a very small `Focus Window Size` (e.g., 3 or 5) can create a noisy focus map. Combining this with `Direct Map Selection` can transfer that noise directly into the output. Adding high `Sharpening Strength` (e.g., > 1.0) will amplify this noise further, likely leading to poor results.
*   **Very High Mask Threshold:** Setting the `Alignment Mask Threshold` extremely high (e.g., 50+) might prevent the alignment algorithm from finding enough features if the image lacks strong edges, potentially leading to misalignment.
*   **Pyramid Level 1 with Large Movements:** If your images have significant shifts or rotation between them, using only `1` `Alignment Pyramid Level` (no pyramid) might not be sufficient for accurate alignment.
*   **Excessive Sharpening:** Regardless of other settings, very high `Sharpening Strength` (e.g., > 1.5) almost always introduces undesirable halos and noise. Apply sharpening cautiously.

***The best approach is always to start with the defaults, inspect the output image carefully, and then adjust one parameter at a time based on the specific issues you observe.***

---

## Installation

This tool requires Python 3 (3.8 or higher recommended) and several Python packages.

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/your-username/Photo-Focus-Stacker.git # Replace with actual URL if different
    cd Photo-Focus-Stacker
    ```

2.  **Create and Activate a Virtual Environment:** (Highly Recommended)

    Using a virtual environment isolates the project's dependencies from your global Python installation, preventing conflicts between projects.

    *   **Windows (Command Prompt - cmd.exe):**
        ```bash
        # Create the virtual environment (replace 'python' with 'py -3' or specific python path if needed)
        python -m venv venv
        # Activate the virtual environment
        venv\Scripts\activate.bat
        # Your prompt should now show (venv) at the beginning
        ```

    *   **Windows (PowerShell):**
        ```powershell
        # Create the virtual environment (replace 'python' with 'py -3' or specific python path if needed)
        python -m venv venv
        # Activate the virtual environment (you might need to adjust execution policy)
        # Run: Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
        venv\Scripts\Activate.ps1
        # Your prompt should now show (venv) at the beginning
        ```

    *   **macOS / Linux (Bash/Zsh):**
        ```bash
        # Create the virtual environment (use python3)
        python3 -m venv venv
        # Activate the virtual environment
        source venv/bin/activate
        # Your prompt should now show (venv) at the beginning
        ```

    *   **Deactivating:** When you're finished working on the project, simply type `deactivate` in the terminal.

3.  **Install Requirements:**

    *Ensure your virtual environment is activated before running this command.*
    ```bash
    pip install -r requirements.txt
    ```
    *Note: This installs packages like OpenCV, NumPy, PyQt5, etc.*

## Running the Application

Once installed, run the GUI using:
```bash
python src/main.py
```

---

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
# Example using the 'direct_map' blending and different parameters
stacker_custom = FocusStacker(
    focus_window_size=5,      # Smaller window for potentially finer focus map
    sharpen_strength=0.8,     # Apply some sharpening
    num_pyramid_levels=4,     # More pyramid levels for alignment
    gradient_threshold=8,     # Stricter alignment mask
    blend_method='direct_map' # Use the new direct map blending
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

### `FocusStacker` Options (Python API)

When initializing `FocusStacker` in your Python code, you can customize its behavior:

*   `focus_window_size` (int, default=7): Window size for the Laplacian Variance Map focus measure. Must be odd.
*   `sharpen_strength` (float, default=0.0): Strength of the final Unsharp Mask filter. Set to 0.0 to disable.
*   `num_pyramid_levels` (int, default=3): Number of levels for the Pyramid ECC alignment. `1` disables the pyramid approach.
*   `gradient_threshold` (int, default=10): Threshold for the ECC alignment gradient mask.
*   `blend_method` (str, default='weighted'): Blending method to use. Options: `'weighted'`, `'direct_map'`.

---

## Contributing

Contributions are very welcome! Whether you're fixing bugs, adding new features, or improving documentation, your help is appreciated.

Feel free to:
- Fork the repository
- Create a feature branch
- Submit pull requests
- Open issues for bugs or feature requests
- Suggest improvements

Let's make this tool even better together!

---

## License

This project is licensed under a Non-Commercial Open Source License (NCOSL). This means:
- ✅ Free for personal and academic use
- ✅ Open source and can be modified
- ✅ Must maintain the same license if distributed
- ❌ Cannot be used for commercial purposes without permission
- ❌ Cannot be sold or included in commercial products

See the LICENSE file for full details.

For commercial licensing inquiries, please open an issue or contact the project maintainer.
