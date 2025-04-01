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

2. **Load Images:** Click "Load Images" and select the image files for your focus stack. The tool expects filenames like `prefix_stackID-imageID.ext` (e.g., `wirbel_0_1.jpg`, `wirbel_0_2.jpg`) and will group images based on the `prefix_stackID` part (e.g., `wirbel_0`).
3. **Configure Parameters:** Adjust the stacking parameters in the UI:
    *   **Alignment Pyramid Levels:** Number of levels for the Pyramid ECC Homography alignment. More levels can increase accuracy for larger misalignments but take longer. `1` means no pyramid. Default: 3.
    *   **Alignment Mask Threshold:** Threshold for the gradient mask used during ECC alignment. Lower values make the mask stricter (focusing ECC on stronger edges), higher values make it more permissive. Default: 10.
    *   **Focus Window Size:** Size of the square window (pixels) used for calculating the sharpness map (Laplacian Variance). Smaller values (e.g., 5, 7) capture finer detail but can be noisier. Larger values (e.g., 9, 11) give smoother maps but might blur focus boundaries. Default: 7.
    *   **Sharpening Strength:** Controls the amount of Unsharp Masking applied to the final image. 0.0 means no sharpening. Values around 0.5-1.0 provide moderate sharpening. Higher values increase sharpness but may amplify noise. Default: 0.0.
    *   *(Note: Alignment is fixed to Pyramid ECC Homography with Masking, Focus Measure is fixed to Laplacian Variance Map, Blending is fixed to Weighted Blending for simplicity and robustness).*
4. **Output Settings:** Optionally set a custom prefix for the output files. If left blank, the original stack name (e.g., `wirbel_0`) will be used.
5. **Process:** Click "Process Stack". The results will be saved in the `results/` directory.

### Tips for Best Results

*   **Stability:** Use a stable setup (tripod, copy stand, scanner) to minimize movement between shots.
*   **Lighting:** Keep lighting consistent across all images in a stack.
*   **Settings:** Use manual focus and consistent camera settings (aperture, shutter speed, ISO).
*   **Overlap:** Ensure sufficient overlap in focus between consecutive images. Small focus steps are better than large ones.
*   **Sequence:** Take enough images to cover the entire depth of field of your subject.

### Recommended Settings

The core algorithms are now fixed to prioritize robustness and potential quality:
*   **Alignment:** Pyramid ECC Homography
*   **Focus Measure:** Laplacian Variance Map
*   **Blending:** Weighted Blending

The main parameters to tune in the UI are:
*   **Alignment Pyramid Levels:** Start with `3`. Increase if alignment seems poor for large movements, decrease to `1` (no pyramid) for speed if alignment is already good.
*   **Alignment Mask Threshold:** Start with `10`. Decrease (e.g., to 5) if alignment struggles in low-contrast areas, increase (e.g., to 20) if background noise seems to interfere with alignment.
*   **Focus Window Size:** Start with `7`. Decrease to `5` for potentially finer detail (if source images are sharp), increase to `9` or `11` if the focus map seems too noisy or results have speckles.
*   **Sharpening Strength:** Start with `0.0`. Increase gradually (e.g., `0.5`, `0.8`, `1.0`) to enhance detail, but reduce if noise or halos become excessive.

*Always inspect your results and adjust these parameters based on the visual outcome.*

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

### `FocusStacker` Options (Python API)

When initializing `FocusStacker` in your Python code, you can customize its behavior:

*   `focus_window_size` (int, default=7): Window size for the Laplacian Variance Map focus measure. Must be odd.
*   `sharpen_strength` (float, default=0.0): Strength of the final Unsharp Mask filter. Set to 0.0 to disable.
*   `num_pyramid_levels` (int, default=3): Number of levels for the Pyramid ECC alignment. `1` disables the pyramid approach.
*   `gradient_threshold` (int, default=10): Threshold for the ECC alignment gradient mask.

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
