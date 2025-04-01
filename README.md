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
    *   **Alignment:** Method used to align images before stacking.
        *   `orb`: Faster, feature-based alignment using ORB features and homography. Good default. (Uses Cross-Check matching).
        *   `ecc`: Slower, potentially more accurate pixel-based alignment (Enhanced Correlation Coefficient). Can handle subtle warps better. Choose Motion Type below.
        *   `akaze`: Feature-based alignment similar to ORB, but uses AKAZE features which can be more robust to scale and rotation changes.
    *   **ECC Motion Type:** (Only for `ecc` alignment) The mathematical model ECC uses to describe the transformation between images. Choose based on expected camera/subject movement:
        *   `TRANSLATION`: Assumes movement is only horizontal (X) and vertical (Y) shifts. Fastest, but only suitable if there's no rotation, scaling, or perspective change.
        *   `AFFINE`: Handles translation, rotation, scaling, and shear (skewing). A good general-purpose choice for many scenarios where perspective distortion is minimal. (6 degrees of freedom).
        *   `HOMOGRAPHY` (Perspective): Handles full perspective transformations, including changes in viewpoint. Most flexible but also most complex and potentially less stable if the movement is actually simpler (e.g., purely affine). Use if you see perspective distortions not corrected by AFFINE. (8 degrees of freedom).
    *   **Focus Measure:** Algorithm to determine the sharpest pixels in each image.
        *   `custom`: An experimental multi-faceted approach.
        *   `laplacian_variance`: Older method, uses absolute Laplacian as a basic sharpness map.
        *   `laplacian_variance_map`: Calculates sharpness based on the variance of the Laplacian within a local window. Generally recommended.
    *   **Focus Window Size:** (Only for `laplacian_variance_map`) Size of the square window (pixels) used to calculate local sharpness variance. Smaller values (e.g., 7) capture finer detail but can be noisier. Larger values (e.g., 11, 15) give smoother maps but might blur focus boundaries. Default: 9.
    *   **Blending:** Method for combining the sharpest parts from aligned images.
        *   `weighted`: Simple and fast blending based directly on focus map values. Good default.
        *   `laplacian`: More complex pyramid-based blending, can produce smoother transitions but is slower. Often used with the Consistency Filter.
    *   **Laplacian Levels:** (Only for `laplacian` blending) Number of pyramid levels used for blending. More levels capture finer details but increase processing time. Default: 5.
    *   **Apply Consistency Filter:** (Recommended for `laplacian` blending) Applies a median filter to the internal map that decides which source image is sharpest at each pixel. Reduces noise and small inconsistent regions.
    *   **Filter Kernel Size:** (Only if Consistency Filter is checked) Size of the median filter kernel (must be odd). Larger values increase smoothing. Default: 5.
    *   ~~**Apply Post-processing:**~~ (This option has been removed).
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

*   `align_method` (str, default='orb'): Method for aligning images.
    *   `'orb'`: Feature-based alignment (ORB features, homography). Uses KNN + Ratio Test matching.
    *   `'ecc'`: Pixel-based alignment (Enhanced Correlation Coefficient).
    *   `'akaze'`: Feature-based alignment (AKAZE features, homography). Uses KNN + Ratio Test matching.
*   `ecc_motion_type` (str, default='AFFINE'): Motion model used only if `align_method='ecc'`. Defines the type of transformation ECC estimates.
    *   `'TRANSLATION'`: Models only X/Y shifts.
    *   `'AFFINE'`: Models translation, rotation, scale, shear.
    *   `'HOMOGRAPHY'`: Models perspective transformations.
*   `focus_measure_method` (str, default='custom'): Method for measuring focus.
    *   `'custom'`: Experimental multi-faceted approach.
    *   `'laplacian_variance'`: Older method using absolute Laplacian map.
    *   `'laplacian_variance_map'`: Recommended method using local variance of Laplacian.
*   `focus_window_size` (int, default=9): Window size for the `'laplacian_variance_map'` method. Must be odd.
*   `blend_method` (str, default='weighted'): Method for blending images.
    *   `'weighted'`: Simple blending based on focus map weights.
    *   `'laplacian'`: Pyramid-based blending for smoother transitions.
*   `consistency_filter` (bool, default=False): Apply median filter to the internal sharpness selection map before Laplacian blending. Helps reduce noise. Recommended if `blend_method='laplacian'`.
*   `consistency_kernel` (int, default=5): Kernel size for the consistency filter (must be odd). Used only if `consistency_filter=True`.
*   ~~`postprocess` (bool, default=True):~~ This parameter has been removed.
*   `laplacian_levels` (int, default=5): Number of pyramid levels used in Laplacian blending. Used only if `blend_method='laplacian'`.

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
