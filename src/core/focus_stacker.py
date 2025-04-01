#!/usr/bin/env python3

import cv2
import numpy as np
import os

# Import functions from refactored modules using relative imports
from . import utils
from . import alignment # Now contains align_images (Pyramid ECC Homography w/ Masking)
from . import focus_measure # Now only contains measure_laplacian_variance_map
from . import blending # Now only contains blend_weighted
from . import postprocessing # For sharpening

class StackingCancelledException(Exception):
    """Custom exception for cancelled stacking process."""
    pass

class FocusStacker:
    def __init__(self,
                 focus_window_size=7,      # Window size for laplacian_variance_map
                 sharpen_strength=0.0,     # Strength for final sharpening
                 num_pyramid_levels=3,     # Levels for Pyramid ECC alignment
                 gradient_threshold=10,    # Threshold for ECC gradient mask
                 blend_method='weighted'   # New parameter for blending method
                 ):
        """
         Initializes the FocusStacker orchestrator.
         Uses Pyramid ECC Homography with Masking for alignment, Laplacian Variance Map for focus.
         Allows choosing between 'weighted' and 'direct_map' blending.

         @param focus_window_size: Window size for focus measure (default: 7).
         @param sharpen_strength: Strength of Unsharp Mask filter (0.0 to disable). Default: 0.0.
         @param num_pyramid_levels: Number of levels for Pyramid ECC alignment (default: 3).
         @param gradient_threshold: Threshold for creating the ECC gradient mask (default: 10).
         @param blend_method: Blending method ('weighted' or 'direct_map'). Default: 'weighted'.
        """
        print("Initializing FocusStacker...")
        # Hardcoded methods are now implicit in the functions called
        self.align_method_desc = f'Pyramid ECC Homography ({num_pyramid_levels} levels, Masked)'
        self.focus_measure_method_desc = f'Laplacian Variance Map (window={focus_window_size})'

        if blend_method not in ['weighted', 'direct_map']:
            print(f"Warning: Invalid blend_method '{blend_method}'. Defaulting to 'weighted'.")
            self.blend_method = 'weighted'
        else:
            self.blend_method = blend_method
        self.blend_method_desc = 'Weighted' if self.blend_method == 'weighted' else 'Direct Map Selection'


        # Store relevant parameters
        self.focus_window_size = focus_window_size
        self.sharpen_strength = sharpen_strength
        self.num_pyramid_levels = num_pyramid_levels
        self.gradient_threshold = gradient_threshold # Store gradient threshold
        # self.blend_method = blend_method # Already assigned above
        self._stop_requested = False # Flag to signal stopping

        # Initialize color profiles (now handled in utils)
        utils.init_color_profiles()
        print(f"  Alignment: {self.align_method_desc}")
        print(f"  Focus Measure: {self.focus_measure_method_desc}")
        print(f"  Blending: {self.blend_method_desc}") # Updated description
        print(f"  Sharpen Strength: {self.sharpen_strength:.2f}")

    def request_stop(self):
        """Sets the flag to stop processing."""
        print("Stop requested for FocusStacker instance.")
        self._stop_requested = True

    def _check_stop_requested(self):
        """Checks if stop was requested and raises exception if so."""
        if self._stop_requested:
            print("Stop request detected during processing.")
            raise StackingCancelledException("Stacking process cancelled by user.")

    def process_stack(self, image_paths, color_space='sRGB'):
        """
        Main processing pipeline for a single stack of images, using simplified methods.

        @param image_paths: List of paths to the images in the stack.
        @param color_space: Target color space for the output (e.g., 'sRGB', 'AdobeRGB').
                            Conversion happens at the end if needed.
        @return: The final processed (stacked and sharpened) image as a float32 NumPy array [0, 1].
        """
        self._stop_requested = False # Reset stop flag at the start of processing
        if not image_paths or len(image_paths) < 2:
            raise ValueError("Focus stacking requires at least 2 image paths.")

        print(f"\n--- Processing stack of {len(image_paths)} images ---")
        base_filenames = [os.path.basename(p) for p in image_paths]
        print(f"Images: {', '.join(base_filenames[:3])}{'...' if len(base_filenames) > 3 else ''}")

        # 1. Load images using utility function
        images = []
        for i, path in enumerate(image_paths):
            self._check_stop_requested() # Check before loading
            print(f"Loading image {i+1}/{len(image_paths)}: {os.path.basename(path)}")
            try:
                img = utils.load_image(path)
                images.append(img)
            except Exception as e:
                print(f"ERROR loading image {path}: {e}")
                raise # Re-raise critical error

        # 2. Align images using Pyramid ECC Homography with Masking
        self._check_stop_requested() # Check before alignment
        print(f"\nAligning images using {self.align_method_desc}...")
        try:
            # Pass pyramid levels and gradient threshold parameters
            aligned_images = alignment.align_images(
                images,
                num_pyramid_levels=self.num_pyramid_levels,
                gradient_threshold=self.gradient_threshold
            )
            print(f"Alignment complete ({len(aligned_images)} images).")
        except Exception as e:
            if not isinstance(e, StackingCancelledException):
                print(f"ERROR during image alignment: {e}")
            raise # Re-raise

        # 3. Calculate focus measures using Laplacian Variance Map
        print(f"\nCalculating focus measures using {self.focus_measure_method_desc}...")
        focus_maps = []
        for i, img in enumerate(aligned_images):
            self._check_stop_requested() # Check before calculating focus for each image
            print(f"Calculating focus for image {i+1}/{len(aligned_images)}")
            try:
                img_gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
                focus_map = focus_measure.measure_laplacian_variance_map(img_gray, window_size=self.focus_window_size)
                focus_maps.append(focus_map.astype(np.float32))
            except Exception as e:
                 if not isinstance(e, StackingCancelledException):
                    print(f"ERROR calculating focus measure for image {i+1}: {e}")
                 raise # Re-raise
        print("Focus measure calculation complete.")

        # 4. Determine sharpest indices (needed for direct map blending)
        self._check_stop_requested() # Check before index calculation
        sharpest_indices = None
        if self.blend_method == 'direct_map':
            print("\nCalculating sharpest image indices...")
            if focus_maps:
                # Stack focus maps along a new axis (axis=0)
                focus_maps_stack = np.stack(focus_maps, axis=0)
                # Find the index of the maximum focus value along the stack axis
                sharpest_indices = np.argmax(focus_maps_stack, axis=0).astype(np.uint16) # Use uint16 for indices
                print("Sharpest indices calculated.")
            else:
                print("Warning: No focus maps available to calculate sharpest indices.")
                # Handle error or fallback? For now, raise error if needed for direct map
                raise ValueError("Cannot perform direct map blending without focus maps.")


        # 5. Blend images using the selected method
        self._check_stop_requested() # Check before blending
        print(f"\nBlending images using {self.blend_method_desc}...")
        try:
            if self.blend_method == 'weighted':
                blended_image = blending.blend_weighted(aligned_images, focus_maps)
            elif self.blend_method == 'direct_map':
                if sharpest_indices is None:
                      raise ValueError("Sharpest indices map is required for direct map blending but was not calculated.")
                # We need to implement blend_direct_map in blending.py
                blended_image = blending.blend_direct_map(aligned_images, sharpest_indices)
            else:
                # Should not happen due to check in __init__, but good practice
                raise ValueError(f"Unsupported blend method: {self.blend_method}")

            print("Blending complete.")
        except Exception as e:
            if not isinstance(e, StackingCancelledException):
                print(f"ERROR during image blending: {e}")
            raise # Re-raise

        # 6. Apply Sharpening (if strength > 0)
        self._check_stop_requested() # Check before sharpening
        if self.sharpen_strength > 0:
            try:
                final_result = postprocessing.apply_unsharp_mask(blended_image, strength=self.sharpen_strength)
            except Exception as e:
                 if not isinstance(e, StackingCancelledException):
                    print(f"ERROR during sharpening: {e}. Returning blended image without sharpening.")
                 final_result = blended_image # Fallback to blended result
        else:
            print("\nSkipping sharpening.")
            final_result = blended_image # No sharpening applied

        # 7. Color space conversion (if needed) using utility function
        self._check_stop_requested() # Check before final conversion
        if color_space != 'sRGB':
            try:
                final_result = utils.convert_color_space(final_result, target_space=color_space, source_space='sRGB')
            except Exception as e:
                 if not isinstance(e, StackingCancelledException):
                    print(f"ERROR during final color space conversion: {e}")
                 raise # Re-raise

        self._check_stop_requested() # Final check before returning
        print("\n--- Stack processing complete! ---")
        return final_result

    # --- Public methods for saving and splitting (using utils) ---

    def save_image(self, img, path, format='JPEG', quality=95, color_space='sRGB'):
        """Saves the image using the utility function."""
        utils.save_image(img, path, format=format, quality=quality, color_space=color_space)

    def split_into_stacks(self, image_paths, stack_size=0):
        """Splits image paths into stacks using the utility function."""
        return utils.split_into_stacks(image_paths, stack_size=stack_size)

    # --- Removed methods that were moved to other modules ---
    # (Removed methods are now handled in respective modules)
