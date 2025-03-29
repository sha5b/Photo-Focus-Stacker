#!/usr/bin/env python3

import cv2
import numpy as np
import os

# Import functions from refactored modules using relative imports
from . import utils
from . import alignment
from . import focus_measure
from . import blending
from . import postprocessing

class FocusStacker:
    def __init__(self,
                 align_method='orb',
                 focus_measure_method='custom',
                 blend_method='weighted',
                 consistency_filter=False, # Option to enable consistency filter
                 consistency_kernel=5,     # Kernel size for consistency filter
                 postprocess=True,         # Option to enable post-processing
                 laplacian_levels=5        # Levels for Laplacian pyramid blending
                 ):
        """
        Initializes the FocusStacker orchestrator.

        @param align_method: 'orb' or 'ecc' (default: 'orb').
        @param focus_measure_method: 'custom' or 'laplacian_variance' (default: 'custom').
        @param blend_method: 'weighted' or 'laplacian' (default: 'weighted').
        @param consistency_filter: Apply consistency filter before blending (default: False).
                                   Currently only effective with 'laplacian' blend method.
        @param consistency_kernel: Kernel size for median filter if consistency_filter is True.
        @param postprocess: Apply contrast/brightness adjustment and sharpening (default: True).
        @param laplacian_levels: Number of levels for Laplacian pyramid blending.
        """
        print("Initializing FocusStacker...")
        self.align_method = align_method
        self.focus_measure_method = focus_measure_method
        self.blend_method = blend_method
        self.consistency_filter = consistency_filter
        self.consistency_kernel = consistency_kernel
        self.postprocess = postprocess
        self.laplacian_levels = laplacian_levels

        # Initialize color profiles (now handled in utils)
        utils.init_color_profiles()
        print(f"  Alignment: {self.align_method}")
        print(f"  Focus Measure: {self.focus_measure_method}")
        print(f"  Blending: {self.blend_method}")
        print(f"  Consistency Filter: {'Enabled (k=' + str(self.consistency_kernel) + ')' if self.consistency_filter else 'Disabled'}")
        print(f"  Post-processing: {'Enabled' if self.postprocess else 'Disabled'}")
        if self.blend_method == 'laplacian':
            print(f"  Laplacian Levels: {self.laplacian_levels}")


    def process_stack(self, image_paths, color_space='sRGB'):
        """
        Main processing pipeline for a single stack of images, using selected methods.

        @param image_paths: List of paths to the images in the stack.
        @param color_space: Target color space for the output (e.g., 'sRGB', 'AdobeRGB').
                            Conversion happens at the end if needed.
        @return: The final processed (stacked and optionally post-processed) image
                 as a float32 NumPy array [0, 1].
        """
        if not image_paths or len(image_paths) < 2:
            raise ValueError("Focus stacking requires at least 2 image paths.")

        print(f"\n--- Processing stack of {len(image_paths)} images ---")
        base_filenames = [os.path.basename(p) for p in image_paths]
        print(f"Images: {', '.join(base_filenames[:3])}{'...' if len(base_filenames) > 3 else ''}")

        # 1. Load images using utility function
        images = []
        for i, path in enumerate(image_paths):
            print(f"Loading image {i+1}/{len(image_paths)}: {os.path.basename(path)}")
            try:
                img = utils.load_image(path)
                images.append(img)
            except Exception as e:
                print(f"ERROR loading image {path}: {e}")
                raise # Re-raise critical error

        # 2. Align images using selected method
        print(f"\nAligning images using '{self.align_method}' method...")
        try:
            if self.align_method == 'orb':
                aligned_images = alignment.align_orb(images)
            elif self.align_method == 'ecc':
                aligned_images = alignment.align_ecc(images) # Placeholder
                print("Warning: ECC alignment is currently a placeholder.")
            else:
                print(f"Warning: Unknown alignment method '{self.align_method}'. Using original images.")
                aligned_images = images
            print(f"Alignment complete ({len(aligned_images)} images).")
        except Exception as e:
            print(f"ERROR during image alignment: {e}")
            raise

        # Store reference image for potential post-processing use
        reference_image = aligned_images[0] if aligned_images else None

        # 3. Calculate focus measures using selected method
        print(f"\nCalculating focus measures using '{self.focus_measure_method}' method...")
        focus_maps = []
        for i, img in enumerate(aligned_images):
            print(f"Calculating focus for image {i+1}/{len(aligned_images)}")
            try:
                if self.focus_measure_method == 'custom':
                    # Custom measure expects color image
                    focus_map = focus_measure.measure_custom(img)
                elif self.focus_measure_method == 'laplacian_variance':
                    # Laplacian variance expects grayscale
                    img_gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
                    # We need a map, not a single value. Apply variance in windows?
                    # For now, let's adapt the custom measure logic slightly or use a simpler map.
                    # Using simple absolute Laplacian as a map for now.
                    lap = cv2.Laplacian(img_gray, cv2.CV_32F, ksize=3)
                    focus_map = np.abs(lap)
                    # Normalize map
                    min_val, max_val = np.min(focus_map), np.max(focus_map)
                    if max_val > min_val: focus_map = (focus_map - min_val) / (max_val - min_val)
                    else: focus_map = np.zeros_like(focus_map)
                    print("  (Using absolute Laplacian map for 'laplacian_variance' method)")

                else:
                    print(f"Warning: Unknown focus measure method '{self.focus_measure_method}'. Skipping focus calculation.")
                    focus_map = np.ones(img.shape[:2], dtype=np.float32) * 0.5 # Default neutral map

                focus_maps.append(focus_map.astype(np.float32))
            except Exception as e:
                print(f"ERROR calculating focus measure for image {i+1}: {e}")
                raise
        print("Focus measure calculation complete.")

        # 4. Blend images using selected method
        print(f"\nBlending images using '{self.blend_method}' method...")
        try:
            if self.blend_method == 'weighted':
                blended_image = blending.blend_weighted(aligned_images, focus_maps)
            elif self.blend_method == 'laplacian':
                # First, determine the sharpest source index map from focus maps
                if not focus_maps:
                    raise ValueError("Focus maps are required for Laplacian blending.")
                focus_stack = np.stack(focus_maps, axis=-1)
                sharpest_indices = np.argmax(focus_stack, axis=-1)

                # Apply consistency filter if enabled
                if self.consistency_filter:
                    indices_to_blend = blending.apply_consistency_filter(sharpest_indices, self.consistency_kernel)
                else:
                    indices_to_blend = sharpest_indices # Use unfiltered indices

                # Pass the (potentially filtered) index map to the blending function
                # **Note:** Need to modify blend_laplacian_pyramid to accept indices_to_blend instead of focus_maps
                blended_image = blending.blend_laplacian_pyramid(aligned_images, indices_to_blend, levels=self.laplacian_levels)

            else:
                print(f"Warning: Unknown blend method '{self.blend_method}'. Averaging images.")
                blended_image = np.mean(np.stack(aligned_images, axis=0), axis=0).astype(np.float32)

            print("Blending complete.")
        except Exception as e:
            print(f"ERROR during image blending: {e}")
            raise

        # 5. Apply Post-processing if enabled
        if self.postprocess and reference_image is not None:
            try:
                final_result = postprocessing.apply_postprocessing(blended_image, reference_image, focus_maps)
            except Exception as e:
                print(f"ERROR during post-processing: {e}. Returning blended image without post-processing.")
                final_result = blended_image # Fallback to blended result
        else:
            print("\nSkipping post-processing.")
            final_result = blended_image # No post-processing applied

        # 6. Color space conversion (if needed) using utility function
        if color_space != 'sRGB':
            try:
                # Assume the result is currently in sRGB after processing
                final_result = utils.convert_color_space(final_result, target_space=color_space, source_space='sRGB')
            except Exception as e:
                print(f"ERROR during final color space conversion: {e}")
                # Decide if this is critical - maybe just warn? For now, raise.
                raise

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
    # _init_color_profiles
    # _load_image
    # _align_images
    # _focus_measure
    # _blend_images (core logic moved, post-processing moved)
    # _convert_color_space
