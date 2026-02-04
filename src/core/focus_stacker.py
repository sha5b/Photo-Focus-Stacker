#!/usr/bin/env python3

# Context: Focus stacking pipeline orchestrator
# Purpose: Load a stack, align frames, compute focus maps, blend, and optionally post-process.
# Notes: Used by the PyQt UI worker thread.

import cv2
import numpy as np
import os

from . import utils
from . import alignment
from . import focus_measure
from . import blending


def apply_unsharp_mask(image, strength=0.5, kernel_size=(5, 5), sigma=1.0):
    """Applies Unsharp Masking to enhance image sharpness."""
    if strength <= 0:
        return image

    # Ensure kernel size is odd
    k_h = kernel_size[0] if kernel_size[0] % 2 != 0 else kernel_size[0] + 1
    k_w = kernel_size[1] if kernel_size[1] % 2 != 0 else kernel_size[1] + 1
    kernel_size = (k_h, k_w)

    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = cv2.addWeighted(image, 1.0 + strength, blurred, -strength, 0)
    return np.clip(sharpened, 0.0, 1.0)

class StackingCancelledException(Exception):
    """Custom exception for cancelled stacking process."""
    pass

class FocusStacker:
    def __init__(self,
                 focus_window_size=7,
                 sharpen_strength=0.0,
                 num_pyramid_levels=3,
                 gradient_threshold=10,
                 blend_method='weighted'
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
        self.align_method_desc = f'Pyramid ECC Homography ({num_pyramid_levels} levels, Masked)'
        self.focus_measure_method_desc = f'Laplacian Variance Map (window={focus_window_size})'

        if blend_method not in ['weighted', 'direct_map', 'laplacian_pyramid', 'guided_weighted', 'luma_weighted_chroma_pick']:
            print(f"Warning: Invalid blend_method '{blend_method}'. Defaulting to 'weighted'.")
            self.blend_method = 'weighted'
        else:
            self.blend_method = blend_method
        if self.blend_method == 'weighted':
            self.blend_method_desc = 'Weighted'
        elif self.blend_method == 'direct_map':
            self.blend_method_desc = 'Direct Map Selection'
        elif self.blend_method == 'laplacian_pyramid':
            self.blend_method_desc = 'Laplacian Pyramid Fusion'
        elif self.blend_method == 'guided_weighted':
            self.blend_method_desc = 'Guided Weighted (Edge-Aware)'
        else:
            self.blend_method_desc = 'Luma Weighted + Chroma Pick (MFF)'

        self.focus_window_size = focus_window_size
        self.sharpen_strength = sharpen_strength
        self.num_pyramid_levels = num_pyramid_levels
        self.gradient_threshold = gradient_threshold
        self._stop_requested = False

        utils.init_color_profiles()
        print(f"  Alignment: {self.align_method_desc}")
        print(f"  Focus Measure: {self.focus_measure_method_desc}")
        print(f"  Blending: {self.blend_method_desc}")
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

    def _compute_focus_map(self, img_rgb_f32: np.ndarray) -> np.ndarray:
        h, w = img_rgb_f32.shape[:2]
        max_dim = int(max(h, w))

        img_gray = cv2.cvtColor((img_rgb_f32 * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)

        target_max_dim = 1200
        if max_dim > target_max_dim:
            scale = float(target_max_dim) / float(max_dim)
            new_w = max(int(round(w * scale)), 2)
            new_h = max(int(round(h * scale)), 2)
            img_gray_small = cv2.resize(img_gray, (new_w, new_h), interpolation=cv2.INTER_AREA)
            window_small = int(max(3, round(self.focus_window_size * scale)))
            if window_small % 2 == 0:
                window_small += 1
            focus_small = focus_measure.measure_laplacian_variance_map(
                img_gray_small,
                window_size=window_small,
                normalize=False,
            ).astype(np.float32)
            focus_map = cv2.resize(focus_small, (w, h), interpolation=cv2.INTER_LINEAR)
            return focus_map.astype(np.float32)

        return focus_measure.measure_laplacian_variance_map(
            img_gray,
            window_size=self.focus_window_size,
            normalize=False,
        ).astype(np.float32)

    def process_stack(self, image_paths, color_space='sRGB'):
        """
        Main processing pipeline for a single stack of images, using simplified methods.

        @param image_paths: List of paths to the images in the stack.
        @param color_space: Target color space for the output (e.g., 'sRGB', 'AdobeRGB').
                            Conversion happens at the end if needed.
        @return: The final processed (stacked and sharpened) image as a float32 NumPy array [0, 1].
        """
        self._stop_requested = False
        if not image_paths or len(image_paths) < 2:
            raise ValueError("Focus stacking requires at least 2 image paths.")

        print(f"\n--- Processing stack of {len(image_paths)} images ---")
        base_filenames = [os.path.basename(p) for p in image_paths]
        print(f"Images: {', '.join(base_filenames[:3])}{'...' if len(base_filenames) > 3 else ''}")

        # 1. Load images using utility function
        images = []
        for i, path in enumerate(image_paths):
            self._check_stop_requested()
            print(f"Loading image {i+1}/{len(image_paths)}: {os.path.basename(path)}")
            try:
                img = utils.load_image(path)
                images.append(img)
            except Exception as e:
                print(f"ERROR loading image {path}: {e}")
                raise

        # 2. Align images using Pyramid ECC Homography with Masking
        self._check_stop_requested()
        print(f"\nAligning images using {self.align_method_desc}...")
        try:
            aligned_images = alignment.align_images(
                images,
                num_pyramid_levels=self.num_pyramid_levels,
                gradient_threshold=self.gradient_threshold
            )
            print(f"Alignment complete ({len(aligned_images)} images).")
        except Exception as e:
            if not isinstance(e, StackingCancelledException):
                print(f"ERROR during image alignment: {e}")
            raise

        # 3. Calculate focus measures using Laplacian Variance Map
        print(f"\nCalculating focus measures using {self.focus_measure_method_desc}...")
        focus_maps = []
        for i, img in enumerate(aligned_images):
            self._check_stop_requested()
            print(f"Calculating focus for image {i+1}/{len(aligned_images)}")
            try:
                focus_maps.append(self._compute_focus_map(img))
            except Exception as e:
                 if not isinstance(e, StackingCancelledException):
                    print(f"ERROR calculating focus measure for image {i+1}: {e}")
                 raise
        print("Focus measure calculation complete.")

        # 4. Determine sharpest indices (needed for direct map blending)
        self._check_stop_requested()
        sharpest_indices = None
        direct_map_focus_maps = None
        if self.blend_method == 'direct_map':
            print("\nCalculating sharpest image indices...")
            if focus_maps:
                smoothed_maps = [
                    cv2.GaussianBlur(fm.astype(np.float32), (0, 0), sigmaX=1.0, sigmaY=1.0, borderType=cv2.BORDER_REFLECT)
                    for fm in focus_maps
                ]
                direct_map_focus_maps = smoothed_maps
                focus_maps_stack = np.stack(smoothed_maps, axis=0)
                sharpest_indices = np.argmax(focus_maps_stack, axis=0).astype(np.uint16)
                print("Sharpest indices calculated.")
            else:
                print("Warning: No focus maps available to calculate sharpest indices.")
                raise ValueError("Cannot perform direct map blending without focus maps.")


        # 5. Blend images using the selected method
        self._check_stop_requested()
        print(f"\nBlending images using {self.blend_method_desc}...")
        try:
            if self.blend_method == 'weighted':
                blended_image = blending.blend_weighted(aligned_images, focus_maps)
            elif self.blend_method == 'direct_map':
                if sharpest_indices is None:
                      raise ValueError("Sharpest indices map is required for direct map blending but was not calculated.")
                blended_image = blending.blend_direct_map(aligned_images, sharpest_indices, focus_maps=direct_map_focus_maps)
            elif self.blend_method == 'guided_weighted':
                blended_image = blending.blend_guided_weighted(aligned_images, focus_maps)
            elif self.blend_method == 'luma_weighted_chroma_pick':
                blended_image = blending.blend_luma_weighted_chroma_pick(aligned_images, focus_maps)
            elif self.blend_method == 'laplacian_pyramid':
                blended_image = blending.blend_laplacian_pyramid(
                    aligned_images,
                    focus_maps,
                    num_levels=self.num_pyramid_levels,
                )
            else:
                raise ValueError(f"Unsupported blend method: {self.blend_method}")

            print("Blending complete.")
        except Exception as e:
            if not isinstance(e, StackingCancelledException):
                print(f"ERROR during image blending: {e}")
            raise

        # 6. Apply Sharpening (if strength > 0)
        self._check_stop_requested()
        if self.sharpen_strength > 0:
            try:
                final_result = apply_unsharp_mask(blended_image, strength=self.sharpen_strength)
            except Exception as e:
                 if not isinstance(e, StackingCancelledException):
                    print(f"ERROR during sharpening: {e}. Returning blended image without sharpening.")
                 final_result = blended_image
        else:
            print("\nSkipping sharpening.")
            final_result = blended_image

        # 7. Color space conversion (if needed) using utility function
        self._check_stop_requested()
        if color_space != 'sRGB':
            try:
                final_result = utils.convert_color_space(final_result, target_space=color_space, source_space='sRGB')
            except Exception as e:
                 if not isinstance(e, StackingCancelledException):
                    print(f"ERROR during final color space conversion: {e}")
                 raise

        self._check_stop_requested()
        print("\n--- Stack processing complete! ---")
        return final_result

    def save_image(self, img, path, format='JPEG', quality=95, color_space='sRGB'):
        """Saves the image using the utility function."""
        utils.save_image(img, path, format=format, quality=quality, color_space=color_space)

    def split_into_stacks(self, image_paths, stack_size=0):
        """Splits image paths into stacks using the utility function."""
        return utils.split_into_stacks(image_paths, stack_size=stack_size)
