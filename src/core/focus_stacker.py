#!/usr/bin/env python3

# Context: Focus stacking pipeline orchestrator
# Purpose: Load a stack, align frames, compute focus maps, blend, and optionally post-process.
# Notes: Used by the PyQt UI worker thread.

import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np

from . import alignment, blending, focus_measure, postprocessing, utils


class StackingCancelledException(Exception):
    """Custom exception for cancelled stacking process."""


def _largest_common_rectangle(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """Find a large, guaranteed-valid rectangle inside a mostly convex warp mask."""
    valid = np.asarray(mask, dtype=bool)
    rows = np.flatnonzero(np.any(valid, axis=1))
    if rows.size == 0:
        return None
    first = np.argmax(valid, axis=1)
    last = valid.shape[1] - np.argmax(valid[:, ::-1], axis=1)
    best = None
    best_area = 0
    y_min, y_max = int(rows[0]), int(rows[-1]) + 1
    candidates = np.unique(np.linspace(y_min, y_max, 25, dtype=int))
    for y0 in candidates[:-1]:
        for y1 in candidates[1:]:
            if y1 <= y0 or not np.all(np.any(valid[y0:y1], axis=1)):
                continue
            x0 = int(np.max(first[y0:y1]))
            x1 = int(np.min(last[y0:y1]))
            area = (y1 - y0) * max(0, x1 - x0)
            if area > best_area and np.all(valid[y0:y1, x0:x1]):
                best = (y0, y1, x0, x1)
                best_area = area
    return best


class FocusStacker:
    def __init__(self,
                 focus_window_size=7,
                 focus_measure_method='laplacian_var',
                 sharpen_strength=0.0,
                 num_pyramid_levels=3,
                 gradient_threshold=10,
                 blend_method='weighted',
                 alignment_model='affine',
                 crop_to_common_area=True,
                 linear_light_blending=True,
                 normalize_exposure=True,
                 focus_analysis_max_dim=0,
                 cache_memory_limit_mb=512,
                 focus_workers=2,
                 photogrammetry_mode=False,
                 progress_callback=None,
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
        requested_alignment_model = 'euclidean' if photogrammetry_mode else alignment_model
        self.align_method_desc = f'Pyramid ECC {requested_alignment_model.title()} ({num_pyramid_levels} levels, Masked)'
        if focus_measure_method not in ['laplacian_var', 'tenengrad', 'sml', 'multiscale']:
            focus_measure_method = 'laplacian_var'
        self.focus_measure_method = focus_measure_method

        if self.focus_measure_method == 'tenengrad':
            self.focus_measure_method_desc = f'Tenengrad Map (window={focus_window_size})'
        elif self.focus_measure_method == 'sml':
            self.focus_measure_method_desc = f'SML Map (window={focus_window_size})'
        elif self.focus_measure_method == 'multiscale':
            self.focus_measure_method_desc = f'Multi-scale Focus Map (window={focus_window_size})'
        else:
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
        self.photogrammetry_mode = bool(photogrammetry_mode)
        self.alignment_model = 'euclidean' if self.photogrammetry_mode else alignment_model
        self.crop_to_common_area = bool(crop_to_common_area) and not self.photogrammetry_mode
        self.linear_light_blending = bool(linear_light_blending)
        self.normalize_exposure = bool(normalize_exposure)
        self.focus_analysis_max_dim = max(0, int(focus_analysis_max_dim))
        self.cache_memory_limit_mb = max(0, int(cache_memory_limit_mb))
        self.focus_workers = max(1, int(focus_workers))
        self._stop_requested = False
        self.output_metadata = None
        self.depth_map = None
        self.confidence_map = None
        self.alignment_diagnostics = []
        self.source_indices = []
        self.source_paths = []
        self.progress_callback = progress_callback

        utils.init_color_profiles()
        print(f"  Alignment: {self.align_method_desc}")
        print(f"  Focus Measure: {self.focus_measure_method_desc}")
        print(f"  Blending: {self.blend_method_desc}")
        print(f"  Sharpen Strength: {self.sharpen_strength:.2f}")

    _intermediate_cache = {}
    _intermediate_cache_order = []
    _intermediate_cache_bytes = 0

    def request_stop(self):
        """Sets the flag to stop processing."""
        print("Stop requested for FocusStacker instance.")
        self._stop_requested = True

    def _check_stop_requested(self):
        """Checks if stop was requested and raises exception if so."""
        if self._stop_requested:
            print("Stop request detected during processing.")
            raise StackingCancelledException("Stacking process cancelled by user.")

    def _report_progress(self, value: int) -> None:
        if self.progress_callback is not None:
            self.progress_callback(max(0, min(100, int(value))))

    def _compute_focus_map(self, img_rgb_f32: np.ndarray) -> np.ndarray:
        h, w = img_rgb_f32.shape[:2]
        max_dim = int(max(h, w))

        img_gray = cv2.cvtColor(img_rgb_f32.astype(np.float32), cv2.COLOR_RGB2GRAY)

        target_max_dim = int(self.focus_analysis_max_dim)
        if target_max_dim > 0 and max_dim > target_max_dim:
            scale = float(target_max_dim) / float(max_dim)
            new_w = max(int(round(w * scale)), 2)
            new_h = max(int(round(h * scale)), 2)
            img_gray_small = cv2.resize(img_gray, (new_w, new_h), interpolation=cv2.INTER_AREA)
            window_small = int(max(3, round(self.focus_window_size * scale)))
            if window_small % 2 == 0:
                window_small += 1
            focus_small = self._measure_focus_gray(img_gray_small, window_small)
            focus_map = cv2.resize(focus_small, (w, h), interpolation=cv2.INTER_LINEAR)
            return focus_map.astype(np.float32)

        if max_dim > 2048:
            return self._compute_focus_map_tiled(img_gray)
        return self._measure_focus_gray(img_gray, self.focus_window_size)

    def _measure_focus_gray(self, img_gray: np.ndarray, window_size: int) -> np.ndarray:
        if self.focus_measure_method == 'multiscale':
            return focus_measure.measure_multiscale_map(img_gray, window_size=window_size, normalize=False).astype(np.float32)
        if self.focus_measure_method == 'tenengrad':
            return focus_measure.measure_tenengrad_map(img_gray, window_size=window_size, normalize=False).astype(np.float32)
        if self.focus_measure_method == 'sml':
            return focus_measure.measure_sml_map(img_gray, window_size=window_size, normalize=False).astype(np.float32)
        return focus_measure.measure_laplacian_variance_map(img_gray, window_size=window_size, normalize=False).astype(np.float32)

    def _compute_focus_map_tiled(self, img_gray: np.ndarray) -> np.ndarray:
        """Evaluate full-resolution focus evidence with overlap-safe bounded tiles."""
        h, w = img_gray.shape[:2]
        result = np.empty((h, w), dtype=np.float32)
        tile_size = 1024
        overlap = max(24, self.focus_window_size * 2)
        for y0 in range(0, h, tile_size):
            for x0 in range(0, w, tile_size):
                self._check_stop_requested()
                y1, x1 = min(y0 + tile_size, h), min(x0 + tile_size, w)
                sy0, sx0 = max(0, y0 - overlap), max(0, x0 - overlap)
                sy1, sx1 = min(h, y1 + overlap), min(w, x1 + overlap)
                measured = self._measure_focus_gray(
                    img_gray[sy0:sy1, sx0:sx1], self.focus_window_size
                )
                result[y0:y1, x0:x1] = measured[
                    y0 - sy0:y1 - sy0, x0 - sx0:x1 - sx0
                ]
        return result

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
        self.source_paths = [os.path.abspath(path) for path in image_paths]

        print(f"\n--- Processing stack of {len(image_paths)} images ---")
        base_filenames = [os.path.basename(p) for p in image_paths]
        print(f"Images: {', '.join(base_filenames[:3])}{'...' if len(base_filenames) > 3 else ''}")

        file_signatures = tuple(
            (os.path.abspath(path), os.path.getsize(path), os.stat(path).st_mtime_ns)
            for path in image_paths
        )
        cache_key = (
            file_signatures,
            int(self.num_pyramid_levels),
            int(self.gradient_threshold),
            int(self.focus_window_size),
            str(self.focus_measure_method),
            int(self.focus_analysis_max_dim),
            str(self.alignment_model),
            bool(self.crop_to_common_area),
            bool(self.normalize_exposure),
        )

        active_cache_limit = self.cache_memory_limit_mb * 1024 * 1024
        while (FocusStacker._intermediate_cache_bytes > active_cache_limit
               and FocusStacker._intermediate_cache_order):
            oldest, oldest_bytes = FocusStacker._intermediate_cache_order.pop(0)
            FocusStacker._intermediate_cache.pop(oldest, None)
            FocusStacker._intermediate_cache_bytes -= oldest_bytes

        cached = FocusStacker._intermediate_cache.get(cache_key)
        cache_backed = cached is not None
        if cached is not None:
            (aligned_images, focus_maps, valid_masks, self.alignment_diagnostics,
             self.output_metadata, self.source_indices) = cached
            print("\nUsing cached alignment + focus maps...")
        else:
            images = []
            for i, path in enumerate(image_paths):
                self._check_stop_requested()
                print(f"Loading image {i+1}/{len(image_paths)}: {os.path.basename(path)}")
                try:
                    loaded = utils.load_image_with_metadata(path)
                    images.append(loaded.image)
                    if i == len(image_paths) // 2:
                        self.output_metadata = loaded.metadata
                except Exception as e:
                    print(f"ERROR loading image {path}: {e}")
                    raise
                self._report_progress(15 * (i + 1) // len(image_paths))

            self._check_stop_requested()
            print(f"\nAligning images using {self.align_method_desc}...")
            try:
                alignment_result = alignment.align_images_detailed(
                    images,
                    num_pyramid_levels=self.num_pyramid_levels,
                    gradient_threshold=self.gradient_threshold,
                    motion_model=self.alignment_model,
                    reference_index=len(images) // 2,
                    failure_policy='exclude',
                    cancel_check=self._check_stop_requested,
                    release_sources=True,
                    progress_callback=lambda done, total: self._report_progress(15 + 40 * done // total),
                )
                aligned_images = alignment_result.images
                valid_masks = alignment_result.valid_masks
                self.alignment_diagnostics = alignment_result.diagnostics
                self.source_indices = alignment_result.source_indices
                del images
                rejected = [d.source_index for d in self.alignment_diagnostics if not d.accepted]
                if rejected:
                    print(f"Excluded {len(rejected)} frame(s) after alignment quality checks: {rejected}")
                print(f"Alignment complete ({len(aligned_images)} images).")
                if self.normalize_exposure:
                    reference_aligned_index = self.source_indices.index(len(image_paths) // 2)
                    aligned_images = utils.match_stack_exposure(
                        aligned_images, valid_masks, reference_aligned_index
                    )
            except Exception as e:
                if not isinstance(e, StackingCancelledException):
                    print(f"ERROR during image alignment: {e}")
                raise

            print(f"\nCalculating focus measures using {self.focus_measure_method_desc}...")
            focus_maps = [None] * len(aligned_images)
            completed = 0
            with ThreadPoolExecutor(max_workers=min(self.focus_workers, len(aligned_images))) as executor:
                futures = {
                    executor.submit(self._compute_focus_map, image): index
                    for index, image in enumerate(aligned_images)
                }
                for future in as_completed(futures):
                    index = futures[future]
                    self._check_stop_requested()
                    try:
                        focus_maps[index] = future.result()
                    except Exception as e:
                        if not isinstance(e, StackingCancelledException):
                            print(f"ERROR calculating focus measure for image {index + 1}: {e}")
                        raise
                    completed += 1
                    self._report_progress(55 + 20 * completed // len(aligned_images))
            print("Focus measure calculation complete.")

            entry_bytes = sum(a.nbytes for a in aligned_images + focus_maps + valid_masks)
            limit_bytes = self.cache_memory_limit_mb * 1024 * 1024
            if limit_bytes and entry_bytes <= limit_bytes:
                FocusStacker._intermediate_cache[cache_key] = (
                    aligned_images, focus_maps, valid_masks, self.alignment_diagnostics,
                    self.output_metadata, self.source_indices
                )
                FocusStacker._intermediate_cache_order.append((cache_key, entry_bytes))
                FocusStacker._intermediate_cache_bytes += entry_bytes
                cache_backed = True
                while FocusStacker._intermediate_cache_bytes > limit_bytes and FocusStacker._intermediate_cache_order:
                    oldest, oldest_bytes = FocusStacker._intermediate_cache_order.pop(0)
                    FocusStacker._intermediate_cache.pop(oldest, None)
                    FocusStacker._intermediate_cache_bytes -= oldest_bytes

        if self.crop_to_common_area:
            common = np.logical_and.reduce([mask > 0.5 for mask in valid_masks])
            common_rectangle = _largest_common_rectangle(common)
            if common_rectangle is not None:
                y0, y1, x0, x1 = common_rectangle
                aligned_images = [img[y0:y1, x0:x1] for img in aligned_images]
                focus_maps = [fm[y0:y1, x0:x1] for fm in focus_maps]
                valid_masks = [mask[y0:y1, x0:x1] for mask in valid_masks]

        blend_depth_map, self.confidence_map = blending.build_focus_depth_map(
            focus_maps, valid_masks=valid_masks,
            min_region_area=max(16, self.focus_window_size * self.focus_window_size),
        )
        source_lookup = np.asarray(self.source_indices, dtype=np.uint16)
        self.depth_map = source_lookup[blend_depth_map]

        # 4. Determine sharpest indices (needed for direct map blending)
        self._check_stop_requested()
        self._report_progress(80)
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
                sharpest_indices = blend_depth_map
                print("Sharpest indices calculated.")
            else:
                print("Warning: No focus maps available to calculate sharpest indices.")
                raise ValueError("Cannot perform direct map blending without focus maps.")


        # 5. Blend images using the selected method
        self._check_stop_requested()
        print(f"\nBlending images using {self.blend_method_desc}...")
        try:
            if self.linear_light_blending:
                if cache_backed:
                    blend_images = [utils.srgb_to_linear(image) for image in aligned_images]
                else:
                    for image_index in range(len(aligned_images)):
                        aligned_images[image_index] = utils.srgb_to_linear(aligned_images[image_index])
                    blend_images = aligned_images
            else:
                blend_images = aligned_images
            if self.blend_method == 'weighted':
                blended_image = blending.blend_weighted(
                    blend_images, focus_maps, valid_masks=valid_masks,
                    depth_map=blend_depth_map, confidence_map=self.confidence_map,
                )
            elif self.blend_method == 'direct_map':
                if sharpest_indices is None:
                      raise ValueError("Sharpest indices map is required for direct map blending but was not calculated.")
                blended_image = blending.blend_direct_map(blend_images, sharpest_indices, focus_maps=direct_map_focus_maps, valid_masks=valid_masks)
            elif self.blend_method == 'guided_weighted':
                blended_image = blending.blend_guided_weighted(
                    blend_images, focus_maps, valid_masks=valid_masks,
                    depth_map=blend_depth_map, confidence_map=self.confidence_map,
                )
            elif self.blend_method == 'luma_weighted_chroma_pick':
                blended_image = blending.blend_luma_weighted_chroma_pick(
                    blend_images, focus_maps, valid_masks=valid_masks,
                    depth_map=blend_depth_map, confidence_map=self.confidence_map,
                )
            elif self.blend_method == 'laplacian_pyramid':
                blended_image = blending.blend_laplacian_pyramid(
                    blend_images,
                    focus_maps,
                    num_levels=self.num_pyramid_levels,
                    valid_masks=valid_masks,
                )
            else:
                raise ValueError(f"Unsupported blend method: {self.blend_method}")

            if self.linear_light_blending:
                blended_image = utils.linear_to_srgb(blended_image)
            print("Blending complete.")
        except Exception as e:
            if not isinstance(e, StackingCancelledException):
                print(f"ERROR during image blending: {e}")
            raise

        # 6. Apply Sharpening (if strength > 0)
        self._check_stop_requested()
        if self.sharpen_strength > 0:
            try:
                final_result = postprocessing.apply_unsharp_mask(blended_image, strength=self.sharpen_strength)
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
        self._report_progress(100)
        return final_result

    def save_image(self, img, path, format='JPEG', quality=95, color_space='sRGB', bit_depth=None):
        """Saves the image using the utility function."""
        utils.save_image(img, path, format=format, quality=quality, color_space=color_space,
                         bit_depth=bit_depth, metadata=self.output_metadata)

    def split_into_stacks(self, image_paths, stack_size=0):
        """Splits image paths into stacks using the utility function."""
        return utils.split_into_stacks(image_paths, stack_size=stack_size)
