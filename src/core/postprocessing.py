#!/usr/bin/env python3

import cv2
import numpy as np
from scipy.fft import fft2, ifft2

# --- Sharpening Kernels ---
# Moved from focus_stacker.py

# Ultra-sharp kernel with extreme micro-detail preservation
_SHARP_KERNEL = np.array([[-4,-4,-4],
                         [-4, 33,-4],
                         [-4,-4,-4]], dtype=np.float32)

_HIGHFREQ_KERNEL = np.array([[-2,-3,-2], [-3, 25,-3], [-2,-3,-2]], dtype=np.float32)

# Enhanced edge detection with multi-directional sensitivity
# _EDGE_KERNEL = np.array([[-3,-3,-3,-3,-3],
#                        [-3, 4, 4, 4,-3],
#                        [-3, 4, 16, 4,-3],
#                        [-3, 4, 4, 4,-3],
#                        [-3,-3,-3,-3,-3]], dtype=np.float32) # Not used in sharpening

_DETAIL_KERNEL = np.array([[-1,-2,-1], [-2, 13,-2], [-1,-2,-1]], dtype=np.float32)


def apply_postprocessing(result_img, reference_img, focus_maps):
    """
    Applies post-processing steps: brightness/contrast matching and focus-aware sharpening.

    @param result_img: The initially blended image (float32 [0, 1]).
    @param reference_img: The reference image (usually the first aligned image)
                           for brightness/contrast matching (float32 [0, 1]).
    @param focus_maps: List of focus maps (float32 [0, 1]) used to create a
                       focus-aware sharpening mask.
    @return: Post-processed image (float32 [0, 1]).
    """
    print("\nApplying post-processing (Contrast/Brightness Adjust, Sharpening)...")
    result = result_img.copy() # Work on a copy

    # --- Preserve original brightness range ---
    try:
        ref_min = float(np.min(reference_img))
        ref_max = float(np.max(reference_img))
        ref_range = ref_max - ref_min

        result_min = float(np.min(result))
        result_max = float(np.max(result))

        # Avoid division by zero if result range is zero
        if (result_max - result_min) > 1e-6:
            result = (result - result_min) * (ref_range / (result_max - result_min)) + ref_min
        else:
            # If result is flat, set it to the reference mean? Or just leave it?
            # Setting to reference mean might be reasonable.
            result = np.full_like(result, float(np.mean(reference_img)))
            print("  Warning: Blended result has zero range. Setting to reference mean brightness.")

        # --- Gentle contrast enhancement per channel, preserving mean ---
        contrast_factor = 1.1 # How much to boost contrast
        for c in range(result.shape[2]): # Iterate through color channels
            channel_mean = float(np.mean(reference_img[...,c]))
            result[...,c] = (result[...,c] - channel_mean) * contrast_factor + channel_mean

        # Clip to original max (and ensure min is 0)
        result = np.clip(result, 0.0, ref_max)
        print("  Applied brightness and contrast adjustments.")

    except Exception as e:
        print(f"  Warning: Error during brightness/contrast adjustment: {e}. Skipping.")
        # Fallback to the unadjusted result if error occurs
        result = result_img.copy()


    # --- Focus-aware Sharpening ---
    try:
        # Compute focus-aware sharpening mask (sharpen less in out-of-focus areas)
        if focus_maps:
            focus_mask_sum = np.zeros_like(result[...,0], dtype=np.float32)
            valid_maps = 0
            for fm in focus_maps:
                if fm is not None and fm.shape == focus_mask_sum.shape:
                    # Ensure focus map is 2D
                    fm_2d = fm[..., 0] if len(fm.shape) > 2 else fm
                    focus_mask_sum += (1.0 - fm_2d) # Invert focus map (higher value = less focus)
                    valid_maps += 1
                else:
                     print(f"  Warning: Skipping invalid focus map in sharpening mask calculation.")

            if valid_maps > 0:
                focus_mask = focus_mask_sum / valid_maps
                # Limit sharpening strength: clip between 0.3 (max sharpen) and 1.0 (no sharpen)
                focus_mask = np.clip(focus_mask, 0.3, 1.0)
            else:
                print("  Warning: No valid focus maps provided for sharpening. Applying uniform sharpening.")
                focus_mask = np.full_like(result[...,0], 0.3, dtype=np.float32) # Uniform max sharpening
        else:
             print("  Warning: No focus maps provided for sharpening. Applying uniform sharpening.")
             focus_mask = np.full_like(result[...,0], 0.3, dtype=np.float32) # Uniform max sharpening


        # Apply multi-kernel sharpening using FFT
        sharp_result = np.zeros_like(result)
        h, w = result.shape[:2]

        # Pre-calculate FFT of kernels
        fft_sharp_kernel = fft2(_SHARP_KERNEL, s=(h, w))
        fft_highfreq_kernel = fft2(_HIGHFREQ_KERNEL, s=(h, w))
        fft_detail_kernel = fft2(_DETAIL_KERNEL, s=(h, w))

        for c in range(result.shape[2]): # Iterate through color channels
            channel = result[...,c]
            fft_channel = fft2(channel)

            # Calculate components using FFT convolution
            sharp = np.real(ifft2(fft_channel * fft_sharp_kernel))
            high_freq = np.real(ifft2(fft_channel * fft_highfreq_kernel))
            fine_detail = np.real(ifft2(fft_channel * fft_detail_kernel))

            # Calculate masks for adaptive sharpening
            # Local variance mask (more detail = more sharpening)
            mean_sq = cv2.GaussianBlur(channel**2, (11, 11), 0)
            sq_mean = cv2.GaussianBlur(channel, (11, 11), 0)**2
            local_var = mean_sq - sq_mean
            min_var = np.min(local_var)
            max_var = np.max(local_var)
            if max_var > min_var:
                 detail_mask = np.clip((local_var - min_var) / (max_var - min_var), 0.4, 1.0)
            else:
                 detail_mask = np.full_like(channel, 0.4) # Default if flat

            # Local contrast mask (more contrast = more sharpening)
            local_contrast_lap = cv2.Laplacian(channel, cv2.CV_32F, ksize=3)
            max_lap = np.max(np.abs(local_contrast_lap))
            if max_lap > 1e-6:
                contrast_mask = np.clip(np.abs(local_contrast_lap) / max_lap, 0.2, 0.6)
            else:
                contrast_mask = np.full_like(channel, 0.2) # Default if flat

            # Combine sharpening components adaptively
            # Sharpness strength modulated by focus, detail, and contrast
            sharp_strength = np.clip(focus_mask * (1.4 + 0.4 * detail_mask), 0.8, 0.99)

            # Add weighted sharpening components to the original channel
            sharp_result[...,c] = channel + \
                                 (sharp - channel) * sharp_strength * 0.5 * contrast_mask + \
                                 (high_freq - channel) * 0.2 * contrast_mask + \
                                 (fine_detail - channel) * 0.1 * contrast_mask
            # Note: Subtracting 'channel' from kernel results before multiplying
            # is equivalent to applying kernel to (img - mean) or similar,
            # preventing overall brightness shifts from sharpening.

        # Final clipping
        result_np = np.clip(sharp_result, 0.0, 1.0)
        print("  Applied focus-aware sharpening.")
        return result_np

    except Exception as e:
        print(f"  Warning: Error during sharpening: {e}. Returning contrast/brightness adjusted image.")
        return result # Return the contrast/brightness adjusted image if sharpening fails
