#!/usr/bin/env python3

import cv2
import numpy as np
import os

def _build_gaussian_pyramid(img, levels):
    """Builds a Gaussian pyramid."""
    pyramid = [img]
    for _ in range(levels - 1):
        img = cv2.pyrDown(img, borderType=cv2.BORDER_REFLECT)
        if img is None or img.shape[0] < 2 or img.shape[1] < 2:
            print(f"      Warning: Pyramid level generation stopped early due to small image size.")
            break
        pyramid.append(img)
    return pyramid

# Only alignment method: Pyramid ECC Homography with Masking
def align_images(images, num_pyramid_levels=3, max_iterations=100, epsilon=1e-5, gradient_threshold=10):
    """
    Align images using pyramid-based ECC (Enhanced Correlation Coefficient)
    with HOMOGRAPHY motion model and an input mask derived from reference image gradients.
    Aligns all images to the first image in the list.

    @param images: List of images (as float32 NumPy arrays [0, 1]) to align.
    @param num_pyramid_levels: Number of pyramid levels to use (e.g., 3 or 4).
    @param max_iterations: Maximum number of iterations for ECC algorithm per level.
    @param epsilon: Termination threshold for ECC algorithm per level.
    @param gradient_threshold: Threshold for creating the gradient mask. Lower values include more pixels.
    @return: List of aligned images (float32 NumPy arrays [0, 1]).
    """
    if not images:
        return []
    if len(images) == 1:
        return images # No alignment needed

    motion_type = cv2.MOTION_HOMOGRAPHY
    motion_type_str = 'HOMOGRAPHY'

    print(f"\nAligning images using Pyramid ECC (Motion: {motion_type_str}, Levels: {num_pyramid_levels}, Masked)...")
    reference_color = images[0]
    aligned_color = [reference_color] # Start with the reference image (color)
    h_full, w_full = reference_color.shape[:2]

    # Convert reference image to grayscale uint8
    try:
        ref_gray_full = cv2.cvtColor((reference_color * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    except cv2.error as e:
        print(f"Error converting reference image to grayscale: {e}")
        print("Cannot proceed with ECC alignment.")
        return images # Return original images

    # --- Create Gradient Mask from Reference Image ---
    print("  Creating gradient mask from reference image...")
    try:
        # Calculate Sobel gradients
        grad_x = cv2.Sobel(ref_gray_full, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(ref_gray_full, cv2.CV_64F, 0, 1, ksize=3)
        # Calculate magnitude
        gradient_magnitude = cv2.magnitude(grad_x, grad_y)
        # Threshold the magnitude to create a binary mask
        # Pixels with gradient magnitude above the threshold are considered for ECC
        _, mask_full = cv2.threshold(gradient_magnitude, gradient_threshold, 255, cv2.THRESH_BINARY)
        mask_full = mask_full.astype(np.uint8) # Convert to uint8 for ECC and pyramid building
        # Optional: Dilate the mask slightly to ensure edges are included
        mask_full = cv2.dilate(mask_full, None, iterations=2)
        print(f"  Gradient mask created with threshold {gradient_threshold}.")
    except Exception as e:
        print(f"  Warning: Failed to create gradient mask: {e}. Proceeding without mask.")
        mask_full = None
    # -----------------------------------------------

    # Build reference pyramid (grayscale)
    ref_pyramid = _build_gaussian_pyramid(ref_gray_full, num_pyramid_levels)
    actual_levels = len(ref_pyramid)
    print(f"  Reference pyramid built with {actual_levels} levels.")

    # Build mask pyramid if mask was created successfully
    mask_pyramid = None
    if mask_full is not None:
        mask_pyramid = _build_gaussian_pyramid(mask_full, actual_levels)
        # Ensure mask pyramid has same number of levels (important if pyramid stopped early)
        if len(mask_pyramid) != actual_levels:
            print("  Warning: Mask pyramid level count mismatch. Disabling mask.")
            mask_pyramid = None
        else:
             print(f"  Mask pyramid built with {actual_levels} levels.")


    # Define termination criteria for ECC
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, max_iterations, epsilon)

    # Align subsequent images to the reference
    for i, img_color in enumerate(images[1:], 1):
        print(f"\nAligning image {i+1}/{len(images)} using Pyramid ECC...")

        try:
            # Convert current image to grayscale uint8
            img_gray_full = cv2.cvtColor((img_color * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)

            # Build image pyramid
            img_pyramid = _build_gaussian_pyramid(img_gray_full, actual_levels)
            if len(img_pyramid) != actual_levels:
                 print(f"  Warning: Image {i+1} pyramid has different levels ({len(img_pyramid)}) than reference ({actual_levels}). Using original image.")
                 aligned_color.append(img_color)
                 continue

            # Initialize warp matrix (Homography is 3x3)
            warp_matrix = np.eye(3, 3, dtype=np.float32)

            # Iterate down the pyramid (from smallest level to largest)
            for level in range(actual_levels - 1, -1, -1):
                print(f"  Processing pyramid level {level} (shape: {ref_pyramid[level].shape})...")
                ref_level = ref_pyramid[level]
                img_level = img_pyramid[level]
                mask_level = mask_pyramid[level] if mask_pyramid else None # Get mask for this level

                # Scale warp matrix from previous level if not the smallest level
                if level < actual_levels - 1:
                    pass # Using the matrix from finer level directly as initial guess for Homography

                # Run ECC algorithm for the current level, potentially with mask
                try:
                    # Pass mask_level to inputMask parameter
                    (cc, warp_matrix_level) = cv2.findTransformECC(ref_level, img_level, warp_matrix, motion_type, criteria, inputMask=mask_level, gaussFiltSize=5)
                    warp_matrix = warp_matrix_level
                    print(f"    ECC finished for level {level}. Correlation: {cc:.4f}")
                except cv2.error as ecc_error:
                     print(f"    Warning: findTransformECC failed for image {i+1} at level {level}: {ecc_error}. Using original image.")
                     warp_matrix = None # Signal failure
                     break # Stop processing levels for this image

            if warp_matrix is None: # Check if ECC failed at any level
                aligned_color.append(img_color)
                continue

            # Warp the original COLOR image using the final transformation matrix
            aligned_img = cv2.warpPerspective(img_color, warp_matrix, (w_full, h_full), flags=cv2.INTER_LANCZOS4 + cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REFLECT)

            aligned_color.append(aligned_img.astype(np.float32)) # Ensure output is float32
            print(f"  Aligned image {i+1} using Pyramid ECC.")

        except cv2.error as e:
            print(f"OpenCV Error aligning image {i+1} with Pyramid ECC: {str(e)}")
            print("  Using original image as fallback.")
            aligned_color.append(img_color)
        except Exception as e:
            print(f"Unexpected Error aligning image {i+1} with Pyramid ECC: {str(e)}")
            print("  Using original image as fallback.")
            aligned_color.append(img_color)

    print(f"\nPyramid ECC Alignment complete. Returning {len(aligned_color)} images.")
    return aligned_color
