#!/usr/bin/env python3

import cv2
import numpy as np
import os

def align_orb(images):
    """
    Align images using ORB feature matching and Homography.
    Aligns all images to the first image in the list.

    @param images: List of images (as float32 NumPy arrays [0, 1]) to align.
    @return: List of aligned images (float32 NumPy arrays [0, 1]).
    """
    if not images:
        return []
    if len(images) == 1:
        return images # No alignment needed for a single image

    print("\nAligning images using ORB features...")
    reference = images[0]
    aligned = [reference] # Start with the reference image

    # Convert reference image to grayscale uint8 for feature detection
    try:
        ref_gray = cv2.cvtColor((reference * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        h, w = ref_gray.shape
    except cv2.error as e:
        print(f"Error converting reference image to grayscale: {e}")
        print("Cannot proceed with alignment.")
        return images # Return original images if conversion fails

    # Initialize ORB detector
    try:
        orb = cv2.ORB_create(nfeatures=2000, scaleFactor=1.2, nlevels=8, edgeThreshold=15, patchSize=31)
    except cv2.error as e:
        print(f"Error creating ORB detector: {e}. Check OpenCV installation.")
        print("Cannot proceed with alignment.")
        return images

    # Find keypoints and descriptors in the reference image
    try:
        kp_ref, des_ref = orb.detectAndCompute(ref_gray, None)
        if des_ref is None or len(kp_ref) == 0:
            print("Warning: No descriptors found in reference image. Skipping alignment for subsequent images.")
            # If reference has no features, we can't align others to it. Return originals.
            return images
    except cv2.error as e:
        print(f"Error detecting features in reference image: {e}")
        print("Cannot proceed with alignment.")
        return images

    # Initialize Brute-Force Matcher
    try:
        # Use crossCheck=False for KNN matching + ratio test
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    except cv2.error as e:
        print(f"Error creating BFMatcher: {e}. Check OpenCV installation.")
        print("Cannot proceed with alignment.")
        return images

    # Align subsequent images to the reference
    for i, img in enumerate(images[1:], 1):
        print(f"Aligning image {i+1}/{len(images)}...")

        try:
            # Convert current image to grayscale uint8
            img_gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)

            # Find keypoints and descriptors in the current image
            kp_img, des_img = orb.detectAndCompute(img_gray, None)

            if des_img is None or len(kp_img) < 4:
                print(f"Warning: Not enough descriptors found in image {i+1}. Using original image.")
                aligned.append(img)
                continue

            # Match descriptors using KNN
            matches_knn = bf.knnMatch(des_ref, des_img, k=2)

            # Apply Lowe's ratio test to filter good matches
            good_matches = []
            ratio_thresh = 0.75 # Standard threshold
            for m, n in matches_knn:
                # Ensure m and n are valid matches (knnMatch can return fewer than k)
                if m is not None and n is not None and hasattr(m, 'distance') and hasattr(n, 'distance'):
                    if m.distance < ratio_thresh * n.distance:
                        good_matches.append(m)

            print(f"Found {len(good_matches)} good matches (ratio={ratio_thresh}) for image {i+1}.")

            # Need at least min_match_count good matches to find homography reliably
            min_match_count = 10 # Minimum match count for robustness
            if len(good_matches) >= min_match_count:
                # Extract location of good matches
                # Ensure indices are valid before accessing keypoints
                src_pts_list = []
                dst_pts_list = []
                # Use the filtered good_matches list
                for m in good_matches:
                    if m.queryIdx < len(kp_ref) and m.trainIdx < len(kp_img):
                         src_pts_list.append(kp_ref[m.queryIdx].pt)
                         dst_pts_list.append(kp_img[m.trainIdx].pt)

                if len(src_pts_list) < min_match_count:
                    print(f"Warning: Not enough valid matches ({len(src_pts_list)}/{min_match_count}) after index check for image {i+1}. Using original image.")
                    aligned.append(img)
                    continue

                src_pts = np.float32(src_pts_list).reshape(-1, 1, 2)
                dst_pts = np.float32(dst_pts_list).reshape(-1, 1, 2)

                # Find homography matrix using RANSAC
                M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)

                if M is None:
                     print(f"Warning: Homography calculation failed for image {i+1} (findHomography returned None). Using original image.")
                     aligned.append(img)
                     continue

                # Warp the current image to align with the reference image
                # Use BORDER_REFLECT to handle edges better than default black
                aligned_img = cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
                aligned.append(aligned_img)
                print(f"Aligned image {i+1} using Homography.")

            else:
                print(f"Warning: Not enough good matches found ({len(good_matches)}/{min_match_count}) for image {i+1}. Using original image.")
                aligned.append(img)

        except cv2.error as e:
            print(f"OpenCV Error aligning image {i+1}: {str(e)}")
            print("Using original image as fallback.")
            aligned.append(img)
        except Exception as e:
            print(f"Unexpected Error aligning image {i+1}: {str(e)}")
            print("Using original image as fallback.")
            aligned.append(img)

    print(f"Alignment complete. Returning {len(aligned)} images.")
    return aligned

# --- ECC Alignment ---
def align_ecc(images, motion_type_str='AFFINE', max_iterations=100, epsilon=1e-5):
    """
    Align images using ECC (Enhanced Correlation Coefficient).
    Aligns all images to the first image in the list.

    @param images: List of images (as float32 NumPy arrays [0, 1]) to align.
    @param motion_type_str: String representing the motion model ('AFFINE', 'HOMOGRAPHY', 'TRANSLATION'). Default: 'AFFINE'.
    @param max_iterations: Maximum number of iterations for ECC algorithm.
    @param epsilon: Termination threshold for ECC algorithm.
    @return: List of aligned images (float32 NumPy arrays [0, 1]).
    """
    if not images:
        return []
    if len(images) == 1:
        return images # No alignment needed

    # Map string motion type to OpenCV constant
    motion_map = {
        'AFFINE': cv2.MOTION_AFFINE,
        'HOMOGRAPHY': cv2.MOTION_HOMOGRAPHY,
        'TRANSLATION': cv2.MOTION_TRANSLATION
    }
    motion_type = motion_map.get(motion_type_str.upper())
    if motion_type is None:
        print(f"Warning: Invalid ECC motion type string '{motion_type_str}'. Defaulting to AFFINE.")
        motion_type = cv2.MOTION_AFFINE
        motion_type_str = 'AFFINE' # Update string for print message

    print(f"\nAligning images using ECC (Motion Type: {motion_type_str})...")
    reference = images[0]
    aligned = [reference] # Start with the reference image

    # Convert reference image to grayscale uint8
    try:
        ref_gray = cv2.cvtColor((reference * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        h, w = ref_gray.shape
    except cv2.error as e:
        print(f"Error converting reference image to grayscale: {e}")
        print("Cannot proceed with ECC alignment.")
        return images # Return original images

    # Define termination criteria for ECC
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, max_iterations, epsilon)

    # Align subsequent images to the reference
    for i, img in enumerate(images[1:], 1):
        print(f"Aligning image {i+1}/{len(images)} using ECC...")

        try:
            # Convert current image to grayscale uint8
            img_gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)

            # Initialize warp matrix based on motion type
            if motion_type == cv2.MOTION_HOMOGRAPHY:
                warp_matrix = np.eye(3, 3, dtype=np.float32)
            else: # Translation, Affine, Euclidean
                warp_matrix = np.eye(2, 3, dtype=np.float32)

            # Run ECC algorithm to find the transformation
            try:
                (cc, warp_matrix_found) = cv2.findTransformECC(ref_gray, img_gray, warp_matrix, motion_type, criteria)
                # Use the found matrix if ECC converged successfully
                warp_matrix = warp_matrix_found
                print(f"  ECC finished for image {i+1}. Correlation: {cc:.4f}")
            except cv2.error as ecc_error:
                 # This can happen if ECC doesn't converge or inputs are unsuitable
                 print(f"  Warning: findTransformECC failed for image {i+1}: {ecc_error}. Using original image.")
                 aligned.append(img)
                 continue # Skip warping for this image

            # Warp the current image using the found transformation
            if motion_type == cv2.MOTION_HOMOGRAPHY:
                 # Use warpPerspective for Homography
                 aligned_img = cv2.warpPerspective(img, warp_matrix, (w, h), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REFLECT)
            else:
                 # Use warpAffine for Translation, Affine, Euclidean
                 aligned_img = cv2.warpAffine(img, warp_matrix, (w, h), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REFLECT)

            aligned.append(aligned_img.astype(np.float32)) # Ensure output is float32
            print(f"  Aligned image {i+1} using ECC.")

        except cv2.error as e:
            print(f"OpenCV Error aligning image {i+1} with ECC: {str(e)}")
            print("  Using original image as fallback.")
            aligned.append(img)
        except Exception as e:
            print(f"Unexpected Error aligning image {i+1} with ECC: {str(e)}")
            print("  Using original image as fallback.")
            aligned.append(img)

    print(f"ECC Alignment complete. Returning {len(aligned)} images.")
    return aligned


# --- AKAZE Alignment ---
def align_akaze(images):
    """
    Align images using AKAZE feature matching and Homography.
    Aligns all images to the first image in the list.

    @param images: List of images (as float32 NumPy arrays [0, 1]) to align.
    @return: List of aligned images (float32 NumPy arrays [0, 1]).
    """
    if not images:
        return []
    if len(images) == 1:
        return images # No alignment needed

    print("\nAligning images using AKAZE features...")
    reference = images[0]
    aligned = [reference] # Start with the reference image

    # Convert reference image to grayscale uint8 for feature detection
    try:
        ref_gray = cv2.cvtColor((reference * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        h, w = ref_gray.shape
    except cv2.error as e:
        print(f"Error converting reference image to grayscale: {e}")
        print("Cannot proceed with alignment.")
        return images # Return original images if conversion fails

    # Initialize AKAZE detector
    try:
        akaze = cv2.AKAZE_create()
    except cv2.error as e:
        print(f"Error creating AKAZE detector: {e}. Check OpenCV installation.")
        print("Cannot proceed with alignment.")
        return images

    # Find keypoints and descriptors in the reference image
    try:
        kp_ref, des_ref = akaze.detectAndCompute(ref_gray, None)
        if des_ref is None or len(kp_ref) == 0:
            print("Warning: No descriptors found in reference image. Skipping alignment for subsequent images.")
            return images
    except cv2.error as e:
        print(f"Error detecting features in reference image: {e}")
        print("Cannot proceed with alignment.")
        return images

    # Initialize Brute-Force Matcher (NORM_HAMMING for binary descriptors like AKAZE)
    try:
        # Use crossCheck=False for KNN matching + ratio test
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    except cv2.error as e:
        print(f"Error creating BFMatcher: {e}. Check OpenCV installation.")
        print("Cannot proceed with alignment.")
        return images

    # Align subsequent images to the reference
    for i, img in enumerate(images[1:], 1):
        print(f"Aligning image {i+1}/{len(images)}...")

        try:
            # Convert current image to grayscale uint8
            img_gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)

            # Find keypoints and descriptors in the current image
            kp_img, des_img = akaze.detectAndCompute(img_gray, None)

            if des_img is None or len(kp_img) < 4:
                print(f"Warning: Not enough descriptors found in image {i+1}. Using original image.")
                aligned.append(img)
                continue

            # Match descriptors using KNN
            matches_knn = bf.knnMatch(des_ref, des_img, k=2)

            # Apply Lowe's ratio test
            good_matches = []
            ratio_thresh = 0.75 # Standard threshold
            for m, n in matches_knn:
                if m is not None and n is not None and hasattr(m, 'distance') and hasattr(n, 'distance'):
                    if m.distance < ratio_thresh * n.distance:
                        good_matches.append(m)

            print(f"Found {len(good_matches)} good matches (ratio={ratio_thresh}) for image {i+1}.")

            # Need at least min_match_count good matches
            min_match_count = 10
            if len(good_matches) >= min_match_count:
                # Extract location of good matches
                src_pts_list = []
                dst_pts_list = []
                for m in good_matches:
                    if m.queryIdx < len(kp_ref) and m.trainIdx < len(kp_img):
                         src_pts_list.append(kp_ref[m.queryIdx].pt)
                         dst_pts_list.append(kp_img[m.trainIdx].pt)

                if len(src_pts_list) < min_match_count:
                    print(f"Warning: Not enough valid matches ({len(src_pts_list)}/{min_match_count}) after index check for image {i+1}. Using original image.")
                    aligned.append(img)
                    continue

                src_pts = np.float32(src_pts_list).reshape(-1, 1, 2)
                dst_pts = np.float32(dst_pts_list).reshape(-1, 1, 2)

                # Find homography matrix using RANSAC
                M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)

                if M is None:
                     print(f"Warning: Homography calculation failed for image {i+1} (findHomography returned None). Using original image.")
                     aligned.append(img)
                     continue

                # Warp the current image
                aligned_img = cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
                aligned.append(aligned_img)
                print(f"Aligned image {i+1} using Homography.")

            else:
                print(f"Warning: Not enough good matches found ({len(good_matches)}/{min_match_count}) for image {i+1}. Using original image.")
                aligned.append(img)

        except cv2.error as e:
            print(f"OpenCV Error aligning image {i+1}: {str(e)}")
            print("Using original image as fallback.")
            aligned.append(img)
        except Exception as e:
            print(f"Unexpected Error aligning image {i+1}: {str(e)}")
            print("Using original image as fallback.")
            aligned.append(img)

    print(f"AKAZE Alignment complete. Returning {len(aligned)} images.")
    return aligned
