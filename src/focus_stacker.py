#!/usr/bin/env python3

import cv2
import numpy as np
import os
from skimage import img_as_float32, img_as_uint
from skimage.color import rgb2lab, lab2rgb
from skimage.filters import gaussian
from skimage.registration import phase_cross_correlation
import PIL.Image
import PIL.ImageCms
from scipy.fft import fft2, ifft2

# Kernels for image processing
laplace_kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)

# Ultra-sharp kernel with extreme micro-detail preservation
sharp_kernel = np.array([[-4,-4,-4],
                        [-4, 33,-4],
                        [-4,-4,-4]], dtype=np.float32)

highfreq_kernel = np.array([[-2,-3,-2], [-3, 25,-3], [-2,-3,-2]], dtype=np.float32)

# Enhanced edge detection with multi-directional sensitivity
edge_kernel = np.array([[-3,-3,-3,-3,-3],
                       [-3, 4, 4, 4,-3],
                       [-3, 4, 16, 4,-3],
                       [-3, 4, 4, 4,-3],
                       [-3,-3,-3,-3,-3]], dtype=np.float32)

detail_kernel = np.array([[-1,-2,-1], [-2, 13,-2], [-1,-2,-1]], dtype=np.float32)

class FocusStacker:
    def __init__(self, radius=8, smoothing=4): # Removed scale_factor
        """
        @param radius: Size of the focus measure window (1-20)
        @param smoothing: Amount of smoothing applied to focus maps (1-10)
        """
        if not 1 <= radius <= 20:
            raise ValueError("Radius must be between 1 and 20")
        if not 1 <= smoothing <= 10:
            raise ValueError("Smoothing must be between 1 and 10")
            
        self.radius = radius
        self.smoothing = smoothing
        self.window_size = 2 * radius + 1
        self._init_color_profiles()

    def _init_color_profiles(self):
        self.color_profiles = {
            'sRGB': PIL.ImageCms.createProfile('sRGB')
        }

    def _load_image(self, path):
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Failed to load image: {path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255

    def _align_images(self, images):
        """Align images using ORB feature matching and Homography."""
        print("\nAligning images using ORB features...")
        reference = images[0]
        aligned = [reference]
        
        # Convert reference image to grayscale
        ref_gray = cv2.cvtColor((reference * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        h, w = ref_gray.shape
        
        # Initialize ORB detector
        # Increased features, adjusted parameters for potentially better matching
        orb = cv2.ORB_create(nfeatures=2000, scaleFactor=1.2, nlevels=8, edgeThreshold=15, patchSize=31) 
        
        # Find keypoints and descriptors in the reference image
        kp_ref, des_ref = orb.detectAndCompute(ref_gray, None)
        if des_ref is None:
            print("Warning: No descriptors found in reference image. Skipping alignment.")
            return images # Return original images if reference has no features

        # Initialize Brute-Force Matcher
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False) # Use crossCheck=False for ratio test

        for i, img in enumerate(images[1:], 1):
            print(f"Aligning image {i+1}/{len(images)}...")
            
            try:
                # Convert current image to grayscale
                img_gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
                
                # Find keypoints and descriptors in the current image
                kp_img, des_img = orb.detectAndCompute(img_gray, None)
                
                if des_img is None or len(kp_img) < 4:
                    print(f"Warning: Not enough descriptors found in image {i+1}. Using original image.")
                    aligned.append(img)
                    continue

                # Match descriptors using KNN
                matches = bf.knnMatch(des_ref, des_img, k=2)
                
                # Apply Lowe's ratio test to filter good matches
                good_matches = []
                if matches:
                    for m, n in matches:
                        if m.distance < 0.75 * n.distance: # Ratio test
                            good_matches.append(m)
                
                print(f"Found {len(good_matches)} good matches for image {i+1}.")

                # Need at least 4 good matches to find homography
                min_match_count = 10 # Increased minimum match count for robustness
                if len(good_matches) >= min_match_count:
                    # Extract location of good matches
                    src_pts = np.float32([kp_ref[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                    dst_pts = np.float32([kp_img[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                    
                    # Find homography matrix using RANSAC
                    M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
                    
                    if M is None:
                         print(f"Warning: Homography calculation failed for image {i+1}. Using original image.")
                         aligned.append(img)
                         continue

                    # Warp the current image to align with the reference image
                    aligned_img = cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
                    aligned.append(aligned_img)
                    print(f"Aligned image {i+1} using Homography.")
                    
                else:
                    print(f"Warning: Not enough good matches found ({len(good_matches)}/{min_match_count}) for image {i+1}. Using original image.")
                    aligned.append(img)
                
            except Exception as e:
                print(f"Error aligning image {i+1}: {str(e)}")
                print("Using original image as fallback")
                aligned.append(img)
        
        return aligned

    def _focus_measure(self, img):
        """
        Calculate a focus measure map for a single image using CPU operations.
        Combines gradient magnitude, Laplacian, high-frequency components, and local contrast.
        """
        if len(img.shape) == 3:
            img_gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
        else: # Assume already grayscale
            img_gray = (img * 255).astype(np.uint8).astype(np.float32)
            
        # Pre-calculate derivatives
        dx = cv2.Sobel(img_gray, cv2.CV_32F, 1, 0, ksize=3)
        dy = cv2.Sobel(img_gray, cv2.CV_32F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(dx*dx + dy*dy)
        laplacian = cv2.Laplacian(img_gray, cv2.CV_32F)
        
        # Multi-scale analysis
        scales = [1.0, 0.5, 0.25]
        weights = [0.6, 0.3, 0.1]
        focus_map = np.zeros_like(img_gray)
        
        for scale, weight in zip(scales, weights):
            # Resize inputs for current scale
            if scale != 1.0:
                scaled_img = cv2.resize(img_gray, None, fx=scale, fy=scale)
                scaled_grad = cv2.resize(gradient_magnitude, None, fx=scale, fy=scale)
                scaled_lap = cv2.resize(laplacian, None, fx=scale, fy=scale)
            else:
                scaled_img = img_gray
                scaled_grad = gradient_magnitude
                scaled_lap = laplacian
            
            # Calculate components for this scale
            high_freq = np.abs(scaled_img - cv2.GaussianBlur(scaled_img, (5,5), 0))
            edge_strength = np.abs(scaled_lap)
            local_mean = cv2.GaussianBlur(scaled_img, (7,7), 1.5)
            local_contrast = np.abs(scaled_img - local_mean)
            
            # Combine components
            scale_measure = high_freq * edge_strength * local_contrast * scaled_grad
            
            # Resize result back to original size if needed
            if scale != 1.0:
                scale_measure = cv2.resize(scale_measure, (img_gray.shape[1], img_gray.shape[0]))
            
            focus_map += weight * scale_measure
            
        # Normalize the combined map
        focus_map = (focus_map - np.min(focus_map)) / (np.max(focus_map) - np.min(focus_map) + 1e-6)
        
        # Optional edge-aware enhancement
        edge_mask = np.clip(np.abs(laplacian) / (np.max(np.abs(laplacian)) + 1e-6), 0, 1)
        focus_map = focus_map * (1.0 + 0.2 * edge_mask)
        
        # Final normalization
        focus_map = np.clip((focus_map - np.min(focus_map)) / (np.max(focus_map) - np.min(focus_map) + 1e-6), 0, 1)
        
        return focus_map.astype(np.float32)

    def _blend_images(self, aligned_images, focus_maps):
        """
        Blend aligned images using their focus maps.
        """
        h, w = aligned_images[0].shape[:2]
        result = np.zeros((h, w, 3), dtype=np.float32)
        weights_sum = np.zeros((h, w, 1), dtype=np.float32)
        
        # Process each image
        for img, fm in zip(aligned_images, focus_maps):
            # Ensure focus map is 2D or extract first channel
            fm_2d = fm[..., 0] if len(fm.shape) > 2 else fm
            
            # Calculate depth gradients (edges in the focus map)
            dx = cv2.Sobel(fm_2d, cv2.CV_32F, 1, 0, ksize=3)
            dy = cv2.Sobel(fm_2d, cv2.CV_32F, 0, 1, ksize=3)
            depth_gradient = np.sqrt(dx*dx + dy*dy)
            
            # Create depth-aware mask (smoother transitions near focus edges)
            depth_mask = cv2.bilateralFilter(depth_gradient, 9, 75, 75)
            depth_mask = (depth_mask - np.min(depth_mask)) / (np.max(depth_mask) - np.min(depth_mask) + 1e-6)
            
            # Multi-scale analysis to refine focus weights
            fm_refined = np.zeros_like(fm_2d)
            scales = [200, 150, 100, 50] # Larger scales capture broader focus regions
            weights = [0.35, 0.3, 0.2, 0.15]
            
            for scale, weight_factor in zip(scales, weights):
                kernel_size = (scale*2+1, scale*2+1)
                sigma = scale/3
                
                fm_blur = cv2.GaussianBlur(fm_2d, kernel_size, sigma)
                edge_strength = np.abs(fm_2d - fm_blur)
                edge_strength *= (1.0 + depth_mask) # Boost edges based on depth gradient
                
                # Thresholding based on local statistics
                edge_sq = edge_strength * edge_strength
                local_std = cv2.GaussianBlur(edge_sq, (25, 25), 0) # Approximate local std dev
                threshold = np.mean(edge_strength) + np.std(edge_strength) * (2.0 + depth_mask)
                
                # Combine weights
                blend_weight = weight_factor * (1.0 + 0.5 * depth_mask)
                fm_refined += np.where(edge_strength > threshold, fm_blur * blend_weight, fm_2d * blend_weight)
                
            # Smooth the refined weights
            smoothed_weights = cv2.bilateralFilter(fm_refined, 11, 100, 100)
            smoothed_weights = cv2.bilateralFilter(smoothed_weights, 7, 50, 50)
            
            # Normalize weights
            weight_map = (smoothed_weights - np.min(smoothed_weights)) / (np.max(smoothed_weights) - np.min(smoothed_weights) + 1e-6)
            weight_map = weight_map.reshape(h, w, 1) # Ensure 3D for broadcasting
            
            # Weighted blending
            result += img * weight_map
            weights_sum += weight_map
            
        # Normalize the final blended image
        result = result / (weights_sum + 1e-10)
        
        # --- Post-processing: Brightness/Contrast/Sharpening ---
        
        # Preserve original brightness range
        ref_img = aligned_images[0]
        ref_min = float(np.min(ref_img))
        ref_max = float(np.max(ref_img))
        ref_range = ref_max - ref_min
        
        result_min = float(np.min(result))
        result_max = float(np.max(result))
        result = (result - result_min) * (ref_range / (result_max - result_min + 1e-6)) + ref_min
        
        # Gentle contrast enhancement per channel, preserving mean
        for c in range(3):
            channel_mean = float(np.mean(ref_img[...,c]))
            result[...,c] = (result[...,c] - channel_mean) * 1.1 + channel_mean
            
        result = np.clip(result, 0.0, ref_max) # Clip to original max
        
        # Compute focus-aware sharpening mask (sharpen less in out-of-focus areas)
        focus_mask = np.zeros_like(result[...,0])
        for fm in focus_maps:
            fm_2d = fm[..., 0] if len(fm.shape) > 2 else fm
            focus_mask += (1.0 - fm_2d) # Invert focus map
        focus_mask /= len(focus_maps)
        focus_mask = np.clip(focus_mask, 0.3, 1.0) # Limit sharpening strength
        
        # Apply multi-kernel sharpening using FFT
        sharp_result = np.zeros_like(result)
        for c in range(3):
            # Calculate components using FFT convolution
            sharp = np.real(ifft2(fft2(result[...,c]) * fft2(sharp_kernel, s=result.shape[:2])))
            high_freq = np.real(ifft2(fft2(result[...,c]) * fft2(highfreq_kernel, s=result.shape[:2])))
            fine_detail = np.real(ifft2(fft2(result[...,c]) * fft2(detail_kernel, s=result.shape[:2])))
            
            # Calculate masks for adaptive sharpening
            local_var = cv2.GaussianBlur(result[...,c]**2, (11, 11), 0) - cv2.GaussianBlur(result[...,c], (11, 11), 0)**2
            detail_mask = np.clip((local_var - np.min(local_var)) / (np.max(local_var) - np.min(local_var) + 1e-6), 0.4, 1.0)
            local_contrast = cv2.Laplacian(result[...,c], cv2.CV_32F)
            contrast_mask = np.clip(np.abs(local_contrast) / (np.max(np.abs(local_contrast)) + 1e-6), 0.2, 0.6)
            
            # Combine sharpening components adaptively
            sharp_strength = np.clip(focus_mask * (1.4 + 0.4 * detail_mask), 0.8, 0.99)
            sharp_result[...,c] = result[...,c] + \
                                 sharp * sharp_strength * 0.5 * contrast_mask + \
                                 high_freq * 0.2 * contrast_mask + \
                                 fine_detail * 0.1 * contrast_mask
        
        result_np = np.clip(sharp_result, 0, 1) # Final clipping
        
        return result_np

    def split_into_stacks(self, image_paths, stack_size):
        """
        Splits a list of image paths into stacks based on filename patterns.
        Assumes filenames end with a sequence number (e.g., img_001.jpg, img_002.jpg).
        """
        import re # Local import ok here as it's self-contained
        
        stacks_dict = {}
        # Try common patterns to extract base name and sequence number
        patterns = [ r'^(.*?)[-_]?(\d+)$', r'^(.*?)[-_]?(\d+)[-_]', r'(\d+)[-_]?(.*?)$' ]
        
        for path in image_paths:
            filename = os.path.basename(path)
            name, _ = os.path.splitext(filename)
            
            matched = False
            for pattern in patterns:
                match = re.match(pattern, name)
                if match:
                    groups = match.groups()
                    # Heuristic to find the 'base name' part vs the number part
                    base_name = groups[0] if len(groups) > 1 and len(groups[0]) > len(groups[1]) else (groups[1] if len(groups) > 1 else groups[0])
                    if not base_name: base_name = "default_stack" # Handle cases like just numbers '001.jpg'
                        
                    if base_name not in stacks_dict:
                        stacks_dict[base_name] = []
                    stacks_dict[base_name].append(path)
                    matched = True
                    break # Stop after first successful pattern match
                    
            if not matched:
                print(f"Warning: Could not determine stack for file: {filename}. Adding to default stack.")
                if "default_stack" not in stacks_dict: stacks_dict["default_stack"] = []
                stacks_dict["default_stack"].append(path)
            
        # Sort images within each stack
        for base_name in stacks_dict:
            stacks_dict[base_name].sort()
            
        # Convert dictionary to list of lists (stacks)
        stacks = list(stacks_dict.values())
        
        # Optional: Check if detected stacks match the expected size
        if stack_size > 0:
            for i, stack in enumerate(stacks):
                if len(stack) != stack_size:
                    print(f"Warning: Stack {i+1} ({os.path.basename(stack[0])[:10]}...) has {len(stack)} images, expected {stack_size}.")
                    # print(f"Stack contents: {[os.path.basename(p) for p in stack]}") # Can be verbose
                
        # Sort stacks based on the first image path for consistent order
        stacks.sort(key=lambda x: x[0] if x else "")
        
        print("\nDetected stacks:")
        for i, stack in enumerate(stacks):
            if stack: print(f"Stack {i+1}: {len(stack)} images starting with {os.path.basename(stack[0])}")
            
        return stacks

    def process_stack(self, image_paths, color_space='sRGB'):
        """
        Main processing pipeline for a single stack of images.
        """
        if len(image_paths) < 2:
            raise ValueError("Focus stacking requires at least 2 images.")
            
        print(f"\n--- Processing stack of {len(image_paths)} images ---")
        # print("Image paths:", image_paths) # Can be verbose
            
        # 1. Load images
        images = []
        for i, path in enumerate(image_paths):
            print(f"Loading image {i+1}/{len(image_paths)}: {os.path.basename(path)}")
            try:
                img = self._load_image(path)
                images.append(img)
            except Exception as e:
                print(f"ERROR loading image {path}: {e}")
                raise # Re-raise critical error
                
        # 2. Align images
        print("\nAligning images...")
        try:
            aligned = self._align_images(images)
            print(f"Alignment complete ({len(aligned)} images).")
        except Exception as e:
            print(f"ERROR during image alignment: {e}")
            raise
            
        # 3. Calculate focus measures
        print("\nCalculating focus measures...")
        focus_maps = []
        for i, img in enumerate(aligned):
            print(f"Calculating focus for image {i+1}/{len(aligned)}")
            try:
                focus_map = self._focus_measure(img)
                focus_maps.append(focus_map)
            except Exception as e:
                print(f"ERROR calculating focus measure for image {i+1}: {e}")
                raise
        print("Focus measure calculation complete.")
                
        # 4. Blend images
        print("\nBlending images...")
        try:
            result = self._blend_images(aligned, focus_maps)
            print("Blending complete.")
        except Exception as e:
            print(f"ERROR during image blending: {e}")
            raise
            
        # 5. Color space conversion (if needed)
        if color_space != 'sRGB':
            print(f"\nConverting to {color_space} color space...")
            try:
                result = self._convert_color_space(result, color_space)
                print("Color space conversion complete.")
            except Exception as e:
                print(f"ERROR during color space conversion: {e}")
                # Decide if this is critical - maybe just warn? For now, raise.
                raise
            
        print("\n--- Stack processing complete! ---")
        return result

    def _convert_color_space(self, img, target_space):
        """Convert image color space using ICC profiles."""
        pil_img = PIL.Image.fromarray((img * 255).astype('uint8'))
        
        source_profile = self.color_profiles.get('sRGB') # Default to sRGB if not specified?
        target_profile = self.color_profiles.get(target_space)
        
        if not source_profile or not target_profile:
             print(f"Warning: Could not find profiles for conversion from sRGB to {target_space}. Skipping conversion.")
             return img # Return original if profiles missing

        try:
            transform = PIL.ImageCms.buildTransformFromOpenProfiles(
                source_profile, target_profile, "RGB", "RGB")
            converted = PIL.ImageCms.applyTransform(pil_img, transform)
            return np.array(converted).astype(np.float32) / 255
        except PIL.ImageCms.PyCMSError as e:
            print(f"Error applying color space transform: {e}")
            return img # Return original on error

    def save_image(self, img, path, format='JPEG', color_space='sRGB'):
        """
        Save the processed image to a file.
        """
        print(f"\nSaving image...")
        print(f"Path: {path}")
        print(f"Format: {format}, Color Space: {color_space}") # Assumes color space handled before saving
        
        try:
            # Convert to 8-bit for saving
            img_8bit = np.clip(img * 255.0 + 0.5, 0, 255).astype(np.uint8)
            pil_img = PIL.Image.fromarray(img_8bit, mode='RGB')
            
            # Add saving options based on format
            save_options = {}
            if format.upper() == 'JPEG':
                save_options['quality'] = 95
                save_options['optimize'] = True
            # Add options for PNG, TIFF etc. here if needed later
            # elif format.upper() == 'PNG':
            #     save_options['compress_level'] = 6 # Example
            
            pil_img.save(path, format=format.upper(), **save_options)
            print(f"Successfully saved image.")
            
        except Exception as e:
            print(f"ERROR saving image: {e}")
            raise
