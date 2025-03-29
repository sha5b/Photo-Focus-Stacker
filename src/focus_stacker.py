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
from scipy.fft import fft2, ifft2 # Use SciPy for FFT on CPU
# Removed cupy import and GPU initialization

# Create reusable kernels for common operations using NumPy
laplace_kernel = np.array([[0, 1, 0],
                          [1, -4, 1],
                          [0, 1, 0]], dtype=np.float32)

# Ultra-sharp kernel with extreme micro-detail preservation
sharp_kernel = np.array([[-4,-4,-4],
                        [-4, 33,-4],
                        [-4,-4,-4]], dtype=np.float32)

# Maximum detail recovery kernel with stronger edge emphasis
highfreq_kernel = np.array([[-2,-3,-2],
                          [-3, 25,-3],
                          [-2,-3,-2]], dtype=np.float32)

# Enhanced edge detection with multi-directional sensitivity
edge_kernel = np.array([[-3,-3,-3,-3,-3],
                       [-3, 4, 4, 4,-3],
                       [-3, 4, 16, 4,-3],
                       [-3, 4, 4, 4,-3],
                       [-3,-3,-3,-3,-3]], dtype=np.float32)

# Fine detail enhancement kernel for microscopic features
detail_kernel = np.array([[-1,-2,-1],
                         [-2, 13,-2],
                         [-1,-2,-1]], dtype=np.float32)

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
        # Removed self.scale_factor
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
        print("\nAligning images using CPU...") # Changed message
        reference = images[0]
        aligned = [reference]
        
        ref_gray = cv2.cvtColor((reference * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        # Removed GPU conversion, use ref_gray directly
        ref_gray_norm = (ref_gray.astype(np.float32) - np.min(ref_gray)) / (np.max(ref_gray) - np.min(ref_gray)) # Normalize using numpy
        
        for i, img in enumerate(images[1:], 1):
            print(f"Aligning image {i+1} with reference...")
            
            try:
                img_gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
                img_gray = cv2.normalize(img_gray, None, 0, 255, cv2.NORM_MINMAX)
                
                # Enhanced multi-scale alignment with finer scale steps
                scales = [1.0, 0.8, 0.6, 0.4, 0.2]  # More granular scale steps
                best_shift = None
                best_error = float('inf')
                
                for scale in scales:
                    if scale != 1.0:
                        width = int(img_gray.shape[1] * scale)
                        height = int(img_gray.shape[0] * scale)
                        scaled_ref = cv2.resize(ref_gray, (width, height))
                        scaled_img = cv2.resize(img_gray, (width, height))
                    else:
                        scaled_ref = ref_gray
                        scaled_img = img_gray
                    
                    # Use NumPy arrays directly
                    scaled_ref_norm = (scaled_ref.astype(np.float32) - np.min(scaled_ref)) / (np.max(scaled_ref) - np.min(scaled_ref))
                    scaled_img_norm = (scaled_img.astype(np.float32) - np.min(scaled_img)) / (np.max(scaled_img) - np.min(scaled_img))
                    
                    # Enhanced phase correlation with higher upsampling using NumPy arrays
                    shift, error, _ = phase_cross_correlation(
                        scaled_ref_norm,
                        scaled_img_norm,
                        upsample_factor=20  # Increased precision
                    )
                    
                    if scale != 1.0:
                        shift = shift / scale
                    
                    shifted_img = cv2.warpAffine(
                        img_gray, 
                        np.float32([[1, 0, -shift[1]], [0, 1, -shift[0]]]),
                        (img_gray.shape[1], img_gray.shape[0]),
                        flags=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_REFLECT
                    )
                    
                    error = -cv2.matchTemplate(
                        ref_gray, 
                        shifted_img, 
                        cv2.TM_CCOEFF_NORMED
                    )[0][0]
                    
                    if error < best_error:
                        best_error = error
                        best_shift = shift
                
                shift = best_shift
                error = best_error
                print(f"Detected shift: {shift}, error: {error}")
                
                M = np.float32([[1, 0, -shift[1]], [0, 1, -shift[0]]])
                aligned_img = cv2.warpAffine(
                    img, M, (img.shape[1], img.shape[0]),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REFLECT
                )
                
                if error > 0.1:
                    try:
                        warp_matrix = np.eye(2, 3, dtype=np.float32)
                        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 1000, 1e-7)
                        _, warp_matrix = cv2.findTransformECC(
                            ref_gray,
                            cv2.cvtColor((aligned_img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY),
                            warp_matrix, cv2.MOTION_EUCLIDEAN, criteria
                        )
                        aligned_img = cv2.warpAffine(
                            aligned_img, warp_matrix, (img.shape[1], img.shape[0]),
                            flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_REFLECT
                        )
                        print("Applied ECC refinement")
                    except Exception as e:
                        print(f"Warning: ECC refinement failed: {str(e)}")
                
                aligned.append(aligned_img)
                print(f"Successfully aligned image {i+1}")
                
            except Exception as e:
                print(f"Error aligning image {i+1}: {str(e)}")
                print("Using original image as fallback")
                aligned.append(img)
        
        return aligned

    def _focus_measure(self, img):
        """
        Optimized focus measure calculation using parallel CPU operations
        """
        if len(img.shape) == 3:
            img_gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
        else:
            img_gray = (img * 255).astype(np.uint8).astype(np.float32)
            
        # Use NumPy arrays directly
        
        # Pre-calculate common image derivatives for all scales
        dx = cv2.Sobel(img_gray, cv2.CV_32F, 1, 0, ksize=3)
        dy = cv2.Sobel(img_gray, cv2.CV_32F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(dx*dx + dy*dy)
        
        # Pre-calculate Laplacian for edge detection
        laplacian = cv2.Laplacian(img_gray, cv2.CV_32F)
        
        # Multi-scale analysis (can be parallelized further if needed, but keep simple for now)
        scales = [1.0, 0.5, 0.25]
        weights = [0.6, 0.3, 0.1]
        
        focus_map = np.zeros_like(img_gray)
        
        for scale, weight in zip(scales, weights):
            if scale != 1.0:
                scaled = cv2.resize(img_gray, None, fx=scale, fy=scale)
                scaled_grad = cv2.resize(gradient_magnitude, None, fx=scale, fy=scale)
                scaled_lap = cv2.resize(laplacian, None, fx=scale, fy=scale)
            else:
                scaled = img_gray
                scaled_grad = gradient_magnitude
                scaled_lap = laplacian
            
            # Frequency analysis
            high_freq = np.abs(scaled - cv2.GaussianBlur(scaled, (5,5), 0))
            
            # Edge detection
            edge_strength = np.abs(scaled_lap)
            
            # Local contrast
            local_mean = cv2.GaussianBlur(scaled, (7,7), 1.5)
            local_contrast = np.abs(scaled - local_mean)
            
            # Combine measures
            scale_measure = high_freq * edge_strength * local_contrast * scaled_grad
            
            # Resize back to original size if needed
            if scale != 1.0:
                scale_measure = cv2.resize(scale_measure, (img_gray.shape[1], img_gray.shape[0]))
            
            focus_map += weight * scale_measure
            
            # Clear intermediate results (NumPy handles memory)
            del high_freq, edge_strength, local_mean, local_contrast, scale_measure
            if scale != 1.0:
                del scaled, scaled_grad, scaled_lap
        
        # Final enhancement
        focus_map = (focus_map - np.min(focus_map)) / (np.max(focus_map) - np.min(focus_map) + 1e-6)
        
        # Edge-aware enhancement
        edge_mask = np.clip(np.abs(laplacian) / (np.max(np.abs(laplacian)) + 1e-6), 0, 1)
        focus_map = focus_map * (1.0 + 0.2 * edge_mask)
        
        # Normalize and cleanup
        focus_map = np.clip((focus_map - np.min(focus_map)) / (np.max(focus_map) - np.min(focus_map) + 1e-6), 0, 1)
        
        # No GPU memory to clear
        del dx, dy, gradient_magnitude, laplacian, focus_map, edge_mask
        
        return focus_map.astype(np.float32)

    def _blend_images(self, aligned_images, focus_maps):
        """
        Enhanced blending with depth-aware processing using NumPy
        """
        h, w = aligned_images[0].shape[:2]
        # Removed scaling logic, process at original resolution
        new_h, new_w = h, w 
            
        # Initialize result arrays using NumPy
        result = np.zeros((new_h, new_w, 3), dtype=np.float32)
        weights_sum = np.zeros((new_h, new_w, 1), dtype=np.float32)
        
        # Pre-process focus maps (already at original size from _focus_measure)
        resized_focus_maps = focus_maps # No resizing needed
        
        # Process each image using NumPy operations
        for img, fm in zip(aligned_images, resized_focus_maps):
            # No GPU cache to clear
            
            # Use original image and focus map (no upscaling)
            img_proc = img # Use original image
            fm_proc = fm # Use original focus map
            
            # Use NumPy arrays directly
            np_img = img_proc
            np_fm = fm_proc
            if len(np_fm.shape) == 2:
                np_fm = np_fm.reshape(np_fm.shape[0], np_fm.shape[1], 1)
            fm_2d = np_fm[..., 0] if len(np_fm.shape) > 2 else np_fm
            
            # Calculate depth gradients using NumPy/OpenCV
            dx = cv2.Sobel(fm_2d, cv2.CV_32F, 1, 0, ksize=3)
            dy = cv2.Sobel(fm_2d, cv2.CV_32F, 0, 1, ksize=3)
            depth_gradient = np.sqrt(dx*dx + dy*dy)
            del dx, dy
            
            # Create depth-aware mask using NumPy/OpenCV
            depth_mask = cv2.bilateralFilter(depth_gradient, 9, 75, 75)
            depth_mask = (depth_mask - np.min(depth_mask)) / (np.max(depth_mask) - np.min(depth_mask) + 1e-6)
            del depth_gradient
            
            # Multi-scale analysis using NumPy/OpenCV
            fm_new = np.zeros_like(fm_2d)
            scales = [200, 150, 100, 50]
            weights = [0.35, 0.3, 0.2, 0.15]
            
            for scale, weight in zip(scales, weights):
                kernel_size = (scale*2+1, scale*2+1)
                sigma = scale/3
                
                # Gaussian blur using OpenCV
                fm_blur = cv2.GaussianBlur(fm_2d, kernel_size, sigma)
                
                # Compute edge strength using NumPy
                edge_strength = np.abs(fm_2d - fm_blur)
                edge_strength *= (1.0 + depth_mask)  # Depth-aware edge boost
                
                # Local statistics using NumPy/OpenCV
                edge_sq = edge_strength * edge_strength
                local_std = cv2.GaussianBlur(edge_sq, (25, 25), 0) # Approximation
                threshold = np.mean(edge_strength) + np.std(edge_strength) * (2.0 + depth_mask)
                
                # Combine with depth-aware weighting using NumPy
                blend_weight = weight * (1.0 + 0.5 * depth_mask)
                fm_new += np.where(edge_strength > threshold,
                                 fm_blur * blend_weight,
                                 fm_2d * blend_weight)
                
                # Clean up scale-specific arrays
                del fm_blur, edge_strength, edge_sq, local_std
            
            # Bilateral filtering using OpenCV
            smoothed = cv2.bilateralFilter(fm_new, 11, 100, 100)
            smoothed = cv2.bilateralFilter(smoothed, 7, 50, 50)
            
            # Normalize and prepare weight using NumPy
            weight_map = (smoothed - np.min(smoothed)) / (np.max(smoothed) - np.min(smoothed) + 1e-6)
            weight_map = weight_map.reshape(weight_map.shape[0], weight_map.shape[1], 1)
            
            # Blend using NumPy
            result += np_img * weight_map
            weights_sum += weight_map
            
            # Clean up image-specific arrays
            del np_img, np_fm, fm_2d, fm_new, smoothed, weight_map, depth_mask
            
        # Normalize result using NumPy
        result = result / (weights_sum + 1e-10)
        
        # Calculate input statistics and preserve original brightness characteristics using NumPy
        ref_img = aligned_images[0] # Use NumPy array directly
        input_mean = float(np.mean(ref_img))
        input_std = float(np.std(ref_img))
        max_ref = float(np.max(ref_img))
        
        # Simpler dynamic range preservation using NumPy
        ref_min = float(np.min(ref_img))
        ref_max = float(np.max(ref_img))
        ref_range = ref_max - ref_min
        
        # Normalize result to match reference range using NumPy
        result_min = float(np.min(result))
        result_max = float(np.max(result))
        result = (result - result_min) * (ref_range / (result_max - result_min + 1e-6)) + ref_min
        
        # Apply gentle contrast enhancement using NumPy
        for c in range(3):
            channel_min = float(np.min(ref_img[...,c]))
            channel_max = float(np.max(ref_img[...,c]))
            channel_mean = float(np.mean(ref_img[...,c]))
            
            result[...,c] = np.clip(result[...,c], channel_min, channel_max)
            result[...,c] = (result[...,c] - channel_mean) * 1.1 + channel_mean
            
        # Final range adjustment using NumPy
        result = np.clip(result, 0.0, max_ref)
        
        # Clean up reference image
        del ref_img
        
        # Debug sharpening mask dimensions
        print(f"\nSharpening mask calculation:")
        print(f"Result shape: {result.shape}")
        print(f"Focus mask initial shape: {result[...,0].shape}")
        
        # Compute focus-aware sharpening mask using original focus maps
        focus_mask = np.zeros_like(result[...,0])
        for i, fm in enumerate(resized_focus_maps): # Use original maps
            print(f"Processing focus map {i+1}:")
            print(f"Original shape: {fm.shape}")
            # No upscaling needed
            np_fm = fm # Use original map
            if len(np_fm.shape) > 2:
                np_fm = np_fm[..., 0]
            print(f"NumPy array shape: {np_fm.shape}")
            focus_mask += (1.0 - np_fm)
            del np_fm
            
        print(f"Final focus mask shape: {focus_mask.shape}")
        focus_mask /= len(focus_maps)
        focus_mask = np.clip(focus_mask, 0.3, 1.0)
        
        # Apply sharpening to full image at once using NumPy/SciPy FFT
        # Moved import to top
        
        sharp_result = np.zeros_like(result)
        for c in range(3):
            # Basic sharpening
            sharp = np.real(ifft2(
                fft2(result[...,c]) * fft2(sharp_kernel, s=result[...,c].shape)
            ))
            
            # High-frequency enhancement
            high_freq = np.real(ifft2(
                fft2(result[...,c]) * fft2(highfreq_kernel, s=result[...,c].shape)
            ))
            
            # Enhanced multi-scale sharpening using NumPy/OpenCV
            local_var = cv2.GaussianBlur(result[...,c] * result[...,c], (11, 11), 0) - \
                       np.power(cv2.GaussianBlur(result[...,c], (11, 11), 0), 2)
            
            detail_mask = np.clip((local_var - np.min(local_var)) / (np.max(local_var) - np.min(local_var) + 1e-6), 0.4, 1.0)
            
            # Fine detail enhancement
            fine_detail = np.real(ifft2(
                fft2(result[...,c]) * fft2(detail_kernel, s=result[...,c].shape)
            ))
            
            # Adaptive multi-scale sharpening
            sharp_strength = np.clip(focus_mask * (1.4 + 0.4 * detail_mask), 0.8, 0.99)
            
            # Gentler detail enhancement using NumPy/OpenCV
            local_contrast = cv2.Laplacian(result[...,c], cv2.CV_32F)
            contrast_mask = np.clip(np.abs(local_contrast) / (np.max(np.abs(local_contrast)) + 1e-6), 0.2, 0.6)
            
            # Combine enhancements with reduced strength
            sharp_result[...,c] = result[...,c] + \
                                 sharp * sharp_strength * 0.5 * contrast_mask + \
                                 high_freq * 0.2 * contrast_mask + \
                                 fine_detail * 0.1 * contrast_mask
            
            # Clear intermediate results
            del sharp, high_freq, local_var, detail_mask, fine_detail, sharp_strength, local_contrast, contrast_mask
        
        result = sharp_result
        del sharp_result
        
        # No downscaling needed as processing is done at original resolution
        result_np = result
            
        # Final normalization and clipping
        result_np = np.clip(result_np, 0, 1)
        
        # No GPU memory to clear
        del focus_mask
        
        return result_np

    def split_into_stacks(self, image_paths, stack_size):
        import re
        
        stacks_dict = {}
        for path in image_paths:
            filename = os.path.basename(path)
            name, ext = os.path.splitext(filename)
            
            patterns = [
                r'^(.*?)[-_]?(\d+)$',
                r'^(.*?)[-_]?(\d+)[-_]',
                r'(\d+)[-_]?(.*?)$'
            ]
            
            matched = False
            for pattern in patterns:
                match = re.match(pattern, name)
                if match:
                    groups = match.groups()
                    base_name = groups[0] if len(groups[0]) > len(groups[1]) else groups[1]
                    if base_name not in stacks_dict:
                        stacks_dict[base_name] = []
                    stacks_dict[base_name].append(path)
                    matched = True
                    break
                    
            if not matched:
                print(f"Warning: Could not match pattern for file: {filename}")
                base_name = name
                if base_name not in stacks_dict:
                    stacks_dict[base_name] = []
                stacks_dict[base_name].append(path)
            
        for base_name in stacks_dict:
            stacks_dict[base_name].sort()
            
        stacks = list(stacks_dict.values())
        
        expected_size = stack_size
        for i, stack in enumerate(stacks):
            if len(stack) != expected_size:
                print(f"Warning: Stack {i+1} has {len(stack)} images, expected {expected_size}")
                print(f"Stack contents: {[os.path.basename(p) for p in stack]}")
                
        stacks.sort(key=lambda x: x[0])
        
        print("\nDetected stacks:")
        for i, stack in enumerate(stacks):
            print(f"Stack {i+1}: {[os.path.basename(p) for p in stack]}")
            
        return stacks

    def process_stack(self, image_paths, color_space='sRGB'):
        if len(image_paths) < 2:
            raise ValueError("At least 2 images are required")
            
        print(f"\nProcessing stack of {len(image_paths)} images...")
        print("Image paths:", image_paths)
            
        images = []
        for i, path in enumerate(image_paths):
            print(f"Loading image {i+1}/{len(image_paths)}: {path}")
            try:
                img = self._load_image(path)
                images.append(img)
                print(f"Successfully loaded image {i+1} with shape {img.shape}")
            except Exception as e:
                print(f"Error loading image {path}: {str(e)}")
                raise
                
        print("\nAligning images...")
        try:
            aligned = self._align_images(images)
            print(f"Successfully aligned {len(aligned)} images")
        except Exception as e:
            print(f"Error during image alignment: {str(e)}")
            raise
            
        print("\nCalculating focus measures...")
        focus_maps = []
        for i, img in enumerate(aligned):
            print(f"Computing focus measure for image {i+1}/{len(aligned)}")
            try:
                focus_map = self._focus_measure(img)
                focus_maps.append(focus_map)
                print(f"Focus measure computed for image {i+1}")
            except Exception as e:
                print(f"Error calculating focus measure for image {i+1}: {str(e)}")
                raise
                
        print("\nBlending images...")
        try:
            result = self._blend_images(aligned, focus_maps)
            print("Successfully blended images")
        except Exception as e:
            print(f"Error during image blending: {str(e)}")
            raise
            
        if color_space != 'sRGB':
            print(f"\nConverting to {color_space} color space...")
            try:
                result = self._convert_color_space(result, color_space)
                print("Color space conversion complete")
            except Exception as e:
                print(f"Error during color space conversion: {str(e)}")
                raise
            
        print("\nStack processing complete!")
        return result

    def _convert_color_space(self, img, target_space):
        pil_img = PIL.Image.fromarray((img * 255).astype('uint8'))
        
        source_profile = self.color_profiles['sRGB']
        target_profile = self.color_profiles[target_space]
        transform = PIL.ImageCms.buildTransformFromOpenProfiles(
            source_profile, target_profile, "RGB", "RGB")
        
        converted = PIL.ImageCms.applyTransform(pil_img, transform)
        
        return np.array(converted).astype(np.float32) / 255
    def save_image(self, img, path, format='JPEG', color_space='sRGB'):
        """
        Save the processed image to a file
        @param img: The image array to save
        @param path: Output file path
        @param format: Image format (JPEG)
        @param color_space: Color space to use (sRGB)
        """
        print(f"\nSaving image as {format}...")
        print(f"Path: {path}")
        
        try:
            # Convert to 8-bit with proper rounding
            img_8bit = np.clip(img * 255.0 + 0.5, 0, 255).astype(np.uint8)
            
            # Create PIL image
            pil_img = PIL.Image.fromarray(img_8bit, mode='RGB')
            
            # Save with format-specific settings
            if format.upper() == 'JPEG':
                pil_img.save(path, format='JPEG', quality=95, optimize=True)
            else:
                raise ValueError(f"Unsupported format: {format}")
                
            print(f"Successfully saved image to {path}")
            
        except Exception as e:
            print(f"Error saving image: {str(e)}")
            raise
