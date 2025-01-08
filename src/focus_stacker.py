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
import cupy as cp  # GPU array operations
# Initialize GPU device
cp.cuda.Device(0).use()

# Create reusable kernels for common operations
laplace_kernel = cp.array([[0, 1, 0],
                          [1, -4, 1],
                          [0, 1, 0]], dtype=cp.float32)

# Ultra-sharp kernel optimized for photogrammetry micro-detail
sharp_kernel = cp.array([[-3,-3,-3],
                        [-3, 25,-3],
                        [-3,-3,-3]], dtype=cp.float32)

# Enhanced high-frequency kernel for maximum detail recovery
highfreq_kernel = cp.array([[-3,-3,-3],
                          [-3, 24,-3],
                          [-3,-3,-3]], dtype=cp.float32)

# Edge enhancement kernel with stronger center weight
edge_kernel = cp.array([[-2,-2,-2,-2,-2],
                       [-2, 3, 3, 3,-2],
                       [-2, 3, 12, 3,-2],
                       [-2, 3, 3, 3,-2],
                       [-2,-2,-2,-2,-2]], dtype=cp.float32)

class FocusStacker:
    def __init__(self, radius=8, smoothing=4, scale_factor=2):
        """
        @param radius: Size of the focus measure window (1-20)
        @param smoothing: Amount of smoothing applied to focus maps (1-10)
        @param scale_factor: Processing scale multiplier (1-4). Higher values may improve detail but increase processing time.
                           1 = original resolution
                           2 = 2x upscaling (default, recommended)
                           3 = 3x upscaling (more detail, slower)
                           4 = 4x upscaling (maximum detail, much slower)
        """
        if not 1 <= radius <= 20:
            raise ValueError("Radius must be between 1 and 20")
        if not 1 <= smoothing <= 10:
            raise ValueError("Smoothing must be between 1 and 10")
        if not 1 <= scale_factor <= 4:
            raise ValueError("Scale factor must be between 1 and 4")
            
        self.radius = radius
        self.smoothing = smoothing
        self.scale_factor = scale_factor
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
        print("\nAligning images using GPU...")
        reference = images[0]
        aligned = [reference]
        
        ref_gray = cv2.cvtColor((reference * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        gpu_ref = cp.asarray(ref_gray.astype(np.float32))
        
        # Convert to GPU and normalize
        gpu_ref = cp.asarray(ref_gray.astype(np.float32))
        gpu_ref = (gpu_ref - cp.min(gpu_ref)) / (cp.max(gpu_ref) - cp.min(gpu_ref))
        
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
                    
                    # Convert to GPU arrays first
                    gpu_scaled_ref = cp.asarray(scaled_ref.astype(np.float32))
                    gpu_scaled_img = cp.asarray(scaled_img.astype(np.float32))
                    
                    # Apply contrast enhancement directly on GPU
                    gpu_scaled_ref = (gpu_scaled_ref - cp.min(gpu_scaled_ref)) / (cp.max(gpu_scaled_ref) - cp.min(gpu_scaled_ref))
                    gpu_scaled_img = (gpu_scaled_img - cp.min(gpu_scaled_img)) / (cp.max(gpu_scaled_img) - cp.min(gpu_scaled_img))
                    
                    # Enhanced phase correlation with higher upsampling
                    shift, error, _ = phase_cross_correlation(
                        gpu_scaled_ref.get(),
                        gpu_scaled_img.get(),
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
        if len(img.shape) == 3:
            img = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            img = (img * 255).astype(np.uint8)
            
        # Convert to GPU array
        gpu_img = cp.asarray(img.astype(np.float32))
        
        # Multi-scale wavelet decomposition for focus detection
        levels = 4
        focus_maps = []
        
        for scale in range(levels):
            # Apply wavelet decomposition
            if scale > 0:
                gpu_scaled = cp.asarray(cv2.resize(cp.asnumpy(gpu_img), None, fx=1.0/(2**scale), fy=1.0/(2**scale)))
            else:
                gpu_scaled = gpu_img
                
            # Pyramid decomposition for better detail analysis
            pyramid_levels = 3
            pyramids = []
            current = gpu_scaled.copy()
            
            for _ in range(pyramid_levels):
                # Gaussian blur for next level
                blurred = cp.asarray(cv2.GaussianBlur(cp.asnumpy(current), (5,5), 0))
                # High-frequency residual
                residual = current - blurred
                pyramids.append(residual)
                current = cp.asarray(cv2.resize(cp.asnumpy(blurred), (blurred.shape[1]//2, blurred.shape[0]//2)))
            
            # Enhanced frequency analysis
            high_freq = cp.zeros_like(gpu_scaled)
            for i, residual in enumerate(pyramids):
                if i > 0:
                    residual = cp.asarray(cv2.resize(cp.asnumpy(residual), (gpu_scaled.shape[1], gpu_scaled.shape[0])))
                weight = 1.0 / (2 ** i)  # Higher weight for finer details
                high_freq += weight * cp.abs(residual)
            
            # Enhanced multi-scale edge detection optimized for photogrammetry
            edges = cp.zeros_like(gpu_scaled)
            # Use finer sigmas for better detail preservation
            for sigma in [0.5, 0.75, 1.0]:
                # Apply Gaussian blur directly on GPU with proper padding
                blurred = gpu_scaled.copy()
                padded = cp.pad(blurred, ((1,1), (1,1)), mode='reflect')
                for _ in range(3):  # Approximate Gaussian with multiple box blurs
                    blurred = (padded[:-2, 1:-1] + padded[1:-1, 1:-1] + padded[2:, 1:-1] +
                             padded[1:-1, :-2] + padded[1:-1, 2:]) / 5.0
                    # Re-pad for next iteration
                    padded = cp.pad(blurred, ((1,1), (1,1)), mode='reflect')
                    
                # Compute edge response using FFT
                edge = cp.abs(cp.fft.fftshift(cp.real(cp.fft.ifft2(
                    cp.fft.fft2(blurred) * cp.fft.fft2(laplace_kernel, s=blurred.shape)
                ))))
                
                # Weight smaller scales more heavily for fine detail preservation
                weight = 1.0 / (sigma * sigma)  # Quadratic weighting
                edges += edge * weight
            
            # Enhanced local contrast analysis optimized for photogrammetry
            local_std = cp.zeros_like(gpu_scaled)
            window_sizes = [3, 5, 7]  # Multiple window sizes for multi-scale analysis
            weights = [0.5, 0.3, 0.2]  # Higher weight for smaller windows (finer details)
            
            for size, weight in zip(window_sizes, weights):
                pad_size = size // 2
                padded = cp.pad(gpu_scaled, pad_size, mode='reflect')
                
                # Compute local statistics using sliding window
                local_var = cp.zeros_like(gpu_scaled)
                for i in range(size):
                    for j in range(size):
                        window = padded[i:i+gpu_scaled.shape[0], j:j+gpu_scaled.shape[1]]
                        diff = window - gpu_scaled
                        local_var += diff * diff
                
                local_std += cp.sqrt(local_var / (size * size)) * weight
            
            # Combine all focus measures with emphasis on high frequencies
            focus_map = high_freq * edges * cp.sqrt(local_std)
            
            # Always ensure focus map matches input dimensions
            if focus_map.shape[:2] != (img.shape[0], img.shape[1]):
                focus_map = cp.asarray(cv2.resize(cp.asnumpy(focus_map), (img.shape[1], img.shape[0])))
            
            focus_maps.append(focus_map)
            
            # Weight maps by scale importance with adaptive weighting for photogrammetry
            weights = [0.6, 0.25, 0.1, 0.05]  # Stronger emphasis on primary detail level
            final_focus = cp.zeros_like(focus_maps[0])
            
            # Enhanced multi-scale combination with subject-aware weighting
            for i, (fm, w) in enumerate(zip(focus_maps, weights)):
                # Calculate local variance to identify high-detail regions (likely the main subject)
                local_var = cp.asarray(cv2.GaussianBlur(cp.asnumpy(fm * fm), (15, 15), 0)) - \
                          cp.power(cp.asarray(cv2.GaussianBlur(cp.asnumpy(fm), (15, 15), 0)), 2)
                
                # Create detail-aware mask
                detail_mask = cp.clip((local_var - cp.min(local_var)) / (cp.max(local_var) - cp.min(local_var) + 1e-6), 0.3, 1.0)
                
                # Apply stronger enhancement to high-detail areas
                enhanced = cp.power(fm, 0.7 + 0.3 * detail_mask)  # Adaptive power based on detail level
                
                # Apply local contrast enhancement with detail-aware strength
                local_mean = cp.asarray(cv2.GaussianBlur(cp.asnumpy(enhanced), (0,0), 1.0))
                local_detail = enhanced - local_mean
                enhanced = enhanced + local_detail * (0.3 + 0.4 * detail_mask)  # Adaptive contrast boost
                
                final_focus += enhanced * w * (0.7 + 0.3 * detail_mask)  # Detail-weighted combination
            
            # Multi-scale contrast enhancement with subject-aware normalization
            focus_map = (final_focus - cp.min(final_focus)) / (cp.max(final_focus) - cp.min(final_focus))
            
            # Calculate detail-aware normalization mask
            detail_mask = cp.asarray(cv2.GaussianBlur(cp.asnumpy(focus_map * focus_map), (25, 25), 0))
            detail_mask = (detail_mask - cp.min(detail_mask)) / (cp.max(detail_mask) - cp.min(detail_mask) + 1e-6)
            
            # Apply adaptive enhancement based on detail level
            focus_map = cp.power(focus_map, 0.5 + 0.4 * detail_mask)  # More enhancement for high-detail areas
        
        return cp.asnumpy(focus_map).astype(np.float32)

    def _blend_images(self, aligned_images, focus_maps):
        # Get dimensions for processing
        h, w = aligned_images[0].shape[:2]
        if self.scale_factor > 1:
            new_h, new_w = h * self.scale_factor, w * self.scale_factor
        else:
            new_h, new_w = h, w
            
        # Initialize result array on GPU
        result = cp.zeros((new_h, new_w, 3), dtype=cp.float32)
        weights_sum = cp.zeros((new_h, new_w, 1), dtype=cp.float32)
        
        # Debug dimensions
        print("\nDimension check before blending:")
        print(f"Target dimensions (h, w): ({h}, {w})")
        print(f"Processing dimensions (new_h, new_w): ({new_h}, {new_w})")
        for i, (img, fm) in enumerate(zip(aligned_images, focus_maps)):
            print(f"Image {i+1} shape: {img.shape}")
            print(f"Focus map {i+1} shape: {fm.shape}")
            
        # Ensure all focus maps match image dimensions before processing
        resized_focus_maps = []
        for i, fm in enumerate(focus_maps):
            if fm.shape[:2] != (h, w):
                print(f"Resizing focus map {i+1} from {fm.shape[:2]} to {(h, w)}")
                fm = cv2.resize(fm, (w, h), interpolation=cv2.INTER_LINEAR)
                print(f"After resize - focus map {i+1} shape: {fm.shape}")
            resized_focus_maps.append(fm)
            
        # Process images one at a time to reduce memory usage
        for img, fm in zip(aligned_images, resized_focus_maps):
            # Clear GPU cache
            cp.get_default_memory_pool().free_all_blocks()
            
            # Scale both image and focus map to processing dimensions
            img_up = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
            fm_up = cv2.resize(fm, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            
            # Ensure proper shape for focus maps
            if len(fm_up.shape) == 2:
                fm_up = fm_up.reshape(fm_up.shape[0], fm_up.shape[1], 1)
        
            # Process focus map
            gpu_fm = cp.asarray(fm_up)
            
            # Apply feathered blending mask
            overlap_size = 100
            if len(gpu_fm.shape) > 2:
                fm_2d = gpu_fm[..., 0]
            else:
                fm_2d = gpu_fm
                
            fm_blur = cp.asarray(cv2.GaussianBlur(cp.asnumpy(fm_2d),
                                                 (overlap_size*2+1, overlap_size*2+1),
                                                 overlap_size/3))
            boundary = cp.abs(fm_2d - fm_blur) > 0.05
            fm_new = cp.where(boundary, fm_blur, fm_2d)
            
            # Edge-aware smoothing
            smoothed = cp.asarray(gaussian(cp.asnumpy(fm_new), sigma=2.0))
            
            # Normalize focus map
            fm_min = cp.min(smoothed)
            fm_max = cp.max(smoothed)
            if fm_max > fm_min:
                weight = (smoothed - fm_min) / (fm_max - fm_min)
            else:
                weight = cp.ones_like(smoothed) / len(aligned_images)
                
            # Reshape weight for RGB blending
            weight = weight.reshape(weight.shape[0], weight.shape[1], 1)
            
            # Add to result
            gpu_img = cp.asarray(img_up)
            result += gpu_img * weight
            weights_sum += weight
            
            # Clear individual arrays
            del gpu_fm, fm_blur, fm_new, smoothed, weight, gpu_img
            cp.get_default_memory_pool().free_all_blocks()
            
        # Normalize result
        result = result / (weights_sum + 1e-10)
        
        # Calculate input statistics and preserve original brightness characteristics
        ref_img = cp.asarray(aligned_images[0])
        input_mean = float(cp.mean(ref_img))
        input_std = float(cp.std(ref_img))
        max_ref = float(cp.max(ref_img))
        
        # Normalize result while preserving original characteristics
        result_mean = float(cp.mean(result))
        result_std = float(cp.std(result))
        # Use a stronger blend factor to stay closer to original brightness
        blend_factor = 0.95  # 95% of original brightness, 5% normalized
        # Apply gentler normalization
        normalized = ((result - result_mean) * (input_std / (result_std + 1e-6)) + input_mean)
        result = result * blend_factor + normalized * (1 - blend_factor)
        # Additional brightness correction to prevent overshooting
        result = cp.clip(result, 0.0, max_ref)
        
        # Clean up reference image
        del ref_img
        
        # Debug sharpening mask dimensions
        print(f"\nSharpening mask calculation:")
        print(f"Result shape: {result.shape}")
        print(f"Focus mask initial shape: {result[...,0].shape}")
        
        # Compute focus-aware sharpening mask using resized focus maps
        focus_mask = cp.zeros_like(result[...,0])
        for i, fm in enumerate(resized_focus_maps):
            print(f"Processing focus map {i+1}:")
            print(f"Original shape: {fm.shape}")
            # Resize focus map to match processing dimensions
            fm_up = cv2.resize(fm, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            print(f"Upscaled shape: {fm_up.shape}")
            gpu_fm = cp.asarray(fm_up)
            if len(gpu_fm.shape) > 2:
                gpu_fm = gpu_fm[..., 0]
            print(f"GPU array shape: {gpu_fm.shape}")
            focus_mask += (1.0 - gpu_fm)
            del gpu_fm, fm_up
            
        print(f"Final focus mask shape: {focus_mask.shape}")
        focus_mask /= len(focus_maps)
        focus_mask = cp.clip(focus_mask, 0.3, 1.0)
        
        # Apply sharpening to full image at once
        sharp_result = cp.zeros_like(result)
        for c in range(3):
            # Basic sharpening
            sharp = cp.real(cp.fft.ifft2(
                cp.fft.fft2(result[...,c]) * cp.fft.fft2(sharp_kernel, s=result[...,c].shape)
            ))
            
            # High-frequency enhancement
            high_freq = cp.real(cp.fft.ifft2(
                cp.fft.fft2(result[...,c]) * cp.fft.fft2(highfreq_kernel, s=result[...,c].shape)
            ))
            
            # Combine enhancements
            sharp_strength = cp.clip(focus_mask * 1.2, 0.7, 0.95)
            sharp_result[...,c] = result[...,c] + sharp * sharp_strength + high_freq * 0.3
            
            # Clear intermediate results
            del sharp, high_freq
            cp.get_default_memory_pool().free_all_blocks()
        
        result = sharp_result
        del sharp_result
        
        # Convert back to CPU and downscale if needed
        if self.scale_factor > 1:
            result_np = cv2.resize(cp.asnumpy(result), (w, h), 
                                 interpolation=cv2.INTER_LANCZOS4)
        else:
            result_np = cp.asnumpy(result)
            
        # Final normalization and clipping
        result_np = np.clip(result_np, 0, 1)
        
        # Clear GPU memory
        del result, focus_mask
        cp.get_default_memory_pool().free_all_blocks()
        
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
