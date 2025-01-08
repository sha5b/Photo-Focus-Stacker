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

# Aggressive sharpening kernel optimized for photogrammetry
sharp_kernel = cp.array([[-1,-1,-1],
                        [-1, 9,-1],
                        [-1,-1,-1]], dtype=cp.float32)

# High-frequency enhancement kernel
highfreq_kernel = cp.array([[-1,-1,-1],
                          [-1, 8,-1],
                          [-1,-1,-1]], dtype=cp.float32)

class FocusStacker:
    def __init__(self, radius=8, smoothing=4):
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
        print("\nAligning images using GPU...")
        reference = images[0]
        aligned = [reference]
        
        ref_gray = cv2.cvtColor((reference * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        gpu_ref = cp.asarray(ref_gray.astype(np.float32))
        
        # Enhanced CLAHE for better feature detection
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gpu_ref = cp.asarray(clahe.apply(ref_gray))
        
        for i, img in enumerate(images[1:], 1):
            print(f"Aligning image {i+1} with reference...")
            
            try:
                img_gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
                img_gray = cv2.normalize(img_gray, None, 0, 255, cv2.NORM_MINMAX)
                
                # Multi-scale alignment with refined error metrics
                scales = [1.0, 0.5, 0.25]
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
                    
                    scaled_ref = clahe.apply(scaled_ref)
                    scaled_img = clahe.apply(scaled_img)
                    
                    gpu_scaled_ref = cp.asarray(scaled_ref.astype(np.float32))
                    gpu_scaled_img = cp.asarray(scaled_img.astype(np.float32))
                    
                    shift, error, _ = phase_cross_correlation(
                        gpu_scaled_ref.get(),
                        gpu_scaled_img.get(),
                        upsample_factor=10
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
            
            # Multi-scale edge detection
            edges = cp.zeros_like(gpu_scaled)
            for sigma in [0.5, 1.0, 2.0]:
                blurred = cp.asarray(cv2.GaussianBlur(cp.asnumpy(gpu_scaled), (0,0), sigma))
                edge = cp.abs(cp.fft.fftshift(cp.real(cp.fft.ifft2(
                    cp.fft.fft2(blurred) * cp.fft.fft2(laplace_kernel, s=blurred.shape)
                ))))
                edges += edge / sigma  # Weight by scale
            
            # Local contrast enhancement
            local_std = cp.zeros_like(gpu_scaled)
            window_size = 5
            pad_size = window_size // 2
            padded = cp.pad(gpu_scaled, pad_size, mode='reflect')
            
            for i in range(window_size):
                for j in range(window_size):
                    window = padded[i:i+gpu_scaled.shape[0], j:j+gpu_scaled.shape[1]]
                    local_std += (window - gpu_scaled) ** 2
                    
            local_std = cp.sqrt(local_std / (window_size * window_size))
            
            # Combine all focus measures with emphasis on high frequencies
            focus_map = high_freq * edges * cp.sqrt(local_std)
            
            if scale > 0:
                focus_map = cp.asarray(cv2.resize(cp.asnumpy(focus_map), (img.shape[1], img.shape[0])))
                
            focus_maps.append(focus_map)
            
        # Weight maps by scale importance (emphasize finer details)
        weights = [0.4, 0.3, 0.2, 0.1]  # Higher weight for finer scales
        final_focus = cp.zeros_like(focus_maps[0])
        for fm, w in zip(focus_maps, weights):
            final_focus += fm * w
            
        # Aggressive contrast enhancement for photogrammetry
        focus_map = (final_focus - cp.min(final_focus)) / (cp.max(final_focus) - cp.min(final_focus))
        focus_map = cp.power(focus_map, 0.5)  # More aggressive contrast enhancement
        
        return cp.asnumpy(focus_map).astype(np.float32)

    def _blend_images(self, aligned_images, focus_maps):
        gpu_aligned = cp.array([cp.asarray(img) for img in aligned_images])
        gpu_focus = cp.array([cp.asarray(fm) for fm in focus_maps])
        
        # Edge-aware smoothing of focus maps
        sigma = 0.5
        for i in range(len(focus_maps)):
            gpu_focus[i] = cp.asarray(gaussian(cp.asnumpy(gpu_focus[i]), sigma=sigma))
        
        # Normalize focus maps individually first
        for i in range(len(focus_maps)):
            fm = gpu_focus[i]
            fm_min = cp.min(fm)
            fm_max = cp.max(fm)
            if fm_max > fm_min:
                gpu_focus[i] = (fm - fm_min) / (fm_max - fm_min)
        
        # Compute weights with better normalization
        weights_sum = cp.sum(gpu_focus, axis=0, keepdims=True)
        weights = cp.where(weights_sum > 0, 
                         gpu_focus / (weights_sum + 1e-10),
                         1.0 / len(focus_maps))
        
        # Ensure weights sum to 1 exactly
        weights_sum = cp.sum(weights, axis=0, keepdims=True)
        weights = weights / (weights_sum + 1e-10)
        
        weights = weights[..., None]
        result = cp.sum(gpu_aligned * weights, axis=0)
        
        # Calculate and preserve brightness distribution
        input_mean = float(cp.mean(gpu_aligned))
        input_std = float(cp.std(gpu_aligned))
        result_mean = float(cp.mean(result))
        result_std = float(cp.std(result))
        
        print(f"Input stats - mean: {input_mean:.3f}, std: {input_std:.3f}")
        print(f"Result stats - mean: {result_mean:.3f}, std: {result_std:.3f}")
        
        # Normalize result to match input distribution
        result = (result - result_mean) * (input_std / result_std) + input_mean
        
        # Verify final stats
        final_mean = float(cp.mean(result))
        final_std = float(cp.std(result))
        print(f"Final stats - mean: {final_mean:.3f}, std: {final_std:.3f}")
        
        # Compute focus-aware sharpening mask with proper shape
        focus_mask = cp.zeros_like(result[...,0])  # Single channel
        for fm in gpu_focus:
            focus_mask += (1.0 - fm)  # Higher values for out-of-focus regions
        focus_mask /= len(gpu_focus)
        focus_mask = cp.clip(focus_mask, 0.3, 1.0)  # Ensure minimum sharpening
        focus_mask = focus_mask[..., None]  # Add channel dimension
        
        # Multi-scale adaptive sharpening
        result_sharp = cp.zeros_like(result)
        
        # First pass: Detail recovery in out-of-focus areas
        for c in range(3):
            # Pyramid decomposition
            current = result[...,c]
            detail_levels = []
            
            for _ in range(3):
                blurred = cp.asarray(cv2.GaussianBlur(cp.asnumpy(current), (5,5), 0))
                detail = current - blurred
                detail_levels.append(detail)
                current = cp.asarray(cv2.resize(cp.asnumpy(blurred), (blurred.shape[1]//2, blurred.shape[0]//2)))
            
            # Enhanced detail reconstruction
            enhanced = result[...,c].copy()
            for i, detail in enumerate(detail_levels):
                if i > 0:
                    detail = cp.asarray(cv2.resize(cp.asnumpy(detail), (result.shape[1], result.shape[0])))
                enhancement = detail * (0.6 * (i + 1))  # Increased enhancement strength
                enhanced += enhancement * focus_mask[...,0]  # Use single channel for enhancement
            
            result_sharp[...,c] = enhanced
        
        # Second pass: Adaptive sharpening
        result_sharp2 = cp.zeros_like(result)
        for c in range(3):
            # Apply stronger sharpening in out-of-focus areas
            sharp = cp.real(cp.fft.ifft2(
                cp.fft.fft2(result_sharp[...,c]) * cp.fft.fft2(sharp_kernel, s=result_sharp[...,c].shape)
            ))
            # Apply stronger sharpening with high-frequency boost
            sharp_strength = 0.7 * focus_mask[...,0]  # Increased sharpening strength
            
            # Additional high-frequency enhancement
            high_freq = cp.real(cp.fft.ifft2(
                cp.fft.fft2(result_sharp[...,c]) * cp.fft.fft2(highfreq_kernel, s=result_sharp[...,c].shape)
            ))
            high_freq_strength = 0.3 * focus_mask[...,0]  # High-frequency boost in out-of-focus areas
            result_sharp2[...,c] = result_sharp[...,c] + sharp * sharp_strength + high_freq * high_freq_strength
        
        # Final blend with focus-aware weighting (expand alpha to match channels)
        alpha = cp.clip(0.8 * focus_mask, 0.4, 0.9)  # Increased blend strength for sharper result
        result = (1 - alpha) * result + alpha * result_sharp2
        
        # Re-normalize to input brightness distribution
        result_mean = float(cp.mean(result))
        result_std = float(cp.std(result))
        result = (result - result_mean) * (input_std / result_std) + input_mean
        
        # Clip while preserving relative brightness
        result_np = cp.asnumpy(result)
        result_np = np.clip(result_np, 0, 1)
        
        # Final brightness check
        final_mean = float(np.mean(result_np))
        print(f"Final output mean: {final_mean:.3f}")
        
        return np.clip(result_np, 0, 1)

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

    def save_image(self, img, path, format_name='JPEG', color_space='sRGB'):
        print(f"\nSaving image as JPEG...")
        print(f"Path: {path}")
        
        try:
            img_8bit = np.clip(img * 255.0 + 0.5, 0, 255).astype(np.uint8)
            
            pil_img = PIL.Image.fromarray(img_8bit, mode='RGB')
            pil_img.save(path, format='JPEG', quality=95, optimize=True)
            print(f"Successfully saved image to {path}")
            
        except Exception as e:
            print(f"Error saving image: {str(e)}")
            raise
