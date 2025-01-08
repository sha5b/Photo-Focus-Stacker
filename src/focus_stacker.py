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
            
            if scale > 0:
                focus_map = cp.asarray(cv2.resize(cp.asnumpy(focus_map), (img.shape[1], img.shape[0])))
                
            focus_maps.append(focus_map)
            
        # Weight maps by scale importance with stronger emphasis on finest details
        weights = [0.5, 0.3, 0.15, 0.05]  # Increased weight for finest scale
        final_focus = cp.zeros_like(focus_maps[0])
        
        # Enhanced multi-scale combination
        for i, (fm, w) in enumerate(zip(focus_maps, weights)):
            # Apply non-linear enhancement to each scale
            enhanced = cp.power(fm, 0.8)  # Less aggressive power for better detail preservation
            # Apply local contrast enhancement
            local_mean = cp.asarray(cv2.GaussianBlur(cp.asnumpy(enhanced), (0,0), 1.0))
            local_detail = enhanced - local_mean
            enhanced = enhanced + local_detail * 0.5  # Boost local contrast
            final_focus += enhanced * w
            
        # Multi-scale contrast enhancement optimized for photogrammetry
        focus_map = (final_focus - cp.min(final_focus)) / (cp.max(final_focus) - cp.min(final_focus))
        # Apply gentler contrast enhancement to preserve more gradients
        focus_map = cp.power(focus_map, 0.7)  # Less aggressive power for better detail preservation
        
        return cp.asnumpy(focus_map).astype(np.float32)

    def _blend_images(self, aligned_images, focus_maps):
        # Process at configured scale
        gpu_aligned = []
        gpu_focus = []
        
        for img, fm in zip(aligned_images, focus_maps):
            if self.scale_factor > 1:
                # Upscale image with optimized parameters
                h, w = img.shape[:2]
                new_h, new_w = h * self.scale_factor, w * self.scale_factor
                
                # Use Lanczos4 for RGB image
                img_up = cv2.resize(img, (new_w, new_h), 
                                  interpolation=cv2.INTER_LANCZOS4)
                
                # Use linear interpolation for focus maps (smoother transitions)
                fm_up = cv2.resize(fm, (new_w, new_h), 
                                 interpolation=cv2.INTER_LINEAR)
            else:
                img_up = img
                fm_up = fm
            
            # Ensure proper shape for focus maps
            if len(fm_up.shape) == 2:
                fm_up = fm_up.reshape(fm_up.shape[0], fm_up.shape[1], 1)
                
            gpu_aligned.append(cp.asarray(img_up))
            gpu_focus.append(cp.asarray(fm_up))
            
        gpu_aligned = cp.array(gpu_aligned)
        gpu_focus = cp.array(gpu_focus)  # Shape will be (N, H, W, 1)
        
        # Create feathered blending masks with larger overlap
        overlap_size = 100  # increased overlap for smoother transitions
        for i in range(len(gpu_focus)):
            # Gaussian blur the focus map edges (handle channel dimension)
            fm = gpu_focus[i]
            if len(fm.shape) > 2:
                fm_2d = fm[..., 0]  # Take first channel
                fm_blur = cp.asarray(cv2.GaussianBlur(cp.asnumpy(fm_2d), 
                                                     (overlap_size*2+1, overlap_size*2+1), 
                                                     overlap_size/3))
                # Create smoother transition at boundaries with lower threshold
                boundary = cp.abs(fm_2d - fm_blur) > 0.05
                fm_new = cp.where(boundary, fm_blur, fm_2d)
                gpu_focus[i] = fm_new.reshape(fm_new.shape[0], fm_new.shape[1], 1)  # Restore channel
            else:
                fm_blur = cp.asarray(cv2.GaussianBlur(cp.asnumpy(fm), 
                                                     (overlap_size*2+1, overlap_size*2+1), 
                                                     overlap_size/3))
                boundary = cp.abs(fm - fm_blur) > 0.05
                gpu_focus[i] = cp.where(boundary, fm_blur, fm)
            
        # Edge-aware smoothing with adaptive radius
        sigma = 2.0  # Increased for smoother transitions
        for i in range(len(gpu_focus)):
            # Handle focus maps with channel dimension
            if len(gpu_focus[i].shape) > 2:
                smoothed = cp.asarray(gaussian(cp.asnumpy(gpu_focus[i][..., 0]), sigma=sigma))
                gpu_focus[i] = smoothed.reshape(smoothed.shape[0], smoothed.shape[1], 1)
            else:
                gpu_focus[i] = cp.asarray(gaussian(cp.asnumpy(gpu_focus[i]), sigma=sigma))
        
        # Normalize focus maps individually first
        for i in range(len(focus_maps)):
            fm = gpu_focus[i]
            if len(fm.shape) > 2:
                fm_2d = fm[..., 0]  # Take first channel
                fm_min = cp.min(fm_2d)
                fm_max = cp.max(fm_2d)
                if fm_max > fm_min:
                    fm_norm = (fm_2d - fm_min) / (fm_max - fm_min)
                    gpu_focus[i] = fm_norm.reshape(fm_norm.shape[0], fm_norm.shape[1], 1)
            else:
                fm_min = cp.min(fm)
                fm_max = cp.max(fm)
                if fm_max > fm_min:
                    gpu_focus[i] = (fm - fm_min) / (fm_max - fm_min)
        
        # Compute weights with better normalization (handle channel dimension)
        gpu_focus_2d = cp.array([fm[..., 0] if len(fm.shape) > 2 else fm for fm in gpu_focus])
        weights_sum = cp.sum(gpu_focus_2d, axis=0, keepdims=True)
        weights = cp.where(weights_sum > 0, 
                         gpu_focus_2d / (weights_sum + 1e-10),
                         1.0 / len(focus_maps))
        
        # Ensure weights sum to 1 exactly
        weights_sum = cp.sum(weights, axis=0, keepdims=True)
        weights = weights / (weights_sum + 1e-10)
        
        # Add channel dimension for RGB blending
        weights = weights.reshape(weights.shape[0], weights.shape[1], weights.shape[2], 1)
        
        # Add channel dimension for RGB blending (weights already have shape (N, H, W, 1))
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
        
        # Compute focus-aware sharpening mask
        focus_mask = cp.zeros_like(result[...,0])  # Single channel
        for fm in gpu_focus:
            # Ensure focus map is 2D
            if len(fm.shape) > 2:
                fm = fm[..., 0]  # Take first channel if multi-dimensional
            focus_mask += (1.0 - fm)
        focus_mask /= len(gpu_focus)
        focus_mask = cp.clip(focus_mask, 0.3, 1.0)  # Ensure minimum sharpening
        
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
                enhanced += enhancement * focus_mask  # Focus mask is already 2D
            
            result_sharp[...,c] = enhanced
        
        # Second pass: Adaptive sharpening
        result_sharp2 = cp.zeros_like(result)
        for c in range(3):
            # Apply stronger sharpening in out-of-focus areas
            sharp = cp.real(cp.fft.ifft2(
                cp.fft.fft2(result_sharp[...,c]) * cp.fft.fft2(sharp_kernel, s=result_sharp[...,c].shape)
            ))
            # Multi-pass sharpening with increasing strength in out-of-focus areas
            sharp_strength = cp.clip(focus_mask * 1.2, 0.7, 0.95)  # Increased base strength
            
            # First high-frequency enhancement pass
            high_freq1 = cp.real(cp.fft.ifft2(
                cp.fft.fft2(result_sharp[...,c]) * cp.fft.fft2(highfreq_kernel, s=result_sharp[...,c].shape)
            ))
            
            # Multi-kernel enhancement for maximum detail preservation
            edge_enhanced = cp.real(cp.fft.ifft2(
                cp.fft.fft2(result_sharp[...,c]) * cp.fft.fft2(edge_kernel, s=result_sharp[...,c].shape)
            ))
            
            # Stronger high-frequency enhancement
            high_freq2 = cp.real(cp.fft.ifft2(
                cp.fft.fft2(result_sharp[...,c]) * cp.fft.fft2(highfreq_kernel, s=result_sharp[...,c].shape)
            ))
            
            # Deconvolution for detail recovery
            psf = cp.array([[0.05, 0.1, 0.05],
                          [0.1, 0.4, 0.1],
                          [0.05, 0.1, 0.05]], dtype=cp.float32)
            deconv = cp.real(cp.fft.ifft2(
                cp.fft.fft2(result_sharp[...,c]) / (cp.fft.fft2(psf, s=result_sharp[...,c].shape) + 1e-4)
            ))
            
            # Combine all enhancements with maximum detail preservation
            high_freq_strength = cp.clip(focus_mask * 1.1, 0.6, 0.9)  # More aggressive strength
            edge_strength = cp.clip(focus_mask * 0.9, 0.5, 0.8)  # Stronger edge enhancement
            deconv_strength = cp.clip(focus_mask * 0.6, 0.3, 0.7)  # More aggressive deconvolution
            
            result_sharp2[...,c] = (result_sharp[...,c] + 
                                  sharp * sharp_strength +
                                  high_freq1 * high_freq_strength +
                                  high_freq2 * (high_freq_strength * 0.7) +  # Increased contribution
                                  edge_enhanced * edge_strength +  # Add edge enhancement
                                  deconv * deconv_strength)
        
        # Multi-scale feathered blending
        def create_feather_weights(size):
            y, x = cp.ogrid[:size[0], :size[1]]
            weights = cp.minimum(x, size[1]-x) * cp.minimum(y, size[0]-y)
            return weights / weights.max()

        # Create feather weights for smooth transitions
        feather = create_feather_weights(result.shape[:2])
        feather = feather.reshape(feather.shape[0], feather.shape[1])  # Ensure 2D shape
        
        # Apply feathered blending across scales
        levels = 4  # Additional pyramid level for finer detail control
        pyramid_result = []
        pyramid_sharp = []
        
        # Build Laplacian pyramids
        current_result = result
        current_sharp = result_sharp2
        
        for _ in range(levels):
            # Gaussian blur
            blurred_result = cp.asarray(cv2.GaussianBlur(cp.asnumpy(current_result), (5,5), 0))
            blurred_sharp = cp.asarray(cv2.GaussianBlur(cp.asnumpy(current_sharp), (5,5), 0))
            
            # Compute and store residual
            pyramid_result.append(current_result - blurred_result)
            pyramid_sharp.append(current_sharp - blurred_sharp)
            
            # Downsample for next level
            current_result = cp.asarray(cv2.resize(cp.asnumpy(blurred_result), 
                                                 (blurred_result.shape[1]//2, blurred_result.shape[0]//2)))
            current_sharp = cp.asarray(cv2.resize(cp.asnumpy(blurred_sharp), 
                                                (blurred_sharp.shape[1]//2, blurred_sharp.shape[0]//2)))
        
        # Add base levels
        pyramid_result.append(current_result)
        pyramid_sharp.append(current_sharp)
        
        # Blend pyramids with feathering
        blended_pyramid = []
        for i, (pr, ps) in enumerate(zip(pyramid_result, pyramid_sharp)):
            # Resize feather weights if needed
            if pr.shape[:2] != feather.shape:
                level_feather = cp.asarray(cv2.resize(cp.asnumpy(feather), (pr.shape[1], pr.shape[0])))
            else:
                level_feather = feather
                
            # Adaptive blending based on detail level
            weight = level_feather * (1.0 - 0.1 * i)  # Reduce feathering influence at finer levels
            weight = weight.reshape(weight.shape[0], weight.shape[1], 1)  # Add channel dim properly
            blend = pr * (1 - weight) + ps * weight
            blended_pyramid.append(blend)
            
        # Reconstruct image from pyramid
        result = blended_pyramid[-1]
        for level in reversed(blended_pyramid[:-1]):
            result = cp.asarray(cv2.resize(cp.asnumpy(result), (level.shape[1], level.shape[0])))
            result += level
            
        # Final adaptive blend with stronger emphasis on enhanced details
        blend_mask = cp.clip(0.95 * focus_mask.reshape(focus_mask.shape[0], focus_mask.shape[1]) * feather, 0.7, 0.98)  # More aggressive blend
        blend_mask = blend_mask.reshape(blend_mask.shape[0], blend_mask.shape[1], 1)  # Add channel dim properly
        result = (1 - blend_mask) * result + blend_mask * result_sharp2
        
        # Downscale if we processed at higher resolution
        if self.scale_factor > 1:
            h, w = aligned_images[0].shape[:2]
            result_np = cp.asnumpy(result)
            
            # Use Lanczos4 for downscaling to preserve sharpness
            result_np = cv2.resize(result_np, (w, h), 
                                 interpolation=cv2.INTER_LANCZOS4)
        else:
            result_np = cp.asnumpy(result)
        
        # Re-normalize to input brightness distribution
        result_mean = float(np.mean(result_np))
        result_std = float(np.std(result_np))
        result_np = (result_np - result_mean) * (input_std / result_std) + input_mean
        
        # Clip while preserving relative brightness
        result_np = np.clip(result_np, 0, 1)
        
        # Final brightness check
        final_mean = float(np.mean(result_np))
        print(f"Final output mean: {final_mean:.3f}")
        
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
