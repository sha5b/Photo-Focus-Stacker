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

class FocusStacker:
    """
    @class FocusStacker
    @brief Implements focus stacking algorithm to combine multiple images
    """
    
    def __init__(self, radius=8, smoothing=4, enhance_sharpness=True, sharpening_params=None):
        """
        Initialize focus stacker with parameters
        @param radius Size of area around each pixel for focus detection (1-20)
        @param smoothing Amount of smoothing for transitions (1-10)
        @param enhance_sharpness Whether to enhance sharpness in poorly focused regions
        @param sharpening_params Dictionary containing sharpening parameters:
            - snr_values: List of 3 SNR values for multi-scale deconvolution
            - focus_threshold: Threshold for determining regions to enhance
        """
        if not 1 <= radius <= 20:
            raise ValueError("Radius must be between 1 and 20")
        if not 1 <= smoothing <= 10:
            raise ValueError("Smoothing must be between 1 and 10")
            
        # Store parameters
        self.radius = radius  # Used for focus measure window and guided filter
        self.smoothing = smoothing  # Used for guided filter epsilon
        self.enhance_sharpness = enhance_sharpness
        
        # Set default sharpening parameters if none provided
        if sharpening_params is None:
            self.sharpening_params = {
                'snr_values': [120],  # Single, more conservative SNR value
                'focus_threshold': 0.2  # More selective enhancement threshold
            }
        else:
            self.sharpening_params = sharpening_params
        
        # Derived parameters
        self.window_size = 2 * radius + 1  # Window size for focus measure
        self.gaussian_blur = 2.0  # Blur sigma for noise reduction
        self.guided_filter_eps = (smoothing / 10.0) * 1e-4  # Epsilon for guided filter
        
        # Color profiles
        self._init_color_profiles()
        
        print(f"\nInitialized focus stacker:")
        print(f"Radius: {radius}")
        print(f"Smoothing: {smoothing}")
        print(f"Enhance sharpness: {enhance_sharpness}")
        if enhance_sharpness:
            print(f"SNR values: {self.sharpening_params['snr_values']}")
            print(f"Focus threshold: {self.sharpening_params['focus_threshold']}")

    def _init_color_profiles(self):
        """Initialize color profile transformations"""
        self.color_profiles = {
            'sRGB': PIL.ImageCms.createProfile('sRGB')
        }

    def _load_image(self, path):
        """
        Load image and convert to float32
        @param path Path to image file
        @return Loaded image as float32 array
        """
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Failed to load image: {path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255

    def _align_images(self, images):
        """
        Align images using enhanced phase correlation
        @param images List of images to align
        @return List of aligned images
        """
        print("\nAligning images...")
        reference = images[0]
        aligned = [reference]
        
        # Convert reference to grayscale once
        ref_gray = cv2.cvtColor((reference * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        
        # Apply minimal blur to reduce noise in alignment
        ref_gray = cv2.GaussianBlur(ref_gray, (3, 3), 0.5)  # Reduced blur
        
        # Enhance contrast for better feature detection
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4,4))  # Smaller tiles
        ref_gray = clahe.apply(ref_gray)
        
        for i, img in enumerate(images[1:], 1):
            print(f"Aligning image {i+1} with reference...")
            
            # Convert to grayscale and enhance
            img_gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            img_gray = cv2.GaussianBlur(img_gray, (3, 3), 0.5)  # Reduced blur
            img_gray = clahe.apply(img_gray)
            
            try:
                # Calculate shift with error checking
                shift, error, _ = phase_cross_correlation(ref_gray, img_gray)
                
                if error > 1:  # High error indicates potential misalignment
                    print(f"Warning: High alignment error ({error:.2f}) for image {i+1}")
                    
                print(f"Detected shift: ({shift[0]:.2f}, {shift[1]:.2f}), Error: {error:.2f}")
                
                # Apply shift using affine transform
                M = np.float32([[1, 0, -shift[1]], [0, 1, -shift[0]]])
                aligned_img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
                aligned.append(aligned_img)
                
            except Exception as e:
                print(f"Error aligning image {i+1}: {str(e)}")
                print("Using original image as fallback")
                aligned.append(img)
        
        print("Alignment complete")
        return aligned

    def _focus_measure(self, img, progress_callback=None, base_progress=0, progress_range=25):
        """
        Calculate focus measure optimized for sharp detail preservation
        @param img Input image
        @param progress_callback Optional callback for progress updates
        @param base_progress Base progress value (0-100)
        @param progress_range Progress range for this operation
        @return Focus measure map
        """
        print("Starting focus measure calculation...")
        print(f"Input image shape: {img.shape}")
        
        if progress_callback:
            progress_callback(base_progress)
        
        if len(img.shape) == 3:
            print("Converting RGB to grayscale...")
            img = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            img = (img * 255).astype(np.uint8)
            
        if progress_callback:
            progress_callback(base_progress + progress_range * 0.2)
            
        # Reduce contrast enhancement to preserve natural appearance
        print("Applying minimal contrast enhancement...")
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))  # Larger tiles, lower contrast
        img = clahe.apply(img)
        
        # Calculate focus measure using multiple techniques
        print("Computing focus measures...")
        focus_map = np.zeros_like(img, dtype=np.float32)
        
        # 1. Modified Laplacian for fine detail
        lap_kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=np.float32)
        modified_laplacian = np.abs(cv2.filter2D(img.astype(np.float32), -1, lap_kernel))
        
        # 2. Gradient magnitude for edge strength
        sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        gradient = np.sqrt(sobelx**2 + sobely**2)
        
        # 3. Local variance for texture detail
        mean = cv2.boxFilter(img.astype(np.float32), -1, (3, 3))
        mean_sq = cv2.boxFilter(img.astype(np.float32) ** 2, -1, (3, 3))
        variance = mean_sq - mean ** 2
        
        if progress_callback:
            progress_callback(base_progress + progress_range * 0.4)
            
        # Combine measures with balanced weights
        focus_map = (modified_laplacian * 0.4 + 
                    gradient * 0.4 + 
                    variance * 0.2)  # Less weight on variance to reduce noise
        
        # Normalize with gentler curve
        focus_map = cv2.normalize(focus_map, None, 0, 1, cv2.NORM_MINMAX)
        focus_map = np.power(focus_map, 0.8)  # Slightly reduced contrast enhancement
        
        # Apply minimal smoothing to reduce noise while preserving edges
        # Convert to 8-bit for bilateral filter
        focus_map_8bit = (focus_map * 255).astype(np.uint8)
        focus_map_8bit = cv2.bilateralFilter(focus_map_8bit, 5, 25, 25)
        focus_map = focus_map_8bit.astype(np.float32) / 255
        
        if progress_callback:
            progress_callback(base_progress + progress_range * 0.8)
        
        # No smoothing to preserve maximum sharpness
        print("Focus measure calculation complete")
        return focus_map.astype(np.float32)

    def _enhance_sharpness_multiscale(self, img):
        """
        Enhance sharpness using adaptive multi-scale approach with strong edge preservation
        @param img Input image
        @return Enhanced image
        """
        # Parameters for balanced multi-scale enhancement
        kernel_sizes = [3]  # Use single small kernel for more controlled enhancement
        snr_values = self.sharpening_params['snr_values']  # Get from parameters
        
        enhanced = img.copy()
        
        # Step 1: Edge-preserving decomposition using bilateral filter
        # Process each channel separately
        base = np.zeros_like(enhanced)
        for c in range(3):
            # Convert to 8-bit for bilateral filter
            channel_8bit = (enhanced[:,:,c] * 255).astype(np.uint8)
            # More conservative bilateral filter with smaller sigmas
            base_8bit = cv2.bilateralFilter(channel_8bit, 5, 15, 15)
            base[:,:,c] = base_8bit.astype(np.float32) / 255
        detail = enhanced - base
        
        # Step 2: Multi-scale enhancement with adaptive filtering
        detail_enhanced = detail.copy()
        
        # Calculate edge mask for adaptive enhancement
        # Convert to grayscale for edge detection
        guide = cv2.cvtColor((enhanced * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        edge_mask = np.zeros_like(guide, dtype=np.float32)
        for size in kernel_sizes:
            sobelx = cv2.Sobel(guide, cv2.CV_32F, 1, 0, ksize=size)
            sobely = cv2.Sobel(guide, cv2.CV_32F, 0, 1, ksize=size)
            edge_strength = np.sqrt(sobelx**2 + sobely**2)
            edge_mask = np.maximum(edge_mask, edge_strength)
        
        edge_mask = cv2.GaussianBlur(edge_mask, (0,0), 1.0)
        edge_mask = cv2.normalize(edge_mask, None, 0, 1, cv2.NORM_MINMAX)
        
        for size, snr in zip(kernel_sizes, snr_values[:len(kernel_sizes)]):
            # Create adaptive PSF based on edge strength
            center = size // 2
            psf = np.zeros((size, size))
            psf[center, center] = 1
            psf = cv2.GaussianBlur(psf, (size, size), 0.3)  # Sharper PSF for finer detail
            psf /= psf.sum()
            
            # Process each channel separately
            for c in range(3):
                # Pad detail layer
                pad = size * 2
                padded = np.pad(detail_enhanced[:,:,c], ((pad,pad), (pad,pad)), mode='reflect')
                padded_edge = np.pad(edge_mask, ((pad,pad), (pad,pad)), mode='reflect')
                
                # Adaptive deconvolution based on edge strength
                for _ in range(1):  # Single iteration to prevent over-enhancement
                    conv = cv2.filter2D(padded, -1, psf)
                    ratio = cv2.filter2D(np.ones_like(padded) / (conv + 1e-10), -1, psf)
                    update = padded * ratio
                    
                    # Apply update adaptively based on edge strength
                    padded = padded * (1 - padded_edge) + update * padded_edge
                
                # Crop and update result
                detail_enhanced[:,:,c] = np.clip(padded[pad:-pad, pad:-pad], -0.3, 0.3)  # More conservative range for natural look
        
        # Step 3: Combine enhanced detail with base layer using refined edge-aware weights
        # Calculate local structure tensor for better edge detection
        dx = cv2.Sobel(guide, cv2.CV_32F, 1, 0, ksize=3)
        dy = cv2.Sobel(guide, cv2.CV_32F, 0, 1, ksize=3)
        local_structure = np.sqrt(dx*dx + dy*dy)
        
        # Combine with edge mask for more precise edge weighting
        weight = local_structure * edge_mask
        weight = cv2.GaussianBlur(weight, (0,0), 0.5)  # Smaller sigma for less spread
        weight = cv2.normalize(weight, None, 0.45, 0.55, cv2.NORM_MINMAX)  # Very conservative range
        
        # Expand weight for broadcasting
        weight = np.repeat(weight[:,:,np.newaxis], 3, axis=2)
        
        enhanced = base + detail_enhanced * weight
        enhanced = np.clip(enhanced, 0, 1)
        
        return enhanced

    def _blend_images(self, aligned_images, focus_maps):
        """
        Blend images using weighted average method with sharpness enhancement
        @param aligned_images List of aligned images
        @param focus_maps List of focus measure maps
        @return Blended image
        """
        print("\nBlending images...")
        print(f"Number of images: {len(aligned_images)}")
        print(f"Image shape: {aligned_images[0].shape}")
        print(f"Focus map shape: {focus_maps[0].shape}")
        
        # Convert to numpy arrays
        aligned_images = np.array(aligned_images)
        focus_maps = np.array(focus_maps)
        
        # Find regions needing enhancement
        max_focus = np.max(focus_maps, axis=0)
        low_focus_mask = max_focus < self.sharpening_params['focus_threshold']
        # Expand mask to match image dimensions
        low_focus_mask = np.repeat(low_focus_mask[:,:,np.newaxis], 3, axis=2)
        
        # Normalize focus maps using local contrast
        print("Computing local contrast weights...")
        kernel_size = max(3, self.window_size // 2)
        kernel = np.ones((kernel_size, kernel_size), dtype=np.float32) / (kernel_size * kernel_size)
        
        weights = []
        for focus_map in focus_maps:
            # Calculate local statistics
            local_mean = cv2.filter2D(focus_map, -1, kernel)
            local_var = cv2.filter2D(focus_map**2, -1, kernel) - local_mean**2
            
            # Weight by local contrast
            weight = np.sqrt(local_var) * focus_map
            weights.append(weight)
            
        weights = np.array(weights)
        
        # Normalize weights per-pixel
        print("Normalizing weights...")
        weights_sum = np.sum(weights, axis=0, keepdims=True)
        weights = np.where(weights_sum > 0, weights / (weights_sum + 1e-10), 1.0 / len(weights))
        
        # More balanced weight distribution
        print("Adjusting weight distribution...")
        weights = np.power(weights, 1.4)  # Stronger emphasis on high confidence regions
        weights_sum = np.sum(weights, axis=0, keepdims=True)
        weights = weights / (weights_sum + 1e-10)
        
        # Expand weights for proper broadcasting
        weights = np.repeat(weights[:, :, :, np.newaxis], 3, axis=3)  # Shape: (N, H, W, 3)
        
        # Blend images
        print("Blending images...")
        result = np.sum(aligned_images * weights, axis=0)  # Shape: (H, W, 3)
            
        # Enhance sharpness in low focus regions if enabled
        if self.enhance_sharpness and np.any(low_focus_mask):
            print("Enhancing sharpness in low focus regions")
            enhanced = self._enhance_sharpness_multiscale(result)
            result = np.where(low_focus_mask, enhanced, result)
            
        return np.clip(result, 0, 1)

    def _guided_filter(self, img, guide):
        """
        Apply guided filter for smooth blending
        @param img Input image
        @param guide Guidance image
        @param radius Filter radius
        @param eps Regularization parameter
        @return Filtered image
        """
        mean_guide = cv2.boxFilter(guide, -1, (self.window_size, self.window_size))
        mean_img = cv2.boxFilter(img, -1, (self.window_size, self.window_size))
        corr_guide = cv2.boxFilter(guide * guide, -1, (self.window_size, self.window_size))
        corr_img_guide = cv2.boxFilter(img * guide, -1, (self.window_size, self.window_size))
        
        var_guide = corr_guide - mean_guide * mean_guide
        cov_img_guide = corr_img_guide - mean_img * mean_guide
        
        a = cov_img_guide / (var_guide + self.guided_filter_eps)
        b = mean_img - a * mean_guide
        
        mean_a = cv2.boxFilter(a, -1, (self.window_size, self.window_size))
        mean_b = cv2.boxFilter(b, -1, (self.window_size, self.window_size))
        
        return mean_a * guide + mean_b

    def split_into_stacks(self, image_paths, stack_size):
        """
        Split images into stacks based on filename patterns
        @param image_paths List of paths to input images
        @param stack_size Number of images per stack
        @return List of lists containing paths for each stack
        """
        import re
        
        # Group files by their base name using multiple patterns
        stacks_dict = {}
        for path in image_paths:
            filename = os.path.basename(path)
            name, ext = os.path.splitext(filename)
            
            # Try different patterns to extract base name and number
            patterns = [
                r'^(.*?)[-_]?(\d+)$',  # name-123 or name_123 or name123
                r'^(.*?)[-_]?(\d+)[-_]',  # name-123-suffix or name_123_suffix
                r'(\d+)[-_]?(.*?)$'  # 123-name or 123_name
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
            
        # Sort files within each stack
        for base_name in stacks_dict:
            stacks_dict[base_name].sort()
            
        # Convert dictionary to list of stacks
        stacks = list(stacks_dict.values())
        
        # Verify stack sizes
        expected_size = stack_size
        for i, stack in enumerate(stacks):
            if len(stack) != expected_size:
                print(f"Warning: Stack {i+1} has {len(stack)} images, expected {expected_size}")
                print(f"Stack contents: {[os.path.basename(p) for p in stack]}")
                
        # Sort stacks by first filename to maintain consistent order
        stacks.sort(key=lambda x: x[0])
        
        print("\nDetected stacks:")
        for i, stack in enumerate(stacks):
            print(f"Stack {i+1}: {[os.path.basename(p) for p in stack]}")
            
        return stacks

    def process_stack(self, image_paths, color_space='sRGB', progress_callback=None):
        """
        Process a stack of images
        @param image_paths List of paths to input images
        @param color_space Target color space
        @param progress_callback Optional callback for progress updates
        @return Processed image
        """
        if len(image_paths) < 2:
            raise ValueError("At least 2 images are required")
            
        print(f"\nProcessing stack of {len(image_paths)} images...")
        print("Image paths:", image_paths)
            
        # Load images
        images = []
        for i, path in enumerate(image_paths):
            print(f"Loading image {i+1}/{len(image_paths)}: {path}")
            try:
                img = self._load_image(path)
                images.append(img)
                print(f"Successfully loaded image {i+1} with shape {img.shape}")
                if progress_callback:
                    progress_callback(int((i + 1) / len(image_paths) * 30))
            except Exception as e:
                print(f"Error loading image {path}: {str(e)}")
                raise
                
        # Align images
        print("\nAligning images...")
        try:
            aligned = self._align_images(images)
            print(f"Successfully aligned {len(aligned)} images")
            if progress_callback:
                progress_callback(50)
        except Exception as e:
            print(f"Error during image alignment: {str(e)}")
            raise
            
        # Calculate focus measures
        print("\nCalculating focus measures...")
        focus_maps = []
        for i, img in enumerate(aligned):
            print(f"Computing focus measure for image {i+1}/{len(aligned)}")
            try:
                # Calculate base progress for this image's focus measure
                base_progress = 50 + (i * 25) // len(aligned)
                progress_range = 25 // len(aligned)
                
                focus_map = self._focus_measure(
                    img, 
                    progress_callback=progress_callback,
                    base_progress=base_progress,
                    progress_range=progress_range
                )
                focus_maps.append(focus_map)
                print(f"Focus measure computed for image {i+1}")
            except Exception as e:
                print(f"Error calculating focus measure for image {i+1}: {str(e)}")
                raise
                
        # Blend images
        print("\nBlending images...")
        try:
            result = self._blend_images(aligned, focus_maps)
            print("Successfully blended images")
            if progress_callback:
                progress_callback(90)
        except Exception as e:
            print(f"Error during image blending: {str(e)}")
            raise
            
        # Convert to target color space
        if color_space != 'sRGB':
            print(f"\nConverting to {color_space} color space...")
            try:
                result = self._convert_color_space(result, color_space)
                print("Color space conversion complete")
            except Exception as e:
                print(f"Error during color space conversion: {str(e)}")
                raise
            
        if progress_callback:
            progress_callback(100)
            
        print("\nStack processing complete!")
        return result

    def _convert_color_space(self, img, target_space):
        """
        Convert image to target color space
        @param img Input image
        @param target_space Target color space name
        @return Converted image
        """
        # Convert numpy array to PIL Image
        pil_img = PIL.Image.fromarray((img * 255).astype('uint8'))
        
        # Create transform
        source_profile = self.color_profiles['sRGB']
        target_profile = self.color_profiles[target_space]
        transform = PIL.ImageCms.buildTransformFromOpenProfiles(
            source_profile, target_profile, "RGB", "RGB")
        
        # Apply transform
        converted = PIL.ImageCms.applyTransform(pil_img, transform)
        
        # Convert back to numpy array
        return np.array(converted).astype(np.float32) / 255

    def save_image(self, img, path, format_name='JPEG', color_space='sRGB'):
        """
        Save processed image
        @param img Image to save
        @param path Output path
        @param format_name Output format (currently only JPEG supported)
        @param color_space Color space
        """
        print(f"\nSaving image as JPEG...")
        print(f"Path: {path}")
        
        try:
            # Convert to 8-bit with careful rounding
            img_8bit = np.clip(img * 255.0 + 0.5, 0, 255).astype(np.uint8)
            
            # Save as high-quality JPEG
            pil_img = PIL.Image.fromarray(img_8bit, mode='RGB')
            pil_img.save(path, format='JPEG', quality=95, optimize=True)
            print(f"Successfully saved image to {path}")
            
        except Exception as e:
            print(f"Error saving image: {str(e)}")
            raise
