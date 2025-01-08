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
    
    def __init__(self, radius=8, smoothing=4, enhance_sharpness=True):
        """
        Initialize focus stacker with parameters
        @param radius Size of area around each pixel for focus detection (1-20)
        @param smoothing Amount of smoothing for transitions (1-10)
        @param enhance_sharpness Whether to enhance sharpness in poorly focused regions
        """
        if not 1 <= radius <= 20:
            raise ValueError("Radius must be between 1 and 20")
        if not 1 <= smoothing <= 10:
            raise ValueError("Smoothing must be between 1 and 10")
            
        # Store parameters
        self.radius = radius  # Used for focus measure window and guided filter
        self.smoothing = smoothing  # Used for guided filter epsilon
        self.enhance_sharpness = enhance_sharpness
        
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
            
        # Enhance local contrast with small tiles
        print("Enhancing local contrast...")
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(2,2))  # Smaller tiles, higher contrast
        img = clahe.apply(img)
        
        # Calculate multi-scale gradients
        print("Computing gradients...")
        gradients = []
        kernel_sizes = [3, 5]  # Focus on smaller kernels for fine detail
        for ksize in kernel_sizes:
            sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=ksize)
            sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=ksize)
            gradient = np.sqrt(sobelx**2 + sobely**2)
            gradients.append(gradient)
            
        if progress_callback:
            progress_callback(base_progress + progress_range * 0.4)
        
        # Calculate multi-scale Laplacian
        print("Computing Laplacian responses...")
        laplacians = []
        kernel_sizes = [3, 5]  # Focus on smaller kernels
        for ksize in kernel_sizes:
            laplacian = cv2.Laplacian(img, cv2.CV_64F, ksize=ksize)
            laplacians.append(np.abs(laplacian))
            
        if progress_callback:
            progress_callback(base_progress + progress_range * 0.6)
        
        # Combine measures with emphasis on high frequencies
        print("Combining focus measures...")
        focus_map = np.zeros_like(img, dtype=np.float32)
        
        # Weight gradients heavily to emphasize edges
        for gradient in gradients:
            focus_map += gradient * 3.0  # Stronger weight for edge detection
            
        # Weight Laplacians to emphasize fine details
        for i, laplacian in enumerate(laplacians):
            weight = 2.0 if i == 0 else 1.0  # More weight to smaller kernel
            focus_map += laplacian * weight
            
        # Normalize and enhance high frequencies
        focus_map = cv2.normalize(focus_map, None, 0, 1, cv2.NORM_MINMAX)
        focus_map = np.power(focus_map, 0.5)  # Increase contrast in high-frequency regions
        
        if progress_callback:
            progress_callback(base_progress + progress_range * 0.8)
        
        # No smoothing to preserve maximum sharpness
        print("Focus measure calculation complete")
        return focus_map.astype(np.float32)

    def _enhance_sharpness_multiscale(self, img):
        """
        Enhance sharpness using multi-scale approach with deconvolution and unsharp masking
        @param img Input image
        @return Enhanced image
        """
        # Parameters for multi-scale enhancement
        kernel_sizes = [3, 5, 7]  # Multiple scales
        snr_values = [200, 150, 100]  # Stronger to weaker enhancement
        
        enhanced = img.copy()
        
        # Apply deconvolution at multiple scales
        for size, snr in zip(kernel_sizes, snr_values):
            # Create point spread function (PSF)
            center = size // 2
            psf = np.zeros((size, size))
            psf[center, center] = 1
            psf = cv2.GaussianBlur(psf, (size, size), 0.5)  # Smaller sigma for sharper result
            psf /= psf.sum()
            
            # Pad image
            pad = size * 2
            padded = np.pad(enhanced, ((pad,pad), (pad,pad)), mode='reflect')
            
            # Apply Wiener deconvolution
            psf_fft = np.fft.fft2(psf, s=padded.shape)
            img_fft = np.fft.fft2(padded)
            psf_conj = np.conj(psf_fft)
            power = np.abs(psf_fft) ** 2
            img_deconv = np.real(np.fft.ifft2(
                img_fft * psf_conj / (power + 1/snr)
            ))
            
            # Crop and update result
            deconv = img_deconv[pad:-pad, pad:-pad]
            enhanced = np.clip(deconv, 0, 1)
        
        # Apply unsharp masking as final step
        blurred = cv2.GaussianBlur(enhanced, (3, 3), 0.5)
        mask = enhanced - blurred
        enhanced = enhanced + mask * 0.5  # Adjust strength here
        
        return np.clip(enhanced, 0, 1)

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
        
        # Find regions with poor focus
        max_focus = np.max(focus_maps, axis=0)
        low_focus_mask = max_focus < 0.2  # Lower threshold to catch more regions
        
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
        
        # Enhance high-confidence regions
        print("Enhancing weight contrast...")
        weights = np.power(weights, 3.0)  # Stronger emphasis on high confidence
        weights_sum = np.sum(weights, axis=0, keepdims=True)
        weights = weights / (weights_sum + 1e-10)
        
        # Blend images
        print("Blending images...")
        result = np.zeros_like(aligned_images[0])
        
        for channel in range(3):
            # Weight each image
            channel_result = np.sum(aligned_images[:,:,:,channel] * weights, axis=0)
            
            # Enhance sharpness in low focus regions if enabled
            if self.enhance_sharpness and np.any(low_focus_mask):
                print(f"Enhancing sharpness in low focus regions for channel {channel}")
                enhanced = self._enhance_sharpness_multiscale(channel_result)
                channel_result = np.where(low_focus_mask, enhanced, channel_result)
                
            result[:,:,channel] = channel_result
            
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

    def save_image(self, img, path, format_name, color_space):
        """
        Save processed image
        @param img Image to save
        @param path Output path
        @param format_name Output format
        @param color_space Color space
        """
        print(f"\nSaving image...")
        print(f"Path: {path}")
        print(f"Format: {format_name}")
        print(f"Color space: {color_space}")
        
        try:
            if format_name in ['TIFF (16-bit)', 'PNG (16-bit)']:
                print("Converting to 16-bit...")
                # Convert each channel separately to prevent color noise
                img_16bit = np.zeros(img.shape, dtype=np.uint16)
                for channel in range(3):
                    # Scale to 16-bit with proper rounding
                    channel_data = np.clip(img[:,:,channel] * 65535.0 + 0.5, 0, 65535)
                    img_16bit[:,:,channel] = channel_data.astype(np.uint16)
                
                print(f"Image converted to 16-bit with shape {img_16bit.shape}")
                
                format_type = 'TIFF' if format_name == 'TIFF (16-bit)' else 'PNG'
                print(f"Saving as {format_type}...")
                
                # Create PIL image with correct mode and color profile
                pil_img = PIL.Image.fromarray(img_16bit, mode='RGB')
                if color_space == 'sRGB':
                    icc = self.color_profiles['sRGB'].tobytes()
                    pil_img.info['icc_profile'] = icc
                
                # Save with appropriate format-specific parameters
                if format_type == 'TIFF':
                    pil_img.save(path, format=format_type, tiffinfo={317: 2}, compression='tiff_deflate')
                else:  # PNG
                    pil_img.save(path, format=format_type, optimize=True)
            else:
                print("Converting to 8-bit JPEG...")
                img_8bit = (img * 255).astype(np.uint8)
                print(f"Image converted to 8-bit with shape {img_8bit.shape}")
                
                print("Saving as JPEG...")
                pil_img = PIL.Image.fromarray(img_8bit, mode='RGB')
                pil_img.save(path, format='JPEG', quality=95, optimize=True)
                
            print(f"Successfully saved image to {path}")
            
        except Exception as e:
            print(f"Error saving image: {str(e)}")
            raise
