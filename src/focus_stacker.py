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

class FocusStacker:
    """
    @class FocusStacker
    @brief Implements focus stacking algorithm to combine multiple images
    """
    
    def __init__(self, radius=8, smoothing=4):
        """
        Initialize focus stacker with parameters
        @param radius Size of area around each pixel for focus detection (1-20)
        @param smoothing Amount of smoothing for transitions (1-10)
        """
        if not 1 <= radius <= 20:
            raise ValueError("Radius must be between 1 and 20")
        if not 1 <= smoothing <= 10:
            raise ValueError("Smoothing must be between 1 and 10")
            
        # Store parameters
        self.radius = radius  # Used for focus measure window
        self.smoothing = smoothing  # Used for transitions
        
        # Derived parameters
        self.window_size = 2 * radius + 1  # Window size for focus measure
        
        # Color profiles
        self._init_color_profiles()

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
        Align images using GPU-accelerated phase correlation
        @param images List of aligned images
        @return List of aligned images
        """
        print("\nAligning images using GPU...")
        reference = images[0]
        aligned = [reference]
        
        # Convert reference to grayscale and upload to GPU
        ref_gray = cv2.cvtColor((reference * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        gpu_ref = cp.asarray(ref_gray.astype(np.float32))
        
        # Apply CLAHE on GPU
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gpu_ref = cp.asarray(clahe.apply(ref_gray))
        
        for i, img in enumerate(images[1:], 1):
            print(f"Aligning image {i+1} with reference...")
            
            try:
                # Convert to grayscale and normalize
                img_gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
                img_gray = cv2.normalize(img_gray, None, 0, 255, cv2.NORM_MINMAX)
                
                # Multi-scale alignment
                scales = [1.0, 0.5, 0.25]  # Try different scales
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
                    
                    # Apply CLAHE for better contrast
                    scaled_ref = clahe.apply(scaled_ref)
                    scaled_img = clahe.apply(scaled_img)
                    
                    # Upload to GPU
                    gpu_scaled_ref = cp.asarray(scaled_ref.astype(np.float32))
                    gpu_scaled_img = cp.asarray(scaled_img.astype(np.float32))
                    
                    # Compute phase correlation
                    shift, error, _ = phase_cross_correlation(
                        gpu_scaled_ref.get(),
                        gpu_scaled_img.get(),
                        upsample_factor=10
                    )
                    
                    # Scale shift back to original size
                    if scale != 1.0:
                        shift = shift / scale
                    
                    # Calculate normalized cross-correlation as error metric
                    shifted_img = cv2.warpAffine(
                        img_gray, 
                        np.float32([[1, 0, -shift[1]], [0, 1, -shift[0]]]),
                        (img_gray.shape[1], img_gray.shape[0]),
                        flags=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_REFLECT
                    )
                    
                    # Compute correlation coefficient
                    error = -cv2.matchTemplate(
                        ref_gray, 
                        shifted_img, 
                        cv2.TM_CCOEFF_NORMED
                    )[0][0]  # Negative because we want to minimize
                    
                    if error < best_error:
                        best_error = error
                        best_shift = shift
                
                shift = best_shift
                error = best_error
                print(f"Detected shift: {shift}, error: {error}")
                
                # Create translation matrix
                M = np.float32([[1, 0, -shift[1]], [0, 1, -shift[0]]])
                
                # Apply transformation with border reflection
                aligned_img = cv2.warpAffine(
                    img, M, (img.shape[1], img.shape[0]),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REFLECT
                )
                
                # Refine alignment using ECC if error is above threshold
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
        
        print("Alignment complete")
        return aligned

    def _focus_measure(self, img):
        """
        Calculate focus measure using enhanced Laplacian and local variance
        @param img Input image
        @return Focus measure map
        """
        if len(img.shape) == 3:
            img = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            img = (img * 255).astype(np.uint8)
            
        # Milder CLAHE for better contrast while preserving lighting
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        img_clahe = clahe.apply(img)
        # Use CLAHE only for focus detection, not the actual image
        img_for_focus = img_clahe
            
        # Upload focus detection image to GPU
        gpu_img = cp.asarray(img_for_focus.astype(np.float32))
        
        # Multi-scale Laplacian for better detail detection
        laplacian_small = cp.abs(cp.fft.fftshift(cp.real(cp.fft.ifft2(cp.fft.fft2(gpu_img) * cp.fft.fft2(laplace_kernel, s=gpu_img.shape)))))
        
        # Calculate local variance (sensitive to texture details)
        mean = cp.average(gpu_img)
        variance = cp.abs(gpu_img - mean) ** 2
        
        # Combine Laplacian and variance with more emphasis on Laplacian
        focus_map = laplacian_small * cp.sqrt(variance)
        
        # Normalize with moderate non-linear enhancement
        focus_map = (focus_map - cp.min(focus_map)) / (cp.max(focus_map) - cp.min(focus_map))
        focus_map = cp.power(focus_map, 0.6)  # Moderate boost to sharp regions
        
        # Download result from GPU
        focus_map = cp.asnumpy(focus_map)
        
        return focus_map.astype(np.float32)

    def _blend_images(self, aligned_images, focus_maps):
        """
        Blend images using enhanced weighted average method with sharpening
        @param aligned_images List of aligned images
        @param focus_maps List of focus measure maps
        @return Blended image
        """
        # Upload data to GPU
        gpu_aligned = cp.array([cp.asarray(img) for img in aligned_images])
        gpu_focus = cp.array([cp.asarray(fm) for fm in focus_maps])
        
        # Apply milder Gaussian smoothing to focus maps
        sigma = 0.5  # Reduced smoothing to preserve detail boundaries
        for i in range(len(focus_maps)):
            gpu_focus[i] = cp.asarray(gaussian(cp.asnumpy(gpu_focus[i]), sigma=sigma))
        
        # Enhanced weight normalization with local max pooling
        weights_sum = cp.sum(gpu_focus, axis=0, keepdims=True)
        max_weights = cp.max(gpu_focus, axis=0, keepdims=True)
        weights = cp.where(weights_sum > 0, 
                         (gpu_focus * max_weights) / (weights_sum + 1e-10),
                         1.0 / len(focus_maps))
        
        # Expand weights for broadcasting
        weights = weights[..., None]
        
        # Blend images
        result = cp.sum(gpu_aligned * weights, axis=0)
        
        # Milder sharpening on GPU
        kernel = cp.array([[-0.5,-0.5,-0.5], [-0.5,5,-0.5], [-0.5,-0.5,-0.5]], dtype=cp.float32)
        result_sharp = cp.zeros_like(result)
        
        # Apply gentler sharpening per channel
        for c in range(3):
            result_sharp[...,c] = cp.real(cp.fft.ifft2(
                cp.fft.fft2(result[...,c]) * cp.fft.fft2(kernel, s=result[...,c].shape)
            ))
        
        # Blend original and sharpened with less intensity
        alpha = 0.4  # Reduced sharpening strength
        result = (1 - alpha) * result + alpha * result_sharp
        
        # Download result from GPU
        result = cp.asnumpy(result)
        
        return np.clip(result, 0, 1)

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

    def process_stack(self, image_paths, color_space='sRGB'):
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
            except Exception as e:
                print(f"Error loading image {path}: {str(e)}")
                raise
                
        # Align images
        print("\nAligning images...")
        try:
            aligned = self._align_images(images)
            print(f"Successfully aligned {len(aligned)} images")
        except Exception as e:
            print(f"Error during image alignment: {str(e)}")
            raise
            
        # Calculate focus measures
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
                
        # Blend images
        print("\nBlending images...")
        try:
            result = self._blend_images(aligned, focus_maps)
            print("Successfully blended images")
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
