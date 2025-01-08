import cv2
import numpy as np
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
    
    def __init__(self):
        """Initialize focus stacker with default parameters"""
        # Parameters for focus measure
        self.window_size = 15  # Size of window for focus measure
        self.gaussian_blur = 2.0  # Blur sigma for noise reduction
        
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
        Align images using phase correlation
        @param images List of images to align
        @return List of aligned images
        """
        reference = images[0]
        aligned = [reference]
        
        for img in images[1:]:
            # Convert to grayscale for alignment
            ref_gray = cv2.cvtColor((reference * 255).astype(np.uint8), 
                                  cv2.COLOR_RGB2GRAY)
            img_gray = cv2.cvtColor((img * 255).astype(np.uint8), 
                                  cv2.COLOR_RGB2GRAY)
            
            # Calculate shift
            shift, error, _ = phase_cross_correlation(ref_gray, img_gray)
            
            # Apply shift using affine transform
            M = np.float32([[1, 0, -shift[1]], [0, 1, -shift[0]]])
            aligned_img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
            aligned.append(aligned_img)
            
        return aligned

    def _focus_measure(self, img):
        """
        Calculate focus measure using Laplacian variance
        @param img Input image
        @return Focus measure map
        """
        if len(img.shape) == 3:
            img = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            img = (img * 255).astype(np.uint8)
            
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(img, (3, 3), self.gaussian_blur)
        
        # Calculate Laplacian
        laplacian = cv2.Laplacian(blurred, cv2.CV_64F)
        
        # Calculate local variance
        focus_map = np.zeros_like(laplacian, dtype=np.float32)
        half_window = self.window_size // 2
        
        for i in range(half_window, img.shape[0] - half_window):
            for j in range(half_window, img.shape[1] - half_window):
                window = laplacian[i-half_window:i+half_window+1,
                                 j-half_window:j+half_window+1]
                focus_map[i, j] = np.var(window)
                
        return focus_map

    def _blend_images(self, aligned_images, focus_maps):
        """
        Blend images based on focus measures
        @param aligned_images List of aligned images
        @param focus_maps List of focus measure maps
        @return Blended image
        """
        focus_maps = np.array(focus_maps)
        max_focus = np.argmax(focus_maps, axis=0)
        
        result = np.zeros_like(aligned_images[0])
        for i in range(len(aligned_images)):
            mask = (max_focus == i)
            result[mask] = aligned_images[i][mask]
            
        # Apply guided filter for smooth transitions
        result = self._guided_filter(result, max_focus.astype(np.float32))
        return np.clip(result, 0, 1)

    def _guided_filter(self, img, guide, radius=10, eps=1e-6):
        """
        Apply guided filter for smooth blending
        @param img Input image
        @param guide Guidance image
        @param radius Filter radius
        @param eps Regularization parameter
        @return Filtered image
        """
        mean_guide = cv2.boxFilter(guide, -1, (radius, radius))
        mean_img = cv2.boxFilter(img, -1, (radius, radius))
        corr_guide = cv2.boxFilter(guide * guide, -1, (radius, radius))
        corr_img_guide = cv2.boxFilter(img * guide, -1, (radius, radius))
        
        var_guide = corr_guide - mean_guide * mean_guide
        cov_img_guide = corr_img_guide - mean_img * mean_guide
        
        a = cov_img_guide / (var_guide + eps)
        b = mean_img - a * mean_guide
        
        mean_a = cv2.boxFilter(a, -1, (radius, radius))
        mean_b = cv2.boxFilter(b, -1, (radius, radius))
        
        return mean_a * guide + mean_b

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
            
        # Load images
        images = []
        for i, path in enumerate(image_paths):
            images.append(self._load_image(path))
            if progress_callback:
                progress_callback(int((i + 1) / len(image_paths) * 30))
                
        # Align images
        aligned = self._align_images(images)
        if progress_callback:
            progress_callback(50)
            
        # Calculate focus measures
        focus_maps = []
        for i, img in enumerate(aligned):
            focus_maps.append(self._focus_measure(img))
            if progress_callback:
                progress_callback(50 + int((i + 1) / len(aligned) * 25))
                
        # Blend images
        result = self._blend_images(aligned, focus_maps)
        if progress_callback:
            progress_callback(90)
            
        # Convert to target color space
        if color_space != 'sRGB':
            result = self._convert_color_space(result, color_space)
            
        if progress_callback:
            progress_callback(100)
            
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
        if format_name in ['TIFF (16-bit)', 'PNG (16-bit)']:
            # Convert to 16-bit
            img_16bit = img_as_uint(img)
            PIL.Image.fromarray(img_16bit).save(
                path,
                format='TIFF' if format_name == 'TIFF (16-bit)' else 'PNG'
            )
        else:
            # Save as 8-bit JPEG
            img_8bit = (img * 255).astype(np.uint8)
            PIL.Image.fromarray(img_8bit).save(path, format='JPEG', quality=95)
