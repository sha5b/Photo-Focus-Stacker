#!/usr/bin/env python3

import cv2
import numpy as np
import os
import re
import PIL.Image
import PIL.ImageCms

# --- Color Profile Management ---

_COLOR_PROFILES = {}

def init_color_profiles():
    """Initializes common color profiles."""
    global _COLOR_PROFILES
    if not _COLOR_PROFILES: # Initialize only once
        try:
            _COLOR_PROFILES['sRGB'] = PIL.ImageCms.createProfile('sRGB')
            print("Initialized sRGB color profile.")
        except Exception as e:
            print(f"Warning: Could not initialize color profiles: {e}")
    return _COLOR_PROFILES

def get_color_profile(name='sRGB'):
    """Gets a pre-initialized color profile."""
    profiles = init_color_profiles() # Ensure initialized
    return profiles.get(name)

# --- Image Loading ---

def load_image(path):
    """Loads an image using OpenCV, converts to RGB float32 [0, 1]."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to load image: {path}")
    # Convert BGR (OpenCV default) to RGB and normalize to [0, 1] float
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

# --- Image Saving ---

def save_image(img, path, format='JPEG', quality=95, color_space='sRGB'):
    """
    Saves the processed image (float32 [0, 1]) to a file.
    Handles conversion to uint8 and applies format-specific options.
    Assumes image is in sRGB for saving unless converted beforehand.
    """
    print(f"\nSaving image...")
    print(f"  Path: {path}")
    print(f"  Format: {format}, Quality: {quality if format.upper() == 'JPEG' else 'N/A'}, Target Color Space (Assumed): {color_space}")

    try:
        # Convert float32 [0, 1] to uint8 [0, 255]
        img_8bit = np.clip(img * 255.0 + 0.5, 0, 255).astype(np.uint8)
        pil_img = PIL.Image.fromarray(img_8bit, mode='RGB')

        # Add saving options based on format
        save_options = {}
        if format.upper() == 'JPEG':
            save_options['quality'] = quality
            save_options['optimize'] = True
        elif format.upper() == 'PNG':
            save_options['compress_level'] = 6 # Example
        elif format.upper() == 'TIFF':
            save_options['compression'] = 'tiff_lzw' # Example

        pil_img.save(path, format=format.upper(), **save_options)
        print(f"Successfully saved image.")

    except Exception as e:
        print(f"ERROR saving image: {e}")
        raise

# --- Stack Splitting ---

def split_into_stacks(image_paths, stack_size=0):
    """
    Splits a list of image paths into stacks based on filename patterns.
    Assumes filenames contain sequence numbers.
    @param image_paths: List of full paths to images.
    @param stack_size: Expected number of images per stack (0 for auto-detect).
    @return: A list of tuples, where each tuple is (base_name, list_of_paths).
    """
    stacks_dict = {}
    # Use a pattern for the format: prefix_stackID-imageID.ext
    # e.g., alienshape_0_1-0.jpg -> base_name = alienshape_0_1
    pattern = r'^(.*_\d+)-(\d+)$' # Group 1: base_name (prefix_stackID), Group 2: image_index

    print(f"\nAttempting to split images into stacks using pattern: '{pattern}'...")
    for path in image_paths:
        filename = os.path.basename(path)
        name, _ = os.path.splitext(filename)

        match = re.match(pattern, name, re.IGNORECASE) # Try matching the specific pattern

        if match:
            base_name = match.group(1).strip() # Group 1 is the base name including the stack ID
            if not base_name:
                base_name = "default_stack" # Fallback if base name is empty

            if base_name not in stacks_dict:
                stacks_dict[base_name] = []
            stacks_dict[base_name].append(path)
        else:
            # Fallback if the specific pattern doesn't match
            print(f"  Warning: Could not determine stack for file: {filename}. Adding to 'default_stack'.")
            if "default_stack" not in stacks_dict: stacks_dict["default_stack"] = []
            stacks_dict["default_stack"].append(path)

    # Sort images within each stack naturally (handles numbers better)
    try:
        import natsort
        for base_name in stacks_dict:
            stacks_dict[base_name] = natsort.natsorted(stacks_dict[base_name])
        print("Sorted images within stacks using natsort.")
    except ImportError:
        print("Warning: natsort package not found. Using basic sort for images within stacks.")
        for base_name in stacks_dict:
            stacks_dict[base_name].sort() # Basic sort as fallback

    # Convert dictionary to list of tuples: [(base_name, paths), ...]
    stack_items = list(stacks_dict.items())

    # Optional: Check if detected stacks match the expected size
    # Operate on stack_items now
    if stack_size > 0:
        print(f"Checking if stacks match expected size: {stack_size}")
        valid_stack_items = []
        for i, (base_name, paths) in enumerate(stack_items):
            if len(paths) == stack_size:
                print(f"  Stack '{base_name}': Found {len(paths)} images (Correct size).")
                valid_stack_items.append((base_name, paths))
            else:
                print(f"  Warning: Stack '{base_name}': Found {len(paths)} images, expected {stack_size}. Skipping this stack.")
        stack_items = valid_stack_items # Keep only stacks with the correct size

    # Sort stacks based on the first image path within each stack for consistent order
    stack_items.sort(key=lambda item: item[1][0] if item[1] else "")

    print("\nDetected Stacks:")
    if not stack_items:
        print("  No valid stacks found.")
    for i, (base_name, paths) in enumerate(stack_items):
        if paths: print(f"  Stack {i+1} ('{base_name}'): {len(paths)} images starting with {os.path.basename(paths[0])}")

    return stack_items # Return list of (base_name, paths) tuples


# --- Color Space Conversion ---

def convert_color_space(img, target_space, source_space='sRGB'):
    """
    Convert image color space using ICC profiles.
    Assumes input image (float32 [0, 1]) is in source_space.
    """
    print(f"Attempting color space conversion from {source_space} to {target_space}...")
    if source_space == target_space:
        print("Source and target spaces are the same. Skipping conversion.")
        return img

    # Convert float32 [0, 1] to uint8 [0, 255] for PIL/ImageCms
    pil_img = PIL.Image.fromarray(np.clip(img * 255.0 + 0.5, 0, 255).astype('uint8'), mode='RGB')

    source_profile = get_color_profile(source_space)
    target_profile = get_color_profile(target_space)

    if not source_profile:
        print(f"Warning: Could not find source profile '{source_space}'. Skipping conversion.")
        return img
    if not target_profile:
        print(f"Warning: Could not find target profile '{target_space}'. Skipping conversion.")
        return img

    try:
        transform = PIL.ImageCms.buildTransformFromOpenProfiles(
            source_profile, target_profile, "RGB", "RGB",
            renderingIntent=PIL.ImageCms.INTENT_PERCEPTUAL # Or RELATIVE_COLORIMETRIC
        )
        converted_pil = PIL.ImageCms.applyTransform(pil_img, transform)

        # Convert back to float32 [0, 1]
        converted_img = np.array(converted_pil).astype(np.float32) / 255.0
        print("Color space conversion successful.")
        return converted_img
    except PIL.ImageCms.PyCMSError as e:
        print(f"Error applying color space transform: {e}. Returning original image.")
        return img
    except Exception as e:
        print(f"Unexpected error during color space conversion: {e}. Returning original image.")
        return img
