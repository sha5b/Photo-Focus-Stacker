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
            # Add other common profiles if needed later
            # _COLOR_PROFILES['AdobeRGB'] = PIL.ImageCms.createProfile('AdobeRGB')
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
            # Note: ICC profile embedding might be needed for color accuracy if not sRGB
            # if color_space != 'sRGB':
            #     profile = get_color_profile(color_space)
            #     if profile: save_options['icc_profile'] = profile.tobytes()
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
    @return: A list of lists, where each inner list is a stack of image paths.
    """
    stacks_dict = {}
    # Try common patterns to extract base name and sequence number
    # Prioritize patterns that capture longer base names
    patterns = [
        r'^(.*?)[\s_-]?(\d{1,5})$',          # BaseName_### or BaseName ###
        r'^(.*?)[\s_-]?(\d{1,5})[\s_-]',     # BaseName_###_Suffix
        r'^(\d{1,5})[\s_-]?(.*?)$',          # ###_BaseName
    ]

    print("\nAttempting to split images into stacks based on filenames...")
    for path in image_paths:
        filename = os.path.basename(path)
        name, _ = os.path.splitext(filename)

        matched = False
        for pattern in patterns:
            match = re.match(pattern, name, re.IGNORECASE) # Ignore case for matching
            if match:
                groups = match.groups()
                # Determine base name and number based on pattern structure
                if pattern == patterns[0] or pattern == patterns[1]: # BaseName first
                    base_name = groups[0].strip() if groups[0] else "default_stack"
                    # seq_num = int(groups[1]) # Could use sequence number for sorting later if needed
                elif pattern == patterns[2]: # Number first
                    base_name = groups[1].strip() if groups[1] else "default_stack"
                    # seq_num = int(groups[0])
                else: # Fallback
                     base_name = "default_stack"

                # Normalize base name (remove trailing numbers/separators if pattern was too greedy)
                base_name = re.sub(r'[\s_-]*\d*$', '', base_name).strip()
                if not base_name: base_name = "default_stack"

                if base_name not in stacks_dict:
                    stacks_dict[base_name] = []
                stacks_dict[base_name].append(path)
                matched = True
                # print(f"  Matched '{filename}' to stack '{base_name}' using pattern {patterns.index(pattern)}") # Debug
                break # Stop after first successful pattern match

        if not matched:
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

    # Convert dictionary to list of lists (stacks)
    stacks = list(stacks_dict.values())

    # Optional: Check if detected stacks match the expected size
    if stack_size > 0:
        print(f"Checking if stacks match expected size: {stack_size}")
        valid_stacks = []
        for i, stack in enumerate(stacks):
            base = os.path.basename(stack[0])[:20]+"..." if stack else "N/A"
            if len(stack) == stack_size:
                print(f"  Stack {i+1} ({base}): Found {len(stack)} images (Correct size).")
                valid_stacks.append(stack)
            else:
                print(f"  Warning: Stack {i+1} ({base}): Found {len(stack)} images, expected {stack_size}. Skipping this stack.")
        stacks = valid_stacks # Keep only stacks with the correct size if stack_size is specified

    # Sort stacks based on the first image path for consistent order
    stacks.sort(key=lambda x: x[0] if x else "")

    print("\nDetected Stacks:")
    if not stacks:
        print("  No valid stacks found.")
    for i, stack in enumerate(stacks):
        if stack: print(f"  Stack {i+1}: {len(stack)} images starting with {os.path.basename(stack[0])}")

    return stacks


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
        # Ensure profiles are in the correct format if loaded differently later
        # source_profile_obj = PIL.ImageCms.getOpenProfile(source_profile)
        # target_profile_obj = PIL.ImageCms.getOpenProfile(target_profile)

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
