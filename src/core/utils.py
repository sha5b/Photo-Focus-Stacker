#!/usr/bin/env python3

# Context: Core utility functions for Photo Focus Stacker
# Purpose: Provide shared helpers for image I/O and color conversion used by the stacking pipeline.
# Notes: Used by `src.core.focus_stacker` and the UI for saving output.

import json
import os
import re
import struct
import zlib
from dataclasses import dataclass
from io import BytesIO
from typing import Optional

import cv2
import numpy as np
import PIL.Image
import PIL.ImageCms
import PIL.ImageOps

# --- Color Profile Management ---

_COLOR_PROFILES = {}


@dataclass(frozen=True)
class ImageMetadata:
    """Metadata carried from the selected reference frame to the output."""
    source_path: str = ""
    icc_profile: Optional[bytes] = None
    exif: Optional[bytes] = None
    dpi: Optional[tuple] = None
    source_bit_depth: int = 8


@dataclass(frozen=True)
class LoadedImage:
    image: np.ndarray
    metadata: ImageMetadata

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


def get_color_profile_bytes(name='sRGB') -> Optional[bytes]:
    profile = get_color_profile(name)
    if profile is None:
        return None
    try:
        return PIL.ImageCms.ImageCmsProfile(profile).tobytes()
    except Exception:
        return None

# --- Image Loading ---

def load_image_with_metadata(path: str) -> LoadedImage:
    """Load RGB data without discarding 16-bit precision and collect metadata."""
    extension = os.path.splitext(path)[1].lower()
    if extension in {".dng", ".nef", ".cr2", ".cr3", ".arw", ".orf", ".rw2", ".raf"}:
        try:
            import rawpy
        except ImportError as exc:
            raise ValueError("RAW input requires the optional 'raw' dependency (rawpy).") from exc
        with rawpy.imread(path) as raw:
            rgb = raw.postprocess(
                output_bps=16, gamma=(1, 1), no_auto_bright=True,
                use_camera_wb=True, output_color=rawpy.ColorSpace.sRGB,
            )
        return LoadedImage(
            rgb.astype(np.float32) / 65535.0,
            ImageMetadata(source_path=path, icc_profile=get_color_profile_bytes("sRGB"), source_bit_depth=16),
        )

    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Failed to load image: {path}")
    icc_profile = None
    exif = None
    dpi = None
    pil_rgb = None
    try:
        with PIL.Image.open(path) as pil_img:
            source_icc = pil_img.info.get("icc_profile")
            icc_profile = source_icc
            dpi = pil_img.info.get("dpi")
            exif_obj = pil_img.getexif()
            if 274 in exif_obj:
                del exif_obj[274]
            exif = exif_obj.tobytes() if exif_obj else None
            if img.dtype == np.uint8:
                pil_rgb = PIL.ImageOps.exif_transpose(pil_img).convert("RGB")
                if source_icc:
                    try:
                        source_profile = PIL.ImageCms.getOpenProfile(BytesIO(source_icc))
                        target_profile = get_color_profile("sRGB")
                        pil_rgb = PIL.ImageCms.profileToProfile(
                            pil_rgb, source_profile, target_profile, outputMode="RGB"
                        )
                        icc_profile = PIL.ImageCms.ImageCmsProfile(target_profile).tobytes()
                    except Exception as exc:
                        print(f"Warning: ICC conversion failed for {path}: {exc}")
                        icc_profile = source_icc
                else:
                    icc_profile = None
    except Exception as exc:
        print(f"Warning: Could not read image metadata for {path}: {exc}")

    if icc_profile is None:
        icc_profile = get_color_profile_bytes("sRGB")

    if pil_rgb is not None:
        img = np.asarray(pil_rgb)
    elif img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if np.issubdtype(img.dtype, np.integer):
        dtype_max = float(np.iinfo(img.dtype).max)
        bit_depth = int(np.iinfo(img.dtype).bits)
        image = img.astype(np.float32) / dtype_max
    elif np.issubdtype(img.dtype, np.floating):
        bit_depth = 32
        image = np.nan_to_num(img.astype(np.float32), nan=0.0, posinf=1.0, neginf=0.0)
        if float(np.max(image)) > 1.0:
            image /= max(float(np.max(image)), 1.0)
    else:
        raise ValueError(f"Unsupported image data type: {img.dtype}")

    return LoadedImage(
        np.clip(image, 0.0, 1.0).astype(np.float32),
        ImageMetadata(path, icc_profile, exif, dpi, bit_depth),
    )


def load_image(path):
    """Backward-compatible image loader returning RGB float32 in [0, 1]."""
    return load_image_with_metadata(path).image

# --- Image Saving ---

def save_image(img, path, format='JPEG', quality=95, color_space='sRGB',
               bit_depth=None, metadata: Optional[ImageMetadata] = None):
    """
    Saves the processed image (float32 [0, 1]) to a file.
    Handles conversion to uint8 and applies format-specific options.
    Assumes image is in sRGB for saving unless converted beforehand.
    """
    print("\nSaving image...")
    print(f"  Path: {path}")
    print(f"  Format: {format}, Quality: {quality if format.upper() == 'JPEG' else 'N/A'}, Target Color Space (Assumed): {color_space}")

    try:
        fmt = format.upper()
        if bit_depth is None:
            bit_depth = 16 if fmt in {"PNG", "TIFF", "TIF"} else 8
        bit_depth = 8 if fmt == "JPEG" else int(bit_depth)
        if bit_depth not in (8, 16):
            raise ValueError("Output bit depth must be 8 or 16.")

        clipped = np.clip(np.asarray(img, dtype=np.float32), 0.0, 1.0)
        if bit_depth == 16:
            img_16bit = np.rint(clipped * 65535.0).astype(np.uint16)
            if fmt in {"TIFF", "TIF"}:
                import tifffile
                extra_tags = []
                if metadata and metadata.icc_profile:
                    extra_tags.append((34675, "B", len(metadata.icc_profile), metadata.icc_profile, False))
                tifffile.imwrite(
                    path, img_16bit, photometric="rgb", compression="deflate",
                    metadata=None, extratags=extra_tags,
                )
            elif fmt == "PNG":
                ok = cv2.imwrite(path, cv2.cvtColor(img_16bit, cv2.COLOR_RGB2BGR))
                if not ok:
                    raise ValueError(f"Failed to encode 16-bit PNG: {path}")
                if metadata and metadata.icc_profile:
                    _inject_png_icc(path, metadata.icc_profile)
            else:
                raise ValueError(f"16-bit output is not supported for {fmt}.")
            print("Successfully saved 16-bit image.")
            return

        img_8bit = np.rint(clipped * 255.0).astype(np.uint8)
        pil_img = PIL.Image.fromarray(img_8bit, mode='RGB')

        # Add saving options based on format
        save_options = {}
        if fmt == 'JPEG':
            save_options['quality'] = quality
            save_options['optimize'] = True
        elif fmt == 'PNG':
            save_options['compress_level'] = 6 # Example
        elif fmt == 'TIFF':
            save_options['compression'] = 'tiff_lzw' # Example

        if metadata is not None:
            if metadata.icc_profile:
                save_options["icc_profile"] = metadata.icc_profile
            if metadata.exif:
                save_options["exif"] = _updated_exif_dimensions(
                    metadata.exif, pil_img.width, pil_img.height
                )
            if metadata.dpi:
                save_options["dpi"] = metadata.dpi

        pil_img.save(path, format=fmt, **save_options)
        print("Successfully saved image.")

    except Exception as e:
        print(f"ERROR saving image: {e}")
        raise


def _updated_exif_dimensions(exif_bytes: bytes, width: int, height: int) -> bytes:
    try:
        exif = PIL.Image.Exif()
        exif.load(exif_bytes)
        exif[40962] = int(width)   # PixelXDimension
        exif[40963] = int(height)  # PixelYDimension
        exif.pop(274, None)        # Orientation was applied during loading.
        return exif.tobytes()
    except Exception:
        return exif_bytes


def _inject_png_icc(path: str, icc_profile: bytes) -> None:
    """Insert a standards-compliant compressed iCCP chunk after PNG IHDR."""
    with open(path, "rb") as handle:
        data = handle.read()
    signature = b"\x89PNG\r\n\x1a\n"
    if not data.startswith(signature) or len(data) < 33:
        raise ValueError("Encoder did not produce a valid PNG file.")
    ihdr_length = struct.unpack(">I", data[8:12])[0]
    insertion = 8 + 12 + ihdr_length
    payload = b"sRGB ICC\x00\x00" + zlib.compress(bytes(icc_profile), level=9)
    chunk_type = b"iCCP"
    chunk = struct.pack(">I", len(payload)) + chunk_type + payload
    chunk += struct.pack(">I", zlib.crc32(chunk_type + payload) & 0xFFFFFFFF)
    with open(path, "wb") as handle:
        handle.write(data[:insertion] + chunk + data[insertion:])


def srgb_to_linear(image: np.ndarray) -> np.ndarray:
    image = np.clip(np.asarray(image, dtype=np.float32), 0.0, 1.0)
    return np.where(image <= 0.04045, image / 12.92, ((image + 0.055) / 1.055) ** 2.4).astype(np.float32)


def linear_to_srgb(image: np.ndarray) -> np.ndarray:
    image = np.clip(np.asarray(image, dtype=np.float32), 0.0, 1.0)
    return np.where(image <= 0.0031308, image * 12.92, 1.055 * np.power(image, 1.0 / 2.4) - 0.055).astype(np.float32)


def match_stack_exposure(images, valid_masks, reference_index: int):
    """Match robust linear-light luminance ranges while retaining source color."""
    if not images:
        return images
    linear_reference = srgb_to_linear(images[reference_index])
    ref_luma = cv2.cvtColor(linear_reference, cv2.COLOR_RGB2GRAY)
    ref_values = ref_luma[np.asarray(valid_masks[reference_index]) > 0.5]
    ref_lo, ref_hi = np.percentile(ref_values, (5.0, 95.0))
    matched = images
    for index, image in enumerate(images):
        if index == reference_index:
            continue
        linear = srgb_to_linear(image)
        luma = cv2.cvtColor(linear, cv2.COLOR_RGB2GRAY)
        values = luma[np.asarray(valid_masks[index]) > 0.5]
        if values.size == 0:
            continue
        lo, hi = np.percentile(values, (5.0, 95.0))
        gain = float(np.clip((ref_hi - ref_lo) / max(hi - lo, 1e-6), 0.5, 2.0))
        offset = float(ref_lo - lo * gain)
        corrected = np.clip(linear * gain + offset, 0.0, 1.0)
        matched[index] = linear_to_srgb(corrected)
    return matched


def save_focus_maps(depth_map, confidence_map, output_path: str) -> None:
    """Save an exact source-index map and a normalized confidence map as 16-bit PNG."""
    if depth_map is None or confidence_map is None:
        return
    stem, _ = os.path.splitext(output_path)
    depth = np.asarray(depth_map, dtype=np.uint16)
    confidence = np.rint(np.clip(confidence_map, 0.0, 1.0) * 65535.0).astype(np.uint16)
    if not cv2.imwrite(f"{stem}_depth.png", depth):
        raise ValueError("Failed to save the depth map.")
    if not cv2.imwrite(f"{stem}_confidence.png", confidence):
        raise ValueError("Failed to save the confidence map.")


def save_alignment_report(diagnostics, output_path: str, source_paths=None) -> None:
    """Write per-frame registration quality next to the rendered image."""
    if diagnostics is None:
        return
    stem, _ = os.path.splitext(output_path)
    rows = []
    for item in diagnostics:
        index = int(item.source_index)
        rows.append({
            "source_index": index,
            "source_path": source_paths[index] if source_paths and index < len(source_paths) else None,
            "accepted": bool(item.accepted),
            "model": str(item.model),
            "correlation": float(item.correlation),
            "overlap_ratio": float(item.overlap_ratio),
            "message": str(item.message),
            "transform": [list(row) for row in item.transform],
        })
    with open(f"{stem}_alignment.json", "w", encoding="utf-8") as handle:
        json.dump({"frames": rows}, handle, indent=2)

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
            if "default_stack" not in stacks_dict:
                stacks_dict["default_stack"] = []
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
        if paths:
            print(f"  Stack {i+1} ('{base_name}'): {len(paths)} images starting with {os.path.basename(paths[0])}")

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
