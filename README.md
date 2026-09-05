# Photo Focus Stacker

High-quality focus stacking for macro photography, microscopy, and per-view
photogrammetry image stacks. The application includes a PyQt GUI and a headless
command-line workflow.

## Highlights

- Pyramid ECC registration with translation, Euclidean, affine, and optional
  homography models.
- Middle-frame reference, exposure-normalized registration, transform quality
  checks, automatic fallback, and rejection of unreliable frames.
- Real warp-validity masks: generated border pixels never participate in fusion.
- Laplacian variance, Tenengrad, SML, and noise-tolerant multi-scale focus maps.
- Spatially regularized focus-depth map and per-pixel confidence map.
- Weighted, guided edge-aware, direct-map, Laplacian-pyramid, and luma/chroma
  fusion modes.
- Linear-light blending by default.
- Robust linear-light exposure normalization across frames.
- 8-bit and 16-bit input; 16-bit TIFF and PNG output.
- ICC handling, compatible EXIF preservation, optional RAW input, and alignment
  diagnostics.
- Photogrammetry preset that constrains alignment geometry and preserves the
  reference canvas.
- File-change-aware, memory-budgeted intermediate cache.
- Overlap-safe tiled full-resolution focus analysis for large images.
- Configurable parallel focus-map workers.
- Responsive cancellation and per-stack processing progress.

## Install and run

Python 3.9 or newer and [uv](https://docs.astral.sh/uv/) are recommended.

```bash
uv sync
uv run photostacker
```

For tests or RAW camera files:

```bash
uv sync --extra test
uv sync --extra raw
uv run pytest
```

The project intentionally uses `opencv-contrib-python-headless`: PyQt5 owns the
GUI, and installing a non-headless OpenCV wheel alongside it can redirect Qt to
OpenCV's incompatible `xcb` plugin. On a GNOME Wayland session the launcher
selects Qt's Wayland backend automatically; an explicit `QT_QPA_PLATFORM` value
is always respected.

RAW support uses `rawpy` and covers common DNG, NEF, CR2/CR3, ARW, ORF, RW2,
and RAF files. RAW development uses camera white balance, disables automatic
brightening, and emits 16-bit RGB working data.

## GUI workflow

1. Load at least two images from one or more focus stacks.
2. Use automatic stack detection, a fixed stack size, or a custom regular
   expression.
3. Choose a preset.
4. Select TIFF or PNG with 16-bit output for maximum quality.
5. Optionally export depth, confidence, and alignment diagnostics.
6. Process the stack and inspect the result preview.

Unreliable frames are excluded. If fewer than two frames survive alignment,
processing stops with an explicit error instead of silently blending an
unaligned image.

## Presets

### Best Quality

- Full-resolution multi-scale focus analysis
- Five-level affine ECC registration
- Guided edge-aware fusion
- Linear-light blending
- Common-valid-area crop

### Photogrammetry

- Euclidean registration only: translation, rotation, and uniform scale
- Full-resolution multi-scale focus analysis
- Guided edge-aware fusion in linear light
- Reference-frame canvas is preserved
- No projective homography or automatic crop

Use one focus stack per camera pose. All poses should use the same preset,
exposure treatment, output dimensions, and naming convention. Exporting the
alignment report is recommended so rejected frames can be audited.

### Balanced

- 2000-pixel focus-analysis limit
- Three-level affine alignment
- Weighted fusion

### Fast Preview

- 1200-pixel focus-analysis limit
- One-level alignment
- Direct-map fusion
- Linear-light conversion disabled

## Command line

When image paths are supplied, `photostacker` runs without launching the GUI:

```bash
uv run photostacker frame_001.tif frame_002.tif frame_003.tif \
  --preset quality --bit-depth 16 --export-maps \
  --output stacked.tif
```

For photogrammetry:

```bash
uv run photostacker pose07_focus_*.tif \
  --preset photogrammetry --bit-depth 16 --export-maps \
  --output pose07_stacked.tif
```

The shell expansion order becomes the focus-depth order. Use zero-padded frame
numbers or explicitly provide the paths in focus order.

## Output files

- `name.tif`, `name.png`, or `name.jpg`: rendered stack
- `name_depth.png`: exact zero-based source-frame index, stored as uint16
- `name_confidence.png`: normalized focus confidence, stored as uint16
- `name_alignment.json`: accepted/rejected status, transform model, ECC
  matrix, correlation, and valid overlap for every source frame

JPEG is always 8-bit. TIFF and PNG default to 16-bit. TIFF supports embedded ICC
profiles; 16-bit PNG output receives a standards-compliant iCCP chunk. JPEG and
8-bit Pillow outputs preserve compatible EXIF metadata from the reference frame,
with EXIF orientation removed after pixels have been normalized.

## Important controls

- **Alignment model**: affine is the general macro default. Euclidean is safer
  when downstream camera geometry matters. Homography is intended only for
  stacks that genuinely require projective correction.
- **Alignment mask threshold**: lower values retain only stronger edges; higher
  values include more low-contrast structure.
- **Focus analysis max dimension**: zero means full resolution. A limit trades
  small-detail accuracy for speed.
- **Focus window size**: smaller windows preserve narrow details; larger windows
  reduce speckle.
- **Cache memory limit**: oversized stacks are not cached. Set zero to disable
  intermediate caching completely.
- **Crop to common valid area**: removes warp borders. Photogrammetry mode keeps
  the canvas fixed and uses validity-weighted filling instead.

## Python API

```python
from src.core.focus_stacker import FocusStacker

stacker = FocusStacker(
    num_pyramid_levels=5,
    alignment_model="affine",
    focus_measure_method="multiscale",
    blend_method="guided_weighted",
    focus_analysis_max_dim=0,
    crop_to_common_area=True,
    linear_light_blending=True,
    cache_memory_limit_mb=512,
)

result = stacker.process_stack(["frame_001.tif", "frame_002.tif"])
stacker.save_image(result, "stacked.tif", format="TIFF", bit_depth=16)

# Available after processing:
depth = stacker.depth_map
confidence = stacker.confidence_map
diagnostics = stacker.alignment_diagnostics
```

## Quality recommendations

- Capture RAW or 16-bit TIFF when possible.
- Lock exposure, white balance, ISO, and lighting across a stack.
- Move focus in one direction with adequate overlap between focus planes.
- Keep sharpening at zero until alignment and depth selection look correct.
- Inspect confidence and alignment output before feeding images into a
  photogrammetry reconstruction.
- Homography can hide capture movement by distorting geometry; use it only when
  the output is not expected to retain a calibrated camera model.

## Development checks

```bash
uv sync --extra test
uv run pytest -q
uv run python -m compileall -q src main.py
```

The tests cover every blend path, the former Direct Map crash, regularized depth
selection, alignment validation and masks, natural stack grouping, 16-bit TIFF
round trips, metadata behavior, color-transfer round trips, and a complete
focus-stacking pipeline.

## License

Non-Commercial Open Source License (NCOSL). See [LICENSE](LICENSE).
