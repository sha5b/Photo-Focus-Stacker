# Photo Focus Stacker

Focus stacking GUI for microscopy/macro stacks (developed for the [OpenScan](https://openscan.eu) community).

## Quick start

```bash
uv sync
uv run photostacker
```

1. Click `Load Images`.
2. Keep `Stack Detection = Auto` unless your naming needs `Fixed size` or `Regex`.
3. Optionally enable `Auto Tune`.
4. Choose:
   - `Focus Measure`
   - `Blending Method`
5. Click `Process Stack`.

Settings persist to `%APPDATA%/photo_focus_stacker_settings.json` (Windows).

## Recommended settings

### Best overall (quality)

- **Alignment Pyramid Levels**: `4-5`
- **Alignment Mask Threshold**: `8-12`
- **Focus Window Size**: `5-7`
- **Focus Measure**: `Tenengrad` (good in low-contrast / flat regions)
- **Blend**: `Guided Weighted (Edge-Aware)` or `Luma Weighted + Chroma Pick (MFF)`
- **Sharpen**: `0.0` (sharpen later if needed)

### Fast preview

- **Alignment Pyramid Levels**: `1-2`
- **Focus Window Size**: `7`
- **Focus Measure**: `Laplacian Variance`
- **Blend**: `Direct Map Selection` (fastest, can artifact)

### When results look noisy / haloed

- Increase **Focus Window Size** (e.g. `9`)
- Prefer **Guided Weighted** or **MFF** over **Direct Map**
- Reduce **Sharpen** (keep `0.0` while tuning)

## Parameter reference (what each setting means)

### Alignment Pyramid Levels

Controls Pyramid ECC alignment from coarse-to-fine.

- **Higher (4-6)**
  - More robust to larger shifts/rotations.
  - Slower.
- **Lower (1-2)**
  - Faster.
  - Can fail if the stack has motion.

### Alignment Mask Threshold

Controls how many pixels are used by ECC by building a gradient mask from the reference image.

- **Lower value (stricter)**
  - Uses only the strongest edges.
  - More robust to noise, but can fail if the subject is low-texture.
- **Higher value (more permissive)**
  - Includes more pixels (weaker edges).
  - Can help low-contrast scenes, but can be influenced by noise.

### Focus Window Size

Size of the local window used to aggregate the focus measure (odd numbers only).

- **Smaller (3-5)**
  - Sharper transitions, more detail.
  - More sensitive to noise.
- **Larger (7-11)**
  - Smoother maps (less speckle).
  - Can slightly soften fine focus boundaries.

### Focus Measure

- **Laplacian Variance** (`laplacian_var`)
  - Very common and fast.
  - Can underperform in flat/homogeneous regions.
- **Tenengrad** (`tenengrad`)
  - Uses gradient energy.
  - Often better in low-contrast regions.
- **SML** (`sml`)
  - Sum-modified Laplacian (`|Lx|+|Ly|`).
  - Often a good alternative to Laplacian variance on textured subjects.

### Sharpening Strength

Unsharp mask applied after blending.

- **0.0**: recommended while tuning
- **0.3-0.8**: mild sharpening
- **> 1.0**: can create halos/noise

### Blending Method

- **Weighted Blending**
  - Smooth, stable baseline.
- **Guided Weighted (Edge-Aware)**
  - Smooths weights in an edge-aware way (guided filter when available).
  - Good general-purpose “quality” option.
- **Direct Map Selection**
  - Chooses sharpest frame per pixel (plus top-2 confidence blending in ambiguous areas).
  - Can produce artifacts if focus maps are noisy.
- **Laplacian Pyramid Fusion**
  - Multi-scale fusion; helps preserve detail at multiple scales.
- **Luma Weighted + Chroma Pick (MFF)**
  - Fuses luminance smoothly and selects/blends chroma from the sharpest frames.
  - Often reduces color seams/halos.

## Best / worst setting combinations

### Best starting points

- **General quality**
  - Levels `4`
  - Mask `8-12`
  - Window `5-7`
  - Focus `Tenengrad`
  - Blend `Guided Weighted` or `MFF`
  - Sharpen `0.0`

- **Low-texture subjects**
  - Focus `Tenengrad`
  - Mask threshold slightly higher (more permissive)
  - Blend `Guided Weighted` or `MFF`

### Common “bad” combos

- **Direct Map + small window (3–5)**
  - Noisy focus maps turn into pixel-level artifacts.

- **High sharpening + any artifacts**
  - Sharpening amplifies halos/noise. Tune sharpening last.

- **Very low alignment levels (1) with motion**
  - Misalignment creates ghosting/blur no blend can fix.

## Methods (what each option does)

### Focus Measure

- **Laplacian Variance**: local variance of Laplacian in a window.
- **Tenengrad**: local average of gradient energy `(Gx^2 + Gy^2)`.
- **SML**: sum-modified Laplacian `|Lx| + |Ly|` averaged in a window.

For speed, focus maps are computed on a downscaled grayscale image for large inputs and upsampled back.

### Blending Method

- **Weighted Blending**: soft weights from focus maps.
- **Guided Weighted (Edge-Aware)**: weight maps are edge-aware smoothed (guided/bilateral) before blending.
- **Direct Map Selection**: picks the sharpest source per pixel, with a top-2 confidence blend in ambiguous regions.
- **Laplacian Pyramid Fusion**: multi-scale fusion using Laplacian pyramids.
- **Luma Weighted + Chroma Pick (MFF)**: fuse luminance smoothly and select/blend chroma from the sharpest frames (reduces color seams).

### Performance note (caching)

Alignment (ECC) is the slowest stage. The stacker caches `aligned_images` + `focus_maps` (bounded) so rerunning the same stack with the same alignment/focus settings can reuse intermediates.

## The math / pipeline (high level)

For each stack:

1. **Load** RGB images as float32 `[0,1]`.
2. **Align** each frame to the first frame using pyramid ECC.

   We estimate a transform by maximizing the Enhanced Correlation Coefficient (ECC) objective between the reference and moving image (at multiple resolutions). A gradient-based mask is used to emphasize informative pixels.

   - Primary model: homography `H` (3x3)
   - Fallback model: affine (2x3) if homography ECC fails

3. **Compute a focus map** per aligned frame.

   Each focus measure returns a non-negative field `f_i(x)`:

   - Laplacian variance: `Var(ΔI)` in a local window
   - Tenengrad: local mean of `Gx^2 + Gy^2`
   - SML: local mean of `|Lx| + |Ly|`

4. **Convert focus maps to weights**.

   For weight-based blends we use a stable per-pixel softmax:

   `w_i(x) = exp(beta * f_i(x)) / sum_j exp(beta * f_j(x))`

   This avoids hard seams and keeps weights normalized.

5. **Blend** using the selected method.

   - Weighted / Guided Weighted: `I(x) = sum_i w_i(x) * I_i(x)`
   - Direct Map: choose argmax focus index (with top-2 confidence mixing in ambiguous areas)
   - Laplacian Pyramid: multi-scale fusion of Laplacian pyramids with Gaussian pyramids of weights
   - MFF (luma/chroma): fuse luminance with weights and pick/blend chroma from sharpest frames

6. Optional **unsharp mask** sharpening.

## Install

Python 3.9+.

```bash
git clone https://github.com/sha5b/Photo-Focus-Stacker.git
cd Photo-Focus-Stacker
uv sync
uv run photostacker
```

## Python API

```python
from src.core.focus_stacker import FocusStacker

stacker = FocusStacker(
    num_pyramid_levels=4,
    gradient_threshold=10,
    focus_window_size=7,
    focus_measure_method="tenengrad",
    blend_method="guided_weighted",
    sharpen_strength=0.0,
)

result = stacker.process_stack(["img0.jpg", "img1.jpg", "img2.jpg"], color_space="sRGB")
```

### `FocusStacker` options

- `num_pyramid_levels`: 1–6
- `gradient_threshold`: 1–100
- `focus_window_size`: 3–21 (odd)
- `focus_measure_method`: `"laplacian_var" | "tenengrad" | "sml"`
- `blend_method`:
  - `"weighted"`
  - `"guided_weighted"`
  - `"direct_map"`
  - `"laplacian_pyramid"`
  - `"luma_weighted_chroma_pick"`
- `sharpen_strength`: 0.0–3.0

## License

Non-Commercial Open Source License (NCOSL). See `LICENSE`.
