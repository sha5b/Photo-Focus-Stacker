import cv2
import numpy as np

from src.core.focus_stacker import FocusStacker, _largest_common_rectangle


def test_common_crop_rectangle_contains_only_valid_pixels():
    mask = np.zeros((30, 40), dtype=bool)
    for row in range(30):
        inset = abs(15 - row) // 4
        mask[row, inset:40 - inset] = True
    rectangle = _largest_common_rectangle(mask)
    assert rectangle is not None
    y0, y1, x0, x1 = rectangle
    assert mask[y0:y1, x0:x1].all()
    assert (y1 - y0) * (x1 - x0) > 600


def test_tiled_focus_map_matches_whole_image_evaluation():
    stacker = FocusStacker(
        focus_measure_method="multiscale", cache_memory_limit_mb=0,
    )
    gray = np.random.default_rng(12).random((2055, 48), dtype=np.float32)
    tiled = stacker._compute_focus_map_tiled(gray)
    whole = stacker._measure_focus_gray(gray, stacker.focus_window_size)
    np.testing.assert_allclose(tiled, whole, atol=1e-6)


def test_direct_map_pipeline_no_longer_crashes(tmp_path):
    rng = np.random.default_rng(8)
    base = (rng.random((96, 96)) * 255).astype(np.uint8)
    paths = []
    for index, sigma in enumerate((0.6, 1.2, 1.8)):
        frame = cv2.GaussianBlur(base, (0, 0), sigma)
        path = tmp_path / f"frame_{index}.png"
        cv2.imwrite(str(path), np.dstack([frame, frame, frame]))
        paths.append(str(path))

    stacker = FocusStacker(
        blend_method="direct_map", focus_measure_method="multiscale",
        num_pyramid_levels=2, cache_memory_limit_mb=0,
    )
    result = stacker.process_stack(paths)
    assert result.ndim == 3 and result.shape[2] == 3
    assert np.isfinite(result).all()
    assert stacker.depth_map.shape == result.shape[:2]
    assert stacker.confidence_map.shape == result.shape[:2]
