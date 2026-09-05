import numpy as np
import pytest
import cv2

from src.core import blending
from src.core.focus_measure import measure_multiscale_map


@pytest.fixture
def stack():
    rng = np.random.default_rng(42)
    images = [rng.random((33, 35, 3), dtype=np.float32) for _ in range(3)]
    maps = [rng.random((33, 35), dtype=np.float32) for _ in range(3)]
    masks = [np.ones((33, 35), dtype=np.float32) for _ in range(3)]
    masks[1][:, :3] = 0
    return images, maps, masks


@pytest.mark.parametrize("method", [
    "weighted", "guided_weighted", "direct_map",
    "laplacian_pyramid", "luma_weighted_chroma_pick",
])
def test_blend_methods_return_finite_rgb(method, stack):
    images, maps, masks = stack
    depth, _ = blending.build_focus_depth_map(maps, masks)
    if method == "direct_map":
        result = blending.blend_direct_map(images, depth, maps, masks)
    elif method == "laplacian_pyramid":
        result = blending.blend_laplacian_pyramid(images, maps, 4, masks)
    else:
        result = getattr(blending, f"blend_{method}")(images, maps, masks)
    assert result.shape == images[0].shape
    assert result.dtype == np.float32
    assert np.isfinite(result).all()
    assert 0.0 <= float(result.min()) <= float(result.max()) <= 1.0


def test_depth_map_removes_isolated_label():
    maps = [np.ones((25, 25), np.float32), np.zeros((25, 25), np.float32)]
    maps[1][12, 12] = 100
    depth, confidence = blending.build_focus_depth_map(maps, min_region_area=16)
    assert depth[12, 12] == 0
    assert confidence.shape == depth.shape


def test_multiscale_depth_recovers_two_focus_planes():
    y, x = np.mgrid[:96, :96]
    texture = (((x // 3) + (y // 3)) % 2).astype(np.float32)
    blurred = cv2.GaussianBlur(texture, (0, 0), 3.0)
    first = texture.copy()
    first[:, 48:] = blurred[:, 48:]
    second = texture.copy()
    second[:, :48] = blurred[:, :48]
    maps = [measure_multiscale_map(first), measure_multiscale_map(second)]
    depth, confidence = blending.build_focus_depth_map(maps, min_region_area=32)
    assert np.mean(depth[:, 8:40] == 0) > 0.90
    assert np.mean(depth[:, 56:88] == 1) > 0.90
    assert float(np.mean(confidence[:, 8:40])) > 0.2
