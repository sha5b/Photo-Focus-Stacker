import numpy as np
import pytest

from src.core.alignment import align_images_detailed


def test_alignment_validates_dimensions():
    with pytest.raises(ValueError, match="same dimensions"):
        align_images_detailed([
            np.zeros((10, 10, 3), np.float32),
            np.zeros((11, 10, 3), np.float32),
        ])


def test_alignment_returns_masks_and_diagnostics():
    rng = np.random.default_rng(5)
    base = rng.random((96, 96, 3), dtype=np.float32)
    shifted = np.roll(base, 2, axis=1)
    result = align_images_detailed(
        [base, shifted], reference_index=0, motion_model="translation",
        num_pyramid_levels=2,
    )
    assert len(result.images) == 2
    assert len(result.valid_masks) == 2
    assert all(item.accepted for item in result.diagnostics)
    assert result.valid_masks[1].mean() < 1.0


def test_alignment_observes_cancellation_between_frames():
    class Cancelled(Exception):
        pass

    calls = 0

    def cancel_check():
        nonlocal calls
        calls += 1
        if calls > 1:
            raise Cancelled

    image = np.random.default_rng(7).random((64, 64, 3), dtype=np.float32)
    with pytest.raises(Cancelled):
        align_images_detailed([image, image.copy()], cancel_check=cancel_check)
