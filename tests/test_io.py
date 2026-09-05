import numpy as np
import tifffile
from PIL import Image

from src.core import utils


def test_16bit_tiff_roundtrip(tmp_path):
    source = np.zeros((7, 9, 3), dtype=np.uint16)
    source[..., 0] = 12345
    source[..., 1] = 32768
    source[..., 2] = 65535
    input_path = tmp_path / "input.tif"
    output_path = tmp_path / "output.tif"
    tifffile.imwrite(input_path, source, photometric="rgb")

    loaded = utils.load_image_with_metadata(str(input_path))
    utils.save_image(loaded.image, str(output_path), format="TIFF", bit_depth=16,
                     metadata=loaded.metadata)
    result = tifffile.imread(output_path)
    assert loaded.metadata.source_bit_depth == 16
    assert result.dtype == np.uint16
    np.testing.assert_array_equal(result, source)
    with tifffile.TiffFile(output_path) as tif:
        assert 34675 in tif.pages[0].tags


def test_16bit_png_contains_icc_profile(tmp_path):
    output = tmp_path / "result.png"
    image = np.random.default_rng(2).random((8, 9, 3), dtype=np.float32)
    metadata = utils.ImageMetadata(icc_profile=utils.get_color_profile_bytes("sRGB"))
    utils.save_image(image, str(output), format="PNG", bit_depth=16, metadata=metadata)
    with Image.open(output) as saved:
        assert saved.info.get("icc_profile") == metadata.icc_profile


def test_jpeg_exif_is_preserved_but_orientation_is_removed(tmp_path):
    path = tmp_path / "source.jpg"
    out = tmp_path / "result.jpg"
    exif = Image.Exif()
    exif[271] = "Test Camera"
    exif[274] = 6
    Image.new("RGB", (8, 5), "red").save(path, exif=exif)
    loaded = utils.load_image_with_metadata(str(path))
    utils.save_image(loaded.image, str(out), format="JPEG", metadata=loaded.metadata)
    with Image.open(out) as image:
        output_exif = image.getexif()
        assert output_exif[271] == "Test Camera"
        assert 274 not in output_exif


def test_srgb_linear_roundtrip():
    values = np.linspace(0, 1, 100, dtype=np.float32)
    restored = utils.linear_to_srgb(utils.srgb_to_linear(values))
    np.testing.assert_allclose(restored, values, atol=2e-6)


def test_exposure_matching_brings_luminance_ranges_together():
    ramp = np.linspace(0.1, 0.8, 64, dtype=np.float32)
    reference = np.broadcast_to(ramp[None, :, None], (32, 64, 3)).copy()
    darker = utils.linear_to_srgb(utils.srgb_to_linear(reference) * 0.55)
    masks = [np.ones((32, 64), np.float32), np.ones((32, 64), np.float32)]
    matched = utils.match_stack_exposure([reference, darker], masks, 0)
    assert abs(float(np.mean(matched[0])) - float(np.mean(matched[1]))) < 0.03
