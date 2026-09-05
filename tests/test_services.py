from src.config.output_settings import OutputSettings
from src.config.settings_store import AppSettings, load_settings, save_settings
from src.config.stack_detection_settings import StackDetectionSettings
from src.config.stacking_settings import StackerSettings
from src.services.stack_detection import detect_stacks


def test_auto_detection_uses_common_numeric_suffix():
    stacks = detect_stacks(["Ella_0_10.jpeg", "Ella_0_2.jpeg", "Ella_0_1.jpeg"])
    assert len(stacks) == 1
    assert stacks[0][0] == "Ella_0"
    assert stacks[0][1] == ["Ella_0_1.jpeg", "Ella_0_2.jpeg", "Ella_0_10.jpeg"]


def test_photogrammetry_settings_constrain_geometry():
    settings = StackerSettings(
        photogrammetry_mode=True, alignment_model="homography",
        crop_to_common_area=True,
    ).validated()
    assert settings.alignment_model == "euclidean"
    assert settings.crop_to_common_area is False


def test_jpeg_forces_eight_bit_output():
    assert OutputSettings(output_format="JPEG", bit_depth=16).validated().bit_depth == 8


def test_settings_store_roundtrip(tmp_path):
    path = tmp_path / "settings.json"
    expected = AppSettings(
        stacker=StackerSettings(photogrammetry_mode=True),
        stack_detection=StackDetectionSettings(mode="common_suffix"),
        output=OutputSettings(output_format="PNG", bit_depth=16, export_maps=True),
    )
    save_settings(expected, str(path))
    actual = load_settings(str(path))
    assert actual.to_dict() == expected.to_dict()
    assert not list(tmp_path.glob("*.tmp"))
