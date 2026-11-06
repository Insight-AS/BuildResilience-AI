import pytest

from app.core.errors import Unprocessable
from app.utils.validation import (cleanup_old_files, get_extension, safe_display_name, sniff_image_type,
                                  validate_identifier, validate_polygon_collection)


def test_safe_name_strips_paths():
    assert safe_display_name("../../etc/passwd") == "passwd"
    assert safe_display_name("C:\\Users\\me\\bad name!.png") == "bad_name_.png"


def test_extension_and_sniffing():
    assert get_extension("A.TIF") == ".tif"
    assert sniff_image_type(b"\x89PNG\r\n\x1a\n....") == "png"
    assert sniff_image_type(b"II*\x00....") == "tiff" and sniff_image_type(b"MZ") is None


def test_identifier_validation():
    assert validate_identifier("B001") == "B001"
    for bad in ("../x", "a b", "", "x" * 100):
        with pytest.raises(Exception):
            validate_identifier(bad)


def test_unclosed_ring_rejected():
    fc = {"type": "FeatureCollection", "features": [{"type": "Feature", "properties": {}, "geometry": {
        "type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1]]]}}]}
    with pytest.raises(Unprocessable):
        validate_polygon_collection(fc, max_features=10)


def test_feature_limit_and_crs_member():
    f = {"type": "Feature", "properties": {}, "geometry": {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]]}}
    with pytest.raises(Unprocessable):
        validate_polygon_collection({"type": "FeatureCollection", "features": [f, f]}, max_features=1)
    with pytest.raises(Unprocessable):
        validate_polygon_collection({"type": "FeatureCollection", "crs": {"properties": {"name": "EPSG:32737"}}, "features": [f]}, max_features=5)


def test_cleanup_old_files(tmp_path):
    import os, time

    old, new = tmp_path / "old.bin", tmp_path / "new.bin"
    old.write_bytes(b"x"); new.write_bytes(b"x")
    os.utime(old, (time.time() - 10 * 3600, time.time() - 10 * 3600))
    assert cleanup_old_files(tmp_path, 1) == 1 and new.exists() and not old.exists()
