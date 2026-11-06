import numpy as np
import pytest

pytest.importorskip("rasterio")
pytest.importorskip("shapely")
pytest.importorskip("pyproj")
from affine import Affine  # noqa: E402

from app.services.polygonization_service import items_to_features, polygonize_mask  # noqa: E402

TF = Affine(1.0, 0, 500000.0, 0, -1.0, 9245000.0)  # 1 m pixels, UTM 37S
CRS = "EPSG:32737"


def _mask():
    m = np.zeros((60, 60), bool)
    m[5:15, 5:15] = True      # 10x10 = 100 m²
    m[30:42, 30:42] = True    # 12x12 = 144 m²
    m[50:52, 50:52] = True    # 2x2 = 4 m²  (below threshold)
    return m


def test_georeferenced_polygons_and_min_area_filter():
    items = polygonize_mask(_mask(), transform=TF, crs=CRS, min_area_m2=20, min_area_px=1, simplify_tolerance_px=0)
    assert sorted(round(i.area_m2) for i in items) == [100, 144]
    assert all(i.geometry.is_valid for i in items)


def test_output_geojson_is_wgs84_with_properties():
    items = polygonize_mask(_mask(), transform=TF, crs=CRS, min_area_m2=20, simplify_tolerance_px=0)
    feats = items_to_features(items, crs=CRS, source="model_prediction")
    assert [f["properties"]["building_id"] for f in feats] == ["B001", "B002"]
    lon, lat = feats[0]["geometry"]["coordinates"][0][0]
    assert 38 < lon < 40 and -8 < lat < -6  # Dar es Salaam area, i.e. properly reprojected
    assert feats[0]["properties"]["source"] == "model_prediction" and feats[0]["properties"]["confidence"] is None


def test_confidence_is_mean_probability():
    prob = np.where(_mask(), 0.9, 0.1).astype(np.float32)
    items = polygonize_mask(_mask(), transform=TF, crs=CRS, probability=prob, min_area_m2=20, simplify_tolerance_px=0)
    assert all(abs(i.confidence - 0.9) < 1e-6 for i in items)


def test_unreferenced_image_uses_pixels_and_no_area_m2():
    items = polygonize_mask(_mask(), min_area_px=20, simplify_tolerance_px=0)
    assert len(items) == 2 and all(i.area_m2 is None for i in items)
    feats = items_to_features(items, crs=None, source="model_prediction")
    assert feats[0]["properties"]["area_m2"] is None and feats[0]["properties"]["area_px"] in (100, 144)


def test_touching_buildings_merge_documented_limitation():
    m = np.zeros((30, 30), bool)
    m[5:15, 5:12] = True
    m[5:15, 12:20] = True  # shares a wall: indistinguishable in a binary mask
    assert len(polygonize_mask(m, min_area_px=5, simplify_tolerance_px=0)) == 1


def test_empty_mask_gives_no_polygons():
    assert polygonize_mask(np.zeros((20, 20), bool), transform=TF, crs=CRS) == []
