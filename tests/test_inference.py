import io

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("rasterio")
pytest.importorskip("shapely")
pytest.importorskip("pyproj")
from tests.conftest import png_bytes  # noqa: E402


def _noise_png(w=256, h=256):
    rng = np.random.default_rng(1)
    b = io.BytesIO()
    Image.fromarray(rng.integers(0, 255, (h, w, 3), dtype=np.uint8)).save(b, format="PNG")
    return b.getvalue()


def test_png_predict_in_demo_mode_is_flagged_and_unreferenced(client):
    r = client.post("/api/predict", files={"file": ("tile.png", png_bytes(256, 256), "image/png")},
                    data={"model": "cnn", "type": "segmentation", "format": "geojson"})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["demo"] is True and d["results_type"] == "simulated_demo" and "DEMO" in d["status_message"]
    assert d["georeferenced"] is False and d["crs"] is None
    assert d["building_count"] == len(d["geojson"]["features"]) > 0
    assert all(f["properties"]["area_m2"] is None for f in d["geojson"]["features"])  # no fabricated metres
    assert all(f["properties"]["source"] == "demo_simulated" for f in d["geojson"]["features"])
    assert d["mask"]["data_uri"].startswith("data:image/png;base64,")
    assert d["statistics"]["building_density_per_km2"] is None
    assert any("georeferenc" in w for w in d["warnings"])
    assert d["processing_time_s"] >= 0
    # summary keys come before bulky payloads
    keys = list(d)
    assert keys.index("statistics") < keys.index("mask") < keys.index("geojson")


def test_temp_upload_is_removed_after_predict(client):
    import os
    from app.core.config import get_settings

    client.post("/api/predict", files={"file": ("tile.png", png_bytes(128, 128), "image/png")})
    assert [f for f in os.listdir(get_settings().tmp_dir) if f != ".gitkeep"] == []


def test_edge_baseline_runs_without_model(client):
    r = client.post("/api/predict", files={"file": ("tile.png", _noise_png(), "image/png")}, data={"model": "edge"})
    assert r.status_code == 200 and r.json()["results_type"] == "edge_detection_baseline" and r.json()["demo"] is False


def test_invalid_model_and_missing_file(client):
    assert client.post("/api/predict", files={"file": ("t.png", png_bytes(), "image/png")}, data={"model": "gpt"}).status_code == 422
    assert client.post("/api/predict").status_code == 400


def test_geotiff_keeps_crs_and_returns_wgs84_with_metric_areas(client, tmp_path):
    import rasterio
    from rasterio.transform import from_origin

    path = tmp_path / "scene.tif"
    arr = np.full((3, 256, 256), 100, dtype=np.uint8)
    with rasterio.open(path, "w", driver="GTiff", width=256, height=256, count=3, dtype="uint8",
                       crs="EPSG:32737", transform=from_origin(500000, 9245000, 0.5, 0.5)) as dst:
        dst.write(arr)
    r = client.post("/api/predict", files={"file": ("scene.tif", path.read_bytes(), "image/tiff")})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["georeferenced"] is True and d["crs"] == "EPSG:32737" and d["output_crs"] == "EPSG:4326"
    f = d["geojson"]["features"][0]
    lon, lat = f["geometry"]["coordinates"][0][0]
    assert 38 < lon < 40 and -8 < lat < -6
    assert f["properties"]["area_m2"] > 0
    assert d["statistics"]["image_area_km2"] == pytest.approx(128 * 128 / 1e6, rel=0.01)
    # georeferenced result is now what /api/footprints serves (still flagged as demo)
    fp = client.get("/api/footprints").json()
    assert fp["demo"] is True and len(fp["features"]) == d["building_count"]
    assert client.get(f"/api/buildings/{f['properties']['building_id']}").status_code == 200
    assert client.get("/api/analysis/" + d["analysis_id"]).status_code == 200
    assert client.get("/api/buildings/NOPE999").status_code == 404
