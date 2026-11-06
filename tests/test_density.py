import math

import pytest

pytest.importorskip("geopandas")
pytest.importorskip("shapely")
from app.services.density_service import calculate_density, density_rows  # noqa: E402
from tests.conftest import fc, square_geojson  # noqa: E402

LAT = -6.82


def _expected_km2(size_deg):
    return (size_deg * 111.32) * (size_deg * 111.32 * math.cos(math.radians(LAT)))


def _scene():
    wards = fc(square_geojson(39.27, LAT, 0.01, ward="A"), square_geojson(39.30, LAT, 0.01, ward="B"))
    b = [square_geojson(39.271 + i * 0.002, LAT + 0.002, 0.0003) for i in range(3)] + [square_geojson(39.305, LAT + 0.002, 0.0003)]
    outside = square_geojson(40.0, LAT, 0.0003)
    return fc(*b, outside), wards


def test_ward_density_uses_metric_area_not_degrees():
    b, w = _scene()
    res = calculate_density(b, w)
    rows = {r["ward"]: r for r in density_rows(res)}
    assert rows["A"]["count"] == 3 and rows["B"]["count"] == 1
    km2 = _expected_km2(0.01)
    assert rows["A"]["density"] == pytest.approx(3 / km2, rel=0.03)
    assert 0 < rows["A"]["coverage"] < 5
    assert res["summary"]["buildings_outside_polygons"] == 1 and res["kind"] == "ward"


def test_grid_fallback_is_labelled():
    b, _ = _scene()
    res = calculate_density(fc(*b["features"][:4]), None, grid_cell_size_m=500)
    assert res["kind"] == "grid" and any("not wards" in n for n in res["notes"])
    assert all(f["properties"]["kind"] == "grid" and f["properties"]["ward"].startswith("Grid cell") for f in res["features"])


def test_density_calculate_endpoint(client):
    b, w = _scene()
    r = client.post("/api/density/calculate", json={"buildings": b, "wards": w})
    assert r.status_code == 200 and len(r.json()["rows"]) == 2
    bad = client.post("/api/density/calculate", json={"buildings": {"type": "FeatureCollection", "features": []}})
    assert bad.status_code == 422
