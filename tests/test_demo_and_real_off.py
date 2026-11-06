import pytest


def test_demo_footprints_never_claim_real_source(client):
    d = client.get("/api/footprints").json()
    assert d["demo"] is True and len(d["features"]) > 50
    sources = {f["properties"]["source"] for f in d["features"]}
    assert sources == {"demo_simulated"}  # never 'model_prediction' or 'ground_truth'


def test_demo_wards_named_demo(client):
    d = client.get("/api/wards").json()
    assert len(d["features"]) == 6 and all(f["properties"]["ward"].startswith("DEMO") for f in d["features"])


def test_demo_density_rows(client):
    pytest.importorskip("geopandas")
    rows = client.get("/api/density").json()
    assert len(rows) == 6
    assert set(rows[0]) == {"ward", "count", "density", "area", "coverage"}
    layer = client.get("/api/density?format=geojson").json()
    assert layer["type"] == "FeatureCollection" and layer["kind"] == "ward"


def test_demo_metrics_are_marked(client):
    d = client.get("/api/metrics").json()
    assert d["status"] == "demo" and d["demo"] is True
    assert "(demo)" in d["summary"]["IoU"]
    assert d["comparison"] == [] and d["training"] == []  # no invented charts


def test_settlement_comparison_unavailable_without_labels(client):
    d = client.get("/api/analytics/settlement-comparison").json()
    assert d["status"] == "unavailable" and "settlement-type annotations" in d["message"]


def test_demo_off_and_no_model_returns_nothing_fake(make_client):
    c = make_client(DEMO_MODE="false")
    assert c.get("/api/health").json()["mode"] == "unavailable"
    assert c.get("/api/footprints").json()["features"] == []
    s = c.get("/api/summary").json()
    assert s["total_buildings"] is None and s["avg_density"] is None
    m = c.get("/api/metrics").json()
    assert m["status"] == "unavailable" and m["summary"]["IoU"] is None
    from tests.conftest import png_bytes

    r = c.post("/api/predict", files={"file": ("a.png", png_bytes(), "image/png")})
    assert r.status_code == 503 and r.json()["error"]["code"] == "model_unavailable"
