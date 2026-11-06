def test_health_structure_and_no_path_leak(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    d = r.json()
    for k in ("status", "model_loaded", "device", "demo_mode", "mode", "checkpoint_found"):
        assert k in d
    assert d["status"] == "ok" and d["model_loaded"] is False and d["demo_mode"] is True and d["mode"] == "demo"
    assert "missing.pth" not in r.text and "buildresilience_test_" not in r.text  # no server paths


def test_demo_header_present(client):
    assert client.get("/api/health").headers.get("x-demo-mode") == "true"


def test_dashboard_summary_is_labelled_demo(client):
    d = client.get("/api/dashboard/summary").json()
    assert d["demo"] is True and "DEMO" in d["label"]
    assert d["total_buildings"] > 0 and d["model"].startswith("DEMO MODE")


def test_frontend_summary_shape_and_demo_suffix(client):
    d = client.get("/api/summary").json()
    for k in ("total_buildings", "total_footprint_area", "avg_density", "wards_analyzed", "model"):
        assert k in d
    assert "(demo)" in d["total_buildings"] and "(demo)" in d["total_footprint_area"]


def test_unknown_route_error_shape(client):
    r = client.get("/api/nope")
    assert r.status_code == 404 and r.json()["error"]["code"] == "http_error"
