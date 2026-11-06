import json
import os

from tests.conftest import fc, png_bytes, square_geojson


def test_reject_bad_extension(client):
    r = client.post("/api/uploads/image", files={"file": ("x.exe", b"MZ....", "application/octet-stream")})
    assert r.status_code == 415 and r.json()["error"]["code"] == "unsupported_file_type"


def test_reject_content_mismatch(client):
    r = client.post("/api/uploads/image", files={"file": ("fake.png", b"this is not an image at all", "image/png")})
    assert r.status_code == 415


def test_reject_empty(client):
    assert client.post("/api/uploads/image", files={"file": ("a.png", b"", "image/png")}).status_code == 422


def test_reject_too_large(client):  # MAX_UPLOAD_SIZE_MB=1 in conftest
    big = png_bytes() + b"\0" * (2 * 1024 * 1024)
    r = client.post("/api/uploads/image", files={"file": ("big.png", big, "image/png")})
    assert r.status_code == 413 and r.json()["error"]["code"] == "file_too_large"


def test_valid_png_upload_reports_no_georeferencing(client):
    r = client.post("/api/uploads/image", files={"file": ("tile.png", png_bytes(), "image/png")})
    assert r.status_code == 201
    d = r.json()
    assert d["georeferenced"] is False and d["crs"] is None and d["width"] == 128
    assert "path" not in d and "upload_id" in d


def test_client_filename_never_used_on_disk(client):
    from app.core.config import get_settings

    r = client.post("/api/uploads/image", files={"file": ("../../evil.png", png_bytes(), "image/png")})
    assert r.status_code == 201
    assert all("evil" not in f for f in os.listdir(get_settings().uploads_dir))


def test_ward_upload_ok_and_served(client):
    good = fc(square_geojson(39.27, -6.82, ward="Ward Z"))
    r = client.post("/api/uploads/ward-boundaries", files={"file": ("w.geojson", json.dumps(good), "application/json")})
    assert r.status_code == 201 and r.json()["feature_count"] == 1
    wards = client.get("/api/wards").json()
    assert wards["features"][0]["properties"]["ward"] == "Ward Z"


def test_ward_upload_rejects_projected_coordinates(client):
    bad = '{"type":"FeatureCollection","features":[{"type":"Feature","properties":{},"geometry":{"type":"Polygon","coordinates":[[[500000,9000000],[500100,9000000],[500100,9000100],[500000,9000100],[500000,9000000]]]}}]}'
    r = client.post("/api/uploads/ward-boundaries", files={"file": ("w.geojson", bad, "application/json")})
    assert r.status_code == 422 and r.json()["error"]["code"] == "invalid_crs"


def test_ward_upload_rejects_points_and_garbage(client):
    pts = '{"type":"FeatureCollection","features":[{"type":"Feature","properties":{},"geometry":{"type":"Point","coordinates":[1,2]}}]}'
    assert client.post("/api/uploads/ward-boundaries", files={"file": ("w.geojson", pts, "application/json")}).status_code == 422
    assert client.post("/api/uploads/ward-boundaries", files={"file": ("w.geojson", "{not json", "application/json")}).status_code == 422
    assert client.post("/api/uploads/ward-boundaries", files={"file": ("w.txt", "{}", "text/plain")}).status_code == 415


def test_settlement_categories_validated(client):
    ok = '{"type":"FeatureCollection","features":[{"type":"Feature","properties":{"category":"Informal"},"geometry":{"type":"Polygon","coordinates":[[[39,-6],[39.1,-6],[39.1,-6.1],[39,-6.1],[39,-6]]]}}]}'
    r = client.post("/api/uploads/settlement-areas", files={"file": ("s.geojson", ok, "application/json")})
    assert r.status_code == 201 and r.json()["categories"] == {"informal": 1}
    bad = ok.replace("Informal", "slum-ish")
    r = client.post("/api/uploads/settlement-areas", files={"file": ("s.geojson", bad, "application/json")})
    assert r.status_code == 422
