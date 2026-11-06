import logging
import uuid

from fastapi import APIRouter, File, UploadFile

from app.core.config import get_settings
from app.core.state import state
from app.utils.geojson_utils import bbox_of
from app.utils.raster_utils import probe_image
from app.utils.validation import (
    cleanup_old_files,
    normalise_settlement_categories,
    normalise_ward_names,
    read_geojson_upload,
    save_image_upload,
    validate_polygon_collection,
)

log = logging.getLogger(__name__)
router = APIRouter(prefix="/uploads", tags=["uploads"])


@router.post("/image", status_code=201)
def upload_image(file: UploadFile = File(...)) -> dict:
    """Validate and store an aerial image (GeoTIFF/TIFF/PNG/JPEG). Use the returned upload_id with /api/predict."""
    s = get_settings()
    s.ensure_dirs()
    cleanup_old_files(s.uploads_dir, s.upload_ttl_hours)
    saved = save_image_upload(file, s.uploads_dir, s.max_upload_bytes)
    try:
        info = probe_image(saved.path, s.max_image_pixels)
    except Exception:
        saved.path.unlink(missing_ok=True)
        raise
    upload_id = uuid.uuid4().hex
    state.register_upload(upload_id, {"path": str(saved.path), "filename": saved.original_name})
    notes = [] if info["georeferenced"] else [
        "No georeferencing found: results will be in pixel coordinates; areas in m² and density will be unavailable."]
    return {"upload_id": upload_id, "filename": saved.original_name, "size_bytes": saved.size,
            **info, "notes": notes, "expires_after_hours": s.upload_ttl_hours}


def _layer_response(kind: str, fc: dict) -> dict:
    return {"status": "stored", "layer": kind, "feature_count": len(fc["features"]), "bbox": bbox_of(fc), "crs": "EPSG:4326"}


@router.post("/ward-boundaries", status_code=201)
def upload_wards(file: UploadFile = File(...)) -> dict:
    """GeoJSON FeatureCollection of ward Polygons/MultiPolygons in EPSG:4326 (name in properties.ward/name)."""
    s = get_settings()
    s.ensure_dirs()
    fc = normalise_ward_names(validate_polygon_collection(read_geojson_upload(file, s.max_upload_bytes),
                                                          max_features=s.max_geojson_features, label="Ward boundaries"))
    state.set_wards(fc)
    return {**_layer_response("wards", fc), "wards": [f["properties"]["ward"] for f in fc["features"]][:200]}


@router.post("/ground-truth", status_code=201)
def upload_ground_truth(file: UploadFile = File(...)) -> dict:
    """Ground-truth building footprints (e.g. Open Cities labels reprojected to EPSG:4326). Shown as source='ground_truth'."""
    s = get_settings()
    s.ensure_dirs()
    fc = validate_polygon_collection(read_geojson_upload(file, s.max_upload_bytes),
                                     max_features=s.max_geojson_features, label="Ground truth")
    state.set_ground_truth(fc)
    return _layer_response("ground_truth", fc)


@router.post("/settlement-areas", status_code=201)
def upload_settlements(file: UploadFile = File(...)) -> dict:
    """Polygons with properties.category = 'planned' | 'informal' (needed for the settlement comparison)."""
    s = get_settings()
    s.ensure_dirs()
    fc = normalise_settlement_categories(validate_polygon_collection(
        read_geojson_upload(file, s.max_upload_bytes), max_features=s.max_geojson_features, label="Settlement areas"))
    state.set_settlements(fc)
    counts: dict[str, int] = {}
    for f in fc["features"]:
        counts[f["properties"]["category"]] = counts.get(f["properties"]["category"], 0) + 1
    return {**_layer_response("settlements", fc), "categories": counts}
