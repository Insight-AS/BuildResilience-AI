from fastapi import APIRouter, Query

from app.core.errors import NotFound
from app.services import dashboard_service
from app.utils.validation import validate_identifier

router = APIRouter(tags=["buildings"])


@router.get("/buildings")
def list_buildings(
    source: str | None = Query(None, pattern="^(prediction|ground_truth)$"),
    limit: int = Query(50_000, ge=1, le=200_000),
    offset: int = Query(0, ge=0),
) -> dict:
    fc = dashboard_service.footprints_fc(source)
    total = len(fc["features"])
    fc["features"] = fc["features"][offset : offset + limit]
    fc["total"], fc["offset"], fc["limit"] = total, offset, limit
    return fc


@router.get("/buildings/{building_id}")
def get_building(building_id: str) -> dict:
    validate_identifier(building_id, "building_id")
    for f in dashboard_service.footprints_fc()["features"]:
        if f["properties"].get("building_id") == building_id or f["properties"].get("id") == building_id:
            return f
    raise NotFound(f"Building '{building_id}' not found.", code="building_not_found")


@router.get("/footprints")
def footprints() -> dict:
    """All footprints (ground truth tagged source='ground_truth'); read by MapExplorer.jsx."""
    return dashboard_service.footprints_fc()


@router.get("/wards")
def wards() -> dict:
    """Ward boundaries FeatureCollection; read by MapExplorer.jsx."""
    return dashboard_service.wards_fc()
