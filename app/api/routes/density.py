from fastapi import APIRouter, Query, Response

from app.core.config import get_settings
from app.schemas.building import DensityCalculateRequest
from app.services import dashboard_service
from app.services.density_service import calculate_density, density_rows
from app.utils.validation import normalise_ward_names, validate_polygon_collection

router = APIRouter(tags=["density"])


@router.get("/density")
def density(response: Response, format: str = Query("rows", pattern="^(rows|geojson)$")):
    """Default: flat ward rows [{ward,count,density,area,coverage}] as read by the React pages.
    `?format=geojson` returns the density polygon layer with summary statistics."""
    res = dashboard_service.current_density()
    response.headers["X-Density-Kind"] = res["kind"] if res else "none"
    if res and res.get("demo"):
        response.headers["X-Demo-Mode"] = "true"
    return dashboard_service.density_response(format)


@router.get("/density/layer")
def density_layer(response: Response) -> dict:
    res = dashboard_service.current_density()
    response.headers["X-Density-Kind"] = res["kind"] if res else "none"
    return dashboard_service.density_response("geojson")


@router.post("/density/calculate")
def density_calculate(req: DensityCalculateRequest) -> dict:
    """Compute density for supplied building footprints and ward polygons (GeoJSON, EPSG:4326).
    Without `wards`, a labelled regular grid is produced instead."""
    s = get_settings()
    b = validate_polygon_collection(req.buildings, max_features=s.max_geojson_features, label="buildings")
    w = None
    if req.wards is not None:
        w = normalise_ward_names(validate_polygon_collection(req.wards, max_features=s.max_geojson_features, label="wards"))
    result = calculate_density(b, w, grid_cell_size_m=req.grid_cell_size_m or s.grid_cell_size_m, max_grid_cells=s.max_grid_cells)
    result["rows"] = density_rows(result)
    return result
