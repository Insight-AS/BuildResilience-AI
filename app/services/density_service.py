"""Building density per ward / uploaded polygon, or per labelled grid cell.

Inputs are GeoJSON in EPSG:4326. All areas are computed in a local UTM CRS (never in degrees).

Definitions
  density  = building_count / polygon_area_km2
  coverage = footprint area inside the polygon / polygon area * 100   (footprints are clipped)
  count    = buildings whose representative point lies inside the polygon (each building is counted
             once; buildings outside every polygon are reported separately)
"""
from __future__ import annotations

import math

from app.core.errors import Unprocessable
from app.services.geospatial_service import fc_to_gdf, gdf_geometry_to_features

PROP_COLS = ["ward", "count", "density", "area", "coverage", "ward_area_km2", "avg_building_area_m2", "kind"]


def _grid_cells(bounds, cell_m: float, max_cells: int):
    from shapely.geometry import box

    minx, miny, maxx, maxy = bounds
    nx, ny = max(1, math.ceil((maxx - minx) / cell_m)), max(1, math.ceil((maxy - miny) / cell_m))
    if nx * ny > max_cells:
        raise Unprocessable(
            f"Grid would have {nx * ny} cells (limit {max_cells}). Increase the cell size.", code="grid_too_large")
    cells = []
    for j in range(ny):
        for i in range(nx):
            cells.append(box(minx + i * cell_m, miny + j * cell_m, minx + (i + 1) * cell_m, miny + (j + 1) * cell_m))
    return cells


def calculate_density(buildings_fc: dict, wards_fc: dict | None = None, *, grid_cell_size_m: float | None = None,
                      max_grid_cells: int = 20_000) -> dict:
    import geopandas as gpd

    b = fc_to_gdf(buildings_fc)
    if b.empty:
        raise Unprocessable("No valid building footprints supplied.", code="no_buildings")

    if wards_fc is not None:
        regions = fc_to_gdf(wards_fc)
        if regions.empty:
            raise Unprocessable("No valid ward polygons supplied.", code="no_wards")
        kind = "ward"
        if "ward" in regions.columns:
            names = [str(v) if v is not None and str(v) != "nan" else f"Ward {i + 1}" for i, v in enumerate(regions["ward"])]
        else:
            names = [f"Ward {i + 1}" for i in range(len(regions))]
        metric = regions.estimate_utm_crs()
        rm = regions.to_crs(metric)
        rm["ward"] = names
    else:
        metric = b.estimate_utm_crs()
        bm0 = b.to_crs(metric)
        cells = _grid_cells(bm0.total_bounds, float(grid_cell_size_m or 250.0), max_grid_cells)
        rm = gpd.GeoDataFrame({"geometry": cells}, crs=metric)
        rm["ward"] = [f"Grid cell G{i + 1:04d}" for i in range(len(rm))]
        kind = "grid"

    bm = b.to_crs(metric)
    bm["_area"] = bm.geometry.area
    rm["_rid"] = range(len(rm))
    rm["_area_m2"] = rm.geometry.area
    rm = rm[rm["_area_m2"] > 0].reset_index(drop=True)

    pts = bm.copy()
    pts["geometry"] = bm.geometry.representative_point()
    joined = gpd.sjoin(pts[["_area", "geometry"]], rm[["_rid", "geometry"]], how="inner", predicate="within")
    joined = joined[~joined.index.duplicated(keep="first")]
    grp = joined.groupby("_rid")["_area"].agg(["count", "mean"])

    sindex = bm.sindex
    rows = []
    for _, r in rm.iterrows():
        rid = int(r["_rid"])
        cand = bm.iloc[sindex.query(r.geometry, predicate="intersects")]
        covered = float(cand.geometry.intersection(r.geometry).area.sum()) if len(cand) else 0.0
        count = int(grp["count"].get(rid, 0))
        mean_area = float(grp["mean"].get(rid, 0.0)) if count else None
        km2 = r["_area_m2"] / 1e6
        rows.append({
            "geometry": r.geometry, "ward": r["ward"], "count": count,
            "density": round(count / km2, 2), "area": round(covered, 2),
            "coverage": round(min(100.0, covered / r["_area_m2"] * 100.0), 2),
            "ward_area_km2": round(km2, 4),
            "avg_building_area_m2": round(mean_area, 2) if mean_area is not None else None,
            "kind": kind,
        })
    out = gpd.GeoDataFrame(rows, geometry="geometry", crs=metric).to_crs("EPSG:4326")
    if kind == "grid":  # drop completely empty grid cells to keep the layer light
        out = out[out["count"] > 0].reset_index(drop=True)

    total_area_km2 = float(sum(r["ward_area_km2"] for r in rows))
    total_count = int(sum(r["count"] for r in rows))
    total_footprint = float(sum(r["area"] for r in rows))
    summary = {
        "kind": kind,
        "polygons_analyzed": len(rows),
        "total_buildings_input": int(len(bm)),
        "total_buildings_assigned": total_count,
        "buildings_outside_polygons": int(len(bm) - total_count),
        "total_footprint_area_m2": round(total_footprint, 2),
        "total_polygon_area_km2": round(total_area_km2, 4),
        "overall_density_per_km2": round(total_count / total_area_km2, 2) if total_area_km2 else None,
        "overall_coverage_pct": round(total_footprint / (total_area_km2 * 1e6) * 100, 2) if total_area_km2 else None,
        "mean_polygon_density_per_km2": round(sum(r["density"] for r in rows) / len(rows), 2) if rows else None,
        "average_building_area_m2": round(float(bm["_area"].mean()), 2),
    }
    notes = ["Areas were computed in a projected CRS (" + str(metric.to_string()) + "), not in degrees."]
    if kind == "grid":
        notes.append("No administrative boundaries were provided: this is a regular analysis grid, not wards.")
    features = gdf_geometry_to_features(out, PROP_COLS)
    return {"type": "FeatureCollection", "kind": kind, "summary": summary, "notes": notes, "features": features}


def density_rows(result: dict) -> list[dict]:
    """Flat rows in the shape the React Density/Overview pages read."""
    return [{k: f["properties"][k] for k in ("ward", "count", "density", "area", "coverage")} for f in result["features"]]
