"""Binary mask -> building footprint polygons (+ GeoJSON features).

Method: 4-connected component labelling (OpenCV) -> rasterio.features.shapes on the label image
(preserves the affine transform) -> Shapely clean-up (make_valid, simplify, min-area filter).

LIMITATION: one connected component is NOT guaranteed to be one real-world building. Buildings
that touch, share walls or overlap in the mask are merged into a single polygon, and one
building split by a tree or shadow may become several polygons. Instance-separation (e.g. a
distance-transform / watershed step or an instance model) is not applied.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from app.utils.geojson_utils import geom_to_geojson

LIMITATION_TEXT = (
    "Each polygon is a connected region of the predicted mask. Touching or overlapping buildings are "
    "merged into one polygon, and a single building split by occlusion may yield several polygons, so "
    "counts are approximate."
)


@dataclass
class BuildingItem:
    geometry: object  # shapely geometry in raster CRS (or pixel space)
    area_px: int
    area_m2: float | None
    confidence: float | None


def polygonize_mask(
    mask: np.ndarray,
    *,
    transform=None,
    crs: str | None = None,
    probability: np.ndarray | None = None,
    valid_mask: np.ndarray | None = None,
    min_area_m2: float = 10.0,
    min_area_px: int = 30,
    simplify_tolerance_px: float = 0.7,
) -> list[BuildingItem]:
    """Polygonize `mask`. When `crs` is None the output is in pixel space (x=column, y=row)."""
    from affine import Affine
    from rasterio.features import shapes
    from shapely.geometry import shape

    from app.services.geospatial_service import AreaCalculator, make_valid_poly, pixel_area_m2

    h, w = mask.shape
    m = (mask > 0).astype(np.uint8)
    if valid_mask is not None:
        m &= valid_mask.astype(np.uint8)
    n, labels = cv2.connectedComponents(m, connectivity=4, ltype=cv2.CV_32S)
    if n <= 1:
        return []

    counts = np.bincount(labels.ravel(), minlength=n)
    conf_sums = None
    if probability is not None:
        conf_sums = np.bincount(labels.ravel(), weights=probability.ravel().astype(np.float64), minlength=n)

    georef = crs is not None and transform is not None
    tf = transform if georef else Affine.identity()
    calc = AreaCalculator(crs, tf * (w / 2.0, h / 2.0)) if georef else None

    # Cheap pre-filter in pixels (conservatively half the threshold) before building polygons.
    if georef:
        px_m2 = pixel_area_m2(tf, crs, w, h)
        pre = max(1, int(math.floor(0.5 * min_area_m2 / px_m2))) if px_m2 > 0 else 1
    else:
        pre = max(1, min_area_px // 2)
    keep = counts >= pre
    keep[0] = False
    if not keep.any():
        return []
    filtered = np.where(keep[labels], labels, 0).astype(np.int32)
    del labels

    tol = simplify_tolerance_px * abs(tf.a) if simplify_tolerance_px > 0 else 0.0
    items: list[BuildingItem] = []
    for geom_dict, val in shapes(filtered, mask=filtered > 0, transform=tf, connectivity=4):
        lab = int(val)
        geom = make_valid_poly(shape(geom_dict))
        if geom.is_empty:
            continue
        if tol > 0:
            geom = make_valid_poly(geom.simplify(tol, preserve_topology=True))
            if geom.is_empty:
                continue
        area_px = int(counts[lab])
        if georef:
            area_m2 = calc.area(geom)
            if area_m2 < min_area_m2:
                continue
        else:
            area_m2 = None
            if area_px < min_area_px:
                continue
        conf = float(conf_sums[lab] / counts[lab]) if conf_sums is not None and counts[lab] else None
        items.append(BuildingItem(geom, area_px, area_m2, conf))
    # Stable, reading-order IDs: top-to-bottom, left-to-right in raster space
    items.sort(key=lambda it: (round(-it.geometry.centroid.y, 6) if georef else round(it.geometry.centroid.y, 3), it.geometry.centroid.x))
    return items


def items_to_features(
    items: list[BuildingItem], *, crs: str | None, source: str, id_prefix: str = "B", extra_props: dict | None = None
) -> list[dict]:
    """GeoJSON features. Georeferenced input is reprojected to EPSG:4326 (what Leaflet expects)."""
    from app.services.geospatial_service import reproject_geom

    feats = []
    for i, it in enumerate(items, start=1):
        geom = reproject_geom(it.geometry, crs, "EPSG:4326") if crs else it.geometry
        bid = f"{id_prefix}{i:03d}"
        area = round(it.area_m2, 2) if it.area_m2 is not None else None
        props = {
            "id": bid,
            "building_id": bid,
            "area_m2": area,
            "area": area,  # alias: the React popup reads `area`
            "area_px": it.area_px,
            "confidence": round(it.confidence, 4) if it.confidence is not None else None,
            "source": source,
        }
        if extra_props:
            props.update(extra_props)
        feats.append({"type": "Feature", "properties": props, "geometry": geom_to_geojson(geom, 7 if crs else 2)})
    return feats


def summarize(items: list[BuildingItem]) -> dict:
    areas = [it.area_m2 for it in items if it.area_m2 is not None]
    out = {"building_count": len(items), "total_area_m2": None, "mean_area_m2": None,
           "median_area_m2": None, "min_area_m2": None, "max_area_m2": None,
           "total_area_px": int(sum(it.area_px for it in items))}
    if areas:
        out.update(total_area_m2=round(float(sum(areas)), 2), mean_area_m2=round(float(np.mean(areas)), 2),
                   median_area_m2=round(float(np.median(areas)), 2), min_area_m2=round(float(min(areas)), 2),
                   max_area_m2=round(float(max(areas)), 2))
    return out
