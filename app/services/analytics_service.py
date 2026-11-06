"""Evaluation metrics. Never fabricates values: anything not computable is None / 'unavailable'.

IoU       = TP / (TP + FP + FN)  = |P ∩ G| / |P ∪ G|
Dice      = 2·|P ∩ G| / (|P| + |G|)
Precision = TP / (TP + FP)
Recall    = TP / (TP + FN)
Building-count error = | predicted count − ground-truth count |
(Pixel-level for masks; area-level for vector footprints. 'Building count' = 4-connected
components of each mask / polygon count, with the same merging caveat as polygonization.)
"""
from __future__ import annotations

import cv2
import numpy as np

from app.core.errors import Unprocessable


def _ratio(num: float, den: float) -> float | None:
    return round(float(num) / float(den), 4) if den > 0 else None


def count_components(mask: np.ndarray, min_pixels: int = 1) -> int:
    n, labels = cv2.connectedComponents((mask > 0).astype(np.uint8), connectivity=4, ltype=cv2.CV_32S)
    if n <= 1:
        return 0
    counts = np.bincount(labels.ravel(), minlength=n)[1:]
    return int((counts >= min_pixels).sum())


def mask_metrics(pred: np.ndarray, gt: np.ndarray, *, min_component_px: int = 1) -> dict:
    """Pixel-level metrics for two same-shaped masks (non-zero = building)."""
    if pred.shape != gt.shape:
        raise Unprocessable(f"Mask sizes differ: prediction {pred.shape[::-1]} vs ground truth {gt.shape[::-1]} (width x height).",
                            code="shape_mismatch")
    p, g = pred > 0, gt > 0
    tp = int(np.logical_and(p, g).sum())
    fp = int(np.logical_and(p, ~g).sum())
    fn = int(np.logical_and(~p, g).sum())
    tn = int(p.size - tp - fp - fn)
    pc, gc = count_components(p, min_component_px), count_components(g, min_component_px)
    return {
        "iou": _ratio(tp, tp + fp + fn),
        "dice": _ratio(2 * tp, 2 * tp + fp + fn),
        "precision": _ratio(tp, tp + fp),
        "recall": _ratio(tp, tp + fn),
        "building_count_error": abs(pc - gc),
        "predicted_building_count": pc,
        "ground_truth_building_count": gc,
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "level": "pixel",
    }


def evaluate_vector(pred_fc: dict, gt_fc: dict, region_wgs84=None) -> dict:
    """Area-based metrics between two footprint collections (optionally restricted to a region)."""
    import geopandas as gpd
    import shapely

    from app.services.geospatial_service import fc_to_gdf

    p, g = fc_to_gdf(pred_fc), fc_to_gdf(gt_fc)
    if p.empty or g.empty:
        return {"status": "unavailable", "message": "Both predictions and ground-truth footprints are required."}
    metric = g.estimate_utm_crs()
    pg, gg = p.to_crs(metric).geometry, g.to_crs(metric).geometry
    if region_wgs84 is not None:
        r = gpd.GeoSeries([region_wgs84], crs="EPSG:4326").to_crs(metric).iloc[0]
        pc, gc = int(pg.representative_point().within(r).sum()), int(gg.representative_point().within(r).sum())
        pg, gg = pg.intersection(r), gg.intersection(r)
    else:
        pc, gc = len(pg), len(gg)
    pu = shapely.union_all(np.asarray(list(pg), dtype=object))
    gu = shapely.union_all(np.asarray(list(gg), dtype=object))
    inter = pu.intersection(gu).area
    pa, ga = pu.area, gu.area
    tp, fp, fn = inter, pa - inter, ga - inter
    if ga == 0:
        return {"status": "unavailable", "message": "No ground-truth footprints inside this region."}
    return {
        "status": "available",
        "iou": _ratio(tp, tp + fp + fn), "dice": _ratio(2 * tp, 2 * tp + fp + fn),
        "precision": _ratio(tp, tp + fp), "recall": _ratio(tp, tp + fn),
        "building_count_error": abs(pc - gc),
        "predicted_building_count": pc, "ground_truth_building_count": gc,
        "predicted_area_m2": round(pa, 2), "ground_truth_area_m2": round(ga, 2),
        "level": "area",
    }


_SETTLEMENT_NEEDS = ("a real (non-demo) georeferenced prediction", "ground-truth building footprints",
                     "settlement-type polygons labelled 'planned' / 'informal'")


def settlement_comparison(pred_fc: dict | None, gt_fc: dict | None, settlements_fc: dict | None, *, pred_is_demo: bool = False) -> dict:
    """Planned vs informal metrics. Only computed when all three inputs genuinely exist."""
    missing = []
    if not pred_fc or pred_is_demo:
        missing.append(_SETTLEMENT_NEEDS[0])
    if not gt_fc:
        missing.append(_SETTLEMENT_NEEDS[1])
    if not settlements_fc:
        missing.append(_SETTLEMENT_NEEDS[2])
    if missing:
        return {"status": "unavailable",
                "message": "Planned vs informal comparison requires settlement-type annotations, ground truth and a real prediction. Missing: "
                           + "; ".join(missing) + ".",
                "missing": missing, "categories": {}}
    from shapely.ops import unary_union

    from app.services.geospatial_service import fc_to_gdf

    regions = fc_to_gdf(settlements_fc)
    results: dict[str, dict] = {}
    for cat in ("planned", "informal"):
        sub = regions[regions["category"] == cat] if "category" in regions.columns else regions.iloc[0:0]
        if sub.empty:
            results[cat] = {"status": "unavailable", "message": f"No '{cat}' polygons were provided."}
            continue
        results[cat] = evaluate_vector(pred_fc, gt_fc, unary_union(list(sub.geometry)))
        results[cat]["polygons"] = int(len(sub))
    ok = [c for c, r in results.items() if r.get("status") == "available"]
    return {"status": "available" if ok else "unavailable",
            "message": None if len(ok) == 2 else "Only categories with valid labels and ground truth are reported.",
            "categories": results, "level": "area"}


def to_frontend_summary(m: dict | None) -> dict:
    """Map our metric dict to the keys the React Performance page reads (None -> 'Awaiting')."""
    m = m or {}
    return {"IoU": m.get("iou"), "Dice": m.get("dice"), "Precision": m.get("precision"),
            "Recall": m.get("recall"), "Building-count error": m.get("building_count_error")}
