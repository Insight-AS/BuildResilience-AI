"""DEMO MODE ONLY. Everything produced here is synthetic and must be labelled as simulated.

Nothing in this module is a model prediction. It exists so the React frontend can be exercised
before a trained checkpoint is available. Real inference lives in segmentation_service.py.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

from app.services.density_service import calculate_density, density_rows
from app.utils.geojson_utils import feature_collection

DEMO_LABEL = "DEMO MODE: simulated data, not produced by a trained model"
DEMO_SOURCE = "demo_simulated"
CENTER_LAT, CENTER_LON = -6.8235, 39.2695  # same placeholder view as the frontend's MAP_CENTER
_SEED = 20240607


def demo_mask(height: int, width: int, seed: int = _SEED) -> np.ndarray:
    """Synthetic 'building-like' rectangles laid out on a jittered block grid."""
    rng = np.random.default_rng(seed)
    cell = max(24, min(height, width) // 12)
    mask = np.zeros((height, width), np.uint8)
    for cy in range(0, height - cell, cell):
        for cx in range(0, width - cell, cell):
            if rng.random() < 0.45:
                continue
            bw, bh = int(cell * rng.uniform(0.35, 0.7)), int(cell * rng.uniform(0.35, 0.7))
            x0 = cx + int(rng.uniform(0.05, 0.25) * cell)
            y0 = cy + int(rng.uniform(0.05, 0.25) * cell)
            cv2.rectangle(mask, (x0, y0), (min(width - 1, x0 + bw), min(height - 1, y0 + bh)), 1, thickness=cv2.FILLED)
    return mask.astype(bool)


def _rect(lat: float, lon: float, w_m: float, l_m: float) -> list[list[float]]:
    dlat = l_m / 111_320.0
    dlon = w_m / (111_320.0 * math.cos(math.radians(lat)))
    return [[lon, lat], [lon + dlon, lat], [lon + dlon, lat + dlat], [lon, lat + dlat], [lon, lat]]


_WARD_NAMES = ["DEMO Ward A", "DEMO Ward B", "DEMO Ward C", "DEMO Ward D", "DEMO Ward E", "DEMO Ward F"]


def demo_wards() -> dict:
    """Six rectangular demo wards (3 columns x 2 rows), each ~400 m x 300 m."""
    feats = []
    for i, name in enumerate(_WARD_NAMES):
        r, c = divmod(i, 3)
        lat = CENTER_LAT + r * 300 / 111_320.0
        lon = CENTER_LON + c * 400 / (111_320.0 * math.cos(math.radians(CENTER_LAT)))
        feats.append({"type": "Feature",
                      "properties": {"ward": name, "source": DEMO_SOURCE, "demo": True},
                      "geometry": {"type": "Polygon", "coordinates": [_rect(lat, lon, 400, 300)]}})
    return feature_collection(feats, demo=True, label=DEMO_LABEL)


def demo_footprints() -> dict:
    """Synthetic footprints; density varies by ward so the choropleth/bar charts look plausible."""
    rng = np.random.default_rng(_SEED)
    density_factor = [0.30, 0.65, 0.45, 0.8, 0.2, 0.55]
    feats, n = [], 0
    for i in range(6):
        r, c = divmod(i, 3)
        for by in range(0, 300, 25):
            for bx in range(0, 400, 25):
                if rng.random() > density_factor[i]:
                    continue
                w_m, l_m = float(rng.uniform(7, 16)), float(rng.uniform(7, 16))
                lat = CENTER_LAT + (r * 300 + by + rng.uniform(1, 6)) / 111_320.0
                lon = CENTER_LON + (c * 400 + bx + rng.uniform(1, 6)) / (111_320.0 * math.cos(math.radians(CENTER_LAT)))
                n += 1
                bid = f"DEMO{n:04d}"
                area = round(w_m * l_m, 2)
                feats.append({"type": "Feature",
                              "properties": {"id": bid, "building_id": bid, "area_m2": area, "area": area,
                                             "confidence": None, "source": DEMO_SOURCE, "demo": True},
                              "geometry": {"type": "Polygon", "coordinates": [_rect(lat, lon, w_m, l_m)]}})
    return feature_collection(feats, demo=True, label=DEMO_LABEL)


def demo_density_rows(fc: dict | None = None, wards: dict | None = None) -> list[dict]:
    res = calculate_density(fc or demo_footprints(), wards or demo_wards())
    return density_rows(res)


def demo_metrics_summary() -> dict:
    """Placeholder numbers, always suffixed so the UI shows they are not real results."""
    return {"IoU": "0.71 (demo)", "Dice": "0.83 (demo)", "Precision": "0.86 (demo)",
            "Recall": "0.81 (demo)", "Building-count error": "12 (demo)"}
