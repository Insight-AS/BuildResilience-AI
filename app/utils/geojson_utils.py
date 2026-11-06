"""Small GeoJSON / encoding helpers."""
from __future__ import annotations

import base64
import io
from typing import Any

import numpy as np
from PIL import Image


def feature_collection(features: list[dict], **extra: Any) -> dict:
    return {"type": "FeatureCollection", **extra, "features": features}


def round_coords(coords: Any, ndigits: int = 7) -> Any:
    if isinstance(coords, (list, tuple)):
        if coords and isinstance(coords[0], (int, float)):
            return [round(float(c), ndigits) for c in coords]
        return [round_coords(c, ndigits) for c in coords]
    return coords


def geom_to_geojson(geom, ndigits: int = 7) -> dict:
    from shapely.geometry import mapping

    g = dict(mapping(geom))
    g["coordinates"] = round_coords(g["coordinates"], ndigits)
    return g


def mask_to_png_base64(mask: np.ndarray, max_px: int) -> dict:
    """Encode a boolean mask as a (possibly downscaled) 8-bit PNG data URI."""
    h, w = mask.shape
    img = Image.fromarray((mask.astype(np.uint8)) * 255, mode="L")
    scale = min(1.0, max_px / max(h, w))
    downscaled = scale < 1.0
    if downscaled:
        img = img.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.NEAREST)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return {
        "data_uri": "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii"),
        "width": img.width,
        "height": img.height,
        "source_width": w,
        "source_height": h,
        "downscaled": downscaled,
        "encoding": "8-bit grayscale PNG; 255 = building, 0 = background",
    }


def bbox_of(fc: dict) -> list[float] | None:
    """[minx, miny, maxx, maxy] of all coordinates in a FeatureCollection."""
    xs: list[float] = []
    ys: list[float] = []

    def walk(c):
        if c and isinstance(c[0], (int, float)):
            xs.append(c[0])
            ys.append(c[1])
        else:
            for sub in c:
                walk(sub)

    for f in fc.get("features", []):
        walk((f.get("geometry") or {}).get("coordinates", []))
    return [min(xs), min(ys), max(xs), max(ys)] if xs else None
