"""CRS handling and metric area computation. Never computes m² from lon/lat degrees."""
from __future__ import annotations

from functools import lru_cache
from typing import Any

from app.utils.geojson_utils import feature_collection  # noqa: F401  (re-export convenience)


def utm_crs_for(lon: float, lat: float):
    from pyproj import CRS

    zone = int((lon + 180) // 6) + 1
    zone = max(1, min(60, zone))
    return CRS.from_epsg((32600 if lat >= 0 else 32700) + zone)


@lru_cache(maxsize=64)
def _transformer(src: str, dst: str):
    from pyproj import Transformer

    return Transformer.from_crs(src, dst, always_xy=True)


def reproject_geom(geom, src: str, dst: str):
    from shapely.ops import transform

    if src == dst:
        return geom
    return transform(_transformer(src, dst).transform, geom)


def make_valid_poly(geom):
    """Return a valid Polygon/MultiPolygon (possibly empty)."""
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
    from shapely.validation import make_valid

    if geom is None or geom.is_empty:
        return Polygon()
    if not geom.is_valid:
        geom = make_valid(geom)
    if geom.geom_type in ("Polygon", "MultiPolygon"):
        return geom
    parts = [g for g in getattr(geom, "geoms", []) if g.geom_type in ("Polygon", "MultiPolygon")]
    return unary_union(parts) if parts else Polygon()


class AreaCalculator:
    """Computes geometry area in m² for a raster/vector CRS.

    * geographic CRS -> reproject to a local UTM zone (chosen at the extent centre)
    * projected CRS  -> use the CRS linear unit conversion to metres
    """

    def __init__(self, crs: str, center_xy: tuple[float, float]):
        from pyproj import CRS

        self.src = CRS.from_user_input(crs)
        self.to_metric = None
        self.factor = 1.0
        if self.src.is_geographic:
            lon, lat = _transformer(self.src.to_string(), "EPSG:4326").transform(*center_xy)
            self.metric_crs = utm_crs_for(lon, lat)
            self.to_metric = _transformer(self.src.to_string(), self.metric_crs.to_string())
        else:
            self.metric_crs = self.src
            if self.src.axis_info:
                self.factor = float(self.src.axis_info[0].unit_conversion_factor or 1.0)

    def area(self, geom) -> float:
        from shapely.ops import transform

        if self.to_metric is not None:
            return float(transform(self.to_metric.transform, geom).area)
        return float(geom.area) * self.factor**2


def pixel_area_m2(transform, crs: str, width: int, height: int) -> float:
    from shapely.geometry import Polygon

    cx, cy = width / 2.0, height / 2.0
    pts = [transform * (cx + dx, cy + dy) for dx, dy in ((0, 0), (1, 0), (1, 1), (0, 1))]
    calc = AreaCalculator(crs, transform * (cx, cy))
    return calc.area(Polygon(pts))


def fc_to_gdf(fc: dict | None):
    """FeatureCollection (EPSG:4326) -> GeoDataFrame with valid, non-empty polygonal geometries."""
    import geopandas as gpd

    feats = (fc or {}).get("features") or []
    if not feats:
        return gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs="EPSG:4326")
    gdf = gpd.GeoDataFrame.from_features(feats, crs="EPSG:4326")
    gdf["geometry"] = gdf.geometry.apply(make_valid_poly)
    return gdf[~gdf.geometry.is_empty].reset_index(drop=True)


def gdf_geometry_to_features(gdf, props_cols: list[str], ndigits: int = 7) -> list[dict[str, Any]]:
    from app.utils.geojson_utils import geom_to_geojson

    out = []
    for _, row in gdf.iterrows():
        out.append({
            "type": "Feature",
            "properties": {c: _py(row[c]) for c in props_cols},
            "geometry": geom_to_geojson(row.geometry, ndigits),
        })
    return out


def _py(v):
    try:
        import numpy as np

        if isinstance(v, np.generic):
            v = v.item()
    except ImportError:  # pragma: no cover
        pass
    if isinstance(v, float) and v != v:  # NaN -> null
        return None
    return v
