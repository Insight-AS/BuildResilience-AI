from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class BuildingProperties(BaseModel):
    id: str
    building_id: str
    area_m2: float | None = None
    area: float | None = None
    area_px: int | None = None
    confidence: float | None = None
    source: str


class BuildingFeature(BaseModel):
    type: Literal["Feature"] = "Feature"
    properties: dict[str, Any]
    geometry: dict[str, Any]


class FeatureCollectionModel(BaseModel):
    type: Literal["FeatureCollection"] = "FeatureCollection"
    features: list[BuildingFeature]
    model_config = {"extra": "allow"}


class DensityCalculateRequest(BaseModel):
    """GeoJSON in EPSG:4326. Omit `wards` to get a clearly labelled grid layer instead."""

    buildings: dict[str, Any]
    wards: dict[str, Any] | None = None
    grid_cell_size_m: float | None = Field(default=None, gt=1, le=50_000)
