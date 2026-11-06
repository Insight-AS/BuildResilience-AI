from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class ConfusionCounts(BaseModel):
    tp: int
    fp: int
    fn: int
    tn: int


class EvaluationResponse(BaseModel):
    iou: float | None
    dice: float | None
    precision: float | None
    recall: float | None
    building_count_error: int
    predicted_building_count: int
    ground_truth_building_count: int
    confusion: ConfusionCounts
    level: str
    model: str
    image_width: int
    image_height: int
    notes: list[str]
    extra: dict[str, Any] | None = None
