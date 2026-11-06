import logging

import numpy as np
from fastapi import APIRouter, File, Form, UploadFile
from PIL import Image

from app.core.config import get_settings
from app.core.errors import Unprocessable
from app.core.state import state
from app.services import analytics_service, dashboard_service
from app.utils.validation import save_image_upload

log = logging.getLogger(__name__)
router = APIRouter(tags=["analytics"])


@router.get("/analytics/metrics")
@router.get("/metrics")  # alias used by src/services/api.js
def metrics() -> dict:
    """Available evaluation metrics. Returns status 'unavailable' (never invented numbers) until
    an evaluation, ground truth + prediction, or models/training_report.json exists."""
    return dashboard_service.metrics_payload()


def _read_mask(upload: UploadFile, threshold: int) -> np.ndarray:
    s = get_settings()
    saved = save_image_upload(upload, s.tmp_dir, s.max_upload_bytes)
    try:
        with Image.open(saved.path) as img:
            if img.width * img.height > s.max_image_pixels:
                raise Unprocessable("Mask is too large.", code="image_too_large")
            arr = np.array(img.convert("L"))
    except (OSError, Image.DecompressionBombError) as exc:
        raise Unprocessable("Could not decode mask image.", code="invalid_image") from exc
    finally:
        saved.path.unlink(missing_ok=True)
    return arr > threshold


@router.post("/analytics/evaluate")
def evaluate(
    prediction_mask: UploadFile = File(...),
    ground_truth_mask: UploadFile = File(...),
    model: str = Form("cnn"),
    threshold: int = Form(0, ge=0, le=254),
    min_component_px: int = Form(1, ge=1, le=10_000),
) -> dict:
    """Compare a predicted mask with a ground-truth mask (same width x height; pixels > threshold = building).
    Use threshold=127 for JPEG masks. Results are also stored for /api/metrics."""
    get_settings().ensure_dirs()
    pred = _read_mask(prediction_mask, threshold)
    gt = _read_mask(ground_truth_mask, threshold)
    m = analytics_service.mask_metrics(pred, gt, min_component_px=min_component_px)
    notes = ["Pixel-level metrics over all pixels in the masks.",
             "Building counts are 4-connected components; touching buildings count as one."]
    if gt.sum() == 0:
        notes.append("Ground-truth mask contains no building pixels: IoU/Dice/Recall are undefined or trivial.")
    entry = {**m, "model": model[:40], "image_width": int(pred.shape[1]), "image_height": int(pred.shape[0]), "notes": notes}
    state.add_evaluation(entry)
    return entry


@router.get("/analytics/settlement-comparison")
def settlement_comparison() -> dict:
    """Planned vs informal metrics, only when settlement polygons, ground truth and a real prediction exist."""
    lp = state.latest_prediction
    try:
        return analytics_service.settlement_comparison(
            lp["fc"] if lp else None, state.ground_truth_fc, state.settlements_fc, pred_is_demo=bool(lp and lp.get("demo")))
    except ImportError as exc:
        log.error("geospatial dependency missing: %s", exc)
        raise
