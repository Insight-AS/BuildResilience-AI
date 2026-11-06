"""Orchestrates: load image -> (real | demo | edge) mask -> polygons -> GeoJSON + statistics."""
from __future__ import annotations

import logging
import time
import uuid
from pathlib import Path

import numpy as np

from app.core.config import get_settings
from app.core.errors import ModelUnavailableError, Unprocessable
from app.core.state import state, utcnow
from app.services import demo_service
from app.services.geospatial_service import pixel_area_m2
from app.services.polygonization_service import LIMITATION_TEXT, items_to_features, polygonize_mask, summarize
from app.services.segmentation_service import edge_baseline_mask, model_manager
from app.utils.geojson_utils import feature_collection, mask_to_png_base64
from app.utils.raster_utils import load_image

log = logging.getLogger(__name__)
VALID_MODELS = {"cnn", "edge"}


def _csv_summary(stats: dict, meta: dict) -> str:
    rows = [("analysis_id", meta["analysis_id"]), ("results_type", meta["results_type"]),
            ("georeferenced", meta["georeferenced"])] + list(stats.items())
    return "metric,value\n" + "\n".join(f"{k},{'' if v is None else v}" for k, v in rows)


def run_inference(*, path: Path, original_name: str, model: str = "cnn", analysis_type: str = "segmentation",
                  output_format: str = "geojson") -> dict:
    t0 = time.perf_counter()
    s = get_settings()
    model = (model or "cnn").lower()
    if model not in VALID_MODELS:
        raise Unprocessable(f"Unknown model '{model}'. Use one of: {', '.join(sorted(VALID_MODELS))}.", code="invalid_model")

    img = load_image(path, s.max_image_pixels)
    warnings = list(img.warnings)
    probability = None

    if model == "edge":
        results_type, demo = "edge_detection_baseline", False
        mask = edge_baseline_mask(img.array, img.valid_mask, s.min_building_area_px)
        engine = {"name": "edge_detection_baseline", "description": "Classical Canny-based baseline (not the CNN)."}
        warnings.append("Edge-detection baseline: a crude non-learning comparison method with no confidence scores.")
    else:
        mode = model_manager.effective_mode()
        if mode == "real":
            probability = model_manager.predict_probability(img.array)
            mask = probability >= s.threshold
            results_type, demo = "model_prediction", False
            engine = {"name": f"{s.architecture}/{s.encoder_name}", "device": model_manager.device_name, "threshold": s.threshold}
        elif mode == "demo":
            mask = demo_service.demo_mask(img.height, img.width)
            results_type, demo = "simulated_demo", True
            engine = {"name": "demo_generator", "description": demo_service.DEMO_LABEL}
            warnings.insert(0, demo_service.DEMO_LABEL + ". These polygons are NOT a prediction of the imagery content.")
        else:
            raise ModelUnavailableError(
                "No trained model checkpoint is loaded and DEMO_MODE=false. Place the checkpoint at MODEL_PATH "
                "(see backend/models/README.md) or set DEMO_MODE=auto.")
    mask = np.logical_and(mask, img.valid_mask)

    items = polygonize_mask(
        mask, transform=img.transform, crs=img.crs, probability=probability, valid_mask=img.valid_mask,
        min_area_m2=s.min_building_area_m2, min_area_px=s.min_building_area_px, simplify_tolerance_px=s.simplify_tolerance_px)
    source = {"model_prediction": "model_prediction", "simulated_demo": demo_service.DEMO_SOURCE,
              "edge_detection_baseline": "edge_baseline"}[results_type]
    features = items_to_features(items, crs=img.crs, source=source, extra_props={"demo": True} if demo else None)
    stats = summarize(items)

    # image-extent statistics (valid pixels only)
    valid_frac = float(img.valid_mask.mean())
    if img.georeferenced:
        px_m2 = pixel_area_m2(img.transform, img.crs, img.width, img.height)
        extent_km2 = px_m2 * float(img.valid_mask.sum()) / 1e6
        stats["image_area_km2"] = round(extent_km2, 4)
        stats["building_density_per_km2"] = round(len(items) / extent_km2, 2) if extent_km2 > 0 else None
        stats["building_coverage_pct"] = (round(stats["total_area_m2"] / (extent_km2 * 1e6) * 100, 2)
                                          if extent_km2 > 0 and stats["total_area_m2"] is not None else None)
    else:
        stats.update(image_area_km2=None, building_density_per_km2=None, building_coverage_pct=None)
        warnings.append("Area (m²) and density are unavailable because the input has no georeferencing.")

    analysis_id = uuid.uuid4().hex
    fc = feature_collection(features, demo=demo, results_type=results_type, georeferenced=img.georeferenced,
                            coordinate_system="EPSG:4326" if img.georeferenced else "pixel (x=column, y=row)",
                            analysis_id=analysis_id)
    meta = {"analysis_id": analysis_id, "results_type": results_type, "georeferenced": img.georeferenced}
    result = {
        "analysis_id": analysis_id,
        "demo": demo,
        "results_type": results_type,
        "status_message": (demo_service.DEMO_LABEL if demo else
                           "Edge-detection baseline result." if results_type == "edge_detection_baseline" else
                           "Result produced by the trained segmentation model."),
        "created_at": utcnow(),
        "analysis_type": analysis_type,
        "engine": engine,
        "georeferenced": img.georeferenced,
        "crs": img.crs,
        "output_crs": "EPSG:4326" if img.georeferenced else None,
        "coordinate_system": "EPSG:4326" if img.georeferenced else "pixel",
        "building_count": stats["building_count"],
        "total_building_area_m2": stats["total_area_m2"],
        "statistics": stats,
        "processing_time_s": None,  # filled below
        "image": {"filename": original_name, "format": img.format, "width": img.width, "height": img.height,
                  "valid_pixel_fraction": round(valid_frac, 4)},
        "warnings": warnings,
        "limitations": [LIMITATION_TEXT],
    }
    if output_format == "csv":
        result["csv_summary"] = _csv_summary(stats, meta)
    # bulky payloads last, so truncated JSON previews still show the summary
    result["mask"] = mask_to_png_base64(mask, s.mask_preview_max_px)
    result["geojson"] = fc
    result["processing_time_s"] = round(time.perf_counter() - t0, 3)

    # Persist
    try:
        import json

        s.outputs_dir.mkdir(parents=True, exist_ok=True)
        (s.outputs_dir / f"{analysis_id}.geojson").write_text(json.dumps(fc), encoding="utf-8")
    except OSError:
        log.warning("could not write output GeoJSON")
    state.add_analysis(analysis_id, result)
    if img.georeferenced and results_type in ("model_prediction", "simulated_demo"):
        state.set_latest_prediction(analysis_id, fc, {"demo": demo, "results_type": results_type, "created_at": result["created_at"],
                                                     "stats": stats})
    log.info("inference complete", extra={"analysis_id": analysis_id, "buildings": stats["building_count"],
                                          "results_type": results_type, "seconds": result["processing_time_s"]})
    return result
