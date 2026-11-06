"""Assembles what the frontend reads (footprints, wards, density rows, summary, metrics).

Rules: real data always wins. Demo content is only produced in effective demo mode and is always
labelled (flags + 'DEMO' naming + '(demo)' suffixes where the UI prints values verbatim).
"""
from __future__ import annotations

import json
import logging

from app.core.config import get_settings
from app.core.state import state
from app.services import analytics_service, demo_service
from app.services.density_service import calculate_density, density_rows
from app.services.segmentation_service import model_manager
from app.utils.geojson_utils import feature_collection

log = logging.getLogger(__name__)
EMPTY_MSG = "No data yet: upload an image and run analysis, or enable demo mode."


def is_demo() -> bool:
    return model_manager.effective_mode() == "demo"


def prediction_features() -> tuple[list[dict], bool]:
    """(features, is_demo_content) for the predicted-footprints layer."""
    lp = state.latest_prediction
    if lp:
        return lp["fc"]["features"], bool(lp.get("demo"))
    if is_demo() and state.ground_truth_fc is None:
        return demo_service.demo_footprints()["features"], True
    return [], False


def ground_truth_features() -> list[dict]:
    if not state.ground_truth_fc:
        return []
    out = []
    for i, f in enumerate(state.ground_truth_fc["features"], start=1):
        props = dict(f.get("properties") or {})
        props.setdefault("id", f"GT{i:03d}")
        props["building_id"] = props.get("building_id") or props["id"]
        props["source"] = "ground_truth"
        out.append({**f, "properties": props})
    return out


def footprints_fc(source: str | None = None) -> dict:
    pred, demo = prediction_features()
    gt = ground_truth_features()
    feats = {"prediction": pred, "ground_truth": gt}.get(source or "", gt + pred)
    extra = {"demo": demo, "label": demo_service.DEMO_LABEL} if demo else {"demo": False}
    if not feats:
        extra["message"] = EMPTY_MSG
    return feature_collection(feats, **extra)


def wards_fc() -> dict:
    if state.wards_fc:
        return feature_collection(state.wards_fc["features"], demo=False)
    if is_demo() and not state.latest_prediction:
        return demo_service.demo_wards()
    return feature_collection([], demo=False, message="No ward boundaries uploaded.")


def current_density() -> dict | None:
    """Density result (wards if available, otherwise a labelled grid) or None."""
    key = ("density", state.version, model_manager.effective_mode())
    if key in state.cache:
        return state.cache[key]
    pred, demo = prediction_features()
    result = None
    if pred:
        wards = state.wards_fc
        if wards is None and demo and not state.latest_prediction:
            wards = demo_service.demo_wards()
        s = get_settings()
        try:
            result = calculate_density(feature_collection(pred), wards, grid_cell_size_m=s.grid_cell_size_m,
                                       max_grid_cells=s.max_grid_cells)
            result["demo"] = demo
        except Exception as exc:  # noqa: BLE001 - never let a density failure break the dashboard
            log.warning("density computation failed: %s", exc)
    state.cache[key] = result
    return result


def density_response(fmt: str | None = None):
    res = current_density()
    if fmt == "geojson":
        return res or feature_collection([], message=EMPTY_MSG)
    return density_rows(res) if res else []


# ------------------------------------------------------------------ metrics --
def load_training_report() -> dict:
    """Optional models/training_report.json exported from the Kaggle notebook (real numbers only)."""
    path = get_settings().models_dir / "training_report.json"
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        log.warning("training_report.json is unreadable")
        return {}


def _vector_metrics_latest() -> dict | None:
    lp = state.latest_prediction
    if not lp or lp.get("demo") or not state.ground_truth_fc:
        return None
    key = ("vector_metrics", state.version)
    if key not in state.cache:
        try:
            state.cache[key] = analytics_service.evaluate_vector(lp["fc"], state.ground_truth_fc)
        except Exception as exc:  # noqa: BLE001
            log.warning("vector evaluation failed: %s", exc)
            state.cache[key] = None
    m = state.cache[key]
    return m if m and m.get("status") == "available" else None


def metrics_payload() -> dict:
    report = load_training_report()
    chosen, source = None, None
    if state.evaluations:
        chosen, source = state.evaluations[-1], "mask_evaluation"
    elif (vm := _vector_metrics_latest()) is not None:
        chosen, source = vm, "vector_overlap_latest_prediction_vs_ground_truth"
    elif isinstance(report.get("summary"), dict):
        s = report["summary"]
        chosen = {"iou": s.get("IoU"), "dice": s.get("Dice"), "precision": s.get("Precision"),
                  "recall": s.get("Recall"), "building_count_error": s.get("Building-count error")}
        source = "training_report.json"
    summary = analytics_service.to_frontend_summary(chosen)
    demo_values = False
    if chosen is None and is_demo():
        summary, demo_values, source = demo_service.demo_metrics_summary(), True, "demo_placeholder"
    available = chosen is not None
    return {
        # --- keys read by the React Performance page ---
        "summary": summary,
        "comparison": report.get("comparison") if isinstance(report.get("comparison"), list) else [],
        "training": report.get("training") if isinstance(report.get("training"), list) else [],
        "experiments": report.get("experiments") if isinstance(report.get("experiments"), list) else [],
        # --- extra keys ---
        "status": "available" if available else ("demo" if demo_values else "unavailable"),
        "demo": demo_values,
        "source": source,
        "metrics": None if chosen is None else {k: chosen.get(k) for k in
                  ("iou", "dice", "precision", "recall", "building_count_error", "level")},
        "evaluations": state.evaluations[-10:],
        "message": None if (available or demo_values) else
        "No evaluation has been run. POST masks to /api/analytics/evaluate, upload ground-truth footprints, "
        "or add models/training_report.json. Metrics are never fabricated.",
    }


# ------------------------------------------------------------------ summary --
def _model_label() -> str:
    st = model_manager.status()
    if st["mode"] == "real":
        return f"Trained model active ({st['architecture']}/{st['encoder']} on {st['device']})"
    if st["mode"] == "demo":
        why = "model checkpoint not loaded" if not st["model_loaded"] else "DEMO_MODE forced"
        return f"DEMO MODE: simulated results ({why}). No trained model is producing these numbers."
    return "No trained model loaded (DEMO_MODE=false). " + (st["load_error"] or "")


def dashboard_summary() -> dict:
    pred, demo = prediction_features()
    dens = current_density()
    areas = [f["properties"].get("area_m2") for f in pred if f["properties"].get("area_m2") is not None]
    total_area = round(float(sum(areas)), 2) if areas else None
    ds = dens["summary"] if dens else {}
    return {
        "mode": model_manager.effective_mode(),
        "demo": demo,
        "label": demo_service.DEMO_LABEL if demo else None,
        "model": _model_label(),
        "total_buildings": len(pred) if pred else None,
        "total_footprint_area_m2": total_area,
        "average_building_area_m2": round(total_area / len(areas), 2) if areas else None,
        "overall_density_per_km2": ds.get("overall_density_per_km2"),
        "mean_polygon_density_per_km2": ds.get("mean_polygon_density_per_km2"),
        "overall_coverage_pct": ds.get("overall_coverage_pct"),
        "density_kind": ds.get("kind"),
        "wards_analyzed": ds.get("polygons_analyzed") if ds.get("kind") == "ward" else None,
        "metrics": metrics_payload()["summary"],
        "latest_analysis_id": state.latest_prediction["analysis_id"] if state.latest_prediction else None,
    }


def frontend_summary() -> dict:
    """Shape read by Overview.jsx. Values are printed verbatim, so display strings carry units / (demo)."""
    d = dashboard_summary()
    tag = " (demo)" if d["demo"] else ""
    dens = d["mean_polygon_density_per_km2"] if d["mean_polygon_density_per_km2"] is not None else d["overall_density_per_km2"]
    return {
        "total_buildings": f"{d['total_buildings']:,}{tag}" if d["total_buildings"] is not None else None,
        "total_footprint_area": f"{d['total_footprint_area_m2']:,.0f} m²{tag}" if d["total_footprint_area_m2"] is not None else None,
        "avg_density": f"{dens:,.1f}{tag}" if dens is not None else None,
        "wards_analyzed": f"{d['wards_analyzed']}{tag}" if d["wards_analyzed"] is not None else None,
        "model": d["model"],
        "mode": d["mode"],
        "demo": d["demo"],
        "details": d,
    }
