import numpy as np
import pytest

from app.core.errors import Unprocessable
from app.services.analytics_service import count_components, mask_metrics, settlement_comparison
from tests.conftest import png_bytes


def _sq(n=20, boxes=()):
    m = np.zeros((n, n), bool)
    for y0, y1, x0, x1 in boxes:
        m[y0:y1, x0:x1] = True
    return m


def test_perfect_overlap():
    m = _sq(boxes=[(2, 8, 2, 8)])
    r = mask_metrics(m, m)
    assert r["iou"] == 1.0 and r["dice"] == 1.0 and r["precision"] == 1.0 and r["recall"] == 1.0
    assert r["building_count_error"] == 0


def test_half_overlap_formulas():
    gt = _sq(boxes=[(0, 10, 0, 10)])      # 100 px
    pred = _sq(boxes=[(0, 10, 5, 15)])    # 100 px, 50 overlap
    r = mask_metrics(pred, gt)
    assert r["confusion"] == {"tp": 50, "fp": 50, "fn": 50, "tn": 400 - 150}
    assert r["iou"] == round(50 / 150, 4)
    assert r["dice"] == round(2 * 50 / 200, 4)
    assert r["precision"] == 0.5 and r["recall"] == 0.5


def test_undefined_metrics_are_none_not_zero():
    r = mask_metrics(_sq(), _sq())  # nothing predicted, nothing in ground truth
    assert r["iou"] is None and r["dice"] is None and r["precision"] is None and r["recall"] is None
    r = mask_metrics(_sq(), _sq(boxes=[(0, 5, 0, 5)]))  # nothing predicted
    assert r["precision"] is None and r["recall"] == 0.0 and r["iou"] == 0.0


def test_building_count_error_absolute():
    gt = _sq(boxes=[(0, 4, 0, 4), (8, 12, 8, 12), (15, 19, 0, 4)])
    pred = _sq(boxes=[(0, 4, 0, 4)])
    r = mask_metrics(pred, gt)
    assert (r["predicted_building_count"], r["ground_truth_building_count"], r["building_count_error"]) == (1, 3, 2)
    assert count_components(_sq(boxes=[(0, 4, 0, 4), (4, 8, 4, 8)])) == 2  # diagonal contact is NOT merged (4-connectivity)


def test_shape_mismatch_raises():
    with pytest.raises(Unprocessable):
        mask_metrics(np.zeros((5, 5)), np.zeros((6, 6)))


def test_settlement_comparison_refuses_without_inputs():
    r = settlement_comparison(None, None, None)
    assert r["status"] == "unavailable" and len(r["missing"]) == 3
    r = settlement_comparison({"features": [1]}, {"features": [1]}, {"features": [1]}, pred_is_demo=True)
    assert r["status"] == "unavailable"  # demo predictions never produce settlement metrics


def test_evaluate_endpoint_and_metrics_store(client):
    from PIL import Image
    import io

    def mask_png(box):
        a = np.zeros((32, 32), np.uint8)
        y0, y1, x0, x1 = box
        a[y0:y1, x0:x1] = 255
        b = io.BytesIO()
        Image.fromarray(a).save(b, format="PNG")
        return b.getvalue()

    files = {"prediction_mask": ("p.png", mask_png((0, 10, 0, 10)), "image/png"),
             "ground_truth_mask": ("g.png", mask_png((0, 10, 5, 15)), "image/png")}
    r = client.post("/api/analytics/evaluate", files=files, data={"model": "cnn"})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["iou"] == round(50 / 150, 4) and d["building_count_error"] == 0
    m = client.get("/api/analytics/metrics").json()
    assert m["status"] == "available" and m["source"] == "mask_evaluation" and m["summary"]["IoU"] == d["iou"]


def test_evaluate_rejects_size_mismatch(client):
    files = {"prediction_mask": ("p.png", png_bytes(32, 32), "image/png"),
             "ground_truth_mask": ("g.png", png_bytes(64, 64), "image/png")}
    r = client.post("/api/analytics/evaluate", files=files)
    assert r.status_code == 422 and r.json()["error"]["code"] == "shape_mismatch"
