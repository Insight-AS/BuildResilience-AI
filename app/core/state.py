"""In-memory application state (+ JSON persistence for wards, ground truth, settlement areas).

This is deliberately simple (single process, no database). State resets on restart except the
three "active" layers below, which are re-loaded from data/active/.
"""
from __future__ import annotations

import json
import logging
import threading
from collections import OrderedDict
from datetime import datetime, timezone
from typing import Any

from app.core.config import get_settings

log = logging.getLogger(__name__)
MAX_ANALYSES = 10
MAX_EVALUATIONS = 25


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class AppState:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self.reset()

    def reset(self) -> None:
        with self._lock:
            self.wards_fc: dict | None = None
            self.ground_truth_fc: dict | None = None
            self.settlements_fc: dict | None = None
            self.latest_prediction: dict | None = None
            self.analyses: OrderedDict[str, dict] = OrderedDict()
            self.evaluations: list[dict] = []
            self.uploads: dict[str, dict] = {}
            self.version = 0
            self.cache: dict[Any, Any] = {}

    # ---- helpers -------------------------------------------------------
    def _bump(self) -> None:
        self.version += 1
        self.cache.clear()

    def _persist(self, name: str, fc: dict | None) -> None:
        try:
            path = get_settings().active_dir / f"{name}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            if fc is None:
                path.unlink(missing_ok=True)
            else:
                path.write_text(json.dumps(fc), encoding="utf-8")
        except OSError:
            log.warning("could not persist layer", extra={"layer": name})

    def load_persisted(self) -> None:
        for attr, name in (("wards_fc", "wards"), ("ground_truth_fc", "ground_truth"), ("settlements_fc", "settlements")):
            path = get_settings().active_dir / f"{name}.json"
            if path.is_file():
                try:
                    setattr(self, attr, json.loads(path.read_text(encoding="utf-8")))
                except (OSError, ValueError):
                    log.warning("ignoring unreadable persisted layer", extra={"layer": name})
        self._bump()

    # ---- layers --------------------------------------------------------
    def set_wards(self, fc: dict | None) -> None:
        with self._lock:
            self.wards_fc = fc
            self._persist("wards", fc)
            self._bump()

    def set_ground_truth(self, fc: dict | None) -> None:
        with self._lock:
            self.ground_truth_fc = fc
            self._persist("ground_truth", fc)
            self._bump()

    def set_settlements(self, fc: dict | None) -> None:
        with self._lock:
            self.settlements_fc = fc
            self._persist("settlements", fc)
            self._bump()

    # ---- predictions / analyses ---------------------------------------
    def set_latest_prediction(self, analysis_id: str, fc: dict, meta: dict) -> None:
        with self._lock:
            self.latest_prediction = {"analysis_id": analysis_id, "fc": fc, **meta}
            self._bump()

    def add_analysis(self, analysis_id: str, result: dict) -> None:
        with self._lock:
            self.analyses[analysis_id] = result
            while len(self.analyses) > MAX_ANALYSES:
                self.analyses.popitem(last=False)

    def get_analysis(self, analysis_id: str) -> dict | None:
        with self._lock:
            return self.analyses.get(analysis_id)

    def add_evaluation(self, entry: dict) -> None:
        with self._lock:
            self.evaluations.append({"evaluated_at": utcnow(), **entry})
            del self.evaluations[:-MAX_EVALUATIONS]
            self._bump()

    def register_upload(self, upload_id: str, info: dict) -> None:
        with self._lock:
            self.uploads[upload_id] = info

    def get_upload(self, upload_id: str) -> dict | None:
        with self._lock:
            return self.uploads.get(upload_id)


state = AppState()
