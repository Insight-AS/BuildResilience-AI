"""Real model inference (U-Net family via segmentation_models_pytorch) and a classical edge baseline.

No demo logic lives here. If the checkpoint is missing or fails to load, `loaded` stays False and
nothing is faked.
"""
from __future__ import annotations

import logging
import threading

import cv2
import numpy as np

from app.core.config import Settings, get_settings
from app.core.errors import ModelUnavailableError

log = logging.getLogger(__name__)
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
_STATE_KEYS = ("model_state_dict", "state_dict", "model", "net", "network")


def _starts(n: int, tile: int, step: int) -> list[int]:
    if n <= tile:
        return [0]
    return list(range(0, n - tile, step)) + [n - tile]


class ModelManager:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._model = None
        self._torch = None
        self.device_name: str | None = None
        self.load_error: str | None = None
        self.checkpoint_found = False
        self.load_attempted = False

    # ------------------------------------------------------------- status --
    @property
    def loaded(self) -> bool:
        return self._model is not None

    def effective_mode(self) -> str:
        """'real' | 'demo' | 'unavailable' (for the CNN model; the edge baseline is always available)."""
        s = get_settings()
        if s.demo_mode == "true":
            return "demo"
        if self.loaded:
            return "real"
        return "demo" if s.demo_mode == "auto" else "unavailable"

    def status(self) -> dict:
        s = get_settings()
        return {
            "model_loaded": self.loaded,
            "checkpoint_found": self.checkpoint_found,
            "architecture": s.architecture,
            "encoder": s.encoder_name,
            "device": self.device_name or s.device,
            "load_error": self.load_error,
            "mode": self.effective_mode(),
        }

    # ------------------------------------------------------------ loading --
    def load(self, settings: Settings | None = None) -> None:
        s = settings or get_settings()
        with self._lock:
            self.load_attempted = True
            self._model, self.load_error = None, None
            self.checkpoint_found = s.model_path.is_file()
            if not self.checkpoint_found:
                self.load_error = "Checkpoint not found at MODEL_PATH."
                log.info("model checkpoint not found; running without CNN model")
                return
            try:
                import segmentation_models_pytorch as smp
                import torch
            except ImportError as exc:
                self.load_error = f"PyTorch / segmentation_models_pytorch not installed ({exc.name})."
                log.warning(self.load_error)
                return
            try:
                want = s.device
                if want == "cuda" and not torch.cuda.is_available():
                    log.warning("DEVICE=cuda requested but CUDA is unavailable; using CPU")
                    want = "cpu"
                device = torch.device("cuda" if (want == "auto" and torch.cuda.is_available()) or want == "cuda" else "cpu")
                model = smp.create_model(s.architecture, encoder_name=s.encoder_name, encoder_weights=None, in_channels=3, classes=1)
                ckpt = torch.load(s.model_path, map_location="cpu", weights_only=True)
                state = ckpt
                if isinstance(ckpt, dict):
                    for k in _STATE_KEYS:
                        if k in ckpt and isinstance(ckpt[k], dict):
                            state = ckpt[k]
                            break
                state = {(k[7:] if k.startswith("module.") else k): v for k, v in state.items()}
                model.load_state_dict(state, strict=True)
                model.to(device).eval()
                self._torch, self._model, self.device_name = torch, model, str(device)
                log.info("model loaded", extra={"device": self.device_name, "arch": s.architecture, "encoder": s.encoder_name})
            except Exception as exc:  # noqa: BLE001 - report any load failure without crashing the API
                msg = str(exc).replace(str(s.model_path), "<MODEL_PATH>")
                self.load_error = f"Failed to load checkpoint: {type(exc).__name__}: {msg[:300]}"
                self._model = None
                log.error(self.load_error)

    # ---------------------------------------------------------- inference --
    def predict_probability(self, rgb: np.ndarray) -> np.ndarray:
        """Return an HxW float32 building probability map using overlapping tiles (mean blending)."""
        if not self.loaded:
            raise ModelUnavailableError("No trained model is loaded.")
        s, torch = get_settings(), self._torch
        h, w, _ = rgb.shape
        tile, step = s.tile_size, s.tile_size - s.tile_overlap
        windows = [(y, x) for y in _starts(h, tile, step) for x in _starts(w, tile, step)]
        acc = np.zeros((h, w), np.float32)
        cnt = np.zeros((h, w), np.float32)
        device = torch.device(self.device_name)
        for i in range(0, len(windows), s.batch_size):
            batch, boxes = [], []
            for y, x in windows[i : i + s.batch_size]:
                crop = rgb[y : y + tile, x : x + tile]
                ch, cw = crop.shape[:2]
                padded = np.zeros((tile, tile, 3), np.uint8)
                padded[:ch, :cw] = crop
                batch.append(padded)
                boxes.append((y, x, ch, cw))
            arr = np.stack(batch).astype(np.float32) / 255.0
            if s.normalize == "imagenet":
                arr = (arr - _IMAGENET_MEAN) / _IMAGENET_STD
            tensor = torch.from_numpy(np.ascontiguousarray(arr.transpose(0, 3, 1, 2))).to(device)
            with torch.inference_mode():
                prob = torch.sigmoid(self._model(tensor))[:, 0].float().cpu().numpy()
            for j, (y, x, ch, cw) in enumerate(boxes):
                acc[y : y + ch, x : x + cw] += prob[j, :ch, :cw]
                cnt[y : y + ch, x : x + cw] += 1.0
        return acc / np.maximum(cnt, 1.0)


model_manager = ModelManager()


def edge_baseline_mask(rgb: np.ndarray, valid: np.ndarray, min_area_px: int) -> np.ndarray:
    """Classical edge-detection baseline (Canny -> morphological close -> filled external contours).

    A deliberately simple comparison baseline, NOT the CNN. Produces no confidence scores.
    """
    h, w = rgb.shape[:2]
    gray = cv2.GaussianBlur(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), (5, 5), 0)
    med = float(np.median(gray[valid])) if valid.any() else float(np.median(gray))
    edges = cv2.Canny(gray, int(max(0, 0.66 * med)), int(min(255, 1.33 * med)))
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.morphologyEx(cv2.dilate(edges, k), cv2.MORPH_CLOSE, k, iterations=2)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros((h, w), np.uint8)
    max_area = 0.05 * h * w
    for c in contours:
        if min_area_px <= cv2.contourArea(c) <= max_area:
            cv2.drawContours(mask, [c], -1, 1, thickness=cv2.FILLED)
    return mask.astype(bool) & valid
