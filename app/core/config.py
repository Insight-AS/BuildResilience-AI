"""Runtime configuration, read from environment variables (a .env file is supported)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

BACKEND_DIR = Path(__file__).resolve().parents[2]
load_dotenv(BACKEND_DIR / ".env", override=False)


def _get(name: str, default: str) -> str:
    value = os.getenv(name)
    return default if value is None or value.strip() == "" else value.strip()


def _int(name: str, default: int) -> int:
    try:
        return int(_get(name, str(default)))
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be an integer") from exc


def _float(name: str, default: float) -> float:
    try:
        return float(_get(name, str(default)))
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be a number") from exc


def _bool(name: str, default: bool) -> bool:
    return _get(name, "true" if default else "false").lower() in {"1", "true", "yes", "on"}


def _tristate(name: str, default: str = "auto") -> str:
    v = _get(name, default).lower()
    if v in {"1", "true", "yes", "on"}:
        return "true"
    if v in {"0", "false", "no", "off"}:
        return "false"
    if v == "auto":
        return "auto"
    raise ValueError(f"Environment variable {name} must be true, false or auto")


def _path(name: str, default: str) -> Path:
    p = Path(_get(name, default))
    return p if p.is_absolute() else (BACKEND_DIR / p).resolve()


@dataclass(frozen=True)
class Settings:
    app_name: str
    environment: str
    host: str
    port: int
    frontend_origins: tuple[str, ...]
    # --- model ---
    model_path: Path
    demo_mode: str  # "true" | "false" | "auto"
    device: str  # auto | cpu | cuda
    architecture: str
    encoder_name: str
    tile_size: int
    tile_overlap: int
    batch_size: int
    threshold: float
    normalize: str  # imagenet | none
    # --- uploads / limits ---
    max_upload_size_mb: int
    max_image_pixels: int
    max_geojson_features: int
    upload_ttl_hours: int
    # --- polygonization ---
    min_building_area_m2: float
    min_building_area_px: int
    simplify_tolerance_px: float
    # --- density ---
    grid_cell_size_m: float
    max_grid_cells: int
    # --- paths ---
    data_dir: Path
    outputs_dir: Path
    models_dir: Path
    # --- misc ---
    log_level: str
    log_json: bool
    mask_preview_max_px: int

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def active_dir(self) -> Path:
        return self.data_dir / "active"

    @property
    def tmp_dir(self) -> Path:
        return self.data_dir / "tmp"

    def ensure_dirs(self) -> None:
        for d in (self.data_dir, self.uploads_dir, self.active_dir, self.tmp_dir, self.outputs_dir, self.models_dir):
            d.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    origins = tuple(
        o.strip().rstrip("/")
        for o in _get("FRONTEND_ORIGIN", "http://localhost:5173,http://127.0.0.1:5173").split(",")
        if o.strip()
    )
    s = Settings(
        app_name="BuildResilience AI Backend",
        environment=_get("APP_ENV", "development"),
        host=_get("HOST", "127.0.0.1"),
        port=_int("PORT", 8000),
        frontend_origins=origins,
        model_path=_path("MODEL_PATH", "models/best_model.pth"),
        demo_mode=_tristate("DEMO_MODE", "auto"),
        device=_get("DEVICE", "auto").lower(),
        architecture=_get("MODEL_ARCHITECTURE", "Unet"),
        encoder_name=_get("MODEL_ENCODER", "resnet34"),
        tile_size=_int("TILE_SIZE", 512),
        tile_overlap=_int("TILE_OVERLAP", 64),
        batch_size=_int("INFERENCE_BATCH_SIZE", 4),
        threshold=_float("PREDICTION_THRESHOLD", 0.5),
        normalize=_get("NORMALIZE", "imagenet").lower(),
        max_upload_size_mb=_int("MAX_UPLOAD_SIZE_MB", 100),
        max_image_pixels=_int("MAX_IMAGE_PIXELS", 25_000_000),
        max_geojson_features=_int("MAX_GEOJSON_FEATURES", 200_000),
        upload_ttl_hours=_int("UPLOAD_TTL_HOURS", 24),
        min_building_area_m2=_float("MIN_BUILDING_AREA_M2", 10.0),
        min_building_area_px=_int("MIN_BUILDING_AREA_PX", 30),
        simplify_tolerance_px=_float("SIMPLIFY_TOLERANCE_PX", 0.7),
        grid_cell_size_m=_float("GRID_CELL_SIZE_M", 250.0),
        max_grid_cells=_int("MAX_GRID_CELLS", 20_000),
        data_dir=_path("DATA_DIR", "data"),
        outputs_dir=_path("OUTPUTS_DIR", "outputs"),
        models_dir=_path("MODELS_DIR", "models"),
        log_level=_get("LOG_LEVEL", "INFO").upper(),
        log_json=_bool("LOG_JSON", False),
        mask_preview_max_px=_int("MASK_PREVIEW_MAX_PX", 2048),
    )
    if s.device not in {"auto", "cpu", "cuda"}:
        raise ValueError("DEVICE must be auto, cpu or cuda")
    if s.normalize not in {"imagenet", "none"}:
        raise ValueError("NORMALIZE must be imagenet or none")
    if s.tile_size % 32 != 0 or s.tile_size < 64:
        raise ValueError("TILE_SIZE must be a multiple of 32 and at least 64")
    if not 0 <= s.tile_overlap < s.tile_size:
        raise ValueError("TILE_OVERLAP must be >= 0 and smaller than TILE_SIZE")
    if not 0.0 < s.threshold < 1.0:
        raise ValueError("PREDICTION_THRESHOLD must be between 0 and 1")
    if s.max_upload_size_mb < 1 or s.batch_size < 1:
        raise ValueError("MAX_UPLOAD_SIZE_MB and INFERENCE_BATCH_SIZE must be >= 1")
    return s
