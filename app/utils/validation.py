"""Upload and GeoJSON validation, safe file naming and stale-file cleanup."""
from __future__ import annotations

import json
import math
import re
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from app.core.errors import BadRequest, PayloadTooLarge, Unprocessable, UnsupportedMedia

IMAGE_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
GEOJSON_EXTENSIONS = {".geojson", ".json"}
MASK_EXTENSIONS = IMAGE_EXTENSIONS
CHUNK = 1024 * 1024
_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_EXT_FAMILY = {".tif": "tiff", ".tiff": "tiff", ".png": "png", ".jpg": "jpeg", ".jpeg": "jpeg"}


@dataclass
class SavedUpload:
    path: Path
    original_name: str
    size: int
    kind: str  # png | jpeg | tiff


def safe_display_name(name: str | None) -> str:
    """Sanitised copy of a client filename. Only used for display/logging, never for disk paths."""
    base = Path((name or "upload").replace("\\", "/")).name
    return re.sub(r"[^A-Za-z0-9._-]", "_", base)[:100] or "upload"


def get_extension(name: str | None) -> str:
    return Path((name or "").replace("\\", "/")).suffix.lower()


def validate_identifier(value: str, what: str = "id") -> str:
    if not _ID_RE.match(value or ""):
        raise BadRequest(f"Invalid {what}.", code="invalid_id")
    return value


def sniff_image_type(head: bytes) -> str | None:
    if head.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if head.startswith(b"\xff\xd8\xff"):
        return "jpeg"
    if head[:4] in (b"II*\x00", b"MM\x00*", b"II+\x00", b"MM\x00+"):
        return "tiff"
    return None


def save_image_upload(upload, dest_dir: Path, max_bytes: int, allowed: set[str] = IMAGE_EXTENSIONS) -> SavedUpload:
    """Stream an UploadFile to dest_dir under a server-generated name, enforcing size and type."""
    ext = get_extension(upload.filename)
    if ext not in allowed:
        raise UnsupportedMedia(f"Unsupported file type '{ext or 'none'}'. Allowed: {', '.join(sorted(allowed))}.")
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = (dest_dir / f"{uuid.uuid4().hex}{ext}").resolve()
    if dest.parent != dest_dir.resolve():  # path-traversal guard (name is server-generated; belt and braces)
        raise BadRequest("Invalid destination.")
    size, head = 0, b""
    try:
        with open(dest, "xb") as out:
            while True:
                chunk = upload.file.read(CHUNK)
                if not chunk:
                    break
                if not head:
                    head = chunk[:16]
                size += len(chunk)
                if size > max_bytes:
                    raise PayloadTooLarge(f"File exceeds the {max_bytes // (1024 * 1024)} MB upload limit.")
                out.write(chunk)
        if size == 0:
            raise Unprocessable("Uploaded file is empty.", code="empty_file")
        kind = sniff_image_type(head)
        if kind is None or kind != _EXT_FAMILY[ext]:
            raise UnsupportedMedia("File content does not match its extension or is not a supported image (PNG, JPEG, TIFF/GeoTIFF).")
    except BaseException:
        dest.unlink(missing_ok=True)
        raise
    return SavedUpload(path=dest, original_name=safe_display_name(upload.filename), size=size, kind=kind)


def read_geojson_upload(upload, max_bytes: int) -> dict:
    ext = get_extension(upload.filename)
    if ext not in GEOJSON_EXTENSIONS:
        raise UnsupportedMedia("GeoJSON files must have a .geojson or .json extension.")
    raw = upload.file.read(max_bytes + 1)
    if len(raw) > max_bytes:
        raise PayloadTooLarge(f"File exceeds the {max_bytes // (1024 * 1024)} MB upload limit.")
    if not raw:
        raise Unprocessable("Uploaded file is empty.", code="empty_file")
    try:
        data = json.loads(raw.decode("utf-8-sig"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise Unprocessable("File is not valid UTF-8 JSON.", code="invalid_json") from exc
    if not isinstance(data, dict):
        raise Unprocessable("GeoJSON root must be an object.", code="invalid_geojson")
    return data


# ----------------------------------------------------------------- GeoJSON --
def _num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _check_position(p, where: str) -> None:
    if not (isinstance(p, (list, tuple)) and len(p) >= 2 and _num(p[0]) and _num(p[1])):
        raise Unprocessable(f"{where}: invalid coordinate.", code="invalid_geojson")
    if not (-180 <= p[0] <= 180 and -90 <= p[1] <= 90):
        raise Unprocessable(
            f"{where}: coordinates must be longitude/latitude in EPSG:4326 (got {p[0]}, {p[1]}). Reproject the file first.",
            code="invalid_crs",
        )


def _check_polygon(coords, where: str) -> None:
    if not isinstance(coords, list) or not coords:
        raise Unprocessable(f"{where}: polygon needs at least one ring.", code="invalid_geojson")
    for ring in coords:
        if not isinstance(ring, list) or len(ring) < 4:
            raise Unprocessable(f"{where}: a ring needs at least 4 positions.", code="invalid_geojson")
        for p in ring:
            _check_position(p, where)
        if ring[0][:2] != ring[-1][:2]:
            raise Unprocessable(f"{where}: ring is not closed.", code="invalid_geojson")


def validate_polygon_collection(data: dict, *, max_features: int, label: str = "GeoJSON") -> dict:
    """Return a normalised FeatureCollection of Polygon/MultiPolygon features in EPSG:4326, or raise."""
    if data.get("type") == "Feature":
        data = {"type": "FeatureCollection", "features": [data]}
    if data.get("type") != "FeatureCollection" or not isinstance(data.get("features"), list):
        raise Unprocessable(f"{label} must be a GeoJSON FeatureCollection.", code="invalid_geojson")
    crs = data.get("crs")
    if crs:
        name = str(((crs.get("properties") or {}).get("name")) if isinstance(crs, dict) else crs).upper()
        if not any(t in name for t in ("4326", "CRS84")):
            raise Unprocessable(f"{label} declares a CRS other than EPSG:4326. Reproject to WGS84 lon/lat first.", code="invalid_crs")
    feats = data["features"]
    if not feats:
        raise Unprocessable(f"{label} contains no features.", code="empty_geojson")
    if len(feats) > max_features:
        raise Unprocessable(f"{label} has {len(feats)} features; the limit is {max_features}.", code="too_many_features")
    out = []
    for i, f in enumerate(feats):
        where = f"feature {i}"
        if not isinstance(f, dict) or f.get("type") != "Feature":
            raise Unprocessable(f"{where}: not a GeoJSON Feature.", code="invalid_geojson")
        g = f.get("geometry")
        if not isinstance(g, dict) or g.get("type") not in ("Polygon", "MultiPolygon"):
            raise Unprocessable(f"{where}: geometry must be Polygon or MultiPolygon.", code="invalid_geometry")
        if g["type"] == "Polygon":
            _check_polygon(g.get("coordinates"), where)
        else:
            polys = g.get("coordinates")
            if not isinstance(polys, list) or not polys:
                raise Unprocessable(f"{where}: empty MultiPolygon.", code="invalid_geometry")
            for poly in polys:
                _check_polygon(poly, where)
        props = f.get("properties")
        out.append({"type": "Feature", "properties": dict(props) if isinstance(props, dict) else {}, "geometry": g})
    return {"type": "FeatureCollection", "features": out}


_WARD_KEYS = ("ward", "ward_name", "WARD_NAME", "Ward", "name", "Name", "NAME", "admin_name")


def normalise_ward_names(fc: dict) -> dict:
    for i, f in enumerate(fc["features"], start=1):
        p = f["properties"]
        name = next((str(p[k]) for k in _WARD_KEYS if p.get(k) not in (None, "")), None)
        p["ward"] = name or f"Ward {i}"
    return fc


SETTLEMENT_CATEGORIES = {"planned", "informal"}


def normalise_settlement_categories(fc: dict) -> dict:
    for i, f in enumerate(fc["features"]):
        p = f["properties"]
        raw = p.get("category") or p.get("settlement_type") or p.get("type")
        cat = str(raw).strip().lower() if raw is not None else ""
        if cat not in SETTLEMENT_CATEGORIES:
            raise Unprocessable(
                f"feature {i}: properties.category must be 'planned' or 'informal' (got {raw!r}).",
                code="invalid_settlement_category",
            )
        p["category"] = cat
    return fc


def cleanup_old_files(directory: Path, ttl_hours: int) -> int:
    """Delete files older than ttl_hours (non-recursive; ignores .gitkeep). Returns number removed."""
    if not directory.is_dir():
        return 0
    cutoff, removed = time.time() - ttl_hours * 3600, 0
    for p in directory.iterdir():
        try:
            if p.is_file() and p.name != ".gitkeep" and p.stat().st_mtime < cutoff:
                p.unlink()
                removed += 1
        except OSError:
            continue
    return removed
