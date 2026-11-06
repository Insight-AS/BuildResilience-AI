"""Image loading for GeoTIFF / TIFF / PNG / JPEG with size guards.

Returns an RGB uint8 array, a validity mask (False = NoData / transparent), and georeferencing
only when it genuinely exists. Nothing is ever fabricated for PNG/JPEG.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from app.core.errors import Unprocessable
from app.utils.validation import sniff_image_type

MIN_SIDE = 16


@dataclass
class LoadedImage:
    array: np.ndarray  # H x W x 3, uint8
    valid_mask: np.ndarray  # H x W, bool
    format: str  # png | jpeg | tiff
    georeferenced: bool = False
    transform: object | None = None  # affine.Affine when georeferenced
    crs: str | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def height(self) -> int:
        return int(self.array.shape[0])

    @property
    def width(self) -> int:
        return int(self.array.shape[1])


def _check_size(w: int, h: int, max_pixels: int) -> None:
    if w < MIN_SIDE or h < MIN_SIDE:
        raise Unprocessable(f"Image is too small ({w}x{h}); minimum is {MIN_SIDE}x{MIN_SIDE} pixels.", code="image_too_small")
    if w * h > max_pixels:
        raise Unprocessable(
            f"Image is too large ({w}x{h} = {w * h / 1e6:.1f} MP). Limit is {max_pixels / 1e6:.0f} MP. "
            "Clip or downsample the raster, or raise MAX_IMAGE_PIXELS if you have enough RAM.",
            code="image_too_large",
        )


def to_uint8(arr: np.ndarray, valid: np.ndarray | None = None) -> tuple[np.ndarray, bool]:
    """Convert (bands, H, W) data of any dtype to uint8. Returns (array, was_rescaled)."""
    if arr.dtype == np.uint8:
        return arr, False
    out = np.zeros(arr.shape, dtype=np.uint8)
    step = max(1, int(np.sqrt(arr.shape[1] * arr.shape[2] / 1_000_000)))
    for b in range(arr.shape[0]):
        band = arr[b].astype(np.float32)
        sample = band[::step, ::step]
        if valid is not None:
            vs = valid[::step, ::step]
            if vs.any():
                sample = sample[vs]
        lo, hi = np.percentile(sample, [2, 98]) if sample.size else (0.0, 1.0)
        if hi <= lo:
            hi = lo + 1.0
        out[b] = np.clip((band - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)
    return out, True


def _load_tiff(path: Path, max_pixels: int) -> LoadedImage:
    import rasterio
    from rasterio.enums import ColorInterp
    from rasterio.errors import NotGeoreferencedWarning, RasterioIOError

    notes: list[str] = []
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", NotGeoreferencedWarning)
            with rasterio.open(path) as ds:
                _check_size(ds.width, ds.height, max_pixels)
                if ds.count < 1:
                    raise Unprocessable("TIFF has no raster bands.", code="invalid_image")
                alpha = next((i + 1 for i, c in enumerate(ds.colorinterp) if c == ColorInterp.alpha), None)
                color = [i + 1 for i in range(ds.count) if i + 1 != alpha]
                if not color:
                    raise Unprocessable("TIFF contains only an alpha band.", code="invalid_image")
                if len(color) >= 3:
                    data = ds.read(color[:3])
                    if len(color) > 3:
                        notes.append(f"{len(color)} colour bands found; using the first three as RGB.")
                else:
                    data = np.repeat(ds.read(color[:1]), 3, axis=0)
                    notes.append("Single/two-band raster treated as grayscale and replicated to RGB.")
                valid = ds.dataset_mask() > 0  # honours nodata and alpha
                georef = ds.crs is not None and not ds.transform.is_identity
                transform = ds.transform if georef else None
                crs = None
                if georef:
                    epsg = ds.crs.to_epsg()
                    crs = f"EPSG:{epsg}" if epsg else ds.crs.to_wkt()
    except RasterioIOError as exc:
        raise Unprocessable("Could not read the TIFF/GeoTIFF file.", code="invalid_image") from exc

    data, rescaled = to_uint8(data, valid)
    if rescaled:
        notes.append("Non-8-bit imagery was rescaled to 8-bit using a 2-98 percentile stretch; "
                     "make sure this matches how the model was trained.")
    if not valid.any():
        raise Unprocessable("Image contains no valid (non-NoData) pixels.", code="empty_image")
    if not georef:
        notes.append("TIFF has no usable georeferencing; results are in pixel coordinates.")
    return LoadedImage(np.ascontiguousarray(np.moveaxis(data, 0, -1)), valid, "tiff", georef, transform, crs, notes)


def _load_pil(path: Path, fmt: str, max_pixels: int) -> LoadedImage:
    notes: list[str] = []
    try:
        with Image.open(path) as img:
            _check_size(img.width, img.height, max_pixels)
            img.load()
            valid = None
            if "A" in img.getbands() or "transparency" in img.info:
                rgba = np.array(img.convert("RGBA"))
                valid = rgba[..., 3] > 0
                rgb = rgba[..., :3]
            elif img.mode in ("I;16", "I", "F"):
                raw = np.array(img).astype(np.float32)[None]
                u8, _ = to_uint8(raw)
                rgb = np.repeat(u8[0][..., None], 3, axis=2)
                notes.append("16-bit/float image rescaled to 8-bit.")
            else:
                rgb = np.array(img.convert("RGB"))
    except Unprocessable:
        raise
    except (OSError, Image.DecompressionBombError, ValueError) as exc:
        raise Unprocessable("Could not decode the image file.", code="invalid_image") from exc
    if valid is None:
        valid = np.ones(rgb.shape[:2], dtype=bool)
    if not valid.any():
        raise Unprocessable("Image is fully transparent.", code="empty_image")
    notes.append("PNG/JPEG carries no georeferencing: results are in pixel coordinates and areas in m² are unavailable.")
    return LoadedImage(np.ascontiguousarray(rgb), valid, fmt, False, None, None, notes)


def load_image(path: Path, max_pixels: int) -> LoadedImage:
    with open(path, "rb") as fh:
        kind = sniff_image_type(fh.read(16))
    if kind == "tiff":
        return _load_tiff(path, max_pixels)
    if kind in ("png", "jpeg"):
        return _load_pil(path, kind, max_pixels)
    raise Unprocessable("Unsupported or corrupt image.", code="invalid_image")


def probe_image(path: Path, max_pixels: int) -> dict:
    """Cheap header-only inspection (no pixel decode) used by the upload endpoint."""
    with open(path, "rb") as fh:
        kind = sniff_image_type(fh.read(16))
    if kind == "tiff":
        import rasterio
        from rasterio.errors import NotGeoreferencedWarning, RasterioIOError

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", NotGeoreferencedWarning)
                with rasterio.open(path) as ds:
                    _check_size(ds.width, ds.height, max_pixels)
                    georef = ds.crs is not None and not ds.transform.is_identity
                    epsg = ds.crs.to_epsg() if ds.crs else None
                    return {"format": "tiff", "width": ds.width, "height": ds.height, "bands": ds.count,
                            "georeferenced": georef, "crs": (f"EPSG:{epsg}" if epsg else ("custom" if georef else None))}
        except RasterioIOError as exc:
            raise Unprocessable("Could not read the TIFF/GeoTIFF file.", code="invalid_image") from exc
    try:
        with Image.open(path) as img:
            _check_size(img.width, img.height, max_pixels)
            return {"format": kind, "width": img.width, "height": img.height, "bands": len(img.getbands()),
                    "georeferenced": False, "crs": None}
    except (OSError, Image.DecompressionBombError) as exc:
        raise Unprocessable("Could not decode the image file.", code="invalid_image") from exc
