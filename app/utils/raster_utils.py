"""Image loading for GeoTIFF / TIFF / PNG / JPEG with size guards.

Returns an RGB uint8 array, a validity mask (False = NoData / transparent), and
georeferencing only when it genuinely exists. Nothing is fabricated for PNG/JPEG.

Oversized GeoTIFFs are automatically downsampled in memory. No permanent
downsampled raster files are created.
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
    """Validate image dimensions against the configured pixel limit."""
    if w < MIN_SIDE or h < MIN_SIDE:
        raise Unprocessable(
            f"Image is too small ({w}x{h}); "
            f"minimum is {MIN_SIDE}x{MIN_SIDE} pixels.",
            code="image_too_small",
        )

    if w * h > max_pixels:
        raise Unprocessable(
            f"Image is too large ({w}x{h} = {w * h / 1e6:.1f} MP). "
            f"Limit is {max_pixels / 1e6:.0f} MP. "
            "Clip or downsample the raster, or raise MAX_IMAGE_PIXELS "
            "if you have enough RAM.",
            code="image_too_large",
        )


def _target_dimensions(
    w: int,
    h: int,
    max_pixels: int,
) -> tuple[int, int]:
    """Calculate dimensions that fit within max_pixels while preserving aspect ratio."""
    if w * h <= max_pixels:
        return w, h

    scale = (max_pixels / (w * h)) ** 0.5

    new_w = max(MIN_SIDE, int(w * scale))
    new_h = max(MIN_SIDE, int(h * scale))

    while new_w * new_h > max_pixels:
        if new_w >= new_h:
            new_w -= 1
        else:
            new_h -= 1

    return new_w, new_h


def to_uint8(
    arr: np.ndarray,
    valid: np.ndarray | None = None,
) -> tuple[np.ndarray, bool]:
    """Convert (bands, H, W) data of any dtype to uint8.

    Returns:
        (array, was_rescaled)
    """
    if arr.dtype == np.uint8:
        return arr, False

    out = np.zeros(arr.shape, dtype=np.uint8)

    step = max(
        1,
        int(np.sqrt(arr.shape[1] * arr.shape[2] / 1_000_000)),
    )

    for b in range(arr.shape[0]):
        band = arr[b].astype(np.float32)

        sample = band[::step, ::step]

        if valid is not None:
            vs = valid[::step, ::step]

            if vs.any():
                sample = sample[vs]

        lo, hi = (
            np.percentile(sample, [2, 98])
            if sample.size
            else (0.0, 1.0)
        )

        if hi <= lo:
            hi = lo + 1.0

        out[b] = np.clip(
            (band - lo) / (hi - lo) * 255.0,
            0,
            255,
        ).astype(np.uint8)

    return out, True


def _load_tiff(
    path: Path,
    max_pixels: int,
) -> LoadedImage:
    import rasterio

    from affine import Affine
    from rasterio.enums import ColorInterp, Resampling
    from rasterio.errors import NotGeoreferencedWarning, RasterioIOError

    notes: list[str] = []

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", NotGeoreferencedWarning)

            with rasterio.open(path) as ds:

                # Check minimum dimensions without rejecting oversized TIFFs.
                if ds.width < MIN_SIDE or ds.height < MIN_SIDE:
                    raise Unprocessable(
                        f"Image is too small ({ds.width}x{ds.height}); "
                        f"minimum is {MIN_SIDE}x{MIN_SIDE} pixels.",
                        code="image_too_small",
                    )

                if ds.count < 1:
                    raise Unprocessable(
                        "TIFF has no raster bands.",
                        code="invalid_image",
                    )

                # Determine target dimensions.
                original_width = ds.width
                original_height = ds.height

                target_width, target_height = _target_dimensions(
                    original_width,
                    original_height,
                    max_pixels,
                )

                downsampled = (
                    target_width != original_width
                    or target_height != original_height
                )

                # Identify alpha and colour bands.
                alpha = next(
                    (
                        i + 1
                        for i, c in enumerate(ds.colorinterp)
                        if c == ColorInterp.alpha
                    ),
                    None,
                )

                color = [
                    i + 1
                    for i in range(ds.count)
                    if i + 1 != alpha
                ]

                if not color:
                    raise Unprocessable(
                        "TIFF contains only an alpha band.",
                        code="invalid_image",
                    )

                # Read the raster directly at the target resolution.
                # Rasterio handles the downsampling without creating
                # a separate raster file on disk.
                if len(color) >= 3:
                    data = ds.read(
                        color[:3],
                        out_shape=(
                            len(color[:3]),
                            target_height,
                            target_width,
                        ),
                        resampling=Resampling.bilinear,
                    )

                    if len(color) > 3:
                        notes.append(
                            f"{len(color)} colour bands found; "
                            "using the first three as RGB."
                        )

                else:
                    data = ds.read(
                        color[:1],
                        out_shape=(
                            1,
                            target_height,
                            target_width,
                        ),
                        resampling=Resampling.bilinear,
                    )

                    data = np.repeat(data, 3, axis=0)

                    notes.append(
                        "Single/two-band raster treated as grayscale "
                        "and replicated to RGB."
                    )

                # Resize the validity mask using nearest-neighbour.
                # This preserves the meaning of NoData/alpha pixels.
                valid = (
                    ds.dataset_mask(
                        out_shape=(
                            target_height,
                            target_width,
                        ),
                        resampling=Resampling.nearest,
                    )
                    > 0
                )

                # Preserve genuine georeferencing.
                georef = (
                    ds.crs is not None
                    and not ds.transform.is_identity
                )

                transform = None
                crs = None

                if georef:
                    # Scale the affine transform to match the resized raster.
                    scale_x = original_width / target_width
                    scale_y = original_height / target_height

                    transform = ds.transform * Affine.scale(
                        scale_x,
                        scale_y,
                    )

                    epsg = ds.crs.to_epsg()

                    crs = (
                        f"EPSG:{epsg}"
                        if epsg
                        else ds.crs.to_wkt()
                    )

    except RasterioIOError as exc:
        raise Unprocessable(
            "Could not read the TIFF/GeoTIFF file.",
            code="invalid_image",
        ) from exc

    if downsampled:
        notes.append(
            f"Large GeoTIFF automatically downsampled in memory "
            f"from {original_width}x{original_height} "
            f"to {target_width}x{target_height} pixels "
            f"to satisfy the {max_pixels / 1e6:.0f} MP limit. "
            "Georeferencing was adjusted to preserve the original extent."
        )

    # Convert to uint8 after downsampling to reduce memory usage.
    data, rescaled = to_uint8(data, valid)

    if rescaled:
        notes.append(
            "Non-8-bit imagery was rescaled to 8-bit using a "
            "2-98 percentile stretch; make sure this matches "
            "how the model was trained."
        )

    if not valid.any():
        raise Unprocessable(
            "Image contains no valid (non-NoData) pixels.",
            code="empty_image",
        )

    if not georef:
        notes.append(
            "TIFF has no usable georeferencing; "
            "results are in pixel coordinates."
        )

    return LoadedImage(
        np.ascontiguousarray(np.moveaxis(data, 0, -1)),
        valid,
        "tiff",
        georef,
        transform,
        crs,
        notes,
    )


def _load_pil(
    path: Path,
    fmt: str,
    max_pixels: int,
) -> LoadedImage:
    notes: list[str] = []

    try:
        with Image.open(path) as img:
            # PNG/JPEG retain the original pixel-limit validation.
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

                rgb = np.repeat(
                    u8[0][..., None],
                    3,
                    axis=2,
                )

                notes.append(
                    "16-bit/float image rescaled to 8-bit."
                )

            else:
                rgb = np.array(img.convert("RGB"))

    except Unprocessable:
        raise

    except (
        OSError,
        Image.DecompressionBombError,
        ValueError,
    ) as exc:
        raise Unprocessable(
            "Could not decode the image file.",
            code="invalid_image",
        ) from exc

    if valid is None:
        valid = np.ones(
            rgb.shape[:2],
            dtype=bool,
        )

    if not valid.any():
        raise Unprocessable(
            "Image is fully transparent.",
            code="empty_image",
        )

    notes.append(
        "PNG/JPEG carries no georeferencing: results are in "
        "pixel coordinates and areas in m² are unavailable."
    )

    return LoadedImage(
        np.ascontiguousarray(rgb),
        valid,
        fmt,
        False,
        None,
        None,
        notes,
    )


def load_image(
    path: Path,
    max_pixels: int,
) -> LoadedImage:
    with open(path, "rb") as fh:
        kind = sniff_image_type(fh.read(16))

    if kind == "tiff":
        return _load_tiff(path, max_pixels)

    if kind in ("png", "jpeg"):
        return _load_pil(path, kind, max_pixels)

    raise Unprocessable(
        "Unsupported or corrupt image.",
        code="invalid_image",
    )


def probe_image(
    path: Path,
    max_pixels: int,
) -> dict:
    """Cheap header-only inspection used by the upload endpoint.

    Oversized TIFFs are accepted here because they will be downsampled
    automatically during inference.
    """
    with open(path, "rb") as fh:
        kind = sniff_image_type(fh.read(16))

    if kind == "tiff":
        import rasterio

        from rasterio.errors import (
            NotGeoreferencedWarning,
            RasterioIOError,
        )

        try:
            with warnings.catch_warnings():
                warnings.simplefilter(
                    "ignore",
                    NotGeoreferencedWarning,
                )

                with rasterio.open(path) as ds:
                    if ds.width < MIN_SIDE or ds.height < MIN_SIDE:
                        raise Unprocessable(
                            f"Image is too small ({ds.width}x{ds.height}); "
                            f"minimum is {MIN_SIDE}x{MIN_SIDE} pixels.",
                            code="image_too_small",
                        )

                    georef = (
                        ds.crs is not None
                        and not ds.transform.is_identity
                    )

                    epsg = ds.crs.to_epsg() if ds.crs else None

                    target_width, target_height = _target_dimensions(
                        ds.width,
                        ds.height,
                        max_pixels,
                    )

                    return {
                        "format": "tiff",
                        "width": ds.width,
                        "height": ds.height,
                        "bands": ds.count,
                        "georeferenced": georef,
                        "crs": (
                            f"EPSG:{epsg}"
                            if epsg
                            else ("custom" if georef else None)
                        ),
                        "downsampled": (
                            target_width != ds.width
                            or target_height != ds.height
                        ),
                        "analysis_width": target_width,
                        "analysis_height": target_height,
                    }

        except RasterioIOError as exc:
            raise Unprocessable(
                "Could not read the TIFF/GeoTIFF file.",
                code="invalid_image",
            ) from exc

    try:
        with Image.open(path) as img:
            _check_size(
                img.width,
                img.height,
                max_pixels,
            )

            return {
                "format": kind,
                "width": img.width,
                "height": img.height,
                "bands": len(img.getbands()),
                "georeferenced": False,
                "crs": None,
                "downsampled": False,
                "analysis_width": img.width,
                "analysis_height": img.height,
            }

    except (
        OSError,
        Image.DecompressionBombError,
    ) as exc:
        raise Unprocessable(
            "Could not decode the image file.",
            code="invalid_image",
        ) from exc