import logging
import uuid
from pathlib import Path

from fastapi import APIRouter, File, Form, UploadFile

from app.core.config import get_settings
from app.core.errors import BadRequest, NotFound, Unprocessable
from app.core.state import state
from app.services.inference_service import run_inference
from app.utils.validation import save_image_upload, validate_identifier

log = logging.getLogger(__name__)
router = APIRouter(tags=["inference"])
_TYPES = {"segmentation", "footprints", "density"}
_FORMATS = {"geojson", "png", "csv"}


@router.post("/inference/predict")
@router.post("/predict")  # alias used by src/services/api.js
def predict(
    file: UploadFile | None = File(None),
    upload_id: str | None = Form(None),
    model: str = Form("cnn"),
    analysis_type: str = Form("segmentation", alias="type"),
    output_format: str = Form("geojson", alias="format"),
) -> dict:
    """Run building extraction on an uploaded image (multipart `file`) or a previous `upload_id`."""
    if analysis_type not in _TYPES:
        raise Unprocessable(f"type must be one of {sorted(_TYPES)}", code="invalid_type")
    if output_format not in _FORMATS:
        raise Unprocessable(f"format must be one of {sorted(_FORMATS)}", code="invalid_format")
    s = get_settings()
    s.ensure_dirs()
    if file is not None and file.filename:
        saved = save_image_upload(file, s.tmp_dir, s.max_upload_bytes)
        try:
            return run_inference(path=saved.path, original_name=saved.original_name, model=model,
                                 analysis_type=analysis_type, output_format=output_format)
        finally:
            saved.path.unlink(missing_ok=True)  # temporary file cleanup
    if upload_id:
        info = state.get_upload(validate_identifier(upload_id, "upload_id"))
        if not info or not Path(info["path"]).is_file():
            raise NotFound("Unknown or expired upload_id.", code="upload_not_found")
        return run_inference(path=Path(info["path"]), original_name=info["filename"], model=model,
                             analysis_type=analysis_type, output_format=output_format)
    raise BadRequest("Provide an image as multipart field 'file' (or an 'upload_id').", code="missing_file")


@router.get("/analysis/{analysis_id}")
def get_analysis(analysis_id: str) -> dict:
    validate_identifier(analysis_id, "analysis_id")
    result = state.get_analysis(analysis_id)
    if result is None:
        raise NotFound("Analysis not found (results are kept in memory for the most recent runs only).", code="analysis_not_found")
    return result
