from fastapi import APIRouter

from app import __version__
from app.core.state import utcnow
from app.schemas.response import HealthResponse
from app.services.segmentation_service import model_manager

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    from app.core.config import get_settings

    st = model_manager.status()
    return HealthResponse(
        status="ok", version=__version__, mode=st["mode"], model_loaded=st["model_loaded"],
        checkpoint_found=st["checkpoint_found"], device=str(st["device"]), demo_mode=st["mode"] == "demo",
        configured_demo_mode=get_settings().demo_mode, architecture=st["architecture"], encoder=st["encoder"],
        model_load_error=st["load_error"], time=utcnow())
