from fastapi import APIRouter

from app.services import dashboard_service

router = APIRouter(tags=["dashboard"])


@router.get("/dashboard/summary")
def dashboard_summary() -> dict:
    """Full dashboard statistics (demo values are flagged and labelled)."""
    return dashboard_service.dashboard_summary()


@router.get("/summary")
def frontend_summary() -> dict:
    """Same data in the shape the React Overview page reads (alias used by src/services/api.js)."""
    return dashboard_service.frontend_summary()
