from fastapi import APIRouter

from app.api.routes import analytics, buildings, dashboard, density, health, inference, uploads

api_router = APIRouter()
for module in (health, dashboard, inference, buildings, density, analytics, uploads):
    api_router.include_router(module.router)
