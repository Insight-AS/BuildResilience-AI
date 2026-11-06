"""FastAPI application factory."""
from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app import __version__
from app.api.router import api_router
from app.core.config import get_settings
from app.core.errors import ApiError
from app.core.logging_config import setup_logging
from app.core.state import state
from app.services.segmentation_service import model_manager
from app.utils.validation import cleanup_old_files

log = logging.getLogger("app")


def _err(status: int, code: str, message: str, details=None) -> JSONResponse:
    body = {"error": {"code": code, "message": message}}
    if details:
        body["error"]["details"] = details
    return JSONResponse(status_code=status, content=body)


@asynccontextmanager
async def lifespan(app: FastAPI):
    s = get_settings()
    setup_logging(s.log_level, s.log_json)
    s.ensure_dirs()
    state.load_persisted()
    cleanup_old_files(s.uploads_dir, s.upload_ttl_hours)
    cleanup_old_files(s.tmp_dir, 1)
    model_manager.load(s)
    st = model_manager.status()
    log.info("backend started", extra={"mode": st["mode"], "model_loaded": st["model_loaded"], "device": st["device"]})
    if st["mode"] == "demo":
        log.warning("DEMO MODE active: responses are simulated and clearly labelled")
    yield


def create_app() -> FastAPI:
    s = get_settings()
    app = FastAPI(title=s.app_name, version=__version__, lifespan=lifespan,
                  description="Building footprint extraction, density and analytics for disaster resilience.")
    app.add_middleware(CORSMiddleware, allow_origins=list(s.frontend_origins), allow_credentials=False,
                       allow_methods=["*"], allow_headers=["*"], expose_headers=["X-Request-ID", "X-Demo-Mode", "X-Density-Kind"])

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        rid, t0 = uuid.uuid4().hex[:12], time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:  # last-resort guard; details are logged, never returned
            log.exception("unhandled error", extra={"request_id": rid, "path": request.url.path})
            response = _err(500, "internal_error", "An internal error occurred.")
        response.headers["X-Request-ID"] = rid
        if model_manager.effective_mode() == "demo":
            response.headers["X-Demo-Mode"] = "true"
        log.info("request", extra={"request_id": rid, "method": request.method, "path": request.url.path,
                                   "status": response.status_code, "ms": round((time.perf_counter() - t0) * 1000, 1)})
        return response

    @app.exception_handler(ApiError)
    async def api_error(_: Request, exc: ApiError):
        return _err(exc.status_code, exc.code, exc.message, exc.details)

    @app.exception_handler(RequestValidationError)
    async def validation_error(_: Request, exc: RequestValidationError):
        details = [{"field": ".".join(str(p) for p in e.get("loc", ())), "message": str(e.get("msg", ""))} for e in exc.errors()]
        return _err(422, "validation_error", "Request validation failed.", details)

    @app.exception_handler(StarletteHTTPException)
    async def http_error(_: Request, exc: StarletteHTTPException):
        return _err(exc.status_code, "http_error", str(exc.detail))

    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception):
        log.exception("unhandled error", extra={"path": request.url.path})
        return _err(500, "internal_error", "An internal error occurred.")

    app.include_router(api_router, prefix="/api")

    @app.get("/", include_in_schema=False)
    def root():
        return {"service": s.app_name, "docs": "/docs", "health": "/api/health"}

    return app


app = create_app()
