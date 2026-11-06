from __future__ import annotations

from pydantic import BaseModel


class ErrorBody(BaseModel):
    code: str
    message: str
    details: list | dict | None = None


class ErrorResponse(BaseModel):
    error: ErrorBody


class HealthResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    status: str
    version: str
    mode: str  # real | demo | unavailable
    model_loaded: bool
    checkpoint_found: bool
    device: str
    demo_mode: bool  # effective: are responses currently simulated?
    configured_demo_mode: str
    architecture: str
    encoder: str
    model_load_error: str | None = None
    time: str
