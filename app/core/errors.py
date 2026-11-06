"""Application errors with HTTP status codes and stable machine-readable codes."""
from __future__ import annotations


class ApiError(Exception):
    def __init__(self, status_code: int, code: str, message: str, details: dict | None = None):
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message
        self.details = details


class BadRequest(ApiError):
    def __init__(self, message: str, code: str = "bad_request", details: dict | None = None):
        super().__init__(400, code, message, details)


class Unprocessable(ApiError):
    def __init__(self, message: str, code: str = "invalid_input", details: dict | None = None):
        super().__init__(422, code, message, details)


class NotFound(ApiError):
    def __init__(self, message: str, code: str = "not_found"):
        super().__init__(404, code, message)


class PayloadTooLarge(ApiError):
    def __init__(self, message: str):
        super().__init__(413, "file_too_large", message)


class UnsupportedMedia(ApiError):
    def __init__(self, message: str):
        super().__init__(415, "unsupported_file_type", message)


class ModelUnavailableError(ApiError):
    def __init__(self, message: str):
        super().__init__(503, "model_unavailable", message)
