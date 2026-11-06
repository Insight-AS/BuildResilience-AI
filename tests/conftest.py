import io
import os
import tempfile

# Configure the environment BEFORE the app is imported.
_TMP = tempfile.mkdtemp(prefix="buildresilience_test_")
os.environ.update(
    DEMO_MODE="true", MODEL_PATH=os.path.join(_TMP, "missing.pth"), DATA_DIR=os.path.join(_TMP, "data"),
    OUTPUTS_DIR=os.path.join(_TMP, "outputs"), MODELS_DIR=os.path.join(_TMP, "models"),
    MAX_UPLOAD_SIZE_MB="1", LOG_LEVEL="WARNING",
)

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from PIL import Image  # noqa: E402


@pytest.fixture
def make_client(monkeypatch):
    """Factory: make_client(DEMO_MODE='false') -> TestClient with fresh settings and state."""
    clients = []

    def _make(**env):
        from app.core.config import get_settings
        from app.core.state import state
        from app.services.segmentation_service import model_manager

        for k, v in env.items():
            monkeypatch.setenv(k, v)
        get_settings.cache_clear()
        state.reset()
        model_manager.__init__()
        from app.main import create_app

        c = TestClient(create_app(), raise_server_exceptions=False)
        c.__enter__()
        clients.append(c)
        return c

    yield _make
    for c in clients:
        c.__exit__(None, None, None)
    from app.core.config import get_settings

    get_settings.cache_clear()


@pytest.fixture
def client(make_client):
    return make_client()


def png_bytes(w=128, h=128, color=(120, 130, 90), mode="RGB") -> bytes:
    buf = io.BytesIO()
    Image.new(mode, (w, h), color).save(buf, format="PNG")
    return buf.getvalue()


def square_geojson(lon, lat, size=0.001, **props):
    ring = [[lon, lat], [lon + size, lat], [lon + size, lat + size], [lon, lat + size], [lon, lat]]
    return {"type": "Feature", "properties": props, "geometry": {"type": "Polygon", "coordinates": [ring]}}


def fc(*features):
    return {"type": "FeatureCollection", "features": list(features)}
