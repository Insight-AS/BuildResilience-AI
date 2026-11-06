# BuildResilience AI — Backend

FastAPI backend for **Building Footprint Mapping for Disaster Resilience**: it takes aerial imagery, extracts building footprint polygons with a segmentation model, computes building density per ward (or per labelled grid cell), evaluates the model, and serves everything to the existing React/Vite frontend.

> **Status honesty.** No trained checkpoint exists yet, so the backend runs in **DEMO MODE**. Everything demo is flagged (`demo: true`, `results_type: "simulated_demo"`, `source: "demo_simulated"`, `DEMO` ward names, `(demo)` suffixes on numbers the UI prints verbatim, and an `X-Demo-Mode: true` header). Real inference only happens when a valid checkpoint loads. Metrics are never fabricated: they are `null` / `"unavailable"` until real evaluation data exists.

---

## 1. Overview

| Capability | How |
|---|---|
| Inputs | GeoTIFF, TIFF, PNG, JPEG (size/pixel limits, magic-byte checks) |
| Model | `segmentation_models_pytorch` (default U-Net + ResNet-34), tiled CPU/CUDA inference |
| Footprints | connected components → `rasterio.features.shapes` → Shapely clean-up → GeoJSON (EPSG:4326) |
| Density | count/km², coverage %, mean area; computed in a projected (UTM) CRS, never in degrees |
| Metrics | IoU, Dice, precision, recall, building-count error; planned-vs-informal only with real labels |
| Edge baseline | `model=edge` runs a classical Canny baseline (the frontend offers it as a comparison) |

## 2. Architecture

```
React (Vite)  ──HTTP──▶  FastAPI routes (app/api/routes)
                              │
          ┌───────────────────┼────────────────────────────┐
          ▼                   ▼                            ▼
   inference_service   dashboard_service              analytics_service
   (orchestration)     (what the UI reads)            (metrics, settlements)
     │  │  │                 │
     │  │  └ demo_service    └ density_service ── geospatial_service (CRS / m²)
     │  └ polygonization_service
     └ segmentation_service (ModelManager + edge baseline)   utils/: raster, geojson, validation
```
Demo logic (`demo_service.py`) is separate from real inference (`segmentation_service.py`). State lives in memory (`core/state.py`); ward / ground-truth / settlement layers are also saved to `data/active/` and reloaded on start.

## 3. Folder structure

```
backend/
├── app/
│   ├── main.py                  # app factory, CORS, error handlers, request logging
│   ├── core/                    # config.py, logging_config.py, errors.py, state.py
│   ├── api/routes/              # health, dashboard, inference, buildings, density, analytics, uploads
│   ├── services/                # inference, segmentation, polygonization, density, analytics,
│   │                            # geospatial, dashboard, demo
│   ├── schemas/                 # pydantic models
│   └── utils/                   # raster_utils, geojson_utils, validation
├── models/                      # put best_model.pth here (git-ignored) + README.md
├── data/  outputs/              # runtime files (git-ignored)
├── tests/  docs/api_examples.md
├── requirements.txt  .env.example  .gitignore  pytest.ini  run.py  README.md
```

## 4–6. Setup and run (Windows PowerShell)

Python **3.11 or 3.12** (64-bit) is recommended; the geospatial wheels (rasterio, fiona, pyproj) are most reliable there.

```powershell
cd backend
python -m venv venv
venv\Scripts\Activate.ps1          # if blocked:  Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
python run.py
```
Open http://127.0.0.1:8000/docs (interactive API) and http://127.0.0.1:8000/api/health.

* `pip install torch` from PyPI on Windows installs the **CPU** build, which is fine for demos. For an NVIDIA GPU, install torch from https://pytorch.org/get-started/locally/ first, then `pip install -r requirements.txt`.
* If `pip` fails on `fiona`/`rasterio`/`geopandas` for your Python version, use Python 3.11, or a conda environment (`conda install -c conda-forge rasterio geopandas fiona`) then `pip install` the rest.
* `pip freeze > requirements.lock.txt` after a working install to pin exact versions for teammates.

## 7. API documentation

All paths are under `/api`. Errors always look like `{"error": {"code": "...", "message": "...", "details": [...]}}`.

The existing frontend calls **different paths than the original spec**, so both are served (the frontend’s `src/services/api.js` was inspected and not modified):

| Frontend calls (`api.js`) | Spec path | Notes |
|---|---|---|
| `GET /health` | `GET /health` | status, `model_loaded`, `device`, `demo_mode`, `mode` |
| `GET /summary` | `GET /dashboard/summary` | `/summary` returns display strings for the 4 Overview cards (`"593 (demo)"`); `/dashboard/summary` returns full numerics |
| `POST /predict` (multipart `file`, `model`, `type`, `format`) | `POST /inference/predict` | also accepts `upload_id` instead of `file` |
| `GET /footprints` | `GET /buildings` | FeatureCollection; ground truth has `source:"ground_truth"` (MapExplorer splits layers on this) |
| – | `GET /buildings/{building_id}` | single Feature |
| `GET /wards` | – | ward FeatureCollection |
| `GET /density` → **array** `[{ward,count,density,area,coverage}]` | `GET /density` | spec wanted a layer; the UI needs rows, so rows are the default. Layer: `GET /density?format=geojson` or `GET /density/layer` |
| – | `POST /density/calculate` | JSON `{buildings, wards?, grid_cell_size_m?}` (EPSG:4326 GeoJSON) |
| `GET /metrics` → `{summary, comparison, training, experiments}` | `GET /analytics/metrics` | same payload on both |
| – | `POST /analytics/evaluate` | multipart `prediction_mask`, `ground_truth_mask`, optional `threshold`, `min_component_px`, `model` |
| – | `GET /analytics/settlement-comparison` | `status: "unavailable"` + reason unless inputs exist |
| `GET /analysis/{id}` | – | a stored `/predict` result (last 10 kept in memory) |
| – | `POST /uploads/image` | validates + stores, returns `upload_id` |
| – | `POST /uploads/ward-boundaries` | GeoJSON polygons, EPSG:4326 |
| – | `POST /uploads/ground-truth` | ground-truth footprints (GeoJSON, EPSG:4326) |
| – | `POST /uploads/settlement-areas` | polygons with `properties.category` = `planned`/`informal` |

**Predict response** (summary first, bulky fields last): `analysis_id`, `demo`, `results_type` (`model_prediction` | `simulated_demo` | `edge_detection_baseline`), `status_message`, `georeferenced`, `crs`, `output_crs`, `building_count`, `total_building_area_m2`, `statistics`, `processing_time_s`, `image`, `warnings`, `limitations`, then `mask` (PNG data URI) and `geojson`. Feature properties: `building_id` (`B001`…), `id`, `area_m2`, `area` (alias for the popup), `area_px`, `confidence`, `source`.

* **PNG/JPEG or un-referenced TIFF** → coordinates are image pixels (`x=column, y=row`), `area_m2`/density are `null`, and a warning says so. Such results are *not* served on the map.
* **GeoTIFF** → CRS and affine transform are preserved; NoData/alpha become a validity mask; areas are computed in metres; output GeoJSON is reprojected to EPSG:4326 for Leaflet.

Examples: `docs/api_examples.md`.

## 8. Demo mode

`DEMO_MODE` in `.env`:

* `auto` (default): demo only while no valid checkpoint is loaded.
* `true`: always simulated (even if a checkpoint exists), useful for UI rehearsals.
* `false`: never simulate; `/predict` returns **503** until a checkpoint loads, and data endpoints return empty/`null`.

In demo mode: six `DEMO Ward A–F` polygons, ~590 synthetic footprints near the frontend’s placeholder centre, density rows computed from them by the real density code, and uploaded images get synthetic rectangles (not a reading of your image). `comparison`/`training` charts stay empty (nothing invented); metric cards show `0.71 (demo)`-style placeholders. The frontend’s own “Sample preview” toggle is independent.

## 9–10. Real model integration and checkpoint placement

1. Copy the Kaggle checkpoint to `backend/models/best_model.pth` (details and export snippet in `models/README.md`).
2. Make `.env` match training: `MODEL_ARCHITECTURE=Unet`, `MODEL_ENCODER=resnet34`, `NORMALIZE=imagenet|none`, `PREDICTION_THRESHOLD`, `TILE_SIZE`.
3. Restart. `GET /api/health` should show `"model_loaded": true, "mode": "real"`. If not, `model_load_error` says why (paths are redacted).
4. Optional: export `models/training_report.json` from your notebook for the Performance page (format in `models/README.md`).

Checkpoint requirements: `state_dict` (optionally nested under `model_state_dict`/`state_dict`/`model`), 3 input channels, 1 output class, loaded with `weights_only=True`. Strict key matching is used so a wrong architecture fails loudly instead of silently producing garbage.

## 11. Connect the React frontend (no frontend changes needed)

The frontend `.env` already contains `VITE_API_BASE_URL=http://localhost:8000/api`. Terminal 1: backend (`python run.py`). Terminal 2, repo root:

```powershell
npm install
npm run dev
```
CORS allows `http://localhost:5173` and `http://127.0.0.1:5173` by default; change `FRONTEND_ORIGIN` (comma-separated) if Vite uses another port.

## 12. Tests

```powershell
cd backend
venv\Scripts\Activate.ps1
pytest -v
```
Tests run without a checkpoint (demo mode, temp directories). Geospatial tests skip automatically if rasterio/geopandas are missing.

## 13. Known limitations

* **Instances:** one connected mask region ≠ one building. Touching/overlapping buildings merge; occluded buildings can split. Counts and count-error are approximate. No watershed/instance separation yet.
* **Metrics are pixel/area based**, as specified; they are not instance-matched (no F1@IoU).
* **Frontend timeout:** `src/services/api.js` aborts every request after **8 s** and reports it as “backend offline”. CPU inference on large rasters can exceed that. Use small tiles for the live demo, run on GPU, or (your call, I did not touch the frontend) raise the timeout for `predict`.
* **Frontend labelling:** the UI only shows its “sample” badge when its own toggle is on, so demo data is labelled via names/suffixes/headers and the Model status panel text (“DEMO MODE: …”).
* **Single process, in-memory state** (last 10 analyses, latest prediction). Restarting loses them; wards/ground truth/settlements persist. No auth, no database.
* **Memory:** rasters above `MAX_IMAGE_PIXELS` (default 25 MP) are rejected rather than streamed window-by-window. Clip or downsample large scenes first.
* **GeoJSON uploads must be EPSG:4326** (reproject Open Cities labels first if they are not). Counts use each building’s representative point; coverage uses clipped areas.
* **Resolution/domain shift:** a model trained at one ground resolution may degrade on imagery with a very different GSD.
* **Settlement comparison** needs your own labelled polygons; the dataset’s city names are deliberately not used as a proxy.
* **Not yet executed end-to-end:** this package was written and checked in an offline sandbox (syntax, imports, pure-logic tests). Run `pytest` on your machine first and report any failure.

## Integrating into the GitHub repo (`backend-dev` branch)

```powershell
git clone https://github.com/Insight-AS/BuildResilience-AI.git
cd BuildResilience-AI
git checkout backend-dev
# unzip so that the folder ends up at BuildResilience-AI\backend\
Expand-Archive -Path "$HOME\Downloads\buildresilience-backend.zip" -DestinationPath . -Force
git status                       # only backend\ should be new/changed
git add backend
git commit -m "feat(backend): FastAPI backend with demo mode and real-model hooks"
git push origin backend-dev
```
Do not merge into `main`. `.gitignore` keeps weights, `.env`, uploads and outputs out of git. If the zip already contains a top-level `backend` folder, extraction into the repo root puts it in the right place.
