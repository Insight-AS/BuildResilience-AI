# Example API responses

> Illustrative, hand-written to match the code's response shapes. They were **not** captured from a live run.
> Compare against your own output at http://127.0.0.1:8000/docs.

## GET /api/health (demo mode, no checkpoint)
```json
{"status":"ok","version":"1.0.0","mode":"demo","model_loaded":false,"checkpoint_found":false,
 "device":"auto","demo_mode":true,"configured_demo_mode":"auto","architecture":"Unet","encoder":"resnet34",
 "model_load_error":"Checkpoint not found at MODEL_PATH.","time":"2026-11-07T10:00:00+00:00"}
```

## GET /api/summary (what Overview.jsx prints verbatim)
```json
{"total_buildings":"593 (demo)","total_footprint_area":"79,482 m² (demo)","avg_density":"412.3 (demo)",
 "wards_analyzed":"6 (demo)","model":"DEMO MODE: simulated results (model checkpoint not loaded). No trained model is producing these numbers.",
 "mode":"demo","demo":true,"details":{"...":"full numeric dashboard summary"}}
```

## GET /api/density
```json
[{"ward":"DEMO Ward A","count":23,"density":191.7,"area":2400.5,"coverage":2.0}, {"ward":"DEMO Ward B","count":78,"density":650.0,"area":8900.1,"coverage":7.4}]
```

## POST /api/predict (PNG, demo mode)
```json
{"analysis_id":"3f0c…","demo":true,"results_type":"simulated_demo",
 "status_message":"DEMO MODE: simulated data, not produced by a trained model",
 "georeferenced":false,"crs":null,"output_crs":null,"coordinate_system":"pixel",
 "building_count":41,"total_building_area_m2":null,
 "statistics":{"building_count":41,"total_area_m2":null,"total_area_px":18890,"image_area_km2":null,
               "building_density_per_km2":null,"building_coverage_pct":null},
 "processing_time_s":0.21,
 "image":{"filename":"tile.png","format":"png","width":512,"height":512,"valid_pixel_fraction":1.0},
 "warnings":["DEMO MODE: … These polygons are NOT a prediction of the imagery content.",
             "PNG/JPEG carries no georeferencing: …","Area (m²) and density are unavailable because the input has no georeferencing."],
 "limitations":["Each polygon is a connected region of the predicted mask. Touching or overlapping buildings are merged …"],
 "mask":{"data_uri":"data:image/png;base64,…","width":512,"height":512,"downscaled":false},
 "geojson":{"type":"FeatureCollection","demo":true,"georeferenced":false,"features":[
   {"type":"Feature","properties":{"id":"B001","building_id":"B001","area_m2":null,"area":null,"area_px":212,
    "confidence":null,"source":"demo_simulated","demo":true},"geometry":{"type":"Polygon","coordinates":[["…pixel x,y…"]]}}]}}
```

## POST /api/predict (GeoTIFF, real model)
Same keys, with `"demo":false`, `"results_type":"model_prediction"`, `"crs":"EPSG:32737"`, `"output_crs":"EPSG:4326"`,
`area_m2` e.g. `120.5`, `confidence` e.g. `0.91`, `"source":"model_prediction"`, and statistics including
`building_density_per_km2` and `building_coverage_pct`.

## POST /api/predict with DEMO_MODE=false and no checkpoint → 503
```json
{"error":{"code":"model_unavailable","message":"No trained model checkpoint is loaded and DEMO_MODE=false. Place the checkpoint at MODEL_PATH (see backend/models/README.md) or set DEMO_MODE=auto."}}
```

## GET /api/analytics/settlement-comparison (no labels)
```json
{"status":"unavailable","message":"Planned vs informal comparison requires settlement-type annotations, ground truth and a real prediction. Missing: a real (non-demo) georeferenced prediction; ground-truth building footprints; settlement-type polygons labelled 'planned' / 'informal'.","missing":["…"],"categories":{}}
```

## POST /api/analytics/evaluate (two 32×32 masks, 50 % overlap)
```json
{"iou":0.3333,"dice":0.5,"precision":0.5,"recall":0.5,"building_count_error":0,
 "predicted_building_count":1,"ground_truth_building_count":1,
 "confusion":{"tp":50,"fp":50,"fn":50,"tn":874},"level":"pixel","model":"cnn","image_width":32,"image_height":32,"notes":["…"]}
```
