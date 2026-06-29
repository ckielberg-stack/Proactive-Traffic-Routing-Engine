# PROACTIVE TRAFFIC ROUTING ENGINE (PTRE) — ROADMAP

## OBJECTIVE

Build a **proactive** traffic routing engine that uses computer vision on live traffic cameras and mathematical fluid dynamics (Kinematic Wave Theory) on sensor data to predict where a traffic queue *will be* in the near future — before vehicles are stuck.

---

## ROADMAP STATUS

| Milestone | Description | Status |
|---|---|---|
| **1 — Data** | Live data ingestion pipeline (46 configured cameras, weather, road conditions, incidents) | ✅ Complete |
| **2 — Perception** | Vision Engine: YOLO-based capacity estimation + anomaly detection | ✅ Complete |
| **2a — Spatial** | ROI Mapper: per-camera pixel → road segment classification | ✅ Complete |
| **2b — Retention** | Smart image retention policy (anomaly + training samples only) | ✅ Complete |
| **2c — Tooling** | Interactive ROI calibration helper + camera discovery script | ✅ Complete |
| **2d — Deployment** | Single-service Docker Compose stack (`main.py`) | ✅ Complete |
| **3 — Physics** | Shockwave Prediction Engine (LWR model) | ✅ Complete |
| **5 — VMS** | VMS & Queue Tail Predictor for preemptive sign activation | ✅ Complete |
| **6 — Operator** | Operator Decision Support API (incidents, VMS, DATEX II) | ✅ Complete |

> **Note:** Phases 4–5 from the original B2C roadmap (Geospatial Routing Integrator, routing penalty API) have been **removed** per ADR-004.

> **Future work:** see [docs/proposals/traffic-safety-improvements.md](proposals/traffic-safety-improvements.md) — eight prioritized safety proposals (weather-adjusted physics, Situation-API capacity inputs, uncertainty bands, stopped-vehicle detection, SMHI forecasts, DATEX II weather export).

---

## COMPLETED WORK

### Milestone 1: Data Ingestion Pipeline ✅

Live pipeline collecting from **46 configured Trafikverket cameras** on E4 Hallunda → Stockholm every 60 seconds:

- **Camera images** — fetched from Trafikverket Camera API (full-size 1280×720)
- **Weather data** — temperature, wind, visibility, precipitation from WeatherMeasurepoint API
- **Road conditions** — surface state, friction data from RoadCondition API
- **Traffic situations** — accidents, roadwork, closures from Situation API (ground truth)
- **Dashboard** — real-time monitoring at `localhost:8080` with interactive Leaflet map, camera validation (remove/restore), weather table, road conditions, incident log
- **Camera exclusion** — cameras can be removed from collection via `excluded_cameras.json`; `main.py` re-reads exclusions each tick (no restart needed)
- **Camera discovery** — `discover_cameras.py` auto-discovers cameras in the bounding box from the Trafikverket API, outputs to `discovered_cameras.json`
- **Graceful shutdown** — SIGINT/SIGTERM handled for clean runtime termination
- **File logging** — structured tick-loop log output to `data/mainloop.log` with rotation

### Milestone 2: Vision & Capacity Engine ✅

`src/vision_engine.py` — YOLOv8n-based perception module:

- **Vehicle detection** with ROI polygon filtering and COCO class filtering (car, motorcycle, bus, truck)
- **Capacity estimation** using simplified Greenshields model: `density × speed`, capped at theoretical max
- **Anomaly detection** — abnormal aspect ratios (sideways vehicles), zero detections + high inflow, speed drop + low detections
- **Sensor fusion fallback** — black image + speed drop >50% → `capacity = 0`, `is_anomaly = True`
- **Blocked lane estimation** from abnormally wide bounding boxes
- **Output:** `CapacityState(timestamp, vehicle_count, blocked_lanes, total_lanes, estimated_capacity_vph, is_anomaly, anomaly_reason, confidence)`

### Milestone 2a: ROI Spatial Awareness ✅

`src/roi_mapper.py` — pixel-to-road-segment classification:

- **Per-camera ROI polygons** defined in `camera_config.json` (road_id, direction, capacity, lanes, pixel polygon)
- **Shapely point-in-polygon** classification on bottom-center detection point (tire contact: `x = (x1+x2)/2`, `y = y2`)
- **Batch classification** — `classify_detections_batch()` groups detections by road segment
- **Multi-ROI analysis** — `VisionEngine.analyze_multi_roi()` returns `MultiSegmentCapacity` with per-segment counts
- **Graceful degradation** — cameras without ROI config fall back to full-frame single-mode
- **Nighttime fallback stub** — headlight/taillight classification concept documented for Swedish winter darkness

### Milestone 2b: Smart Retention Policy ✅

`retention.py` — minimises disk I/O from ~860 MB/day to ~5 MB/day:

- **Anomaly retention** → `storage/anomalies/{date}/{cam}_{time}.jpg` (human debugging)
- **Training sample retention** → `storage/training/{date}/{cam}_{time}.jpg` (1 frame/camera every 4 hours, randomised start offsets)
- **Schedule persistence** — training schedule saved to `data/training_schedule.json`, survives restarts

### Milestone 2c: Tooling ✅

- **Interactive ROI helper** — `roi_helper.py` (389 lines): OpenCV interactive polygon drawing tool for camera ROI calibration. Fetches live camera images, lets the user click to define polygon ROIs, prompts for road metadata, and saves directly to `camera_config.json`
- **Camera discovery** — `discover_cameras.py` queries Trafikverket API for all active cameras within the bounding box

### Milestone 2d: Docker Deployment ✅

- **Dockerfile** — Python 3.12-slim, health check against `/health`
- **docker-compose.yml** — single `trafik` service running `main.py` on port 8080
- **Runtime data volume** — `./data` mounted into the service
- **Log rotation** — JSON file driver with max-size limits

### Dashboard ✅

Full-featured monitoring dashboard (`main.py` + `templates/` + `static/`):

- **Interactive Leaflet map** — all 46 configured cameras plotted with clickable markers and photo popups
- **Camera grid** — latest images from each camera with live updates
- **Weather table** — air/road temperature, wind, visibility, humidity, precipitation
- **Road conditions table** — surface state, warnings, friction data
- **Incidents table** — traffic situations (accidents, roadwork, closures) with severity
- **Live logs** — scrollable log viewer with error highlighting
- **Camera management** — exclude/restore cameras dynamically via API
- **Status bar** — cycle count, last update, total images, disk usage, interval, active incidents

### Test Suite ✅

Current default pytest collection: **330 tests**, with **1 live smoke test
deselected** by the pytest configuration.

Verified on 2026-06-29 with:

```bash
.venv/bin/python -m pytest --collect-only -q
```

Coverage spans vision, ROI mapping, physics, VMS orchestration, operator API,
deployment shape, retention, replay evaluation, shipped config integration,
route-linear mapping, weather/SMHI inputs, and the live Trafikverket smoke test.

### Pydantic Domain Models ✅

`src/models.py` — typed data contracts:

- `SensorReading` — upstream radar/loop-detector data
- `CameraMetadata` — camera → road network mapping
- `CapacityState` — single-frame vision engine output
- `RoadSegmentState` — per-segment multi-ROI output
- `MultiSegmentCapacity` — aggregated multi-ROI frame output
- `QueuePrediction` — physics engine output (wave speed, queue lengths at time intervals)
- `VMSGantry` — static VMS sign configuration (ID, coordinates, chainage)
- `VMSRecommendation` — VMS activation recommendation with urgency and message
- `IncidentReport` — AI-verified incident with thumbnail and capacity drop

---

## ARCHITECTURE DECISIONS

### ADR-001: In-Memory Image Processing (No Standard Disk Writes)

**Context:** 46 configured cameras × 1440 cycles/day = ~66,000 HD images/day. Saving all to disk causes I/O bottlenecks and storage bloat.

**Decision:** Images are processed **entirely in RAM**:
```
API → fetch_image_bytes() → cv2.imdecode() → np.ndarray
    → VisionEngine.analyze_array() → CapacityState (metadata)
    → RetentionPolicy.maybe_retain() → discard or save
    → del frame (GC reclaims ~2 MB)
```

**Only metadata** (vehicle counts, capacity VPH, anomaly flags) is persisted to `sensor_data.jsonl` and `vision_state.json`.

### ADR-002: Smart Retention Policy (`retention.py`)

Physical `.jpg` files are saved **only** under two conditions:

1. **Anomaly detected** → `storage/anomalies/{date}/{cam}_{time}.jpg`  
   Required for human debugging of false positives.

2. **Training sample** → `storage/training/{date}/{cam}_{time}.jpg`  
   Exactly 1 frame per camera every 4 hours (randomized start offsets).  
   Builds a long-term dataset for model fine-tuning (snow, night, sun glare).

**Storage impact:** ~860 MB/day → ~5 MB/day.

### ADR-003: ROI Spatial Awareness (Pixel → Road Segment Mapping)

**Context:** YOLO evaluates static 2D images — it cannot infer geographic context (E4 vs off-ramp) or travel direction from low-framerate stills. We need to map 2D pixel coordinates to 3D physical road segments using static polygons.

**Decision:** Per-camera Regions of Interest (ROIs) defined in `camera_config.json`:
- Each ROI has a `road_id`, `direction_relative_to_camera`, `capacity_vph`, `num_lanes`, and a pixel `polygon`
- `ROIMapper` (Shapely-based) classifies detections via point-in-polygon tests on the **bottom-center** point (tire contact: `x = (x1+x2)/2`, `y = y2`)
- Detections outside all ROIs are **discarded** (filters background traffic, parked cars, false positives)
- `VisionEngine.analyze_multi_roi()` returns `MultiSegmentCapacity` with per-segment counts
- Cameras without ROI config fall back to full-frame single-mode (backward compatible)

**Future:** Nighttime headlight/taillight classification stub added for Swedish winter darkness fallback.

### ADR-004: B2G Strategic Pivot (Consumer Routing → Traffic Management)

**Context:** The target customer changed from B2C consumer navigation to B2G — Swedish Transport Administration (Trafikverket) and their Traffic Management Centers (Trafik Stockholm). We are building an **Automated Incident Verification and Predictive VMS Copilot** for human traffic control operators.

**Decision:**
- **Remove** all routing/graph logic (OSRM, Valhalla, `osmnx`, dynamic edge weights). These were never implemented — only planned in the roadmap.
- **Keep** Phases 1–3 (Data, Vision, Physics) unchanged — they produce the same `CapacityState` and `QueuePrediction` outputs regardless of downstream consumer.
- **New Phase 5** — VMS & Queue Tail Predictor: uses chainage-based linear referencing along the highway corridor to predict when queue tails will reach physical VMS gantry positions and triggers preemptive speed reductions.
- **New Phase 6** — Operator Decision Support API: FastAPI endpoints delivering verified incident telemetry, VMS recommendations, and DATEX II XML export for integration with the National Traffic Management System (NTS).

**Rationale:** The B2G customer doesn't route vehicles — they manage infrastructure (VMS signs, incident response, traffic flow). The same perception and physics engines are reused, but the output layer serves operator decision support instead of navigation.

---

## REMAINING PHASES

### Phase 3: Shockwave Prediction Engine (`physics_engine.py`) ✅
**Goal:** Calculate backward propagation speed of traffic queues.
1. Implement LWR kinematic wave formula:  
   `Wave_Speed = (Inflow_Volume − Bottleneck_Capacity) / (Jam_Density − Inflow_Density)`
2. Assume `Jam_Density` ≈ 133 vehicles/km/lane
3. `predict_queue_length(wave_speed)` → queue length at T+1, T+3, T+5, T+10 min
4. Output: `QueuePrediction(growth_speed_kmh, lengths_at_intervals: dict, origin_lat, origin_lng)`

### Phase 5: VMS & Queue Tail Predictor (`vms_orchestrator.py`) ✅
**Goal:** Predict when the tail of the traffic jam will pass upstream physical VMS gantries.
1. Load VMS gantry coordinates from `vms_config.json` (chainage-based linear referencing)
2. Using `QueuePrediction` output, project the queue tail position at T+1, T+3, T+5 min
3. Find nearest VMS gantry ≥1000m upstream of predicted queue tail
4. Generate activation recommendation: "KÖVARNING 70 km/h" with urgency and ETA
5. Output: `list[VMSRecommendation]`

### Phase 6: Operator Decision Support API (`operator_api.py`)
**Goal:** Provide instant verified telemetry to human traffic control operators.
1. FastAPI endpoints for Control Room frontend:
   - `GET /api/v1/operator/active-incidents` — AI-verified incidents with YOLO thumbnails
   - `GET /api/v1/operator/vms-recommendations` — active VMS recommendations with queue growth data
   - `GET /api/v1/export/datex2` — DATEX II XML for NTS integration
2. Reduces operator cognitive load with pre-verified, structured incident data

---

## TECHNICAL STACK

| Layer | Technology |
|---|---|
| Language | Python 3.11+ |
| Vision | `ultralytics` (YOLOv8n), `opencv-python` |
| Math | `numpy`, `pandas`, `scipy` |
| Geospatial | `shapely` |
| Backend | `FastAPI`, `pydantic`, `uvicorn` |
| XML/DATEX II | `lxml` |
| Data API | Trafikverket Open API v2 (XML/JSON) |
| Deployment | Docker, Docker Compose |
| Testing | `pytest`, `httpx` |

## KEY FILES

| File | Purpose |
|---|---|
| `legacy/collect.py` | Historical collection loop reference; not part of default deployment |
| `main_loop.py` | **Tick-based orchestrator** — 60s polling, concurrent fetch, physics → VMS pipeline |
| `config.py` | Camera IDs, coordinates, API config (46 cameras) |
| `retention.py` | Smart image retention policy |
| `legacy/dashboard.py` | Historical dashboard server reference; superseded by `main.py` |
| `src/vision_engine.py` | YOLO perception + capacity estimation (545 lines) |
| `src/physics_engine.py` | **LWR shockwave model** — queue propagation speed and tail projection |
| `src/models.py` | Pydantic domain models (incl. `TickResult`, `VMSStatusSnapshot`) |
| `src/roi_mapper.py` | Pixel → road segment ROI classification |
| `src/vms_orchestrator.py` | VMS & Queue Tail Predictor with narrative summaries (Phase 5) |
| `src/operator_api.py` | Operator Decision Support API (Phase 6) |
| `vms_config.json` | VMS gantry coordinates and metadata |
| `roi_helper.py` | Interactive OpenCV ROI calibration tool |
| `discover_cameras.py` | Camera discovery from Trafikverket API |
| `camera_config.json` | Per-camera ROI polygon definitions |
| `static/` | Dashboard frontend (HTML/CSS/JS + Leaflet) |
| `Dockerfile` | Container image (Python 3.12-slim) |
| `docker-compose.yml` | Single-service deployment running `main.py` |
| `tests/` | Unit tests (physics engine, vision engine, ROI mapper, VMS orchestrator, operator API) |

### ADR-005: Tick-Based Discrete Architecture

**Context:** The system receives static `.jpg` images every 60 seconds — there is no video stream, no temporal continuity between frames. The previous `collect.py` ran a sequential monolithic loop that wasted ~30s per cycle on serial API calls.

**Decision:** Replace the monolithic collection with a **discrete tick-based architecture** (`main_loop.py`):

- Each tick is **stateless**: evaluates the world from scratch, no cross-tick memory
- **Concurrent fetching** via `ThreadPoolExecutor`: cameras, sensors, and VMS statuses fetched in parallel
- Pipeline per tick: `fetch → vision (YOLO) → physics (LWR) → VMS orchestrator → persist`
- **VMS status polling** creates a ground-truth log of when human operators actually activated signs
- Physics engine computes **predictive queue tail propagation** — the system's core value over humans who cannot do shockwave math in real-time

**Value proposition:**
> The system does NOT race humans to detect crashes (humans have live video). The system predicts **when the queue tail will reach upstream VMS signs** and recommends preemptive activation.
