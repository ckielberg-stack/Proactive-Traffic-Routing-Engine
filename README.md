# PTRE — Proactive Traffic Routing Engine

**Open-source prototype for automated incident verification and predictive VMS recommendations.**

PTRE is an experimental traffic-management research project for Swedish road data. It monitors live cameras along the E4/E20 corridor (Hallunda → Stockholm/Karlbergskanalen, ~15 km northbound), runs YOLO vehicle detection on each camera every 60 seconds, feeds the resulting capacity estimates into a kinematic-wave queue model, and produces preemptive Variable Message Sign (VMS) recommendations *before* congestion reaches each gantry.

This repository is intentionally published as a **prototype**, not as a finished control-room product. It is meant to be read, forked, tested, challenged, and developed further by people interested in traffic operations, computer vision, DATEX II, prediction models, operator tooling, and public-infrastructure software.

## Prototype Status

PTRE is useful as a working reference implementation, but it is not production certified.

- It depends on live Trafikverket API access and corridor-specific configuration.
- The VMS "ground truth" uses Situation API records as a proxy because live VMS panel state is not exposed publicly.
- The queue model, confidence scoring, camera ROIs, and operator UX need field validation before any operational use.
- Local runtime data, captured frames, anomaly images, and training images are intentionally excluded from the public repo.
- You should treat the current system as a foundation for experiments, not as safety-critical traffic-control software.

## Handoff: What This Prototype Is Solving

The main handoff is this: PTRE is trying to turn sparse public traffic data into an earlier, explainable VMS decision. Human operators can see that a crash or slowdown exists; this prototype focuses on the part that is hard to do manually in real time: estimating how fast the queue tail will propagate upstream and which VMS gantry should warn drivers before the queue arrives.

The current work has been aimed at four connected problems:

| Problem | Current approach | Where to continue |
|---|---|---|
| **Detect capacity loss from cameras** | YOLO vehicle detection, ROI polygons, density estimation, anomaly flags | Improve ROI calibration, add Swedish traffic-camera training data, validate false positives in night/winter/glare conditions |
| **Turn observations into physics** | LWR shockwave calculation in [src/physics_engine.py](src/physics_engine.py), using upstream inflow, bottleneck capacity, jam density, and per-segment queue-tail propagation | Field-validate the model, compare against replay data, tune density/capacity assumptions, and make multi-direction corridor support explicit |
| **Recommend VMS action before impact** | [src/vms_orchestrator.py](src/vms_orchestrator.py) maps predicted queue-tail ETAs to configured gantries in [vms_config.json](vms_config.json) | Validate gantry targeting rules with operators and connect to a real VMS status feed if available |
| **Measure whether PTRE was early and right** | Situation API `SPEEDMANAGEMENTID` records are used as a proxy for human action; [src/evaluation_logger.py](src/evaluation_logger.py) records camera-to-camera "Prophecy" predictions | Replace proxy ground truth with authoritative activation logs, add replay fixtures, and publish repeatable evaluation metrics |

For the deepest next-contributor context, start with [docs/handoff.md](docs/handoff.md). It explains the tick cycle, the piecewise LWR shockwave algorithm, the VMS recommendation flow, and known limitations in more detail than this README.

## What To Build Next

Good directions for contributors:

- Port the corridor configuration to other Swedish roads or other traffic-data providers.
- Improve camera ROI calibration and add a repeatable calibration workflow.
- Replace heuristic confidence scoring with evaluated, versioned model metrics.
- Add synthetic and replay-based test fixtures so contributors can run more of the system without live API access.
- Improve the DATEX II export path and validate it against downstream consumers.
- Separate prototype dashboard concerns from stable API contracts.
- Add sample, non-sensitive demo data so the UI can be explored without publishing captured camera frames.

## Overview

| | |
|---|---|
| **Corridor** | E4/E20 northbound, Hallunda → Karlbergskanalen |
| **Inputs** | 46 traffic cameras · 30 TrafficFlow sensor stations · 21 TravelTimeRoute segments · Situation API (incidents + VMS proxy) |
| **Outputs** | 8 VMS gantries with ETA recommendations · DATEX II XML export · operator dashboard · `/api/v1/*` endpoints |
| **Cadence** | 60-second tick — concurrent fetch → YOLO → physics → recommendations |
| **Stack** | Python 3.12 · FastAPI · Ultralytics YOLOv8 · OpenCV · Shapely · Pydantic · pytest |
| **Status** | Open-source prototype · research/demo use · needs validation before operational use |

## Quick Start

To try PTRE as it is, you need your own Trafikverket API key. Create one through Trafikverket's developer/API portal, then place it in a local `.env` file as shown below. The repository does not include credentials or live runtime data.

```bash
# 1. Virtual environment + dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. API key (create a .env file in the repo root)
cp .env.example .env
# Then edit .env and set TRAFIKVERKET_API_KEY.

# 3. Smoke test — run a single tick and exit
python main.py --once

# 4. Run continuously (default: API on :8081 + 60s tick loop)
python main.py

# 5. Verify
curl http://localhost:8081/health
open http://localhost:8081/        # TMC dashboard
```

`main.py` is the unified entry point — it runs the tick loop in a background thread (`asyncio.to_thread`) while serving the FastAPI app, dashboard pages, and operator API on the same port. Override with `--port 8080` to match the Docker Compose `dashboard` service.

Validate ROI physical-length calibration before trusting density-driven predictions:

```bash
python -m src.roi_length_calibration validate --config camera_config.json
```

Replay persisted ticks and write versioned prediction metrics without live API access:

```bash
python -m src.replay_evaluator tests/fixtures/replay_sample.jsonl --output docs/baselines/trafik-005-sample-metrics.json --corridor-length-km 10.0
```

The live Trafikverket API key belongs in `.env`. Do not commit local `.env`, `data/`, `storage/`, or captured images.

YOLO model weights are also kept out of Git. By default the vision engine uses `yolov8n.pt`; Ultralytics will download/cache the weight file locally on first use if it is not already present.

## Architecture

```
                              ┌─────────────────────────────────────────────────────┐
                              │                    60-SECOND TICK                    │
                              │                                                       │
  Camera API ─────────────────┼──► YOLOv8 ──► ROI Mapper ──► Density Smoother ──┐    │
  TrafficFlow Sensor API ─────┼──────────────────────────────────────────────┐   │   │
  TravelTimeRoute API ────────┼──► Travel Time Calibrator ──────────────┐    │   │   │
  Situation API (VMS proxy) ──┼──────────────────────────────────────┐  │    │   │   │
                              │                                       ▼  ▼    ▼   ▼   │
                              │              Physics Engine (LWR shockwave)         │
                              │                            │                          │
                              │                            ▼                          │
                              │          VMS Orchestrator · Incident Builder         │
                              │                            │                          │
                              └────────────────────────────┼──────────────────────────┘
                                                           │
                                       ┌───────────────────┼───────────────────┐
                                       ▼                   ▼                   ▼
                                  JSONL store        Operator API         Web Dashboard
                                (sensor_data,      (/api/v1/operator/*,  (TMC, sensors,
                                 anomalies,         /api/v1/export/      cameras, map,
                                 evaluation)        datex2)              anomalies, ...)
```

Each tick fetches concurrently, runs stateless YOLO inference, smooths observed density, calibrates the LWR free-flow speed from live travel-time observations, propagates queue tails to upstream VMS gantries, and atomically swaps the resulting snapshot into the API state for serving.

## Web Dashboard

Once `python main.py` is running, the dashboard is at **http://localhost:8081/** (or `:8080` under Docker Compose). All pages share a Jinja2 layout in [templates/base.html](templates/base.html).

| Path | Page | Purpose |
|---|---|---|
| `/` | TMC overview | Active incidents, VMS recommendations, Camera-to-Camera Prophecy hit rate |
| `/cameras` | Camera grid | Live thumbnails with YOLO overlay, ROI polygons, per-camera capacity drop |
| `/sensors` | Sensor data | TrafficFlow station readings (volume vph, speed km/h), mapped to nearest route-linear camera node |
| `/travel-times` | Travel times | Per-route delay vs free-flow, corridor-level congestion status |
| `/map` | Corridor map | Interactive Leaflet map of cameras, sensors, and VMS gantries |
| `/anomalies` | Anomaly log | Persisted anomaly events with annotated frames from local `storage/anomalies/` |
| `/system` | System health | Last-tick timestamp, pipeline stats, calibration confidence |
| `/logs` | Live logs | Tail of `data/mainloop.log` with level filtering |

Operator camera exclusions (used to silence noisy or out-of-corridor cameras) are persisted to `data/excluded_cameras.json` and toggled via `DELETE /api/cameras/{id}` and `POST /api/cameras/{id}/restore` from the camera page.

## Operator API

Versioned, stable surface for control-room frontends and NTS integration. Defined in [src/operator_api.py](src/operator_api.py).

| Endpoint | Description |
|---|---|
| `GET /api/v1/operator/active-incidents` | AI-verified incidents with base64 JPEG thumbnails and YOLO bounding boxes |
| `GET /api/v1/operator/vms-recommendations` | VMS recommendations + `proxy_ground_truth_active` flag (was a human already there?) |
| `GET /api/v1/export/datex2` | DATEX II v3 XML — `SituationPublication` for NTS ingestion |
| `GET /api/v1/evaluation/stats` | Camera-to-Camera Prophecy hit rate (predicted vs subsequent observation) |
| `GET /api/v1/evaluation/log` | Recent prophecy events for the dashboard feed |
| `GET /health` | Service health + last-tick timestamp + pipeline counts |

Example:

```bash
curl -s http://localhost:8081/api/v1/operator/active-incidents | jq '.count, .incidents[0].camera_id'
curl -s http://localhost:8081/api/v1/export/datex2 > datex2.xml
```

Optional auth:

- Leave `PTRE_API_TOKEN` unset for unchanged local research behavior.
- Set `PTRE_API_TOKEN` in deployed/shared environments to require a bearer token for API and dashboard routes. `/health` remains public for container health checks.
- API clients can send `Authorization: Bearer $PTRE_API_TOKEN` or `X-PTRE-API-Token: $PTRE_API_TOKEN`.
- Browser dashboard sessions can open `/?token=$PTRE_API_TOKEN` once; the app stores an HTTP-only session cookie for subsequent dashboard/API requests.

## Internal Data API

Helpers serving the dashboard pages. Defined in [main.py](main.py). Treat as internal — subject to change.

| Endpoint | Purpose |
|---|---|
| `GET /api/v1/cameras` | Per-camera capacity, vehicle count, anomaly flag from latest tick |
| `GET /api/v1/sensors` | Latest sensor readings + camera mapping |
| `GET /api/v1/travel-times` | Per-route TT + corridor summary (NB/SB split, status, total delay) |
| `GET /api/v1/calibration/status` | Free-flow speed adaptation from TravelTimeRoute |
| `GET /api/v1/anomalies?limit=N&camera_id=…` | Anomaly event log |
| `GET /api/v1/anomaly-image/{date}/{filename}` | Serve a saved annotated anomaly frame |
| `GET /api/v1/camera-image/{camera_id}` | Proxy a live Trafikverket camera JPEG |
| `GET /api/v1/camera-detections/{camera_id}` | Run on-demand YOLO on a camera, return boxes + ROI classification |
| `GET /api/v1/camera-config` | ROI polygon and exclusion-zone config |
| `GET /api/v1/logs?lines=N` | Tail `data/mainloop.log` with level parsing |
| `GET /api/v1/status` | Pipeline status snapshot from `data/status.json` |

## Core Components

**Entry points (repo root):**

| File | Purpose |
|---|---|
| [main.py](main.py) | **Canonical entry point** — FastAPI + 60s tick loop + dashboard pages, single process |
| [main_loop.py](main_loop.py) | Headless tick orchestrator (imported by `main.py`; can also run standalone) |
| [collect.py](collect.py) | Legacy standalone data collector (run by the `collector` Docker container) |
| [dashboard.py](dashboard.py) | Legacy dashboard server (superseded by `main.py`) |
| [config.py](config.py) | Centralized config — API URLs, camera IDs, sensor IDs, route IDs, thresholds |

**Pipeline modules ([src/](src/)):**

| File | Purpose |
|---|---|
| [src/vision_engine.py](src/vision_engine.py) | YOLOv8 perception → vehicle detections + capacity estimation |
| [src/roi_mapper.py](src/roi_mapper.py) | Pixel → road-segment classification with BEV homography support |
| [src/density_smoother.py](src/density_smoother.py) | Smooths observed traffic density across ticks to suppress flicker |
| [src/physics_engine.py](src/physics_engine.py) | LWR shockwave algorithm — bottleneck detection, piecewise queue-tail propagation, T+1/3/5/10 min lengths |
| [src/travel_time_calibrator.py](src/travel_time_calibrator.py) | Adapts free-flow speed from TravelTimeRoute API observations |
| [src/vms_orchestrator.py](src/vms_orchestrator.py) | Maps queue tail trajectory → gantry ETAs → VMS recommendations |
| [src/incident_builder.py](src/incident_builder.py) | Converts capacity states + YOLO frames into `IncidentReport` objects |
| [src/anomaly_store.py](src/anomaly_store.py) | Persistent anomaly event log (JSONL) with annotated frames |
| [src/evaluation_logger.py](src/evaluation_logger.py) | Records predictions and evaluates them against subsequent ticks (Prophecy) |
| [src/replay_evaluator.py](src/replay_evaluator.py) | Offline JSONL replay metrics for prediction precision, recall, ETA error, distance error, and VMS lead time |
| [src/operator_api.py](src/operator_api.py) | FastAPI app — `/api/v1/operator/*`, DATEX II export, atomic state injection |
| [src/models.py](src/models.py) | Pydantic domain models — `IncidentReport`, `QueuePrediction`, `VMSRecommendation`, etc. |

## Configuration

| Source | Used for |
|---|---|
| `.env` (env var `TRAFIKVERKET_API_KEY`) | **Required** — Trafikverket Datex API key |
| `.env` (optional `DATA_DIR`) | Override the default `./data` output directory |
| local YOLO `.pt` weights | Downloaded/cached locally by Ultralytics; not committed to Git |
| [config.py](config.py) | Camera IDs (46), sensor SiteIds (30), TravelTimeRoute IDs (21), bounding box, retry/backoff, anomaly thresholds |
| [camera_config.json](camera_config.json) | Per-camera ROI polygons, exclusion zones, homography matrices |
| [vms_config.json](vms_config.json) | 8 VMS gantries with `vms_id`, `lat`/`lng`, `chainage_km`, direction |
| `data/excluded_cameras.json` | Runtime camera exclusions toggled from the dashboard |

## Data Storage

```
data/
├── status.json               # Latest pipeline status (last tick, counts)
├── vision_state.json         # Latest capacity states (fallback for dashboard)
├── camera_info_cache.json    # Cached Trafikverket camera metadata (5-min TTL)
├── excluded_cameras.json     # Operator-toggled exclusions
├── mainloop.log              # Rotating tick log
└── 2026-05-16/
    ├── sensor_data.jsonl     # All tick data (vision, sensors, VMS, predictions)
    └── images/               # Captured camera frames (if enabled)

storage/
├── anomalies/                # Annotated anomaly JPEGs, grouped by date
└── training/                 # Reserved for offline training data
```

Most processing happens in memory — only metadata, predictions, and anomaly frames persist.

`data/` and `storage/` are local runtime directories and are ignored by Git. They may contain captured traffic-camera frames, logs, generated annotations, and other environment-specific artifacts. The public repository should not depend on those files being present.

## VMS Ground-Truth Strategy

The public Trafikverket API does **not** expose live VMS panel state. PTRE polls `Situation.Deviation` records with the `SPEEDMANAGEMENTID` prefix as a proxy for human-operator action timestamps. Each tick logs these with `source: "situation_api_proxy"`, building a historical comparison dataset:

```
AI predicted VMS needed at T₁  →  Human operator acted at T₂
Δ = T₂ − T₁  (PTRE's speed advantage over the human-in-the-loop)
```

The `proxy_ground_truth_active` flag on every `/api/v1/operator/vms-recommendations` entry tells the control room whether a human has already activated a speed advisory on the same road segment.

## Testing

```bash
pytest tests/ -v --ignore=tests/smoke_test.py
```

Run `pytest -v` for the current count. The 10 unit test modules under [tests/](tests/) cover:

- [test_physics_engine.py](tests/test_physics_engine.py) — LWR kinematic wave model
- [test_vision_engine.py](tests/test_vision_engine.py) — YOLO + capacity estimation
- [test_roi_mapper.py](tests/test_roi_mapper.py) — Pixel → segment classification, BEV homography
- [test_density_smoother.py](tests/test_density_smoother.py) — Cross-tick density smoothing
- [test_travel_time_calibrator.py](tests/test_travel_time_calibrator.py) — Free-flow speed adaptation
- [test_vms_orchestrator.py](tests/test_vms_orchestrator.py) — Recommendation generation
- [test_incident_builder.py](tests/test_incident_builder.py) — Capacity → IncidentReport
- [test_evaluation_logger.py](tests/test_evaluation_logger.py) — Camera-to-Camera Prophecy
- [test_replay_evaluator.py](tests/test_replay_evaluator.py) — Offline replay metrics and artifact generation
- [test_sensor_anomaly.py](tests/test_sensor_anomaly.py) — Sensor anomaly detection
- [test_operator_api.py](tests/test_operator_api.py) — FastAPI endpoints + DATEX II

[tests/smoke_test.py](tests/smoke_test.py) is an integration test that hits the live Trafikverket API and is excluded from default runs.

## Deployment

```bash
docker compose up -d
```

[docker-compose.yml](docker-compose.yml) brings up two services:

- **`collector`** — runs [collect.py](collect.py) (legacy collection of weather/road/camera data into the shared `./data` volume).
- **`dashboard`** — runs `python main.py --host 0.0.0.0 --port 8080`, serving the unified API + dashboard at **http://localhost:8080**. Mounts `static/`, `templates/`, and `camera_config.json`.

Both containers share `./data` via volume mount. The collector's health check fails if `data/collector.log` is older than 3 minutes.

## Contributing

This project is open for further development. See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines. Practical contributions are especially welcome when they make the prototype easier to validate, replay, port, or operate safely:

- small, testable changes over large rewrites
- fixtures or replay data that do not expose private keys or captured operational material
- clear notes about assumptions, failure modes, and confidence limits
- docs that help others reproduce a result with their own Trafikverket API key

Before opening a pull request, run the unit tests and keep generated runtime files out of Git.

## Documentation

- [docs/roadmap.md](docs/roadmap.md) — phases, ADRs, and remaining work
- [docs/project_state.md](docs/project_state.md) — current status snapshot
- [docs/handoff.md](docs/handoff.md) — context for the next contributor
- [docs/notes/](docs/notes/) · [docs/plans/](docs/plans/) — working notes and plan drafts

## License

This project is licensed under the GNU Affero General Public License v3.0. See [LICENSE](LICENSE).

PTRE uses [Ultralytics YOLO](https://github.com/ultralytics/ultralytics), which is distributed under AGPL-3.0 unless you have a separate Enterprise license from Ultralytics.
