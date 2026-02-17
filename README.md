# 🚦 PTRE — Proactive Traffic Routing Engine

**Automated Incident Verification and Predictive VMS Copilot** for Swedish traffic management operators (Trafikverket / Trafik Stockholm).

The system monitors traffic cameras along the E4 corridor (Södertälje → Stockholm), detects incidents via YOLO, and uses kinematic wave theory to predict queue tail propagation — recommending preemptive VMS sign activations before the queue reaches upstream gantries.

## Quick Start

```bash
# 1. Virtual environment + dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. API key
cp .env.example .env  # Add your TRAFIKVERKET_API_KEY

# 3. Run one tick (test)
python main_loop.py --once

# 4. Run continuous monitoring (60s ticks)
python main_loop.py

# 5. Start operator API
python -m src.operator_api  # → http://localhost:8081
```

## Architecture

```
Camera API ─┐
Sensor API ──┼── ThreadPool ── YOLO ── Physics (LWR) ── VMS Orchestrator ── JSONL
Situation API┘                                                            ── API
```

Each **60-second tick** concurrently fetches data, runs stateless YOLO inference, computes shockwave propagation, and generates operator recommendations — all in memory.

## API Endpoints

| Endpoint | Description |
|---|---|
| `GET /api/v1/operator/active-incidents` | AI-verified incidents with YOLO thumbnails |
| `GET /api/v1/operator/vms-recommendations` | VMS recommendations + `proxy_ground_truth_active` flag |
| `GET /api/v1/export/datex2` | DATEX II XML for NTS integration |
| `GET /health` | Service health with pipeline metadata |

## Key Components

| File | Purpose |
|---|---|
| `main_loop.py` | Tick-based orchestrator — 60s concurrent fetch → YOLO → physics → VMS |
| `src/physics_engine.py` | LWR kinematic wave model — queue propagation speed |
| `src/vms_orchestrator.py` | Queue tail → VMS gantry ETA prediction |
| `src/operator_api.py` | FastAPI operator endpoints + DATEX II export |
| `src/vision_engine.py` | YOLOv8 perception + capacity estimation |
| `src/roi_mapper.py` | Pixel → road segment classification |
| `src/models.py` | Pydantic domain models |
| `collect.py` | Legacy standalone data collector |

## Testing

```bash
# Run all tests
python -m pytest tests/ -v --ignore=tests/smoke_test.py

# 92 tests passing across 5 suites:
# Physics Engine (13) | Vision Engine (16) | ROI Mapper (16)
# VMS Orchestrator (18) | Operator API (21)
```

## VMS Ground-Truth Strategy

The public Trafikverket API does **not** expose live VMS panel state. We poll `Situation.Deviation` records with `SPEEDMANAGEMENTID` prefix as a proxy for human operator action timestamps. Each tick logs these with `source: "situation_api_proxy"` to build a historical comparison dataset:

```
AI predicted VMS needed at T₁  →  Human operator acted at T₂
Value = T₂ - T₁ (our speed advantage)
```

## Data Storage

```
data/
├── status.json              # Dashboard state
├── vision_state.json        # Latest capacity states
├── mainloop.log             # Rotating log
└── 2026-02-16/
    └── sensor_data.jsonl    # All tick data (vision, sensors, VMS, predictions)
```

Estimated storage: **~5 MB/day** (in-memory processing, only metadata persisted).

## Deployment

```bash
docker compose up -d  # collector + dashboard + operator-api
```

## Documentation

- [Roadmap & ADRs](docs/roadmap.md) — architecture decisions and remaining phases
- [Project State](docs/project_state.md) — current status and what's next
