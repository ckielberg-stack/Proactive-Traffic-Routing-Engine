# Project State — PTRE (Proactive Traffic Routing Engine)

> Last updated: 2026-02-16

## System Identity

**Automated Incident Verification and Predictive VMS Copilot** for Trafikverket / Trafik Stockholm. A B2G traffic management tool that predicts queue tail propagation and recommends preemptive VMS sign activations — something human operators cannot do mathematically in real-time.

## Architecture

```
60s tick │ ThreadPoolExecutor
         ├── Camera API  → YOLO → CapacityState[]
         ├── Sensor API  → SensorReading[]
         └── Situation API (SPEEDMANAGEMENTID proxy) → VMSStatusSnapshot[]
                │
                ▼
         Physics Engine (LWR kinematic wave)
                │
                ▼
         QueuePrediction[] → VMS Orchestrator → VMSRecommendation[]
                │
                ▼
         ├── JSONL persistence (ground-truth log)
         ├── Operator API (FastAPI)
         └── DATEX II XML export
```

## Phase Completion

| Phase | Component | Status |
|---|---|---|
| 1 | Data Ingestion (`collect.py`, `config.py`) | ✅ Complete |
| 2 | Vision Engine (`vision_engine.py`, `roi_mapper.py`) | ✅ Complete |
| 3 | Physics Engine (`physics_engine.py`) | ✅ Complete |
| — | Tick Architecture (`main_loop.py`) | ✅ Complete |
| 5 | VMS Orchestrator (`vms_orchestrator.py`) | ✅ Complete |
| 6 | Operator API (`operator_api.py`) | ✅ Complete |

## Implemented Features

- **Tick-based architecture** — stateless 60s discrete polling, concurrent fetching
- **In-memory YOLO inference** — no disk I/O for images, smart retention only
- **Multi-ROI spatial awareness** — pixel→road segment mapping per camera
- **LWR shockwave model** — `w = (Q_in − Q_cap) / (k_jam − k_in)`
- **VMS queue tail prediction** — ETA to upstream gantry positions
- **Situation API proxy polling** — `SPEEDMANAGEMENTID` as human operator action timestamp
- **Ground-truth enrichment** — `proxy_ground_truth_active` on every VMS recommendation
- **DATEX II export** — `SituationPublication` + `SpeedManagement` records for NTS
- **Operator narrative summaries** — Swedish-language descriptions for control room

## Test Coverage

**92 tests passing** (0 failures):
- Physics Engine: 13
- Vision Engine: 16
- ROI Mapper: 16
- VMS Orchestrator: 18
- Operator API: 21

## Key Design Decisions (ADRs)

| ADR | Decision |
|---|---|
| 001 | In-memory image processing — only metadata persisted |
| 002 | Smart retention — anomalies + training samples only |
| 003 | ROI spatial awareness — pixel→road segment via Shapely |
| 004 | B2G pivot — consumer routing → traffic management |
| 005 | Tick-based discrete architecture — stateless 60s cycles |

## Known Limitations

1. **No live VMS panel state** — public API only exposes speed advisories, not physical sign hardware state
2. **VMS proxy is approximate** — `SPEEDMANAGEMENTID` deviations are roadwork-related speed limits, not real-time VMS activations
3. **YOLO model untrained** — using default YOLOv8n weights, no fine-tuning on Swedish traffic cameras yet
4. **Single corridor** — hardcoded to E4 Södertälje→Stockholm, needs generalization for other highways

## What's Next

- [ ] Wire `main_loop.py` tick output into `operator_api.py` state setters (live integration)
- [ ] Deploy on VPS with Docker Compose (`collector` + `dashboard` + `operator-api`)
- [ ] Fine-tune YOLO on Swedish camera images (night, winter, sun glare)
- [ ] Build control room dashboard frontend consuming the Operator API
- [ ] Formal DATEX II XSD validation against Trafikverket's NTS schemas
