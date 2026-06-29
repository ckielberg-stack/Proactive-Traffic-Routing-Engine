# Project State — PTRE (Proactive Traffic Routing Engine)

> Last updated: 2026-06-29

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
         Density Smoother (EMA α=0.4)       ← NEW: Expert Audit Fix 3
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
| 1 | Data Ingestion (`main_loop.py`, `config.py`) | ✅ Complete |
| 2 | Vision Engine (`vision_engine.py`, `roi_mapper.py`) | ✅ Complete |
| 3 | Physics Engine (`physics_engine.py`) | ✅ Complete |
| — | Tick Architecture (`main_loop.py`) | ✅ Complete |
| 5 | VMS Orchestrator (`vms_orchestrator.py`) | ✅ Complete |
| 6 | Operator API (`operator_api.py`) | ✅ Complete |
| 7 | Expert Audit Fixes (4 critical flaws) | ✅ Complete |

## Expert Audit Fixes (2026-02-18)

Four critical theoretical and geometric flaws were identified by an expert audit in macroscopic traffic flow and computer vision. All four have been resolved:

| Fix | Problem | Solution | Files Changed |
|---|---|---|---|
| **1. Flow vs Capacity** | `_estimate_capacity()` computed `q = k × v` (flow), not capacity. Falsely triggered bottlenecks on empty roads. | Vision Engine outputs density (`veh/km/lane`). Physics Engine triggers on `density > k_critical (45)` instead of `is_anomaly`. Static `Q_cap = 2000 vph/lane` for free-flow. | `vision_engine.py`, `physics_engine.py`, `models.py` |
| **2. BEV Homography** | 1D polynomial (`np.polyfit`) only mapped Y-pixel → meters, ignoring horizontal perspective distortion. | 4-point homography via `cv2.getPerspectiveTransform`. `roi_mapper.py` projects detections to BEV when H matrix available, pixel-space fallback for legacy cameras. | `roi_mapper.py`, `roi_helper.py`, `camera_config.json` |
| **3. Temporal Smoothing** | Stateless tick vulnerable to transient occlusions (bus blocking camera for 1 frame → false congestion). | EMA smoother (`α=0.4`) applied per-camera between vision output and physics engine. Only intentional break of ADR-005. | `density_smoother.py` (NEW), `main_loop.py` |
| **4. ROI Horizon Hardcap** | YOLOv8n detection degrades past ~150m, creating artificially low density. | Non-blocking warning when ROI depth > 150m during calibration (both legacy ruler and BEV modes). | `roi_helper.py` |

## Implemented Features

- **Tick-based architecture** — stateless 60s discrete polling, concurrent fetching
- **In-memory YOLO inference** — no disk I/O for images, smart retention only
- **Multi-ROI spatial awareness** — pixel→road segment mapping per camera
- **BEV homography projection** — 4-point calibration for accurate physical-plane mapping
- **Density-based congestion detection** — `observed_density_veh_km_lane` on every `CapacityState`
- **Temporal EMA smoothing** — dampens transient occlusion spikes before physics evaluation
- **LWR shockwave model** — `w = (Q_in − Q_cap) / (k_jam − k_in)`
- **VMS queue tail prediction** — ETA to upstream gantry positions
- **Situation API proxy polling** — `SPEEDMANAGEMENTID` as human operator action timestamp
- **Ground-truth enrichment** — `proxy_ground_truth_active` on every VMS recommendation
- **DATEX II export** — `SituationPublication` + `SpeedManagement` records for NTS
- **Operator narrative summaries** — Swedish-language descriptions for control room
- **ROI horizon validation** — warns when ROI extends past YOLOv8n effective range (~150m)

## Test Coverage

Current default pytest collection: **330 tests**, with **1 live smoke test
deselected** by the pytest configuration.

Verified on 2026-06-29 with:

```bash
.venv/bin/python -m pytest --collect-only -q
```

The suite covers the tick pipeline, shipped config integration, deployment
shape, FastAPI/operator surfaces, DATEX export, VMS orchestration, physics,
vision, ROI mapping, retention, replay evaluation, route-linear mapping,
situation/weather/SMHI inputs, and live Trafikverket smoke coverage.

## Key Design Decisions (ADRs)

| ADR | Decision |
|---|---|
| 001 | In-memory image processing — only metadata persisted |
| 002 | Smart retention — anomalies + training samples only |
| 003 | ROI spatial awareness — pixel→road segment via Shapely |
| 004 | B2G pivot — consumer routing → traffic management |
| 005 | Tick-based discrete architecture — stateless 60s cycles |
| 006 | Temporal density smoothing — EMA α=0.4, sole exception to ADR-005 |
| 007 | Density-based bottleneck detection — `k > k_critical` replaces `is_anomaly` trigger |
| 008 | BEV homography — 4-point calibration replaces 1D polyfit (backward-compatible) |

## Known Limitations

1. **No live VMS panel state** — public API only exposes speed advisories, not physical sign hardware state
2. **VMS proxy is approximate** — `SPEEDMANAGEMENTID` deviations are roadwork-related speed limits, not real-time VMS activations
3. **YOLO model untrained** — using default YOLOv8n weights, no fine-tuning on Swedish traffic cameras yet
4. **Single corridor** — hardcoded to E4 Södertälje→Stockholm, needs generalization for other highways
5. **BEV calibration not yet run** — homography matrices need to be computed per camera using `roi_helper.py` (`b` key)

## What's Next

- [ ] Run BEV calibration (`roi_helper.py` → `b` key) on all configured cameras
- [ ] Deploy on VPS with Docker Compose (`main.py` single-service runtime)
- [ ] Fine-tune YOLO on Swedish camera images (night, winter, sun glare)
- [ ] Build control room dashboard frontend consuming the Operator API
- [ ] Formal DATEX II XSD validation against Trafikverket's NTS schemas
