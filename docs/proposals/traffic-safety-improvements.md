# Traffic-Safety Improvement Proposals — Trafikverket API + External Tools

**Status:** Proposal (not yet implemented)
**Date:** 2026-06-09
**Question addressed:** *How can we make traffic safer by using data from the Trafikverket API combined with other tools?*

This document proposes eight prioritized improvements (P1–P8). Every proposal is
grounded in the current codebase: each lists the safety rationale, the data
source involved, the concrete integration point (file + function), rough
effort, and dependencies. Sequencing logic and rejected alternatives are
documented at the end.

---

## 1. Problem statement & safety framing

PTRE's core safety contribution is preventing **secondary rear-end collisions
at queue tails** — the system predicts where a queue tail will be and warns
upstream drivers via VMS before they arrive at standstill traffic. Two
systematic blind spots limit that contribution today:

1. **Dry-road physics assumptions.** The LWR model uses fixed constants —
   `FREE_FLOW_SPEED_KMH = 110` (`src/physics_engine.py:53`),
   `K_CRITICAL_VEH_KM_LANE = 45` (`src/physics_engine.py:65`), and a static
   per-lane capacity cap. On wet, snowy, or icy roads, real capacity and safe
   speed drop 10–30 %, so PTRE **under-predicts queue growth exactly when
   secondary collisions are most lethal** (low friction = longer braking
   distance at the queue tail). The system also cannot issue any
   slippery-road warning today.
2. **Single-sensor bottleneck detection.** YOLOv8n vision is the only trigger
   for bottleneck detection, and it is known-weak at night and in snow
   (issue 006). Authoritative, camera-independent incident signals from the
   Situation API are fetched only through a narrow speed-advisory filter
   (`main_loop.py:838-858`) and never reach the physics engine.

The proposals below close these blind spots first, then improve operator
trust (uncertainty), detection coverage (stopped vehicles), anticipation
(forecasts), and reach (DATEX II).

---

## 2. Prioritized proposals

### P1 — Weather + RoadCondition in the tick loop *(recommended quick win)*

| | |
|---|---|
| **Data source** | Trafikverket `WeatherMeasurepoint` (schemaversion 2) + `RoadCondition` (schemaversion 1.2) |
| **Effort** | ~2–4 days |
| **Dependencies** | None |

**Safety rationale.** Weather-adjusted physics makes queue predictions
correct in degraded conditions (earlier, more accurate VMS activation), and
road-condition warnings enable direct "HALKA" (slippery road) driver
messaging — a safety message the system cannot produce at all today.

**Key finding: this is a port, not new API work.** Working, production-tested
queries already exist in the legacy collector:

- `fetch_weather_data()` — `collect.py:417-469`: air/surface temperature,
  humidity, dewpoint, visibility, wind, precipitation, 5-minute rain sum and
  snow water-equivalent, with bbox filtering.
- `fetch_road_conditions()` — `collect.py:472-509`: `ConditionCode`,
  `ConditionText`, `Warning`, `RoadNumber` for Stockholm county, with an
  `_is_e4_road()` corridor filter (`collect.py:512`).
- `dashboard.py` already renders both record types from JSONL.

They were never ported into `main_loop.py`'s 60-second tick.

**Integration points:**

1. **Fetch** — port both functions into `main_loop.py` and add two futures to
   the `tick_once()` `ThreadPoolExecutor`, mirroring `fetch_travel_times`
   (timeout + try/except pattern).
2. **Adapter** — new `src/weather_adapter.py`:
   `WeatherAdapter.compute(weather, road_conditions) -> WeatherAdjustment`.
   Classify the worst observed corridor surface state (`RoadCondition`
   `ConditionCode`/`Warning` takes precedence; fall back to surface temp
   ≤ 0 °C + precipitation from `WeatherMeasurepoint`). **Fail-safe rule:
   degrade to `dry` / low-confidence when data is stale or missing — never
   raise capacity on missing data.** Mirror `TravelTimeCalibrator`'s
   structure for testability.
3. **Physics injection** — follow the exact precedent of the calibrator,
   which already mutates `physics.free_flow_speed` each tick
   (`main_loop.py:1230-1235`). Compose:
   `physics.free_flow_speed = calibrated_speed * adj.free_flow_factor`, and
   make `K_CRITICAL_VEH_KM_LANE` / capacity an instance-level, factor-adjustable
   input in `PhysicsEngine`. Lowering critical density under degraded surface
   makes bottleneck evaluation trigger *earlier* — the safety-conservative
   direction.
4. **VMS messages** — extend `_build_message(urgency)`
   (`src/vms_orchestrator.py:400`) to accept a surface status, producing e.g.
   `"HALKA — KÖVARNING 50 km/h"`; add a `generate_weather_recommendations()`
   that maps `RoadCondition.Warning == True` records to the nearest gantry
   via the existing chainage helpers and emits standalone HALKA advisories
   even when no queue exists.
5. **Persistence** — write records in `_persist_tick` using the legacy
   `"type": "weather"` / `"type": "road_condition"` JSONL shapes so
   `dashboard.py` and existing JSONL consumers keep working unchanged.
6. **Adjustment table** — conservative HCM-style factors in `config.py`,
   documented as tunable, e.g.
   `{"dry": (1.00, 1.00), "wet": (0.92, 0.90), "snow": (0.85, 0.75), "ice": (0.75, 0.65)}`
   (free-flow factor, capacity factor).

**Deferred sub-idea:** weather-gated YOLO confidence thresholds in
`src/vision_engine.py`. Without labeled night/snow evaluation data
(issue 006), any threshold change is unvalidated. Safe interim step:
reflect degraded weather in `CapacityState.confidence` only.

---

### P2 — Situation accident/roadwork deviations as direct physics inputs

| | |
|---|---|
| **Data source** | Trafikverket `Situation` API (deviation types beyond the current speed-advisory filter) |
| **Effort** | ~3–5 days |
| **Dependencies** | None (compounds with P1) |

**Safety rationale.** YOLOv8n is a single point of failure for bottleneck
detection. Accident and roadwork deviations are camera-independent,
authoritative capacity-loss signals: a confirmed accident should (a) trigger
or corroborate a bottleneck even when the camera sees nothing (night, snow,
glare), and (b) raise the confidence of vision-detected bottlenecks at the
same chainage.

**Current gap.** The main loop's only Situation query
(`fetch_vms_status`, `main_loop.py:838-858`) filters exclusively on
`Deviation.MessageCode = 'Hastighetsbegränsning gäller'` — the
SPEEDMANAGEMENTID proxy used as VMS ground truth. Accident (`Olycka`) and
roadwork (`Vägarbete`) deviations are not fetched in the tick at all.

**Integration points:**

- Widen the query (or add a separate `fetch_situations()`) to include
  accident/roadwork `MessageType` with `Deviation.Geometry.WGS84`,
  `NumberOfLanesRestricted`, and `SeverityCode`.
- Project deviation coordinates to corridor chainage with
  `build_route_chainage_map` (`src/route_chainage.py`).
- Feed into `PhysicsEngine.compute()` as a per-node capacity override (lane
  restriction → fractional capacity drop) or a synthetic high-density
  `CapacityState`.
- Add corroboration flags in `src/incident_builder.py` (vision-detected +
  Situation-confirmed = high confidence).

The lane-restriction → capacity mapping needs care (lane closures don't
reduce capacity strictly proportionally); start conservative and calibrate
against replay data.

---

### P3 — Prediction uncertainty bands *(existing issue 008)*

| | |
|---|---|
| **Data source** | Internal — existing confidence fields and replay error statistics |
| **Effort** | ~3–4 days |
| **Dependencies** | None; prerequisite for P6 |

**Safety rationale.** Operators acting on a false-precise "queue reaches
VMS-4003 in 6.5 min" either over-trust (activate wrongly, eroding driver
compliance — a real safety effect: drivers ignore signs that cry wolf) or
under-trust (ignore the one prediction that mattered). An honest interval
("4–9 min") calibrates operator response.

**Inputs already exist:** `QueuePrediction.data_confidence`, per-segment
local/fallback/missing data counts in the physics engine,
`CapacityState.confidence`, and ETA-error baselines from
`src/replay_evaluator.py`.

**Integration points:** `src/models.py` (`QueuePrediction.eta_interval`,
bounds on `VMSRecommendation`), `src/vms_orchestrator.py`
(`_classify_urgency`, `_build_narrative`), `src/operator_api.py`,
`templates/tmc.html`.

---

### P4 — Stopped-vehicle detection via cross-tick persistence

| | |
|---|---|
| **Data source** | Existing camera feed (no new API) |
| **Effort** | ~3–5 days |
| **Dependencies** | None |

**Safety rationale.** A stationary vehicle in a live lane is the
highest-risk precursor of a secondary collision. Today a stopped car that
YOLO sees is just "1 vehicle, low density" — invisible to every downstream
component. `VisionEngine._check_anomalies` (`src/vision_engine.py`) is three
single-frame heuristics; nothing persists detections across ticks.

**Approach:** compare bounding boxes across consecutive ticks per camera —
IoU > ~0.7 at the same position for ≥ 2–3 ticks while sensor speed at the
node is otherwise normal ⇒ stopped-vehicle flag. Cross-tick state has an
explicit precedent: `DensitySmoother` (ADR-006 sanctioned breaking the
stateless-tick rule for exactly this kind of need).

**Integration points:** new small `src/track_persistence.py` keyed per
camera (same singleton pattern as `_get_density_smoother()` in
`main_loop.py`); hook in `fetch_cameras()` after `engine.analyze_multi_roi`;
new anomaly reason feeding `record_anomaly` and `src/incident_builder.py`.

**Evaluated and rejected: wrong-way driver detection.** With one still image
per 60 seconds there is no motion-direction information; this requires video
or much higher frame rates and should not be attempted on the current feed.

---

### P5 — SMHI open-data forecasts: anticipate, don't just observe

| | |
|---|---|
| **Data source** | SMHI open-data point forecast API (metfcst) — free, no key required |
| **Effort** | ~2 days (after P1) |
| **Dependencies** | P1 (extends `src/weather_adapter.py`) |

**Safety rationale.** `WeatherMeasurepoint` observes current conditions;
SMHI forecasts anticipate them. Knowing snowfall starts in 45 minutes lets
PTRE pre-degrade physics thresholds and pre-stage HALKA advisories *before*
friction drops — true proactivity, matching the project's identity ("predict
where a queue *will be*").

**Integration points:** a forecast source in the P1 `WeatherAdapter`, one
config block in `config.py` (corridor reference point(s), poll interval —
forecasts only need refreshing every ~30 min, not every tick).

---

### P6 — Learned residual correction for LWR *(existing issue 007)*

| | |
|---|---|
| **Data source** | Internal — accumulated JSONL tick history + Prophecy evaluations |
| **Effort** | ~1–2 weeks |
| **Dependencies** | P3 first; weeks of accumulated logged ticks |

**Safety rationale.** Systematic per-segment ETA bias means VMS activates
late on specific segments. Keep LWR as the explainable baseline and learn
only the residual from logged history (`src/evaluation_logger.py`,
`src/replay_evaluator.py`).

**Honest sequencing note:** this is blocked on data volume and on P3 — the
residual statistics are also the natural source of the uncertainty
interval. Do P3 first; revisit this after sustained data collection.

---

### P7 — DATEX II export of weather/safety situation records

| | |
|---|---|
| **Data source** | P1 outputs |
| **Effort** | ~2–3 days |
| **Dependencies** | P1 |

**Safety rationale.** Once PTRE knows the road is slippery, exporting
`slipperyRoad`-type situation records makes its warnings consumable by the
National Traffic Management System and downstream navigation providers —
**drivers outside the eight VMS gantries also get warned.**

**Integration points:** `_build_datex2_xml()` (`src/operator_api.py:353`) —
add a weather situation-record type alongside the existing incident and
speed-management records; also a good moment to add the long-pending XSD
validation of the export.

---

### P8 — Detour advisories (routing, reframed) — *deprioritized*

| | |
|---|---|
| **Data source** | OSRM/OpenStreetMap — **offline only** |
| **Effort** | ~1 week, if ever |
| **Dependencies** | Business decision |

ADR-004 (`docs/roadmap.md`) explicitly removed routing engines: the B2G
customer manages infrastructure, not vehicle routing. Reinstating
OSRM/Valhalla in the tick would reverse an architectural decision and is
**not recommended**.

The defensible slice: a small set of **precomputed static detour
descriptions** per closure scenario (computed offline with OSRM, stored as
config), surfaced as advisory text in `VMSRecommendation` / DATEX II
(e.g. `"LÅNG KÖ — VÄLJ VÄG 73"`). No routing engine in the tick, no graph
state, no new runtime dependency.

---

## 3. Priority logic & sequencing

```
P1 (weather in tick) ──┬──> P5 (SMHI forecast)
                       └──> P7 (DATEX II weather export)
P2 (Situation → physics)        [independent, do in parallel with P1]
P3 (uncertainty bands) ───────> P6 (learned residual; also needs weeks of data)
P4 (stopped vehicles)           [independent]
P8 (detour advisories)          [deprioritized — business decision first]
```

- **P1 first**: cheapest (queries already written), and it raises prediction
  correctness precisely in the conditions where queue-tail collisions kill.
  Its fail-safe design (never raise capacity on missing data) means it can
  only make the system more conservative.
- **P2 second**: removes the single-sensor blind spot — the largest remaining
  detection gap.
- **P3 third**: makes everything else operator-trustworthy; prerequisite
  for P6.
- **P4** is the highest-value *new detection class* and can proceed any time.
- **P5/P7** compound P1; **P6** needs P3 plus data history; **P8** awaits a
  product decision.

**Evaluated and parked (not proposed):**

- *Trafiklab / GTFS-RT public-transport data* — negligible motorway-safety
  value for this corridor.
- *OpenStreetMap as a tick input* — parked; useful only as an offline
  calibration aid for lane counts and ramp positions in
  `camera_config.json`.
- *Wrong-way driver detection* — infeasible on 60-second stills (see P4).
- *Weather-gated YOLO thresholds* — unvalidated without issue-006 evaluation
  data (see P1).

---

## 4. Documentation observations (noted, not fixed here)

- **Camera count inconsistency:** `config.py` and `README.md` say 46
  cameras; `docs/roadmap.md` (Milestone 1) and `docs/handoff.md` say 53.
  `config.CAMERA_IDS` actually contains 46 entries.
- **Direction hardcoded:** northbound-only, with an explicit TODO at
  `src/physics_engine.py:241` (`direction = "northbound"  # TODO: derive
  from CapacityState/ROI road_id`). Southbound support is a prerequisite for
  any multi-corridor expansion.
- **Roadmap status table** marks Milestone 3 (Physics) and 6 (Operator API)
  as "Not started" although both are implemented and tested — worth a
  refresh.
