# Learning Notes Index

Use `docs/notes/_template-learning-note.md` for new notes.

## High-signal recurring lessons

- Prediction quality depends on calibrated spatial inputs and preserving derived density across the perception-to-physics handoff. Open when physics produces too few predictions, VMS recommendations are mostly sensor-driven, `RoadSegmentState` changes, `roi_length_meters` validation fails, or ROI calibration is being changed: [2026-06-05 - Traffic Prediction Density Inputs](2026-06-05-traffic-prediction-density-inputs.md).
- Keep cameras, sensors, physics segments, and VMS gantries on the same route-linear chainage datum. Open when changing `CAMERA_COORDS`, sensor mapping, VMS matching, queue ETA logic, or TravelTimeRoute spatial validation: [2026-06-06 - Route-Linear Chainage Datum](2026-06-06-route-linear-chainage-datum.md).
- Keep local speed/inflow provenance visible through physics output. Open when changing `SegmentTrafficState`, `SegmentSpeed`, TrafficFlow aggregation, TravelTimeRoute fallback, or prediction confidence: [2026-06-06 - Segment-Local Sensor Fusion](2026-06-06-segment-local-sensor-fusion.md).
- Prediction uncertainty must be computed at the physics boundary and carried through VMS/API/dashboard surfaces. Open when changing `QueuePrediction` confidence fields, ETA interval logic, VMS urgency classification, JSONL queue records, or learned residual correction: [2026-06-16 - Prediction Uncertainty Bands](2026-06-16-prediction-uncertainty-bands.md).
- Learned residual correction should adjust ETA metadata without mutating base LWR wave speed or queue geometry. Open when changing residual buckets, corrected ETA fields, replay ETA comparison, or VMS corrected activation timing: [2026-06-27 - ETA Residual Correction Boundary](2026-06-27-eta-residual-correction-boundary.md).
- Situation accident/roadwork records are capacity-loss inputs, not VMS proxy ground truth. Open when changing `Situation` ingestion, `fetch_vms_status`, synthetic/corroborated `CapacityState`, or accident/roadwork physics handoff: [2026-06-13 - Situation Capacity Inputs](2026-06-13-situation-capacity-inputs.md).
- Weather surface adjustments must lower both bottleneck trigger thresholds and capacity caps, while preserving legacy weather/road-condition JSONL records. Open when changing `WeatherMeasurepoint`, `RoadCondition`, `WeatherAdapter`, `PhysicsEngine.critical_density_veh_km_lane`, or HALKA VMS warnings: [2026-06-13 - Weather Surface Physics Adjustment](2026-06-13-weather-surface-physics-adjustment.md).
- SMHI forecasts are an escalation-only, UTC-normalized, poll-throttled input that pre-degrades physics and pre-stages HALKA before friction drops. Open when changing `SMHIForecastSource`, `WeatherAdapter.compute(forecast=...)`, `proactive_halka`, forecast persistence, or SMHI tick wiring/stubbing: [2026-06-23 - SMHI Forecast Proactive Weather Adjustment](2026-06-23-smhi-forecast-proactive-weather.md).
- DATEX safety exports should use derived weather context and keep accident/roadwork Situation records separate from VMS proxy ground truth. Open when changing `_build_datex2_xml`, `set_pipeline_snapshot`, weather/situation DATEX records, or XML validation: [2026-06-28 - DATEX Safety Context Source Boundary](2026-06-28-datex-safety-context-source-boundary.md).
- Default deployment boundaries should exclude quarantined legacy entry points structurally and be regression-tested. Open when changing `Dockerfile`, `docker-compose.yml`, `legacy/`, healthchecks, or default runtime entrypoints: [2026-06-28 - Default Deployment Boundary](2026-06-28-default-deployment-boundary.md).
- Hot-path extractions need an explicit transitional facade when old tests or scripts patch `main_loop` globals. Open when moving tick orchestration, source fetchers, camera workers, fusion helpers, or persistence out of an entrypoint: [2026-06-29 - Hot-Path Extraction Facade](2026-06-29-hot-path-extraction-facade.md).
- Stopped-vehicle persistence must be applied after density smoothing and before fused capacity/physics, with local-speed gating and northbound ROI filtering. Open when changing `TrackPersistence`, `fetch_cameras`, `VisionEngine` detection metadata, stopped-vehicle anomalies, or multi-ROI direction filtering: [2026-06-17 - Stopped Vehicle Cross-Tick Persistence](2026-06-17-stopped-vehicle-cross-tick-persistence.md).
- Parallel camera workers need isolated inference state and locked persistence state. Open when changing `fetch_cameras`, `VisionEngine` lifetime, `RetentionPolicy`, camera worker pools, or tick duration behavior: [2026-06-10 - Camera Worker Concurrency State](2026-06-10-camera-worker-concurrency-state.md).
- Replay metrics must pair precision/recall with ETA, distance, expiry, and VMS lead-time checks. Open when changing `QueuePrediction`, TravelTimeRoute matching, replay fixtures, or model comparison workflows: [2026-06-06 - Replay Evaluation Metrics](2026-06-06-replay-evaluation-metrics.md).
- New audit/proposal docs must be synchronized into canonical `.ai/` before `continue` can pick them up. Open when triaging roadmap work, adding docs under `docs/audit/` or `docs/proposals/`, changing task order, or seeing `.ai/state.json.next` point at stale work: [2026-06-10 - AI Queue Triage Synchronization](2026-06-10-ai-queue-triage-synchronization.md).

## Execution State

- 2026-06-10 - [AI Queue Triage Synchronization](2026-06-10-ai-queue-triage-synchronization.md) - Keywords: `.ai/state.json`, `.ai/tasks.json`, `ledger.jsonl`, triage, audit plan, proposal docs, duplicate tasks, queue order. Open when converting planning docs into executable queue state or resolving stale `next` pointers.

## Architecture

- 2026-06-29 - [Hot-Path Extraction Facade](2026-06-29-hot-path-extraction-facade.md) - Keywords: `main_loop.py`, `tick_once`, compatibility shims, monkeypatch sync, hot-path extraction, facade tests. Open when extracting orchestration or migrating callers from an entrypoint to focused `src/` modules.

## Deployment

- 2026-06-28 - [Default Deployment Boundary](2026-06-28-default-deployment-boundary.md) - Keywords: Dockerfile, docker-compose, single-service runtime, legacy quarantine, healthcheck, deployment-shape tests. Open when changing default service topology, entrypoints, or Docker copy boundaries.

## Operator & Export

- 2026-06-28 - [DATEX Safety Context Source Boundary](2026-06-28-datex-safety-context-source-boundary.md) - Keywords: DATEX II, `_build_datex2_xml`, `WeatherAdjustment`, `SituationDeviation`, safety context, VMS proxy separation, XML well-formedness. Open when changing DATEX weather exports, Situation safety records, or operator API snapshot fields.

## Prediction & Calibration

- 2026-06-27 - [ETA Residual Correction Boundary](2026-06-27-eta-residual-correction-boundary.md) - Keywords: learned residuals, base ETA, corrected ETA, LWR boundary, residual buckets, false-positive preservation, VMS activation timing. Open when changing residual correction, replay ETA comparison, or corrected ETA persistence.
- 2026-06-16 - [Prediction Uncertainty Bands](2026-06-16-prediction-uncertainty-bands.md) - Keywords: `QueuePrediction`, `VMSRecommendation`, ETA intervals, confidence bands, uncertainty reason, upper-bound urgency, JSONL/DATEX/dashboard propagation. Open when changing prediction confidence, VMS urgency, or learned residual uncertainty.
- 2026-06-13 - [Situation Capacity Inputs](2026-06-13-situation-capacity-inputs.md) - Keywords: `Situation`, accidents, roadwork, capacity factors, synthetic `CapacityState`, corroboration, route-linear chainage, VMS proxy separation. Open when adding external incident inputs or changing Situation API handling.
- 2026-06-13 - [Weather Surface Physics Adjustment](2026-06-13-weather-surface-physics-adjustment.md) - Keywords: `WeatherMeasurepoint`, `RoadCondition`, `WeatherAdapter`, surface factors, critical density, capacity cap, HALKA VMS, JSONL persistence. Open when changing weather-adjusted physics, road-condition warnings, or slippery-road VMS advisories.
- 2026-06-23 - [SMHI Forecast Proactive Weather Adjustment](2026-06-23-smhi-forecast-proactive-weather.md) - Keywords: `SMHIForecastSource`, `WeatherForecast`, metfcst `pcat`, escalation-only worst-of, `proactive_halka`, HALKRISK, UTC normalization, poll throttle, `fetch_smhi_forecast` stubbing. Open when changing forecast ingestion, anticipatory physics pre-degradation, or pre-staged HALKA advisories.
- 2026-06-05 - [Traffic Prediction Density Inputs](2026-06-05-traffic-prediction-density-inputs.md) - Keywords: multi-ROI, `RoadSegmentState`, `observed_density_veh_km_lane`, `roi_length_meters`, `roi_length_calibration`, LWR gating, queue predictions. Open when improving ETA accuracy, camera density estimation, or VMS prediction hit rate.
- 2026-06-06 - [Route-Linear Chainage Datum](2026-06-06-route-linear-chainage-datum.md) - Keywords: route-linear chainage, `src.route_chainage`, `build_camera_chainage_map`, `build_node_inflows`, VMS chainage, sensor-to-node mapping. Open when changing spatial mapping, queue ETA distances, or sensor/VMS matching.
- 2026-06-06 - [Segment-Local Sensor Fusion](2026-06-06-segment-local-sensor-fusion.md) - Keywords: `SegmentTrafficState`, `SegmentSpeed.local_speed_kmh`, TrafficFlow volume-weighted speed, TravelTimeRoute fallback, local/fallback/missing diagnostics. Open when changing LWR input fusion or prediction confidence.
- 2026-06-06 - [Replay Evaluation Metrics](2026-06-06-replay-evaluation-metrics.md) - Keywords: `src.replay_evaluator`, replay JSONL, precision/recall, ETA error, distance error, prediction expiry, VMS lead time. Open when comparing model changes or adding replay fixtures.

## Performance

- 2026-06-17 - [Stopped Vehicle Cross-Tick Persistence](2026-06-17-stopped-vehicle-cross-tick-persistence.md) - Keywords: `TrackPersistence`, `vehicle_stopped`, IoU, cross-tick state, local speed gate, `_vehicle_detections`, northbound ROI filtering, density smoothing order. Open when changing stopped-vehicle detection or adding new temporal perception state.
- 2026-06-10 - [Camera Worker Concurrency State](2026-06-10-camera-worker-concurrency-state.md) - Keywords: `fetch_cameras`, bounded worker pool, thread-local `VisionEngine`, YOLO predictor lock, `RetentionPolicy`, training schedule persistence, deterministic future ordering. Open when parallelizing camera work or changing tick duration behavior.

## Chronological Notes

| Date | Title | Retrieval cues |
|---|---|---|
| 2026-06-29 | [Hot-Path Extraction Facade](2026-06-29-hot-path-extraction-facade.md) | `main_loop.py` facade, tick orchestration extraction, monkeypatch sync, direct owner tests plus compatibility integration |
| 2026-06-28 | [Default Deployment Boundary](2026-06-28-default-deployment-boundary.md) | Dockerfile explicit COPY, Compose single-service runtime, legacy quarantine, healthcheck targets, deployment-shape regression tests |
| 2026-06-28 | [DATEX Safety Context Source Boundary](2026-06-28-datex-safety-context-source-boundary.md) | DATEX weather safety export, derived WeatherAdjustment source, Situation accident/roadwork export, VMS proxy separation, XML well-formedness |
| 2026-06-27 | [ETA Residual Correction Boundary](2026-06-27-eta-residual-correction-boundary.md) | ETA-only learned residuals, base vs corrected ETA, residual buckets, preserve LWR geometry, replay ETA delta without false-positive change |
| 2026-06-23 | [SMHI Forecast Proactive Weather Adjustment](2026-06-23-smhi-forecast-proactive-weather.md) | SMHI metfcst point forecast, escalation-only worst-of, proactive HALKA pre-staging, UTC normalization, poll throttle, forecast persistence/status fields |
| 2026-06-17 | [Stopped Vehicle Cross-Tick Persistence](2026-06-17-stopped-vehicle-cross-tick-persistence.md) | Cross-tick stopped vehicles, IoU box persistence, local speed gating, density smoothing order, northbound-only ROI promotion |
| 2026-06-16 | [Prediction Uncertainty Bands](2026-06-16-prediction-uncertainty-bands.md) | Queue/VMS uncertainty fields, ETA interval bounds, upper-bound urgency downgrade, confidence propagation across API/JSONL/dashboard |
| 2026-06-13 | [Situation Capacity Inputs](2026-06-13-situation-capacity-inputs.md) | Situation accident/roadwork ingestion, synthetic capacity states, incident corroboration, route-linear deviation matching |
| 2026-06-13 | [Weather Surface Physics Adjustment](2026-06-13-weather-surface-physics-adjustment.md) | WeatherMeasurepoint/RoadCondition tick ingestion, weather-adjusted critical density and capacity cap, HALKA standalone VMS warnings |
| 2026-06-10 | [Camera Worker Concurrency State](2026-06-10-camera-worker-concurrency-state.md) | Camera worker pools, thread-local inference engines, retention schedule locking, deterministic parallel output order |
| 2026-06-10 | [AI Queue Triage Synchronization](2026-06-10-ai-queue-triage-synchronization.md) | Canonical `.ai` state, triage queue sync, stale `next`, audit/proposal task promotion, duplicate planning items |
| 2026-06-06 | [Replay Evaluation Metrics](2026-06-06-replay-evaluation-metrics.md) | Offline replay metrics, TravelTimeRoute route spans, expired predictions, false positives, baseline metrics artifacts |
| 2026-06-06 | [Segment-Local Sensor Fusion](2026-06-06-segment-local-sensor-fusion.md) | Segment-local speed/inflow, source labels, TrafficFlow vs TravelTime precedence, LWR confidence diagnostics |
| 2026-06-06 | [Route-Linear Chainage Datum](2026-06-06-route-linear-chainage-datum.md) | Shared chainage datum, route projection, same-latitude cameras, sensor inflows mapping to wrong node, VMS matching by chainage |
| 2026-06-05 | [Traffic Prediction Density Inputs](2026-06-05-traffic-prediction-density-inputs.md) | Multi-ROI density handoff, max segment density aggregation, ROI length validation, missing/default lengths, no physics predictions despite camera anomalies |
