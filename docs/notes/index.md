# Learning Notes Index

Use `docs/notes/_template-learning-note.md` for new notes.

## High-signal recurring lessons

- Prediction quality depends on calibrated spatial inputs and preserving derived density across the perception-to-physics handoff. Open when physics produces too few predictions, VMS recommendations are mostly sensor-driven, `RoadSegmentState` changes, `roi_length_meters` validation fails, or ROI calibration is being changed: [2026-06-05 - Traffic Prediction Density Inputs](2026-06-05-traffic-prediction-density-inputs.md).
- Keep cameras, sensors, physics segments, and VMS gantries on the same route-linear chainage datum. Open when changing `CAMERA_COORDS`, sensor mapping, VMS matching, queue ETA logic, or TravelTimeRoute spatial validation: [2026-06-06 - Route-Linear Chainage Datum](2026-06-06-route-linear-chainage-datum.md).
- Keep local speed/inflow provenance visible through physics output. Open when changing `SegmentTrafficState`, `SegmentSpeed`, TrafficFlow aggregation, TravelTimeRoute fallback, or prediction confidence: [2026-06-06 - Segment-Local Sensor Fusion](2026-06-06-segment-local-sensor-fusion.md).
- Parallel camera workers need isolated inference state and locked persistence state. Open when changing `fetch_cameras`, `VisionEngine` lifetime, `RetentionPolicy`, camera worker pools, or tick duration behavior: [2026-06-10 - Camera Worker Concurrency State](2026-06-10-camera-worker-concurrency-state.md).
- Replay metrics must pair precision/recall with ETA, distance, expiry, and VMS lead-time checks. Open when changing `QueuePrediction`, TravelTimeRoute matching, replay fixtures, or model comparison workflows: [2026-06-06 - Replay Evaluation Metrics](2026-06-06-replay-evaluation-metrics.md).
- New audit/proposal docs must be synchronized into canonical `.ai/` before `continue` can pick them up. Open when triaging roadmap work, adding docs under `docs/audit/` or `docs/proposals/`, changing task order, or seeing `.ai/state.json.next` point at stale work: [2026-06-10 - AI Queue Triage Synchronization](2026-06-10-ai-queue-triage-synchronization.md).

## Execution State

- 2026-06-10 - [AI Queue Triage Synchronization](2026-06-10-ai-queue-triage-synchronization.md) - Keywords: `.ai/state.json`, `.ai/tasks.json`, `ledger.jsonl`, triage, audit plan, proposal docs, duplicate tasks, queue order. Open when converting planning docs into executable queue state or resolving stale `next` pointers.

## Prediction & Calibration

- 2026-06-05 - [Traffic Prediction Density Inputs](2026-06-05-traffic-prediction-density-inputs.md) - Keywords: multi-ROI, `RoadSegmentState`, `observed_density_veh_km_lane`, `roi_length_meters`, `roi_length_calibration`, LWR gating, queue predictions. Open when improving ETA accuracy, camera density estimation, or VMS prediction hit rate.
- 2026-06-06 - [Route-Linear Chainage Datum](2026-06-06-route-linear-chainage-datum.md) - Keywords: route-linear chainage, `src.route_chainage`, `build_camera_chainage_map`, `build_node_inflows`, VMS chainage, sensor-to-node mapping. Open when changing spatial mapping, queue ETA distances, or sensor/VMS matching.
- 2026-06-06 - [Segment-Local Sensor Fusion](2026-06-06-segment-local-sensor-fusion.md) - Keywords: `SegmentTrafficState`, `SegmentSpeed.local_speed_kmh`, TrafficFlow volume-weighted speed, TravelTimeRoute fallback, local/fallback/missing diagnostics. Open when changing LWR input fusion or prediction confidence.
- 2026-06-06 - [Replay Evaluation Metrics](2026-06-06-replay-evaluation-metrics.md) - Keywords: `src.replay_evaluator`, replay JSONL, precision/recall, ETA error, distance error, prediction expiry, VMS lead time. Open when comparing model changes or adding replay fixtures.

## Performance

- 2026-06-10 - [Camera Worker Concurrency State](2026-06-10-camera-worker-concurrency-state.md) - Keywords: `fetch_cameras`, bounded worker pool, thread-local `VisionEngine`, YOLO predictor lock, `RetentionPolicy`, training schedule persistence, deterministic future ordering. Open when parallelizing camera work or changing tick duration behavior.

## Chronological Notes

| Date | Title | Retrieval cues |
|---|---|---|
| 2026-06-10 | [Camera Worker Concurrency State](2026-06-10-camera-worker-concurrency-state.md) | Camera worker pools, thread-local inference engines, retention schedule locking, deterministic parallel output order |
| 2026-06-10 | [AI Queue Triage Synchronization](2026-06-10-ai-queue-triage-synchronization.md) | Canonical `.ai` state, triage queue sync, stale `next`, audit/proposal task promotion, duplicate planning items |
| 2026-06-06 | [Replay Evaluation Metrics](2026-06-06-replay-evaluation-metrics.md) | Offline replay metrics, TravelTimeRoute route spans, expired predictions, false positives, baseline metrics artifacts |
| 2026-06-06 | [Segment-Local Sensor Fusion](2026-06-06-segment-local-sensor-fusion.md) | Segment-local speed/inflow, source labels, TrafficFlow vs TravelTime precedence, LWR confidence diagnostics |
| 2026-06-06 | [Route-Linear Chainage Datum](2026-06-06-route-linear-chainage-datum.md) | Shared chainage datum, route projection, same-latitude cameras, sensor inflows mapping to wrong node, VMS matching by chainage |
| 2026-06-05 | [Traffic Prediction Density Inputs](2026-06-05-traffic-prediction-density-inputs.md) | Multi-ROI density handoff, max segment density aggregation, ROI length validation, missing/default lengths, no physics predictions despite camera anomalies |
