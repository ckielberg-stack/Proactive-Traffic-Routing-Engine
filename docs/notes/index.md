# Learning Notes Index

Use `/Users/chips/antigravity/trafik/docs/notes/_template-learning-note.md` for new notes.

## High-signal recurring lessons

- Prediction quality depends on calibrated spatial inputs and preserving derived density across the perception-to-physics handoff. Open when physics produces too few predictions, VMS recommendations are mostly sensor-driven, `RoadSegmentState` changes, `roi_length_meters` validation fails, or ROI calibration is being changed: [2026-06-05 - Traffic Prediction Density Inputs](2026-06-05-traffic-prediction-density-inputs.md).
- Keep cameras, sensors, physics segments, and VMS gantries on the same route-linear chainage datum. Open when changing `CAMERA_COORDS`, sensor mapping, VMS matching, queue ETA logic, or TravelTimeRoute spatial validation: [2026-06-06 - Route-Linear Chainage Datum](2026-06-06-route-linear-chainage-datum.md).
- Keep local speed/inflow provenance visible through physics output. Open when changing `SegmentTrafficState`, `SegmentSpeed`, TrafficFlow aggregation, TravelTimeRoute fallback, or prediction confidence: [2026-06-06 - Segment-Local Sensor Fusion](2026-06-06-segment-local-sensor-fusion.md).

## Prediction & Calibration

- 2026-06-05 - [Traffic Prediction Density Inputs](2026-06-05-traffic-prediction-density-inputs.md) - Keywords: multi-ROI, `RoadSegmentState`, `observed_density_veh_km_lane`, `roi_length_meters`, `roi_length_calibration`, LWR gating, queue predictions. Open when improving ETA accuracy, camera density estimation, or VMS prediction hit rate.
- 2026-06-06 - [Route-Linear Chainage Datum](2026-06-06-route-linear-chainage-datum.md) - Keywords: route-linear chainage, `src.route_chainage`, `build_camera_chainage_map`, `build_node_inflows`, VMS chainage, sensor-to-node mapping. Open when changing spatial mapping, queue ETA distances, or sensor/VMS matching.
- 2026-06-06 - [Segment-Local Sensor Fusion](2026-06-06-segment-local-sensor-fusion.md) - Keywords: `SegmentTrafficState`, `SegmentSpeed.local_speed_kmh`, TrafficFlow volume-weighted speed, TravelTimeRoute fallback, local/fallback/missing diagnostics. Open when changing LWR input fusion or prediction confidence.

## Chronological Notes

| Date | Title | Retrieval cues |
|---|---|---|
| 2026-06-06 | [Segment-Local Sensor Fusion](2026-06-06-segment-local-sensor-fusion.md) | Segment-local speed/inflow, source labels, TrafficFlow vs TravelTime precedence, LWR confidence diagnostics |
| 2026-06-06 | [Route-Linear Chainage Datum](2026-06-06-route-linear-chainage-datum.md) | Shared chainage datum, route projection, same-latitude cameras, sensor inflows mapping to wrong node, VMS matching by chainage |
| 2026-06-05 | [Traffic Prediction Density Inputs](2026-06-05-traffic-prediction-density-inputs.md) | Multi-ROI density handoff, max segment density aggregation, ROI length validation, missing/default lengths, no physics predictions despite camera anomalies |
