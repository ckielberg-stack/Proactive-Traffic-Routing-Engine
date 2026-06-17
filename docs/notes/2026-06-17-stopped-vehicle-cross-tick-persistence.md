# 2026-06-17 - Stopped Vehicle Cross-Tick Persistence

## Context

TRAFIK-031 added stopped-vehicle detection to the tick pipeline by comparing
YOLO vehicle boxes across consecutive camera ticks. The camera workers remain
parallel and use thread-local `VisionEngine` instances, while the stopped
vehicle tracker is shared across ticks.

## What I Learned

Cross-tick perception state should live at the orchestrator boundary, not
inside per-worker inference engines. `fetch_cameras()` can expose detection
metadata from the same YOLO inference that produced capacity, but the stopped
vehicle decision needs sensor-derived node speed that is only available after
the concurrent camera and TrafficFlow futures complete.

For stopped vehicles, the persistence signal must be applied after density
smoothing. If it is applied before the EMA pass, the forced bottleneck density
can be damped below the physics threshold and the queue-tail prediction path
will miss the safety event.

## Reuse Rules

- Keep mutable cross-tick trackers as explicit singletons in `main_loop.py`,
  with internal locking when camera workers can update or age state.
- Carry YOLO detection metadata through worker records as internal data, then
  remove it before JSONL persistence unless it becomes a compact event payload.
- Gate stopped-vehicle detections with local node speed; if speed is unknown or
  already congested, let existing queue/sensor anomaly paths handle it.
- Apply stopped-vehicle promotion after density smoothing and before fused
  capacity derivation and physics.
- For multi-ROI cameras, promote only the corridor direction used by physics;
  keep opposite-direction detections out of northbound bottleneck state.

## Failure Signals

- `vehicle_stopped` appears during normal queue congestion with low sensor
  speed.
- `vision_result` JSONL records contain raw `_vehicle_detections`.
- A stopped vehicle is logged as an anomaly but does not produce any
  `QueuePrediction`.
- Southbound-only ROI detections create northbound VMS recommendations.
- Camera worker tests become order-dependent after adding persistence.

## Next Checklist

- Tune IoU, tick count, and speed gate with live replay clips before changing
  defaults.
- Verify local speed mapping for every production camera before trusting
  stopped-vehicle alerts operationally.
- If annotated stopped-vehicle thumbnails become required, move the speed-gated
  persistence decision into a phase that still has access to decoded frames.
