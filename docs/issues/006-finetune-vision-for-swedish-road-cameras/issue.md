# Issue 006 - Fine-Tune Vision For Swedish Road Cameras

## Problem

The vision engine uses generic YOLO vehicle classes. Swedish traffic cameras have difficult conditions: night, rain, snow, glare, occlusion, stopped traffic, and perspective-heavy motorway views.

## Impact

Detection misses or hallucinations affect density, capacity, anomaly flags, and therefore queue predictions. The physics model can only be as good as the perception state it receives.

## Suggested Approach

- Curate a local/private labeled dataset from retained frames, respecting privacy and repo data rules.
- Fine-tune or evaluate a stronger vehicle detector against YOLOv8n.
- Add night-mode fallback using headlight/taillight clusters for low-light direction/count verification.
- Track confidence calibration by condition: daylight, night, rain, snow, glare, and camera availability.

## Acceptance Criteria

- A documented training/evaluation workflow exists.
- Offline evaluation reports precision/recall by condition and camera group.
- The runtime can select or configure the improved detector.
- Low-light fallback improves or clearly diagnoses night detection failures.

## References

- `src/vision_engine.py`
- `retention.py`
- `storage/training/`
- `camera_config.json`
