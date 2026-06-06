# Issue 001 - Preserve Multi-ROI Density In CapacityState

## Problem

All configured cameras use multi-ROI analysis, but `main_loop.fetch_cameras()` aggregates `MultiSegmentCapacity` into `CapacityState` without setting `observed_density_veh_km_lane`. The field falls back to `0.0`, and `PhysicsEngine.compute()` skips states below `K_CRITICAL_VEH_KM_LANE`.

## Impact

Queue predictions can be suppressed even when one or more ROI segments are congested. This likely reduces predictive VMS recommendations and makes the system depend too heavily on sensor-only anomaly recommendations.

## Suggested Approach

- Add `observed_density_veh_km_lane` to `RoadSegmentState`, or expose segment density in another structured way from `VisionEngine.analyze_multi_roi()`.
- Aggregate segment density into the camera-level `CapacityState`. Start with max segment density for bottleneck detection, or a lane/length-weighted density if the physics model expects a whole-camera representative value.
- Persist density in `vision_records` for debugging.
- Ensure temporal smoothing operates on meaningful density values.

## Acceptance Criteria

- A congested multi-ROI segment produces a camera-level `CapacityState.observed_density_veh_km_lane` above critical density.
- Existing single-ROI behavior remains unchanged.
- A regression test proves a high-density multi-ROI camera can reach `PhysicsEngine.compute()`.
- Dashboard/API records expose enough density information to debug prediction gating.

## References

- `main_loop.py` multi-ROI aggregation
- `src/vision_engine.py` per-ROI density calculation
- `src/physics_engine.py` density gate
- `docs/notes/2026-06-05-traffic-prediction-density-inputs.md`
