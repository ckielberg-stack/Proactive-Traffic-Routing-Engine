# 2026-06-05 - Traffic Prediction Density Inputs

## Context

PTRE predicts queue propagation from YOLO-derived density/capacity states, TrafficFlow sensor inflows, TravelTimeRoute calibration, and an LWR shockwave model. Prediction quality depends on preserving density from per-camera perception into the physics engine and on using realistic physical ROI lengths.

## What I Learned

All configured cameras currently use multi-ROI analysis. In the multi-ROI path, `main_loop.fetch_cameras()` aggregates `RoadSegmentState` values into a `CapacityState`, but the aggregated state does not carry `observed_density_veh_km_lane`; it falls back to the model default of `0.0`. The physics engine skips bottlenecks below `K_CRITICAL_VEH_KM_LANE`, so this can suppress queue predictions even when per-ROI density crossed the critical threshold.

The fix is to make density part of the multi-ROI contract: store `observed_density_veh_km_lane` on each `RoadSegmentState`, then aggregate the maximum segment density into the camera-level `CapacityState`. The max preserves a localized bottleneck better than averaging across clear opposite-direction or adjacent ROIs.

The ROI config also lacked `roi_length_meters` for every ROI, so `ROIRegion` defaulted each physical road segment length to 100 m. That fallback made density sensitive to a hidden calibration assumption instead of the actual road length visible in each camera.

The first repeatable calibration pass uses `src.roi_length_calibration` to estimate ROI length from polygon major/minor axes scaled by Swedish lane width (`num_lanes * 3.5 m`). This is a clear improvement over a universal 100 m default and keeps validation CI-friendly, but the values are still `estimated`; BEV homography or surveyed calibration should replace them when available.

## Reuse Rules

- When changing multi-ROI perception, verify that aggregated `CapacityState` preserves a representative density for physics, not only vehicle count, capacity, and anomaly flags.
- Use max segment density for camera-level physics gating unless the downstream model becomes direction-aware.
- Treat `roi_length_meters` as prediction-critical data, not UI metadata.
- Run `python -m src.roi_length_calibration validate --config camera_config.json` after changing ROI polygons.
- Keep `roi_length_source` and `roi_length_confidence` honest; estimated geometry should not be presented as surveyed truth.
- Prefer route-linear chainage and calibrated ROI lengths before tuning LWR constants or adding a more complex model.
- Add tests around the handoff from `MultiSegmentCapacity` to `CapacityState` whenever prediction gating depends on derived fields.

## Failure Signals

- Many camera anomalies but few or no `QueuePrediction` records.
- Smoothed density remains near `0.0` for cameras with ROI detections.
- ROI length validation reports missing/default/suspicious `roi_length_meters`.
- VMS recommendations come mostly from sensor anomalies rather than queue-tail predictions.

## Next Checklist

- Replace lane-width geometry estimates with BEV/surveyed calibration where possible.
- Compare prediction counts before and after ROI length calibration.
- Add replay fixtures that assert a congested ROI produces a physics prediction.
