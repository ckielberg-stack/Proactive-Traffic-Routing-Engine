# Issue 002 - Calibrate ROI Physical Lengths

## Problem

Every ROI currently lacks `roi_length_meters`, so `ROIRegion` defaults to `100.0` meters. Density is computed from vehicle count divided by ROI length, so inaccurate lengths directly bias congestion detection and shockwave prediction.

## Impact

The same vehicle count can produce either a false congestion signal or a missed bottleneck depending on the actual visible road length. Prediction quality cannot be trusted until ROI physical lengths are calibrated.

## Suggested Approach

- Run or improve the ROI calibration workflow for all camera ROIs.
- Store `roi_length_meters` in `camera_config.json`.
- Add a validation script that reports missing, defaulted, or suspicious ROI lengths.
- Consider a minimum calibration quality flag per ROI so low-confidence geometry can reduce prediction confidence.

## Acceptance Criteria

- All active ROIs have non-default `roi_length_meters`.
- A validation command fails or warns clearly when ROI length data is missing.
- Density calculations use calibrated lengths for every configured camera.
- Prediction QA includes before/after counts for bottlenecks and VMS recommendations.

## References

- `camera_config.json`
- `src/roi_mapper.py`
- `src/vision_engine.py`
- `roi_helper.py`
