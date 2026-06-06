# Issue 007 - Add Learned Residual Correction To LWR Predictions

## Problem

The LWR model is explainable and useful, but its constants and inputs cannot capture every segment-specific behavior, ramp effect, weather pattern, or systematic timing bias.

## Impact

Pure physics predictions may be consistently early or late on particular segments or time windows, limiting ETA accuracy even after input calibration.

## Suggested Approach

- Keep LWR as the baseline prediction.
- Learn a residual correction for ETA or wave speed using historical prediction errors.
- Start with simple online calibration or a regularized model using segment id, time of day, day type, sensor confidence, travel-time status, and perception confidence.
- Store correction metadata so operators can still understand the base physics result and the adjustment.

## Acceptance Criteria

- The system can report base LWR ETA and corrected ETA separately.
- Residual correction is disabled safely when insufficient history exists.
- Replay evaluation shows whether correction improves ETA error without increasing false positives.
- Tests cover fallback behavior and correction bounds.

## References

- `src/physics_engine.py`
- `src/evaluation_logger.py`
- `src/travel_time_calibrator.py`
