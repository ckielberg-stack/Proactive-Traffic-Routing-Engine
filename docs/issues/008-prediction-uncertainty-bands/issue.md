# Issue 008 - Add Prediction Uncertainty Bands

## Problem

VMS recommendations currently present point estimates for queue growth and activation timing. Inputs can be uncertain due to camera confidence, missing sensors, ROI calibration quality, and recent model error.

## Impact

Operators need to know whether an ETA is precise enough for immediate action or should be treated as advisory. Point estimates can create false confidence.

## Suggested Approach

- Propagate uncertainty from perception confidence, ROI calibration status, sensor availability, and recent replay error.
- Represent queue-tail ETA as an interval, for example 4-7 minutes, alongside the point estimate.
- Include uncertainty in urgency classification and operator narrative.
- Add dashboard/API fields for confidence and interval bounds.

## Acceptance Criteria

- `QueuePrediction` or `VMSRecommendation` exposes ETA interval/confidence fields.
- Recommendations degrade gracefully when uncertainty is high.
- Tests cover high-confidence, low-confidence, and missing-data scenarios.
- Operator-facing summaries make uncertainty visible without hiding the actionable recommendation.

## References

- `src/models.py`
- `src/vms_orchestrator.py`
- `src/operator_api.py`
- `templates/tmc.html`
