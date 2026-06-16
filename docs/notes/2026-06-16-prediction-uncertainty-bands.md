# 2026-06-16 - Prediction Uncertainty Bands

## Context

TRAFIK-008 added confidence-aware queue length bands and VMS ETA intervals. The system already had local/fallback/missing segment diagnostics, camera confidence, ROI calibration confidence, and replay ETA error baselines, but those signals were not part of the prediction contract.

## What I Learned

Uncertainty needs to be computed at the physics boundary and carried forward as first-class data. If each downstream surface recomputes confidence independently, urgency, operator narrative, JSONL replay records, DATEX exports, and the dashboard can disagree about how trustworthy the same prediction is.

ETA interval urgency should use the upper bound, not the point estimate. That makes the recommendation conservative when data is weak while still preserving the actionable point ETA for operators.

## Reuse Rules

- Keep `QueuePrediction` as the source of truth for prediction confidence, uncertainty reason, and queue length bounds.
- Let `VMSRecommendation` copy uncertainty fields from the triggering prediction instead of inventing a separate confidence model.
- Classify VMS urgency from the ETA upper bound when an interval is available.
- Preserve point-estimate fields for compatibility, but add interval fields to API, JSONL, and dashboard surfaces together.
- Treat ROI length confidence as prediction-critical provenance alongside sensor locality and camera confidence.

## Failure Signals

- VMS urgency is immediate/soon while the ETA upper bound is advisory.
- Dashboard shows an interval but JSONL or DATEX exports only contain the point ETA.
- Low camera confidence, missing segment data, or unknown ROI length still produces a narrow ETA interval.
- Replay evaluation cannot explain why predictions were widened or downgraded.

## Next Checklist

- Replace the fixed replay-error floor with recent route/segment residual statistics once enough replay history exists.
- When adding learned residual correction, feed residual uncertainty into the same `QueuePrediction` interval fields.
- Keep tests covering high-confidence, fallback, and missing-data uncertainty paths when changing physics inputs.
