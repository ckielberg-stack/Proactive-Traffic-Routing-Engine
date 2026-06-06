# 2026-06-06 - Replay Evaluation Metrics

## Context

TRAFIK-005 added offline replay metrics over persisted JSONL ticks. The goal was to move beyond corridor-level TravelTime hit rate and measure whether queue predictions were early, spatially close, and useful before proxy VMS activation.

## What I Learned

Replay evaluation needs an explicit route-linear span model for TravelTimeRoute segments. Without reconstructing segment spans from ordered route IDs and route lengths, a corridor-level hit can hide wrong ETA or queue-tail distance.

Prediction expiry must be part of the metric contract. Otherwise old predictions can accidentally match later congestion and inflate recall while hiding false positives.

## Reuse Rules

- Use `src.replay_evaluator` for offline JSONL metrics instead of `TravelTimeCalibrator.evaluate_accuracy` when comparing model changes.
- Keep metrics versioned with `METRICS_VERSION` so artifacts can be compared across evaluator changes.
- Treat `queue_prediction` records as active only until the configured expiry window.
- Report both corridor-level and per-route metrics; precision/recall alone is not enough without ETA and distance error.

## Failure Signals

- Precision or recall improves while `mean_abs_eta_error_minutes` or `mean_distance_error_km` gets worse.
- Predictions created long before a congestion event still count as hits.
- Replay command requires a live API key or imports the tick loop.
- Baseline metrics cannot be regenerated exactly from a small fixture.

## Next Checklist

- When changing queue physics, regenerate replay metrics and compare the same `version`.
- If TravelTimeRoute geometry becomes available, replace length-proportional route spans with endpoint geometry.
- Keep sample fixtures small and deterministic so CI can validate artifact generation.
