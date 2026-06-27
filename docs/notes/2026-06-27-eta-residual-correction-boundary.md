# 2026-06-27 - ETA Residual Correction Boundary

## Context

TRAFIK-007 added learned residual correction for LWR queue predictions. The key design choice was to correct ETA surfaces while preserving base LWR wave speed, queue length, and queue-tail geometry.

## What I Learned

Residual learning is safest as metadata at the prediction boundary. If correction mutates wave speed or queue length directly, downstream VMS selection, replay distance error, and operator explanations can no longer distinguish physics behavior from learned timing bias.

Keeping base and corrected ETA fields side by side lets replay evaluation show whether residuals improve timing without changing prediction creation or false-positive counts.

## Reuse Rules

- Keep `growth_speed_kmh`, `lengths_at_minutes`, and LWR uncertainty bands as the base physics result.
- Apply residual correction as a bounded ETA offset only after `PhysicsEngine.compute()`.
- Train residuals from matched replay or TravelTimeRoute timing error, then require enough bucketed history before enabling correction.
- Persist base ETA, corrected ETA, correction amount, sample count, and disabled reason together.
- When evaluating residual changes, compare ETA error deltas while confirming false-positive counts are unchanged.

## Failure Signals

- Learned correction changes queue-tail geometry or distance error.
- VMS recommendations expose only corrected ETA and hide the base LWR ETA.
- Correction is enabled with too few matched samples or without a disabled reason.
- Replay metrics improve ETA error but silently change prediction count or false-positive count.

## Next Checklist

- Before changing residual logic, verify base LWR fields remain unchanged.
- Keep correction bounds conservative until replay artifacts show stable gains.
- Add route/camera/time bucket tests for any new residual feature dimensions.
