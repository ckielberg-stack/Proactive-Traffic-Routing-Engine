# 2026-06-13 - Situation Capacity Inputs

## Context

TRAFIK-030 added Trafikverket `Situation` accident and roadwork deviations as direct inputs to the tick-loop physics pipeline.

## What I Learned

Authoritative Situation records should not be forced into the VMS proxy-ground-truth model. Speed-management deviations describe operator action, while accident and roadwork deviations describe upstream capacity loss. Keeping them as a separate domain model lets the tick loop persist raw incident records, corroborate vision anomalies, or synthesize a capacity state when vision is missing.

The least invasive physics integration is to merge Situation capacity impacts into `CapacityState` before `PhysicsEngine.compute()`. This preserves the existing physics API and VMS queue recommendation flow while still allowing camera-independent bottlenecks.

## Reuse Rules

- Keep `fetch_vms_status()` speed-management-only; add separate Situation fetchers for incident inputs.
- Project Situation geometry onto the same route-linear datum before matching cameras or VMS.
- Apply Situation impacts after local-speed/weather capacity derivation so authoritative capacity loss is not overwritten.
- Persist `"type": "situation"` records separately from `vms_status` ground-truth records.

## Failure Signals

- Accident or roadwork deviations appear as VMS proxy statuses.
- Situation-only incidents are logged but produce no capacity state or queue prediction.
- Camera anomaly confidence does not increase when an authoritative Situation record maps to the same chainage.
- Out-of-corridor E4 geometry produces a corridor chainage instead of preserving only the raw record.

## Next Checklist

- Test both existing-state corroboration and synthetic-state creation.
- Include two Situation query paths in tick tests: speed-management proxy and accident/roadwork inputs.
- Keep capacity-factor mappings conservative until replay data can calibrate lane-restriction behavior.
