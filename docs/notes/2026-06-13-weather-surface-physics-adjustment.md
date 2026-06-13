# 2026-06-13 - Weather Surface Physics Adjustment

## Context

TRAFIK-029 ported Trafikverket `WeatherMeasurepoint` and `RoadCondition` data into the tick loop so degraded road surfaces can affect LWR queue prediction and VMS warnings.

## What I Learned

Weather adjustment is not only a free-flow speed problem. Wet, snow, and ice conditions need to lower the capacity cap used during camera capacity derivation and the critical-density threshold used by physics. Lowering only speed can leave bottleneck detection too late or keep dry-road maximum capacity in place.

Road-condition warnings are operational safety records as well as physics inputs. They should remain persisted in the legacy JSONL shape for dashboard/API consumers, while a separate adjustment snapshot records the derived model factors.

## Reuse Rules

- Apply surface factors before capacity derivation and before `PhysicsEngine.compute()`.
- Keep `PhysicsEngine.critical_density_veh_km_lane` instance-level when per-tick conditions can change.
- Treat `RoadCondition.warning == True` as authoritative for HALKA recommendations.
- Preserve `"type": "weather"` and `"type": "road_condition"` records when changing weather ingestion.

## Failure Signals

- Ice/snow changes free-flow speed but camera capacity remains capped at dry-road capacity.
- Queue predictions do not trigger earlier in degraded weather despite lower friction.
- Dashboard weather or road-condition tables go empty after tick-loop changes.
- HALKA VMS advisories only appear when a queue prediction also exists.

## Next Checklist

- Verify degraded-weather tests cover capacity cap, critical density, JSONL persistence, and standalone VMS advisories.
- Keep route-linear chainage mapping for any weather/road-condition record with geometry.
- Do not change YOLO confidence thresholds from weather data without labeled weather/night evaluation data.
