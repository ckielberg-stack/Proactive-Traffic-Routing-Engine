# 2026-06-06 - Route-Linear Chainage Datum

## Context

PTRE projects queue growth from camera bottlenecks to upstream camera nodes and VMS gantries. TRAFIK-003 replaced latitude-sorted camera chainage and nearest-latitude sensor matching with a shared route-linear datum for the E4 northbound corridor.

## What I Learned

Latitude is not a safe proxy for corridor distance. Closely spaced cameras can share nearly identical latitudes, and the road curves around interchanges, ramps, and bridges. If cameras, sensors, and VMS gantries use different spatial assumptions, the physics engine can compute wrong segment distances and attach local inflow to the wrong node.

The safer repo-local approach is to preserve the curated corridor order as an offline route reference, project lat/lng points onto that line in local metres, and scale the projection to the same 15.8 km chainage datum used by `vms_config.json`.

## Reuse Rules

- Keep camera chainage, sensor-to-node inflow mapping, sensor anomaly camera lookup, and VMS position matching on the same route-linear datum.
- Do not sort `CAMERA_COORDS` or sensor positions by latitude to infer corridor order.
- When adding new spatial consumers, prefer `src.route_chainage` projection helpers over ad hoc distance or latitude matching.
- Treat TravelTimeRoute spatial matching as incomplete until segment endpoints or geometry are available.

## Failure Signals

- Queue ETA changes unexpectedly after camera coordinate edits.
- Multiple nearby sensors map to a camera that is not adjacent in corridor order.
- VMS recommendations look geographically plausible but are wrong by chainage.
- Tests only verify north/south latitude order and miss curves or same-latitude positions.

## Next Checklist

- Before changing prediction, inspect whether all participants use route-linear chainage.
- Add same-latitude or out-of-order-latitude tests for new spatial mapping code.
- Revisit TravelTimeRoute accuracy matching once route segment geometry is available.
