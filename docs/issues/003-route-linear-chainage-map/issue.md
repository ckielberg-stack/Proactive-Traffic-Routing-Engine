# Issue 003 - Replace Latitude-Derived Chainage With Route-Linear Chainage

## Problem

Camera chainage is currently derived by sorting latitude and interpolating over the corridor length. Sensor-to-camera inflow mapping also uses nearest latitude. This is approximate and can be wrong around curves, ramps, bridges, and closely spaced cameras.

## Impact

Queue-tail ETA quality depends on distance between bottlenecks, cameras, sensors, and VMS gantries. Bad chainage or sensor matching creates wrong segment distances, wrong local inflows, and wrong VMS timing.

## Suggested Approach

- Create a canonical route-linear map for cameras, sensors, VMS gantries, and TravelTimeRoute segments.
- Prefer a route polyline with map-matched projection distance over latitude interpolation.
- Store chainage explicitly in configuration or a generated checked-in mapping file.
- Update `build_camera_chainage_map()` and `build_node_inflows()` to use the route-linear reference.

## Acceptance Criteria

- Camera chainage is loaded from explicit route-linear data, not inferred from latitude.
- Sensors map to nearest upstream/downstream route position, not just nearest latitude.
- Tests cover out-of-order latitudes or closely spaced cameras.
- VMS ETA calculations use the same chainage datum as the physics engine.

## References

- `main_loop.py`
- `config.py`
- `vms_config.json`
