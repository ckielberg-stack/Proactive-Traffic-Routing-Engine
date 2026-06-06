# Issue 004 - Use Segment-Local Sensor Speeds And Inflows

## Problem

The physics engine supports node inflows, but upstream speed is still often global or fallback-based. TravelTimeRoute and TrafficFlow data are not yet fused into a consistent per-segment speed/inflow state.

## Impact

LWR wave speed depends on both flow and upstream density. Using global speed can overstate or understate queue growth in segments with ramps, localized slowdown, or partial recovery.

## Suggested Approach

- Build a per-segment traffic state from TrafficFlow station volumes, station speeds, and TravelTimeRoute speeds.
- Pass segment-local speed into the wave-speed calculation instead of only the aggregate sensor fallback.
- Handle missing sensor data with explicit confidence degradation rather than silent global averaging.
- Separate direction-aware station mapping from lane aggregation.

## Acceptance Criteria

- `PhysicsEngine` can compute each segment wave speed using local inflow and local speed.
- Missing local data is visible in prediction confidence or diagnostics.
- Unit tests show different local speeds produce different segment speeds.
- Tick logs expose how many segments used local data vs fallback data.

## References

- `main_loop.py`
- `src/physics_engine.py`
- `src/models.py`
- `config.py`
