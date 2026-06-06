# 2026-06-06 - Segment-Local Sensor Fusion

## Context

TRAFIK-004 fused TrafficFlow station volumes/speeds and TravelTimeRoute speeds into the LWR physics handoff. The goal was to stop using one aggregate speed for every upstream segment while keeping missing local data visible.

## What I Learned

TrafficFlow readings are the highest-confidence local source because station coordinates can be projected onto the same route-linear camera datum as physics and VMS. When multiple station or lane readings map to one camera node, inflow should be summed and speed should be volume-weighted when volume is positive.

TravelTimeRoute readings expose route IDs, names, lengths, and speeds, but this repo does not yet have route endpoint geometry. They are useful as a northbound, route-order speed fallback, not as station-local truth.

## Reuse Rules

- Prefer `SegmentTrafficState` over separate ad hoc `node_inflows` and `node_speeds` maps.
- Keep source labels on physics output whenever aggregate or TravelTime fallback data is used.
- Let TrafficFlow station speed override TravelTimeRoute speed for the same camera node.
- Do not blend southbound TravelTimeRoute readings into northbound physics.

## Failure Signals

- Segment wave speeds remain identical when adjacent upstream sensor speeds differ.
- Tick logs show only aggregate fallback data despite populated TrafficFlow station readings.
- TravelTimeRoute fallback appears high confidence or masks station-local slowdowns.
- Queue predictions lack source diagnostics for local vs fallback segment inputs.

## Next Checklist

- Before tuning LWR constants, inspect `SegmentSpeed.local_speed_kmh` and source fields.
- Add real TravelTimeRoute endpoint/geometry mapping before treating route speeds as segment-local.
- Include local/fallback/missing segment counts in replay evaluation metrics.
