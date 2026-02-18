"""
Incident report builder for Operator API state.

Converts anomaly-marked ``CapacityState`` entries into ``IncidentReport``
objects consumed by ``/api/v1/operator/active-incidents``.
"""

from __future__ import annotations

from src.models import CapacityState, IncidentReport

FREE_FLOW_PER_LANE_VPH: float = 2200.0


def build_incident_reports(
    capacity_states: list[CapacityState],
    camera_coords: dict[str, tuple[float, float]] | None = None,
) -> list[IncidentReport]:
    """Build incident reports from anomaly capacity states."""
    coords_map = camera_coords or {}
    incidents: list[IncidentReport] = []

    for state in capacity_states:
        if not state.is_anomaly:
            continue

        total_lanes = max(state.total_lanes, 1)
        baseline_capacity = total_lanes * FREE_FLOW_PER_LANE_VPH
        if baseline_capacity > 0:
            drop_pct = (1.0 - (state.estimated_capacity_vph / baseline_capacity)) * 100.0
        else:
            drop_pct = 0.0
        drop_pct = round(max(0.0, min(drop_pct, 100.0)), 1)

        coords = coords_map.get(state.camera_id)
        lat = coords[0] if coords else None
        lng = coords[1] if coords else None

        incidents.append(
            IncidentReport(
                timestamp=state.timestamp,
                camera_id=state.camera_id,
                incident_type=state.anomaly_reason or "congestion",
                lanes_affected=min(max(state.blocked_lanes, 0), total_lanes),
                total_lanes=total_lanes,
                capacity_drop_percentage=drop_pct,
                thumbnail_base64=None,
                confidence=state.confidence,
                lat=lat,
                lng=lng,
            )
        )

    return incidents
