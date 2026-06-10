"""Shared traffic-engineering constants used across PTRE modules."""

# Swedish motorway free-flow baseline on the E4 corridor.
FREE_FLOW_SPEED_KMH: float = 110.0

# Static theoretical maximum lane capacity. Keep this single source of truth
# so incident/evaluation baselines match the vision and physics pipeline.
Q_CAP_VPH_PER_LANE: float = 2_000.0
FREE_FLOW_PER_LANE_VPH: float = Q_CAP_VPH_PER_LANE

# Density thresholds for the LWR/vision handoff.
JAM_DENSITY_VEH_KM_LANE: float = 133.0
K_CRITICAL_VEH_KM_LANE: float = 45.0
