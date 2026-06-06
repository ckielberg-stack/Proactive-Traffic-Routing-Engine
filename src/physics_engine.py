"""
Shockwave Prediction Engine (Phase 3) — Piecewise LWR Kinematic Wave Model.

Computes backward-propagating queue dynamics using the Lighthill–Whitham–
Richards (LWR) model with **multi-segment spatial iteration**.  Instead of
a single linear queue projection, the engine walks backward through camera
nodes segment-by-segment, using the local inflow at each node.

This naturally handles lateral sources (on-ramps) and sinks (off-ramps):
the observed inflow at Node N inherently includes any downstream ramp
traffic relative to Node N-1, without needing an explicit ramp database.

Key Formula (per segment)
-------------------------
    wave_speed = (Q_in - Q_cap) / (k_jam - k_in)

Where:
    Q_in   = local inflow at the upstream end of the segment (veh/h)
    Q_cap  = bottleneck capacity (veh/h) — the flow constraint
    k_jam  = jam density ≈ 133 veh/km/lane (Swedish standard)
    k_in   = inflow density = Q_in / v_in

Iteration halts when wave_speed ≤ 0 at a segment (queue stops growing
or dissolves upstream of an off-ramp / low-demand zone).

This engine is **stateless per tick**.
"""

from __future__ import annotations

import logging
from datetime import datetime

from src.models import (
    CapacityState,
    QueuePrediction,
    SegmentSpeed,
    SegmentTrafficState,
    SensorReading,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

#: Jam density in vehicles per km per lane (Swedish Transport Admin default).
JAM_DENSITY_VEH_KM_LANE: float = 133.0

#: Free-flow speed on the E4 motorway (km/h).
FREE_FLOW_SPEED_KMH: float = 110.0

#: Default time horizons to project queue length (minutes).
DEFAULT_TIME_HORIZONS: list[int] = [1, 3, 5, 10]

#: Minimum capacity drop (VPH) to consider a segment a "bottleneck".
#: Below this threshold, the delta is too small for meaningful predictions.
MIN_CAPACITY_DROP_VPH: float = 200.0

# --- Expert Audit Fix 1: Flow vs Capacity ---
#: Critical density threshold (veh/km/lane).  Bottleneck evaluation only
#: triggers when the vision engine's observed density exceeds this value.
K_CRITICAL_VEH_KM_LANE: float = 45.0


# ---------------------------------------------------------------------------
# Physics Engine
# ---------------------------------------------------------------------------


class PhysicsEngine:
    """Piecewise LWR shockwave calculator.

    Call :meth:`compute` once per tick with the current vision and sensor
    data.  Returns a list of ``QueuePrediction`` — one for each detected
    bottleneck — with per-segment wave speeds.
    """

    def __init__(
        self,
        jam_density: float = JAM_DENSITY_VEH_KM_LANE,
        free_flow_speed: float = FREE_FLOW_SPEED_KMH,
        time_horizons: list[int] | None = None,
    ) -> None:
        self.jam_density = jam_density
        self.free_flow_speed = free_flow_speed
        self.time_horizons = time_horizons or DEFAULT_TIME_HORIZONS

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(
        self,
        capacity_states: list[CapacityState],
        sensor: SensorReading | None,
        camera_chainage_map: dict[str, float] | None = None,
        camera_coords_map: dict[str, tuple[float, float]] | None = None,
        now: datetime | None = None,
        node_inflows: dict[str, float] | None = None,
        node_traffic_states: dict[str, SegmentTrafficState] | None = None,
    ) -> list[QueuePrediction]:
        """Evaluate all capacity states and return queue predictions.

        Parameters
        ----------
        capacity_states:
            Vision engine outputs for this tick.
        sensor:
            Global upstream sensor reading (fallback inflow).
        camera_chainage_map:
            ``{camera_id: chainage_km}`` — linear reference for each camera.
        camera_coords_map:
            ``{camera_id: (lat, lng)}`` — geographic coordinates.
        now:
            Timestamp to stamp the output.  Defaults to ``datetime.now()``.
        node_inflows:
            ``{camera_id: measured_volume_vph}`` — per-node inflow
            estimates from sensors or visual fallback.  If a node is
            missing, falls back to the global ``sensor`` reading.
        node_traffic_states:
            ``{camera_id: SegmentTrafficState}`` — per-node inflow/speed
            estimates with source/confidence diagnostics. Takes precedence
            over ``node_inflows``.
        """
        now = now or datetime.now()
        camera_chainage_map = camera_chainage_map or {}
        camera_coords_map = camera_coords_map or {}
        node_inflows = node_inflows or {}
        node_traffic_states = node_traffic_states or {
            camera_id: SegmentTrafficState(
                local_inflow_vph=inflow_vph,
                inflow_source="traffic_flow",
                confidence="medium",
            )
            for camera_id, inflow_vph in node_inflows.items()
        }

        predictions: list[QueuePrediction] = []

        if sensor is None and not node_traffic_states:
            logger.debug("No sensor data — skipping physics computation")
            return predictions

        # Build sorted node list: (chainage_km, camera_id) ascending
        sorted_nodes = sorted(camera_chainage_map.items(), key=lambda x: x[1])

        for state in capacity_states:
            # Expert Audit Fix 1: trigger on density, not is_anomaly
            if state.observed_density_veh_km_lane < K_CRITICAL_VEH_KM_LANE:
                continue  # Density below critical — no congestion breakdown

            prediction = self._evaluate_bottleneck_piecewise(
                state=state,
                sensor=sensor,
                sorted_nodes=sorted_nodes,
                coords_map=camera_coords_map,
                node_traffic_states=node_traffic_states,
                now=now,
            )
            if prediction is not None:
                predictions.append(prediction)

        return predictions

    # ------------------------------------------------------------------
    # Internal — Piecewise iteration
    # ------------------------------------------------------------------

    def _evaluate_bottleneck_piecewise(
        self,
        state: CapacityState,
        sensor: SensorReading | None,
        sorted_nodes: list[tuple[str, float]],
        coords_map: dict[str, tuple[float, float]],
        node_traffic_states: dict[str, SegmentTrafficState],
        now: datetime,
    ) -> QueuePrediction | None:
        """Run piecewise LWR calculation for a single bottleneck.

        Walks backward (upstream) from the bottleneck through camera
        nodes, computing per-segment wave speeds using local inflow
        at each node.

        Iteration halts when:
        - We run out of upstream nodes
        - Wave speed becomes ≤ 0 (queue stops growing)
        """
        bottleneck_capacity_vph = state.estimated_capacity_vph
        total_lanes = max(state.total_lanes, 1)

        # Global sensor fallback values
        global_inflow = sensor.inflow_volume_vph if sensor else None
        global_speed = sensor.average_speed_kmh if sensor else self.free_flow_speed
        global_inflow_source = "aggregate" if sensor else "missing"
        global_speed_source = "aggregate" if sensor else "model_free_flow"

        # First check: does inflow even exceed capacity?
        # Use per-node inflow for the bottleneck camera, or global sensor
        bottleneck_inflow, bottleneck_inflow_source = self._resolve_local_inflow(
            state.camera_id,
            node_traffic_states,
            global_inflow,
            global_inflow_source,
        )
        bottleneck_speed, bottleneck_speed_source = self._resolve_local_speed(
            state.camera_id,
            node_traffic_states,
            global_speed,
            global_speed_source,
        )
        if bottleneck_inflow is None:
            logger.debug(
                f"Camera {state.camera_id}: no inflow data available"
            )
            return None

        capacity_drop = bottleneck_inflow - bottleneck_capacity_vph
        if capacity_drop < MIN_CAPACITY_DROP_VPH:
            logger.debug(
                f"Camera {state.camera_id}: capacity drop {capacity_drop:.0f} VPH "
                f"below threshold ({MIN_CAPACITY_DROP_VPH:.0f})"
            )
            return None

        # Find bottleneck position in sorted node list
        bottleneck_idx = None
        for i, (cam_id, _ch) in enumerate(sorted_nodes):
            if cam_id == state.camera_id:
                bottleneck_idx = i
                break

        # --- Determine upstream direction ---
        # For northbound traffic (Södertälje → Stockholm): upstream = south
        #   = decreasing chainage → iterate backward (idx - 1, idx - 2, ...)
        # For southbound traffic: upstream = north
        #   = increasing chainage → iterate forward (idx + 1, idx + 2, ...)
        # Default to northbound if no direction info available.
        direction = "northbound"  # TODO: derive from CapacityState/ROI road_id
        upstream_step = -1 if direction == "northbound" else 1

        # --- Piecewise backward iteration ---
        segment_speeds: list[SegmentSpeed] = []
        local_data_segments = 0
        fallback_data_segments = 0
        missing_data_segments = 0

        if bottleneck_idx is not None and sorted_nodes:
            prev_cam_id = state.camera_id
            prev_chainage = sorted_nodes[bottleneck_idx][1]
            idx = bottleneck_idx + upstream_step

            while 0 <= idx < len(sorted_nodes):
                upstream_cam_id, upstream_chainage = sorted_nodes[idx]
                distance_km = abs(prev_chainage - upstream_chainage)

                if distance_km < 0.001:
                    # Skip degenerate zero-distance segments
                    idx += upstream_step
                    continue

                # Get local inflow and speed at this upstream node.
                local_inflow, inflow_source = self._resolve_local_inflow(
                    upstream_cam_id,
                    node_traffic_states,
                    global_inflow,
                    global_inflow_source,
                )
                if local_inflow is None:
                    missing_data_segments += 1
                    break  # No inflow data — can't continue
                local_speed, speed_source = self._resolve_local_speed(
                    upstream_cam_id,
                    node_traffic_states,
                    global_speed,
                    global_speed_source,
                )

                # Compute segment wave speed
                wave_speed = self._lwr_wave_speed(
                    inflow_vph=local_inflow,
                    bottleneck_capacity_vph=bottleneck_capacity_vph,
                    upstream_speed_kmh=local_speed,
                    num_lanes=total_lanes,
                )

                # GUARD: if wave speed ≤ 0, queue stops growing here
                if wave_speed <= 0:
                    logger.debug(
                        f"Segment {prev_cam_id}→{upstream_cam_id}: "
                        f"wave_speed={wave_speed:.2f} ≤ 0, halting iteration"
                    )
                    break

                segment_speeds.append(SegmentSpeed(
                    from_camera=prev_cam_id,
                    to_camera=upstream_cam_id,
                    distance_km=round(distance_km, 4),
                    wave_speed_kmh=round(wave_speed, 2),
                    local_inflow_vph=local_inflow,
                    local_speed_kmh=round(local_speed, 1),
                    inflow_source=inflow_source,
                    speed_source=speed_source,
                ))
                if inflow_source == "traffic_flow" and speed_source == "traffic_flow":
                    local_data_segments += 1
                else:
                    fallback_data_segments += 1

                prev_cam_id = upstream_cam_id
                prev_chainage = upstream_chainage
                idx += upstream_step

        # --- Compute lengths_at_minutes via time accumulation ---
        lengths_at_minutes = self._compute_lengths_piecewise(
            segment_speeds, self.time_horizons
        )

        # --- Weighted average growth_speed for backward compat ---
        if segment_speeds:
            total_dist = sum(s.distance_km for s in segment_speeds)
            if total_dist > 0:
                avg_speed = total_dist / sum(
                    s.distance_km / s.wave_speed_kmh
                    for s in segment_speeds
                )
            else:
                avg_speed = segment_speeds[0].wave_speed_kmh
        else:
            # No segments — fall back to simple single-segment calculation
            avg_speed = self._lwr_wave_speed(
                inflow_vph=bottleneck_inflow,
                bottleneck_capacity_vph=bottleneck_capacity_vph,
                upstream_speed_kmh=bottleneck_speed,
                num_lanes=total_lanes,
            )
            if avg_speed <= 0:
                return None
            if (
                bottleneck_inflow_source == "traffic_flow"
                and bottleneck_speed_source == "traffic_flow"
            ):
                local_data_segments = 1
            else:
                fallback_data_segments = 1
            # Simple linear projection (legacy behavior)
            lengths_at_minutes = {
                t: round(avg_speed * (t / 60), 3)
                for t in self.time_horizons
            }

        data_confidence = self._data_confidence(
            local_data_segments,
            fallback_data_segments,
            missing_data_segments,
        )

        coords = coords_map.get(state.camera_id, (0.0, 0.0))
        chainage_km = 0.0
        for cam_id, ch in sorted_nodes:
            if cam_id == state.camera_id:
                chainage_km = ch
                break

        logger.info(
            f"🌊 Piecewise prediction @ {state.camera_id}: "
            f"segments={len(segment_speeds)}, "
            f"local={local_data_segments}, "
            f"fallback={fallback_data_segments}, "
            f"missing={missing_data_segments}, "
            f"confidence={data_confidence}, "
            f"avg_wave={avg_speed:.1f} km/h, "
            f"Q+5min={lengths_at_minutes.get(5, 0):.2f} km"
        )

        return QueuePrediction(
            timestamp=now,
            camera_id=state.camera_id,
            origin_lat=coords[0],
            origin_lng=coords[1],
            origin_chainage_km=chainage_km,
            growth_speed_kmh=round(avg_speed, 2),
            segment_speeds=segment_speeds,
            lengths_at_minutes=lengths_at_minutes,
            local_data_segments=local_data_segments,
            fallback_data_segments=fallback_data_segments,
            missing_data_segments=missing_data_segments,
            data_confidence=data_confidence,
        )

    @staticmethod
    def _resolve_local_inflow(
        camera_id: str,
        node_traffic_states: dict[str, SegmentTrafficState],
        global_inflow: float | None,
        global_inflow_source: str,
    ) -> tuple[float | None, str]:
        traffic_state = node_traffic_states.get(camera_id)
        if traffic_state and traffic_state.local_inflow_vph is not None:
            return traffic_state.local_inflow_vph, traffic_state.inflow_source
        return global_inflow, global_inflow_source

    @staticmethod
    def _resolve_local_speed(
        camera_id: str,
        node_traffic_states: dict[str, SegmentTrafficState],
        global_speed: float,
        global_speed_source: str,
    ) -> tuple[float, str]:
        traffic_state = node_traffic_states.get(camera_id)
        if traffic_state and traffic_state.local_speed_kmh is not None:
            return traffic_state.local_speed_kmh, traffic_state.speed_source
        return global_speed, global_speed_source

    @staticmethod
    def _data_confidence(
        local_data_segments: int,
        fallback_data_segments: int,
        missing_data_segments: int,
    ) -> str:
        if missing_data_segments > 0:
            return "low"
        if fallback_data_segments > 0:
            return "medium"
        if local_data_segments > 0:
            return "high"
        return "low"

    # ------------------------------------------------------------------
    # Piecewise time-distance accumulation
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_lengths_piecewise(
        segments: list[SegmentSpeed],
        time_horizons: list[int],
    ) -> dict[int, float]:
        """Compute queue lengths at each time horizon via segment accumulation.

        Walks through segments, summing transit time per segment.  For each
        time horizon T, interpolates within the segment where T is reached.
        """
        if not segments:
            return {}

        # Pre-compute cumulative time and distance through segments
        cum_time_h: list[float] = []   # Cumulative hours to traverse each segment
        cum_dist_km: list[float] = []  # Cumulative distance
        total_time = 0.0
        total_dist = 0.0

        for seg in segments:
            seg_time_h = seg.distance_km / seg.wave_speed_kmh  # hours
            total_time += seg_time_h
            total_dist += seg.distance_km
            cum_time_h.append(total_time)
            cum_dist_km.append(total_dist)

        result: dict[int, float] = {}
        for t_min in time_horizons:
            t_hours = t_min / 60.0
            if t_hours <= 0:
                result[t_min] = 0.0
                continue

            # Find how far the queue extends in t_hours
            queue_km = 0.0
            time_used = 0.0

            for i, seg in enumerate(segments):
                seg_time_h = seg.distance_km / seg.wave_speed_kmh
                remaining_time = t_hours - time_used

                if seg_time_h <= remaining_time:
                    # Queue traverses this entire segment
                    queue_km += seg.distance_km
                    time_used += seg_time_h
                else:
                    # Queue partially traverses this segment — interpolate
                    fraction = remaining_time / seg_time_h
                    queue_km += seg.distance_km * fraction
                    break
            else:
                # Queue extends beyond all known segments — extrapolate
                # using the last segment's wave speed
                remaining_time = t_hours - time_used
                if remaining_time > 0 and segments:
                    queue_km += segments[-1].wave_speed_kmh * remaining_time

            result[t_min] = round(queue_km, 3)

        return result

    # ------------------------------------------------------------------
    # LWR core formula (unchanged)
    # ------------------------------------------------------------------

    def _lwr_wave_speed(
        self,
        inflow_vph: float,
        bottleneck_capacity_vph: float,
        upstream_speed_kmh: float,
        num_lanes: int,
    ) -> float:
        """Compute the LWR shockwave propagation speed.

        Formula::

            w = (Q_in - Q_cap) / (k_jam - k_in)

        where k_in = Q_in / v_in (per lane), k_jam is the jam density
        (per lane), and all volumes are per lane.

        Returns the wave speed in km/h.  Positive = queue growing upstream.
        """
        # Convert to per-lane values
        q_in_per_lane = inflow_vph / num_lanes
        q_cap_per_lane = bottleneck_capacity_vph / num_lanes

        # Inflow density (veh/km/lane)
        if upstream_speed_kmh > 0:
            k_in = q_in_per_lane / upstream_speed_kmh
        else:
            # If speed is zero, use jam density
            k_in = self.jam_density

        # Denominator: jam_density - inflow_density
        denominator = self.jam_density - k_in
        if denominator <= 0:
            # Already at jam density — physically impossible to grow faster
            # Return a high but bounded wave speed
            logger.warning(
                f"Inflow density ({k_in:.1f} veh/km/lane) ≥ jam density "
                f"({self.jam_density:.1f}) — capping wave speed"
            )
            return self.free_flow_speed * 0.5  # Heuristic cap

        # Numerator: inflow volume - bottleneck capacity (per lane)
        numerator = q_in_per_lane - q_cap_per_lane

        wave_speed = numerator / denominator
        return max(wave_speed, 0.0)  # Don't return negative (queue dissolving)
