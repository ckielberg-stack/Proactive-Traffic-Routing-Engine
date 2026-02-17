"""
Shockwave Prediction Engine (Phase 3) — LWR Kinematic Wave Model.

Computes backward-propagating queue dynamics using the Lighthill–Whitham–
Richards (LWR) model.  Each tick receives the *current* bottleneck capacity
(from the vision engine) and the *current* upstream inflow (from the sensor
API) and calculates the instantaneous shockwave speed.

Key Formula
-----------
    wave_speed = (Q_in - Q_cap) / (k_jam - k_in)

Where:
    Q_in   = upstream inflow volume (veh/h)
    Q_cap  = bottleneck capacity (veh/h)
    k_jam  = jam density ≈ 133 veh/km/lane (Swedish standard)
    k_in   = inflow density = Q_in / v_in

A *positive* wave speed means the queue tail is propagating upstream
(i.e. the queue is growing).  Zero means the queue is stationary.
A negative value means the queue is dissolving.

This engine is **stateless per tick**: it does not maintain cross-tick
memory.  Historical trend analysis can be done offline against the
persisted JSONL data.
"""

from __future__ import annotations

import logging
from datetime import datetime

from src.models import CapacityState, QueuePrediction, SensorReading

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


# ---------------------------------------------------------------------------
# Physics Engine
# ---------------------------------------------------------------------------


class PhysicsEngine:
    """Stateless LWR shockwave calculator.

    Call :meth:`compute` once per tick with the current vision and sensor
    data.  Returns a list of ``QueuePrediction`` — one for each detected
    bottleneck.
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
    ) -> list[QueuePrediction]:
        """Evaluate all capacity states and return queue predictions.

        Parameters
        ----------
        capacity_states:
            Vision engine outputs for this tick.
        sensor:
            Upstream sensor reading (inflow volume + speed).
        camera_chainage_map:
            ``{camera_id: chainage_km}`` — linear reference for each camera.
        camera_coords_map:
            ``{camera_id: (lat, lng)}`` — geographic coordinates.
        now:
            Timestamp to stamp the output.  Defaults to ``datetime.now()``.
        """
        now = now or datetime.now()
        camera_chainage_map = camera_chainage_map or {}
        camera_coords_map = camera_coords_map or {}

        predictions: list[QueuePrediction] = []

        if sensor is None:
            logger.debug("No sensor data — skipping physics computation")
            return predictions

        for state in capacity_states:
            if not state.is_anomaly:
                continue  # No bottleneck detected by vision

            prediction = self._evaluate_bottleneck(
                state, sensor, camera_chainage_map, camera_coords_map, now,
            )
            if prediction is not None:
                predictions.append(prediction)

        return predictions

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _evaluate_bottleneck(
        self,
        state: CapacityState,
        sensor: SensorReading,
        chainage_map: dict[str, float],
        coords_map: dict[str, tuple[float, float]],
        now: datetime,
    ) -> QueuePrediction | None:
        """Run LWR calculation for a single bottleneck camera."""

        inflow_vph = sensor.inflow_volume_vph
        bottleneck_capacity_vph = state.estimated_capacity_vph
        total_lanes = max(state.total_lanes, 1)

        # Check if the capacity drop is significant enough
        capacity_drop = inflow_vph - bottleneck_capacity_vph
        if capacity_drop < MIN_CAPACITY_DROP_VPH:
            logger.debug(
                f"Camera {state.camera_id}: capacity drop {capacity_drop:.0f} VPH "
                f"below threshold ({MIN_CAPACITY_DROP_VPH:.0f})"
            )
            return None

        # Compute wave speed using LWR formula
        wave_speed = self._lwr_wave_speed(
            inflow_vph=inflow_vph,
            bottleneck_capacity_vph=bottleneck_capacity_vph,
            upstream_speed_kmh=sensor.average_speed_kmh,
            num_lanes=total_lanes,
        )

        if wave_speed <= 0:
            logger.debug(
                f"Camera {state.camera_id}: wave speed {wave_speed:.2f} km/h "
                f"(non-positive → queue not growing)"
            )
            return None

        # Project queue length at each time horizon
        lengths_at_minutes = {
            t: round(wave_speed * (t / 60), 3)  # km = (km/h) × (min/60)
            for t in self.time_horizons
        }

        chainage_km = chainage_map.get(state.camera_id, 0.0)
        coords = coords_map.get(state.camera_id, (0.0, 0.0))

        logger.info(
            f"🌊 Queue prediction @ {state.camera_id}: "
            f"wave={wave_speed:.1f} km/h, "
            f"Q+5min={lengths_at_minutes.get(5, 0):.2f} km"
        )

        return QueuePrediction(
            timestamp=now,
            camera_id=state.camera_id,
            origin_lat=coords[0],
            origin_lng=coords[1],
            origin_chainage_km=chainage_km,
            growth_speed_kmh=round(wave_speed, 2),
            lengths_at_minutes=lengths_at_minutes,
        )

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
