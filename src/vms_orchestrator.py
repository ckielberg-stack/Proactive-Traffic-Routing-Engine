"""
VMS & Queue Tail Predictor (Phase 5).

Predicts when the backward-propagating tail of a traffic jam will reach
upstream Variable Message Sign (VMS) gantries.  Generates preemptive
activation recommendations so operators can display speed warnings
(e.g. "KÖVARNING 70 km/h") *before* drivers encounter the queue.

Algorithm
---------
1. Load VMS gantry positions from ``vms_config.json`` (chainage-based).
2. Given a ``QueuePrediction`` from the Physics Engine, project the queue
   tail position at T+1, T+3, T+5 minutes using linear extrapolation.
3. For each time step, find the nearest VMS gantry that is ≥1000 m
   upstream of the predicted queue tail.
4. Produce ``VMSRecommendation`` objects with message text, urgency, and
   estimated time before the queue reaches the sign.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

from config import E4_NORTHBOUND_CORRIDOR_LENGTH_KM, E4_NORTHBOUND_ROUTE_POINTS
from src.models import (
    QueuePrediction,
    SensorAnomaly,
    VMSGantry,
    VMSRecommendation,
    VMSStatusSnapshot,
)
from src.route_chainage import RouteProjector

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Minimum distance (km) the VMS must be *upstream* of the queue tail
#: to allow drivers enough reaction time.
MIN_UPSTREAM_DISTANCE_KM: float = 1.0

#: Default time horizons to evaluate (minutes).
DEFAULT_TIME_HORIZONS: list[int] = [1, 3, 5]

#: Default VMS config path relative to project root.
DEFAULT_VMS_CONFIG: Path = Path(__file__).resolve().parent.parent / "vms_config.json"


# ---------------------------------------------------------------------------
# VMS Orchestrator
# ---------------------------------------------------------------------------


class VMSOrchestrator:
    """Predicts queue tail reach and recommends VMS activations.

    Parameters
    ----------
    config_path:
        Path to the ``vms_config.json`` file.  Defaults to project root.
    """

    def __init__(
        self,
        config_path: Path | str | None = None,
        route_points: Sequence[tuple[float, float]] | None = None,
        corridor_length_km: float = E4_NORTHBOUND_CORRIDOR_LENGTH_KM,
    ) -> None:
        self._config_path = Path(config_path) if config_path else DEFAULT_VMS_CONFIG
        self._route_points = (
            list(route_points)
            if route_points is not None
            else E4_NORTHBOUND_ROUTE_POINTS
        )
        self._corridor_length_km = corridor_length_km
        self._gantries: list[VMSGantry] = []
        self._load_config()

    # ------------------------------------------------------------------
    # Config loading
    # ------------------------------------------------------------------

    def _load_config(self) -> None:
        """Parse ``vms_config.json`` into a sorted list of ``VMSGantry``."""
        if not self._config_path.exists():
            logger.warning("VMS config not found at %s — no gantries loaded", self._config_path)
            return

        with open(self._config_path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)

        gantries_raw = raw.get("gantries", [])
        self._gantries = [VMSGantry(**g) for g in gantries_raw]
        # Sort by chainage ascending (south → north along E4)
        self._gantries.sort(key=lambda g: g.chainage_km)
        logger.info("Loaded %d VMS gantries from %s", len(self._gantries), self._config_path)

    @property
    def gantries(self) -> list[VMSGantry]:
        return list(self._gantries)

    # ------------------------------------------------------------------
    # Queue tail projection
    # ------------------------------------------------------------------

    @staticmethod
    def predict_queue_tail_chainage(
        prediction: QueuePrediction,
        time_minutes: int,
    ) -> float:
        """Calculate the chainage (km) of the queue tail at T+``time_minutes``.

        When piecewise segment data is available, walks backward through
        segments accumulating time.  Otherwise falls back to the legacy
        linear ``growth_speed_kmh`` extrapolation.

        Parameters
        ----------
        prediction:
            Physics engine output describing the bottleneck.
        time_minutes:
            Future offset in minutes.

        Returns
        -------
        float
            Chainage (km) of the predicted queue tail.
        """
        # Use lengths_at_minutes if the exact horizon is pre-computed
        if time_minutes in prediction.lengths_at_minutes:
            distance_km = prediction.lengths_at_minutes[time_minutes]
            return prediction.origin_chainage_km - distance_km

        # Fallback: linear extrapolation with weighted-average speed
        distance_km = prediction.growth_speed_kmh * (time_minutes / 60.0)
        return prediction.origin_chainage_km - distance_km

    # ------------------------------------------------------------------
    # Upstream VMS selection
    # ------------------------------------------------------------------

    def find_upstream_vms(
        self,
        queue_tail_chainage_km: float,
    ) -> VMSGantry | None:
        """Find the nearest VMS that is ≥ ``MIN_UPSTREAM_DISTANCE_KM`` ahead
        of (i.e. lower chainage than) the queue tail.

        "Upstream" means *before* the queue tail in the direction of travel,
        i.e. at a *lower* chainage on the northbound E4.  Drivers approaching
        from the south see this VMS before they reach the queue.

        Returns ``None`` if no qualifying gantry exists.
        """
        threshold = queue_tail_chainage_km - MIN_UPSTREAM_DISTANCE_KM
        candidates = [g for g in self._gantries if g.chainage_km <= threshold]
        if not candidates:
            return None
        # Nearest upstream = highest chainage among candidates
        return max(candidates, key=lambda g: g.chainage_km)

    def find_nearest_vms_by_lat(
        self,
        lat: float,
    ) -> VMSGantry | None:
        """Find the VMS gantry nearest to a given latitude.

        Used for sensor-based anomalies where we don't have a chainage
        but do have station coordinates.

        Returns ``None`` if no gantries are loaded.
        """
        if not self._gantries:
            return None
        return min(self._gantries, key=lambda g: abs(g.lat - lat))

    def find_nearest_vms_by_chainage(
        self,
        chainage_km: float,
    ) -> VMSGantry | None:
        """Find the VMS gantry nearest to a route-linear chainage."""
        if not self._gantries:
            return None
        return min(self._gantries, key=lambda g: abs(g.chainage_km - chainage_km))

    def find_nearest_vms_by_position(
        self,
        lat: float,
        lng: float,
    ) -> VMSGantry | None:
        """Find the VMS nearest to a lat/lng position on the route datum."""
        projector = RouteProjector(
            self._route_points,
            self._corridor_length_km,
        )
        chainage_km = projector.project_chainage((lat, lng))
        if chainage_km is None:
            return None
        return self.find_nearest_vms_by_chainage(chainage_km)

    # ------------------------------------------------------------------
    # Recommendation generation
    # ------------------------------------------------------------------

    def generate_recommendations(
        self,
        prediction: QueuePrediction,
        time_horizons: list[int] | None = None,
        now: datetime | None = None,
        vms_statuses: list[VMSStatusSnapshot] | None = None,
        surface_state: str | None = None,
    ) -> list[VMSRecommendation]:
        """Produce VMS activation recommendations for each time horizon.

        Parameters
        ----------
        prediction:
            Physics engine queue prediction.
        time_horizons:
            List of T+N minute offsets to evaluate.  Defaults to [1, 3, 5].
        now:
            Override for current timestamp (useful in tests).
        vms_statuses:
            Current polled VMS sign statuses — used to include real-world
            sign state in recommendations and narrative summaries.

        Returns
        -------
        list[VMSRecommendation]
            One recommendation per triggered VMS (deduplicated by vms_id).
        """
        if now is None:
            now = datetime.now()
        if time_horizons is None:
            time_horizons = DEFAULT_TIME_HORIZONS

        # Build a lookup for current VMS statuses
        status_lookup: dict[str, VMSStatusSnapshot] = {}
        if vms_statuses:
            for s in vms_statuses:
                status_lookup[s.vms_id] = s

        seen_vms: set[str] = set()
        recommendations: list[VMSRecommendation] = []

        for t_min in sorted(time_horizons):
            tail_km = self.predict_queue_tail_chainage(prediction, t_min)
            vms = self.find_upstream_vms(tail_km)

            if vms is None or vms.vms_id in seen_vms:
                continue

            seen_vms.add(vms.vms_id)

            distance_to_vms = tail_km - vms.chainage_km
            if prediction.growth_speed_kmh > 0:
                eta_minutes = (distance_to_vms / prediction.growth_speed_kmh) * 60.0
            else:
                eta_minutes = float("inf")

            urgency = _classify_urgency(eta_minutes)
            message = _build_message(urgency, surface_state=surface_state)

            # Resolve current VMS sign state (None = not polled)
            current_status = status_lookup.get(vms.vms_id)
            current_status_str = (
                current_status.displayed_message if current_status.is_active else "OFF"
            ) if current_status is not None else None

            # Build operator-facing narrative summary
            summary = _build_narrative(
                prediction=prediction,
                vms=vms,
                eta_minutes=eta_minutes,
                current_status_str=current_status_str,
            )

            recommendations.append(
                VMSRecommendation(
                    timestamp=now,
                    vms_id=vms.vms_id,
                    vms_name=vms.name,
                    recommended_message=message,
                    urgency=urgency,
                    queue_growth_speed_kmh=prediction.growth_speed_kmh,
                    distance_queue_tail_to_vms_km=round(distance_to_vms, 2),
                    estimated_activation_minutes=round(eta_minutes, 1),
                    triggering_camera_id=prediction.camera_id,
                    current_vms_status=current_status_str,
                    summary=summary,
                )
            )

        return recommendations

    # ------------------------------------------------------------------
    # Sensor-based VMS recommendations
    # ------------------------------------------------------------------

    def generate_sensor_recommendations(
        self,
        anomalies: list[SensorAnomaly],
        now: datetime | None = None,
        vms_statuses: list[VMSStatusSnapshot] | None = None,
    ) -> list[VMSRecommendation]:
        """Generate VMS warnings for sensor-detected speed drops.

        Unlike queue-tail predictions (which forecast *future* congestion),
        sensor anomalies represent *current* measured congestion.  These
        recommendations are therefore always ``urgency='immediate'``.

        Only ``severity='severe'`` anomalies trigger VMS recommendations.
        ``severity='warning'`` anomalies are logged to the anomaly store
        for operator awareness but do not generate VMS activation requests.

        Parameters
        ----------
        anomalies:
            Sensor anomalies from the current tick.
        now:
            Override for current timestamp (useful in tests).
        vms_statuses:
            Current polled VMS sign statuses.

        Returns
        -------
        list[VMSRecommendation]
            One recommendation per triggered VMS (deduplicated by vms_id).
        """
        if now is None:
            now = datetime.now()

        # Build a lookup for current VMS statuses
        status_lookup: dict[str, VMSStatusSnapshot] = {}
        if vms_statuses:
            for s in vms_statuses:
                status_lookup[s.vms_id] = s

        seen_vms: set[str] = set()
        recommendations: list[VMSRecommendation] = []

        for anomaly in anomalies:
            # Only severe anomalies trigger VMS recommendations
            if anomaly.severity != "severe":
                continue

            # Find the nearest VMS to this sensor station on the route datum.
            vms = self.find_nearest_vms_by_position(anomaly.lat, anomaly.lng)
            if vms is None or vms.vms_id in seen_vms:
                continue

            seen_vms.add(vms.vms_id)

            # Resolve current VMS sign state
            current_status = status_lookup.get(vms.vms_id)
            current_status_str = (
                current_status.displayed_message if current_status.is_active else "OFF"
            ) if current_status is not None else None

            summary = _build_sensor_narrative(anomaly, vms, current_status_str)

            recommendations.append(
                VMSRecommendation(
                    timestamp=now,
                    vms_id=vms.vms_id,
                    vms_name=vms.name,
                    recommended_message="KÖVARNING 50 km/h",
                    urgency="immediate",
                    queue_growth_speed_kmh=0.0,  # Not a prediction — measured now
                    distance_queue_tail_to_vms_km=0.0,
                    estimated_activation_minutes=0.0,
                    triggering_camera_id=anomaly.nearest_camera_id or f"sensor_{anomaly.site_id}",
                    current_vms_status=current_status_str,
                    summary=summary,
                )
            )

        return recommendations

    # ------------------------------------------------------------------
    # Weather/RoadCondition-based VMS recommendations
    # ------------------------------------------------------------------

    def generate_weather_recommendations(
        self,
        road_conditions: list[dict],
        now: datetime | None = None,
        vms_statuses: list[VMSStatusSnapshot] | None = None,
    ) -> list[VMSRecommendation]:
        """Generate standalone HALKA advisories for active RoadCondition warnings."""
        if now is None:
            now = datetime.now()

        status_lookup: dict[str, VMSStatusSnapshot] = {}
        if vms_statuses:
            for status in vms_statuses:
                status_lookup[status.vms_id] = status

        seen_vms: set[str] = set()
        recommendations: list[VMSRecommendation] = []

        for condition in road_conditions:
            if condition.get("warning") is not True:
                continue

            vms = self._find_weather_warning_vms(condition)
            if vms is None or vms.vms_id in seen_vms:
                continue

            seen_vms.add(vms.vms_id)
            current_status = status_lookup.get(vms.vms_id)
            current_status_str = (
                current_status.displayed_message if current_status.is_active else "OFF"
            ) if current_status is not None else None
            summary = _build_weather_narrative(condition, vms, current_status_str)

            recommendations.append(
                VMSRecommendation(
                    timestamp=now,
                    vms_id=vms.vms_id,
                    vms_name=vms.name,
                    recommended_message="HALKA - VARNING",
                    urgency="immediate",
                    queue_growth_speed_kmh=0.0,
                    distance_queue_tail_to_vms_km=0.0,
                    estimated_activation_minutes=0.0,
                    triggering_camera_id=(
                        f"road_condition_{condition.get('id')}"
                        if condition.get("id")
                        else "road_condition"
                    ),
                    current_vms_status=current_status_str,
                    summary=summary,
                )
            )

        return recommendations

    def _find_weather_warning_vms(self, condition: dict) -> VMSGantry | None:
        chainage = condition.get("chainage_km")
        if chainage is not None:
            try:
                return self.find_nearest_vms_by_chainage(float(chainage))
            except (TypeError, ValueError):
                pass

        lat = condition.get("lat")
        lng = condition.get("lng")
        if lat is not None and lng is not None:
            try:
                vms = self.find_nearest_vms_by_position(float(lat), float(lng))
            except (TypeError, ValueError):
                vms = None
            if vms is not None:
                return vms

        e4_gantries = [gantry for gantry in self._gantries if gantry.road == "E4"]
        return e4_gantries[0] if e4_gantries else None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _classify_urgency(eta_minutes: float) -> str:
    """Map ETA to urgency level."""
    if eta_minutes <= 2.0:
        return "immediate"
    if eta_minutes <= 5.0:
        return "soon"
    return "advisory"


def _build_message(urgency: str, surface_state: str | None = None) -> str:
    """Generate the Swedish VMS display message."""
    messages = {
        "immediate": "KÖVARNING 50 km/h",
        "soon": "KÖVARNING 70 km/h",
    }
    message = messages.get(urgency, "VARNING - Köbildning framför")
    if surface_state in {"snow", "ice"}:
        return f"HALKA - {message}"
    return message


def _build_narrative(
    prediction: QueuePrediction,
    vms: VMSGantry,
    eta_minutes: float,
    current_status_str: str | None,
) -> str:
    """Generate a physics-driven operator narrative.

    Example output::

        Incident verified. Queue tail growing backwards at 12 km/h.
        Queue will reach VMS-4003 (Kungens Kurva) in 6.5 min.
        Current VMS status: OFF.
        Recommendation: Activate 70 km/h warning in 4 min.
    """
    parts: list[str] = []

    # 1. Queue growth info
    parts.append(
        f"Kö växer bakåt med {prediction.growth_speed_kmh:.0f} km/h."
    )

    # 2. ETA to this VMS
    if eta_minutes < float("inf"):
        parts.append(
            f"Kön når {vms.name} ({vms.vms_id}) om {eta_minutes:.1f} min."
        )
    else:
        parts.append(
            f"Kön når {vms.name} ({vms.vms_id}) — ETA okänd (kö stillastående)."
        )

    # 3. Current VMS status
    if current_status_str is not None:
        parts.append(f"Nuvarande VMS-status: {current_status_str}.")

    # 4. Recommendation
    if eta_minutes <= 2.0:
        parts.append("Rekommendation: Aktivera KÖVARNING 50 km/h OMEDELBART.")
    elif eta_minutes <= 5.0:
        lead_time = max(eta_minutes - 1.0, 0.5)
        parts.append(
            f"Rekommendation: Aktivera 70 km/h varning om {lead_time:.0f} min."
        )
    else:
        parts.append("Rekommendation: Bevakning — aktivera vid behov.")

    return " ".join(parts)


def _build_sensor_narrative(
    anomaly: SensorAnomaly,
    vms: VMSGantry,
    current_status_str: str | None,
) -> str:
    """Generate an operator narrative for a sensor-detected speed anomaly.

    Example output::

        Sensorlarm: Station 2790 mäter 35.0 km/h (gräns 70 km/h, 50%).
        Närmaste VMS: Kristineberg (VMS-4008).
        Nuvarande VMS-status: OFF.
        Rekommendation: Aktivera KÖVARNING 50 km/h OMEDELBART.
    """
    parts: list[str] = [
        f"Sensorlarm: Station {anomaly.site_id} mäter "
        f"{anomaly.measured_speed_kmh:.0f} km/h "
        f"(gräns {anomaly.road_speed_limit_kmh} km/h, "
        f"{anomaly.speed_ratio * 100:.0f}%).",
        f"Närmaste VMS: {vms.name} ({vms.vms_id}).",
    ]

    if current_status_str is not None:
        parts.append(f"Nuvarande VMS-status: {current_status_str}.")

    parts.append("Rekommendation: Aktivera KÖVARNING 50 km/h OMEDELBART.")
    return " ".join(parts)


def _build_weather_narrative(
    condition: dict,
    vms: VMSGantry,
    current_status_str: str | None,
) -> str:
    """Generate an operator narrative for a slippery-road warning."""
    location = condition.get("location") or condition.get("road_number") or "E4"
    text = condition.get("condition_text") or "väglagsvarning"
    parts = [
        f"Väglagsvarning: {text} vid {location}.",
        f"Närmaste VMS: {vms.name} ({vms.vms_id}).",
        "Rekommendation: Aktivera HALKA-varning OMEDELBART.",
    ]
    if current_status_str is not None:
        parts.insert(2, f"Nuvarande VMS-status: {current_status_str}.")
    return " ".join(parts)
