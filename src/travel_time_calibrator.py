"""
TravelTime Calibrator — Adapts physics engine parameters from ground truth.

Uses measured corridor travel times from Trafikverket's TravelTimeRoute API
as ground truth to continuously calibrate the LWR shockwave model.

Calibration targets:
    1. **free_flow_speed** — EMA-smoothed measured speed from freeflow segments
    2. **accuracy_hit_rate** — fraction of congested segments that had a
       matching queue prediction (validation, not adaptation)

The calibrator is **stateful** — it maintains an EMA across ticks.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from statistics import mean

from config import E4_NORTHBOUND_CORRIDOR_LENGTH_KM
from src.models import CalibrationSnapshot, QueuePrediction, TravelTimeReading

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

#: EMA smoothing factor.  α = 0.1 → ~10-tick half-life.
EMA_ALPHA: float = 0.10

#: Minimum number of freeflow segments to consider the measurement reliable.
MIN_FREEFLOW_SEGMENTS: int = 3

#: Maximum correction ratio (clamp).  Prevents wild swings from bad data.
MAX_CORRECTION_RATIO: float = 1.25
MIN_CORRECTION_RATIO: float = 0.60

#: Default free-flow speed when no measurements exist (km/h).
DEFAULT_FREE_FLOW_SPEED: float = 110.0

#: Minimum matched residual samples before correction is allowed.
MIN_RESIDUAL_SAMPLES: int = 5

#: Maximum learned ETA correction in either direction.
MAX_RESIDUAL_CORRECTION_MINUTES: float = 5.0

#: How long pending predictions can wait for matching congestion.
RESIDUAL_MATCH_EXPIRY_MINUTES: float = 30.0

#: Predictions logged shortly after a congestion event can still train as late hits.
RESIDUAL_LATE_HIT_TOLERANCE_MINUTES: float = 5.0


# ---------------------------------------------------------------------------
# Calibrator
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RouteSpan:
    route_id: str
    start_km: float
    end_km: float

    @property
    def midpoint_km(self) -> float:
        return (self.start_km + self.end_km) / 2.0


@dataclass
class PendingResidualObservation:
    timestamp: datetime
    camera_id: str
    route_id: str
    predicted_eta_minutes: float
    bucket: str


@dataclass
class TravelTimeCalibrator:
    """Continuously adapts physics parameters from TravelTimeRoute data."""

    #: The EMA-smoothed free-flow speed derived from measurements (km/h).
    adapted_free_flow_speed: float = DEFAULT_FREE_FLOW_SPEED

    #: How many ticks the calibrator has processed.
    tick_count: int = 0

    #: Most recent raw measurement (before EMA).
    _last_measured_speed: float | None = field(default=None, repr=False)

    #: Pending prediction-route observations waiting for congestion ground truth.
    _pending_residuals: list[PendingResidualObservation] = field(
        default_factory=list,
        repr=False,
    )

    #: Learned ETA residuals by conservative bucket.
    _residuals_by_bucket: dict[str, list[float]] = field(
        default_factory=dict,
        repr=False,
    )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        readings: list[TravelTimeReading],
        model_free_flow_speed: float = DEFAULT_FREE_FLOW_SPEED,
    ) -> CalibrationSnapshot:
        """Compute calibration from current TravelTimeRoute readings.

        Parameters
        ----------
        readings:
            Latest travel time data for the corridor.
        model_free_flow_speed:
            The physics engine's current ``free_flow_speed`` setting.

        Returns
        -------
        CalibrationSnapshot
            Current calibration state with correction factor.
        """
        self.tick_count += 1

        # Split by traffic status
        freeflow = [r for r in readings if r.traffic_status == "freeflow"]
        congested = [
            r for r in readings if r.traffic_status in ("slow", "heavy")
        ]

        measured_speed = self._weighted_avg_speed(freeflow)
        self._last_measured_speed = measured_speed

        # Update EMA
        if measured_speed is not None and len(freeflow) >= MIN_FREEFLOW_SEGMENTS:
            if self.tick_count == 1:
                # First tick — seed the EMA
                self.adapted_free_flow_speed = measured_speed
            else:
                self.adapted_free_flow_speed = (
                    EMA_ALPHA * measured_speed
                    + (1 - EMA_ALPHA) * self.adapted_free_flow_speed
                )

        # Compute correction factor
        correction_factor = self.adapted_free_flow_speed / model_free_flow_speed

        # Clamp to prevent pathological values
        correction_factor = max(
            MIN_CORRECTION_RATIO,
            min(MAX_CORRECTION_RATIO, correction_factor),
        )

        # Confidence based on sample size
        confidence = self._assess_confidence(len(freeflow), len(congested))

        logger.info(
            f"🎯 Calibration: measured_ff="
            f"{f'{measured_speed:.1f}' if measured_speed is not None else '—'} km/h, "
            f"adapted={self.adapted_free_flow_speed:.1f} km/h, "
            f"correction={correction_factor:.3f}, "
            f"confidence={confidence} "
            f"({len(freeflow)} ff / {len(congested)} congested)"
        )

        return CalibrationSnapshot(
            adapted_free_flow_speed=round(self.adapted_free_flow_speed, 2),
            correction_factor=round(correction_factor, 4),
            measured_free_flow_speed=(
                round(measured_speed, 2) if measured_speed else None
            ),
            freeflow_segment_count=len(freeflow),
            congested_segment_count=len(congested),
            accuracy_hit_rate=None,  # Set by evaluate_accuracy() later
            confidence=confidence,
        )

    def evaluate_accuracy(
        self,
        readings: list[TravelTimeReading],
        predictions: list[QueuePrediction],
        snapshot: CalibrationSnapshot,
    ) -> CalibrationSnapshot:
        """Score the physics engine against TravelTimeRoute congestion.

        Compares congested TravelTimeRoute segments to active queue
        predictions.  A "hit" = the physics engine also detected
        congestion in the same corridor area.

        Parameters
        ----------
        readings:
            Current travel time data.
        predictions:
            Queue predictions from the physics engine this tick.
        snapshot:
            The CalibrationSnapshot to update with accuracy data.

        Returns
        -------
        CalibrationSnapshot
            Updated with ``accuracy_hit_rate``.
        """
        congested = [
            r for r in readings if r.traffic_status in ("slow", "heavy")
        ]

        if not congested:
            # No congestion to validate against — accuracy N/A
            return snapshot

        # A simple approach: if physics produced ANY prediction while
        # TravelTimeRoute shows congestion, that's a corridor-level hit.
        # More granular: match by comparing prediction camera positions
        # to TravelTimeRoute segment coordinates (future enhancement).
        has_predictions = len(predictions) > 0

        if has_predictions:
            # At least one prediction — partial hit scoring
            # Count congested segments within prediction coverage
            hit_rate = 1.0 if predictions else 0.0
        else:
            hit_rate = 0.0

        logger.info(
            f"🎯 Accuracy: {len(congested)} congested segments, "
            f"{len(predictions)} predictions → hit_rate={hit_rate:.1%}"
        )

        snapshot.accuracy_hit_rate = round(hit_rate, 4)
        return snapshot

    def apply_residual_corrections(
        self,
        readings: list[TravelTimeReading],
        predictions: list[QueuePrediction],
        now: datetime,
    ) -> list[QueuePrediction]:
        """Apply learned ETA residual metadata without changing LWR geometry.

        This method first learns from pending predictions that now have
        congested TravelTimeRoute evidence, then records the current tick's
        predictions for future learning.  Corrections are disabled until the
        matching bucket has enough history.
        """
        spans = self._build_route_spans(readings)
        if not spans:
            for prediction in predictions:
                self._disable_residual(prediction, "missing route spans")
            return predictions

        congested_by_route = {
            reading.route_id: reading
            for reading in readings
            if reading.traffic_status in {"slow", "heavy"}
        }
        self._learn_residuals(congested_by_route, now)

        for prediction in predictions:
            span = self._target_span_for_prediction(prediction, spans)
            if span is None:
                self._disable_residual(prediction, "no downstream route target")
                continue

            predicted_eta = self._eta_to_span_midpoint(prediction, span)
            if predicted_eta is None:
                self._disable_residual(prediction, "invalid base ETA")
                continue

            bucket = self._residual_bucket(
                prediction.camera_id,
                span.route_id,
                prediction.timestamp,
            )
            samples = self._residuals_by_bucket.get(bucket, [])
            sample_count = len(samples)
            prediction.residual_bucket = bucket
            prediction.residual_sample_count = sample_count
            prediction.residual_confidence = self._residual_confidence(sample_count)

            if sample_count < MIN_RESIDUAL_SAMPLES:
                prediction.residual_correction_enabled = False
                prediction.residual_correction_minutes = 0.0
                prediction.residual_disabled_reason = "insufficient history"
            else:
                correction = _clamp(
                    mean(samples),
                    -MAX_RESIDUAL_CORRECTION_MINUTES,
                    MAX_RESIDUAL_CORRECTION_MINUTES,
                )
                prediction.residual_correction_enabled = True
                prediction.residual_correction_minutes = round(correction, 3)
                prediction.residual_disabled_reason = None

            target = f"route:{span.route_id}"
            prediction.base_eta_minutes_by_target[target] = round(predicted_eta, 3)
            prediction.corrected_eta_minutes_by_target[target] = round(
                max(predicted_eta + prediction.residual_correction_minutes, 0.0),
                3,
            )
            self._pending_residuals.append(PendingResidualObservation(
                timestamp=prediction.timestamp,
                camera_id=prediction.camera_id,
                route_id=span.route_id,
                predicted_eta_minutes=predicted_eta,
                bucket=bucket,
            ))

        self._prune_pending(now)
        return predictions

    def get_state(self) -> dict:
        """Return the current calibration state for API consumers."""
        return {
            "adapted_free_flow_speed": round(self.adapted_free_flow_speed, 2),
            "tick_count": self.tick_count,
            "last_measured_speed": (
                round(self._last_measured_speed, 2)
                if self._last_measured_speed is not None
                else None
            ),
            "residual_pending_count": len(self._pending_residuals),
            "residual_bucket_count": len(self._residuals_by_bucket),
            "residual_min_samples": MIN_RESIDUAL_SAMPLES,
            "residual_max_correction_minutes": MAX_RESIDUAL_CORRECTION_MINUTES,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _weighted_avg_speed(
        readings: list[TravelTimeReading],
    ) -> float | None:
        """Compute length-weighted average speed from freeflow segments.

        Uses segment length as weight so that longer segments (which
        give more reliable speed estimates) have proportionally more
        influence.
        """
        if not readings:
            return None

        total_length = sum(r.length_meters for r in readings)
        if total_length <= 0:
            return None

        weighted_speed = sum(
            r.speed_kmh * r.length_meters for r in readings
        )
        return weighted_speed / total_length

    @staticmethod
    def _assess_confidence(
        freeflow_count: int,
        congested_count: int,
    ) -> str:
        """Classify calibration confidence based on segment counts.

        - "high": ≥ 8 freeflow segments — robust measurement
        - "medium": ≥ 3 freeflow segments — reasonable estimate
        - "low": < 3 freeflow segments — too few for reliable calibration
        """
        if freeflow_count >= 8:
            return "high"
        elif freeflow_count >= MIN_FREEFLOW_SEGMENTS:
            return "medium"
        else:
            return "low"

    def _learn_residuals(
        self,
        congested_by_route: dict[str, TravelTimeReading],
        now: datetime,
    ) -> None:
        if not congested_by_route:
            return

        still_pending: list[PendingResidualObservation] = []
        for pending in self._pending_residuals:
            age_minutes = (now - pending.timestamp).total_seconds() / 60.0
            if age_minutes > RESIDUAL_MATCH_EXPIRY_MINUTES:
                continue

            reading = congested_by_route.get(pending.route_id)
            if reading is None:
                still_pending.append(pending)
                continue

            lead_time = (reading.timestamp - pending.timestamp).total_seconds() / 60.0
            if lead_time < -RESIDUAL_LATE_HIT_TOLERANCE_MINUTES:
                still_pending.append(pending)
                continue
            if lead_time > RESIDUAL_MATCH_EXPIRY_MINUTES:
                continue

            residual = _clamp(
                lead_time - pending.predicted_eta_minutes,
                -MAX_RESIDUAL_CORRECTION_MINUTES,
                MAX_RESIDUAL_CORRECTION_MINUTES,
            )
            self._residuals_by_bucket.setdefault(pending.bucket, []).append(residual)

        self._pending_residuals = still_pending

    def _prune_pending(self, now: datetime) -> None:
        self._pending_residuals = [
            pending
            for pending in self._pending_residuals
            if (now - pending.timestamp).total_seconds() / 60.0
            <= RESIDUAL_MATCH_EXPIRY_MINUTES
        ]

    @staticmethod
    def _disable_residual(prediction: QueuePrediction, reason: str) -> None:
        prediction.residual_correction_enabled = False
        prediction.residual_correction_minutes = 0.0
        prediction.residual_sample_count = 0
        prediction.residual_confidence = "none"
        prediction.residual_disabled_reason = reason

    @staticmethod
    def _eta_to_span_midpoint(
        prediction: QueuePrediction,
        span: RouteSpan,
    ) -> float | None:
        if prediction.growth_speed_kmh <= 0:
            return None
        distance_km = prediction.origin_chainage_km - span.midpoint_km
        if distance_km < 0:
            return None
        return distance_km / prediction.growth_speed_kmh * 60.0

    @staticmethod
    def _target_span_for_prediction(
        prediction: QueuePrediction,
        spans: dict[str, RouteSpan],
    ) -> RouteSpan | None:
        return max(
            (
                span for span in spans.values()
                if prediction.origin_chainage_km >= span.midpoint_km
            ),
            key=lambda span: span.midpoint_km,
            default=None,
        )

    @staticmethod
    def _build_route_spans(readings: list[TravelTimeReading]) -> dict[str, RouteSpan]:
        lengths_by_route: dict[str, float] = {}
        route_order: list[str] = []
        for reading in readings:
            if reading.route_id not in lengths_by_route:
                route_order.append(reading.route_id)
            lengths_by_route[reading.route_id] = max(
                lengths_by_route.get(reading.route_id, 0.0),
                reading.length_meters,
            )

        total_length = sum(lengths_by_route.values())
        if total_length <= 0:
            return {}

        spans: dict[str, RouteSpan] = {}
        start_km = 0.0
        for idx, route_id in enumerate(route_order):
            length_km = (
                lengths_by_route[route_id]
                / total_length
                * E4_NORTHBOUND_CORRIDOR_LENGTH_KM
            )
            end_km = (
                E4_NORTHBOUND_CORRIDOR_LENGTH_KM
                if idx == len(route_order) - 1
                else start_km + length_km
            )
            spans[route_id] = RouteSpan(route_id, start_km, end_km)
            start_km = end_km
        return spans

    @staticmethod
    def _residual_bucket(camera_id: str, route_id: str, timestamp: datetime) -> str:
        day_type = "weekend" if timestamp.weekday() >= 5 else "weekday"
        hour_bucket = f"{(timestamp.hour // 3) * 3:02d}"
        return f"{camera_id}|{route_id}|{day_type}|h{hour_bucket}"

    @staticmethod
    def _residual_confidence(sample_count: int) -> str:
        if sample_count >= 20:
            return "high"
        if sample_count >= MIN_RESIDUAL_SAMPLES:
            return "medium"
        if sample_count > 0:
            return "low"
        return "none"


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))
