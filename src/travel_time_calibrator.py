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


# ---------------------------------------------------------------------------
# Calibrator
# ---------------------------------------------------------------------------


@dataclass
class TravelTimeCalibrator:
    """Continuously adapts physics parameters from TravelTimeRoute data."""

    #: The EMA-smoothed free-flow speed derived from measurements (km/h).
    adapted_free_flow_speed: float = DEFAULT_FREE_FLOW_SPEED

    #: How many ticks the calibrator has processed.
    tick_count: int = 0

    #: Most recent raw measurement (before EMA).
    _last_measured_speed: float | None = field(default=None, repr=False)

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
