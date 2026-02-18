"""
Unit tests for TravelTimeCalibrator.

Tests the EMA-smoothed free-flow speed adaptation, correction factor
computation, accuracy scoring, and confidence classification.
"""

from datetime import datetime

import pytest

from src.models import CalibrationSnapshot, QueuePrediction, TravelTimeReading
from src.travel_time_calibrator import (
    DEFAULT_FREE_FLOW_SPEED,
    EMA_ALPHA,
    MAX_CORRECTION_RATIO,
    MIN_CORRECTION_RATIO,
    MIN_FREEFLOW_SEGMENTS,
    TravelTimeCalibrator,
)


# ======================================================================
# Helpers
# ======================================================================

def _make_reading(
    speed_kmh: float = 90.0,
    length_meters: float = 1000.0,
    traffic_status: str = "freeflow",
    delay: float = 0.0,
) -> TravelTimeReading:
    """Build a TravelTimeReading with sensible defaults."""
    ff_seconds = length_meters / (speed_kmh / 3.6) if speed_kmh > 0 else 60.0
    return TravelTimeReading(
        timestamp=datetime.now(),
        route_id="test_route",
        name="E4/E20 N Test Segment",
        travel_time_seconds=ff_seconds + delay,
        free_flow_seconds=ff_seconds,
        speed_kmh=speed_kmh,
        length_meters=length_meters,
        traffic_status=traffic_status,
        delay_seconds=delay,
    )


def _make_prediction() -> QueuePrediction:
    """Build a minimal QueuePrediction."""
    return QueuePrediction(
        timestamp=datetime.now(),
        camera_id="cam_test",
        origin_lat=59.3,
        origin_lng=18.0,
        origin_chainage_km=5.0,
        growth_speed_kmh=15.0,
        lengths_at_minutes={1: 0.25, 5: 1.25},
    )


# ======================================================================
# Weighted average speed
# ======================================================================


class TestWeightedAverageSpeed:
    """Tests for _weighted_avg_speed helper."""

    def test_empty_list_returns_none(self):
        assert TravelTimeCalibrator._weighted_avg_speed([]) is None

    def test_single_reading(self):
        r = _make_reading(speed_kmh=80.0, length_meters=2000.0)
        result = TravelTimeCalibrator._weighted_avg_speed([r])
        assert result == pytest.approx(80.0)

    def test_length_weighted_average(self):
        """Longer segments should have more weight."""
        r1 = _make_reading(speed_kmh=100.0, length_meters=3000.0)  # 3 km at 100
        r2 = _make_reading(speed_kmh=60.0, length_meters=1000.0)   # 1 km at 60
        result = TravelTimeCalibrator._weighted_avg_speed([r1, r2])
        expected = (100 * 3000 + 60 * 1000) / 4000  # = 90.0
        assert result == pytest.approx(expected)

    def test_zero_length_returns_none(self):
        r = _make_reading(speed_kmh=80.0, length_meters=0.0)
        assert TravelTimeCalibrator._weighted_avg_speed([r]) is None


# ======================================================================
# Confidence classification
# ======================================================================


class TestConfidenceClassification:
    """Tests for _assess_confidence."""

    def test_high_confidence(self):
        assert TravelTimeCalibrator._assess_confidence(10, 2) == "high"

    def test_medium_confidence(self):
        assert TravelTimeCalibrator._assess_confidence(5, 0) == "medium"

    def test_low_confidence(self):
        assert TravelTimeCalibrator._assess_confidence(2, 0) == "low"

    def test_boundary_high_medium(self):
        assert TravelTimeCalibrator._assess_confidence(8, 0) == "high"
        assert TravelTimeCalibrator._assess_confidence(7, 0) == "medium"

    def test_boundary_medium_low(self):
        assert TravelTimeCalibrator._assess_confidence(3, 0) == "medium"
        assert TravelTimeCalibrator._assess_confidence(2, 0) == "low"


# ======================================================================
# EMA update
# ======================================================================


class TestEMAUpdate:
    """Tests for the update() method — EMA-smoothed calibration."""

    def test_first_tick_seeds_ema(self):
        """First tick should set adapted speed directly (no smoothing)."""
        cal = TravelTimeCalibrator()
        readings = [_make_reading(speed_kmh=85.0) for _ in range(5)]
        snapshot = cal.update(readings, model_free_flow_speed=110.0)

        assert snapshot.adapted_free_flow_speed == pytest.approx(85.0, abs=0.1)
        assert snapshot.freeflow_segment_count == 5
        assert cal.tick_count == 1

    def test_ema_smoothing(self):
        """Second tick should apply EMA smoothing."""
        cal = TravelTimeCalibrator()

        # Tick 1: seed at 85 km/h
        r1 = [_make_reading(speed_kmh=85.0) for _ in range(5)]
        cal.update(r1, model_free_flow_speed=110.0)

        # Tick 2: new measurement at 95 km/h
        r2 = [_make_reading(speed_kmh=95.0) for _ in range(5)]
        snap2 = cal.update(r2, model_free_flow_speed=110.0)

        # EMA: 0.1 * 95 + 0.9 * 85 = 86.0
        assert snap2.adapted_free_flow_speed == pytest.approx(86.0, abs=0.1)

    def test_ema_converges_over_many_ticks(self):
        """EMA should gradually converge to the measured value."""
        cal = TravelTimeCalibrator()

        # 50 ticks all at 95 km/h — should converge close to 95
        for _ in range(50):
            readings = [_make_reading(speed_kmh=95.0) for _ in range(10)]
            snap = cal.update(readings, model_free_flow_speed=110.0)

        assert snap.adapted_free_flow_speed == pytest.approx(95.0, abs=0.5)

    def test_too_few_freeflow_segments_no_update(self):
        """With < MIN_FREEFLOW_SEGMENTS freeflow readings, don't update EMA."""
        cal = TravelTimeCalibrator()

        # Tick 1: seed with enough segments
        seed = [_make_reading(speed_kmh=90.0) for _ in range(5)]
        cal.update(seed, model_free_flow_speed=110.0)

        # Tick 2: only 2 freeflow segments (below threshold)
        few = [_make_reading(speed_kmh=50.0) for _ in range(2)]
        snap = cal.update(few, model_free_flow_speed=110.0)

        # Should NOT have shifted to 50 — EMA stays at seed value
        assert snap.adapted_free_flow_speed == pytest.approx(90.0, abs=0.1)
        assert snap.confidence == "low"

    def test_no_readings_no_crash(self):
        """Empty readings should not crash."""
        cal = TravelTimeCalibrator()
        snap = cal.update([], model_free_flow_speed=110.0)

        assert snap.adapted_free_flow_speed == DEFAULT_FREE_FLOW_SPEED
        assert snap.freeflow_segment_count == 0
        assert snap.confidence == "low"


# ======================================================================
# Correction factor
# ======================================================================


class TestCorrectionFactor:
    """Tests for correction factor computation and clamping."""

    def test_correction_below_one(self):
        """When measured speed < model speed, correction < 1."""
        cal = TravelTimeCalibrator()
        readings = [_make_reading(speed_kmh=80.0) for _ in range(5)]
        snap = cal.update(readings, model_free_flow_speed=110.0)

        assert snap.correction_factor < 1.0
        assert snap.correction_factor == pytest.approx(80.0 / 110.0, abs=0.01)

    def test_correction_above_one(self):
        """When measured speed > model speed, correction > 1."""
        cal = TravelTimeCalibrator()
        readings = [_make_reading(speed_kmh=115.0) for _ in range(5)]
        snap = cal.update(readings, model_free_flow_speed=110.0)

        assert snap.correction_factor > 1.0

    def test_correction_clamped_high(self):
        """Correction factor should not exceed MAX_CORRECTION_RATIO."""
        cal = TravelTimeCalibrator()
        readings = [_make_reading(speed_kmh=200.0) for _ in range(5)]
        snap = cal.update(readings, model_free_flow_speed=110.0)

        assert snap.correction_factor <= MAX_CORRECTION_RATIO

    def test_correction_clamped_low(self):
        """Correction factor should not go below MIN_CORRECTION_RATIO."""
        cal = TravelTimeCalibrator()
        readings = [_make_reading(speed_kmh=30.0) for _ in range(5)]
        snap = cal.update(readings, model_free_flow_speed=110.0)

        assert snap.correction_factor >= MIN_CORRECTION_RATIO


# ======================================================================
# Accuracy evaluation
# ======================================================================


class TestAccuracyEvaluation:
    """Tests for evaluate_accuracy()."""

    def test_no_congestion_no_scoring(self):
        """When all segments are freeflow, accuracy is N/A."""
        cal = TravelTimeCalibrator()
        readings = [_make_reading(traffic_status="freeflow") for _ in range(5)]
        snap = cal.update(readings, model_free_flow_speed=110.0)

        result = cal.evaluate_accuracy(readings, [], snap)
        assert result.accuracy_hit_rate is None

    def test_congestion_with_predictions_hit(self):
        """Congested segments + predictions = hit rate 1.0."""
        cal = TravelTimeCalibrator()
        readings = [
            _make_reading(traffic_status="slow", speed_kmh=40.0),
            _make_reading(traffic_status="heavy", speed_kmh=20.0),
        ]
        snap = cal.update(readings, model_free_flow_speed=110.0)
        predictions = [_make_prediction()]

        result = cal.evaluate_accuracy(readings, predictions, snap)
        assert result.accuracy_hit_rate == 1.0

    def test_congestion_without_predictions_miss(self):
        """Congested segments with no predictions = hit rate 0.0."""
        cal = TravelTimeCalibrator()
        readings = [_make_reading(traffic_status="slow", speed_kmh=40.0)]
        snap = cal.update(readings, model_free_flow_speed=110.0)

        result = cal.evaluate_accuracy(readings, [], snap)
        assert result.accuracy_hit_rate == 0.0


# ======================================================================
# State serialization
# ======================================================================


class TestCalibrationState:
    """Tests for get_state() method."""

    def test_initial_state(self):
        cal = TravelTimeCalibrator()
        state = cal.get_state()

        assert state["adapted_free_flow_speed"] == DEFAULT_FREE_FLOW_SPEED
        assert state["tick_count"] == 0
        assert state["last_measured_speed"] is None

    def test_state_after_update(self):
        cal = TravelTimeCalibrator()
        readings = [_make_reading(speed_kmh=85.0) for _ in range(5)]
        cal.update(readings, model_free_flow_speed=110.0)

        state = cal.get_state()
        assert state["tick_count"] == 1
        assert state["last_measured_speed"] == pytest.approx(85.0, abs=0.1)
        assert state["adapted_free_flow_speed"] == pytest.approx(85.0, abs=0.1)
