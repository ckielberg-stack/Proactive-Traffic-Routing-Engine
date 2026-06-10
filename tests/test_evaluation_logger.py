"""
Unit tests for the Camera-to-Camera Prophecy system (evaluation_logger.py).

Tests the prediction recording, evaluation logic, expiry, and edge cases.
"""

from datetime import datetime, timedelta

import pytest

from src.evaluation_logger import (
    CAPACITY_DROP_FRACTION,
    EVALUATION_TOLERANCE_SECONDS,
    FREE_FLOW_PER_LANE_VPH,
    MAX_ETA_MINUTES,
    MIN_ETA_MINUTES,
    EvaluationLogger,
    Prophecy,
)
from src.models import CapacityState, QueuePrediction


# ======================================================================
# Fixtures
# ======================================================================


@pytest.fixture
def chainage_map() -> dict[str, float]:
    """Five cameras along a 10 km corridor, evenly spaced."""
    return {
        "CAM_01": 0.0,   # Southernmost
        "CAM_02": 2.5,
        "CAM_03": 5.0,
        "CAM_04": 7.5,
        "CAM_05": 10.0,  # Northernmost (bottleneck candidate)
    }


@pytest.fixture
def eval_logger(chainage_map: dict[str, float], tmp_path) -> EvaluationLogger:
    return EvaluationLogger(
        chainage_map=chainage_map,
        data_dir=str(tmp_path),
    )


@pytest.fixture
def bottleneck_prediction() -> QueuePrediction:
    """Queue growing backwards from CAM_05 (chainage 10.0 km) at 5 km/h."""
    return QueuePrediction(
        timestamp=datetime(2026, 2, 16, 14, 0, 0),
        camera_id="CAM_05",
        origin_lat=59.30,
        origin_lng=18.00,
        origin_chainage_km=10.0,
        growth_speed_kmh=5.0,
        lengths_at_minutes={1: 0.083, 3: 0.25, 5: 0.417, 10: 0.833},
    )


@pytest.fixture
def mid_corridor_prediction() -> QueuePrediction:
    """Queue at CAM_03 (chainage 5.0 km) at 10 km/h."""
    return QueuePrediction(
        timestamp=datetime(2026, 2, 16, 14, 0, 0),
        camera_id="CAM_03",
        origin_lat=59.27,
        origin_lng=17.95,
        origin_chainage_km=5.0,
        growth_speed_kmh=10.0,
        lengths_at_minutes={1: 0.167, 3: 0.5, 5: 0.833},
    )


# ======================================================================
# Prophecy recording
# ======================================================================


class TestRecordProphecies:
    def test_creates_prophecy_with_correct_eta(
        self,
        eval_logger: EvaluationLogger,
        bottleneck_prediction: QueuePrediction,
    ) -> None:
        """Prophecy ETA should be distance / speed × 60."""
        now = datetime(2026, 2, 16, 14, 0, 0)
        prophecies = eval_logger.record_prophecies([bottleneck_prediction], now)

        assert len(prophecies) == 1
        p = prophecies[0]
        assert p.source_camera_id == "CAM_05"
        assert p.target_camera_id == "CAM_04"
        assert p.source_chainage_km == 10.0
        assert p.target_chainage_km == 7.5
        # Distance = 2.5 km, speed = 5 km/h → ETA = 30 min
        assert p.predicted_eta_minutes == 30.0
        assert p.status == "pending"
        assert p.predicted_impact_time == now + timedelta(minutes=30)

    def test_mid_corridor_targets_correct_upstream(
        self,
        eval_logger: EvaluationLogger,
        mid_corridor_prediction: QueuePrediction,
    ) -> None:
        """CAM_03 (chainage 5.0) should target CAM_02 (chainage 2.5)."""
        now = datetime(2026, 2, 16, 14, 0, 0)
        prophecies = eval_logger.record_prophecies(
            [mid_corridor_prediction], now
        )

        assert len(prophecies) == 1
        p = prophecies[0]
        assert p.target_camera_id == "CAM_02"
        # Distance = 2.5 km, speed = 10 km/h → ETA = 15 min
        assert p.predicted_eta_minutes == 15.0

    def test_no_prophecy_for_southernmost_camera(
        self,
        eval_logger: EvaluationLogger,
    ) -> None:
        """CAM_01 is the southernmost — no upstream camera exists."""
        pred = QueuePrediction(
            timestamp=datetime(2026, 2, 16, 14, 0, 0),
            camera_id="CAM_01",
            origin_lat=59.24,
            origin_lng=17.84,
            origin_chainage_km=0.0,
            growth_speed_kmh=5.0,
            lengths_at_minutes={1: 0.083},
        )
        prophecies = eval_logger.record_prophecies(
            [pred], datetime(2026, 2, 16, 14, 0, 0)
        )
        assert len(prophecies) == 0

    def test_no_prophecy_when_eta_too_slow(
        self,
        eval_logger: EvaluationLogger,
    ) -> None:
        """Very slow queue (ETA > 30 min) should not create a prophecy."""
        pred = QueuePrediction(
            timestamp=datetime(2026, 2, 16, 14, 0, 0),
            camera_id="CAM_05",
            origin_lat=59.30,
            origin_lng=18.00,
            origin_chainage_km=10.0,
            growth_speed_kmh=0.5,  # Very slow → ETA = 300 min
            lengths_at_minutes={1: 0.008},
        )
        prophecies = eval_logger.record_prophecies(
            [pred], datetime(2026, 2, 16, 14, 0, 0)
        )
        assert len(prophecies) == 0

    def test_stats_increment_on_creation(
        self,
        eval_logger: EvaluationLogger,
        bottleneck_prediction: QueuePrediction,
    ) -> None:
        now = datetime(2026, 2, 16, 14, 0, 0)
        eval_logger.record_prophecies([bottleneck_prediction], now)
        stats = eval_logger.get_stats()
        assert stats["total_prophecies_created"] == 1
        assert stats["pending"] == 1


# ======================================================================
# Prophecy evaluation
# ======================================================================


class TestEvaluatePending:
    def test_verified_success_when_anomaly_at_target(
        self,
        eval_logger: EvaluationLogger,
        bottleneck_prediction: QueuePrediction,
    ) -> None:
        """If target camera shows anomaly at predicted time, VERIFIED_SUCCESS."""
        now = datetime(2026, 2, 16, 14, 0, 0)
        eval_logger.record_prophecies([bottleneck_prediction], now)

        # Advance to predicted impact time
        impact_time = now + timedelta(minutes=30)
        target_state = CapacityState(
            timestamp=impact_time,
            camera_id="CAM_04",
            vehicle_count=3,
            blocked_lanes=1,
            total_lanes=3,
            estimated_capacity_vph=500.0,
            is_anomaly=True,
            anomaly_reason="vehicle_stopped",
            confidence=0.85,
        )

        resolved = eval_logger.evaluate_pending([target_state], impact_time)
        assert len(resolved) == 1
        assert resolved[0].status == "VERIFIED_SUCCESS"
        assert eval_logger.pending_count == 0

    def test_verified_success_when_capacity_low(
        self,
        eval_logger: EvaluationLogger,
        bottleneck_prediction: QueuePrediction,
    ) -> None:
        """If capacity drops below 50% of free-flow, VERIFIED_SUCCESS."""
        now = datetime(2026, 2, 16, 14, 0, 0)
        eval_logger.record_prophecies([bottleneck_prediction], now)

        impact_time = now + timedelta(minutes=30)
        # 3 lanes × shared per-lane baseline. 50% remains above 2000 VPH.
        target_state = CapacityState(
            timestamp=impact_time,
            camera_id="CAM_04",
            vehicle_count=5,
            blocked_lanes=0,
            total_lanes=3,
            estimated_capacity_vph=2000.0,
            is_anomaly=False,
            confidence=0.90,
        )

        resolved = eval_logger.evaluate_pending([target_state], impact_time)
        assert len(resolved) == 1
        assert resolved[0].status == "VERIFIED_SUCCESS"

    def test_failed_when_target_is_healthy(
        self,
        eval_logger: EvaluationLogger,
        bottleneck_prediction: QueuePrediction,
    ) -> None:
        """If target camera is healthy at predicted time, FAILED."""
        now = datetime(2026, 2, 16, 14, 0, 0)
        eval_logger.record_prophecies([bottleneck_prediction], now)

        impact_time = now + timedelta(minutes=30)
        target_state = CapacityState(
            timestamp=impact_time,
            camera_id="CAM_04",
            vehicle_count=12,
            blocked_lanes=0,
            total_lanes=3,
            estimated_capacity_vph=5500.0,
            is_anomaly=False,
            confidence=0.92,
        )

        resolved = eval_logger.evaluate_pending([target_state], impact_time)
        assert len(resolved) == 1
        assert resolved[0].status == "FAILED"
        assert eval_logger.get_stats()["failed"] == 1

    def test_stays_pending_before_impact_time(
        self,
        eval_logger: EvaluationLogger,
        bottleneck_prediction: QueuePrediction,
    ) -> None:
        """Prophecy should remain pending if impact time hasn't arrived."""
        now = datetime(2026, 2, 16, 14, 0, 0)
        eval_logger.record_prophecies([bottleneck_prediction], now)

        # Only 5 minutes later — impact is at T+30
        early = now + timedelta(minutes=5)
        target_state = CapacityState(
            timestamp=early,
            camera_id="CAM_04",
            vehicle_count=12,
            blocked_lanes=0,
            total_lanes=3,
            estimated_capacity_vph=5500.0,
            is_anomaly=False,
            confidence=0.92,
        )

        resolved = eval_logger.evaluate_pending([target_state], early)
        assert len(resolved) == 0
        assert eval_logger.pending_count == 1

    def test_expired_past_window(
        self,
        eval_logger: EvaluationLogger,
        bottleneck_prediction: QueuePrediction,
    ) -> None:
        """Prophecies older than 30 min past impact time should EXPIRE."""
        now = datetime(2026, 2, 16, 14, 0, 0)
        eval_logger.record_prophecies([bottleneck_prediction], now)

        # 61 min after impact (30 min ETA + 30 min expiry + 1 min buffer)
        way_past = now + timedelta(minutes=61)
        resolved = eval_logger.evaluate_pending([], way_past)
        assert len(resolved) == 1
        assert resolved[0].status == "EXPIRED"

    def test_hit_rate_calculation(
        self,
        eval_logger: EvaluationLogger,
        mid_corridor_prediction: QueuePrediction,
    ) -> None:
        """Hit rate should be verified / (verified + failed)."""
        now = datetime(2026, 2, 16, 14, 0, 0)

        # Create and verify one prophecy
        eval_logger.record_prophecies([mid_corridor_prediction], now)
        impact = now + timedelta(minutes=15)
        success_state = CapacityState(
            timestamp=impact,
            camera_id="CAM_02",
            vehicle_count=2,
            blocked_lanes=1,
            total_lanes=3,
            estimated_capacity_vph=400.0,
            is_anomaly=True,
            anomaly_reason="congestion",
            confidence=0.88,
        )
        eval_logger.evaluate_pending([success_state], impact)

        stats = eval_logger.get_stats()
        assert stats["hit_rate"] == 1.0
        assert stats["verified_success"] == 1
        assert stats["failed"] == 0


# ======================================================================
# JSONL persistence
# ======================================================================


class TestJSONLPersistence:
    def test_writes_valid_jsonl(
        self,
        eval_logger: EvaluationLogger,
        bottleneck_prediction: QueuePrediction,
        tmp_path,
    ) -> None:
        """JSONL file should contain valid JSON on each line."""
        import json

        now = datetime(2026, 2, 16, 14, 0, 0)
        eval_logger.record_prophecies([bottleneck_prediction], now)

        jsonl_path = tmp_path / "evaluation_metrics.jsonl"
        assert jsonl_path.exists()

        lines = jsonl_path.read_text().strip().split("\n")
        assert len(lines) >= 1

        for line in lines:
            record = json.loads(line)
            assert "prophecy_id" in record
            assert "source_camera_id" in record
            assert "target_camera_id" in record
            assert record["status"] == "pending"
