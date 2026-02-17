"""
Unit tests for the Shockwave Prediction Engine (Phase 3).

Tests the LWR kinematic wave model:
    wave_speed = (Q_in - Q_cap) / (k_jam - k_in)
"""

from datetime import datetime

import pytest

from src.models import CapacityState, QueuePrediction, SensorReading
from src.physics_engine import (
    JAM_DENSITY_VEH_KM_LANE,
    MIN_CAPACITY_DROP_VPH,
    PhysicsEngine,
)


# ======================================================================
# Fixtures
# ======================================================================


@pytest.fixture
def engine() -> PhysicsEngine:
    return PhysicsEngine()


@pytest.fixture
def bottleneck_state() -> CapacityState:
    """A camera showing a bottleneck: anomaly detected, low capacity."""
    return CapacityState(
        timestamp=datetime(2026, 2, 16, 14, 0, 0),
        camera_id="CAM_TEST_01",
        vehicle_count=8,
        blocked_lanes=1,
        total_lanes=3,
        estimated_capacity_vph=1200.0,  # Reduced from ~4500 free-flow
        is_anomaly=True,
        anomaly_reason="vehicle_stopped",
        confidence=0.85,
    )


@pytest.fixture
def normal_state() -> CapacityState:
    """A camera showing normal traffic — no anomaly."""
    return CapacityState(
        timestamp=datetime(2026, 2, 16, 14, 0, 0),
        camera_id="CAM_TEST_02",
        vehicle_count=15,
        blocked_lanes=0,
        total_lanes=3,
        estimated_capacity_vph=4200.0,
        is_anomaly=False,
        anomaly_reason=None,
        confidence=0.92,
    )


@pytest.fixture
def upstream_sensor() -> SensorReading:
    """Upstream sensor showing normal inflow."""
    return SensorReading(
        timestamp=datetime(2026, 2, 16, 14, 0, 0),
        inflow_volume_vph=4000.0,
        average_speed_kmh=95.0,
    )


@pytest.fixture
def chainage_map() -> dict[str, float]:
    return {"CAM_TEST_01": 8.0, "CAM_TEST_02": 5.0}


@pytest.fixture
def coords_map() -> dict[str, tuple[float, float]]:
    return {"CAM_TEST_01": (59.30, 18.00), "CAM_TEST_02": (59.27, 17.91)}


# ======================================================================
# LWR wave speed tests
# ======================================================================


class TestLWRWaveSpeed:
    def test_positive_wave_speed_when_queue_growing(self, engine: PhysicsEngine) -> None:
        """When inflow exceeds bottleneck capacity, wave speed is positive."""
        speed = engine._lwr_wave_speed(
            inflow_vph=4000.0,
            bottleneck_capacity_vph=1200.0,
            upstream_speed_kmh=95.0,
            num_lanes=3,
        )
        assert speed > 0
        # Sanity check: speed should be reasonable (< 30 km/h for typical queues)
        assert speed < 30.0

    def test_zero_wave_speed_when_balanced(self, engine: PhysicsEngine) -> None:
        """When inflow equals capacity, wave speed is zero (no queue growth)."""
        speed = engine._lwr_wave_speed(
            inflow_vph=3000.0,
            bottleneck_capacity_vph=3000.0,
            upstream_speed_kmh=95.0,
            num_lanes=3,
        )
        assert speed == 0.0

    def test_zero_wave_speed_when_capacity_exceeds_inflow(self, engine: PhysicsEngine) -> None:
        """When capacity exceeds inflow, wave speed is zero (queue dissolving)."""
        speed = engine._lwr_wave_speed(
            inflow_vph=2000.0,
            bottleneck_capacity_vph=3000.0,
            upstream_speed_kmh=95.0,
            num_lanes=3,
        )
        assert speed == 0.0

    def test_wave_speed_increases_with_larger_drop(self, engine: PhysicsEngine) -> None:
        """Larger capacity drops produce faster queue growth."""
        speed_small = engine._lwr_wave_speed(
            inflow_vph=4000.0,
            bottleneck_capacity_vph=3000.0,
            upstream_speed_kmh=95.0,
            num_lanes=3,
        )
        speed_large = engine._lwr_wave_speed(
            inflow_vph=4000.0,
            bottleneck_capacity_vph=1000.0,
            upstream_speed_kmh=95.0,
            num_lanes=3,
        )
        assert speed_large > speed_small

    def test_zero_upstream_speed_returns_capped(self, engine: PhysicsEngine) -> None:
        """When upstream speed is zero (full stop), wave speed is capped."""
        speed = engine._lwr_wave_speed(
            inflow_vph=4000.0,
            bottleneck_capacity_vph=1000.0,
            upstream_speed_kmh=0.0,
            num_lanes=3,
        )
        # Should return the heuristic cap (55 km/h)
        assert speed == engine.free_flow_speed * 0.5


# ======================================================================
# Queue prediction computation tests
# ======================================================================


class TestQueuePrediction:
    def test_produces_prediction_for_bottleneck(
        self,
        engine: PhysicsEngine,
        bottleneck_state: CapacityState,
        upstream_sensor: SensorReading,
        chainage_map: dict[str, float],
        coords_map: dict[str, tuple[float, float]],
    ) -> None:
        """Bottleneck camera with anomaly should produce a prediction."""
        predictions = engine.compute(
            capacity_states=[bottleneck_state],
            sensor=upstream_sensor,
            camera_chainage_map=chainage_map,
            camera_coords_map=coords_map,
        )
        assert len(predictions) == 1
        pred = predictions[0]
        assert isinstance(pred, QueuePrediction)
        assert pred.camera_id == "CAM_TEST_01"
        assert pred.growth_speed_kmh > 0
        assert pred.origin_chainage_km == 8.0

    def test_no_prediction_for_normal_traffic(
        self,
        engine: PhysicsEngine,
        normal_state: CapacityState,
        upstream_sensor: SensorReading,
    ) -> None:
        """Normal traffic (no anomaly) should produce no predictions."""
        predictions = engine.compute(
            capacity_states=[normal_state],
            sensor=upstream_sensor,
        )
        assert len(predictions) == 0

    def test_no_prediction_without_sensor(
        self,
        engine: PhysicsEngine,
        bottleneck_state: CapacityState,
    ) -> None:
        """Without sensor data, no physics computation is possible."""
        predictions = engine.compute(
            capacity_states=[bottleneck_state],
            sensor=None,
        )
        assert len(predictions) == 0

    def test_queue_lengths_at_time_horizons(
        self,
        engine: PhysicsEngine,
        bottleneck_state: CapacityState,
        upstream_sensor: SensorReading,
        chainage_map: dict[str, float],
    ) -> None:
        """Queue lengths should increase with time."""
        predictions = engine.compute(
            capacity_states=[bottleneck_state],
            sensor=upstream_sensor,
            camera_chainage_map=chainage_map,
        )
        assert len(predictions) == 1
        lengths = predictions[0].lengths_at_minutes

        # Check that all default horizons are present
        assert 1 in lengths
        assert 3 in lengths
        assert 5 in lengths
        assert 10 in lengths

        # Lengths should increase with time
        assert lengths[1] < lengths[3] < lengths[5] < lengths[10]

    def test_no_prediction_when_drop_below_threshold(
        self,
        engine: PhysicsEngine,
        upstream_sensor: SensorReading,
    ) -> None:
        """If capacity drop is too small, no prediction is generated."""
        state = CapacityState(
            timestamp=datetime(2026, 2, 16, 14, 0, 0),
            camera_id="CAM_TEST_01",
            vehicle_count=10,
            blocked_lanes=0,
            total_lanes=3,
            estimated_capacity_vph=3900.0,  # Close to inflow of 4000
            is_anomaly=True,
            anomaly_reason="minor_slowdown",
        )
        predictions = engine.compute(
            capacity_states=[state],
            sensor=upstream_sensor,
        )
        # 4000 - 3900 = 100, below MIN_CAPACITY_DROP_VPH (200)
        assert len(predictions) == 0

    def test_prediction_has_correct_coordinates(
        self,
        engine: PhysicsEngine,
        bottleneck_state: CapacityState,
        upstream_sensor: SensorReading,
        chainage_map: dict[str, float],
        coords_map: dict[str, tuple[float, float]],
    ) -> None:
        """Prediction should carry the camera's geographic coordinates."""
        predictions = engine.compute(
            capacity_states=[bottleneck_state],
            sensor=upstream_sensor,
            camera_chainage_map=chainage_map,
            camera_coords_map=coords_map,
        )
        pred = predictions[0]
        assert pred.origin_lat == 59.30
        assert pred.origin_lng == 18.00


# ======================================================================
# Edge cases
# ======================================================================


class TestEdgeCases:
    def test_empty_capacity_states(self, engine: PhysicsEngine, upstream_sensor: SensorReading) -> None:
        predictions = engine.compute(capacity_states=[], sensor=upstream_sensor)
        assert predictions == []

    def test_custom_time_horizons(self, engine: PhysicsEngine) -> None:
        custom_engine = PhysicsEngine(time_horizons=[2, 7, 15])
        assert custom_engine.time_horizons == [2, 7, 15]

    def test_custom_jam_density(self) -> None:
        engine = PhysicsEngine(jam_density=150.0)
        assert engine.jam_density == 150.0
