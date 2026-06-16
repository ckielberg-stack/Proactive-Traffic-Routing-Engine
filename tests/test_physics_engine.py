"""
Unit tests for the Piecewise Shockwave Prediction Engine (Phase 3).

Tests the LWR kinematic wave model with multi-segment spatial iteration:
    wave_speed = (Q_in - Q_cap) / (k_jam - k_in)

The piecewise engine walks backward through camera nodes segment-by-segment,
using local inflow at each node to compute per-segment wave speeds.
"""

from datetime import datetime

import pytest

from src.models import (
    CapacityState,
    QueuePrediction,
    SegmentTrafficState,
    SensorReading,
)
from src.physics_engine import (
    JAM_DENSITY_VEH_KM_LANE,
    K_CRITICAL_VEH_KM_LANE,
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
    """A camera showing a bottleneck: density above k_critical, low capacity."""
    return CapacityState(
        timestamp=datetime(2026, 2, 16, 14, 0, 0),
        camera_id="CAM_03",
        vehicle_count=8,
        blocked_lanes=1,
        total_lanes=3,
        estimated_capacity_vph=1200.0,  # Reduced from ~6000 free-flow
        observed_density_veh_km_lane=55.0,  # Above k_critical (45)
        is_anomaly=True,
        anomaly_reason="density_exceeds_k_critical",
        confidence=0.85,
    )


@pytest.fixture
def normal_state() -> CapacityState:
    """A camera showing normal traffic — density below k_critical."""
    return CapacityState(
        timestamp=datetime(2026, 2, 16, 14, 0, 0),
        camera_id="CAM_02",
        vehicle_count=15,
        blocked_lanes=0,
        total_lanes=3,
        estimated_capacity_vph=6000.0,
        observed_density_veh_km_lane=10.0,  # Well below k_critical
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
    """Five cameras sorted south→north by chainage."""
    return {
        "CAM_01": 0.0,
        "CAM_02": 0.5,
        "CAM_03": 1.0,
        "CAM_04": 1.5,
        "CAM_05": 2.0,
    }


@pytest.fixture
def coords_map() -> dict[str, tuple[float, float]]:
    return {
        "CAM_01": (59.25, 17.90),
        "CAM_02": (59.27, 17.91),
        "CAM_03": (59.30, 18.00),
        "CAM_04": (59.32, 18.01),
        "CAM_05": (59.33, 18.02),
    }


# ======================================================================
# LWR wave speed tests (unchanged — tests the core formula)
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
# Piecewise queue prediction tests
# ======================================================================


class TestPiecewisePrediction:
    def test_queue_prediction_defaults_keep_legacy_constructors_valid(self) -> None:
        pred = QueuePrediction(
            timestamp=datetime(2026, 2, 16, 14, 0, 0),
            camera_id="CAM",
            origin_lat=59.0,
            origin_lng=18.0,
            origin_chainage_km=5.0,
            growth_speed_kmh=8.0,
            lengths_at_minutes={1: 0.133},
        )

        assert pred.prediction_confidence == 0.0
        assert pred.uncertainty_level == "low"
        assert pred.length_lower_at_minutes == {}
        assert pred.length_upper_at_minutes == {}

    def test_produces_prediction_with_segments(
        self,
        engine: PhysicsEngine,
        bottleneck_state: CapacityState,
        upstream_sensor: SensorReading,
        chainage_map: dict[str, float],
        coords_map: dict[str, tuple[float, float]],
    ) -> None:
        """Bottleneck with upstream nodes should produce segment speeds."""
        predictions = engine.compute(
            capacity_states=[bottleneck_state],
            sensor=upstream_sensor,
            camera_chainage_map=chainage_map,
            camera_coords_map=coords_map,
            node_inflows={
                "CAM_01": 4000.0,
                "CAM_02": 4000.0,
                "CAM_03": 4000.0,
            },
        )
        assert len(predictions) == 1
        pred = predictions[0]
        assert isinstance(pred, QueuePrediction)
        assert pred.camera_id == "CAM_03"
        # Should have segments going upstream from CAM_03 → CAM_02 → CAM_01
        assert len(pred.segment_speeds) == 2
        assert pred.segment_speeds[0].from_camera == "CAM_03"
        assert pred.segment_speeds[0].to_camera == "CAM_02"
        assert pred.segment_speeds[1].from_camera == "CAM_02"
        assert pred.segment_speeds[1].to_camera == "CAM_01"

    def test_southbound_state_iterates_toward_increasing_chainage(
        self,
        engine: PhysicsEngine,
        chainage_map: dict[str, float],
    ) -> None:
        """Southbound bottlenecks use the north/up-chainage side as upstream."""
        state = CapacityState(
            timestamp=datetime(2026, 2, 16, 14, 0, 0),
            camera_id="CAM_03",
            vehicle_count=8,
            blocked_lanes=0,
            total_lanes=3,
            estimated_capacity_vph=1200.0,
            observed_density_veh_km_lane=55.0,
            road_id="E4_Southbound",
            traffic_direction="southbound",
            is_anomaly=True,
            anomaly_reason="density_exceeds_k_critical",
            confidence=0.85,
        )

        predictions = engine.compute(
            capacity_states=[state],
            sensor=None,
            camera_chainage_map=chainage_map,
            node_inflows={
                "CAM_03": 4000.0,
                "CAM_04": 4000.0,
                "CAM_05": 4000.0,
            },
        )

        assert len(predictions) == 1
        pred = predictions[0]
        assert len(pred.segment_speeds) == 2
        assert pred.segment_speeds[0].from_camera == "CAM_03"
        assert pred.segment_speeds[0].to_camera == "CAM_04"
        assert pred.segment_speeds[1].from_camera == "CAM_04"
        assert pred.segment_speeds[1].to_camera == "CAM_05"

    def test_varying_inflows_produce_different_segment_speeds(
        self,
        engine: PhysicsEngine,
        bottleneck_state: CapacityState,
        chainage_map: dict[str, float],
    ) -> None:
        """Different local inflows should produce different wave speeds."""
        sensor = SensorReading(
            timestamp=datetime(2026, 2, 16, 14, 0, 0),
            inflow_volume_vph=4000.0,
            average_speed_kmh=95.0,
        )
        predictions = engine.compute(
            capacity_states=[bottleneck_state],
            sensor=sensor,
            camera_chainage_map=chainage_map,
            node_inflows={
                "CAM_01": 3000.0,  # Lower inflow (past an off-ramp)
                "CAM_02": 5000.0,  # Higher inflow (on-ramp added cars)
                "CAM_03": 4000.0,  # Bottleneck local inflow
            },
        )
        assert len(predictions) == 1
        pred = predictions[0]
        assert len(pred.segment_speeds) == 2
        # CAM_02 has higher inflow → faster wave speed
        # CAM_01 has lower inflow → slower wave speed
        speed_cam02 = pred.segment_speeds[0].wave_speed_kmh  # CAM_03→CAM_02
        speed_cam01 = pred.segment_speeds[1].wave_speed_kmh  # CAM_02→CAM_01
        assert speed_cam02 > speed_cam01

    def test_varying_local_speeds_produce_different_segment_speeds(
        self,
        engine: PhysicsEngine,
        bottleneck_state: CapacityState,
        chainage_map: dict[str, float],
    ) -> None:
        """Same inflow but different local speeds should alter LWR wave speed."""
        predictions = engine.compute(
            capacity_states=[bottleneck_state],
            sensor=None,
            camera_chainage_map=chainage_map,
            node_traffic_states={
                "CAM_01": SegmentTrafficState(
                    local_inflow_vph=4000.0,
                    local_speed_kmh=90.0,
                    inflow_source="traffic_flow",
                    speed_source="traffic_flow",
                    confidence="high",
                ),
                "CAM_02": SegmentTrafficState(
                    local_inflow_vph=4000.0,
                    local_speed_kmh=40.0,
                    inflow_source="traffic_flow",
                    speed_source="traffic_flow",
                    confidence="high",
                ),
                "CAM_03": SegmentTrafficState(
                    local_inflow_vph=4000.0,
                    local_speed_kmh=50.0,
                    inflow_source="traffic_flow",
                    speed_source="traffic_flow",
                    confidence="high",
                ),
            },
        )

        assert len(predictions) == 1
        pred = predictions[0]
        assert len(pred.segment_speeds) == 2
        speed_cam02 = pred.segment_speeds[0].wave_speed_kmh
        speed_cam01 = pred.segment_speeds[1].wave_speed_kmh
        assert speed_cam02 != speed_cam01
        assert speed_cam02 > speed_cam01
        assert pred.segment_speeds[0].local_speed_kmh == 40.0
        assert pred.segment_speeds[0].speed_source == "traffic_flow"
        assert pred.local_data_segments == 2
        assert pred.fallback_data_segments == 0
        assert pred.data_confidence == "high"

    def test_high_confidence_local_data_produces_narrow_uncertainty_band(
        self,
        engine: PhysicsEngine,
        bottleneck_state: CapacityState,
        chainage_map: dict[str, float],
    ) -> None:
        bottleneck_state.confidence = 0.95
        bottleneck_state.roi_length_confidence = "high"

        predictions = engine.compute(
            capacity_states=[bottleneck_state],
            sensor=None,
            camera_chainage_map=chainage_map,
            node_traffic_states={
                "CAM_01": SegmentTrafficState(
                    local_inflow_vph=4000.0,
                    local_speed_kmh=95.0,
                    inflow_source="traffic_flow",
                    speed_source="traffic_flow",
                    confidence="high",
                ),
                "CAM_02": SegmentTrafficState(
                    local_inflow_vph=4000.0,
                    local_speed_kmh=95.0,
                    inflow_source="traffic_flow",
                    speed_source="traffic_flow",
                    confidence="high",
                ),
                "CAM_03": SegmentTrafficState(
                    local_inflow_vph=4000.0,
                    local_speed_kmh=95.0,
                    inflow_source="traffic_flow",
                    speed_source="traffic_flow",
                    confidence="high",
                ),
            },
        )

        pred = predictions[0]
        length = pred.lengths_at_minutes[5]
        assert pred.uncertainty_level == "high"
        assert pred.prediction_confidence == pytest.approx(0.812)
        assert pred.length_lower_at_minutes[5] == pytest.approx(round(length * 0.85, 3))
        assert pred.length_upper_at_minutes[5] == pytest.approx(round(length * 1.15, 3))

    def test_fallback_data_produces_medium_uncertainty_band(
        self,
        engine: PhysicsEngine,
        bottleneck_state: CapacityState,
        upstream_sensor: SensorReading,
        chainage_map: dict[str, float],
    ) -> None:
        bottleneck_state.confidence = 0.95
        bottleneck_state.roi_length_confidence = "high"

        predictions = engine.compute(
            capacity_states=[bottleneck_state],
            sensor=upstream_sensor,
            camera_chainage_map=chainage_map,
        )

        pred = predictions[0]
        length = pred.lengths_at_minutes[5]
        assert pred.data_confidence == "medium"
        assert pred.uncertainty_level == "medium"
        assert pred.uncertainty_reason == "fallback segment data"
        assert pred.length_lower_at_minutes[5] == pytest.approx(round(length * 0.70, 3))
        assert pred.length_upper_at_minutes[5] == pytest.approx(round(length * 1.30, 3))

    def test_missing_data_and_low_camera_confidence_produces_widest_band(
        self,
        engine: PhysicsEngine,
        bottleneck_state: CapacityState,
        chainage_map: dict[str, float],
    ) -> None:
        bottleneck_state.confidence = 0.4
        bottleneck_state.roi_length_confidence = None

        predictions = engine.compute(
            capacity_states=[bottleneck_state],
            sensor=None,
            camera_chainage_map=chainage_map,
            node_traffic_states={
                "CAM_03": SegmentTrafficState(
                    local_inflow_vph=4000.0,
                    local_speed_kmh=95.0,
                    inflow_source="traffic_flow",
                    speed_source="traffic_flow",
                    confidence="high",
                ),
            },
        )

        pred = predictions[0]
        length = pred.lengths_at_minutes[5]
        assert pred.missing_data_segments == 1
        assert pred.uncertainty_level == "low"
        assert pred.uncertainty_reason == "missing segment data"
        assert pred.length_lower_at_minutes[5] == pytest.approx(round(length * 0.50, 3))
        assert pred.length_upper_at_minutes[5] == pytest.approx(round(length * 1.50, 3))

    def test_wave_speed_zero_halts_iteration(
        self,
        engine: PhysicsEngine,
        bottleneck_state: CapacityState,
        chainage_map: dict[str, float],
    ) -> None:
        """When local inflow ≤ bottleneck capacity, iteration stops (correction #3)."""
        sensor = SensorReading(
            timestamp=datetime(2026, 2, 16, 14, 0, 0),
            inflow_volume_vph=4000.0,
            average_speed_kmh=95.0,
        )
        predictions = engine.compute(
            capacity_states=[bottleneck_state],
            sensor=sensor,
            camera_chainage_map=chainage_map,
            node_inflows={
                "CAM_01": 800.0,   # Below bottleneck capacity → queue stops here
                "CAM_02": 4000.0,  # Normal inflow
                "CAM_03": 4000.0,  # Bottleneck
            },
        )
        assert len(predictions) == 1
        pred = predictions[0]
        # Should only have 1 segment (CAM_03→CAM_02), because at CAM_01
        # inflow (800) < capacity (1200), so wave_speed ≤ 0 → halt
        assert len(pred.segment_speeds) == 1
        assert pred.segment_speeds[0].to_camera == "CAM_02"

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

    def test_no_prediction_without_sensor_or_inflows(
        self,
        engine: PhysicsEngine,
        bottleneck_state: CapacityState,
    ) -> None:
        """Without sensor data or node_inflows, no physics computation."""
        predictions = engine.compute(
            capacity_states=[bottleneck_state],
            sensor=None,
        )
        assert len(predictions) == 0

    def test_global_sensor_fallback(
        self,
        engine: PhysicsEngine,
        bottleneck_state: CapacityState,
        upstream_sensor: SensorReading,
        chainage_map: dict[str, float],
    ) -> None:
        """When node_inflows not provided, falls back to global sensor."""
        predictions = engine.compute(
            capacity_states=[bottleneck_state],
            sensor=upstream_sensor,
            camera_chainage_map=chainage_map,
        )
        assert len(predictions) == 1
        pred = predictions[0]
        # All segments should use the global sensor's inflow
        for seg in pred.segment_speeds:
            assert seg.local_inflow_vph == 4000.0

    def test_missing_local_speed_uses_explicit_fallback_diagnostics(
        self,
        engine: PhysicsEngine,
        bottleneck_state: CapacityState,
        upstream_sensor: SensorReading,
    ) -> None:
        """Missing local speed falls back to aggregate sensor speed visibly."""
        predictions = engine.compute(
            capacity_states=[bottleneck_state],
            sensor=upstream_sensor,
            camera_chainage_map={"CAM_02": 0.5, "CAM_03": 1.0},
            node_traffic_states={
                "CAM_02": SegmentTrafficState(
                    local_inflow_vph=4000.0,
                    inflow_source="traffic_flow",
                    confidence="medium",
                ),
                "CAM_03": SegmentTrafficState(
                    local_inflow_vph=4000.0,
                    local_speed_kmh=55.0,
                    inflow_source="traffic_flow",
                    speed_source="traffic_flow",
                    confidence="high",
                ),
            },
        )

        assert len(predictions) == 1
        pred = predictions[0]
        assert len(pred.segment_speeds) == 1
        segment = pred.segment_speeds[0]
        assert segment.local_speed_kmh == upstream_sensor.average_speed_kmh
        assert segment.inflow_source == "traffic_flow"
        assert segment.speed_source == "aggregate"
        assert pred.local_data_segments == 0
        assert pred.fallback_data_segments == 1
        assert pred.data_confidence == "medium"

    def test_lengths_increase_with_time(
        self,
        engine: PhysicsEngine,
        bottleneck_state: CapacityState,
        upstream_sensor: SensorReading,
        chainage_map: dict[str, float],
    ) -> None:
        """Queue lengths should increase with time horizons."""
        predictions = engine.compute(
            capacity_states=[bottleneck_state],
            sensor=upstream_sensor,
            camera_chainage_map=chainage_map,
            node_inflows={
                "CAM_01": 4000.0,
                "CAM_02": 4000.0,
                "CAM_03": 4000.0,
            },
        )
        assert len(predictions) == 1
        lengths = predictions[0].lengths_at_minutes
        assert 1 in lengths
        assert 3 in lengths
        assert 5 in lengths
        assert 10 in lengths
        assert lengths[1] < lengths[3] < lengths[5] < lengths[10]

    def test_prediction_has_correct_coordinates(
        self,
        engine: PhysicsEngine,
        bottleneck_state: CapacityState,
        upstream_sensor: SensorReading,
        chainage_map: dict[str, float],
        coords_map: dict[str, tuple[float, float]],
    ) -> None:
        """Prediction should carry the bottleneck camera's coordinates."""
        predictions = engine.compute(
            capacity_states=[bottleneck_state],
            sensor=upstream_sensor,
            camera_chainage_map=chainage_map,
            camera_coords_map=coords_map,
        )
        pred = predictions[0]
        assert pred.origin_lat == 59.30
        assert pred.origin_lng == 18.00

    def test_no_prediction_when_drop_below_threshold(
        self,
        engine: PhysicsEngine,
        upstream_sensor: SensorReading,
        chainage_map: dict[str, float],
    ) -> None:
        """If capacity drop is too small, no prediction is generated."""
        state = CapacityState(
            timestamp=datetime(2026, 2, 16, 14, 0, 0),
            camera_id="CAM_03",
            vehicle_count=10,
            blocked_lanes=0,
            total_lanes=3,
            estimated_capacity_vph=3900.0,  # Close to inflow of 4000
            observed_density_veh_km_lane=50.0,  # Above k_critical to trigger
            is_anomaly=True,
            anomaly_reason="density_exceeds_k_critical",
        )
        predictions = engine.compute(
            capacity_states=[state],
            sensor=upstream_sensor,
            camera_chainage_map=chainage_map,
        )
        # 4000 - 3900 = 100, below MIN_CAPACITY_DROP_VPH (200)
        assert len(predictions) == 0

    def test_no_prediction_when_density_below_k_critical(
        self,
        engine: PhysicsEngine,
        upstream_sensor: SensorReading,
        chainage_map: dict[str, float],
    ) -> None:
        """Even with is_anomaly=True, density below k_critical should produce no prediction."""
        state = CapacityState(
            timestamp=datetime(2026, 2, 16, 14, 0, 0),
            camera_id="CAM_03",
            vehicle_count=5,
            blocked_lanes=1,
            total_lanes=3,
            estimated_capacity_vph=1200.0,
            observed_density_veh_km_lane=20.0,  # Below k_critical!
            is_anomaly=True,
            anomaly_reason="abnormal_aspect_ratio (1 boxes)",
        )
        predictions = engine.compute(
            capacity_states=[state],
            sensor=upstream_sensor,
            camera_chainage_map=chainage_map,
        )
        # Density below k_critical → physics engine should skip this
        assert len(predictions) == 0

    def test_weather_adjusted_critical_density_triggers_earlier(
        self,
        upstream_sensor: SensorReading,
        chainage_map: dict[str, float],
    ) -> None:
        """Degraded surface lowers the density threshold for conservative warnings."""
        engine = PhysicsEngine(critical_density_veh_km_lane=30.0)
        state = CapacityState(
            timestamp=datetime(2026, 2, 16, 14, 0, 0),
            camera_id="CAM_03",
            vehicle_count=5,
            blocked_lanes=0,
            total_lanes=3,
            estimated_capacity_vph=1200.0,
            observed_density_veh_km_lane=35.0,
            is_anomaly=False,
        )

        predictions = engine.compute(
            capacity_states=[state],
            sensor=upstream_sensor,
            camera_chainage_map=chainage_map,
        )

        assert len(predictions) == 1

    def test_single_camera_no_chainage_uses_legacy_fallback(
        self,
        engine: PhysicsEngine,
        bottleneck_state: CapacityState,
        upstream_sensor: SensorReading,
    ) -> None:
        """Without chainage map, falls back to single-segment behavior."""
        predictions = engine.compute(
            capacity_states=[bottleneck_state],
            sensor=upstream_sensor,
        )
        assert len(predictions) == 1
        pred = predictions[0]
        assert pred.growth_speed_kmh > 0
        # No segment data in legacy fallback
        assert len(pred.segment_speeds) == 0
        # Should still have lengths_at_minutes from linear projection
        assert 5 in pred.lengths_at_minutes

    def test_growth_speed_is_weighted_average(
        self,
        engine: PhysicsEngine,
        bottleneck_state: CapacityState,
        chainage_map: dict[str, float],
    ) -> None:
        """growth_speed_kmh should be the harmonic (distance-weighted) average."""
        sensor = SensorReading(
            timestamp=datetime(2026, 2, 16, 14, 0, 0),
            inflow_volume_vph=4000.0,
            average_speed_kmh=95.0,
        )
        predictions = engine.compute(
            capacity_states=[bottleneck_state],
            sensor=sensor,
            camera_chainage_map=chainage_map,
            node_inflows={
                "CAM_01": 4000.0,
                "CAM_02": 4000.0,
                "CAM_03": 4000.0,
            },
        )
        pred = predictions[0]
        # With uniform inflow, all segment speeds should be equal
        # and growth_speed_kmh should match them
        if len(pred.segment_speeds) >= 2:
            s0 = pred.segment_speeds[0].wave_speed_kmh
            s1 = pred.segment_speeds[1].wave_speed_kmh
            assert abs(s0 - s1) < 0.01  # Same inflow → same speed
            assert abs(pred.growth_speed_kmh - s0) < 0.01


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

    def test_node_inflows_without_sensor(
        self,
        engine: PhysicsEngine,
        bottleneck_state: CapacityState,
        chainage_map: dict[str, float],
    ) -> None:
        """node_inflows alone (no global sensor) should work."""
        predictions = engine.compute(
            capacity_states=[bottleneck_state],
            sensor=None,
            camera_chainage_map=chainage_map,
            node_inflows={
                "CAM_01": 4000.0,
                "CAM_02": 4000.0,
                "CAM_03": 4000.0,
            },
        )
        assert len(predictions) == 1
        # Should have segments even without global sensor
        assert len(predictions[0].segment_speeds) >= 1
