"""Regression tests for multi-ROI density handoff into physics."""

from datetime import datetime

import pytest

from main_loop import _aggregate_multi_roi_capacity
from src.models import CameraMetadata, MultiSegmentCapacity, RoadSegmentState
from src.physics_engine import PhysicsEngine


def test_multi_roi_aggregate_uses_northbound_density_for_physics() -> None:
    multi_state = MultiSegmentCapacity(
        timestamp=datetime.now(),
        camera_id="CAM_03",
        segments=[
            RoadSegmentState(
                road_id="E4_NB",
                direction="towards",
                vehicle_count=3,
                capacity_vph=4000.0,
                observed_density_veh_km_lane=10.0,
                num_lanes=2,
                confidence=0.8,
            ),
            RoadSegmentState(
                road_id="E4_SB",
                direction="away",
                vehicle_count=12,
                capacity_vph=1000.0,
                observed_density_veh_km_lane=60.0,
                num_lanes=2,
                is_anomaly=True,
                anomaly_reason="density_exceeds_k_critical",
                confidence=0.9,
            ),
        ],
    )
    camera_meta = CameraMetadata(
        camera_id="CAM_03",
        name="Camera 03",
        lat=59.0,
        lng=18.0,
        num_lanes=4,
    )

    state, road_segments = _aggregate_multi_roi_capacity(
        multi_state, camera_meta,
    )

    assert state.observed_density_veh_km_lane == 10.0
    assert state.traffic_direction == "northbound"
    assert state.road_id == "E4_NB"
    assert state.is_anomaly is False
    assert state.anomaly_reason is None
    assert road_segments["E4_SB"]["density_veh_km_lane"] == 60.0
    assert road_segments["E4_SB"]["traffic_direction"] == "southbound"


def test_southbound_only_density_does_not_reach_northbound_physics_engine() -> None:
    multi_state = MultiSegmentCapacity(
        timestamp=datetime.now(),
        camera_id="CAM_03",
        segments=[
            RoadSegmentState(
                road_id="E4_SB",
                direction="away",
                vehicle_count=12,
                capacity_vph=1000.0,
                observed_density_veh_km_lane=60.0,
                num_lanes=2,
                is_anomaly=True,
                anomaly_reason="density_exceeds_k_critical",
                confidence=0.9,
            ),
        ],
    )
    camera_meta = CameraMetadata(
        camera_id="CAM_03",
        name="Camera 03",
        lat=59.0,
        lng=18.0,
        num_lanes=2,
    )
    state, _road_segments = _aggregate_multi_roi_capacity(
        multi_state, camera_meta,
    )

    assert state.observed_density_veh_km_lane == 0.0
    assert state.is_anomaly is False

    predictions = PhysicsEngine().compute(
        [state],
        sensor=None,
        camera_chainage_map={"CAM_02": 1.0, "CAM_03": 2.0},
        camera_coords_map={"CAM_03": (59.0, 18.0)},
        node_inflows={"CAM_02": 2200.0, "CAM_03": 2200.0},
    )

    assert predictions == []


def test_northbound_density_above_critical_reaches_physics_engine() -> None:
    multi_state = MultiSegmentCapacity(
        timestamp=datetime.now(),
        camera_id="CAM_03",
        segments=[
            RoadSegmentState(
                road_id="E4_NB",
                direction="towards",
                vehicle_count=12,
                capacity_vph=1000.0,
                observed_density_veh_km_lane=60.0,
                num_lanes=2,
                is_anomaly=True,
                anomaly_reason="density_exceeds_k_critical",
                confidence=0.9,
            ),
            RoadSegmentState(
                road_id="E4_SB",
                direction="away",
                vehicle_count=2,
                capacity_vph=4000.0,
                observed_density_veh_km_lane=10.0,
                num_lanes=2,
                confidence=0.8,
            ),
        ],
    )
    camera_meta = CameraMetadata(
        camera_id="CAM_03",
        name="Camera 03",
        lat=59.0,
        lng=18.0,
        num_lanes=4,
    )
    state, _road_segments = _aggregate_multi_roi_capacity(
        multi_state, camera_meta,
    )

    predictions = PhysicsEngine().compute(
        [state],
        sensor=None,
        camera_chainage_map={"CAM_02": 1.0, "CAM_03": 2.0},
        camera_coords_map={"CAM_03": (59.0, 18.0)},
        node_inflows={"CAM_02": 2200.0, "CAM_03": 2200.0},
    )

    assert state.observed_density_veh_km_lane == 60.0
    assert state.traffic_direction == "northbound"
    assert len(predictions) == 1
    assert predictions[0].camera_id == "CAM_03"


def test_road_segment_density_rejects_negative_values() -> None:
    with pytest.raises(Exception):
        RoadSegmentState(
            road_id="E4_SB",
            direction="away",
            vehicle_count=1,
            capacity_vph=1000.0,
            observed_density_veh_km_lane=-0.1,
        )
