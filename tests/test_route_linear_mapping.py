from datetime import datetime

from main_loop import (
    build_camera_chainage_map,
    build_node_inflows,
    build_node_traffic_states,
    build_travel_time_speed_states,
    _find_nearest_camera,
)
from src.models import SensorReading, TravelTimeReading


def test_camera_chainage_uses_route_order_not_latitude_sort() -> None:
    camera_coords = {
        "CAM_A": (0.0, 0.0),
        "CAM_B": (0.0, 0.01),
        "CAM_C": (-0.01, 0.02),
    }
    route_points = [
        camera_coords["CAM_A"],
        camera_coords["CAM_B"],
        camera_coords["CAM_C"],
    ]

    chainages = build_camera_chainage_map(
        camera_coords,
        route_points=route_points,
        corridor_length_km=10.0,
    )

    assert chainages["CAM_A"] < chainages["CAM_B"] < chainages["CAM_C"]


def test_node_inflows_use_route_position_and_sum_duplicate_sensors() -> None:
    now = datetime(2026, 6, 6, 12, 0, 0)
    camera_coords = {
        "CAM_A": (0.0, 0.0),
        "CAM_B": (0.0, 0.02),
        "CAM_C": (0.02, 0.02),
    }
    sensor_coords = {
        101: (0.0, 0.018),
        102: (0.0, 0.019),
    }
    route_points = [
        camera_coords["CAM_A"],
        camera_coords["CAM_B"],
        camera_coords["CAM_C"],
    ]
    sensor_readings = [
        SensorReading(
            timestamp=now,
            site_id=101,
            inflow_volume_vph=1200.0,
            average_speed_kmh=70.0,
        ),
        SensorReading(
            timestamp=now,
            site_id=102,
            inflow_volume_vph=800.0,
            average_speed_kmh=65.0,
        ),
    ]

    inflows = build_node_inflows(
        sensor_readings,
        camera_coords=camera_coords,
        sensor_coords=sensor_coords,
        route_points=route_points,
        corridor_length_km=10.0,
    )

    assert inflows == {"CAM_B": 2000.0}


def test_node_traffic_states_sum_inflow_and_weight_speed_by_volume() -> None:
    now = datetime(2026, 6, 6, 12, 0, 0)
    camera_coords = {
        "CAM_A": (0.0, 0.0),
        "CAM_B": (0.0, 0.02),
        "CAM_C": (0.02, 0.02),
    }
    sensor_coords = {
        101: (0.0, 0.018),
        102: (0.0, 0.019),
    }
    route_points = [
        camera_coords["CAM_A"],
        camera_coords["CAM_B"],
        camera_coords["CAM_C"],
    ]
    sensor_readings = [
        SensorReading(
            timestamp=now,
            site_id=101,
            inflow_volume_vph=1200.0,
            average_speed_kmh=70.0,
        ),
        SensorReading(
            timestamp=now,
            site_id=102,
            inflow_volume_vph=800.0,
            average_speed_kmh=50.0,
        ),
    ]

    states = build_node_traffic_states(
        sensor_readings,
        camera_coords=camera_coords,
        sensor_coords=sensor_coords,
        route_points=route_points,
        corridor_length_km=10.0,
    )

    assert states["CAM_B"].local_inflow_vph == 2000.0
    assert states["CAM_B"].local_speed_kmh == 62.0
    assert states["CAM_B"].inflow_source == "traffic_flow"
    assert states["CAM_B"].speed_source == "traffic_flow"
    assert states["CAM_B"].confidence == "high"


def test_node_traffic_states_average_speed_when_all_sensor_volumes_are_zero() -> None:
    now = datetime(2026, 6, 6, 12, 0, 0)
    camera_coords = {
        "CAM_A": (0.0, 0.0),
        "CAM_B": (0.0, 0.02),
    }
    sensor_coords = {
        101: (0.0, 0.019),
        102: (0.0, 0.0195),
    }
    route_points = [camera_coords["CAM_A"], camera_coords["CAM_B"]]
    sensor_readings = [
        SensorReading(
            timestamp=now,
            site_id=101,
            inflow_volume_vph=0.0,
            average_speed_kmh=30.0,
        ),
        SensorReading(
            timestamp=now,
            site_id=102,
            inflow_volume_vph=0.0,
            average_speed_kmh=60.0,
        ),
    ]

    states = build_node_traffic_states(
        sensor_readings,
        camera_coords=camera_coords,
        sensor_coords=sensor_coords,
        route_points=route_points,
        corridor_length_km=10.0,
    )

    assert states["CAM_B"].local_inflow_vph == 0.0
    assert states["CAM_B"].local_speed_kmh == 45.0


def test_travel_time_speed_states_use_only_ordered_northbound_routes() -> None:
    now = datetime(2026, 6, 6, 12, 0, 0)
    readings = [
        TravelTimeReading(
            timestamp=now,
            route_id="SB",
            name="Southbound",
            travel_time_seconds=10.0,
            free_flow_seconds=10.0,
            speed_kmh=10.0,
            length_meters=10000.0,
            traffic_status="heavy",
            delay_seconds=0.0,
        ),
        TravelTimeReading(
            timestamp=now,
            route_id="NB_2",
            name="Northbound 2",
            travel_time_seconds=10.0,
            free_flow_seconds=10.0,
            speed_kmh=80.0,
            length_meters=1000.0,
            traffic_status="freeflow",
            delay_seconds=0.0,
        ),
        TravelTimeReading(
            timestamp=now,
            route_id="NB_1",
            name="Northbound 1",
            travel_time_seconds=10.0,
            free_flow_seconds=10.0,
            speed_kmh=40.0,
            length_meters=1000.0,
            traffic_status="slow",
            delay_seconds=0.0,
        ),
    ]

    states = build_travel_time_speed_states(
        readings,
        camera_chainage_map={"CAM_A": 2.0, "CAM_B": 7.0},
        route_ids=["NB_1", "NB_2"],
        corridor_length_km=10.0,
    )

    assert states["CAM_A"].local_speed_kmh == 40.0
    assert states["CAM_B"].local_speed_kmh == 80.0
    assert states["CAM_A"].speed_source == "travel_time"
    assert "SB" not in states


def test_traffic_flow_speed_overrides_travel_time_fallback() -> None:
    now = datetime(2026, 6, 6, 12, 0, 0)
    camera_coords = {
        "CAM_A": (0.0, 0.0),
        "CAM_B": (0.0, 0.02),
    }
    route_points = [camera_coords["CAM_A"], camera_coords["CAM_B"]]
    sensor_readings = [
        SensorReading(
            timestamp=now,
            site_id=101,
            inflow_volume_vph=500.0,
            average_speed_kmh=65.0,
        )
    ]
    travel_time_readings = [
        TravelTimeReading(
            timestamp=now,
            route_id="724",
            name="Northbound",
            travel_time_seconds=10.0,
            free_flow_seconds=10.0,
            speed_kmh=35.0,
            length_meters=1000.0,
            traffic_status="slow",
            delay_seconds=0.0,
        )
    ]

    states = build_node_traffic_states(
        sensor_readings,
        travel_time_readings=travel_time_readings,
        camera_coords=camera_coords,
        sensor_coords={101: (0.0, 0.019)},
        route_points=route_points,
        corridor_length_km=10.0,
    )

    assert states["CAM_B"].local_speed_kmh == 65.0
    assert states["CAM_B"].speed_source == "traffic_flow"


def test_find_nearest_camera_uses_route_position_when_lng_is_available() -> None:
    camera_coords = {
        "CAM_A": (0.0, 0.0),
        "CAM_B": (0.0, 0.02),
        "CAM_C": (0.02, 0.02),
    }
    route_points = [
        camera_coords["CAM_A"],
        camera_coords["CAM_B"],
        camera_coords["CAM_C"],
    ]

    nearest = _find_nearest_camera(
        0.0,
        0.018,
        camera_coords=camera_coords,
        route_points=route_points,
        corridor_length_km=10.0,
    )

    assert nearest == "CAM_B"
