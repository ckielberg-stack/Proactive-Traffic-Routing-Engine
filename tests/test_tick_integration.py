"""Offline integration coverage for the tick orchestration path."""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

import config
import main_loop
from src.models import (
    CapacityState,
    MultiSegmentCapacity,
    RoadSegmentState,
    TickResult,
)
from src.vms_orchestrator import VMSOrchestrator


class _NoRoiMapper:
    def has_rois(self, _camera_id: str) -> bool:
        return False


class _HasRoiMapper:
    def has_rois(self, _camera_id: str) -> bool:
        return True


class _NoopRetention:
    def maybe_retain(
        self,
        _raw_bytes: bytes,
        _camera_id: str,
        _now: datetime,
        _state: CapacityState,
    ) -> None:
        return None


class _FakeVisionEngine:
    def analyze_array(self, _frame: np.ndarray, meta) -> CapacityState:
        self.last_vehicle_detections = [
            {
                "xyxy": (10.0, 10.0, 50.0, 50.0),
                "confidence": 0.88,
                "class_id": 2,
                "class_name": "car",
            }
        ]
        return CapacityState(
            timestamp=datetime(2026, 6, 10, 12, 0, 0),
            camera_id=meta.camera_id,
            vehicle_count=18,
            blocked_lanes=0,
            total_lanes=2,
            estimated_capacity_vph=1000.0,
            observed_density_veh_km_lane=70.0,
            road_id="E4_Northbound",
            traffic_direction="northbound",
            is_anomaly=False,
            confidence=0.92,
        )


class _FakeMultiRoiVisionEngine:
    def analyze_multi_roi(
        self,
        _frame: np.ndarray,
        meta,
        _roi_mapper,
    ) -> MultiSegmentCapacity:
        return MultiSegmentCapacity(
            timestamp=datetime(2026, 6, 10, 12, 0, 0),
            camera_id=meta.camera_id,
            segments=[
                RoadSegmentState(
                    road_id="E4_Northbound",
                    direction="away",
                    vehicle_count=18,
                    capacity_vph=1000.0,
                    observed_density_veh_km_lane=70.0,
                    num_lanes=2,
                    confidence=0.92,
                ),
                RoadSegmentState(
                    road_id="E4_Southbound",
                    direction="towards",
                    vehicle_count=20,
                    capacity_vph=800.0,
                    observed_density_veh_km_lane=90.0,
                    num_lanes=2,
                    is_anomaly=True,
                    anomaly_reason="density_exceeds_k_critical",
                    confidence=0.9,
                ),
            ],
            detections_by_road_id={
                "E4_Northbound": [
                    {
                        "xyxy": (10.0, 10.0, 50.0, 50.0),
                        "confidence": 0.88,
                        "class_id": 2,
                        "class_name": "car",
                    }
                ],
                "E4_Southbound": [
                    {
                        "xyxy": (100.0, 100.0, 150.0, 150.0),
                        "confidence": 0.9,
                        "class_id": 2,
                        "class_name": "car",
                    }
                ],
            },
        )


class _FakeSouthboundOnlyMultiRoiVisionEngine:
    def analyze_multi_roi(
        self,
        _frame: np.ndarray,
        meta,
        _roi_mapper,
    ) -> MultiSegmentCapacity:
        return MultiSegmentCapacity(
            timestamp=datetime(2026, 6, 10, 12, 0, 0),
            camera_id=meta.camera_id,
            segments=[
                RoadSegmentState(
                    road_id="E4_Northbound",
                    direction="away",
                    vehicle_count=1,
                    capacity_vph=4000.0,
                    observed_density_veh_km_lane=5.0,
                    num_lanes=2,
                    confidence=0.8,
                ),
                RoadSegmentState(
                    road_id="E4_Southbound",
                    direction="towards",
                    vehicle_count=1,
                    capacity_vph=4000.0,
                    observed_density_veh_km_lane=5.0,
                    num_lanes=2,
                    confidence=0.9,
                ),
            ],
            detections_by_road_id={
                "E4_Southbound": [
                    {
                        "xyxy": (100.0, 100.0, 150.0, 150.0),
                        "confidence": 0.9,
                        "class_id": 2,
                        "class_name": "car",
                    }
                ],
            },
        )


def test_fused_capacity_uses_local_speed_for_congested_density() -> None:
    state = CapacityState(
        timestamp=datetime(2026, 6, 10, 12, 0, 0),
        camera_id="CAM_B",
        vehicle_count=18,
        blocked_lanes=0,
        total_lanes=2,
        estimated_capacity_vph=4000.0,
        observed_density_veh_km_lane=70.0,
        confidence=0.9,
    )

    main_loop._derive_capacity_from_fused_state(
        state,
        local_speed_kmh=20.0,
        fallback_speed_kmh=110.0,
    )

    assert state.estimated_capacity_vph == pytest.approx(2800.0)
    assert state.is_anomaly is True
    assert state.anomaly_reason == "density_exceeds_k_critical"


def test_fused_capacity_honors_zero_local_speed() -> None:
    state = CapacityState(
        timestamp=datetime(2026, 6, 10, 12, 0, 0),
        camera_id="CAM_B",
        vehicle_count=18,
        blocked_lanes=0,
        total_lanes=2,
        estimated_capacity_vph=4000.0,
        observed_density_veh_km_lane=70.0,
        confidence=0.9,
    )

    main_loop._derive_capacity_from_fused_state(
        state,
        local_speed_kmh=0.0,
        fallback_speed_kmh=110.0,
    )

    assert state.estimated_capacity_vph == 0.0
    assert state.is_anomaly is True
    assert state.anomaly_reason == "density_exceeds_k_critical"


@pytest.mark.parametrize("is_anomaly", [False, True])
def test_fused_capacity_preserves_unavailable_zero_capacity_frame(
    is_anomaly: bool,
) -> None:
    state = CapacityState(
        timestamp=datetime(2026, 6, 10, 12, 0, 0),
        camera_id="CAM_B",
        vehicle_count=0,
        blocked_lanes=0,
        total_lanes=2,
        estimated_capacity_vph=0.0,
        observed_density_veh_km_lane=0.0,
        is_anomaly=is_anomaly,
        anomaly_reason="image_unreadable" if is_anomaly else None,
        confidence=0.0,
    )

    main_loop._derive_capacity_from_fused_state(
        state,
        local_speed_kmh=20.0,
        fallback_speed_kmh=110.0,
    )

    assert state.estimated_capacity_vph == 0.0
    assert state.is_anomaly is is_anomaly


def test_fetch_cameras_isolates_camera_failures_and_preserves_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = datetime(2026, 6, 10, 12, 0, 0)

    def fake_api_request(_xml_query: str) -> dict:
        return {
            "RESPONSE": {
                "RESULT": [
                    {
                        "Camera": [
                            {
                                "Id": "CAM_A",
                                "Name": "Camera A",
                                "PhotoUrl": "https://example.invalid/cam-a.jpg",
                                "HasFullSizePhoto": False,
                            },
                            {
                                "Id": "CAM_B",
                                "Name": "Camera B",
                                "PhotoUrl": "https://example.invalid/cam-b.jpg",
                                "HasFullSizePhoto": False,
                            },
                            {
                                "Id": "CAM_C",
                                "Name": "Camera C",
                                "PhotoUrl": "https://example.invalid/cam-c.jpg",
                                "HasFullSizePhoto": False,
                            },
                            {
                                "Id": "CAM_D",
                                "Name": "Camera D",
                                "PhotoUrl": "https://example.invalid/cam-d.jpg",
                                "HasFullSizePhoto": False,
                            },
                        ]
                    }
                ]
            }
        }

    def fake_fetch_image_bytes(url: str) -> bytes:
        if "cam-a" in url:
            time.sleep(0.02)
        if "cam-b" in url:
            return None
        if "cam-c" in url:
            raise RuntimeError("camera transport down")
        return b"offline-jpeg"

    created_engines: list[_FakeVisionEngine] = []

    def fake_worker_engine() -> _FakeVisionEngine:
        engine = _FakeVisionEngine()
        created_engines.append(engine)
        return engine

    monkeypatch.setattr(main_loop, "api_request", fake_api_request)
    monkeypatch.setattr(main_loop, "fetch_image_bytes", fake_fetch_image_bytes)
    monkeypatch.setattr(
        main_loop,
        "decode_frame",
        lambda _raw: np.zeros((16, 16, 3), dtype=np.uint8),
    )
    monkeypatch.setattr(main_loop, "_get_camera_worker_vision_engine", fake_worker_engine)
    monkeypatch.setattr(main_loop, "_get_retention_policy", lambda: _NoopRetention())
    monkeypatch.setattr(main_loop, "_get_roi_mapper", lambda: _NoRoiMapper())
    monkeypatch.setattr(main_loop, "_camera_worker_count", lambda _count: 2)

    records, states = main_loop.fetch_cameras(
        ["CAM_A", "CAM_B", "CAM_C", "CAM_D"],
        now,
    )

    assert [record["camera_id"] for record in records] == [
        "CAM_A",
        "CAM_B",
        "CAM_C",
        "CAM_D",
    ]
    assert [record["status"] for record in records] == [
        "ok",
        "fetch_failed",
        "error",
        "ok",
    ]
    assert records[2]["error"] == "camera transport down"
    assert all(isinstance(record["duration_ms"], int) for record in records)
    assert [state.camera_id for state in states] == ["CAM_A", "CAM_D"]
    assert len(created_engines) == 2
    assert created_engines[0] is not created_engines[1]


@pytest.fixture
def reset_main_loop_globals(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(main_loop, "_tick_count", 0)
    monkeypatch.setattr(main_loop, "_start_time", None)
    monkeypatch.setattr(main_loop, "_camera_chainage_map", None)
    monkeypatch.setattr(main_loop, "_vision_engine", None)
    monkeypatch.setattr(main_loop, "_retention_policy", None)
    monkeypatch.setattr(main_loop, "_roi_mapper", None)
    monkeypatch.setattr(main_loop, "_physics_engine", None)
    monkeypatch.setattr(main_loop, "_vms_orchestrator", None)
    monkeypatch.setattr(main_loop, "_density_smoother", None)
    monkeypatch.setattr(main_loop, "_track_persistence", None)
    monkeypatch.setattr(main_loop, "_travel_time_calibrator", None)
    monkeypatch.setattr(main_loop, "_weather_adapter", None)


def test_tick_once_runs_offline_through_camera_sensor_travel_time_vms_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    reset_main_loop_globals: None,
) -> None:
    camera_coords = {
        "CAM_A": (59.2417, 17.8366),
        "CAM_B": (59.2543, 17.8619),
    }
    sensor_coords = {
        101: camera_coords["CAM_A"],
        102: camera_coords["CAM_B"],
    }
    api_calls: list[str] = []
    fetched_urls: list[str] = []

    def fake_api_request(xml_query: str) -> dict:
        api_calls.append(xml_query)

        if 'objecttype="Camera"' in xml_query:
            return {
                "RESPONSE": {
                    "RESULT": [
                        {
                            "Camera": [
                                {
                                    "Id": "CAM_B",
                                    "Name": "Offline camera B",
                                    "PhotoUrl": "https://example.invalid/cam-b.jpg",
                                    "HasFullSizePhoto": True,
                                }
                            ]
                        }
                    ]
                }
            }

        if 'objecttype="TrafficFlow"' in xml_query:
            return {
                "RESPONSE": {
                    "RESULT": [
                        {
                            "TrafficFlow": [
                                {
                                    "SiteId": 101,
                                    "VehicleFlowRate": 5000,
                                    "AverageVehicleSpeed": 70,
                                    "SpecificLane": "1",
                                },
                                {
                                    "SiteId": 102,
                                    "VehicleFlowRate": 5000,
                                    "AverageVehicleSpeed": 20,
                                    "SpecificLane": "1",
                                },
                            ]
                        }
                    ]
                }
            }

        if 'objecttype="Situation"' in xml_query:
            return {
                "RESPONSE": {
                    "RESULT": [
                        {
                            "Situation": [
                                {
                                    "Deviation": [
                                        {
                                            "Id": "ACC-1",
                                            "RoadNumber": "E4",
                                            "MessageType": "Olycka",
                                            "MessageCode": "Olycka",
                                            "SeverityCode": "medium",
                                            "NumberOfLanesRestricted": 1,
                                            "LocationDescriptor": "Fittja",
                                            "Geometry": {
                                                "WGS84": "POINT (17.8619 59.2543)",
                                            },
                                        }
                                    ]
                                }
                            ]
                        }
                    ]
                }
            }

        if 'objecttype="TravelTimeRoute"' in xml_query:
            return {
                "RESPONSE": {
                    "RESULT": [
                        {
                            "TravelTimeRoute": [
                                {
                                    "Id": "724",
                                    "Name": "Offline route 724",
                                    "TravelTime": 180,
                                    "FreeFlowTravelTime": 160,
                                    "Speed": 62,
                                    "Length": 3000,
                                    "TrafficStatus": "slow",
                                }
                            ]
                        }
                    ]
                }
            }

        if 'objecttype="WeatherMeasurepoint"' in xml_query:
            return {
                "RESPONSE": {
                    "RESULT": [
                        {
                            "WeatherMeasurepoint": [
                                {
                                    "Id": "W-1",
                                    "Name": "Offline weather",
                                    "Geometry": {"WGS84": "POINT (17.8619 59.2543)"},
                                    "Observation": {
                                        "Sample": datetime(2026, 6, 10, 12, 0, 0).isoformat(),
                                        "Air": {
                                            "Temperature": {"Value": 1.0},
                                            "RelativeHumidity": {"Value": 90},
                                            "Dewpoint": {"Value": 0.0},
                                            "VisibleDistance": {"Value": 4000},
                                        },
                                        "Surface": {"Temperature": {"Value": -1.0}},
                                        "Wind": [
                                            {
                                                "Speed": {"Value": 4.0},
                                                "Direction": {"Value": 180},
                                            }
                                        ],
                                        "Weather": {"Precipitation": "Snö"},
                                        "Aggregated5minutes": {
                                            "Precipitation": {
                                                "RainSum": {"Value": 0.0},
                                                "SnowSum": {
                                                    "WaterEquivalent": {"Value": 0.2}
                                                },
                                            }
                                        },
                                    },
                                }
                            ]
                        }
                    ]
                }
            }

        if 'objecttype="RoadCondition"' in xml_query:
            return {
                "RESPONSE": {
                    "RESULT": [
                        {
                            "RoadCondition": [
                                {
                                    "Id": "RC-1",
                                    "LocationText": "E4 Kungens kurva",
                                    "ConditionText": "Halka",
                                    "ConditionInfo": ["Isfläckar"],
                                    "ConditionCode": "ice",
                                    "Warning": True,
                                    "RoadNumber": "E4",
                                    "StartTime": "2026-06-10T11:55:00",
                                    "Geometry": {"WGS84": "POINT (17.8619 59.2543)"},
                                }
                            ]
                        }
                    ]
                }
            }

        raise AssertionError(f"Unexpected Trafikverket query: {xml_query}")

    def fake_fetch_image_bytes(url: str) -> bytes:
        fetched_urls.append(url)
        return b"offline-jpeg"

    monkeypatch.setattr(main_loop, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(main_loop, "api_request", fake_api_request)
    monkeypatch.setattr(main_loop, "fetch_image_bytes", fake_fetch_image_bytes)
    monkeypatch.setattr(
        main_loop,
        "decode_frame",
        lambda _raw: np.zeros((16, 16, 3), dtype=np.uint8),
    )
    monkeypatch.setattr(
        main_loop,
        "_get_camera_worker_vision_engine",
        lambda: _FakeVisionEngine(),
    )
    monkeypatch.setattr(main_loop, "_get_retention_policy", lambda: _NoopRetention())
    monkeypatch.setattr(main_loop, "_get_roi_mapper", lambda: _NoRoiMapper())
    monkeypatch.setattr(main_loop, "CAMERA_COORDS", camera_coords)
    monkeypatch.setattr(main_loop, "SENSOR_COORDS", sensor_coords)
    monkeypatch.setattr(config, "SENSOR_SITE_IDS", [101, 102])
    monkeypatch.setattr(
        main_loop,
        "_build_camera_chainage_map",
        lambda: {"CAM_A": 1.0, "CAM_B": 2.0},
    )
    monkeypatch.setattr(
        main_loop,
        "_vms_orchestrator",
        VMSOrchestrator(
            config_path=Path(__file__).resolve().parents[1] / "vms_config.json"
        ),
    )

    result = main_loop.tick_once(["CAM_B"])

    assert isinstance(result, TickResult)
    assert result.tick_number == 1
    assert [state.camera_id for state in result.capacity_states] == ["CAM_B"]
    assert result.capacity_states[0].observed_density_veh_km_lane == pytest.approx(
        70.0
    )
    assert result.capacity_states[0].estimated_capacity_vph == pytest.approx(1800.0)
    assert result.capacity_states[0].situation_confirmed is True
    assert result.capacity_states[0].situation_ids == ["ACC-1"]
    assert len(result.sensor_readings) == 2
    assert {reading.site_id for reading in result.sensor_readings} == {101, 102}
    assert [reading.route_id for reading in result.travel_time_readings] == ["724"]
    assert [record["station_id"] for record in result.weather_records] == ["W-1"]
    assert [record["id"] for record in result.road_condition_records] == ["RC-1"]
    assert result.weather_adjustment is not None
    assert result.weather_adjustment.surface_state == "ice"
    assert [d.deviation_id for d in result.situation_deviations] == ["ACC-1"]
    assert result.vms_statuses
    assert any(
        status.vms_id == "VMS-4001" and not status.is_active
        for status in result.vms_statuses
    )
    assert len(result.queue_predictions) == 1
    prediction = result.queue_predictions[0]
    assert prediction.camera_id == "CAM_B"
    assert prediction.local_data_segments >= 1
    assert prediction.segment_speeds
    assert result.vms_recommendations
    assert any(
        rec.triggering_camera_id == "CAM_B"
        for rec in result.vms_recommendations
    )
    assert any(
        rec.triggering_camera_id == "road_condition_RC-1"
        for rec in result.vms_recommendations
    )

    assert any('objecttype="Camera"' in query for query in api_calls)
    assert any('objecttype="TrafficFlow"' in query for query in api_calls)
    assert sum('objecttype="Situation"' in query for query in api_calls) == 2
    assert any('objecttype="TravelTimeRoute"' in query for query in api_calls)
    assert any('objecttype="WeatherMeasurepoint"' in query for query in api_calls)
    assert any('objecttype="RoadCondition"' in query for query in api_calls)
    assert fetched_urls == ["https://example.invalid/cam-b.jpg?type=fullsize"]

    day_dir = tmp_path / result.timestamp.strftime("%Y-%m-%d")
    jsonl_path = day_dir / "sensor_data.jsonl"
    assert jsonl_path.exists()
    records = [
        json.loads(line)
        for line in jsonl_path.read_text(encoding="utf-8").splitlines()
    ]
    record_types = {record["type"] for record in records}
    assert {
        "vision_result",
        "sensor_reading",
        "vms_status",
        "queue_prediction",
        "vms_recommendation",
        "travel_time",
        "calibration",
        "weather",
        "road_condition",
        "weather_adjustment",
        "situation",
    }.issubset(record_types)
    queue_record = next(record for record in records if record["type"] == "queue_prediction")
    assert "prediction_confidence" in queue_record
    assert "uncertainty_level" in queue_record
    assert "uncertainty_reason" in queue_record
    assert "length_lower_at_minutes" in queue_record
    assert "length_upper_at_minutes" in queue_record
    vms_record = next(record for record in records if record["type"] == "vms_recommendation")
    assert "eta_lower_minutes" in vms_record
    assert "eta_upper_minutes" in vms_record
    assert "confidence" in vms_record
    assert "uncertainty_level" in vms_record
    assert (tmp_path / "vision_state.json").exists()
    assert (tmp_path / "status.json").exists()


def test_tick_once_promotes_persistent_vehicle_to_stopped_anomaly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    reset_main_loop_globals: None,
) -> None:
    camera_coords = {"CAM_A": (59.2417, 17.8366)}
    sensor_coords = {101: camera_coords["CAM_A"]}

    def fake_api_request(xml_query: str) -> dict:
        if 'objecttype="Camera"' in xml_query:
            return {
                "RESPONSE": {
                    "RESULT": [
                        {
                            "Camera": [
                                {
                                    "Id": "CAM_A",
                                    "Name": "Camera A",
                                    "PhotoUrl": "https://example.invalid/cam-a.jpg",
                                    "HasFullSizePhoto": False,
                                }
                            ]
                        }
                    ]
                }
            }
        if 'objecttype="TrafficFlow"' in xml_query:
            return {
                "RESPONSE": {
                    "RESULT": [
                        {
                            "TrafficFlow": [
                                {
                                    "SiteId": 101,
                                    "VehicleFlowRate": 1200,
                                    "AverageVehicleSpeed": 90,
                                }
                            ]
                        }
                    ]
                }
            }
        if 'objecttype="Situation"' in xml_query:
            return {"RESPONSE": {"RESULT": [{"Situation": []}]}}
        if 'objecttype="TravelTimeRoute"' in xml_query:
            return {"RESPONSE": {"RESULT": [{"TravelTimeRoute": []}]}}
        if 'objecttype="WeatherMeasurepoint"' in xml_query:
            return {"RESPONSE": {"RESULT": [{"WeatherMeasurepoint": []}]}}
        if 'objecttype="RoadCondition"' in xml_query:
            return {"RESPONSE": {"RESULT": [{"RoadCondition": []}]}}
        raise AssertionError(f"Unexpected Trafikverket query: {xml_query}")

    monkeypatch.setattr(main_loop, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(main_loop, "api_request", fake_api_request)
    monkeypatch.setattr(main_loop, "fetch_image_bytes", lambda _url: b"offline-jpeg")
    monkeypatch.setattr(
        main_loop,
        "decode_frame",
        lambda _raw: np.zeros((16, 16, 3), dtype=np.uint8),
    )
    monkeypatch.setattr(
        main_loop,
        "_get_camera_worker_vision_engine",
        lambda: _FakeVisionEngine(),
    )
    monkeypatch.setattr(main_loop, "_get_retention_policy", lambda: _NoopRetention())
    monkeypatch.setattr(main_loop, "_get_roi_mapper", lambda: _NoRoiMapper())
    monkeypatch.setattr(main_loop, "CAMERA_COORDS", camera_coords)
    monkeypatch.setattr(main_loop, "SENSOR_COORDS", sensor_coords)
    monkeypatch.setattr(config, "SENSOR_SITE_IDS", [101])
    monkeypatch.setattr(main_loop, "_build_camera_chainage_map", lambda: {"CAM_A": 1.0})
    monkeypatch.setattr(
        main_loop,
        "_vms_orchestrator",
        VMSOrchestrator(
            config_path=Path(__file__).resolve().parents[1] / "vms_config.json"
        ),
    )

    main_loop.tick_once(["CAM_A"])
    main_loop.tick_once(["CAM_A"])
    result = main_loop.tick_once(["CAM_A"])

    state = result.capacity_states[0]
    assert state.is_anomaly is True
    assert state.anomaly_reason == "vehicle_stopped"
    assert state.blocked_lanes == 1
    assert state.observed_density_veh_km_lane > main_loop.K_CRITICAL_VEH_KM_LANE
    assert state.confidence == pytest.approx(0.92)

    jsonl_path = tmp_path / result.timestamp.strftime("%Y-%m-%d") / "sensor_data.jsonl"
    records = [
        json.loads(line)
        for line in jsonl_path.read_text(encoding="utf-8").splitlines()
    ]
    stopped_records = [
        record
        for record in records
        if record.get("type") == "vision_result"
        and record.get("anomaly_reason") == "vehicle_stopped"
    ]
    assert stopped_records
    assert stopped_records[-1]["stopped_vehicle"]["persistence_ticks"] == 3

    anomaly_log = tmp_path / "anomaly_log.jsonl"
    assert anomaly_log.exists()
    assert "vehicle_stopped" in anomaly_log.read_text(encoding="utf-8")

    vision_state = json.loads((tmp_path / "vision_state.json").read_text(encoding="utf-8"))
    assert vision_state["cameras"][0]["anomaly_reason"] == "vehicle_stopped"


def test_tick_once_situation_only_deviation_creates_physics_bottleneck(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    reset_main_loop_globals: None,
) -> None:
    camera_coords = {
        "CAM_A": (59.2417, 17.8366),
        "CAM_B": (59.2543, 17.8619),
    }
    sensor_coords = {101: camera_coords["CAM_B"]}
    api_calls: list[str] = []

    def fake_api_request(xml_query: str) -> dict:
        api_calls.append(xml_query)
        if 'objecttype="TrafficFlow"' in xml_query:
            return {
                "RESPONSE": {
                    "RESULT": [
                        {
                            "TrafficFlow": [
                                {
                                    "SiteId": 101,
                                    "VehicleFlowRate": 5000,
                                    "AverageVehicleSpeed": 70,
                                }
                            ]
                        }
                    ]
                }
            }
        if 'objecttype="Situation"' in xml_query:
            if "Hastighetsbegränsning gäller" in xml_query:
                return {"RESPONSE": {"RESULT": [{"Situation": []}]}}
            return {
                "RESPONSE": {
                    "RESULT": [
                        {
                            "Situation": [
                                {
                                    "Deviation": [
                                        {
                                            "Id": "ACC-SYN",
                                            "RoadNumber": "E4",
                                            "MessageType": "Accident",
                                            "MessageCode": "Accident",
                                            "SeverityCode": "high",
                                            "NumberOfLanesRestricted": 1,
                                            "LocationDescriptor": "Fittja",
                                            "Geometry": {
                                                "WGS84": "POINT (17.8619 59.2543)",
                                            },
                                        }
                                    ]
                                }
                            ]
                        }
                    ]
                }
            }
        if 'objecttype="TravelTimeRoute"' in xml_query:
            return {"RESPONSE": {"RESULT": [{"TravelTimeRoute": []}]}}
        if 'objecttype="WeatherMeasurepoint"' in xml_query:
            return {"RESPONSE": {"RESULT": [{"WeatherMeasurepoint": []}]}}
        if 'objecttype="RoadCondition"' in xml_query:
            return {"RESPONSE": {"RESULT": [{"RoadCondition": []}]}}
        raise AssertionError(f"Unexpected Trafikverket query: {xml_query}")

    monkeypatch.setattr(main_loop, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(main_loop, "api_request", fake_api_request)
    monkeypatch.setattr(main_loop, "CAMERA_COORDS", camera_coords)
    monkeypatch.setattr(main_loop, "SENSOR_COORDS", sensor_coords)
    monkeypatch.setattr(config, "SENSOR_SITE_IDS", [101])
    monkeypatch.setattr(
        main_loop,
        "_build_camera_chainage_map",
        lambda: {"CAM_A": 1.0, "CAM_B": 2.0},
    )
    monkeypatch.setattr(main_loop, "_find_nearest_camera", lambda _lat, _lng: "CAM_B")
    monkeypatch.setattr(
        main_loop,
        "_vms_orchestrator",
        VMSOrchestrator(
            config_path=Path(__file__).resolve().parents[1] / "vms_config.json"
        ),
    )

    result = main_loop.tick_once([])

    assert [d.deviation_id for d in result.situation_deviations] == ["ACC-SYN"]
    assert len(result.capacity_states) == 1
    state = result.capacity_states[0]
    assert state.camera_id == "CAM_B"
    assert state.anomaly_reason == "situation_confirmed_accident"
    assert state.situation_confirmed is True
    assert state.estimated_capacity_vph == pytest.approx(1400.0)
    assert len(result.queue_predictions) == 1
    assert result.queue_predictions[0].camera_id == "CAM_B"
    assert not any('objecttype="Camera"' in query for query in api_calls)
    assert sum('objecttype="Situation"' in query for query in api_calls) == 2

    jsonl_path = (
        tmp_path
        / result.timestamp.strftime("%Y-%m-%d")
        / "sensor_data.jsonl"
    )
    records = [
        json.loads(line)
        for line in jsonl_path.read_text(encoding="utf-8").splitlines()
    ]
    assert any(record["type"] == "situation" for record in records)


def test_tick_once_runs_offline_through_multi_roi_aggregation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    reset_main_loop_globals: None,
) -> None:
    camera_coords = {
        "CAM_A": (59.2417, 17.8366),
        "CAM_B": (59.2543, 17.8619),
    }
    sensor_coords = {
        101: camera_coords["CAM_A"],
        102: camera_coords["CAM_B"],
    }

    def fake_api_request(xml_query: str) -> dict:
        if 'objecttype="Camera"' in xml_query:
            return {
                "RESPONSE": {
                    "RESULT": [
                        {
                            "Camera": [
                                {
                                    "Id": "CAM_B",
                                    "Name": "Offline camera B",
                                    "PhotoUrl": "https://example.invalid/cam-b.jpg",
                                    "HasFullSizePhoto": False,
                                }
                            ]
                        }
                    ]
                }
            }
        if 'objecttype="TrafficFlow"' in xml_query:
            return {
                "RESPONSE": {
                    "RESULT": [
                        {
                            "TrafficFlow": [
                                {
                                    "SiteId": 101,
                                    "VehicleFlowRate": 5000,
                                    "AverageVehicleSpeed": 70,
                                    "SpecificLane": "1",
                                },
                                {
                                    "SiteId": 102,
                                    "VehicleFlowRate": 5000,
                                    "AverageVehicleSpeed": 20,
                                    "SpecificLane": "1",
                                },
                            ]
                        }
                    ]
                }
            }
        if 'objecttype="Situation"' in xml_query:
            return {"RESPONSE": {"RESULT": [{"Situation": []}]}}
        if 'objecttype="TravelTimeRoute"' in xml_query:
            return {"RESPONSE": {"RESULT": [{"TravelTimeRoute": []}]}}
        if 'objecttype="WeatherMeasurepoint"' in xml_query:
            return {"RESPONSE": {"RESULT": [{"WeatherMeasurepoint": []}]}}
        if 'objecttype="RoadCondition"' in xml_query:
            return {"RESPONSE": {"RESULT": [{"RoadCondition": []}]}}
        raise AssertionError(f"Unexpected Trafikverket query: {xml_query}")

    monkeypatch.setattr(main_loop, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(main_loop, "api_request", fake_api_request)
    monkeypatch.setattr(main_loop, "fetch_image_bytes", lambda _url: b"offline-jpeg")
    monkeypatch.setattr(
        main_loop,
        "decode_frame",
        lambda _raw: np.zeros((16, 16, 3), dtype=np.uint8),
    )
    monkeypatch.setattr(
        main_loop,
        "_get_camera_worker_vision_engine",
        lambda: _FakeMultiRoiVisionEngine(),
    )
    monkeypatch.setattr(main_loop, "_get_retention_policy", lambda: _NoopRetention())
    monkeypatch.setattr(main_loop, "_get_roi_mapper", lambda: _HasRoiMapper())
    monkeypatch.setattr(main_loop, "CAMERA_COORDS", camera_coords)
    monkeypatch.setattr(main_loop, "SENSOR_COORDS", sensor_coords)
    monkeypatch.setattr(config, "SENSOR_SITE_IDS", [101, 102])
    monkeypatch.setattr(
        main_loop,
        "_build_camera_chainage_map",
        lambda: {"CAM_A": 1.0, "CAM_B": 2.0},
    )
    monkeypatch.setattr(
        main_loop,
        "_vms_orchestrator",
        VMSOrchestrator(
            config_path=Path(__file__).resolve().parents[1] / "vms_config.json"
        ),
    )

    result = main_loop.tick_once(["CAM_B"])

    assert len(result.capacity_states) == 1
    state = result.capacity_states[0]
    assert state.camera_id == "CAM_B"
    assert state.road_id == "E4_Northbound"
    assert state.traffic_direction == "northbound"
    assert state.vehicle_count == 18
    assert state.observed_density_veh_km_lane == pytest.approx(70.0)
    assert state.estimated_capacity_vph == pytest.approx(2800.0)
    assert result.weather_records == []
    assert result.road_condition_records == []
    assert result.weather_adjustment is not None
    assert result.weather_adjustment.surface_state == "dry"
    assert len(result.queue_predictions) == 1
    assert result.queue_predictions[0].camera_id == "CAM_B"
    assert any(
        rec.triggering_camera_id == "CAM_B"
        for rec in result.vms_recommendations
    )

    jsonl_path = (
        tmp_path
        / result.timestamp.strftime("%Y-%m-%d")
        / "sensor_data.jsonl"
    )
    records = [
        json.loads(line)
        for line in jsonl_path.read_text(encoding="utf-8").splitlines()
    ]
    vision_record = next(record for record in records if record["type"] == "vision_result")
    assert vision_record["road_segments"]["E4_Northbound"]["density_veh_km_lane"] == 70.0
    assert vision_record["road_segments"]["E4_Southbound"]["density_veh_km_lane"] == 90.0
    assert vision_record["road_segments"]["E4_Southbound"]["traffic_direction"] == "southbound"


def test_tick_once_ignores_southbound_only_persistent_vehicle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    reset_main_loop_globals: None,
) -> None:
    camera_coords = {"CAM_B": (59.2543, 17.8619)}
    sensor_coords = {102: camera_coords["CAM_B"]}

    def fake_api_request(xml_query: str) -> dict:
        if 'objecttype="Camera"' in xml_query:
            return {
                "RESPONSE": {
                    "RESULT": [
                        {
                            "Camera": [
                                {
                                    "Id": "CAM_B",
                                    "Name": "Offline camera B",
                                    "PhotoUrl": "https://example.invalid/cam-b.jpg",
                                    "HasFullSizePhoto": False,
                                }
                            ]
                        }
                    ]
                }
            }
        if 'objecttype="TrafficFlow"' in xml_query:
            return {
                "RESPONSE": {
                    "RESULT": [
                        {
                            "TrafficFlow": [
                                {
                                    "SiteId": 102,
                                    "VehicleFlowRate": 1200,
                                    "AverageVehicleSpeed": 90,
                                }
                            ]
                        }
                    ]
                }
            }
        if 'objecttype="Situation"' in xml_query:
            return {"RESPONSE": {"RESULT": [{"Situation": []}]}}
        if 'objecttype="TravelTimeRoute"' in xml_query:
            return {"RESPONSE": {"RESULT": [{"TravelTimeRoute": []}]}}
        if 'objecttype="WeatherMeasurepoint"' in xml_query:
            return {"RESPONSE": {"RESULT": [{"WeatherMeasurepoint": []}]}}
        if 'objecttype="RoadCondition"' in xml_query:
            return {"RESPONSE": {"RESULT": [{"RoadCondition": []}]}}
        raise AssertionError(f"Unexpected Trafikverket query: {xml_query}")

    monkeypatch.setattr(main_loop, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(main_loop, "api_request", fake_api_request)
    monkeypatch.setattr(main_loop, "fetch_image_bytes", lambda _url: b"offline-jpeg")
    monkeypatch.setattr(
        main_loop,
        "decode_frame",
        lambda _raw: np.zeros((16, 16, 3), dtype=np.uint8),
    )
    monkeypatch.setattr(
        main_loop,
        "_get_camera_worker_vision_engine",
        lambda: _FakeSouthboundOnlyMultiRoiVisionEngine(),
    )
    monkeypatch.setattr(main_loop, "_get_retention_policy", lambda: _NoopRetention())
    monkeypatch.setattr(main_loop, "_get_roi_mapper", lambda: _HasRoiMapper())
    monkeypatch.setattr(main_loop, "CAMERA_COORDS", camera_coords)
    monkeypatch.setattr(main_loop, "SENSOR_COORDS", sensor_coords)
    monkeypatch.setattr(config, "SENSOR_SITE_IDS", [102])
    monkeypatch.setattr(main_loop, "_build_camera_chainage_map", lambda: {"CAM_B": 1.0})
    monkeypatch.setattr(
        main_loop,
        "_vms_orchestrator",
        VMSOrchestrator(
            config_path=Path(__file__).resolve().parents[1] / "vms_config.json"
        ),
    )

    main_loop.tick_once(["CAM_B"])
    main_loop.tick_once(["CAM_B"])
    result = main_loop.tick_once(["CAM_B"])

    assert result.capacity_states[0].anomaly_reason != "vehicle_stopped"
    jsonl_path = tmp_path / result.timestamp.strftime("%Y-%m-%d") / "sensor_data.jsonl"
    records = [
        json.loads(line)
        for line in jsonl_path.read_text(encoding="utf-8").splitlines()
    ]
    assert not any(record.get("stopped_vehicle") for record in records)
