"""Narrow integration tests against shipped VMS and camera configs."""

from datetime import datetime
from pathlib import Path

import pytest

from src import trafikverket_sources
from src.fusion_pipeline import _aggregate_multi_roi_capacity
from src.models import (
    CameraMetadata,
    MultiSegmentCapacity,
    QueuePrediction,
    RoadSegmentState,
    SegmentTrafficState,
    VMSStatusSnapshot,
)
from src.operator_api import _match_proxy_ground_truth
from src.physics_engine import PhysicsEngine
from src.roi_mapper import ROIMapper
from src.trafikverket_sources import fetch_vms_status
from src.vms_orchestrator import VMSOrchestrator


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VMS_CONFIG = PROJECT_ROOT / "vms_config.json"
CAMERA_CONFIG = PROJECT_ROOT / "camera_config.json"
NOW = datetime(2026, 6, 10, 12, 0, 0)


def test_vms_orchestrator_uses_shipped_gantry_names_for_recommendations() -> None:
    orchestrator = VMSOrchestrator(config_path=VMS_CONFIG)

    gantry_by_id = {gantry.vms_id: gantry for gantry in orchestrator.gantries}
    assert gantry_by_id["VMS-4003"].name == "Kungens Kurva"
    assert gantry_by_id["VMS-4003"].road == "E4"

    prediction = QueuePrediction(
        timestamp=NOW,
        camera_id="SE_STA_CAMERA_Orion_466",
        origin_lat=59.28,
        origin_lng=17.93,
        origin_chainage_km=7.0,
        growth_speed_kmh=12.0,
        lengths_at_minutes={1: 0.2, 3: 0.6, 5: 1.0},
    )

    recommendations = orchestrator.generate_recommendations(
        prediction,
        time_horizons=[1, 3, 5],
        now=NOW,
    )

    assert recommendations
    rec_by_id = {rec.vms_id: rec for rec in recommendations}
    rec = rec_by_id["VMS-4003"]
    assert rec.vms_name == "Kungens Kurva"
    assert "E4" not in rec.vms_name
    assert rec.triggering_camera_id == "SE_STA_CAMERA_Orion_466"
    assert rec.recommended_message.startswith(("KÖVARNING", "VARNING"))


def test_proxy_ground_truth_matches_shipped_gantry_by_road_metadata() -> None:
    orchestrator = VMSOrchestrator(config_path=VMS_CONFIG)
    recommendation = orchestrator.generate_recommendations(
        QueuePrediction(
            timestamp=NOW,
            camera_id="SE_STA_CAMERA_Orion_466",
            origin_lat=59.28,
            origin_lng=17.93,
            origin_chainage_km=7.0,
            growth_speed_kmh=12.0,
            lengths_at_minutes={1: 0.2},
        ),
        time_horizons=[1],
        now=NOW,
    )[0]
    assert recommendation.vms_id == "VMS-4003"
    assert recommendation.vms_name == "Kungens Kurva"

    active, speed_limit, deviation_id = _match_proxy_ground_truth(
        recommendation,
        [
            VMSStatusSnapshot(
                timestamp=NOW,
                vms_id="SE_STA_SPEEDMANAGEMENTID_1_999",
                vms_name="E4 — active speed-management deviation",
                is_active=True,
                displayed_message="70 km/h",
                speed_limit=70,
                road_number="E4",
                geometry_wgs84="POINT (17.914 59.272)",
                lat=59.272,
                lng=17.914,
                chainage_km=5.1,
            )
        ],
    )

    assert active is True
    assert speed_limit == 70
    assert deviation_id == "SE_STA_SPEEDMANAGEMENTID_1_999"


def test_fetch_vms_status_carries_situation_road_and_chainage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_api_request(_xml_query: str) -> dict:
        return {
            "RESPONSE": {
                "RESULT": [
                    {
                        "Situation": [
                            {
                                "Deviation": [
                                    {
                                        "Id": "SE_STA_SPEEDMANAGEMENTID_1_999",
                                        "RoadNumber": "E4",
                                        "TemporaryLimit": "70 km/h",
                                        "LocationDescriptor": "Kungens Kurva",
                                        "Geometry": {
                                            "WGS84": "POINT (17.914 59.272)",
                                        },
                                    }
                                ]
                            }
                        ]
                    }
                ]
            }
        }

    monkeypatch.setattr(trafikverket_sources, "api_request", fake_api_request)

    statuses = fetch_vms_status(NOW)

    active_status = next(
        s for s in statuses
        if s.vms_id == "SE_STA_SPEEDMANAGEMENTID_1_999"
    )
    assert active_status.road_number == "E4"
    assert active_status.geometry_wgs84 == "POINT (17.914 59.272)"
    assert active_status.lat == pytest.approx(59.272)
    assert active_status.lng == pytest.approx(17.914)
    assert active_status.chainage_km is not None
    assert active_status.chainage_km == pytest.approx(5.1, abs=0.5)


def test_fetch_vms_status_rejects_out_of_corridor_e4_chainage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_api_request(_xml_query: str) -> dict:
        return {
            "RESPONSE": {
                "RESULT": [
                    {
                        "Situation": [
                            {
                                "Deviation": [
                                    {
                                        "Id": "SE_STA_SPEEDMANAGEMENTID_1_888",
                                        "RoadNumber": "E4",
                                        "TemporaryLimit": "70 km/h",
                                        "LocationDescriptor": "Outside corridor",
                                        "Geometry": {
                                            "WGS84": "POINT (18.40 59.80)",
                                        },
                                    }
                                ]
                            }
                        ]
                    }
                ]
            }
        }

    monkeypatch.setattr(trafikverket_sources, "api_request", fake_api_request)

    statuses = fetch_vms_status(NOW)

    active_status = next(
        s for s in statuses
        if s.vms_id == "SE_STA_SPEEDMANAGEMENTID_1_888"
    )
    assert active_status.road_number == "E4"
    assert active_status.lat == pytest.approx(59.80)
    assert active_status.lng == pytest.approx(18.40)
    assert active_status.chainage_km is None


def test_southbound_roi_from_shipped_camera_config_does_not_predict_northbound() -> None:
    mapper = ROIMapper(CAMERA_CONFIG)
    rois = mapper.get_rois("SE_STA_CAMERA_Orion_466")
    southbound = next(roi for roi in rois if roi.road_id == "E4_Southbound")
    northbound = next(roi for roi in rois if roi.road_id == "E4_Northbound")

    multi_state = MultiSegmentCapacity(
        timestamp=NOW,
        camera_id="SE_STA_CAMERA_Orion_466",
        segments=[
            RoadSegmentState(
                road_id=southbound.road_id,
                direction=southbound.direction_relative_to_camera,
                vehicle_count=10,
                capacity_vph=1200.0,
                observed_density_veh_km_lane=60.0,
                num_lanes=southbound.num_lanes,
                is_anomaly=True,
                anomaly_reason="density_exceeds_k_critical",
                confidence=0.9,
            ),
            RoadSegmentState(
                road_id=northbound.road_id,
                direction=northbound.direction_relative_to_camera,
                vehicle_count=2,
                capacity_vph=northbound.capacity_vph,
                observed_density_veh_km_lane=10.0,
                num_lanes=northbound.num_lanes,
                confidence=0.8,
            ),
        ],
    )
    camera_meta = CameraMetadata(
        camera_id="SE_STA_CAMERA_Orion_466",
        name="Orion 466",
        lat=59.272,
        lng=17.914,
        num_lanes=southbound.num_lanes + northbound.num_lanes,
    )

    state, road_segments = _aggregate_multi_roi_capacity(multi_state, camera_meta)
    assert road_segments["E4_Southbound"]["density_veh_km_lane"] == 60.0
    assert road_segments["E4_Northbound"]["density_veh_km_lane"] == 10.0
    assert state.traffic_direction == "northbound"
    assert state.road_id == "E4_Northbound"
    assert state.observed_density_veh_km_lane == 10.0

    predictions = PhysicsEngine().compute(
        [state],
        sensor=None,
        camera_chainage_map={
            "SOUTH_OF_CAMERA": 4.5,
            "SE_STA_CAMERA_Orion_466": 5.0,
            "NORTH_OF_CAMERA": 5.5,
        },
        camera_coords_map={"SE_STA_CAMERA_Orion_466": (59.272, 17.914)},
        node_traffic_states={
            "SOUTH_OF_CAMERA": SegmentTrafficState(
                local_inflow_vph=7000.0,
                local_speed_kmh=90.0,
                inflow_source="traffic_flow",
                speed_source="traffic_flow",
                confidence="high",
            ),
            "SE_STA_CAMERA_Orion_466": SegmentTrafficState(
                local_inflow_vph=7000.0,
                local_speed_kmh=50.0,
                inflow_source="traffic_flow",
                speed_source="traffic_flow",
                confidence="high",
            ),
            "NORTH_OF_CAMERA": SegmentTrafficState(
                local_inflow_vph=7000.0,
                local_speed_kmh=90.0,
                inflow_source="traffic_flow",
                speed_source="traffic_flow",
                confidence="high",
            ),
        },
        now=NOW,
    )

    assert predictions == []
