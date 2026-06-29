"""Tests for Situation accident/roadwork capacity inputs."""

from __future__ import annotations

from datetime import datetime

import pytest

from src import trafikverket_sources
from src.fusion_pipeline import apply_situation_capacity_impacts
from src.models import CapacityState, SituationDeviation
from src.trafikverket_sources import _situation_capacity_factor, fetch_situation_deviations

NOW = datetime(2026, 6, 13, 12, 0, 0)


def test_fetch_situation_deviations_filters_accidents_and_roadwork(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_api_request(xml_query: str) -> dict:
        assert 'objecttype="Situation"' in xml_query
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
                                        "MessageCode": "Olycka med påverkan",
                                        "SeverityCode": "high",
                                        "NumberOfLanesRestricted": 1,
                                        "LocationDescriptor": "Fittja",
                                        "Geometry": {
                                            "WGS84": "POINT (17.8619 59.2543)",
                                        },
                                    },
                                    {
                                        "Id": "RW-1",
                                        "RoadNumber": "E 4",
                                        "MessageType": "Vägarbete",
                                        "MessageCode": "Vägarbete",
                                        "SeverityCode": "medium",
                                        "NumberOfLanesRestricted": 0,
                                        "LocationDescriptor": "Kungens Kurva",
                                        "Geometry": {
                                            "WGS84": "POINT (17.9142 59.2725)",
                                        },
                                    },
                                    {
                                        "Id": "OTHER-ROAD",
                                        "RoadNumber": "73",
                                        "MessageType": "Olycka",
                                    },
                                    {
                                        "Id": "INFO-1",
                                        "RoadNumber": "E4",
                                        "MessageType": "Information",
                                    },
                                ]
                            }
                        ]
                    }
                ]
            }
        }

    monkeypatch.setattr(trafikverket_sources, "api_request", fake_api_request)
    monkeypatch.setattr(
        trafikverket_sources,
        "_find_nearest_camera",
        lambda _lat, _lng: "CAM_NEAREST",
    )

    deviations = fetch_situation_deviations(NOW)

    assert [d.deviation_id for d in deviations] == ["ACC-1", "RW-1"]
    assert [d.deviation_type for d in deviations] == ["accident", "roadwork"]
    assert all(d.chainage_km is not None for d in deviations)
    assert all(d.nearest_camera_id == "CAM_NEAREST" for d in deviations)


def test_fetch_situation_deviations_persists_raw_record_without_corridor_chainage(
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
                                        "Id": "ACC-OUT",
                                        "RoadNumber": "E4",
                                        "MessageType": "Accident",
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

    deviations = fetch_situation_deviations(NOW)

    assert len(deviations) == 1
    assert deviations[0].deviation_id == "ACC-OUT"
    assert deviations[0].chainage_km is None
    assert deviations[0].nearest_camera_id is None


def test_situation_capacity_factor_uses_type_severity_and_lanes() -> None:
    assert _situation_capacity_factor("accident", None) == 0.45
    assert _situation_capacity_factor("roadwork", None) == 0.65
    assert _situation_capacity_factor("accident", 1, total_lanes=2) == 0.45
    assert _situation_capacity_factor("accident", 1, "high", total_lanes=2) == 0.35
    assert _situation_capacity_factor("roadwork", 2, total_lanes=2) == 0.25


def test_apply_situation_impact_corrobates_existing_state() -> None:
    state = CapacityState(
        timestamp=NOW,
        camera_id="CAM_B",
        vehicle_count=12,
        blocked_lanes=0,
        total_lanes=2,
        estimated_capacity_vph=4000.0,
        observed_density_veh_km_lane=20.0,
        is_anomaly=False,
        confidence=0.4,
    )
    deviation = SituationDeviation(
        timestamp=NOW,
        deviation_id="ACC-1",
        deviation_type="accident",
        road_number="E4",
        chainage_km=2.0,
        nearest_camera_id="CAM_B",
        capacity_factor=0.45,
        number_of_lanes_restricted=1,
    )

    count = apply_situation_capacity_impacts(
        [state],
        [deviation],
        now=NOW,
        camera_chainage_map={"CAM_B": 2.0},
        critical_density_veh_km_lane=45.0,
    )

    assert count == 1
    assert state.is_anomaly is True
    assert state.estimated_capacity_vph == pytest.approx(1800.0)
    assert state.observed_density_veh_km_lane == pytest.approx(46.0)
    assert state.blocked_lanes == 1
    assert state.situation_confirmed is True
    assert state.situation_ids == ["ACC-1"]
    assert state.situation_types == ["accident"]


def test_apply_situation_impact_creates_synthetic_state() -> None:
    deviation = SituationDeviation(
        timestamp=NOW,
        deviation_id="RW-1",
        deviation_type="roadwork",
        road_number="E4",
        chainage_km=5.0,
        nearest_camera_id="CAM_X",
        capacity_factor=0.65,
    )
    states: list[CapacityState] = []

    count = apply_situation_capacity_impacts(
        states,
        [deviation],
        now=NOW,
        camera_chainage_map={"CAM_X": 5.0},
        critical_density_veh_km_lane=45.0,
    )

    assert count == 1
    assert len(states) == 1
    state = states[0]
    assert state.camera_id == "CAM_X"
    assert state.estimated_capacity_vph == pytest.approx(2600.0)
    assert state.observed_density_veh_km_lane == pytest.approx(46.0)
    assert state.anomaly_reason == "situation_confirmed_roadwork"
    assert state.situation_confirmed is True
