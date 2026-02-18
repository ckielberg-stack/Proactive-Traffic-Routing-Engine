from datetime import datetime

from src.models import CapacityState


def test_build_incidents_from_anomalies_with_coords() -> None:
    from src.incident_builder import build_incident_reports

    states = [
        CapacityState(
            timestamp=datetime(2026, 2, 16, 14, 0, 0),
            camera_id="CAM_A",
            vehicle_count=4,
            blocked_lanes=1,
            total_lanes=3,
            estimated_capacity_vph=2200.0,
            is_anomaly=True,
            anomaly_reason="vehicle_stopped",
            confidence=0.91,
        ),
        CapacityState(
            timestamp=datetime(2026, 2, 16, 14, 0, 0),
            camera_id="CAM_B",
            vehicle_count=10,
            blocked_lanes=0,
            total_lanes=3,
            estimated_capacity_vph=5500.0,
            is_anomaly=False,
            anomaly_reason=None,
            confidence=0.88,
        ),
    ]

    reports = build_incident_reports(
        states,
        camera_coords={"CAM_A": (59.30, 18.00)},
    )

    assert len(reports) == 1
    report = reports[0]
    assert report.camera_id == "CAM_A"
    assert report.incident_type == "vehicle_stopped"
    assert report.lanes_affected == 1
    assert report.capacity_drop_percentage == 66.7
    assert report.lat == 59.30
    assert report.lng == 18.00
    assert report.thumbnail_base64 is None


def test_build_incidents_defaults_and_clamps_capacity_drop() -> None:
    from src.incident_builder import build_incident_reports

    states = [
        CapacityState(
            timestamp=datetime(2026, 2, 16, 14, 5, 0),
            camera_id="CAM_X",
            vehicle_count=8,
            blocked_lanes=0,
            total_lanes=2,
            estimated_capacity_vph=99999.0,
            is_anomaly=True,
            anomaly_reason=None,
            confidence=0.70,
        ),
    ]

    reports = build_incident_reports(states, camera_coords={})
    report = reports[0]

    assert report.incident_type == "congestion"
    assert report.capacity_drop_percentage == 0.0
    assert report.lat is None
    assert report.lng is None
