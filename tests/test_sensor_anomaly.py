"""
Unit tests for sensor-based anomaly detection.

Tests that speed drops below road speed limits are correctly detected,
classified by severity, and produce VMS recommendations when severe.
"""

from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pytest

from config import (
    DEFAULT_ROAD_SPEED_LIMIT,
    SENSOR_COORDS,
    SENSOR_ROAD_SPEED_LIMITS,
    SENSOR_SEVERE_DROP_RATIO,
    SENSOR_SPEED_DROP_RATIO,
)
from src.models import SensorAnomaly, SensorReading, VMSGantry, VMSRecommendation
from src.vms_orchestrator import VMSOrchestrator


# ======================================================================
# Fixtures
# ======================================================================


@pytest.fixture
def now() -> datetime:
    return datetime(2026, 2, 18, 14, 0, 0)


@pytest.fixture
def normal_reading(now: datetime) -> SensorReading:
    """A sensor reporting normal speed (above threshold)."""
    return SensorReading(
        timestamp=now,
        site_id=2790,
        inflow_volume_vph=3000.0,
        average_speed_kmh=60.0,  # 86% of 70 → fine
    )


@pytest.fixture
def warning_reading(now: datetime) -> SensorReading:
    """A sensor reporting a speed drop that triggers a warning (50%)."""
    return SensorReading(
        timestamp=now,
        site_id=2790,
        inflow_volume_vph=2500.0,
        average_speed_kmh=30.0,  # 43% of 70 → warning (< 50%, >= 35%)
    )


@pytest.fixture
def severe_reading(now: datetime) -> SensorReading:
    """A sensor reporting a severe speed drop (< 35% of limit)."""
    return SensorReading(
        timestamp=now,
        site_id=2790,
        inflow_volume_vph=2000.0,
        average_speed_kmh=20.0,  # 29% of 70 → severe
    )


@pytest.fixture
def borderline_reading(now: datetime) -> SensorReading:
    """Speed exactly at the 50% threshold boundary → no anomaly."""
    return SensorReading(
        timestamp=now,
        site_id=2790,
        inflow_volume_vph=3500.0,
        average_speed_kmh=35.0,  # Exactly 50% — threshold is >=, so no anomaly
    )


@pytest.fixture
def vms_config_path(tmp_path: Path) -> Path:
    """Create a minimal VMS config for testing."""
    import json

    config = {
        "gantries": [
            {
                "vms_id": "VMS-TEST-001",
                "name": "Test Gantry South",
                "lat": 59.30,
                "lng": 18.00,
                "chainage_km": 5.0,
            },
            {
                "vms_id": "VMS-TEST-002",
                "name": "Test Gantry North",
                "lat": 59.35,
                "lng": 18.01,
                "chainage_km": 10.0,
            },
        ]
    }
    config_path = tmp_path / "vms_config.json"
    config_path.write_text(json.dumps(config))
    return config_path


@pytest.fixture
def orchestrator(vms_config_path: Path) -> VMSOrchestrator:
    return VMSOrchestrator(config_path=vms_config_path)


# ======================================================================
# detect_sensor_anomalies tests
# ======================================================================


class TestDetectSensorAnomalies:
    """Tests for the detect_sensor_anomalies function in main_loop."""

    def test_no_anomaly_above_threshold(
        self, normal_reading: SensorReading, now: datetime
    ) -> None:
        """Speed 60 on a 70 road → no anomaly (86% of limit)."""
        from main_loop import detect_sensor_anomalies

        anomalies = detect_sensor_anomalies([normal_reading], now)
        assert len(anomalies) == 0

    def test_detects_speed_below_threshold(
        self, warning_reading: SensorReading, now: datetime
    ) -> None:
        """Speed 30 on a 70 road → anomaly with severity 'warning'."""
        from main_loop import detect_sensor_anomalies

        anomalies = detect_sensor_anomalies([warning_reading], now)
        assert len(anomalies) == 1
        assert anomalies[0].severity == "warning"
        assert anomalies[0].site_id == 2790
        assert anomalies[0].measured_speed_kmh == 30.0
        assert anomalies[0].road_speed_limit_kmh == 70

    def test_detects_severe_speed_drop(
        self, severe_reading: SensorReading, now: datetime
    ) -> None:
        """Speed 20 on a 70 road → anomaly with severity 'severe'."""
        from main_loop import detect_sensor_anomalies

        anomalies = detect_sensor_anomalies([severe_reading], now)
        assert len(anomalies) == 1
        assert anomalies[0].severity == "severe"
        assert anomalies[0].measured_speed_kmh == 20.0

    def test_borderline_no_anomaly(
        self, borderline_reading: SensorReading, now: datetime
    ) -> None:
        """Speed exactly at 50% threshold → no anomaly (>= check)."""
        from main_loop import detect_sensor_anomalies

        anomalies = detect_sensor_anomalies([borderline_reading], now)
        assert len(anomalies) == 0

    def test_anomaly_has_coordinates(
        self, warning_reading: SensorReading, now: datetime
    ) -> None:
        """Anomaly lat/lng populated from SENSOR_COORDS."""
        from main_loop import detect_sensor_anomalies

        anomalies = detect_sensor_anomalies([warning_reading], now)
        assert len(anomalies) == 1
        # Station 2790 should have coordinates from SENSOR_COORDS
        expected_coords = SENSOR_COORDS.get(2790)
        if expected_coords:
            assert anomalies[0].lat == expected_coords[0]
            assert anomalies[0].lng == expected_coords[1]

    def test_speed_ratio_calculation(
        self, warning_reading: SensorReading, now: datetime
    ) -> None:
        """Speed ratio is correctly calculated as measured/limit."""
        from main_loop import detect_sensor_anomalies

        anomalies = detect_sensor_anomalies([warning_reading], now)
        assert len(anomalies) == 1
        # 30 / 70 ≈ 0.429
        assert abs(anomalies[0].speed_ratio - 30.0 / 70.0) < 0.01

    def test_uses_station_specific_limit(self, now: datetime) -> None:
        """Station with a specific speed limit uses that limit."""
        from main_loop import detect_sensor_anomalies

        # Create a reading for a station we know is in the config
        reading = SensorReading(
            timestamp=now,
            site_id=1274,  # Hallunda — 70 km/h in config
            inflow_volume_vph=2000.0,
            average_speed_kmh=30.0,  # 43% → warning
        )
        anomalies = detect_sensor_anomalies([reading], now)
        assert len(anomalies) == 1
        assert anomalies[0].road_speed_limit_kmh == 70

    def test_unknown_station_uses_default(self, now: datetime) -> None:
        """Unknown station falls back to DEFAULT_ROAD_SPEED_LIMIT."""
        from main_loop import detect_sensor_anomalies

        reading = SensorReading(
            timestamp=now,
            site_id=99999,  # Not in SENSOR_ROAD_SPEED_LIMITS
            inflow_volume_vph=2000.0,
            average_speed_kmh=20.0,
        )
        anomalies = detect_sensor_anomalies([reading], now)
        assert len(anomalies) == 1
        assert anomalies[0].road_speed_limit_kmh == DEFAULT_ROAD_SPEED_LIMIT

    def test_multiple_readings_mixed(self, now: datetime) -> None:
        """Multiple readings: one normal, one warning, one severe."""
        from main_loop import detect_sensor_anomalies

        readings = [
            SensorReading(
                timestamp=now, site_id=2790,
                inflow_volume_vph=3000.0, average_speed_kmh=60.0,  # OK
            ),
            SensorReading(
                timestamp=now, site_id=2786,
                inflow_volume_vph=2500.0, average_speed_kmh=30.0,  # Warning
            ),
            SensorReading(
                timestamp=now, site_id=2788,
                inflow_volume_vph=1500.0, average_speed_kmh=15.0,  # Severe
            ),
        ]
        anomalies = detect_sensor_anomalies(readings, now)
        assert len(anomalies) == 2
        severities = {a.severity for a in anomalies}
        assert "warning" in severities
        assert "severe" in severities

    def test_skip_reading_without_site_id(self, now: datetime) -> None:
        """Readings without a site_id are silently skipped."""
        from main_loop import detect_sensor_anomalies

        reading = SensorReading(
            timestamp=now,
            site_id=None,
            inflow_volume_vph=2000.0,
            average_speed_kmh=10.0,  # Would be severe if checked
        )
        anomalies = detect_sensor_anomalies([reading], now)
        assert len(anomalies) == 0


# ======================================================================
# VMS sensor recommendation tests
# ======================================================================


class TestSensorVMSRecommendations:
    """Tests for generate_sensor_recommendations in VMSOrchestrator."""

    def test_severe_anomaly_generates_vms_recommendation(
        self, orchestrator: VMSOrchestrator, now: datetime
    ) -> None:
        """A severe anomaly produces a VMS recommendation."""
        anomaly = SensorAnomaly(
            timestamp=now,
            site_id=2790,
            measured_speed_kmh=20.0,
            road_speed_limit_kmh=70,
            speed_ratio=0.286,
            volume_vph=2000.0,
            severity="severe",
            lat=59.3321,
            lng=18.012,
        )
        recs = orchestrator.generate_sensor_recommendations([anomaly], now=now)
        assert len(recs) == 1
        assert recs[0].urgency == "immediate"
        assert "KÖVARNING" in recs[0].recommended_message
        assert "Sensorlarm" in recs[0].summary

    def test_warning_anomaly_no_vms_recommendation(
        self, orchestrator: VMSOrchestrator, now: datetime
    ) -> None:
        """A warning anomaly does NOT produce a VMS recommendation."""
        anomaly = SensorAnomaly(
            timestamp=now,
            site_id=2790,
            measured_speed_kmh=30.0,
            road_speed_limit_kmh=70,
            speed_ratio=0.429,
            volume_vph=2500.0,
            severity="warning",
            lat=59.3321,
            lng=18.012,
        )
        recs = orchestrator.generate_sensor_recommendations([anomaly], now=now)
        assert len(recs) == 0

    def test_no_duplicate_vms_for_multiple_severe(
        self, orchestrator: VMSOrchestrator, now: datetime
    ) -> None:
        """Multiple severe anomalies near the same VMS → only one rec."""
        anomalies = [
            SensorAnomaly(
                timestamp=now, site_id=2790,
                measured_speed_kmh=20.0, road_speed_limit_kmh=70,
                speed_ratio=0.286, volume_vph=2000.0, severity="severe",
                lat=59.3321, lng=18.012,
            ),
            SensorAnomaly(
                timestamp=now, site_id=2786,
                measured_speed_kmh=18.0, road_speed_limit_kmh=70,
                speed_ratio=0.257, volume_vph=1800.0, severity="severe",
                lat=59.3330, lng=18.013,
            ),
        ]
        recs = orchestrator.generate_sensor_recommendations(anomalies, now=now)
        # Both are near the same gantry, so should be deduplicated
        vms_ids = [r.vms_id for r in recs]
        assert len(vms_ids) == len(set(vms_ids))

    def test_recommendation_fields_populated(
        self, orchestrator: VMSOrchestrator, now: datetime
    ) -> None:
        """VMS recommendation fields are correctly populated."""
        anomaly = SensorAnomaly(
            timestamp=now,
            site_id=2790,
            measured_speed_kmh=15.0,
            road_speed_limit_kmh=70,
            speed_ratio=0.214,
            volume_vph=1500.0,
            severity="severe",
            nearest_camera_id="SE_STA_CAMERA_0_50438756",
            lat=59.3321,
            lng=18.012,
        )
        recs = orchestrator.generate_sensor_recommendations([anomaly], now=now)
        assert len(recs) == 1
        rec = recs[0]
        assert rec.timestamp == now
        assert rec.estimated_activation_minutes == 0.0
        assert rec.queue_growth_speed_kmh == 0.0
        assert rec.triggering_camera_id == "SE_STA_CAMERA_0_50438756"

    def test_empty_anomalies_no_recommendations(
        self, orchestrator: VMSOrchestrator, now: datetime
    ) -> None:
        """Empty anomaly list → no recommendations."""
        recs = orchestrator.generate_sensor_recommendations([], now=now)
        assert len(recs) == 0

    def test_no_gantries_no_recommendations(
        self, tmp_path: Path, now: datetime
    ) -> None:
        """Orchestrator with no gantries → no recommendations."""
        import json

        config_path = tmp_path / "empty_config.json"
        config_path.write_text(json.dumps({"gantries": []}))
        empty_orch = VMSOrchestrator(config_path=config_path)

        anomaly = SensorAnomaly(
            timestamp=now, site_id=2790,
            measured_speed_kmh=10.0, road_speed_limit_kmh=70,
            speed_ratio=0.143, volume_vph=1000.0, severity="severe",
            lat=59.3321, lng=18.012,
        )
        recs = empty_orch.generate_sensor_recommendations([anomaly], now=now)
        assert len(recs) == 0


# ======================================================================
# find_nearest_vms_by_lat tests
# ======================================================================


class TestFindNearestVMSByLat:
    def test_finds_nearest_gantry(self, orchestrator: VMSOrchestrator) -> None:
        """Picks the gantry closest by latitude."""
        # lat=59.31 is closer to Test Gantry South (59.30)
        vms = orchestrator.find_nearest_vms_by_lat(59.31)
        assert vms is not None
        assert vms.vms_id == "VMS-TEST-001"

    def test_finds_north_gantry(self, orchestrator: VMSOrchestrator) -> None:
        """lat=59.34 is closer to Test Gantry North (59.35)."""
        vms = orchestrator.find_nearest_vms_by_lat(59.34)
        assert vms is not None
        assert vms.vms_id == "VMS-TEST-002"

    def test_no_gantries_returns_none(self, tmp_path: Path) -> None:
        """Empty gantry list → returns None."""
        import json

        config_path = tmp_path / "empty.json"
        config_path.write_text(json.dumps({"gantries": []}))
        empty_orch = VMSOrchestrator(config_path=config_path)
        assert empty_orch.find_nearest_vms_by_lat(59.30) is None


# ======================================================================
# SensorAnomaly model tests
# ======================================================================


class TestSensorAnomalyModel:
    def test_model_creation(self, now: datetime) -> None:
        """SensorAnomaly can be created with all required fields."""
        anomaly = SensorAnomaly(
            timestamp=now,
            site_id=2790,
            measured_speed_kmh=35.0,
            road_speed_limit_kmh=70,
            speed_ratio=0.5,
            volume_vph=2500.0,
            severity="warning",
            lat=59.3321,
            lng=18.012,
        )
        assert anomaly.site_id == 2790
        assert anomaly.severity == "warning"
        assert anomaly.nearest_camera_id is None

    def test_model_serialization(self, now: datetime) -> None:
        """SensorAnomaly serializes to dict correctly."""
        anomaly = SensorAnomaly(
            timestamp=now,
            site_id=2790,
            measured_speed_kmh=35.0,
            road_speed_limit_kmh=70,
            speed_ratio=0.5,
            volume_vph=2500.0,
            severity="warning",
            nearest_camera_id="CAM_01",
            lat=59.3321,
            lng=18.012,
        )
        d = anomaly.model_dump()
        assert d["site_id"] == 2790
        assert d["nearest_camera_id"] == "CAM_01"
        assert d["severity"] == "warning"
