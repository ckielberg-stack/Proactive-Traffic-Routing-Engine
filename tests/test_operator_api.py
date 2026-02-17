"""
Unit tests for the Operator Decision Support API (Phase 6).

Tests cover:
- Active incidents endpoint
- VMS recommendations with proxy ground-truth enrichment
- DATEX II XML export (incidents + SpeedManagement records)
- Health check with pipeline metadata
"""

from datetime import datetime

import pytest
from fastapi.testclient import TestClient

from src.models import (
    IncidentReport,
    QueuePrediction,
    VMSRecommendation,
    VMSStatusSnapshot,
)
from src.operator_api import (
    app,
    set_active_incidents,
    set_active_predictions,
    set_active_recommendations,
    set_active_vms_statuses,
    set_last_tick_time,
    set_vms_orchestrator,
    _match_proxy_ground_truth,
)
from src.vms_orchestrator import VMSOrchestrator


# ======================================================================
# Fixtures
# ======================================================================


@pytest.fixture(autouse=True)
def _reset_state() -> None:
    """Reset API state before each test."""
    set_active_incidents([])
    set_active_predictions([])
    set_active_vms_statuses([])
    set_active_recommendations([])
    set_vms_orchestrator(VMSOrchestrator())  # default config
    set_last_tick_time(None)


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture
def sample_incident() -> IncidentReport:
    return IncidentReport(
        timestamp=datetime(2026, 2, 16, 14, 0, 0),
        camera_id="CAM_TEST_01",
        incident_type="vehicle_stopped",
        lanes_affected=1,
        total_lanes=3,
        capacity_drop_percentage=33.3,
        thumbnail_base64="dGVzdA==",  # base64("test")
        confidence=0.88,
        lat=59.30,
        lng=18.00,
    )


@pytest.fixture
def sample_prediction() -> QueuePrediction:
    return QueuePrediction(
        timestamp=datetime(2026, 2, 16, 14, 0, 0),
        camera_id="CAM_TEST_01",
        origin_lat=59.30,
        origin_lng=18.00,
        origin_chainage_km=8.6,
        growth_speed_kmh=8.0,
        lengths_at_minutes={1: 0.133, 3: 0.4, 5: 0.667},
    )


@pytest.fixture
def sample_vms_recommendation() -> VMSRecommendation:
    return VMSRecommendation(
        timestamp=datetime(2026, 2, 16, 14, 0, 0),
        vms_id="VMS-4003",
        vms_name="Kungens Kurva E4",
        recommended_message="KÖVARNING 70 km/h",
        urgency="soon",
        queue_growth_speed_kmh=8.0,
        distance_queue_tail_to_vms_km=1.2,
        estimated_activation_minutes=6.5,
        triggering_camera_id="CAM_TEST_01",
        current_vms_status="OFF",
        summary="Kö växer bakåt med 8 km/h.",
    )


@pytest.fixture
def active_proxy_status() -> VMSStatusSnapshot:
    return VMSStatusSnapshot(
        timestamp=datetime(2026, 2, 16, 14, 0, 0),
        vms_id="SE_STA_SPEEDMANAGEMENTID_1_123",
        vms_name="E4 — Hallunda-Kungens Kurva",
        is_active=True,
        displayed_message="Rekommenderad hastighet: 70km/h",
        speed_limit=70,
    )


@pytest.fixture
def inactive_proxy_status() -> VMSStatusSnapshot:
    return VMSStatusSnapshot(
        timestamp=datetime(2026, 2, 16, 14, 0, 0),
        vms_id="VMS-4003",
        vms_name="Kungens Kurva",
        is_active=False,
        displayed_message=None,
        speed_limit=None,
    )


# ======================================================================
# Active Incidents endpoint
# ======================================================================


class TestActiveIncidents:
    def test_empty_returns_zero(self, client: TestClient) -> None:
        resp = client.get("/api/v1/operator/active-incidents")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 0
        assert data["incidents"] == []

    def test_returns_injected_incident(
        self, client: TestClient, sample_incident: IncidentReport
    ) -> None:
        set_active_incidents([sample_incident])
        resp = client.get("/api/v1/operator/active-incidents")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 1
        inc = data["incidents"][0]
        assert inc["incident_type"] == "vehicle_stopped"
        assert inc["lanes_affected"] == 1
        assert inc["capacity_drop_percentage"] == 33.3
        assert inc["thumbnail_base64"] == "dGVzdA=="
        assert inc["camera_id"] == "CAM_TEST_01"

    def test_response_has_required_fields(
        self, client: TestClient, sample_incident: IncidentReport
    ) -> None:
        set_active_incidents([sample_incident])
        resp = client.get("/api/v1/operator/active-incidents")
        inc = resp.json()["incidents"][0]
        required_fields = {
            "incident_type",
            "lanes_affected",
            "capacity_drop_percentage",
            "thumbnail_base64",
        }
        assert required_fields.issubset(inc.keys())


# ======================================================================
# VMS Recommendations endpoint
# ======================================================================


class TestVMSRecommendations:
    def test_empty_returns_zero(self, client: TestClient) -> None:
        resp = client.get("/api/v1/operator/vms-recommendations")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 0
        assert data["recommendations"] == []

    def test_returns_enriched_recommendations(
        self,
        client: TestClient,
        sample_vms_recommendation: VMSRecommendation,
    ) -> None:
        set_active_recommendations([sample_vms_recommendation])
        resp = client.get("/api/v1/operator/vms-recommendations")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 1

        enriched = data["recommendations"][0]
        # New schema: nested recommendation + ground truth
        assert "recommendation" in enriched
        assert "proxy_ground_truth_active" in enriched
        rec = enriched["recommendation"]
        assert rec["vms_id"] == "VMS-4003"
        assert rec["recommended_message"] == "KÖVARNING 70 km/h"
        assert rec["urgency"] == "soon"

    def test_ground_truth_false_when_no_active_proxy(
        self,
        client: TestClient,
        sample_vms_recommendation: VMSRecommendation,
    ) -> None:
        set_active_recommendations([sample_vms_recommendation])
        resp = client.get("/api/v1/operator/vms-recommendations")
        enriched = resp.json()["recommendations"][0]
        assert enriched["proxy_ground_truth_active"] is False
        assert enriched["proxy_speed_limit"] is None

    def test_ground_truth_true_when_active_proxy_matches(
        self,
        client: TestClient,
        sample_vms_recommendation: VMSRecommendation,
        active_proxy_status: VMSStatusSnapshot,
    ) -> None:
        set_active_recommendations([sample_vms_recommendation])
        set_active_vms_statuses([active_proxy_status])
        resp = client.get("/api/v1/operator/vms-recommendations")
        enriched = resp.json()["recommendations"][0]
        # "E4" is in the proxy vms_name, and "E4" is in the rec vms_name
        assert enriched["proxy_ground_truth_active"] is True
        assert enriched["proxy_speed_limit"] == 70

    def test_fallback_computes_from_predictions(
        self, client: TestClient, sample_prediction: QueuePrediction,
    ) -> None:
        """When no pre-computed recommendations exist, compute on-the-fly."""
        set_active_predictions([sample_prediction])
        resp = client.get("/api/v1/operator/vms-recommendations")
        assert resp.status_code == 200
        # Whether recs are generated depends on VMS config — just check 200

    def test_ai_prediction_timestamp(
        self,
        client: TestClient,
        sample_vms_recommendation: VMSRecommendation,
    ) -> None:
        now = datetime(2026, 2, 16, 14, 30, 0)
        set_active_recommendations([sample_vms_recommendation])
        set_last_tick_time(now)
        resp = client.get("/api/v1/operator/vms-recommendations")
        data = resp.json()
        assert data["ai_prediction_timestamp"] is not None


# ======================================================================
# Ground-truth matching logic
# ======================================================================


class TestProxyGroundTruthMatching:
    def test_match_by_vms_id(self) -> None:
        rec = VMSRecommendation(
            timestamp=datetime.now(),
            vms_id="VMS-4003",
            vms_name="Kungens Kurva",
            recommended_message="KÖVARNING 70 km/h",
            urgency="soon",
            queue_growth_speed_kmh=8.0,
            distance_queue_tail_to_vms_km=1.0,
            estimated_activation_minutes=6.0,
            triggering_camera_id="CAM_01",
            summary="Test",
        )
        statuses = [VMSStatusSnapshot(
            timestamp=datetime.now(),
            vms_id="VMS-4003",
            vms_name="Some name",
            is_active=True,
            displayed_message="70km/h",
            speed_limit=70,
        )]
        active, speed, dev_id = _match_proxy_ground_truth(rec, statuses)
        assert active is True
        assert speed == 70
        assert dev_id == "VMS-4003"

    def test_match_by_road_name(self) -> None:
        rec = VMSRecommendation(
            timestamp=datetime.now(),
            vms_id="VMS-4003",
            vms_name="Kungens Kurva E4",
            recommended_message="KÖVARNING 70 km/h",
            urgency="soon",
            queue_growth_speed_kmh=8.0,
            distance_queue_tail_to_vms_km=1.0,
            estimated_activation_minutes=6.0,
            triggering_camera_id="CAM_01",
            summary="Test",
        )
        statuses = [VMSStatusSnapshot(
            timestamp=datetime.now(),
            vms_id="SE_STA_SPEEDMANAGEMENTID_1_999",
            vms_name="E4 — somewhere on E4",
            is_active=True,
            displayed_message="70km/h",
            speed_limit=70,
        )]
        active, speed, dev_id = _match_proxy_ground_truth(rec, statuses)
        assert active is True
        assert speed == 70

    def test_no_match_when_inactive(self) -> None:
        rec = VMSRecommendation(
            timestamp=datetime.now(),
            vms_id="VMS-4003",
            vms_name="Kungens Kurva E4",
            recommended_message="KÖVARNING 70 km/h",
            urgency="soon",
            queue_growth_speed_kmh=8.0,
            distance_queue_tail_to_vms_km=1.0,
            estimated_activation_minutes=6.0,
            triggering_camera_id="CAM_01",
            summary="Test",
        )
        statuses = [VMSStatusSnapshot(
            timestamp=datetime.now(),
            vms_id="VMS-4003",
            vms_name="Kungens Kurva",
            is_active=False,
            displayed_message=None,
            speed_limit=None,
        )]
        active, speed, dev_id = _match_proxy_ground_truth(rec, statuses)
        assert active is False
        assert speed is None

    def test_no_match_when_different_road(self) -> None:
        rec = VMSRecommendation(
            timestamp=datetime.now(),
            vms_id="VMS-4003",
            vms_name="Kungens Kurva E4",
            recommended_message="KÖVARNING 70 km/h",
            urgency="soon",
            queue_growth_speed_kmh=8.0,
            distance_queue_tail_to_vms_km=1.0,
            estimated_activation_minutes=6.0,
            triggering_camera_id="CAM_01",
            summary="Test",
        )
        statuses = [VMSStatusSnapshot(
            timestamp=datetime.now(),
            vms_id="SE_STA_SPEEDMANAGEMENTID_1_111",
            vms_name="Väg 73 — somewhere else",
            is_active=True,
            displayed_message="50km/h",
            speed_limit=50,
        )]
        active, speed, dev_id = _match_proxy_ground_truth(rec, statuses)
        assert active is False


# ======================================================================
# DATEX II export endpoint
# ======================================================================


class TestDatex2Export:
    def test_empty_returns_valid_xml(self, client: TestClient) -> None:
        resp = client.get("/api/v1/export/datex2")
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "application/xml"
        content = resp.text
        assert '<?xml version="1.0"' in content
        assert "d2LogicalModel" in content
        assert "datex2.eu" in content

    def test_xml_contains_incident_data(
        self, client: TestClient, sample_incident: IncidentReport
    ) -> None:
        set_active_incidents([sample_incident])
        resp = client.get("/api/v1/export/datex2")
        content = resp.text
        assert "vehicle_stopped" in content
        assert "CAM_TEST_01" in content
        assert "situation" in content
        assert "SituationPublication" in content

    def test_xml_contains_speed_management(
        self,
        client: TestClient,
        sample_vms_recommendation: VMSRecommendation,
    ) -> None:
        set_active_recommendations([sample_vms_recommendation])
        resp = client.get("/api/v1/export/datex2")
        content = resp.text
        assert "speedManagement" in content
        assert "VMS-4003" in content
        assert "KÖVARNING 70 km/h" in content
        assert "PTRE-VMS-" in content

    def test_xml_speed_management_includes_operator_status(
        self,
        client: TestClient,
        sample_vms_recommendation: VMSRecommendation,
        active_proxy_status: VMSStatusSnapshot,
    ) -> None:
        set_active_recommendations([sample_vms_recommendation])
        set_active_vms_statuses([active_proxy_status])
        resp = client.get("/api/v1/export/datex2")
        content = resp.text
        assert "implemented" in content  # human operator already acted

    def test_xml_has_correct_namespace(self, client: TestClient) -> None:
        resp = client.get("/api/v1/export/datex2")
        content = resp.text
        assert "http://datex2.eu/schema/3/d2Payload" in content

    def test_content_disposition_header(self, client: TestClient) -> None:
        resp = client.get("/api/v1/export/datex2")
        assert "datex2_export.xml" in resp.headers.get("content-disposition", "")


# ======================================================================
# Health check
# ======================================================================


class TestHealthCheck:
    def test_health_returns_ok(self, client: TestClient) -> None:
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "active_incidents" in data
        assert "active_recommendations" in data
        assert "proxy_statuses_polled" in data

    def test_health_reflects_pipeline_state(
        self, client: TestClient, sample_incident: IncidentReport
    ) -> None:
        set_active_incidents([sample_incident])
        set_last_tick_time(datetime(2026, 2, 16, 14, 30, 0))
        resp = client.get("/health")
        data = resp.json()
        assert data["active_incidents"] == 1
        assert data["last_tick"] is not None
