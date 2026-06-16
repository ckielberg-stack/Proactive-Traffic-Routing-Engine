"""
Unit tests for the VMS & Queue Tail Predictor (Phase 5).
"""

from datetime import datetime
from pathlib import Path

import pytest

from src.models import QueuePrediction, VMSGantry, VMSRecommendation
from src.vms_orchestrator import (
    MIN_UPSTREAM_DISTANCE_KM,
    VMSOrchestrator,
    _build_message,
    _classify_urgency,
)


# ======================================================================
# Fixtures
# ======================================================================

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def vms_config_path(tmp_path: Path) -> Path:
    """Create a minimal VMS config for testing."""
    import json

    config = {
        "gantries": [
            {
                "vms_id": "VMS-T1",
                "name": "Test Gantry South",
                "lat": 59.25,
                "lng": 17.85,
                "road": "E4",
                "direction": "northbound",
                "chainage_km": 2.0,
            },
            {
                "vms_id": "VMS-T2",
                "name": "Test Gantry Mid",
                "lat": 59.28,
                "lng": 17.92,
                "road": "E4",
                "direction": "northbound",
                "chainage_km": 5.0,
            },
            {
                "vms_id": "VMS-T3",
                "name": "Test Gantry North",
                "lat": 59.32,
                "lng": 18.01,
                "road": "E4",
                "direction": "northbound",
                "chainage_km": 10.0,
            },
        ],
    }
    path = tmp_path / "vms_config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


@pytest.fixture
def orchestrator(vms_config_path: Path) -> VMSOrchestrator:
    return VMSOrchestrator(config_path=vms_config_path)


@pytest.fixture
def mock_prediction() -> QueuePrediction:
    """Bottleneck at chainage 10 km, queue growing upstream at 8 km/h."""
    return QueuePrediction(
        timestamp=datetime(2026, 2, 16, 14, 0, 0),
        camera_id="CAM_TEST",
        origin_lat=59.32,
        origin_lng=18.01,
        origin_chainage_km=10.0,
        growth_speed_kmh=8.0,
        lengths_at_minutes={1: 0.133, 3: 0.4, 5: 0.667},
    )


# ======================================================================
# Config loading tests
# ======================================================================


class TestVMSConfigLoading:
    def test_loads_correct_count(self, orchestrator: VMSOrchestrator) -> None:
        assert len(orchestrator.gantries) == 3

    def test_gantries_sorted_by_chainage(self, orchestrator: VMSOrchestrator) -> None:
        chainages = [g.chainage_km for g in orchestrator.gantries]
        assert chainages == sorted(chainages)

    def test_gantry_has_required_fields(self, orchestrator: VMSOrchestrator) -> None:
        g = orchestrator.gantries[0]
        assert g.vms_id == "VMS-T1"
        assert g.name == "Test Gantry South"
        assert g.road == "E4"
        assert g.chainage_km == 2.0

    def test_missing_config_loads_empty(self, tmp_path: Path) -> None:
        orch = VMSOrchestrator(config_path=tmp_path / "nonexistent.json")
        assert orch.gantries == []


# ======================================================================
# Queue tail projection tests
# ======================================================================


class TestQueueTailProjection:
    def test_tail_at_t0(self, mock_prediction: QueuePrediction) -> None:
        tail = VMSOrchestrator.predict_queue_tail_chainage(mock_prediction, 0)
        assert tail == 10.0  # Origin

    def test_tail_at_t1(self, mock_prediction: QueuePrediction) -> None:
        # 8 km/h × 1/60 h = 0.1333 km upstream
        tail = VMSOrchestrator.predict_queue_tail_chainage(mock_prediction, 1)
        assert abs(tail - (10.0 - 0.1333)) < 0.01

    def test_tail_at_t5(self, mock_prediction: QueuePrediction) -> None:
        # 8 km/h × 5/60 h = 0.6667 km upstream
        tail = VMSOrchestrator.predict_queue_tail_chainage(mock_prediction, 5)
        assert abs(tail - (10.0 - 0.6667)) < 0.01

    def test_zero_growth_speed(self) -> None:
        pred = QueuePrediction(
            timestamp=datetime.now(),
            camera_id="CAM",
            origin_lat=59.0,
            origin_lng=18.0,
            origin_chainage_km=5.0,
            growth_speed_kmh=0.0,
            lengths_at_minutes={1: 0.0},
        )
        tail = VMSOrchestrator.predict_queue_tail_chainage(pred, 5)
        assert tail == 5.0  # No movement


# ======================================================================
# Upstream VMS selection tests
# ======================================================================


class TestUpstreamVMSSelection:
    def test_finds_nearest_upstream_vms(self, orchestrator: VMSOrchestrator) -> None:
        # Queue tail at chainage 5.5 → threshold = 5.5 - 1.0 = 4.5
        # VMS-T2 at 5.0 > 4.5 → excluded (too close)
        # VMS-T1 at 2.0 ≤ 4.5 → selected (nearest qualifying)
        vms = orchestrator.find_upstream_vms(5.5)
        assert vms is not None
        assert vms.vms_id == "VMS-T1"

    def test_vms_must_be_far_enough_upstream(self, orchestrator: VMSOrchestrator) -> None:
        # Queue tail at chainage 5.8 → threshold is 4.8
        # VMS-T2 at 5.0 > 4.8 → excluded
        # VMS-T1 at 2.0 ≤ 4.8 → selected
        vms = orchestrator.find_upstream_vms(5.8)
        assert vms is not None
        assert vms.vms_id == "VMS-T1"

    def test_no_vms_upstream_when_too_close(self, orchestrator: VMSOrchestrator) -> None:
        # Queue tail at chainage 2.5 → threshold = 2.5 - 1.0 = 1.5
        # VMS-T1 at 2.0 > 1.5 → excluded (too close)
        # No gantries ≤ 1.5 → None
        vms = orchestrator.find_upstream_vms(2.5)
        assert vms is None

    def test_no_vms_at_all_upstream(self, orchestrator: VMSOrchestrator) -> None:
        # Queue tail at chainage 2.0 → threshold is 1.0
        # VMS-T1 at 2.0 > 1.0 → excluded
        vms = orchestrator.find_upstream_vms(2.0)
        assert vms is None

    def test_exact_minimum_distance(self, orchestrator: VMSOrchestrator) -> None:
        # Queue tail at 3.0 → threshold is 2.0 → VMS-T1 at 2.0 ≤ 2.0 → selected
        vms = orchestrator.find_upstream_vms(3.0)
        assert vms is not None
        assert vms.vms_id == "VMS-T1"


# ======================================================================
# Recommendation generation tests
# ======================================================================


class TestRecommendationGeneration:
    def test_generates_recommendations(
        self, orchestrator: VMSOrchestrator, mock_prediction: QueuePrediction
    ) -> None:
        recs = orchestrator.generate_recommendations(mock_prediction)
        assert len(recs) > 0
        assert all(isinstance(r, VMSRecommendation) for r in recs)

    def test_recommendations_deduplicated_by_vms_id(
        self, orchestrator: VMSOrchestrator, mock_prediction: QueuePrediction
    ) -> None:
        recs = orchestrator.generate_recommendations(mock_prediction)
        vms_ids = [r.vms_id for r in recs]
        assert len(vms_ids) == len(set(vms_ids))

    def test_recommendation_has_swedish_message(
        self, orchestrator: VMSOrchestrator, mock_prediction: QueuePrediction
    ) -> None:
        recs = orchestrator.generate_recommendations(mock_prediction)
        for r in recs:
            assert "VARNING" in r.recommended_message or "KÖVARNING" in r.recommended_message

    def test_empty_when_no_gantries(self, tmp_path: Path) -> None:
        orch = VMSOrchestrator(config_path=tmp_path / "none.json")
        pred = QueuePrediction(
            timestamp=datetime.now(),
            camera_id="CAM",
            origin_lat=59.0,
            origin_lng=18.0,
            origin_chainage_km=5.0,
            growth_speed_kmh=8.0,
            lengths_at_minutes={1: 0.133},
        )
        recs = orch.generate_recommendations(pred)
        assert recs == []

    def test_halka_prefix_for_degraded_surface(
        self, orchestrator: VMSOrchestrator, mock_prediction: QueuePrediction
    ) -> None:
        recs = orchestrator.generate_recommendations(
            mock_prediction,
            surface_state="ice",
        )

        assert recs
        assert all(rec.recommended_message.startswith("HALKA - ") for rec in recs)

    def test_recommendation_exposes_eta_interval_and_confidence(
        self, orchestrator: VMSOrchestrator, mock_prediction: QueuePrediction
    ) -> None:
        mock_prediction.prediction_confidence = 0.82
        mock_prediction.uncertainty_level = "high"
        mock_prediction.length_lower_at_minutes = {5: 0.567}
        mock_prediction.length_upper_at_minutes = {5: 0.767}

        recs = orchestrator.generate_recommendations(mock_prediction)

        assert recs
        rec = recs[0]
        assert rec.eta_lower_minutes is not None
        assert rec.eta_upper_minutes is not None
        assert rec.eta_lower_minutes <= rec.estimated_activation_minutes
        assert rec.eta_upper_minutes >= rec.estimated_activation_minutes
        assert rec.confidence == pytest.approx(0.82)
        assert rec.uncertainty_level == "high"

    def test_low_confidence_upper_eta_downgrades_urgency(
        self, orchestrator: VMSOrchestrator
    ) -> None:
        prediction = QueuePrediction(
            timestamp=datetime(2026, 2, 16, 14, 0, 0),
            camera_id="CAM_TEST",
            origin_lat=59.32,
            origin_lng=18.01,
            origin_chainage_km=10.0,
            growth_speed_kmh=60.0,
            lengths_at_minutes={1: 2.0},
            length_lower_at_minutes={1: 1.0},
            length_upper_at_minutes={1: 3.0},
            prediction_confidence=0.2,
            uncertainty_level="low",
        )

        recs = orchestrator.generate_recommendations(
            prediction,
            time_horizons=[1],
        )

        assert recs
        assert recs[0].estimated_activation_minutes == pytest.approx(3.0)
        assert recs[0].eta_upper_minutes == pytest.approx(5.5)
        assert recs[0].urgency == "advisory"

    def test_narrative_includes_eta_interval_without_hiding_recommendation(
        self, orchestrator: VMSOrchestrator, mock_prediction: QueuePrediction
    ) -> None:
        mock_prediction.prediction_confidence = 0.6
        mock_prediction.uncertainty_level = "medium"
        mock_prediction.length_lower_at_minutes = {5: 0.5}
        mock_prediction.length_upper_at_minutes = {5: 0.8}

        recs = orchestrator.generate_recommendations(mock_prediction)

        assert recs
        assert "ETA-intervall" in recs[0].summary
        assert "medel säkerhet" in recs[0].summary
        assert "Rekommendation:" in recs[0].summary


class TestWeatherRecommendations:
    def test_generates_standalone_halka_warning_by_chainage(
        self,
        orchestrator: VMSOrchestrator,
    ) -> None:
        recs = orchestrator.generate_weather_recommendations(
            [
                {
                    "id": "RC-1",
                    "warning": True,
                    "condition_text": "Halka",
                    "location": "E4 Kungens kurva",
                    "chainage_km": 5.2,
                }
            ],
            now=datetime(2026, 6, 13, 12, 0, 0),
        )

        assert len(recs) == 1
        assert recs[0].vms_id == "VMS-T2"
        assert recs[0].recommended_message == "HALKA - VARNING"
        assert recs[0].urgency == "immediate"
        assert recs[0].triggering_camera_id == "road_condition_RC-1"

    def test_ignores_non_warning_road_conditions(
        self,
        orchestrator: VMSOrchestrator,
    ) -> None:
        recs = orchestrator.generate_weather_recommendations(
            [
                {
                    "id": "RC-1",
                    "warning": False,
                    "condition_text": "Våt vägbana",
                    "chainage_km": 5.2,
                }
            ]
        )

        assert recs == []


# ======================================================================
# Helper function tests
# ======================================================================


class TestUrgencyClassification:
    def test_immediate(self) -> None:
        assert _classify_urgency(1.5) == "immediate"

    def test_soon(self) -> None:
        assert _classify_urgency(3.0) == "soon"

    def test_advisory(self) -> None:
        assert _classify_urgency(10.0) == "advisory"

    def test_boundary_immediate_soon(self) -> None:
        assert _classify_urgency(2.0) == "immediate"

    def test_boundary_soon_advisory(self) -> None:
        assert _classify_urgency(5.0) == "soon"


class TestBuildMessage:
    def test_immediate_message(self) -> None:
        assert _build_message("immediate") == "KÖVARNING 50 km/h"

    def test_soon_message(self) -> None:
        assert _build_message("soon") == "KÖVARNING 70 km/h"

    def test_advisory_message(self) -> None:
        assert "VARNING" in _build_message("advisory")

    def test_halka_prefix(self) -> None:
        assert _build_message("immediate", surface_state="ice") == "HALKA - KÖVARNING 50 km/h"
