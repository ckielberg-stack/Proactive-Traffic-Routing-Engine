"""
Unit tests for the PTRE domain models and vision engine.
"""

from datetime import datetime

import numpy as np
import pytest

from src.models import CameraMetadata, CapacityState, SensorReading
from src.vision_engine import (
    BLACK_IMAGE_THRESHOLD,
    FREE_FLOW_SPEED_KMH,
    SPEED_DROP_RATIO,
    VisionEngine,
)


# ======================================================================
# Fixtures
# ======================================================================


@pytest.fixture
def camera_meta() -> CameraMetadata:
    return CameraMetadata(
        camera_id="TEST_CAM_01",
        name="Test Camera",
        lat=59.25,
        lng=17.85,
        num_lanes=3,
        road="E4",
    )


@pytest.fixture
def normal_sensor() -> SensorReading:
    return SensorReading(
        timestamp=datetime.now(),
        inflow_volume_vph=1200.0,
        average_speed_kmh=90.0,
    )


@pytest.fixture
def speed_drop_sensor() -> SensorReading:
    """Sensor reading with >50 % speed drop from free flow."""
    return SensorReading(
        timestamp=datetime.now(),
        inflow_volume_vph=800.0,
        average_speed_kmh=FREE_FLOW_SPEED_KMH * SPEED_DROP_RATIO - 10,
    )


@pytest.fixture
def engine() -> VisionEngine:
    return VisionEngine(confidence=0.25)


# ======================================================================
# Model validation tests
# ======================================================================


class TestCapacityStateModel:
    def test_valid_construction(self) -> None:
        state = CapacityState(
            timestamp=datetime.now(),
            camera_id="cam1",
            vehicle_count=5,
            blocked_lanes=0,
            total_lanes=3,
            estimated_capacity_vph=1500.0,
        )
        assert state.vehicle_count == 5
        assert state.is_anomaly is False

    def test_negative_vehicle_count_rejected(self) -> None:
        with pytest.raises(Exception):
            CapacityState(
                timestamp=datetime.now(),
                camera_id="cam1",
                vehicle_count=-1,
                blocked_lanes=0,
                total_lanes=3,
                estimated_capacity_vph=0,
            )

    def test_zero_total_lanes_rejected(self) -> None:
        with pytest.raises(Exception):
            CapacityState(
                timestamp=datetime.now(),
                camera_id="cam1",
                vehicle_count=0,
                blocked_lanes=0,
                total_lanes=0,
                estimated_capacity_vph=0,
            )

    def test_confidence_bounds(self) -> None:
        with pytest.raises(Exception):
            CapacityState(
                timestamp=datetime.now(),
                camera_id="cam1",
                vehicle_count=0,
                blocked_lanes=0,
                total_lanes=1,
                estimated_capacity_vph=0,
                confidence=1.5,
            )


class TestSensorReadingModel:
    def test_valid_construction(self) -> None:
        sr = SensorReading(
            timestamp=datetime.now(),
            inflow_volume_vph=1500,
            average_speed_kmh=80,
        )
        assert sr.inflow_volume_vph == 1500

    def test_negative_speed_rejected(self) -> None:
        with pytest.raises(Exception):
            SensorReading(
                timestamp=datetime.now(),
                inflow_volume_vph=1000,
                average_speed_kmh=-5,
            )


class TestCameraMetadataModel:
    def test_defaults(self) -> None:
        meta = CameraMetadata(
            camera_id="cam_x", name="Test", lat=59.0, lng=17.0
        )
        assert meta.num_lanes == 2
        assert meta.road == "E4"


# ======================================================================
# VisionEngine unit tests
# ======================================================================


class TestVisionEngineBlackImage:
    """Test sensor-fusion fallback for black / unavailable images."""

    def test_black_image_with_speed_drop(
        self, engine: VisionEngine, camera_meta: CameraMetadata, speed_drop_sensor: SensorReading, tmp_path
    ) -> None:
        # Create a nearly-black image
        black = np.zeros((480, 640, 3), dtype=np.uint8) + (BLACK_IMAGE_THRESHOLD - 5)
        path = tmp_path / "black.jpg"
        import cv2
        cv2.imwrite(str(path), black)

        state = engine.analyze_frame(str(path), camera_meta, sensor=speed_drop_sensor)

        assert state.is_anomaly is True
        assert state.estimated_capacity_vph == 0.0
        assert state.anomaly_reason == "black_image_with_speed_drop"
        assert state.blocked_lanes == camera_meta.num_lanes

    def test_black_image_without_speed_drop(
        self, engine: VisionEngine, camera_meta: CameraMetadata, normal_sensor: SensorReading, tmp_path
    ) -> None:
        black = np.zeros((480, 640, 3), dtype=np.uint8) + (BLACK_IMAGE_THRESHOLD - 5)
        path = tmp_path / "black.jpg"
        import cv2
        cv2.imwrite(str(path), black)

        state = engine.analyze_frame(str(path), camera_meta, sensor=normal_sensor)

        # Should NOT be flagged as anomaly — just low confidence
        assert state.is_anomaly is False
        assert state.estimated_capacity_vph == 0.0
        assert state.confidence == 0.0

    def test_unreadable_image(
        self, engine: VisionEngine, camera_meta: CameraMetadata, tmp_path
    ) -> None:
        path = tmp_path / "nonexistent.jpg"
        state = engine.analyze_frame(str(path), camera_meta)
        assert state.is_anomaly is True
        assert state.anomaly_reason == "image_unreadable"


class TestVisionEngineCapacityEstimation:
    """Test the Greenshields capacity estimation logic."""

    def test_zero_vehicles_gives_zero_capacity(self, engine: VisionEngine) -> None:
        cap = engine._estimate_capacity(vehicle_count=0, speed_kmh=100, num_lanes=3)
        assert cap == 0.0

    def test_positive_vehicles(self, engine: VisionEngine) -> None:
        # 10 vehicles / 0.3 km = 33.3 veh/km × 80 km/h = 2666.7 VPH
        cap = engine._estimate_capacity(vehicle_count=10, speed_kmh=80, num_lanes=3)
        assert cap > 0
        assert cap <= 6000  # 3 lanes × 2000 max

    def test_capacity_capped_at_max(self, engine: VisionEngine) -> None:
        # Very high density should be capped
        cap = engine._estimate_capacity(vehicle_count=100, speed_kmh=100, num_lanes=2)
        assert cap == 4000.0  # 2 × 2000


class TestVisionEngineIsBlack:
    def test_black_frame_detected(self, engine: VisionEngine) -> None:
        black = np.zeros((100, 100, 3), dtype=np.uint8)
        assert engine._is_black_image(black) is True

    def test_bright_frame_not_black(self, engine: VisionEngine) -> None:
        bright = np.full((100, 100, 3), 200, dtype=np.uint8)
        assert engine._is_black_image(bright) is False

    def test_borderline_frame(self, engine: VisionEngine) -> None:
        # Right at threshold → should NOT be black
        border = np.full((100, 100, 3), BLACK_IMAGE_THRESHOLD, dtype=np.uint8)
        assert engine._is_black_image(border) is False
