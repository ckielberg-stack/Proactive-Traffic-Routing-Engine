"""
Unit tests for the PTRE domain models and vision engine.

Updated for Expert Audit Fix 1: Vision Engine now outputs density
(veh/km/lane) instead of flow-as-capacity.
"""

from datetime import datetime

import numpy as np
import pytest

from src.models import CameraMetadata, CapacityState, SensorReading
from src.vision_engine import (
    BLACK_IMAGE_THRESHOLD,
    DEFAULT_ROI_LENGTH_KM,
    FREE_FLOW_SPEED_KMH,
    JAM_DENSITY_VEH_KM_LANE,
    K_CRITICAL_VEH_KM_LANE,
    Q_CAP_VPH_PER_LANE,
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
        assert state.observed_density_veh_km_lane == 0.0  # default

    def test_valid_construction_with_density(self) -> None:
        state = CapacityState(
            timestamp=datetime.now(),
            camera_id="cam1",
            vehicle_count=5,
            blocked_lanes=0,
            total_lanes=3,
            estimated_capacity_vph=1500.0,
            observed_density_veh_km_lane=33.3,
        )
        assert state.observed_density_veh_km_lane == 33.3

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

    def test_negative_density_rejected(self) -> None:
        with pytest.raises(Exception):
            CapacityState(
                timestamp=datetime.now(),
                camera_id="cam1",
                vehicle_count=0,
                blocked_lanes=0,
                total_lanes=1,
                estimated_capacity_vph=0,
                observed_density_veh_km_lane=-1.0,
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


# ======================================================================
# Density estimation tests (Expert Audit Fix 1)
# ======================================================================


class TestVisionEngineDensityEstimation:
    """Test the new density-based estimation (replaces Greenshields capacity)."""

    def test_zero_vehicles_gives_zero_density(self, engine: VisionEngine) -> None:
        density = engine._estimate_density(vehicle_count=0, num_lanes=3)
        assert density == 0.0

    def test_basic_density_calculation(self, engine: VisionEngine) -> None:
        # 10 vehicles / 0.1 km ROI / 3 lanes = 33.33 veh/km/lane
        density = engine._estimate_density(
            vehicle_count=10, num_lanes=3, roi_length_km=0.1
        )
        assert abs(density - 33.33) < 0.1

    def test_density_per_lane_normalisation(self, engine: VisionEngine) -> None:
        # Same vehicle count, different lanes → different density
        d1 = engine._estimate_density(vehicle_count=10, num_lanes=1)
        d2 = engine._estimate_density(vehicle_count=10, num_lanes=2)
        assert d1 > d2  # More lanes → lower per-lane density

    def test_density_clamped_at_jam_density(self, engine: VisionEngine) -> None:
        # 200 vehicles in 0.1 km with 1 lane = 2000 veh/km/lane → clamped to 133
        density = engine._estimate_density(
            vehicle_count=200, num_lanes=1, roi_length_km=0.1
        )
        assert density == JAM_DENSITY_VEH_KM_LANE

    def test_density_below_k_critical(self, engine: VisionEngine) -> None:
        # 3 vehicles / 0.1 km / 3 lanes = 10 veh/km/lane → below k_critical
        density = engine._estimate_density(
            vehicle_count=3, num_lanes=3, roi_length_km=0.1
        )
        assert density < K_CRITICAL_VEH_KM_LANE

    def test_density_above_k_critical(self, engine: VisionEngine) -> None:
        # 15 vehicles / 0.1 km / 3 lanes = 50 veh/km/lane → above k_critical
        density = engine._estimate_density(
            vehicle_count=15, num_lanes=3, roi_length_km=0.1
        )
        assert density > K_CRITICAL_VEH_KM_LANE

    def test_custom_roi_length(self, engine: VisionEngine) -> None:
        # 10 vehicles / 0.5 km / 2 lanes = 10 veh/km/lane
        density = engine._estimate_density(
            vehicle_count=10, num_lanes=2, roi_length_km=0.5
        )
        assert abs(density - 10.0) < 0.1


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
