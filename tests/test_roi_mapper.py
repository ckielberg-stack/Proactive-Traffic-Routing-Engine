"""
Unit tests for the ROI Mapper module.
"""

import json
import os
import tempfile

import pytest

from src.roi_mapper import ROIMapper, ROIRegion


# ======================================================================
# Fixtures
# ======================================================================

SAMPLE_CONFIG = {
    "_comment": "Test config",
    "cameras": {
        "CAM_001": {
            "rois": [
                {
                    "road_id": "E4_Northbound",
                    "direction_relative_to_camera": "away",
                    "capacity_vph": 2000,
                    "num_lanes": 3,
                    "polygon": [[100, 100], [400, 100], [400, 500], [100, 500]],
                },
                {
                    "road_id": "Offramp_A",
                    "direction_relative_to_camera": "towards",
                    "capacity_vph": 600,
                    "num_lanes": 1,
                    "polygon": [[500, 200], [700, 200], [700, 500], [500, 500]],
                },
            ]
        },
        "CAM_INVALID": {
            "rois": [
                {
                    "road_id": "Bad_Poly",
                    "direction_relative_to_camera": "away",
                    "capacity_vph": 1000,
                    "num_lanes": 2,
                    "polygon": [[10, 10], [20, 20]],  # Only 2 vertices → invalid
                },
            ]
        },
    },
}


@pytest.fixture
def config_path(tmp_path) -> str:
    """Write a temp camera_config.json and return its path."""
    p = tmp_path / "camera_config.json"
    p.write_text(json.dumps(SAMPLE_CONFIG), encoding="utf-8")
    return str(p)


@pytest.fixture
def mapper(config_path: str) -> ROIMapper:
    return ROIMapper(config_path)


# ======================================================================
# Config loading
# ======================================================================


class TestROIMapperLoading:
    def test_loads_valid_camera(self, mapper: ROIMapper) -> None:
        assert mapper.has_rois("CAM_001")
        rois = mapper.get_rois("CAM_001")
        assert len(rois) == 2
        assert rois[0].road_id == "E4_Northbound"
        assert rois[1].road_id == "Offramp_A"

    def test_invalid_polygon_skipped(self, mapper: ROIMapper) -> None:
        """Camera with < 3 vertex polygon should have no ROIs."""
        assert not mapper.has_rois("CAM_INVALID")
        assert mapper.get_rois("CAM_INVALID") == []

    def test_unconfigured_camera(self, mapper: ROIMapper) -> None:
        assert not mapper.has_rois("UNKNOWN_CAM")
        assert mapper.get_rois("UNKNOWN_CAM") == []

    def test_missing_config_file(self, tmp_path) -> None:
        """Graceful degradation when config file doesn't exist."""
        m = ROIMapper(str(tmp_path / "nonexistent.json"))
        assert m.configured_cameras == []

    def test_malformed_json(self, tmp_path) -> None:
        """Graceful degradation on invalid JSON."""
        p = tmp_path / "bad.json"
        p.write_text("{invalid json!}", encoding="utf-8")
        m = ROIMapper(str(p))
        assert m.configured_cameras == []

    def test_configured_cameras_list(self, mapper: ROIMapper) -> None:
        assert "CAM_001" in mapper.configured_cameras


# ======================================================================
# Detection classification
# ======================================================================


class TestROIMapperClassification:
    def test_point_inside_first_roi(self, mapper: ROIMapper) -> None:
        """Point clearly inside E4_Northbound polygon."""
        region = mapper.classify_detection("CAM_001", 250, 300)
        assert region is not None
        assert region.road_id == "E4_Northbound"
        assert region.direction_relative_to_camera == "away"

    def test_point_inside_second_roi(self, mapper: ROIMapper) -> None:
        """Point clearly inside Offramp_A polygon."""
        region = mapper.classify_detection("CAM_001", 600, 350)
        assert region is not None
        assert region.road_id == "Offramp_A"

    def test_point_outside_all_rois(self, mapper: ROIMapper) -> None:
        """Point in the gap between polygons → discard."""
        region = mapper.classify_detection("CAM_001", 450, 300)
        assert region is None

    def test_point_far_outside(self, mapper: ROIMapper) -> None:
        """Point far outside all polygons."""
        region = mapper.classify_detection("CAM_001", 900, 100)
        assert region is None

    def test_unconfigured_camera_returns_none(self, mapper: ROIMapper) -> None:
        region = mapper.classify_detection("UNKNOWN_CAM", 250, 300)
        assert region is None


# ======================================================================
# Batch classification
# ======================================================================


class TestROIMapperBatch:
    def test_batch_classification(self, mapper: ROIMapper) -> None:
        detections = [
            {"xyxy": (150, 50, 350, 400)},   # Bottom-center (250, 400) → E4_Northbound
            {"xyxy": (550, 150, 650, 350)},   # Bottom-center (600, 350) → Offramp_A
            {"xyxy": (800, 50, 900, 100)},    # Bottom-center (850, 100) → outside
        ]
        segments = mapper.classify_detections_batch("CAM_001", detections)

        assert "E4_Northbound" in segments
        assert len(segments["E4_Northbound"]) == 1

        assert "Offramp_A" in segments
        assert len(segments["Offramp_A"]) == 1

        # Third detection (outside) should not appear in any segment
        total_classified = sum(len(v) for v in segments.values())
        assert total_classified == 2

    def test_empty_detections(self, mapper: ROIMapper) -> None:
        segments = mapper.classify_detections_batch("CAM_001", [])
        assert segments == {}

    def test_all_outside(self, mapper: ROIMapper) -> None:
        detections = [
            {"xyxy": (800, 50, 900, 100)},
            {"xyxy": (0, 0, 10, 10)},
        ]
        segments = mapper.classify_detections_batch("CAM_001", detections)
        assert segments == {}


# ======================================================================
# ROIRegion model
# ======================================================================


class TestROIRegionModel:
    def test_shapely_polygon_property(self) -> None:
        roi = ROIRegion(
            road_id="test",
            direction_relative_to_camera="away",
            capacity_vph=1000,
            num_lanes=2,
            polygon=[[0, 0], [100, 0], [100, 100], [0, 100]],
        )
        poly = roi.shapely_polygon
        assert poly.is_valid
        assert poly.area > 0

    def test_defaults(self) -> None:
        roi = ROIRegion(
            road_id="test",
            direction_relative_to_camera="towards",
            capacity_vph=500,
            polygon=[[0, 0], [10, 0], [10, 10]],
        )
        assert roi.num_lanes == 2  # default
