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


# ======================================================================
# BEV Homography (Expert Audit Fix 2)
# ======================================================================


# Identity homography: pixel coords map 1:1 to BEV coords
IDENTITY_H = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

# A simple scale+translate homography: BEV_x = pixel_x * 0.1, BEV_y = pixel_y * 0.1
# (i.e. 10 pixels = 1 meter)
SCALE_H = [[0.1, 0, 0], [0, 0.1, 0], [0, 0, 1]]

BEV_CONFIG = {
    "_comment": "Test config with BEV homography",
    "cameras": {
        "CAM_BEV": {
            "homography_matrix": IDENTITY_H,
            "rois": [
                {
                    "road_id": "E4_NB",
                    "direction_relative_to_camera": "away",
                    "capacity_vph": 2000,
                    "num_lanes": 3,
                    "polygon": [[100, 100], [400, 100], [400, 500], [100, 500]],
                },
            ],
        },
        "CAM_SCALE": {
            "homography_matrix": SCALE_H,
            "rois": [
                {
                    "road_id": "E4_SB",
                    "direction_relative_to_camera": "towards",
                    "capacity_vph": 2000,
                    "num_lanes": 2,
                    "polygon": [[0, 0], [100, 0], [100, 100], [0, 100]],
                },
            ],
        },
        "CAM_NO_H": {
            "rois": [
                {
                    "road_id": "E4_Legacy",
                    "direction_relative_to_camera": "away",
                    "capacity_vph": 1500,
                    "num_lanes": 2,
                    "polygon": [[100, 100], [400, 100], [400, 500], [100, 500]],
                },
            ],
        },
    },
}


@pytest.fixture
def bev_config_path(tmp_path) -> str:
    p = tmp_path / "bev_config.json"
    p.write_text(json.dumps(BEV_CONFIG), encoding="utf-8")
    return str(p)


@pytest.fixture
def bev_mapper(bev_config_path: str) -> ROIMapper:
    return ROIMapper(bev_config_path)


class TestBEVHomography:
    def test_has_homography(self, bev_mapper: ROIMapper) -> None:
        assert bev_mapper.has_homography("CAM_BEV")
        assert bev_mapper.has_homography("CAM_SCALE")
        assert not bev_mapper.has_homography("CAM_NO_H")

    def test_pixel_to_bev_identity(self, bev_mapper: ROIMapper) -> None:
        """Identity homography → BEV coords equal pixel coords."""
        result = bev_mapper.pixel_to_bev("CAM_BEV", 250.0, 300.0)
        assert result is not None
        bev_x, bev_y = result
        assert abs(bev_x - 250.0) < 0.01
        assert abs(bev_y - 300.0) < 0.01

    def test_pixel_to_bev_scale(self, bev_mapper: ROIMapper) -> None:
        """Scale homography → BEV coords are 1/10 of pixel coords."""
        result = bev_mapper.pixel_to_bev("CAM_SCALE", 500.0, 300.0)
        assert result is not None
        bev_x, bev_y = result
        assert abs(bev_x - 50.0) < 0.01
        assert abs(bev_y - 30.0) < 0.01

    def test_pixel_to_bev_no_homography(self, bev_mapper: ROIMapper) -> None:
        result = bev_mapper.pixel_to_bev("CAM_NO_H", 250.0, 300.0)
        assert result is None

    def test_classify_detection_with_identity_h(self, bev_mapper: ROIMapper) -> None:
        """Identity H: detection inside ROI should still be classified correctly."""
        region = bev_mapper.classify_detection("CAM_BEV", 250, 300)
        assert region is not None
        assert region.road_id == "E4_NB"

    def test_classify_detection_outside_with_identity_h(self, bev_mapper: ROIMapper) -> None:
        """Identity H: detection outside ROI should return None."""
        region = bev_mapper.classify_detection("CAM_BEV", 50, 50)
        assert region is None

    def test_classify_detection_with_scale_h(self, bev_mapper: ROIMapper) -> None:
        """Scale H: pixel (50,50) maps to BEV (5,5) which is inside
        the ROI polygon [[0,0]..[100,100]] transformed to [[0,0]..[10,10]]."""
        region = bev_mapper.classify_detection("CAM_SCALE", 50, 50)
        assert region is not None
        assert region.road_id == "E4_SB"

    def test_legacy_camera_pixel_space(self, bev_mapper: ROIMapper) -> None:
        """Camera without H uses pixel-space fallback (backward compat)."""
        region = bev_mapper.classify_detection("CAM_NO_H", 250, 300)
        assert region is not None
        assert region.road_id == "E4_Legacy"

    def test_transform_polygon_to_bev(self, bev_mapper: ROIMapper) -> None:
        polygon = [[100, 100], [400, 100], [400, 500], [100, 500]]
        result = bev_mapper.transform_polygon_to_bev("CAM_SCALE", polygon)
        assert result is not None
        # Should be scaled by 0.1
        assert abs(result[0][0] - 10.0) < 0.01
        assert abs(result[0][1] - 10.0) < 0.01
        assert abs(result[2][0] - 40.0) < 0.01
        assert abs(result[2][1] - 50.0) < 0.01

    def test_compute_roi_area_m2(self, bev_mapper: ROIMapper) -> None:
        """Scale H maps 100×100px ROI to 10×10m → area = 100 m²."""
        roi = bev_mapper.get_rois("CAM_SCALE")[0]
        area = bev_mapper.compute_roi_area_m2("CAM_SCALE", roi)
        assert area is not None
        assert abs(area - 100.0) < 0.1

    def test_compute_roi_area_no_homography(self, bev_mapper: ROIMapper) -> None:
        roi = bev_mapper.get_rois("CAM_NO_H")[0]
        area = bev_mapper.compute_roi_area_m2("CAM_NO_H", roi)
        assert area is None

    def test_invalid_homography_shape(self, tmp_path) -> None:
        """Config with wrong-shaped H matrix should gracefully ignore it."""
        config = {
            "cameras": {
                "CAM_BAD_H": {
                    "homography_matrix": [[1, 0], [0, 1]],  # 2x2 not 3x3
                    "rois": [
                        {
                            "road_id": "test",
                            "direction_relative_to_camera": "away",
                            "capacity_vph": 1000,
                            "polygon": [[0, 0], [100, 0], [100, 100], [0, 100]],
                        }
                    ],
                }
            }
        }
        p = tmp_path / "bad_h.json"
        p.write_text(json.dumps(config), encoding="utf-8")
        m = ROIMapper(str(p))
        assert not m.has_homography("CAM_BAD_H")
        # Should still work in pixel-space
        region = m.classify_detection("CAM_BAD_H", 50, 50)
        assert region is not None

