"""Tests for ROI physical-length calibration utilities."""

from src.roi_length_calibration import (
    ESTIMATE_SOURCE,
    apply_estimated_lengths,
    estimate_roi_length_meters,
    validate_roi_lengths,
)


def test_estimate_roi_length_from_lane_width_geometry() -> None:
    length = estimate_roi_length_meters(
        polygon=[[0, 0], [200, 0], [200, 100], [0, 100]],
        num_lanes=2,
    )

    assert length == 14.0


def test_validate_flags_missing_and_default_lengths() -> None:
    config = {
        "cameras": {
            "CAM_1": {
                "rois": [
                    {
                        "road_id": "E4_NB",
                        "polygon": [[0, 0], [200, 0], [200, 100], [0, 100]],
                    },
                    {
                        "road_id": "E4_SB",
                        "roi_length_meters": 100.0,
                        "polygon": [[0, 0], [200, 0], [200, 100], [0, 100]],
                    },
                ]
            }
        }
    }

    issues = validate_roi_lengths(config)

    assert [issue.severity for issue in issues] == ["error", "error"]
    assert "missing" in issues[0].message
    assert "100 m fallback" in issues[1].message


def test_apply_estimated_lengths_stamps_metadata() -> None:
    config = {
        "cameras": {
            "CAM_1": {
                "rois": [
                    {
                        "road_id": "E4_NB",
                        "num_lanes": 3,
                        "polygon": [[0, 0], [300, 0], [300, 100], [0, 100]],
                    }
                ]
            }
        }
    }

    updated, issues = apply_estimated_lengths(config)

    roi = config["cameras"]["CAM_1"]["rois"][0]
    assert updated == 1
    assert issues == []
    assert roi["roi_length_meters"] == 31.5
    assert roi["roi_length_source"] == ESTIMATE_SOURCE
    assert roi["roi_length_confidence"] == "estimated"
    assert config["roi_length_calibration"]["source"] == ESTIMATE_SOURCE


def test_apply_estimated_lengths_preserves_existing_non_default() -> None:
    config = {
        "cameras": {
            "CAM_1": {
                "rois": [
                    {
                        "road_id": "E4_NB",
                        "num_lanes": 3,
                        "roi_length_meters": 42.0,
                        "polygon": [[0, 0], [300, 0], [300, 100], [0, 100]],
                    }
                ]
            }
        }
    }

    updated, issues = apply_estimated_lengths(config)

    assert updated == 0
    assert issues == []
    assert config["cameras"]["CAM_1"]["rois"][0]["roi_length_meters"] == 42.0
