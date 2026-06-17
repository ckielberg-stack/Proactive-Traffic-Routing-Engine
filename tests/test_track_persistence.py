from datetime import datetime, timedelta

import pytest

from src.track_persistence import TrackPersistence, box_iou


def _det(box: tuple[float, float, float, float], confidence: float = 0.8) -> dict:
    return {"xyxy": box, "confidence": confidence, "class_id": 2}


def test_box_iou_matches_overlapping_boxes() -> None:
    assert box_iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)
    assert box_iou((0, 0, 10, 10), (20, 20, 30, 30)) == pytest.approx(0.0)


def test_same_box_across_required_ticks_triggers() -> None:
    tracker = TrackPersistence(required_ticks=3, free_flow_speed_kmh=100.0)
    now = datetime(2026, 6, 17, 12, 0, 0)
    detections = [_det((10, 10, 50, 50), 0.81)]

    assert tracker.update("CAM_A", detections, timestamp=now, local_speed_kmh=80.0) is None
    assert (
        tracker.update(
            "CAM_A",
            detections,
            timestamp=now + timedelta(minutes=1),
            local_speed_kmh=80.0,
        )
        is None
    )
    event = tracker.update(
        "CAM_A",
        detections,
        timestamp=now + timedelta(minutes=2),
        local_speed_kmh=80.0,
    )

    assert event is not None
    assert event.reason == "vehicle_stopped"
    assert event.persistence_ticks == 3
    assert event.confidence == pytest.approx(0.81)


def test_iou_below_threshold_starts_new_track() -> None:
    tracker = TrackPersistence(required_ticks=2, free_flow_speed_kmh=100.0)
    now = datetime(2026, 6, 17, 12, 0, 0)

    tracker.update(
        "CAM_A",
        [_det((10, 10, 50, 50))],
        timestamp=now,
        local_speed_kmh=80.0,
    )
    event = tracker.update(
        "CAM_A",
        [_det((100, 100, 140, 140))],
        timestamp=now + timedelta(minutes=1),
        local_speed_kmh=80.0,
    )

    assert event is None


def test_missing_detections_expire_tracks() -> None:
    tracker = TrackPersistence(
        required_ticks=2,
        max_missed_ticks=0,
        free_flow_speed_kmh=100.0,
    )
    now = datetime(2026, 6, 17, 12, 0, 0)

    tracker.update(
        "CAM_A",
        [_det((10, 10, 50, 50))],
        timestamp=now,
        local_speed_kmh=80.0,
    )
    assert (
        tracker.update(
            "CAM_A",
            [],
            timestamp=now + timedelta(minutes=1),
            local_speed_kmh=80.0,
        )
        is None
    )
    assert (
        tracker.update(
            "CAM_A",
            [_det((10, 10, 50, 50))],
            timestamp=now + timedelta(minutes=2),
            local_speed_kmh=80.0,
        )
        is None
    )


def test_low_speed_suppresses_stopped_event() -> None:
    tracker = TrackPersistence(required_ticks=2, free_flow_speed_kmh=100.0)
    now = datetime(2026, 6, 17, 12, 0, 0)

    tracker.update(
        "CAM_A",
        [_det((10, 10, 50, 50))],
        timestamp=now,
        local_speed_kmh=40.0,
    )
    event = tracker.update(
        "CAM_A",
        [_det((10, 10, 50, 50))],
        timestamp=now + timedelta(minutes=1),
        local_speed_kmh=40.0,
    )

    assert event is None


def test_multiple_boxes_returns_strongest_mature_track() -> None:
    tracker = TrackPersistence(required_ticks=2, free_flow_speed_kmh=100.0)
    now = datetime(2026, 6, 17, 12, 0, 0)

    tracker.update(
        "CAM_A",
        [
            _det((10, 10, 50, 50), 0.6),
            _det((100, 100, 150, 150), 0.9),
        ],
        timestamp=now,
        local_speed_kmh=90.0,
    )
    event = tracker.update(
        "CAM_A",
        [
            _det((10, 10, 50, 50), 0.6),
            _det((100, 100, 150, 150), 0.9),
        ],
        timestamp=now + timedelta(minutes=1),
        local_speed_kmh=90.0,
    )

    assert event is not None
    assert event.box == (100, 100, 150, 150)
    assert event.confidence == pytest.approx(0.9)
