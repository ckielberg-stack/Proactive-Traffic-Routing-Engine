"""Cross-tick vehicle-box persistence for stopped-vehicle detection."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from threading import Lock
from typing import Any


DEFAULT_IOU_THRESHOLD = 0.7
DEFAULT_REQUIRED_TICKS = 3
DEFAULT_MAX_MISSED_TICKS = 1
DEFAULT_NORMAL_SPEED_RATIO = 0.7


@dataclass(frozen=True)
class StoppedVehicleEvent:
    """A vehicle-like bounding box persisted long enough to be considered stopped."""

    camera_id: str
    box: tuple[float, float, float, float]
    confidence: float
    persistence_ticks: int
    reason: str = "vehicle_stopped"

    def as_record(self) -> dict[str, Any]:
        return {
            "camera_id": self.camera_id,
            "box": [round(v, 2) for v in self.box],
            "confidence": round(self.confidence, 3),
            "persistence_ticks": self.persistence_ticks,
            "reason": self.reason,
        }


@dataclass
class _Track:
    box: tuple[float, float, float, float]
    confidence: float
    hits: int
    missed: int
    last_seen: datetime


def box_iou(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    """Return intersection-over-union for two xyxy boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter_w = max(0.0, ix2 - ix1)
    inter_h = max(0.0, iy2 - iy1)
    intersection = inter_w * inter_h
    if intersection <= 0.0:
        return 0.0

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    if union <= 0.0:
        return 0.0
    return intersection / union


class TrackPersistence:
    """Track near-stationary vehicle boxes across discrete camera ticks."""

    def __init__(
        self,
        *,
        iou_threshold: float = DEFAULT_IOU_THRESHOLD,
        required_ticks: int = DEFAULT_REQUIRED_TICKS,
        max_missed_ticks: int = DEFAULT_MAX_MISSED_TICKS,
        normal_speed_ratio: float = DEFAULT_NORMAL_SPEED_RATIO,
        free_flow_speed_kmh: float = 110.0,
    ) -> None:
        if not 0.0 < iou_threshold <= 1.0:
            raise ValueError("iou_threshold must be in (0, 1]")
        if required_ticks < 2:
            raise ValueError("required_ticks must be >= 2")
        if max_missed_ticks < 0:
            raise ValueError("max_missed_ticks must be >= 0")
        if not 0.0 < normal_speed_ratio <= 1.0:
            raise ValueError("normal_speed_ratio must be in (0, 1]")

        self.iou_threshold = iou_threshold
        self.required_ticks = required_ticks
        self.max_missed_ticks = max_missed_ticks
        self.normal_speed_ratio = normal_speed_ratio
        self.free_flow_speed_kmh = free_flow_speed_kmh
        self._tracks: dict[str, list[_Track]] = {}
        self._lock = Lock()

    def update(
        self,
        camera_id: str,
        detections: list[dict[str, Any]],
        *,
        timestamp: datetime,
        local_speed_kmh: float | None,
    ) -> StoppedVehicleEvent | None:
        """Update tracks for one camera and return a stopped event if gated in."""
        with self._lock:
            tracks = self._update_tracks(camera_id, detections, timestamp)
            if not self._speed_is_normal(local_speed_kmh):
                return None

            mature = [track for track in tracks if track.hits >= self.required_ticks]
            if not mature:
                return None
            best = max(mature, key=lambda t: (t.hits, t.confidence))
            return StoppedVehicleEvent(
                camera_id=camera_id,
                box=best.box,
                confidence=best.confidence,
                persistence_ticks=best.hits,
            )

    def mark_camera_missed(self, camera_id: str) -> None:
        """Age tracks when a camera has no usable frame this tick."""
        with self._lock:
            tracks = self._tracks.get(camera_id)
            if not tracks:
                return
            for track in tracks:
                track.missed += 1
            self._tracks[camera_id] = [
                track for track in tracks if track.missed <= self.max_missed_ticks
            ]
            if not self._tracks[camera_id]:
                self._tracks.pop(camera_id, None)

    def reset(self, camera_id: str | None = None) -> None:
        """Clear one camera's tracks, or all tracks when camera_id is omitted."""
        with self._lock:
            if camera_id is None:
                self._tracks.clear()
            else:
                self._tracks.pop(camera_id, None)

    def _update_tracks(
        self,
        camera_id: str,
        detections: list[dict[str, Any]],
        timestamp: datetime,
    ) -> list[_Track]:
        current = list(self._tracks.get(camera_id, []))
        boxes = [_normalise_detection(det) for det in detections]
        boxes = [box for box in boxes if box is not None]

        matched_track_indexes: set[int] = set()
        for box, confidence in boxes:
            best_index: int | None = None
            best_iou = 0.0
            for index, track in enumerate(current):
                if index in matched_track_indexes:
                    continue
                iou = box_iou(track.box, box)
                if iou > best_iou:
                    best_iou = iou
                    best_index = index

            if best_index is not None and best_iou >= self.iou_threshold:
                track = current[best_index]
                track.box = box
                track.confidence = max(track.confidence, confidence)
                track.hits += 1
                track.missed = 0
                track.last_seen = timestamp
                matched_track_indexes.add(best_index)
            else:
                current.append(
                    _Track(
                        box=box,
                        confidence=confidence,
                        hits=1,
                        missed=0,
                        last_seen=timestamp,
                    )
                )
                matched_track_indexes.add(len(current) - 1)

        for index, track in enumerate(current):
            if index not in matched_track_indexes:
                track.missed += 1

        current = [track for track in current if track.missed <= self.max_missed_ticks]
        if current:
            self._tracks[camera_id] = current
        else:
            self._tracks.pop(camera_id, None)
        return current

    def _speed_is_normal(self, local_speed_kmh: float | None) -> bool:
        if local_speed_kmh is None:
            return False
        return local_speed_kmh >= self.free_flow_speed_kmh * self.normal_speed_ratio


def _normalise_detection(
    detection: dict[str, Any],
) -> tuple[tuple[float, float, float, float], float] | None:
    xyxy = detection.get("xyxy")
    if xyxy is None or len(xyxy) != 4:
        return None
    box = tuple(float(v) for v in xyxy)
    confidence = float(detection.get("confidence", 0.0))
    return box, confidence
