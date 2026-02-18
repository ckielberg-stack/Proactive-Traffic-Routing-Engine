"""
ROI (Region of Interest) Mapper for the Proactive Traffic Routing Engine (PTRE).

Maps 2D pixel coordinates from YOLO detections to physical road segments
using predefined polygons per camera.  The object detection model evaluates
static 2D images — it cannot infer geographic context or travel direction.
This module bridges that gap by assigning each detection to a named road
segment via point-in-polygon tests.

Usage::

    from src.roi_mapper import ROIMapper

    mapper = ROIMapper("camera_config.json")
    rois = mapper.get_rois("SE_STA_CAMERA_Orion_412")
    region = mapper.classify_detection("SE_STA_CAMERA_Orion_412", x=400, y=600)
    if region:
        print(region.road_id, region.direction_relative_to_camera)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from pydantic import BaseModel, Field
from shapely.geometry import Point, Polygon

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class ROIRegion(BaseModel):
    """A single Region of Interest polygon mapping pixels to a road segment."""

    road_id: str
    direction_relative_to_camera: str = Field(
        description="'towards' (traffic approaching camera) or 'away' (traffic receding)"
    )
    capacity_vph: float = Field(ge=0, description="Theoretical max throughput for this segment")
    num_lanes: int = Field(ge=1, default=2)
    roi_length_meters: float = Field(
        default=100.0,
        ge=1.0,
        description="Physical road length visible in this ROI (meters). "
        "Used to convert vehicle count → density for shockwave math.",
    )
    polygon: list[list[int]] = Field(
        description="List of [x, y] pixel coordinates defining the ROI boundary"
    )

    @property
    def shapely_polygon(self) -> Polygon:
        """Return a Shapely Polygon for point-in-polygon tests."""
        return Polygon(self.polygon)


# ---------------------------------------------------------------------------
# ROI Mapper
# ---------------------------------------------------------------------------


class ROIMapper:
    """Loads per-camera ROI configurations and classifies detections.

    Parameters
    ----------
    config_path:
        Path to ``camera_config.json``.  If the file does not exist or is
        malformed, the mapper gracefully degrades — all cameras return
        empty ROI lists (full-frame fallback).
    """

    def __init__(self, config_path: str | Path = "camera_config.json") -> None:
        self._config_path = Path(config_path)
        self._camera_rois: dict[str, list[ROIRegion]] = {}
        self._shapely_cache: dict[str, list[tuple[ROIRegion, Polygon]]] = {}
        self._load_config()

    # -- loading --------------------------------------------------------------

    def _load_config(self) -> None:
        """Parse camera_config.json into structured ROIRegion objects."""
        if not self._config_path.exists():
            logger.warning(
                "ROI config not found: %s — all cameras use full-frame fallback",
                self._config_path,
            )
            return

        try:
            raw = json.loads(self._config_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            logger.error("Failed to parse ROI config: %s", e)
            return

        cameras_raw = raw.get("cameras", {})
        for cam_id, cam_data in cameras_raw.items():
            rois: list[ROIRegion] = []
            for roi_raw in cam_data.get("rois", []):
                try:
                    roi = ROIRegion(**roi_raw)
                    if len(roi.polygon) < 3:
                        logger.warning(
                            "ROI '%s' for camera '%s' has < 3 vertices — skipping",
                            roi.road_id,
                            cam_id,
                        )
                        continue
                    if "roi_length_meters" not in roi_raw:
                        logger.warning(
                            "⚠️  ROI '%s' for camera '%s' missing roi_length_meters "
                            "— defaulting to 100 m. Re-run roi_helper.py to calibrate.",
                            roi.road_id,
                            cam_id,
                        )
                    rois.append(roi)
                except Exception as e:
                    logger.error(
                        "Invalid ROI definition for camera '%s': %s", cam_id, e
                    )
            if rois:
                self._camera_rois[cam_id] = rois
                # Pre-build Shapely polygons for fast lookups
                self._shapely_cache[cam_id] = [
                    (roi, Polygon(roi.polygon)) for roi in rois
                ]

        logger.info(
            "ROI config loaded: %d camera(s) with ROI definitions",
            len(self._camera_rois),
        )

    # -- public API -----------------------------------------------------------

    def has_rois(self, camera_id: str) -> bool:
        """Return True if this camera has ROI definitions."""
        return camera_id in self._camera_rois

    def get_rois(self, camera_id: str) -> list[ROIRegion]:
        """Return ROI regions for a camera, or empty list if unconfigured."""
        return self._camera_rois.get(camera_id, [])

    def classify_detection(
        self, camera_id: str, x: float, y: float
    ) -> ROIRegion | None:
        """Determine which ROI polygon contains the point (x, y).

        Uses the **bottom-center** convention: callers should pass
        ``x = (x1 + x2) / 2``, ``y = y2`` (tire contact point on road).

        Returns the matching ROIRegion, or None if the point falls
        outside all defined ROIs (detection should be discarded).
        """
        cached = self._shapely_cache.get(camera_id)
        if not cached:
            return None

        point = Point(x, y)
        for roi, poly in cached:
            if poly.contains(point):
                return roi

        return None

    def classify_detections_batch(
        self,
        camera_id: str,
        detections: list[dict],
    ) -> dict[str, list[dict]]:
        """Classify a batch of YOLO detections into road segments.

        Parameters
        ----------
        camera_id:
            Camera that produced the detections.
        detections:
            List of detection dicts with ``xyxy`` key
            ``(x1, y1, x2, y2)``.

        Returns
        -------
        dict mapping ``road_id`` → list of detection dicts that fall
        within that segment.  Detections outside all ROIs are discarded.
        """
        segments: dict[str, list[dict]] = {}

        for det in detections:
            x1, y1, x2, y2 = det["xyxy"]
            # Bottom-center = tire contact point (minimises perspective shift)
            bx = (x1 + x2) / 2
            by = y2

            region = self.classify_detection(camera_id, bx, by)
            if region is not None:
                segments.setdefault(region.road_id, []).append(det)

        return segments

    @property
    def configured_cameras(self) -> list[str]:
        """Return list of camera IDs that have ROI definitions."""
        return list(self._camera_rois.keys())
