"""
ROI (Region of Interest) Mapper for the Proactive Traffic Routing Engine (PTRE).

Maps 2D pixel coordinates from YOLO detections to physical road segments
using predefined polygons per camera.  The object detection model evaluates
static 2D images — it cannot infer geographic context or travel direction.
This module bridges that gap by assigning each detection to a named road
segment via point-in-polygon tests.

Expert Audit Fix 2: When a homography matrix is present for a camera, all
detection points are projected into Bird's-Eye View (BEV) coordinates before
classification.  This eliminates perspective distortion errors from the
1D polynomial approach.

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

import cv2
import numpy as np
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
    roi_length_confidence: str | None = Field(
        default=None,
        description="Confidence class for roi_length_meters calibration",
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
        self._homography: dict[str, np.ndarray] = {}  # camera_id → 3×3 H matrix
        self._exclusion_zones: dict[str, list[tuple[int, int, int, int]]] = {}
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
            # -- Load homography matrix if present (Expert Audit Fix 2) --
            h_raw = cam_data.get("homography_matrix")
            if h_raw is not None:
                try:
                    H = np.array(h_raw, dtype=np.float64)
                    if H.shape == (3, 3):
                        self._homography[cam_id] = H
                        logger.info(
                            "BEV homography loaded for camera '%s'", cam_id
                        )
                    else:
                        logger.warning(
                            "homography_matrix for '%s' has wrong shape %s — ignoring",
                            cam_id,
                            H.shape,
                        )
                except (ValueError, TypeError) as e:
                    logger.warning(
                        "Invalid homography_matrix for '%s': %s — ignoring",
                        cam_id,
                        e,
                    )
            else:
                logger.debug(
                    "Camera '%s' has no homography_matrix — using pixel-space fallback",
                    cam_id,
                )

            # -- Load exclusion zones (static false-positive suppression) --
            ez_raw = cam_data.get("exclusion_zones", [])
            zones: list[tuple[int, int, int, int]] = []
            for zone in ez_raw:
                if isinstance(zone, list) and len(zone) == 4:
                    zones.append(tuple(int(v) for v in zone))  # type: ignore[arg-type]
            if zones:
                self._exclusion_zones[cam_id] = zones
                logger.info(
                    "Loaded %d exclusion zone(s) for camera '%s'",
                    len(zones),
                    cam_id,
                )

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
            "ROI config loaded: %d camera(s) with ROI definitions, "
            "%d with BEV homography, %d with exclusion zones",
            len(self._camera_rois),
            len(self._homography),
            len(self._exclusion_zones),
        )

    # -- Exclusion zones (static false-positive suppression) -------------------

    def get_exclusion_zones(
        self, camera_id: str,
    ) -> list[tuple[int, int, int, int]]:
        """Return exclusion zone rectangles ``(x1, y1, x2, y2)`` for a camera."""
        return self._exclusion_zones.get(camera_id, [])

    def is_excluded(self, camera_id: str, cx: float, cy: float) -> bool:
        """Return True if point (cx, cy) falls inside any exclusion zone."""
        for x1, y1, x2, y2 in self._exclusion_zones.get(camera_id, []):
            if x1 <= cx <= x2 and y1 <= cy <= y2:
                return True
        return False

    # -- BEV transform (Expert Audit Fix 2) -----------------------------------

    def has_homography(self, camera_id: str) -> bool:
        """Return True if this camera has a BEV homography matrix."""
        return camera_id in self._homography

    def pixel_to_bev(
        self, camera_id: str, x: float, y: float
    ) -> tuple[float, float] | None:
        """Project a pixel point to Bird's-Eye View coordinates.

        Returns (bev_x, bev_y) in physical meters, or None if no
        homography is available for this camera.
        """
        H = self._homography.get(camera_id)
        if H is None:
            return None

        pts = np.array([[[x, y]]], dtype=np.float64)
        transformed = cv2.perspectiveTransform(pts, H)
        bev_x, bev_y = transformed[0][0]
        return (float(bev_x), float(bev_y))

    def transform_polygon_to_bev(
        self, camera_id: str, polygon: list[list[int]]
    ) -> list[list[float]] | None:
        """Transform an entire ROI polygon to BEV coordinates.

        Returns list of [bev_x, bev_y] points, or None if no homography.
        """
        H = self._homography.get(camera_id)
        if H is None:
            return None

        pts = np.array([polygon], dtype=np.float64)
        transformed = cv2.perspectiveTransform(pts, H)
        return [[float(p[0]), float(p[1])] for p in transformed[0]]

    def compute_roi_area_m2(self, camera_id: str, roi: ROIRegion) -> float | None:
        """Compute the physical area of an ROI in BEV plane (m²).

        Returns None if no homography is available.
        """
        bev_poly = self.transform_polygon_to_bev(camera_id, roi.polygon)
        if bev_poly is None:
            return None

        return Polygon(bev_poly).area

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

        When a homography matrix is available for this camera,
        both the detection point and ROI polygons are projected to
        BEV coordinates for accurate comparison (Expert Audit Fix 2).

        Returns the matching ROIRegion, or None if the point falls
        outside all defined ROIs (detection should be discarded).
        """
        cached = self._shapely_cache.get(camera_id)
        if not cached:
            return None

        # If homography is available, project to BEV plane
        if self.has_homography(camera_id):
            bev_pt = self.pixel_to_bev(camera_id, x, y)
            if bev_pt is not None:
                point = Point(bev_pt[0], bev_pt[1])
                for roi, _pixel_poly in cached:
                    bev_poly_pts = self.transform_polygon_to_bev(
                        camera_id, roi.polygon
                    )
                    if bev_poly_pts:
                        bev_poly = Polygon(bev_poly_pts)
                        if bev_poly.contains(point):
                            return roi
                return None

        # Fallback: pixel-space point-in-polygon (legacy behavior)
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
