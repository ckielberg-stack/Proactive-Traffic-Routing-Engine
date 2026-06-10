"""
Vision & Capacity Engine for the Proactive Traffic Routing Engine (PTRE).

Analyses highway camera frames with YOLO to calculate real-time road
capacity (Vehicles Per Hour) and detect anomalies (accidents, blockages).

Supports two modes:

1. **Single-ROI** (legacy): ``analyze_array()`` / ``analyze_frame()`` →
   returns a single ``CapacityState`` for the entire frame.

2. **Multi-ROI**: ``analyze_multi_roi()`` → classifies each detection
   into predefined road-segment polygons and returns a
   ``MultiSegmentCapacity`` with per-segment counts and capacity.

Usage::

    from src.vision_engine import VisionEngine
    from src.roi_mapper import ROIMapper
    from src.models import CameraMetadata, SensorReading

    engine = VisionEngine()
    mapper = ROIMapper("camera_config.json")

    # Single-ROI (backward compatible)
    state = engine.analyze_array(frame, meta)

    # Multi-ROI
    multi = engine.analyze_multi_roi(frame, meta, mapper)
    for seg in multi.segments:
        print(seg.road_id, seg.vehicle_count, seg.capacity_vph)
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from src.models import (
    CameraMetadata,
    CapacityState,
    MultiSegmentCapacity,
    RoadSegmentState,
    SensorReading,
)
from src.roi_mapper import ROIMapper

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# COCO class IDs for vehicles (https://docs.ultralytics.com/datasets/detect/coco/)
# ---------------------------------------------------------------------------
VEHICLE_CLASS_IDS: set[int] = {
    2,   # car
    3,   # motorcycle
    5,   # bus
    7,   # truck
}

# ---------------------------------------------------------------------------
# Traffic engineering constants
# ---------------------------------------------------------------------------
from src.traffic_constants import (
    FREE_FLOW_SPEED_KMH,
    JAM_DENSITY_VEH_KM_LANE,
    K_CRITICAL_VEH_KM_LANE,
    Q_CAP_VPH_PER_LANE,
)

DEFAULT_ROI_LENGTH_KM: float = 0.1       # Legacy fallback (100 m) for uncalibrated ROIs
BLACK_IMAGE_THRESHOLD: int = 15           # Mean pixel value below this → "black"
SPEED_DROP_RATIO: float = 0.50            # >50 % speed drop → severe event

# Anomaly: bounding-box width/height ratio above this suggests a sideways vehicle
ABNORMAL_ASPECT_RATIO: float = 3.5

# ---------------------------------------------------------------------------
# TODO: Implement Headlight/Taillight classification
# In harsh Swedish winter nights, standard chassis detection fails.
# Classifying red clusters (taillights = moving away) vs white clusters
# (headlights = moving towards) will serve as secondary verification
# for travel direction and counting.
#
# Approach:
#   1. Convert frame to HSV colour space.
#   2. Threshold for red hue range (taillights) and high-value white
#      (headlights) in the night-time luminance regime.
#   3. Cluster qualifying pixels (DBSCAN or connected components).
#   4. Each cluster centroid is treated as a "light detection" and
#      classified into the matching ROI polygon for direction inference.
#   5. Use as fallback when YOLO confidence is below a threshold or
#      when the frame mean brightness is below a night-mode cutoff.
# ---------------------------------------------------------------------------


class VisionEngine:
    """Perception module that converts camera frames into capacity estimates.

    The YOLO model is loaded lazily on first ``analyze_frame`` call so that
    import-time stays fast.
    """

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        confidence: float = 0.25,
    ) -> None:
        self._model_path = model_path
        self._confidence = confidence
        self._model: YOLO | None = None

    # -- lazy model loading --------------------------------------------------

    @property
    def model(self) -> YOLO:
        """Return the loaded YOLO model (download on first use)."""
        if self._model is None:
            logger.info("Loading YOLO model: %s …", self._model_path)
            self._model = YOLO(self._model_path)
            logger.info("YOLO model loaded successfully.")
        return self._model

    # -- public API -----------------------------------------------------------

    def analyze_frame(
        self,
        image_path: str | Path,
        camera_meta: CameraMetadata,
        sensor: SensorReading | None = None,
        roi_polygon: list[tuple[int, int]] | None = None,
    ) -> CapacityState:
        """Analyse a camera frame from disk and return a CapacityState.

        Thin wrapper around :meth:`analyze_array` — loads the image from
        *image_path* and delegates.  Prefer ``analyze_array`` in pipelines
        where the image is already in memory.
        """
        image_path = Path(image_path)

        frame = cv2.imread(str(image_path))
        if frame is None:
            logger.warning("Could not read image: %s", image_path)
            return self._fallback_state(
                datetime.now(), camera_meta, sensor, reason="image_unreadable"
            )

        return self.analyze_array(frame, camera_meta, sensor, roi_polygon)

    def analyze_array(
        self,
        frame: np.ndarray,
        camera_meta: CameraMetadata,
        sensor: SensorReading | None = None,
        roi_polygon: list[tuple[int, int]] | None = None,
    ) -> CapacityState:
        """Analyse an already-decoded camera frame entirely in memory.

        Parameters
        ----------
        frame:
            BGR image as a NumPy array (e.g. from ``cv2.imdecode``).
        camera_meta:
            Static metadata for this camera (lanes, road, etc.).
        sensor:
            Optional upstream sensor reading for sensor-fusion / fallback.
        roi_polygon:
            Optional polygon (list of (x, y) pixel coordinates) defining
            the Region of Interest.  If *None*, the entire frame is used.
        """
        now = datetime.now()

        # -- Step 1: Check for black / unavailable image ---------------------
        if self._is_black_image(frame):
            return self._handle_black_image(now, camera_meta, sensor)

        # -- Step 2: Run YOLO inference --------------------------------------
        # Run YOLO once on the full frame. Filter by ROI afterwards to
        # avoid double inference while still knowing the total frame count
        # for anomaly checks (prevents false positives when vehicles are
        # visible in opposite-direction traffic outside the ROI).
        all_frame_detections = self._detect_vehicles(frame, roi_polygon=None)
        if roi_polygon and len(roi_polygon) >= 3:
            h, w = frame.shape[:2]
            roi_mask = np.zeros((h, w), dtype=np.uint8)
            pts = np.array(roi_polygon, dtype=np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(roi_mask, [pts], 255)
            detections = []
            for d in all_frame_detections:
                x1, _, x2, y2 = d["xyxy"]
                cx = int((x1 + x2) / 2)
                cy = min(int(y2), h - 1)
                cx = min(cx, w - 1)
                if roi_mask[cy, cx] != 0:
                    detections.append(d)
        else:
            detections = all_frame_detections

        # -- Step 3: Check for anomalous bounding boxes ----------------------
        anomaly, anomaly_reason = self._check_anomalies(
            detections, sensor, camera_meta,
            total_frame_detections=len(all_frame_detections),
        )

        # -- Step 4: Estimate density (Expert Audit Fix 1) --------------------
        vehicle_count = len(detections)
        avg_conf = (
            float(np.mean([d["confidence"] for d in detections]))
            if detections
            else 0.0
        )

        density_veh_km_lane = self._estimate_density(
            vehicle_count, camera_meta.num_lanes
        )

        # Determine capacity: use static Q_cap for free-flow;
        # when density exceeds k_critical, compute reduced actual flow
        # using q = k * v (fundamental diagram) as the bottleneck throughput.
        speed = sensor.average_speed_kmh if sensor else FREE_FLOW_SPEED_KMH
        if density_veh_km_lane > K_CRITICAL_VEH_KM_LANE:
            # Congestion: actual flow = density × speed (reduced throughput)
            capacity_vph = density_veh_km_lane * camera_meta.num_lanes * speed
            max_cap = Q_CAP_VPH_PER_LANE * camera_meta.num_lanes
            capacity_vph = min(capacity_vph, max_cap)
            # Mark as congestion anomaly if not already flagged
            if not anomaly:
                anomaly = True
                anomaly_reason = "density_exceeds_k_critical"
        else:
            # Free-flow: road is at full theoretical capacity
            capacity_vph = Q_CAP_VPH_PER_LANE * camera_meta.num_lanes

        # If anomaly detected, reduce capacity proportionally
        blocked_lanes = 0
        if anomaly:
            blocked_lanes = self._estimate_blocked_lanes(
                detections, camera_meta.num_lanes
            )
            if blocked_lanes > 0:
                lane_fraction = (
                    (camera_meta.num_lanes - blocked_lanes) / camera_meta.num_lanes
                )
                capacity_vph *= max(lane_fraction, 0.0)

        return CapacityState(
            timestamp=now,
            camera_id=camera_meta.camera_id,
            vehicle_count=vehicle_count,
            blocked_lanes=blocked_lanes,
            total_lanes=camera_meta.num_lanes,
            estimated_capacity_vph=round(capacity_vph, 1),
            observed_density_veh_km_lane=round(density_veh_km_lane, 2),
            is_anomaly=anomaly,
            anomaly_reason=anomaly_reason,
            confidence=round(avg_conf, 3),
        )

    def analyze_multi_roi(
        self,
        frame: np.ndarray,
        camera_meta: CameraMetadata,
        roi_mapper: ROIMapper,
        sensor: SensorReading | None = None,
    ) -> MultiSegmentCapacity:
        """Analyse a frame with per-road-segment ROI classification.

        Runs YOLO inference once on the full frame, then classifies each
        detection into the predefined ROI polygons for this camera using
        the bottom-center point (tire contact).  Detections outside all
        ROIs are counted but **discarded** from capacity calculations.

        Parameters
        ----------
        frame:
            BGR image as a NumPy array.
        camera_meta:
            Static metadata for this camera.
        roi_mapper:
            Loaded ``ROIMapper`` instance with camera ROI definitions.
        sensor:
            Optional upstream sensor reading for speed estimates.

        Returns
        -------
        MultiSegmentCapacity with one ``RoadSegmentState`` per defined ROI.
        """
        now = datetime.now()

        # -- Black image check -----------------------------------------------
        if self._is_black_image(frame):
            # Return empty segments for all configured ROIs
            rois = roi_mapper.get_rois(camera_meta.camera_id)
            return MultiSegmentCapacity(
                timestamp=now,
                camera_id=camera_meta.camera_id,
                segments=[
                    RoadSegmentState(
                        road_id=roi.road_id,
                        direction=roi.direction_relative_to_camera,
                        vehicle_count=0,
                        capacity_vph=0.0,
                        observed_density_veh_km_lane=0.0,
                        num_lanes=roi.num_lanes,
                    )
                    for roi in rois
                ],
                unmatched_detections=0,
            )

        # -- Run YOLO on full frame (no ROI mask) ----------------------------
        all_detections = self._detect_vehicles(frame, roi_polygon=None)

        # -- Classify detections into road segments --------------------------
        segment_dets = roi_mapper.classify_detections_batch(
            camera_meta.camera_id, all_detections
        )

        # Count unmatched (outside all ROIs → discarded)
        matched_count = sum(len(dets) for dets in segment_dets.values())
        unmatched = len(all_detections) - matched_count

        # -- Build per-segment states ----------------------------------------
        rois = roi_mapper.get_rois(camera_meta.camera_id)
        speed = sensor.average_speed_kmh if sensor else FREE_FLOW_SPEED_KMH
        segments: list[RoadSegmentState] = []

        for roi in rois:
            dets = segment_dets.get(roi.road_id, [])
            count = len(dets)
            avg_conf = (
                float(np.mean([d["confidence"] for d in dets]))
                if dets
                else 0.0
            )

            # Estimate density using per-ROI physical length
            roi_length_km = roi.roi_length_meters / 1000.0
            density_vkl = self._estimate_density(
                count, roi.num_lanes, roi_length_km=roi_length_km,
            )

            # Determine capacity from density vs k_critical
            if density_vkl > K_CRITICAL_VEH_KM_LANE:
                capacity = density_vkl * roi.num_lanes * speed
                capacity = min(capacity, Q_CAP_VPH_PER_LANE * roi.num_lanes)
            else:
                capacity = Q_CAP_VPH_PER_LANE * roi.num_lanes

            # Check anomalies within this segment
            anomaly, anomaly_reason = self._check_anomalies(
                dets, sensor, camera_meta,
                total_frame_detections=len(all_detections),
            )
            if density_vkl > K_CRITICAL_VEH_KM_LANE and not anomaly:
                anomaly = True
                anomaly_reason = "density_exceeds_k_critical"

            segments.append(
                RoadSegmentState(
                    road_id=roi.road_id,
                    direction=roi.direction_relative_to_camera,
                    vehicle_count=count,
                    capacity_vph=round(capacity, 1),
                    observed_density_veh_km_lane=round(density_vkl, 2),
                    num_lanes=roi.num_lanes,
                    is_anomaly=anomaly,
                    anomaly_reason=anomaly_reason,
                    confidence=round(avg_conf, 3),
                )
            )

        logger.debug(
            "Multi-ROI %s: %d total detections, %d matched, %d unmatched",
            camera_meta.camera_id,
            len(all_detections),
            matched_count,
            unmatched,
        )

        return MultiSegmentCapacity(
            timestamp=now,
            camera_id=camera_meta.camera_id,
            segments=segments,
            unmatched_detections=unmatched,
        )

    # -- internal helpers ----------------------------------------------------

    def _is_black_image(self, frame: np.ndarray) -> bool:
        """Return True if the frame is nearly all black (camera off)."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return float(np.mean(gray)) < BLACK_IMAGE_THRESHOLD

    def _handle_black_image(
        self,
        now: datetime,
        meta: CameraMetadata,
        sensor: SensorReading | None,
    ) -> CapacityState:
        """Sensor-fusion fallback for a black / unavailable image.

        If the upstream sensor also shows a severe speed drop (> 50 %),
        we assume a serious accident → capacity = 0.
        Otherwise we simply mark low confidence.
        """
        if sensor and sensor.average_speed_kmh < FREE_FLOW_SPEED_KMH * SPEED_DROP_RATIO:
            logger.warning(
                "Black image + severe speed drop (%.1f km/h) → capacity=0",
                sensor.average_speed_kmh,
            )
            return CapacityState(
                timestamp=now,
                camera_id=meta.camera_id,
                vehicle_count=0,
                blocked_lanes=meta.num_lanes,
                total_lanes=meta.num_lanes,
                estimated_capacity_vph=0.0,
                is_anomaly=True,
                anomaly_reason="black_image_with_speed_drop",
                confidence=0.0,
            )

        # Black image but no sensor data or speed is still ok
        logger.info("Black image without speed drop — low confidence estimate.")
        return CapacityState(
            timestamp=now,
            camera_id=meta.camera_id,
            vehicle_count=0,
            blocked_lanes=0,
            total_lanes=meta.num_lanes,
            estimated_capacity_vph=0.0,
            is_anomaly=False,
            anomaly_reason="black_image_no_sensor_confirmation",
            confidence=0.0,
        )

    def _detect_vehicles(
        self,
        frame: np.ndarray,
        roi_polygon: list[tuple[int, int]] | None,
    ) -> list[dict]:
        """Run YOLO and return vehicle detections inside the ROI.

        Each detection dict has keys: ``xyxy``, ``class_id``, ``class_name``,
        ``confidence``, ``aspect_ratio``.
        """
        results = self.model.predict(
            source=frame,
            conf=self._confidence,
            imgsz=640,
            verbose=False,
            save=False,
        )

        detections: list[dict] = []
        if not results or len(results) == 0:
            return detections

        result = results[0]
        if result.boxes is None or len(result.boxes) == 0:
            return detections

        boxes_xyxy = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)

        # Build ROI mask once (if provided)
        roi_mask: np.ndarray | None = None
        if roi_polygon is not None and len(roi_polygon) >= 3:
            h, w = frame.shape[:2]
            roi_mask = np.zeros((h, w), dtype=np.uint8)
            pts = np.array(roi_polygon, dtype=np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(roi_mask, [pts], 255)

        for i in range(len(boxes_xyxy)):
            cls_id = int(class_ids[i])
            if cls_id not in VEHICLE_CLASS_IDS:
                continue

            x1, y1, x2, y2 = boxes_xyxy[i]
            # Bottom-center = tire contact point on road surface.
            # This minimises perspective distortion compared to
            # bounding-box center, since the bottom edge represents
            # where the vehicle touches the ground plane.
            cx = (x1 + x2) / 2
            cy = y2

            # Filter by ROI
            if roi_mask is not None:
                cy_clamp = min(int(cy), roi_mask.shape[0] - 1)
                cx_clamp = min(int(cx), roi_mask.shape[1] - 1)
                if roi_mask[cy_clamp, cx_clamp] == 0:
                    continue

            width = x2 - x1
            height = y2 - y1
            aspect = width / max(height, 1.0)

            detections.append(
                {
                    "xyxy": (float(x1), float(y1), float(x2), float(y2)),
                    "class_id": cls_id,
                    "class_name": self.model.names.get(cls_id, f"class_{cls_id}"),
                    "confidence": float(confs[i]),
                    "aspect_ratio": float(aspect),
                }
            )

        return detections

    def _check_anomalies(
        self,
        detections: list[dict],
        sensor: SensorReading | None,
        meta: CameraMetadata,
        *,
        total_frame_detections: int | None = None,
    ) -> tuple[bool, str | None]:
        """Detect anomalous conditions from detections + sensor data.

        Parameters
        ----------
        detections:
            Vehicle detections *inside* the ROI polygon.
        total_frame_detections:
            Total vehicle detections in the *entire frame* (all ROIs +
            unmatched).  Used for cases 2 & 3 to avoid false positives
            when vehicles are visible outside the current ROI (e.g.
            opposite-direction traffic).
        """
        # Fall back to ROI count if full-frame count not provided
        frame_count = (
            total_frame_detections
            if total_frame_detections is not None
            else len(detections)
        )

        # Case 1: abnormally wide bounding boxes (sideways vehicle / debris)
        wide_boxes = [
            d for d in detections if d["aspect_ratio"] > ABNORMAL_ASPECT_RATIO
        ]
        if wide_boxes:
            return True, f"abnormal_aspect_ratio ({len(wide_boxes)} boxes)"

        # Case 2: zero detections IN ENTIRE FRAME but high sensor inflow.
        # Only triggers when the camera truly sees nothing — not when
        # vehicles are visible in other ROIs / opposite direction.
        if (
            frame_count == 0
            and sensor is not None
            and sensor.inflow_volume_vph > 500
        ):
            return True, "zero_detections_high_inflow"

        # Case 3: sensor speed drop with few vehicles in entire frame
        if (
            sensor is not None
            and sensor.average_speed_kmh < FREE_FLOW_SPEED_KMH * SPEED_DROP_RATIO
            and frame_count < 2
        ):
            return True, "speed_drop_low_detections"

        return False, None

    def _estimate_density(
        self,
        vehicle_count: int,
        num_lanes: int,
        roi_length_km: float = DEFAULT_ROI_LENGTH_KM,
    ) -> float:
        """Estimate road density in vehicles per km per lane.

        Expert Audit Fix 1: This method outputs DENSITY, not flow.
        The old ``_estimate_capacity`` incorrectly computed ``q = k × v``
        (current flow) and labelled it as capacity.  Now the Vision Engine's
        only spatial job is to calculate localised density ``k``.

        The caller decides whether ``k > k_critical`` (congestion) or not
        (free-flow) and sets capacity accordingly.

        A safety clamp prevents density from exceeding the theoretical
        jam density (133 veh/km/lane).  This guards against absurd values
        when YOLO hallucinates overlapping bounding boxes in snow/rain.
        """
        if vehicle_count == 0:
            return 0.0

        density_per_km = vehicle_count / roi_length_km
        density_per_km_lane = density_per_km / max(num_lanes, 1)

        # Safety clamp: cap at jam density per lane
        if density_per_km_lane > JAM_DENSITY_VEH_KM_LANE:
            logger.warning(
                "Density %.1f veh/km/lane exceeds jam density %.1f "
                "(count=%d, ROI=%.0f m, lanes=%d) — clamping",
                density_per_km_lane,
                JAM_DENSITY_VEH_KM_LANE,
                vehicle_count,
                roi_length_km * 1000,
                num_lanes,
            )
            density_per_km_lane = JAM_DENSITY_VEH_KM_LANE

        return density_per_km_lane

    def _estimate_blocked_lanes(
        self,
        detections: list[dict],
        total_lanes: int,
    ) -> int:
        """Heuristic: count lanes blocked by abnormally wide bounding boxes."""
        wide = [d for d in detections if d["aspect_ratio"] > ABNORMAL_ASPECT_RATIO]
        if not wide:
            return 0
        # Each abnormally wide box blocks ~1 lane; cap at total - 1
        return min(len(wide), total_lanes - 1)

    def _fallback_state(
        self,
        now: datetime,
        meta: CameraMetadata,
        sensor: SensorReading | None,
        reason: str,
    ) -> CapacityState:
        """Return a zero-capacity state when the image is unreadable."""
        return CapacityState(
            timestamp=now,
            camera_id=meta.camera_id,
            vehicle_count=0,
            blocked_lanes=0,
            total_lanes=meta.num_lanes,
            estimated_capacity_vph=0.0,
            is_anomaly=True,
            anomaly_reason=reason,
            confidence=0.0,
        )
