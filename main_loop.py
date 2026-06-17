#!/usr/bin/env python3
"""
Tick-Based Main Loop for the Proactive Traffic Routing Engine (PTRE).

Implements a discrete 60-second polling architecture (like a cron job).
Each tick:
  1. Concurrently fetches camera images, sensor data, and VMS statuses
  2. Runs stateless YOLO inference → CapacityState[]
  3. Runs LWR physics engine → QueuePrediction[]
  4. Runs VMS orchestrator → VMSRecommendation[]
  5. Persists all data to JSONL + dashboard state files

The system does NOT race humans to detect crashes — humans have live video.
Our value is **predictive queue tail modeling** using kinematic wave theory:
predicting *when* the queue will reach upstream VMS signs so operators can
activate speed warnings preemptively.

Usage:
    python main_loop.py              # Run continuous 60s ticks
    python main_loop.py --once       # Run one tick (for testing)
    python main_loop.py --interval 30  # Custom interval
"""

from __future__ import annotations

import argparse
import json
import logging
import logging.handlers
import os
import re
import signal
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from threading import Lock, local
from typing import Callable

import cv2
import numpy as np
import requests
from dotenv import load_dotenv

from config import (
    API_KEY,
    API_URL,
    BBOX,
    CAMERA_COORDS,
    CAMERA_IDS,
    DATA_DIR,
    DEFAULT_ROAD_SPEED_LIMIT,
    E4_NORTHBOUND_CORRIDOR_LENGTH_KM,
    E4_NORTHBOUND_ROUTE_POINTS,
    E4_NORTHBOUND_TRAVEL_TIME_ROUTE_IDS,
    E4_TRAVEL_TIME_ROUTE_IDS,
    INTERVAL_SECONDS,
    MAX_RETRIES,
    RETRY_BACKOFF,
    SENSOR_COORDS,
    SENSOR_ROAD_SPEED_LIMITS,
    SENSOR_SEVERE_DROP_RATIO,
    SENSOR_SPEED_DROP_RATIO,
)
from retention import RetentionPolicy
from src.models import (
    CalibrationSnapshot,
    CameraMetadata,
    CapacityState,
    MultiSegmentCapacity,
    SituationDeviation,
    SensorAnomaly,
    SensorReading,
    SegmentTrafficState,
    TickResult,
    TravelTimeReading,
    VMSStatusSnapshot,
)
from src.travel_time_calibrator import TravelTimeCalibrator
from src.anomaly_store import record_anomaly, get_total_count
from src.density_smoother import DensitySmoother
from src.physics_engine import PhysicsEngine
from src.roi_mapper import ROIMapper
from src.route_chainage import (
    build_route_chainage_map,
    find_nearest_by_chainage,
    RouteProjector,
)
from src.track_persistence import TrackPersistence
from src.traffic_constants import (
    FREE_FLOW_SPEED_KMH,
    K_CRITICAL_VEH_KM_LANE,
    Q_CAP_VPH_PER_LANE,
)
from src.vision_engine import VisionEngine
from src.vms_orchestrator import VMSOrchestrator
from src.weather_adapter import WeatherAdapter

load_dotenv()

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"
LOG_DATE_FORMAT = "%H:%M:%S"

logger = logging.getLogger("mainloop")
logger.setLevel(logging.DEBUG)

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT))
logger.addHandler(ch)


def setup_file_logger(data_dir: str) -> None:
    os.makedirs(data_dir, exist_ok=True)
    fh = logging.handlers.RotatingFileHandler(
        os.path.join(data_dir, "mainloop.log"),
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT))
    logger.addHandler(fh)





# ---------------------------------------------------------------------------
# Graceful shutdown
# ---------------------------------------------------------------------------
_shutdown = False


def _signal_handler(sig, frame):
    global _shutdown
    _shutdown = True
    logger.info("🛑 Avslutningssignal mottagen — stänger efter denna tick")


signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def parse_point_wgs84(geom: str) -> tuple[float, float] | None:
    """Parse 'POINT (lng lat)' → (lat, lng) or None."""
    if not geom or "POINT" not in geom:
        return None
    try:
        parts = geom.replace("POINT (", "").replace(")", "").strip().split()
        return float(parts[1]), float(parts[0])
    except (ValueError, IndexError):
        return None


def get_deviation_wgs84(dev: dict) -> str | None:
    """Read Deviation.Geometry.WGS84 from nested or flattened API payloads."""
    geometry = dev.get("Geometry")
    if isinstance(geometry, dict):
        wgs84 = geometry.get("WGS84")
        return str(wgs84) if wgs84 else None

    wgs84 = dev.get("Geometry.WGS84")
    return str(wgs84) if wgs84 else None


def in_bbox(lat: float, lng: float) -> bool:
    return (
        BBOX["min_lat"] <= lat <= BBOX["max_lat"]
        and BBOX["min_lng"] <= lng <= BBOX["max_lng"]
    )


def project_e4_northbound_chainage(position: tuple[float, float]) -> float | None:
    """Project a lat/lng point onto the configured E4 northbound route datum."""
    if not in_bbox(*position):
        return None
    try:
        projector = RouteProjector(
            E4_NORTHBOUND_ROUTE_POINTS,
            E4_NORTHBOUND_CORRIDOR_LENGTH_KM,
        )
        chainage = projector.project_chainage(position)
    except (TypeError, ValueError):
        return None
    return round(chainage, 2) if chainage is not None else None


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def api_request(xml_query: str, retries: int = MAX_RETRIES) -> dict | None:
    for attempt in range(1, retries + 1):
        try:
            resp = requests.post(
                API_URL,
                data=xml_query.encode("utf-8"),
                headers={"Content-Type": "text/xml; charset=utf-8"},
                timeout=30,
            )
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            wait = RETRY_BACKOFF ** attempt
            logger.warning(f"API error (attempt {attempt}/{retries}): {e} — waiting {wait}s")
            if attempt < retries:
                time.sleep(wait)
    logger.error(f"API call failed after {retries} attempts")
    return None


def _val(d: dict, key: str):
    """Get nested .Value from a Trafikverket observation field."""
    sub = d.get(key, {})
    return sub.get("Value") if isinstance(sub, dict) else None


def _nested(d: dict, *keys):
    for key in keys:
        if isinstance(d, dict):
            d = d.get(key)
        else:
            return None
    return d


def fetch_image_bytes(url: str, retries: int = MAX_RETRIES) -> bytes | None:
    """Fetch image from URL into memory. No disk I/O."""
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            return resp.content
        except requests.RequestException as e:
            wait = RETRY_BACKOFF ** attempt
            logger.warning(f"Image fetch error (attempt {attempt}/{retries}): {e}")
            if attempt < retries:
                time.sleep(wait)
    return None


def decode_frame(raw_bytes: bytes) -> np.ndarray | None:
    """Decode JPEG bytes to a BGR numpy array in memory."""
    arr = np.frombuffer(raw_bytes, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


# ---------------------------------------------------------------------------
# Lazy singletons
# ---------------------------------------------------------------------------
_vision_engine: VisionEngine | None = None
_retention_policy: RetentionPolicy | None = None
_camera_worker_local = local()
_retention_lock = Lock()
_roi_mapper: ROIMapper | None = None
_physics_engine: PhysicsEngine | None = None
_vms_orchestrator: VMSOrchestrator | None = None
_density_smoother: DensitySmoother | None = None
_track_persistence: TrackPersistence | None = None
_travel_time_calibrator: TravelTimeCalibrator | None = None
_weather_adapter: WeatherAdapter | None = None


def _get_vision_engine() -> VisionEngine:
    global _vision_engine
    if _vision_engine is None:
        _vision_engine = VisionEngine()
    return _vision_engine


def _get_camera_worker_vision_engine() -> VisionEngine:
    """Return a thread-local vision engine for parallel camera inference."""
    engine = getattr(_camera_worker_local, "vision_engine", None)
    if engine is None:
        engine = VisionEngine()
        _camera_worker_local.vision_engine = engine
    return engine


def _get_retention_policy() -> RetentionPolicy:
    global _retention_policy
    if _retention_policy is None:
        _retention_policy = RetentionPolicy(base_dir=os.path.dirname(__file__))
    return _retention_policy


def _get_roi_mapper() -> ROIMapper:
    global _roi_mapper
    if _roi_mapper is None:
        config_path = os.path.join(os.path.dirname(__file__), "camera_config.json")
        _roi_mapper = ROIMapper(config_path)
    return _roi_mapper


def _get_physics_engine() -> PhysicsEngine:
    global _physics_engine
    if _physics_engine is None:
        _physics_engine = PhysicsEngine()
    return _physics_engine


def _get_vms_orchestrator() -> VMSOrchestrator:
    global _vms_orchestrator
    if _vms_orchestrator is None:
        _vms_orchestrator = VMSOrchestrator()
    return _vms_orchestrator


def _get_calibrator() -> TravelTimeCalibrator:
    global _travel_time_calibrator
    if _travel_time_calibrator is None:
        _travel_time_calibrator = TravelTimeCalibrator()
    return _travel_time_calibrator


def _get_weather_adapter() -> WeatherAdapter:
    global _weather_adapter
    if _weather_adapter is None:
        _weather_adapter = WeatherAdapter()
    return _weather_adapter


def _get_density_smoother() -> DensitySmoother:
    """Persistent density smoother — maintains EMA state across ticks."""
    global _density_smoother
    if _density_smoother is None:
        _density_smoother = DensitySmoother(alpha=0.4)
    return _density_smoother


def _get_track_persistence() -> TrackPersistence:
    """Persistent vehicle-box tracker — maintains minimal state across ticks."""
    global _track_persistence
    if _track_persistence is None:
        _track_persistence = TrackPersistence(free_flow_speed_kmh=FREE_FLOW_SPEED_KMH)
    return _track_persistence


# ---------------------------------------------------------------------------
# Camera chainage mapping (for physics engine)
# ---------------------------------------------------------------------------

def build_camera_chainage_map(
    camera_coords: dict[str, tuple[float, float]] | None = None,
    route_points: list[tuple[float, float]] | None = None,
    corridor_length_km: float = E4_NORTHBOUND_CORRIDOR_LENGTH_KM,
) -> dict[str, float]:
    """Build a route-linear camera chainage map.

    Parameters
    ----------
    camera_coords:
        Mapping of camera_id → (lat, lng).  Falls back to
        ``config.CAMERA_COORDS`` when not provided.
    route_points:
        Ordered route control points from Hallunda heading northbound.
    corridor_length_km:
        Chainage length matching the VMS datum.
    """
    coords = camera_coords if camera_coords is not None else CAMERA_COORDS
    route = route_points if route_points is not None else E4_NORTHBOUND_ROUTE_POINTS
    return {
        str(camera_id): chainage
        for camera_id, chainage in build_route_chainage_map(
            coords,
            route,
            corridor_length_km,
        ).items()
    }


# Backward-compatible alias
_build_camera_chainage_map = build_camera_chainage_map


def build_travel_time_speed_states(
    travel_time_readings: list[TravelTimeReading],
    camera_chainage_map: dict[str, float] | None = None,
    route_ids: list[str] | None = None,
    corridor_length_km: float = E4_NORTHBOUND_CORRIDOR_LENGTH_KM,
) -> dict[str, SegmentTrafficState]:
    """Map ordered northbound TravelTimeRoute speeds onto camera chainage spans."""
    if not travel_time_readings:
        return {}

    camera_chainages = camera_chainage_map or build_camera_chainage_map()
    ordered_route_ids = route_ids or E4_NORTHBOUND_TRAVEL_TIME_ROUTE_IDS
    route_order = {route_id: idx for idx, route_id in enumerate(ordered_route_ids)}
    northbound = [
        reading for reading in travel_time_readings
        if reading.route_id in route_order
        and reading.length_meters > 0
        and reading.speed_kmh > 0
    ]
    if not camera_chainages or not northbound:
        return {}

    northbound.sort(key=lambda reading: route_order[reading.route_id])
    total_length_m = sum(reading.length_meters for reading in northbound)
    if total_length_m <= 0:
        return {}

    states: dict[str, SegmentTrafficState] = {}
    span_start_km = 0.0
    for idx, reading in enumerate(northbound):
        span_length_km = (reading.length_meters / total_length_m) * corridor_length_km
        span_end_km = corridor_length_km if idx == len(northbound) - 1 else (
            span_start_km + span_length_km
        )

        for camera_id, chainage_km in camera_chainages.items():
            in_span = (
                span_start_km <= chainage_km <= span_end_km
                if idx == len(northbound) - 1
                else span_start_km <= chainage_km < span_end_km
            )
            if not in_span or camera_id in states:
                continue
            states[camera_id] = SegmentTrafficState(
                local_speed_kmh=round(reading.speed_kmh, 1),
                speed_source="travel_time",
                confidence="medium",
            )

        span_start_km = span_end_km

    return states


def build_node_traffic_states(
    sensor_readings: list[SensorReading],
    travel_time_readings: list[TravelTimeReading] | None = None,
    camera_coords: dict[str, tuple[float, float]] | None = None,
    sensor_coords: dict[int, tuple[float, float]] | None = None,
    route_points: list[tuple[float, float]] | None = None,
    corridor_length_km: float = E4_NORTHBOUND_CORRIDOR_LENGTH_KM,
) -> dict[str, SegmentTrafficState]:
    """Map local TrafficFlow and TravelTimeRoute data to camera nodes.

    When multiple sensors map to the same camera, their flows are summed
    and their speeds are volume-weighted when volume is available.

    Parameters
    ----------
    sensor_readings:
        Per-station readings from ``fetch_sensor_data`` (with ``site_id``).
    camera_coords:
        Mapping of camera_id → (lat, lng).  Falls back to
        ``config.CAMERA_COORDS`` when not provided.
    sensor_coords:
        Mapping of TrafficFlow SiteId → (lat, lng). Falls back to
        ``config.SENSOR_COORDS`` when not provided.
    route_points:
        Ordered route control points from Hallunda heading northbound.
    corridor_length_km:
        Chainage length matching the VMS datum.

    Returns
    -------
    dict[str, SegmentTrafficState]
        Mapping of camera_id → local traffic state.
    """
    coords = camera_coords if camera_coords is not None else CAMERA_COORDS
    sensors = sensor_coords if sensor_coords is not None else SENSOR_COORDS
    route = route_points if route_points is not None else E4_NORTHBOUND_ROUTE_POINTS
    if not coords:
        return {}

    camera_chainages = build_camera_chainage_map(
        coords,
        route_points=route,
        corridor_length_km=corridor_length_km,
    )
    states: dict[str, SegmentTrafficState] = {}

    if travel_time_readings:
        states.update(build_travel_time_speed_states(
            travel_time_readings,
            camera_chainage_map=camera_chainages,
            corridor_length_km=corridor_length_km,
        ))

    if not sensors or not sensor_readings:
        return states

    sensor_chainages = build_route_chainage_map(
        sensors,
        route,
        corridor_length_km,
    )
    if not camera_chainages or not sensor_chainages:
        return states

    aggregates: dict[str, dict[str, float | int]] = {}

    for reading in sensor_readings:
        if reading.site_id is None:
            continue
        sensor_chainage = sensor_chainages.get(reading.site_id)
        if sensor_chainage is None:
            continue

        best_cam = find_nearest_by_chainage(sensor_chainage, camera_chainages)
        if best_cam is None:
            continue

        best_cam_id = str(best_cam)
        aggregate = aggregates.setdefault(
            best_cam_id,
            {
                "volume": 0.0,
                "weighted_speed": 0.0,
                "speed_weight": 0.0,
                "speed_sum": 0.0,
                "speed_count": 0,
            },
        )
        aggregate["volume"] = float(aggregate["volume"]) + reading.inflow_volume_vph
        aggregate["speed_sum"] = float(aggregate["speed_sum"]) + reading.average_speed_kmh
        aggregate["speed_count"] = int(aggregate["speed_count"]) + 1
        if reading.inflow_volume_vph > 0:
            aggregate["weighted_speed"] = (
                float(aggregate["weighted_speed"])
                + reading.average_speed_kmh * reading.inflow_volume_vph
            )
            aggregate["speed_weight"] = (
                float(aggregate["speed_weight"]) + reading.inflow_volume_vph
            )

    for camera_id, aggregate in aggregates.items():
        volume = float(aggregate["volume"])
        speed_weight = float(aggregate["speed_weight"])
        speed_count = int(aggregate["speed_count"])
        if speed_weight > 0:
            speed = float(aggregate["weighted_speed"]) / speed_weight
        elif speed_count > 0:
            speed = float(aggregate["speed_sum"]) / speed_count
        else:
            speed = None

        states[camera_id] = SegmentTrafficState(
            local_inflow_vph=round(volume, 1),
            local_speed_kmh=round(speed, 1) if speed is not None else None,
            inflow_source="traffic_flow",
            speed_source="traffic_flow" if speed is not None else "missing",
            confidence="high" if speed is not None else "medium",
        )

    return states


def build_node_inflows(
    sensor_readings: list[SensorReading],
    camera_coords: dict[str, tuple[float, float]] | None = None,
    sensor_coords: dict[int, tuple[float, float]] | None = None,
    route_points: list[tuple[float, float]] | None = None,
    corridor_length_km: float = E4_NORTHBOUND_CORRIDOR_LENGTH_KM,
) -> dict[str, float]:
    """Compatibility wrapper returning only per-camera inflow volumes."""
    traffic_states = build_node_traffic_states(
        sensor_readings,
        camera_coords=camera_coords,
        sensor_coords=sensor_coords,
        route_points=route_points,
        corridor_length_km=corridor_length_km,
    )
    return {
        camera_id: state.local_inflow_vph
        for camera_id, state in traffic_states.items()
        if state.local_inflow_vph is not None
    }


# ---------------------------------------------------------------------------
# Concurrent data fetchers
# ---------------------------------------------------------------------------

def _draw_annotated_frame(
    frame: np.ndarray,
    engine: VisionEngine,
    camera_id: str,
    anomaly_reason: str | None,
    now: datetime,
) -> np.ndarray:
    """Draw YOLO bounding boxes on a copy of the frame for anomaly debugging."""
    annotated = frame.copy()
    try:
        results = engine.model.predict(
            source=frame, conf=engine._confidence, imgsz=640,
            verbose=False, save=False,
        )
        if results and len(results) > 0:
            result = results[0]
            if result.boxes is not None and len(result.boxes) > 0:
                boxes_xyxy = result.boxes.xyxy.cpu().numpy()
                confs = result.boxes.conf.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy().astype(int)
                for i in range(len(boxes_xyxy)):
                    x1, y1, x2, y2 = [int(v) for v in boxes_xyxy[i]]
                    cls_id = int(class_ids[i])
                    label = engine.model.names.get(cls_id, str(cls_id))
                    conf = float(confs[i])
                    # Color by type: vehicle=green, other=gray
                    color = (0, 255, 0) if cls_id in {2, 3, 5, 7} else (128, 128, 128)
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(
                        annotated, f"{label} {conf:.2f}",
                        (x1, max(y1 - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1,
                    )
        # Stamp anomaly reason on bottom
        h, w = annotated.shape[:2]
        cv2.rectangle(annotated, (0, h - 28), (w, h), (0, 0, 0), -1)
        cv2.putText(
            annotated,
            f"ANOMALY: {anomaly_reason or 'unknown'} | {camera_id} | {now.strftime('%H:%M:%S')}",
            (8, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1,
        )
    except Exception as e:
        logger.warning("Could not annotate frame for %s: %s", camera_id, e)
    return annotated


def _save_annotated_image(
    annotated: np.ndarray,
    camera_id: str,
    now: datetime,
    base_dir: str = ".",
) -> str | None:
    """Save an annotated anomaly frame to storage/anomalies/<date>/."""
    try:
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H-%M-%S")
        safe_id = camera_id.replace("/", "_")
        out_dir = os.path.join(base_dir, "storage", "anomalies", date_str)
        os.makedirs(out_dir, exist_ok=True)
        filename = f"{safe_id}_{time_str}_annotated.jpg"
        path = os.path.join(out_dir, filename)
        cv2.imwrite(path, annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
        logger.debug("Saved annotated anomaly image: %s", path)
        return path
    except Exception as e:
        logger.error("Failed to save annotated image: %s", e)
        return None


def _aggregate_multi_roi_capacity(
    multi_state: MultiSegmentCapacity,
    camera_meta: CameraMetadata,
) -> tuple[CapacityState, dict[str, dict]]:
    """Collapse per-ROI vision output into the camera-level state."""
    segment_directions = {
        id(seg): _traffic_direction_from_road_id(seg.road_id)
        for seg in multi_state.segments
    }
    northbound_segments = [
        seg
        for seg in multi_state.segments
        if segment_directions[id(seg)] == "northbound"
    ]
    has_directional_segments = any(
        direction in {"northbound", "southbound"}
        for direction in segment_directions.values()
    )
    if northbound_segments:
        physics_segments = northbound_segments
        traffic_direction = "northbound"
        road_id = northbound_segments[0].road_id
    elif has_directional_segments:
        # The current physics/VMS corridor is northbound E4. Keep southbound
        # ROI details in diagnostics, but do not promote them to a northbound
        # bottleneck state.
        physics_segments = []
        traffic_direction = "northbound"
        road_id = None
    else:
        physics_segments = multi_state.segments
        traffic_direction = None
        road_id = None

    total_vehicles = sum(s.vehicle_count for s in physics_segments)
    total_capacity = sum(s.capacity_vph for s in physics_segments)
    total_lanes = sum(s.num_lanes for s in physics_segments) or camera_meta.num_lanes
    any_anomaly = any(s.is_anomaly for s in physics_segments)
    anomaly_reasons = [
        s.anomaly_reason
        for s in physics_segments
        if s.is_anomaly and s.anomaly_reason
    ]
    max_density = max(
        (s.observed_density_veh_km_lane for s in physics_segments),
        default=0.0,
    )
    roi_length_confidence = _weakest_roi_length_confidence(physics_segments)

    state = CapacityState(
        timestamp=multi_state.timestamp,
        camera_id=multi_state.camera_id,
        vehicle_count=total_vehicles,
        blocked_lanes=0,
        total_lanes=total_lanes,
        estimated_capacity_vph=round(total_capacity, 1),
        observed_density_veh_km_lane=round(max_density, 2),
        road_id=road_id,
        traffic_direction=traffic_direction,
        is_anomaly=any_anomaly,
        anomaly_reason="; ".join(anomaly_reasons) if anomaly_reasons else None,
        confidence=round(
            float(np.mean([s.confidence for s in physics_segments]))
            if physics_segments else 0.0,
            3,
        ),
        roi_length_confidence=roi_length_confidence,
    )
    road_segments_data = {
        seg.road_id: {
            "direction": seg.direction,
            "count": seg.vehicle_count,
            "capacity_vph": seg.capacity_vph,
            "density_veh_km_lane": seg.observed_density_veh_km_lane,
            "roi_length_confidence": seg.roi_length_confidence,
            "traffic_direction": segment_directions[id(seg)],
        }
        for seg in multi_state.segments
    }
    return state, road_segments_data


def _northbound_detections_for_persistence(
    multi_state: MultiSegmentCapacity,
) -> list[dict]:
    """Return detections from the northbound physics corridor only."""
    detections: list[dict] = []
    has_directional_roads = False
    for road_id, road_detections in multi_state.detections_by_road_id.items():
        direction = _traffic_direction_from_road_id(road_id)
        if direction in {"northbound", "southbound"}:
            has_directional_roads = True
        if direction == "northbound":
            detections.extend(road_detections)
    if detections or has_directional_roads:
        return detections

    for road_detections in multi_state.detections_by_road_id.values():
        detections.extend(road_detections)
    return detections


def _apply_stopped_vehicle_detection(
    *,
    vision_records: list[dict],
    capacity_states: list[CapacityState],
    node_traffic_states: dict[str, SegmentTrafficState],
    now: datetime,
    critical_density_veh_km_lane: float,
) -> int:
    """Promote persistent vehicle boxes to stopped-vehicle anomaly states."""
    tracker = _get_track_persistence()
    state_by_camera = {state.camera_id: state for state in capacity_states}
    applied = 0

    for record in vision_records:
        camera_id = record.get("camera_id")
        if not isinstance(camera_id, str):
            continue

        if record.get("status") != "ok":
            tracker.mark_camera_missed(camera_id)
            record.pop("_vehicle_detections", None)
            continue

        detections = record.pop("_vehicle_detections", [])
        if not isinstance(detections, list):
            detections = []

        node_state = node_traffic_states.get(camera_id)
        local_speed = node_state.local_speed_kmh if node_state else None
        event = tracker.update(
            camera_id,
            detections,
            timestamp=now,
            local_speed_kmh=local_speed,
        )
        if event is None:
            continue

        state = state_by_camera.get(camera_id)
        if state is None:
            continue

        state.is_anomaly = True
        state.anomaly_reason = "vehicle_stopped"
        state.blocked_lanes = max(state.blocked_lanes, 1)
        state.observed_density_veh_km_lane = round(
            max(state.observed_density_veh_km_lane, critical_density_veh_km_lane + 1.0),
            2,
        )
        state.confidence = round(max(state.confidence, event.confidence), 3)
        record["is_anomaly"] = state.is_anomaly
        record["anomaly_reason"] = state.anomaly_reason
        record["confidence"] = state.confidence
        record["stopped_vehicle"] = event.as_record()
        record_anomaly(
            DATA_DIR,
            timestamp=now,
            camera_id=camera_id,
            camera_name=str(record.get("camera_name") or camera_id),
            anomaly_reason=state.anomaly_reason,
            confidence=state.confidence,
            vehicle_count=state.vehicle_count,
            capacity_vph=state.estimated_capacity_vph,
            image_path=record.get("annotated_path"),
        )
        applied += 1

    return applied


def _weakest_roi_length_confidence(segments: list) -> str | None:
    """Return the weakest ROI length confidence among selected physics segments."""
    if not segments:
        return None

    rank = {
        "high": 3,
        "surveyed": 3,
        "medium": 2,
        "estimated": 2,
        "low": 1,
        "unknown": 1,
    }
    weakest_value: str | None = None
    weakest_rank = 4
    for segment in segments:
        value = getattr(segment, "roi_length_confidence", None)
        normalized = (value or "unknown").lower()
        current_rank = rank.get(normalized, 1)
        if current_rank < weakest_rank:
            weakest_rank = current_rank
            weakest_value = value or "unknown"
    return weakest_value


def _traffic_direction_from_road_id(road_id: str) -> str | None:
    """Best-effort normalized traffic direction from ROI road identifiers."""
    normalized = road_id.lower().replace("-", "_")
    parts = [part for chunk in normalized.split("_") for part in chunk.split()]
    if "northbound" in parts or "nb" in parts:
        return "northbound"
    if "southbound" in parts or "sb" in parts:
        return "southbound"
    return None


def _derive_capacity_from_fused_state(
    state: CapacityState,
    *,
    local_speed_kmh: float | None,
    fallback_speed_kmh: float | None,
    critical_density_veh_km_lane: float = K_CRITICAL_VEH_KM_LANE,
    capacity_factor: float = 1.0,
) -> None:
    """Recompute camera-level capacity/anomaly after smoothing and speed fusion."""
    if (
        state.estimated_capacity_vph == 0.0
        and state.observed_density_veh_km_lane == 0.0
        and state.vehicle_count == 0
        and state.confidence == 0.0
    ):
        return

    if local_speed_kmh is not None:
        speed = local_speed_kmh
    elif fallback_speed_kmh is not None:
        speed = fallback_speed_kmh
    else:
        speed = FREE_FLOW_SPEED_KMH
    lanes = max(state.total_lanes, 1)
    max_capacity = Q_CAP_VPH_PER_LANE * lanes * min(capacity_factor, 1.0)
    density = state.observed_density_veh_km_lane

    if density > critical_density_veh_km_lane:
        state.estimated_capacity_vph = round(
            min(density * lanes * speed, max_capacity),
            1,
        )
        if not state.is_anomaly or state.anomaly_reason == "density_exceeds_k_critical":
            state.is_anomaly = True
            state.anomaly_reason = "density_exceeds_k_critical"
    else:
        state.estimated_capacity_vph = round(max_capacity, 1)
        if state.anomaly_reason == "density_exceeds_k_critical":
            state.is_anomaly = False
            state.anomaly_reason = None

    if state.is_anomaly and state.blocked_lanes > 0:
        open_lanes = max(lanes - state.blocked_lanes, 0)
        lane_fraction = open_lanes / lanes
        state.estimated_capacity_vph = round(
            state.estimated_capacity_vph * lane_fraction,
            1,
        )


def _apply_fused_capacity_derivation(
    capacity_states: list[CapacityState],
    node_traffic_states: dict[str, SegmentTrafficState],
    aggregate_sensor: SensorReading | None,
    fallback_speed_kmh: float,
    critical_density_veh_km_lane: float = K_CRITICAL_VEH_KM_LANE,
    capacity_factor: float = 1.0,
) -> None:
    """Apply post-smoothing capacity derivation using local traffic speeds."""
    aggregate_speed = aggregate_sensor.average_speed_kmh if aggregate_sensor else None
    for state in capacity_states:
        node_state = node_traffic_states.get(state.camera_id)
        local_speed = node_state.local_speed_kmh if node_state else None
        _derive_capacity_from_fused_state(
            state,
            local_speed_kmh=local_speed,
            fallback_speed_kmh=(
                aggregate_speed
                if aggregate_speed is not None
                else fallback_speed_kmh
            ),
            critical_density_veh_km_lane=critical_density_veh_km_lane,
            capacity_factor=capacity_factor,
        )


def apply_situation_capacity_impacts(
    capacity_states: list[CapacityState],
    situation_deviations: list[SituationDeviation],
    *,
    now: datetime,
    camera_chainage_map: dict[str, float],
    critical_density_veh_km_lane: float,
    default_total_lanes: int = 2,
) -> int:
    """Merge authoritative Situation capacity impacts into camera states."""
    if not situation_deviations or not camera_chainage_map:
        return 0

    state_by_camera = {state.camera_id: state for state in capacity_states}
    strongest_by_camera: dict[str, SituationDeviation] = {}

    for deviation in situation_deviations:
        camera_id = deviation.nearest_camera_id
        if camera_id is None and deviation.chainage_km is not None:
            nearest = find_nearest_by_chainage(
                deviation.chainage_km,
                camera_chainage_map,
            )
            camera_id = str(nearest) if nearest is not None else None
            deviation.nearest_camera_id = camera_id
        if camera_id is None:
            continue

        current = strongest_by_camera.get(camera_id)
        if current is None or deviation.capacity_factor < current.capacity_factor:
            strongest_by_camera[camera_id] = deviation

    for camera_id, deviation in strongest_by_camera.items():
        state = state_by_camera.get(camera_id)
        if state is None:
            chainage = camera_chainage_map.get(camera_id)
            coords = CAMERA_COORDS.get(camera_id)
            lanes = default_total_lanes
            capacity = Q_CAP_VPH_PER_LANE * lanes * deviation.capacity_factor
            state = CapacityState(
                timestamp=now,
                camera_id=camera_id,
                vehicle_count=0,
                blocked_lanes=_blocked_lanes_from_deviation(deviation, lanes),
                total_lanes=lanes,
                estimated_capacity_vph=round(capacity, 1),
                observed_density_veh_km_lane=round(critical_density_veh_km_lane + 1.0, 2),
                road_id="E4_Northbound",
                traffic_direction="northbound",
                is_anomaly=True,
                anomaly_reason=f"situation_confirmed_{deviation.deviation_type}",
                confidence=0.95,
                situation_confirmed=True,
                situation_ids=[deviation.deviation_id],
                situation_types=[deviation.deviation_type],
            )
            capacity_states.append(state)
            state_by_camera[camera_id] = state
            if chainage is not None and coords is None:
                logger.debug(
                    "Synthetic situation state for %s at chainage %.2f km",
                    camera_id,
                    chainage,
                )
            continue

        lanes = max(state.total_lanes, default_total_lanes, 1)
        state.total_lanes = lanes
        state.blocked_lanes = max(
            state.blocked_lanes,
            _blocked_lanes_from_deviation(deviation, lanes),
        )
        situation_capacity = Q_CAP_VPH_PER_LANE * lanes * deviation.capacity_factor
        state.estimated_capacity_vph = round(
            min(state.estimated_capacity_vph, situation_capacity),
            1,
        )
        state.observed_density_veh_km_lane = round(
            max(
                state.observed_density_veh_km_lane,
                critical_density_veh_km_lane + 1.0,
            ),
            2,
        )
        state.is_anomaly = True
        reason = f"situation_confirmed_{deviation.deviation_type}"
        if not state.anomaly_reason:
            state.anomaly_reason = reason
        elif reason not in state.anomaly_reason:
            state.anomaly_reason = f"{state.anomaly_reason}+{reason}"
        state.confidence = max(state.confidence, 0.95)
        state.situation_confirmed = True
        if deviation.deviation_id not in state.situation_ids:
            state.situation_ids.append(deviation.deviation_id)
        if deviation.deviation_type not in state.situation_types:
            state.situation_types.append(deviation.deviation_type)

    return len(strongest_by_camera)


def _blocked_lanes_from_deviation(
    deviation: SituationDeviation,
    total_lanes: int,
) -> int:
    if deviation.number_of_lanes_restricted is not None:
        return min(max(deviation.number_of_lanes_restricted, 0), total_lanes)
    if deviation.deviation_type == "accident":
        return min(1, total_lanes)
    return 0


def _camera_worker_count(camera_count: int) -> int:
    """Bound per-camera fetch/inference concurrency."""
    if camera_count <= 0:
        return 0
    return min(camera_count, 8)


def _camera_failure_record(
    *,
    now: datetime,
    camera_id: str,
    camera_name: str | None = None,
    status: str,
    duration_ms: int,
    error: str | None = None,
) -> dict:
    record = {
        "type": "vision_result",
        "timestamp": now.isoformat(),
        "camera_id": camera_id,
        "status": status,
        "duration_ms": duration_ms,
    }
    if camera_name:
        record["camera_name"] = camera_name
    if error:
        record["error"] = error
    return record


def _process_camera(
    cam: dict,
    now: datetime,
    engine_factory: Callable[[], VisionEngine],
    retention: RetentionPolicy,
    roi_mapper: ROIMapper,
) -> tuple[dict | None, CapacityState | None]:
    camera_started = time.monotonic()
    cam_id = cam.get("Id", "unknown")
    cam_name = cam.get("Name", cam_id)
    photo_url = cam.get("PhotoUrl", "")
    if not photo_url:
        return None, None

    try:
        if cam.get("HasFullSizePhoto"):
            photo_url = photo_url + "?type=fullsize"

        raw_bytes = fetch_image_bytes(photo_url)
        duration_ms = int((time.monotonic() - camera_started) * 1000)
        if raw_bytes is None:
            logger.warning("📷 %s fetch failed after %sms", cam_name, duration_ms)
            return _camera_failure_record(
                now=now,
                camera_id=cam_id,
                camera_name=cam_name,
                status="fetch_failed",
                duration_ms=duration_ms,
            ), None

        frame = decode_frame(raw_bytes)
        duration_ms = int((time.monotonic() - camera_started) * 1000)
        if frame is None:
            logger.warning("📷 %s decode failed after %sms", cam_name, duration_ms)
            return _camera_failure_record(
                now=now,
                camera_id=cam_id,
                camera_name=cam_name,
                status="decode_failed",
                duration_ms=duration_ms,
            ), None

        coords = CAMERA_COORDS.get(cam_id, (0.0, 0.0))
        meta = CameraMetadata(
            camera_id=cam_id, name=cam_name, lat=coords[0], lng=coords[1],
        )
        engine = engine_factory()

        road_segments_data = None
        if roi_mapper.has_rois(cam_id):
            multi_state = engine.analyze_multi_roi(frame, meta, roi_mapper)
            state, road_segments_data = _aggregate_multi_roi_capacity(
                multi_state, meta,
            )
            vehicle_detections = _northbound_detections_for_persistence(multi_state)
        else:
            state = engine.analyze_array(frame, meta)
            vehicle_detections = list(getattr(engine, "last_vehicle_detections", []))

        with _retention_lock:
            retained_path = retention.maybe_retain(raw_bytes, cam_id, now, state)

        annotated_path: str | None = None
        if state.is_anomaly:
            annotated_frame = _draw_annotated_frame(
                frame, engine, cam_id, state.anomaly_reason, now,
            )
            annotated_path = _save_annotated_image(
                annotated_frame, cam_id, now,
            )
            record_anomaly(
                DATA_DIR,
                timestamp=now,
                camera_id=cam_id,
                camera_name=cam_name,
                anomaly_reason=state.anomaly_reason,
                confidence=state.confidence,
                vehicle_count=state.vehicle_count,
                capacity_vph=state.estimated_capacity_vph,
                image_path=annotated_path,
            )

        duration_ms = int((time.monotonic() - camera_started) * 1000)
        record = {
            "type": "vision_result",
            "timestamp": now.isoformat(),
            "camera_id": cam_id,
            "camera_name": cam_name,
            "status": "ok",
            "vehicle_count": state.vehicle_count,
            "capacity_vph": state.estimated_capacity_vph,
            "is_anomaly": state.is_anomaly,
            "anomaly_reason": state.anomaly_reason,
            "confidence": state.confidence,
            "retained_path": retained_path,
            "annotated_path": annotated_path,
            "road_segments": road_segments_data,
            "_vehicle_detections": vehicle_detections,
            "duration_ms": duration_ms,
        }

        anomaly_tag = f" 🚨 {state.anomaly_reason}" if state.is_anomaly else ""
        logger.info(
            f"📷 {cam_name} — {state.vehicle_count} vehicles, "
            f"{state.estimated_capacity_vph:.0f} VPH in {duration_ms}ms{anomaly_tag}"
        )
        return record, state
    except Exception as e:
        duration_ms = int((time.monotonic() - camera_started) * 1000)
        logger.error(
            "Camera %s failed after %sms: %s",
            cam_id,
            duration_ms,
            e,
            exc_info=True,
        )
        return _camera_failure_record(
            now=now,
            camera_id=cam_id,
            camera_name=cam_name,
            status="error",
            duration_ms=duration_ms,
            error=str(e),
        ), None


def fetch_cameras(camera_ids: list[str], now: datetime) -> tuple[list[dict], list[CapacityState]]:
    """Fetch camera images into RAM, run YOLO, apply retention, return metadata."""
    if not camera_ids:
        return [], []

    id_filter = "\n".join(f'<EQ name="Id" value="{cid}" />' for cid in camera_ids)
    xml_query = f"""
    <REQUEST>
        <LOGIN authenticationkey="{API_KEY}" />
        <QUERY objecttype="Camera" schemaversion="1">
            <FILTER>
                <OR>
                    {id_filter}
                </OR>
            </FILTER>
        </QUERY>
    </REQUEST>
    """
    data = api_request(xml_query)
    if not data:
        return [], []

    results = data.get("RESPONSE", {}).get("RESULT", [])
    cameras = results[0].get("Camera", []) if results else []
    if not cameras:
        logger.info("📷 Camera query returned no cameras")
        return [], []

    engine_factory = _get_camera_worker_vision_engine
    retention = _get_retention_policy()
    roi_mapper = _get_roi_mapper()

    vision_records: list[dict] = []
    capacity_states: list[CapacityState] = []

    fetch_started = time.monotonic()
    max_workers = _camera_worker_count(len(cameras))
    logger.info(
        "📷 Processing %s cameras with %s workers",
        len(cameras),
        max_workers,
    )

    ordered_results: list[tuple[dict | None, CapacityState | None] | None] = [
        None
    ] * len(cameras)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _process_camera,
                cam,
                now,
                engine_factory,
                retention,
                roi_mapper,
            ): index
            for index, cam in enumerate(cameras)
        }
        for future in as_completed(futures):
            ordered_results[futures[future]] = future.result()

    for result in ordered_results:
        if result is None:
            continue
        record, state = result
        if record is not None:
            vision_records.append(record)
        if state is not None:
            capacity_states.append(state)

    fetch_duration_ms = int((time.monotonic() - fetch_started) * 1000)
    logger.info(
        "📷 Camera batch completed in %sms: %s ok, %s failed/skipped",
        fetch_duration_ms,
        sum(1 for r in vision_records if r.get("status") == "ok"),
        sum(1 for r in vision_records if r.get("status") != "ok"),
    )

    return vision_records, capacity_states


def fetch_sensor_data(now: datetime) -> list[SensorReading]:
    """Fetch TrafficFlow sensor data for E4 corridor stations.

    Queries the Trafikverket TrafficFlow API filtered to curated
    northbound SiteIds along the Hallunda → Kristineberg corridor
    (see ``config.SENSOR_SITE_IDS``).

    Returns one ``SensorReading`` per station (lanes aggregated).
    """
    from config import SENSOR_SITE_IDS

    if not SENSOR_SITE_IDS:
        logger.warning("No SENSOR_SITE_IDS configured — skipping sensor fetch")
        return []

    # Build OR filter for all SiteIds
    site_filters = "\n".join(
        f'                    <EQ name="SiteId" value="{sid}" />'
        for sid in SENSOR_SITE_IDS
    )

    xml_query = f"""
    <REQUEST>
        <LOGIN authenticationkey="{API_KEY}" />
        <QUERY objecttype="TrafficFlow" schemaversion="1" limit="500">
            <FILTER>
                <OR>
{site_filters}
                </OR>
            </FILTER>
            <INCLUDE>SiteId</INCLUDE>
            <INCLUDE>VehicleFlowRate</INCLUDE>
            <INCLUDE>AverageVehicleSpeed</INCLUDE>
            <INCLUDE>SpecificLane</INCLUDE>
            <INCLUDE>MeasurementTime</INCLUDE>
        </QUERY>
    </REQUEST>
    """
    data = api_request(xml_query)
    if not data:
        return []

    results = data.get("RESPONSE", {}).get("RESULT", [])
    flows = results[0].get("TrafficFlow", []) if results else []

    # Aggregate per SiteId: sum flows across lanes, mean speed
    site_data: dict[int, dict[str, list[float]]] = {}
    for flow in flows:
        try:
            sid = flow.get("SiteId")
            volume = flow.get("VehicleFlowRate", 0) or 0
            speed = flow.get("AverageVehicleSpeed", 0) or 0
            if sid is None or volume <= 0:
                continue
            if sid not in site_data:
                site_data[sid] = {"volumes": [], "speeds": []}
            site_data[sid]["volumes"].append(float(volume))
            site_data[sid]["speeds"].append(float(speed))
        except (ValueError, TypeError):
            continue

    # Produce one SensorReading per station (sum of lane flows, mean speed)
    readings: list[SensorReading] = []
    for sid, agg in site_data.items():
        total_flow = sum(agg["volumes"])
        mean_speed = sum(agg["speeds"]) / len(agg["speeds"]) if agg["speeds"] else 0
        readings.append(SensorReading(
            timestamp=now,
            site_id=sid,
            inflow_volume_vph=round(total_flow, 1),
            average_speed_kmh=round(mean_speed, 1),
        ))

    logger.info(
        f"🔢 Sensor data: {len(readings)} stations from {len(flows)} lane readings "
        f"({len(SENSOR_SITE_IDS)} SiteIds configured)"
    )
    return readings


def fetch_weather_data(now: datetime) -> list[dict]:
    """Fetch WeatherMeasurepoint observations near the monitored corridor."""
    xml_query = f"""
    <REQUEST>
        <LOGIN authenticationkey="{API_KEY}" />
        <QUERY objecttype="WeatherMeasurepoint" schemaversion="2">
            <FILTER />
        </QUERY>
    </REQUEST>
    """

    data = api_request(xml_query)
    if not data:
        return []

    results = data.get("RESPONSE", {}).get("RESULT", [])
    points = results[0].get("WeatherMeasurepoint", []) if results else []

    weather_records: list[dict] = []
    for point in points:
        coords = parse_point_wgs84(point.get("Geometry", {}).get("WGS84", ""))
        if not coords or not in_bbox(*coords):
            continue

        obs = point.get("Observation", {})
        air = obs.get("Air", {})
        surface = obs.get("Surface", {})
        wind_list = obs.get("Wind", [])
        wind = wind_list[0] if wind_list else {}
        weather = obs.get("Weather", {})
        agg5 = obs.get("Aggregated5minutes", {})

        weather_records.append({
            "type": "weather",
            "timestamp": now.isoformat(),
            "station_id": point.get("Id", ""),
            "station_name": point.get("Name", ""),
            "sample_time": obs.get("Sample", ""),
            "air_temp_c": _val(air, "Temperature"),
            "air_humidity_pct": _val(air, "RelativeHumidity"),
            "air_dewpoint_c": _val(air, "Dewpoint"),
            "visibility_m": _val(air, "VisibleDistance"),
            "wind_speed_ms": _val(wind, "Speed"),
            "wind_dir_deg": _val(wind, "Direction"),
            "surface_temp_c": _val(surface, "Temperature"),
            "precipitation": weather.get("Precipitation", None),
            "precip_rain_sum": _nested(
                agg5,
                "Precipitation",
                "RainSum",
                "Value",
            ),
            "precip_snow_water_eq": _nested(
                agg5,
                "Precipitation",
                "SnowSum",
                "WaterEquivalent",
                "Value",
            ),
        })

    logger.info("🌡  Weather data: %s corridor station(s)", len(weather_records))
    return weather_records


def fetch_road_conditions(now: datetime) -> list[dict]:
    """Fetch E4/E20 RoadCondition records for Stockholm county."""
    xml_query = f"""
    <REQUEST>
        <LOGIN authenticationkey="{API_KEY}" />
        <QUERY objecttype="RoadCondition" schemaversion="1.2">
            <FILTER>
                <EQ name="CountyNo" value="1" />
            </FILTER>
        </QUERY>
    </REQUEST>
    """

    data = api_request(xml_query)
    if not data:
        return []

    results = data.get("RESPONSE", {}).get("RESULT", [])
    conditions = results[0].get("RoadCondition", []) if results else []

    road_records: list[dict] = []
    for condition in conditions:
        road_number = condition.get("RoadNumber", "")
        if not _is_e4_road(road_number):
            continue

        geometry_wgs84 = _extract_wgs84(condition)
        position = parse_point_wgs84(geometry_wgs84 or "")
        chainage_km = project_e4_northbound_chainage(position) if position else None

        road_records.append({
            "type": "road_condition",
            "timestamp": now.isoformat(),
            "id": condition.get("Id", ""),
            "location": condition.get("LocationText", ""),
            "condition_text": condition.get("ConditionText", ""),
            "condition_info": condition.get("ConditionInfo", []),
            "condition_code": condition.get("ConditionCode"),
            "warning": condition.get("Warning", False),
            "road_number": road_number,
            "start_time": condition.get("StartTime", ""),
            "geometry_wgs84": geometry_wgs84,
            "lat": position[0] if position else None,
            "lng": position[1] if position else None,
            "chainage_km": chainage_km,
        })

    logger.info("🛣  Road conditions: %s E4/E20 record(s)", len(road_records))
    return road_records


def _extract_wgs84(record: dict) -> str | None:
    geometry = record.get("Geometry")
    if isinstance(geometry, dict):
        wgs84 = geometry.get("WGS84")
        return str(wgs84) if wgs84 else None
    wgs84 = record.get("Geometry.WGS84")
    return str(wgs84) if wgs84 else None


def _is_e4_road(road_number: str) -> bool:
    if not road_number:
        return False
    normalized = road_number.strip().upper()
    return normalized in ("E 4", "E4", "E 20", "E20", "E4/E20", "E 4/E 20")


def fetch_situation_deviations(now: datetime) -> list[SituationDeviation]:
    """Fetch accident/roadwork Situation deviations as capacity-impact inputs."""
    xml_query = f"""
    <REQUEST>
        <LOGIN authenticationkey="{API_KEY}" />
        <QUERY objecttype="Situation" schemaversion="1" limit="200">
            <FILTER>
                <EQ name="Deviation.CountyNo" value="1" />
            </FILTER>
            <INCLUDE>Deviation.Id</INCLUDE>
            <INCLUDE>Deviation.RoadNumber</INCLUDE>
            <INCLUDE>Deviation.MessageType</INCLUDE>
            <INCLUDE>Deviation.MessageCode</INCLUDE>
            <INCLUDE>Deviation.SeverityCode</INCLUDE>
            <INCLUDE>Deviation.NumberOfLanesRestricted</INCLUDE>
            <INCLUDE>Deviation.LocationDescriptor</INCLUDE>
            <INCLUDE>Deviation.Geometry.WGS84</INCLUDE>
            <INCLUDE>Deviation.StartTime</INCLUDE>
            <INCLUDE>Deviation.CreationTime</INCLUDE>
        </QUERY>
    </REQUEST>
    """
    data = api_request(xml_query)
    if not data:
        return []

    results = data.get("RESPONSE", {}).get("RESULT", [])
    situations = results[0].get("Situation", []) if results else []

    deviations: list[SituationDeviation] = []
    for situation in situations:
        for dev in situation.get("Deviation", []):
            road = dev.get("RoadNumber", "")
            if not _is_e4_road(road):
                continue

            deviation_type = _classify_situation_deviation(dev)
            if deviation_type is None:
                continue

            geometry_wgs84 = get_deviation_wgs84(dev)
            position = parse_point_wgs84(geometry_wgs84 or "")
            chainage_km = (
                project_e4_northbound_chainage(position)
                if position and str(road).upper().replace(" ", "") in {"E4", "E4/E20", "E20"}
                else None
            )
            nearest_camera_id = (
                _find_nearest_camera(position[0], position[1])
                if position and chainage_km is not None
                else None
            )
            lanes_restricted = _parse_lanes_restricted(
                dev.get("NumberOfLanesRestricted")
            )

            deviations.append(SituationDeviation(
                timestamp=now,
                deviation_id=str(dev.get("Id", "")),
                deviation_type=deviation_type,
                message_type=_string_or_none(dev.get("MessageType")),
                message_code=_string_or_none(dev.get("MessageCode")),
                severity_code=_string_or_none(dev.get("SeverityCode")),
                number_of_lanes_restricted=lanes_restricted,
                road_number=_string_or_none(road),
                location=_string_or_none(dev.get("LocationDescriptor")),
                geometry_wgs84=geometry_wgs84,
                lat=position[0] if position else None,
                lng=position[1] if position else None,
                chainage_km=chainage_km,
                nearest_camera_id=nearest_camera_id,
                capacity_factor=_situation_capacity_factor(
                    deviation_type,
                    lanes_restricted,
                    dev.get("SeverityCode"),
                ),
                start_time=_string_or_none(dev.get("StartTime")),
                creation_time=_string_or_none(dev.get("CreationTime")),
            ))

    logger.info(
        "🚧 Situation deviations: %s accident/roadwork record(s)",
        len(deviations),
    )
    return deviations


def _classify_situation_deviation(dev: dict) -> str | None:
    text = " ".join(
        str(value or "")
        for value in (
            dev.get("MessageType"),
            dev.get("MessageCode"),
            dev.get("Id"),
        )
    ).lower()
    if any(token in text for token in ("olycka", "accident")):
        return "accident"
    if any(token in text for token in ("vägarbete", "vagarbete", "roadwork", "road work")):
        return "roadwork"
    return None


def _parse_lanes_restricted(value) -> int | None:
    if value in (None, ""):
        return None
    try:
        return max(int(value), 0)
    except (TypeError, ValueError):
        match = re.search(r"\d+", str(value))
        return int(match.group(0)) if match else None


def _string_or_none(value) -> str | None:
    if value in (None, ""):
        return None
    return str(value)


def _situation_capacity_factor(
    deviation_type: str,
    lanes_restricted: int | None,
    severity_code: str | None = None,
    total_lanes: int = 2,
) -> float:
    base = 0.45 if deviation_type == "accident" else 0.65
    severity = str(severity_code or "").lower()
    if any(token in severity for token in ("high", "stor", "severe", "major")):
        base = min(base, 0.35)
    if lanes_restricted is not None and lanes_restricted > 0:
        lane_factor = max(total_lanes - lanes_restricted, 0) / max(total_lanes, 1)
        base = min(base, lane_factor)
    return round(max(base, 0.25), 2)


def fetch_vms_status(now: datetime) -> list[VMSStatusSnapshot]:
    """Poll VMS-proxy ground truth from the Trafikverket Situation API.

    The public API does NOT expose live physical VMS panel state.
    Instead, we poll ``Situation.Deviation`` records filtered by
    ``MessageCode = 'Hastighetsbegränsning gäller'`` (SPEEDMANAGEMENTID).
    These represent temporary speed advisories set by human operators —
    the closest available proxy for "when did the operator act?".

    Each polled deviation becomes a ``VMSStatusSnapshot`` with
    ``source='situation_api_proxy'``.  In production (post-B2G sale),
    this will be replaced by a direct TMC feed.
    """
    xml_query = f"""
    <REQUEST>
        <LOGIN authenticationkey="{API_KEY}" />
        <QUERY objecttype="Situation" schemaversion="1" limit="100">
            <FILTER>
                <AND>
                    <EQ name="Deviation.CountyNo" value="1" />
                    <EQ name="Deviation.MessageCode"
                        value="Hastighetsbegränsning gäller" />
                </AND>
            </FILTER>
            <INCLUDE>Deviation.Id</INCLUDE>
            <INCLUDE>Deviation.RoadNumber</INCLUDE>
            <INCLUDE>Deviation.TemporaryLimit</INCLUDE>
            <INCLUDE>Deviation.LocationDescriptor</INCLUDE>
            <INCLUDE>Deviation.Geometry.WGS84</INCLUDE>
            <INCLUDE>Deviation.StartTime</INCLUDE>
            <INCLUDE>Deviation.CreationTime</INCLUDE>
        </QUERY>
    </REQUEST>
    """
    data = api_request(xml_query)
    statuses: list[VMSStatusSnapshot] = []

    if data:
        results = data.get("RESPONSE", {}).get("RESULT", [])
        situations = results[0].get("Situation", []) if results else []

        for sit in situations:
            for dev in sit.get("Deviation", []):
                dev_id = dev.get("Id", "")
                # Only process SPEEDMANAGEMENT deviations
                if "SPEEDMANAGEMENT" not in dev_id:
                    continue

                temp_limit = dev.get("TemporaryLimit", "") or ""
                road = dev.get("RoadNumber", "")
                location = dev.get("LocationDescriptor", "")
                geometry_wgs84 = get_deviation_wgs84(dev)
                position = parse_point_wgs84(geometry_wgs84 or "")
                chainage_km = (
                    project_e4_northbound_chainage(position)
                    if position and road == "E4"
                    else None
                )

                speed_limit = _parse_speed_limit(temp_limit)
                display_msg = temp_limit if temp_limit else None

                statuses.append(VMSStatusSnapshot(
                    timestamp=now,
                    vms_id=dev_id,
                    vms_name=f"{road} — {location[:60]}" if location else road,
                    is_active=bool(temp_limit),
                    displayed_message=display_msg,
                    speed_limit=speed_limit,
                    road_number=road or None,
                    geometry_wgs84=geometry_wgs84,
                    lat=position[0] if position else None,
                    lng=position[1] if position else None,
                    chainage_km=chainage_km,
                ))

    # Also include our configured gantries that have no matching
    # Situation deviation (mark as inactive)
    active_roads = {s.vms_name.split(" —")[0].strip() for s in statuses}
    orchestrator = _get_vms_orchestrator()
    for gantry in orchestrator.gantries:
        # Check if any speed management already covers this gantry's road
        if gantry.road not in active_roads:
            statuses.append(VMSStatusSnapshot(
                timestamp=now,
                vms_id=gantry.vms_id,
                vms_name=gantry.name,
                is_active=False,
                displayed_message=None,
                speed_limit=None,
                road_number=gantry.road,
                lat=gantry.lat,
                lng=gantry.lng,
                chainage_km=gantry.chainage_km,
            ))

    active_count = sum(1 for s in statuses if s.is_active)
    logger.info(
        f"🚦 VMS proxy: {len(statuses)} entries polled "
        f"({active_count} active speed advisories)"
    )
    return statuses


# ---------------------------------------------------------------------------
# Phase 1d: Travel time fetch
# ---------------------------------------------------------------------------


def fetch_travel_times(now: datetime) -> list[TravelTimeReading]:
    """Fetch measured corridor travel times from TravelTimeRoute API.

    Queries Trafikverket's Bluetooth/ANPR-based travel time measurements
    for E4/E20 route segments in Stockholm county.  Returns one
    ``TravelTimeReading`` per segment with actual vs. free-flow times.
    """
    route_ids_filter = "".join(
        f'<EQ name="Id" value="{rid}" />'
        for rid in E4_TRAVEL_TIME_ROUTE_IDS
    )

    xml_query = f"""
    <REQUEST>
        <LOGIN authenticationkey="{API_KEY}" />
        <QUERY objecttype="TravelTimeRoute" schemaversion="1.5" limit="50">
            <FILTER>
                <AND>
                    <EQ name="CountyNo" value="1" />
                    <OR>
                        {route_ids_filter}
                    </OR>
                </AND>
            </FILTER>
            <INCLUDE>Id</INCLUDE>
            <INCLUDE>Name</INCLUDE>
            <INCLUDE>TravelTime</INCLUDE>
            <INCLUDE>FreeFlowTravelTime</INCLUDE>
            <INCLUDE>Speed</INCLUDE>
            <INCLUDE>Length</INCLUDE>
            <INCLUDE>TrafficStatus</INCLUDE>
            <INCLUDE>MeasureTime</INCLUDE>
        </QUERY>
    </REQUEST>
    """

    data = api_request(xml_query)
    readings: list[TravelTimeReading] = []

    if data:
        results = data.get("RESPONSE", {}).get("RESULT", [])
        routes = results[0].get("TravelTimeRoute", []) if results else []

        for r in routes:
            try:
                tt = float(r.get("TravelTime", 0) or 0)
                ff = float(r.get("FreeFlowTravelTime", 0) or 0)
                readings.append(TravelTimeReading(
                    timestamp=now,
                    route_id=str(r.get("Id", "")),
                    name=r.get("Name", "Unknown"),
                    travel_time_seconds=tt,
                    free_flow_seconds=ff,
                    speed_kmh=float(r.get("Speed", 0) or 0),
                    length_meters=float(r.get("Length", 0) or 0),
                    traffic_status=r.get("TrafficStatus", "unknown"),
                    delay_seconds=round(tt - ff, 2),
                ))
            except (ValueError, TypeError) as e:
                logger.debug(f"Skipping malformed TravelTimeRoute: {e}")

    # Log summary
    total_delay = sum(t.delay_seconds for t in readings)
    slow_count = sum(1 for t in readings if t.traffic_status != "freeflow")
    logger.info(
        f"🕐 Travel times: {len(readings)} routes fetched "
        f"(corridor delay: {total_delay:+.0f}s, "
        f"{slow_count} non-freeflow)"
    )
    return readings


def _parse_speed_limit(text: str) -> int | None:
    """Extract speed limit integer from Swedish text.

    Examples:
        'Hastighet: 70km/h' → 70
        'Rekommenderad hastighet: 50km/h' → 50
        '' → None
    """
    match = re.search(r"(\d+)\s*km/h", text, re.IGNORECASE)
    return int(match.group(1)) if match else None


# ---------------------------------------------------------------------------
# Sensor anomaly detection
# ---------------------------------------------------------------------------


def detect_sensor_anomalies(
    sensor_readings: list[SensorReading],
    now: datetime,
) -> list[SensorAnomaly]:
    """Check sensor speeds against road speed limits.

    This closes the critical detection gap where sensor-measured speed
    drops (e.g. 35 km/h on a 70 km/h road) were invisible to the system
    because anomaly detection was camera-only.

    Parameters
    ----------
    sensor_readings:
        Latest sensor data from the Trafikverket API.
    now:
        Current timestamp for the anomaly records.

    Returns
    -------
    list[SensorAnomaly]
        Anomalies for stations where speed is below threshold.
    """
    anomalies: list[SensorAnomaly] = []

    for reading in sensor_readings:
        if reading.site_id is None:
            continue

        # Look up the road speed limit for this station
        road_limit = SENSOR_ROAD_SPEED_LIMITS.get(
            reading.site_id, DEFAULT_ROAD_SPEED_LIMIT
        )

        # Skip if speed is above the warning threshold
        if reading.average_speed_kmh >= road_limit * SENSOR_SPEED_DROP_RATIO:
            continue

        # Calculate speed ratio
        speed_ratio = (
            reading.average_speed_kmh / road_limit if road_limit > 0 else 0.0
        )

        # Classify severity
        severity = (
            "severe"
            if reading.average_speed_kmh < road_limit * SENSOR_SEVERE_DROP_RATIO
            else "warning"
        )

        # Get station coordinates
        coords = SENSOR_COORDS.get(reading.site_id, (0.0, 0.0))

        # Find nearest camera for cross-referencing
        nearest_cam = _find_nearest_camera(coords[0], coords[1]) if coords[0] else None

        anomaly = SensorAnomaly(
            timestamp=now,
            site_id=reading.site_id,
            measured_speed_kmh=reading.average_speed_kmh,
            road_speed_limit_kmh=road_limit,
            speed_ratio=round(speed_ratio, 3),
            volume_vph=reading.inflow_volume_vph,
            severity=severity,
            nearest_camera_id=nearest_cam,
            lat=coords[0],
            lng=coords[1],
        )
        anomalies.append(anomaly)

        # Log with appropriate severity
        icon = "🚨" if severity == "severe" else "⚠️"
        logger.warning(
            f"{icon} Sensor anomaly: station {reading.site_id} "
            f"at {reading.average_speed_kmh:.0f} km/h "
            f"(limit {road_limit} km/h, {speed_ratio*100:.0f}%) "
            f"→ {severity}"
        )

        # Record to anomaly store
        record_anomaly(
            DATA_DIR,
            timestamp=now,
            camera_id=f"sensor_{reading.site_id}",
            camera_name=f"Sensor {reading.site_id}",
            anomaly_reason=f"sensor_speed_{severity}",
            confidence=1.0 - speed_ratio,  # Higher confidence when bigger drop
            vehicle_count=0,
            capacity_vph=reading.inflow_volume_vph,
            image_path=None,
        )

    return anomalies


def _find_nearest_camera(
    lat: float,
    lng: float | None = None,
    camera_coords: dict[str, tuple[float, float]] | None = None,
    route_points: list[tuple[float, float]] | None = None,
    corridor_length_km: float = E4_NORTHBOUND_CORRIDOR_LENGTH_KM,
) -> str | None:
    """Find the camera ID nearest to a route position."""
    coords = camera_coords if camera_coords is not None else CAMERA_COORDS
    route = route_points if route_points is not None else E4_NORTHBOUND_ROUTE_POINTS
    if not coords:
        return None
    if lng is not None:
        camera_chainages = build_camera_chainage_map(
            coords,
            route_points=route,
            corridor_length_km=corridor_length_km,
        )
        target_chainages = build_route_chainage_map(
            {"target": (lat, lng)},
            route,
            corridor_length_km,
        )
        target_chainage = target_chainages.get("target")
        nearest = (
            find_nearest_by_chainage(target_chainage, camera_chainages)
            if target_chainage is not None
            else None
        )
        if nearest is not None:
            return str(nearest)

    return min(
        coords,
        key=lambda cid: abs(coords[cid][0] - lat),
    )


# ---------------------------------------------------------------------------
# Tick orchestration
# ---------------------------------------------------------------------------

_tick_count = 0
_start_time: str | None = None
_camera_chainage_map: dict[str, float] | None = None


def tick_once(camera_ids: list[str]) -> TickResult:
    """Execute one discrete 60-second tick cycle.

    This is the core orchestration function. It:
    1. Concurrently fetches cameras, sensors, and VMS statuses
    2. Runs physics engine on the results
    3. Runs VMS orchestrator on the physics output
    4. Persists everything to JSONL
    """
    global _tick_count, _start_time, _camera_chainage_map

    tick_started = time.monotonic()
    _tick_count += 1
    now = datetime.now()
    if not _start_time:
        _start_time = now.isoformat()

    if _camera_chainage_map is None:
        _camera_chainage_map = _build_camera_chainage_map()

    logger.info(f"{'=' * 60}")
    logger.info(f"⏰ TICK #{_tick_count} — {now.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"{'=' * 60}")

    # ---- Phase 1: Concurrent data fetch ----
    vision_records: list[dict] = []
    capacity_states: list[CapacityState] = []
    sensor_readings: list[SensorReading] = []
    vms_statuses: list[VMSStatusSnapshot] = []
    travel_time_readings: list[TravelTimeReading] = []
    weather_records: list[dict] = []
    road_condition_records: list[dict] = []
    situation_deviations: list[SituationDeviation] = []

    fetch_started = time.monotonic()
    with ThreadPoolExecutor(max_workers=7) as executor:
        future_cameras = executor.submit(fetch_cameras, camera_ids, now)
        future_sensors = executor.submit(fetch_sensor_data, now)
        future_vms = executor.submit(fetch_vms_status, now)
        future_tt = executor.submit(fetch_travel_times, now)
        future_weather = executor.submit(fetch_weather_data, now)
        future_road_conditions = executor.submit(fetch_road_conditions, now)
        future_situations = executor.submit(fetch_situation_deviations, now)

        try:
            vision_records, capacity_states = future_cameras.result(timeout=55)
        except Exception as e:
            logger.error(f"Camera fetch failed: {e}", exc_info=True)

        try:
            sensor_readings = future_sensors.result(timeout=10)
        except Exception as e:
            logger.error(f"Sensor fetch failed: {e}", exc_info=True)

        try:
            vms_statuses = future_vms.result(timeout=10)
        except Exception as e:
            logger.error(f"VMS status fetch failed: {e}", exc_info=True)

        try:
            travel_time_readings = future_tt.result(timeout=10)
        except Exception as e:
            logger.error(f"Travel time fetch failed: {e}", exc_info=True)

        try:
            weather_records = future_weather.result(timeout=10)
        except Exception as e:
            logger.error(f"Weather fetch failed: {e}", exc_info=True)

        try:
            road_condition_records = future_road_conditions.result(timeout=10)
        except Exception as e:
            logger.error(f"Road condition fetch failed: {e}", exc_info=True)

        try:
            situation_deviations = future_situations.result(timeout=10)
        except Exception as e:
            logger.error(f"Situation deviation fetch failed: {e}", exc_info=True)
    fetch_duration_ms = int((time.monotonic() - fetch_started) * 1000)
    logger.info("⏱️  Data fetch phase completed in %sms", fetch_duration_ms)

    weather_adjustment = _get_weather_adapter().compute(
        weather_records=weather_records,
        road_condition_records=road_condition_records,
        now=now,
    )
    logger.info(
        "🌦  Surface adjustment: %s free_flow=%.2f capacity=%.2f confidence=%s (%s)",
        weather_adjustment.surface_state,
        weather_adjustment.free_flow_factor,
        weather_adjustment.capacity_factor,
        weather_adjustment.confidence,
        weather_adjustment.reason,
    )

    # ---- Phase 2: Apply temporal density smoothing (Expert Audit Fix 3) ----
    smoother = _get_density_smoother()
    for state in capacity_states:
        smoothed = smoother.update(
            state.camera_id, state.observed_density_veh_km_lane
        )
        state.observed_density_veh_km_lane = round(smoothed, 2)

    # ---- Phase 2b: Sensor-based anomaly detection ----
    sensor_anomalies = detect_sensor_anomalies(sensor_readings, now)
    if sensor_anomalies:
        logger.info(
            f"🚨 {len(sensor_anomalies)} sensor speed anomalies detected "
            f"({sum(1 for a in sensor_anomalies if a.severity == 'severe')} severe)"
        )

    # ---- Phase 2c: TravelTime calibration (adapts physics model) ----
    calibrator = _get_calibrator()
    calibration_snapshot: CalibrationSnapshot | None = None
    if travel_time_readings:
        calibration_snapshot = calibrator.update(
            readings=travel_time_readings,
            model_free_flow_speed=FREE_FLOW_SPEED_KMH,
        )

    # ---- Phase 3: Physics engine (shockwave propagation) ----
    physics = _get_physics_engine()

    model_free_flow_speed = FREE_FLOW_SPEED_KMH
    if calibration_snapshot and calibration_snapshot.confidence != "low":
        model_free_flow_speed = calibration_snapshot.adapted_free_flow_speed
    physics.free_flow_speed = round(
        model_free_flow_speed * weather_adjustment.free_flow_factor,
        2,
    )
    physics.critical_density_veh_km_lane = round(
        K_CRITICAL_VEH_KM_LANE * weather_adjustment.capacity_factor,
        2,
    )

    # Build per-camera traffic state from sensor stations and travel-time spans
    node_traffic_states = build_node_traffic_states(
        sensor_readings,
        travel_time_readings=travel_time_readings,
    )
    node_inflows = {
        camera_id: state.local_inflow_vph
        for camera_id, state in node_traffic_states.items()
        if state.local_inflow_vph is not None
    }
    traffic_flow_speed_nodes = sum(
        1 for state in node_traffic_states.values()
        if state.speed_source == "traffic_flow"
    )
    travel_time_speed_nodes = sum(
        1 for state in node_traffic_states.values()
        if state.speed_source == "travel_time"
    )
    if node_traffic_states:
        logger.info(
            f"🗺️  Traffic states: {len(node_inflows)} inflow nodes, "
            f"{traffic_flow_speed_nodes} TrafficFlow speed nodes, "
            f"{travel_time_speed_nodes} TravelTime fallback speed nodes"
        )

    stopped_vehicle_count = _apply_stopped_vehicle_detection(
        vision_records=vision_records,
        capacity_states=capacity_states,
        node_traffic_states=node_traffic_states,
        now=now,
        critical_density_veh_km_lane=physics.critical_density_veh_km_lane,
    )
    if stopped_vehicle_count:
        logger.info(
            "🛑 Detected %s stopped vehicle candidate(s)",
            stopped_vehicle_count,
        )

    # Aggregate sensor as fallback for cameras without a nearby station
    aggregate_sensor: SensorReading | None = None
    if sensor_readings:
        mean_volume = sum(s.inflow_volume_vph for s in sensor_readings) / len(sensor_readings)
        mean_speed = sum(s.average_speed_kmh for s in sensor_readings) / len(sensor_readings)
        aggregate_sensor = SensorReading(
            timestamp=now,
            inflow_volume_vph=round(mean_volume, 1),
            average_speed_kmh=round(mean_speed, 1),
        )

    _apply_fused_capacity_derivation(
        capacity_states,
        node_traffic_states,
        aggregate_sensor,
        physics.free_flow_speed,
        physics.critical_density_veh_km_lane,
        weather_adjustment.capacity_factor,
    )
    situation_impact_count = apply_situation_capacity_impacts(
        capacity_states,
        situation_deviations,
        now=now,
        camera_chainage_map=_camera_chainage_map,
        critical_density_veh_km_lane=physics.critical_density_veh_km_lane,
    )
    if situation_impact_count:
        logger.info(
            "🚧 Applied %s Situation capacity impact(s)",
            situation_impact_count,
        )

    queue_predictions = physics.compute(
        capacity_states=capacity_states,
        sensor=aggregate_sensor,
        camera_chainage_map=_camera_chainage_map,
        camera_coords_map=CAMERA_COORDS,
        now=now,
        node_traffic_states=node_traffic_states if node_traffic_states else None,
    )
    if queue_predictions:
        local_segments = sum(p.local_data_segments for p in queue_predictions)
        fallback_segments = sum(p.fallback_data_segments for p in queue_predictions)
        missing_segments = sum(p.missing_data_segments for p in queue_predictions)
        logger.info(
            f"🌊 Segment data sources: local={local_segments}, "
            f"fallback={fallback_segments}, missing={missing_segments}"
        )

    # Post-physics: evaluate prediction accuracy against TravelTime data
    if calibration_snapshot and travel_time_readings:
        calibration_snapshot = calibrator.evaluate_accuracy(
            readings=travel_time_readings,
            predictions=queue_predictions,
            snapshot=calibration_snapshot,
        )

    # ---- Phase 4: VMS orchestrator ----
    orchestrator = _get_vms_orchestrator()
    vms_recommendations = []

    # 4a: Queue-tail based VMS recommendations (predictive)
    for prediction in queue_predictions:
        recs = orchestrator.generate_recommendations(
            prediction=prediction,
            now=now,
            vms_statuses=vms_statuses,
            surface_state=weather_adjustment.surface_state,
        )
        vms_recommendations.extend(recs)

    # 4b: Sensor-based VMS recommendations (immediate)
    if sensor_anomalies:
        sensor_recs = orchestrator.generate_sensor_recommendations(
            anomalies=sensor_anomalies,
            now=now,
            vms_statuses=vms_statuses,
        )
        vms_recommendations.extend(sensor_recs)

    weather_recs = orchestrator.generate_weather_recommendations(
        road_conditions=road_condition_records,
        now=now,
        vms_statuses=vms_statuses,
    )
    vms_recommendations.extend(weather_recs)

    # ---- Phase 5: Persist ----
    result = TickResult(
        tick_number=_tick_count,
        timestamp=now,
        capacity_states=capacity_states,
        sensor_readings=sensor_readings if aggregate_sensor else [],
        sensor_anomalies=sensor_anomalies,
        vms_statuses=vms_statuses,
        queue_predictions=queue_predictions,
        vms_recommendations=vms_recommendations,
        travel_time_readings=travel_time_readings,
        calibration=calibration_snapshot,
        weather_records=weather_records,
        road_condition_records=road_condition_records,
        weather_adjustment=weather_adjustment,
        situation_deviations=situation_deviations,
    )

    _persist_tick(result, vision_records, now)

    # ---- Phase 6: Summary logging ----
    ok = sum(1 for r in vision_records if r.get("status") == "ok")
    vision_anomalies = sum(1 for s in capacity_states if s.is_anomaly)
    tick_duration_ms = int((time.monotonic() - tick_started) * 1000)
    logger.info(
        f"📊 Tick #{_tick_count}: "
        f"{ok} cameras OK, {vision_anomalies} vision anomalies, "
        f"{len(sensor_anomalies)} sensor anomalies, "
        f"{len(sensor_readings)} sensors, "
        f"{len(queue_predictions)} predictions, "
        f"{len(vms_recommendations)} VMS recommendations, "
        f"{len(travel_time_readings)} travel times, "
        f"{len(weather_records)} weather stations, "
        f"{len(road_condition_records)} road conditions, "
        f"{len(situation_deviations)} situation deviations, "
        f"{tick_duration_ms}ms total"
    )

    for rec in vms_recommendations:
        logger.info(f"🚦 VMS: {rec.summary}")

    return result


def _persist_tick(
    result: TickResult,
    vision_records: list[dict],
    now: datetime,
) -> None:
    """Write tick data to JSONL and dashboard state files."""
    today = now.strftime("%Y-%m-%d")
    day_dir = os.path.join(DATA_DIR, today)
    os.makedirs(day_dir, exist_ok=True)

    jsonl_path = os.path.join(day_dir, "sensor_data.jsonl")

    # Collect all records for JSONL
    all_records: list[dict] = []

    # Vision records
    all_records.extend(vision_records)

    # Weather and road-condition records keep the legacy JSONL shape used by dashboard.py.
    all_records.extend(result.weather_records)
    all_records.extend(result.road_condition_records)

    for deviation in result.situation_deviations:
        record = deviation.model_dump(mode="json")
        record["type"] = "situation"
        all_records.append(record)

    # Sensor readings
    for s in result.sensor_readings:
        all_records.append({
            "type": "sensor_reading",
            "timestamp": now.isoformat(),
            "inflow_volume_vph": s.inflow_volume_vph,
            "average_speed_kmh": s.average_speed_kmh,
        })

    # VMS status snapshots (ground-truth log)
    for vs in result.vms_statuses:
        all_records.append({
            "type": "vms_status",
            "source": "situation_api_proxy",
            "timestamp": now.isoformat(),
            "vms_id": vs.vms_id,
            "vms_name": vs.vms_name,
            "is_active": vs.is_active,
            "displayed_message": vs.displayed_message,
            "speed_limit": vs.speed_limit,
        })

    # Queue predictions
    for qp in result.queue_predictions:
        all_records.append({
            "type": "queue_prediction",
            "timestamp": now.isoformat(),
            "camera_id": qp.camera_id,
            "growth_speed_kmh": qp.growth_speed_kmh,
            "origin_chainage_km": qp.origin_chainage_km,
            "lengths_at_minutes": qp.lengths_at_minutes,
            "prediction_confidence": qp.prediction_confidence,
            "uncertainty_level": qp.uncertainty_level,
            "uncertainty_reason": qp.uncertainty_reason,
            "length_lower_at_minutes": qp.length_lower_at_minutes,
            "length_upper_at_minutes": qp.length_upper_at_minutes,
        })

    # VMS recommendations
    for rec in result.vms_recommendations:
        all_records.append({
            "type": "vms_recommendation",
            "timestamp": now.isoformat(),
            "vms_id": rec.vms_id,
            "urgency": rec.urgency,
            "recommended_message": rec.recommended_message,
            "eta_minutes": rec.estimated_activation_minutes,
            "eta_lower_minutes": rec.eta_lower_minutes,
            "eta_upper_minutes": rec.eta_upper_minutes,
            "confidence": rec.confidence,
            "uncertainty_level": rec.uncertainty_level,
            "current_vms_status": rec.current_vms_status,
            "summary": rec.summary,
        })

    # Sensor anomalies
    for sa in result.sensor_anomalies:
        all_records.append({
            "type": "sensor_anomaly",
            "timestamp": now.isoformat(),
            "site_id": sa.site_id,
            "measured_speed_kmh": sa.measured_speed_kmh,
            "road_speed_limit_kmh": sa.road_speed_limit_kmh,
            "speed_ratio": sa.speed_ratio,
            "volume_vph": sa.volume_vph,
            "severity": sa.severity,
            "nearest_camera_id": sa.nearest_camera_id,
            "lat": sa.lat,
            "lng": sa.lng,
        })

    # Travel time readings
    for tt in result.travel_time_readings:
        all_records.append({
            "type": "travel_time",
            "timestamp": now.isoformat(),
            "route_id": tt.route_id,
            "name": tt.name,
            "travel_time_seconds": tt.travel_time_seconds,
            "free_flow_seconds": tt.free_flow_seconds,
            "delay_seconds": tt.delay_seconds,
            "speed_kmh": tt.speed_kmh,
            "traffic_status": tt.traffic_status,
            "length_meters": tt.length_meters,
        })

    # Calibration snapshot
    if result.calibration:
        cal = result.calibration
        all_records.append({
            "type": "calibration",
            "timestamp": now.isoformat(),
            "adapted_free_flow_speed": cal.adapted_free_flow_speed,
            "correction_factor": cal.correction_factor,
            "measured_free_flow_speed": cal.measured_free_flow_speed,
            "freeflow_segment_count": cal.freeflow_segment_count,
            "congested_segment_count": cal.congested_segment_count,
            "accuracy_hit_rate": cal.accuracy_hit_rate,
            "confidence": cal.confidence,
        })

    if result.weather_adjustment:
        adj = result.weather_adjustment
        all_records.append({
            "type": "weather_adjustment",
            "timestamp": now.isoformat(),
            "surface_state": adj.surface_state,
            "free_flow_factor": adj.free_flow_factor,
            "capacity_factor": adj.capacity_factor,
            "confidence": adj.confidence,
            "reason": adj.reason,
            "warning_count": len(adj.warning_records),
        })

    # Write JSONL
    if all_records:
        with open(jsonl_path, "a", encoding="utf-8") as f:
            for record in all_records:
                f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        logger.info(f"✅ Saved {len(all_records)} records → {os.path.basename(jsonl_path)}")

    # Write latest vision state for dashboard
    state_path = os.path.join(DATA_DIR, "vision_state.json")
    try:
        state_data = {
            "timestamp": now.isoformat(),
            "cameras": [s.model_dump(mode="json") for s in result.capacity_states],
        }
        with open(state_path, "w", encoding="utf-8") as f:
            json.dump(state_data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"Could not write vision_state.json: {e}")

    # Write status.json for dashboard
    _update_status(result, vision_records, now)


def _update_status(
    result: TickResult,
    vision_records: list[dict],
    now: datetime,
) -> None:
    """Write status.json for the dashboard to read."""
    status = {
        "running": True,
        "start_time": _start_time,
        "last_update": now.isoformat(),
        "tick_count": _tick_count,
        "interval_seconds": INTERVAL_SECONDS,
        "architecture": "tick-based",
        "total_anomalies": get_total_count(DATA_DIR),
        "last_tick": {
            "cameras_ok": sum(1 for r in vision_records if r.get("status") == "ok"),
            "cameras_failed": sum(1 for r in vision_records if r.get("status") != "ok"),
            "total_vehicles": sum(s.vehicle_count for s in result.capacity_states),
            "anomalies": sum(1 for s in result.capacity_states if s.is_anomaly),
            "sensor_readings": len(result.sensor_readings),
            "queue_predictions": len(result.queue_predictions),
            "vms_recommendations": len(result.vms_recommendations),
            "vms_statuses_polled": len(result.vms_statuses),
            "travel_time_routes": len(result.travel_time_readings),
            "weather_stations": len(result.weather_records),
            "road_conditions": len(result.road_condition_records),
            "situation_deviations": len(result.situation_deviations),
            "situation_capacity_impacts": sum(
                1 for s in result.capacity_states if s.situation_confirmed
            ),
            "surface_state": (
                result.weather_adjustment.surface_state
                if result.weather_adjustment
                else "unknown"
            ),
            "surface_adjustment_confidence": (
                result.weather_adjustment.confidence
                if result.weather_adjustment
                else "unknown"
            ),
        },
    }

    status_path = os.path.join(DATA_DIR, "status.json")
    try:
        with open(status_path, "w", encoding="utf-8") as f:
            json.dump(status, f, ensure_ascii=False, default=str, indent=2)
    except Exception as e:
        logger.error(f"Could not write status.json: {e}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Tick-based main loop for PTRE — discrete 60s polling"
    )
    parser.add_argument("--once", action="store_true", help="Run one tick only")
    parser.add_argument("--interval", type=int, default=INTERVAL_SECONDS, help="Seconds between ticks")
    args = parser.parse_args()

    if not API_KEY:
        print("❌ Missing API key. Set TRAFIKVERKET_API_KEY in .env")
        sys.exit(1)

    setup_file_logger(DATA_DIR)

    # Use all configured cameras
    camera_ids = list(CAMERA_IDS)

    # Filter out excluded cameras
    excluded_file = os.path.join(DATA_DIR, "excluded_cameras.json")
    try:
        with open(excluded_file, "r", encoding="utf-8") as f:
            excluded = set(json.load(f))
        camera_ids = [c for c in camera_ids if c not in excluded]
    except (FileNotFoundError, json.JSONDecodeError):
        pass

    logger.info(f"🎯 Monitoring {len(camera_ids)} cameras")
    logger.info(f"📂 Data dir: {DATA_DIR}")
    logger.info(f"⏱  Interval: {args.interval}s (tick-based architecture)")
    logger.info(f"🌊 Physics engine: LWR kinematic wave model")
    logger.info(f"🚦 VMS orchestrator: predictive queue tail modeling")

    if args.once:
        logger.info("🔂 Running one tick (--once)")
        tick_once(camera_ids)
        return

    logger.info("🚀 Starting tick-based main loop... (Ctrl+C to stop)")

    while not _shutdown:
        # Re-read exclusions each tick
        try:
            with open(excluded_file, "r", encoding="utf-8") as f:
                excluded = set(json.load(f))
            camera_ids = [c for c in CAMERA_IDS if c not in excluded]
        except (FileNotFoundError, json.JSONDecodeError):
            camera_ids = list(CAMERA_IDS)

        try:
            tick_once(camera_ids)
        except Exception as e:
            logger.error(f"💥 Tick error: {e}", exc_info=True)

        # Sleep with shutdown check
        for _ in range(args.interval):
            if _shutdown:
                break
            time.sleep(1)

    logger.info("👋 Main loop stopped. Goodbye!")


if __name__ == "__main__":
    main()
