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
    SensorAnomaly,
    SensorReading,
    TickResult,
    TravelTimeReading,
    VMSStatusSnapshot,
)
from src.travel_time_calibrator import TravelTimeCalibrator
from src.anomaly_store import record_anomaly, get_total_count
from src.density_smoother import DensitySmoother
from src.physics_engine import PhysicsEngine
from src.roi_mapper import ROIMapper
from src.vision_engine import VisionEngine
from src.vms_orchestrator import VMSOrchestrator

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


def in_bbox(lat: float, lng: float) -> bool:
    return (
        BBOX["min_lat"] <= lat <= BBOX["max_lat"]
        and BBOX["min_lng"] <= lng <= BBOX["max_lng"]
    )


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
_roi_mapper: ROIMapper | None = None
_physics_engine: PhysicsEngine | None = None
_vms_orchestrator: VMSOrchestrator | None = None
_density_smoother: DensitySmoother | None = None
_travel_time_calibrator: TravelTimeCalibrator | None = None


def _get_vision_engine() -> VisionEngine:
    global _vision_engine
    if _vision_engine is None:
        _vision_engine = VisionEngine()
    return _vision_engine


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


def _get_density_smoother() -> DensitySmoother:
    """Persistent density smoother — maintains EMA state across ticks."""
    global _density_smoother
    if _density_smoother is None:
        _density_smoother = DensitySmoother(alpha=0.4)
    return _density_smoother


# ---------------------------------------------------------------------------
# Camera chainage mapping (for physics engine)
# ---------------------------------------------------------------------------

# Maps camera_id → approximate chainage (km) along the E4 corridor.
# Derived from sorted latitude of CAMERA_COORDS — south to north.
# In production this would come from the VMS config or a separate mapping.
def build_camera_chainage_map(
    camera_coords: dict[str, tuple[float, float]] | None = None,
) -> dict[str, float]:
    """Build a rough chainage map from camera coordinates.

    Sort cameras by latitude (south → north) and assign proportional
    chainage along the 15.8 km corridor (matching vms_config.json datum).

    Parameters
    ----------
    camera_coords:
        Mapping of camera_id → (lat, lng).  Falls back to
        ``config.CAMERA_COORDS`` when not provided.
    """
    coords = camera_coords if camera_coords is not None else CAMERA_COORDS
    if not coords:
        return {}

    sorted_cameras = sorted(coords.items(), key=lambda x: x[1][0])
    min_lat = sorted_cameras[0][1][0]
    max_lat = sorted_cameras[-1][1][0]
    lat_range = max_lat - min_lat

    corridor_km = 15.8  # From vms_config.json — Hallunda to Kristineberg

    chainage_map: dict[str, float] = {}
    for cam_id, (lat, _lng) in sorted_cameras:
        if lat_range > 0:
            fraction = (lat - min_lat) / lat_range
        else:
            fraction = 0.0
        chainage_map[cam_id] = round(fraction * corridor_km, 2)

    return chainage_map


# Backward-compatible alias
_build_camera_chainage_map = build_camera_chainage_map


def build_node_inflows(
    sensor_readings: list[SensorReading],
    camera_coords: dict[str, tuple[float, float]] | None = None,
) -> dict[str, float]:
    """Map sensor station readings to the nearest camera node.

    For each sensor with a ``site_id`` and known coordinates, find the
    camera with the smallest latitude distance and assign its measured
    volume as that camera's local inflow.

    When multiple sensors map to the same camera, their flows are summed
    (they typically represent different lanes at the same location).

    Parameters
    ----------
    sensor_readings:
        Per-station readings from ``fetch_sensor_data`` (with ``site_id``).
    camera_coords:
        Mapping of camera_id → (lat, lng).  Falls back to
        ``config.CAMERA_COORDS`` when not provided.

    Returns
    -------
    dict[str, float]
        Mapping of camera_id → total inflow volume (VPH).
    """
    from config import SENSOR_COORDS

    coords = camera_coords if camera_coords is not None else CAMERA_COORDS
    if not coords or not sensor_readings:
        return {}

    cam_items = list(coords.items())  # [(cam_id, (lat, lng)), ...]

    node_inflows: dict[str, float] = {}

    for reading in sensor_readings:
        if reading.site_id is None:
            continue
        sensor_pos = SENSOR_COORDS.get(reading.site_id)
        if sensor_pos is None:
            continue

        # Find nearest camera by latitude (corridor is roughly N-S)
        best_cam = min(
            cam_items,
            key=lambda c: abs(c[1][0] - sensor_pos[0]),
        )[0]

        node_inflows[best_cam] = (
            node_inflows.get(best_cam, 0.0) + reading.inflow_volume_vph
        )

    return node_inflows


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
    total_vehicles = sum(s.vehicle_count for s in multi_state.segments)
    total_capacity = sum(s.capacity_vph for s in multi_state.segments)
    any_anomaly = any(s.is_anomaly for s in multi_state.segments)
    anomaly_reasons = [
        s.anomaly_reason
        for s in multi_state.segments
        if s.is_anomaly and s.anomaly_reason
    ]
    max_density = max(
        (s.observed_density_veh_km_lane for s in multi_state.segments),
        default=0.0,
    )

    state = CapacityState(
        timestamp=multi_state.timestamp,
        camera_id=multi_state.camera_id,
        vehicle_count=total_vehicles,
        blocked_lanes=0,
        total_lanes=camera_meta.num_lanes,
        estimated_capacity_vph=round(total_capacity, 1),
        observed_density_veh_km_lane=round(max_density, 2),
        is_anomaly=any_anomaly,
        anomaly_reason="; ".join(anomaly_reasons) if anomaly_reasons else None,
        confidence=round(
            float(np.mean([s.confidence for s in multi_state.segments]))
            if multi_state.segments else 0.0,
            3,
        ),
    )
    road_segments_data = {
        seg.road_id: {
            "direction": seg.direction,
            "count": seg.vehicle_count,
            "capacity_vph": seg.capacity_vph,
            "density_veh_km_lane": seg.observed_density_veh_km_lane,
        }
        for seg in multi_state.segments
    }
    return state, road_segments_data


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

    engine = _get_vision_engine()
    retention = _get_retention_policy()
    roi_mapper = _get_roi_mapper()

    vision_records: list[dict] = []
    capacity_states: list[CapacityState] = []

    for cam in cameras:
        cam_id = cam.get("Id", "unknown")
        cam_name = cam.get("Name", cam_id)
        photo_url = cam.get("PhotoUrl", "")
        if not photo_url:
            continue

        if cam.get("HasFullSizePhoto"):
            photo_url = photo_url + "?type=fullsize"

        raw_bytes = fetch_image_bytes(photo_url)
        if raw_bytes is None:
            vision_records.append({
                "type": "vision_result",
                "timestamp": now.isoformat(),
                "camera_id": cam_id,
                "status": "fetch_failed",
            })
            continue

        frame = decode_frame(raw_bytes)
        if frame is None:
            continue

        coords = CAMERA_COORDS.get(cam_id, (0.0, 0.0))
        meta = CameraMetadata(
            camera_id=cam_id, name=cam_name, lat=coords[0], lng=coords[1],
        )

        # Run YOLO (stateless — each frame evaluated independently)
        road_segments_data = None
        if roi_mapper.has_rois(cam_id):
            multi_state = engine.analyze_multi_roi(frame, meta, roi_mapper)
            state, road_segments_data = _aggregate_multi_roi_capacity(
                multi_state, meta,
            )
        else:
            state = engine.analyze_array(frame, meta)

        # Smart retention
        retained_path = retention.maybe_retain(raw_bytes, cam_id, now, state)

        # Anomaly persistence: save annotated image + log event
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
        }
        vision_records.append(record)
        capacity_states.append(state)

        anomaly_tag = f" 🚨 {state.anomaly_reason}" if state.is_anomaly else ""
        logger.info(
            f"📷 {cam_name} — {state.vehicle_count} vehicles, "
            f"{state.estimated_capacity_vph:.0f} VPH{anomaly_tag}"
        )

        del frame, raw_bytes

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

                speed_limit = _parse_speed_limit(temp_limit)
                display_msg = temp_limit if temp_limit else None

                statuses.append(VMSStatusSnapshot(
                    timestamp=now,
                    vms_id=dev_id,
                    vms_name=f"{road} — {location[:60]}" if location else road,
                    is_active=bool(temp_limit),
                    displayed_message=display_msg,
                    speed_limit=speed_limit,
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
        nearest_cam = _find_nearest_camera(coords[0]) if coords[0] else None

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


def _find_nearest_camera(lat: float) -> str | None:
    """Find the camera ID nearest to a given latitude."""
    if not CAMERA_COORDS:
        return None
    return min(
        CAMERA_COORDS,
        key=lambda cid: abs(CAMERA_COORDS[cid][0] - lat),
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

    with ThreadPoolExecutor(max_workers=4) as executor:
        future_cameras = executor.submit(fetch_cameras, camera_ids, now)
        future_sensors = executor.submit(fetch_sensor_data, now)
        future_vms = executor.submit(fetch_vms_status, now)
        future_tt = executor.submit(fetch_travel_times, now)

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
        from src.physics_engine import FREE_FLOW_SPEED_KMH
        calibration_snapshot = calibrator.update(
            readings=travel_time_readings,
            model_free_flow_speed=FREE_FLOW_SPEED_KMH,
        )

    # ---- Phase 3: Physics engine (shockwave propagation) ----
    physics = _get_physics_engine()

    # Apply calibrated free-flow speed
    if calibration_snapshot and calibration_snapshot.confidence != "low":
        physics.free_flow_speed = calibration_snapshot.adapted_free_flow_speed

    # Build per-camera inflow map from nearest sensor stations
    node_inflows = build_node_inflows(sensor_readings)
    if node_inflows:
        logger.info(
            f"🗺️  Mapped {len(sensor_readings)} sensors → "
            f"{len(node_inflows)} camera nodes"
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

    queue_predictions = physics.compute(
        capacity_states=capacity_states,
        sensor=aggregate_sensor,
        camera_chainage_map=_camera_chainage_map,
        camera_coords_map=CAMERA_COORDS,
        now=now,
        node_inflows=node_inflows if node_inflows else None,
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
    )

    _persist_tick(result, vision_records, now)

    # ---- Phase 6: Summary logging ----
    ok = sum(1 for r in vision_records if r.get("status") == "ok")
    vision_anomalies = sum(1 for s in capacity_states if s.is_anomaly)
    logger.info(
        f"📊 Tick #{_tick_count}: "
        f"{ok} cameras OK, {vision_anomalies} vision anomalies, "
        f"{len(sensor_anomalies)} sensor anomalies, "
        f"{len(sensor_readings)} sensors, "
        f"{len(queue_predictions)} predictions, "
        f"{len(vms_recommendations)} VMS recommendations, "
        f"{len(travel_time_readings)} travel times"
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
