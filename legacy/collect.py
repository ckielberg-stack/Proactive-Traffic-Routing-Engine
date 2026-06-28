#!/usr/bin/env python3
"""
Samlar kamerabilder och sensordata från Trafikverkets API varje minut.

Användning:
    python collect.py              # Kör kontinuerligt
    python collect.py --once       # Kör en enda cykel (för test)
    python collect.py --discover   # Auto-discover kameror och börja samla

Stoppa med Ctrl+C.
"""
import argparse
import json
import logging
import os
import signal
import sys
import time
from datetime import datetime
from logging.handlers import RotatingFileHandler

import cv2
import numpy as np
import requests

from config import (
    API_KEY,
    API_URL,
    BBOX,
    CAMERA_COORDS,
    CAMERA_IDS,
    DATA_DIR,
    INTERVAL_SECONDS,
    MAX_RETRIES,
    RETRY_BACKOFF,
)
from retention import RetentionPolicy
from src.models import CameraMetadata, CapacityState, MultiSegmentCapacity
from src.roi_mapper import ROIMapper
from src.vision_engine import VisionEngine

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

logger = logging.getLogger("trafikcollector")
logger.setLevel(logging.DEBUG)

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT))
logger.addHandler(ch)


def setup_file_logger(data_dir: str) -> None:
    os.makedirs(data_dir, exist_ok=True)
    log_path = os.path.join(data_dir, "collector.log")
    fh = RotatingFileHandler(log_path, maxBytes=10_000_000, backupCount=5)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT))
    logger.addHandler(fh)


# ---------------------------------------------------------------------------
# Graceful shutdown
# ---------------------------------------------------------------------------
_shutdown = False


def _signal_handler(sig, frame):
    global _shutdown
    logger.info("🛑 Avslutningssignal mottagen, avslutar efter pågående cykel...")
    _shutdown = True


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
            logger.warning(
                f"API-fel (försök {attempt}/{retries}): {e} — väntar {wait}s"
            )
            if attempt < retries:
                time.sleep(wait)
    logger.error(f"API-anrop misslyckades efter {retries} försök")
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
            logger.warning(
                f"Bildfel (försök {attempt}/{retries}): {e} — väntar {wait}s"
            )
            if attempt < retries:
                time.sleep(wait)
    logger.error(f"Kunde inte ladda ner bild: {url}")
    return None


def decode_frame(raw_bytes: bytes) -> np.ndarray | None:
    """Decode JPEG bytes to a BGR numpy array in memory."""
    arr = np.frombuffer(raw_bytes, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return frame


# ---------------------------------------------------------------------------
# Vision Engine, Retention & ROI (lazy singletons)
# ---------------------------------------------------------------------------
_vision_engine: VisionEngine | None = None
_retention_policy: RetentionPolicy | None = None
_roi_mapper: ROIMapper | None = None


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


# ---------------------------------------------------------------------------
# Auto-discover cameras
# ---------------------------------------------------------------------------
def auto_discover_cameras(max_cameras: int = 4) -> list[str]:
    xml_query = f"""
    <REQUEST>
        <LOGIN authenticationkey="{API_KEY}" />
        <QUERY objecttype="Camera" schemaversion="1">
            <FILTER>
                <AND>
                    <EQ name="Active" value="true" />
                    <EQ name="HasFullSizePhoto" value="true" />
                </AND>
            </FILTER>
        </QUERY>
    </REQUEST>
    """
    data = api_request(xml_query)
    if not data:
        return []

    results = data.get("RESPONSE", {}).get("RESULT", [])
    cameras = results[0].get("Camera", []) if results else []

    # Filter by bounding box
    in_area = []
    for cam in cameras:
        coords = parse_point_wgs84(cam.get("Geometry", {}).get("WGS84", ""))
        if coords and in_bbox(*coords):
            in_area.append(cam)

    selected = in_area[:max_cameras]

    logger.info(f"🔍 Auto-discover: {len(in_area)} kameror i området, valde {len(selected)}")
    for cam in selected:
        logger.info(f"   📷 {cam.get('Id')} — {cam.get('Name', '?')}")

    return [cam["Id"] for cam in selected]


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------
def get_today_dir() -> str:
    today = datetime.now().strftime("%Y-%m-%d")
    day_dir = os.path.join(DATA_DIR, today)
    os.makedirs(day_dir, exist_ok=True)
    return day_dir


# ---------------------------------------------------------------------------
# In-Memory Camera Processing Pipeline
# ---------------------------------------------------------------------------

def process_cameras(
    camera_ids: list[str], now: datetime,
) -> tuple[list[dict], list[CapacityState]]:
    """Fetch camera images into RAM, run YOLO, apply retention, return metadata.

    Returns (vision_records, capacity_states) — no standard images saved to disk.
    """
    if not camera_ids:
        return [], []

    # Batch-fetch camera metadata from API
    id_filter = "\n".join(
        f'<EQ name="Id" value="{cid}" />' for cid in camera_ids
    )
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
    multi_segment_states: list[MultiSegmentCapacity] = []

    for cam in cameras:
        cam_id = cam.get("Id", "unknown")
        cam_name = cam.get("Name", cam_id)
        photo_url = cam.get("PhotoUrl", "")
        if not photo_url:
            logger.warning(f"Kamera {cam_id} saknar PhotoUrl")
            continue

        # Use full-size image (1280×720)
        if cam.get("HasFullSizePhoto"):
            photo_url = photo_url + "?type=fullsize"

        # --- Step 1: Fetch bytes into RAM ---
        raw_bytes = fetch_image_bytes(photo_url)
        if raw_bytes is None:
            vision_records.append({
                "type": "vision_result",
                "timestamp": now.isoformat(),
                "camera_id": cam_id,
                "camera_name": cam_name,
                "photo_time": cam.get("PhotoTime", ""),
                "status": "fetch_failed",
                "vehicle_count": 0,
                "is_anomaly": False,
                "capacity_vph": 0.0,
                "retained_path": None,
                "road_segments": None,
            })
            logger.warning(f"📷 {cam_name} — fetch misslyckades")
            continue

        # --- Step 2: Decode to numpy array ---
        frame = decode_frame(raw_bytes)
        if frame is None:
            logger.warning(f"📷 {cam_name} — kunde inte avkoda bild")
            continue

        # --- Step 3: Build CameraMetadata ---
        coords = CAMERA_COORDS.get(cam_id, (0.0, 0.0))
        meta = CameraMetadata(
            camera_id=cam_id,
            name=cam_name,
            lat=coords[0],
            lng=coords[1],
        )

        # --- Step 4: Run YOLO in-memory (multi-ROI or single-mode) ---
        road_segments_data = None
        if roi_mapper.has_rois(cam_id):
            # Multi-ROI path: per-segment output
            multi_state = engine.analyze_multi_roi(frame, meta, roi_mapper)
            multi_segment_states.append(multi_state)

            # Build road_segments dict for the record
            road_segments_data = {
                seg.road_id: {
                    "direction": seg.direction,
                    "count": seg.vehicle_count,
                    "capacity_vph": seg.capacity_vph,
                    "num_lanes": seg.num_lanes,
                    "is_anomaly": seg.is_anomaly,
                    "anomaly_reason": seg.anomaly_reason,
                    "confidence": seg.confidence,
                }
                for seg in multi_state.segments
            }

            # Also produce a CapacityState for backward compatibility
            total_vehicles = sum(s.vehicle_count for s in multi_state.segments)
            total_capacity = sum(s.capacity_vph for s in multi_state.segments)
            any_anomaly = any(s.is_anomaly for s in multi_state.segments)
            anomaly_reasons = [
                s.anomaly_reason for s in multi_state.segments if s.is_anomaly
            ]
            state = CapacityState(
                timestamp=multi_state.timestamp,
                camera_id=cam_id,
                vehicle_count=total_vehicles,
                blocked_lanes=0,
                total_lanes=meta.num_lanes,
                estimated_capacity_vph=round(total_capacity, 1),
                is_anomaly=any_anomaly,
                anomaly_reason="; ".join(anomaly_reasons) if anomaly_reasons else None,
                confidence=round(
                    float(np.mean([s.confidence for s in multi_state.segments]))
                    if multi_state.segments else 0.0,
                    3,
                ),
            )
        else:
            # Single-mode fallback (full-frame, no ROIs)
            state = engine.analyze_array(frame, meta)

        # --- Step 5: Smart retention ---
        retained_path = retention.maybe_retain(
            raw_bytes, cam_id, now, state,
        )

        # --- Step 6: Build record (metadata only) ---
        size_kb = len(raw_bytes) / 1024
        record = {
            "type": "vision_result",
            "timestamp": now.isoformat(),
            "camera_id": cam_id,
            "camera_name": cam_name,
            "photo_url": photo_url,
            "photo_time": cam.get("PhotoTime", ""),
            "status": "ok",
            "vehicle_count": state.vehicle_count,
            "blocked_lanes": state.blocked_lanes,
            "total_lanes": state.total_lanes,
            "capacity_vph": state.estimated_capacity_vph,
            "is_anomaly": state.is_anomaly,
            "anomaly_reason": state.anomaly_reason,
            "confidence": state.confidence,
            "image_size_bytes": len(raw_bytes),
            "retained_path": retained_path,
            "road_segments": road_segments_data,
        }
        vision_records.append(record)
        capacity_states.append(state)

        # Log result
        anomaly_tag = f" 🚨 {state.anomaly_reason}" if state.is_anomaly else ""
        retained_tag = f" 💾 {os.path.basename(retained_path)}" if retained_path else ""
        roi_tag = f" 🗺️  {len(road_segments_data)} segments" if road_segments_data else ""
        logger.info(
            f"📷 {cam_name} — {state.vehicle_count} fordon, "
            f"{state.estimated_capacity_vph:.0f} VPH, "
            f"{size_kb:.0f} KB (in-mem)"
            f"{roi_tag}{anomaly_tag}{retained_tag}"
        )

        # --- Step 7: Release memory ---
        del frame, raw_bytes

    return vision_records, capacity_states


def fetch_weather_data(now: datetime) -> list[dict]:
    """Hämtar väderdata från mätpunkter nära sträckan."""
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

    weather_records = []
    for pt in points:
        coords = parse_point_wgs84(pt.get("Geometry", {}).get("WGS84", ""))
        if not coords or not in_bbox(*coords):
            continue

        obs = pt.get("Observation", {})
        air = obs.get("Air", {})
        surface = obs.get("Surface", {})
        wind_list = obs.get("Wind", [])
        wind = wind_list[0] if wind_list else {}
        weather = obs.get("Weather", {})
        agg5 = obs.get("Aggregated5minutes", {})

        record = {
            "type": "weather",
            "timestamp": now.isoformat(),
            "station_id": pt.get("Id", ""),
            "station_name": pt.get("Name", ""),
            "sample_time": obs.get("Sample", ""),
            "air_temp_c": _val(air, "Temperature"),
            "air_humidity_pct": _val(air, "RelativeHumidity"),
            "air_dewpoint_c": _val(air, "Dewpoint"),
            "visibility_m": _val(air, "VisibleDistance"),
            "wind_speed_ms": _val(wind, "Speed"),
            "wind_dir_deg": _val(wind, "Direction"),
            "surface_temp_c": _val(surface, "Temperature"),
            "precipitation": weather.get("Precipitation", None),
            "precip_rain_sum": _nested(agg5, "Precipitation", "RainSum", "Value"),
            "precip_snow_water_eq": _nested(agg5, "Precipitation", "SnowSum", "WaterEquivalent", "Value"),
        }
        weather_records.append(record)

    logger.info(f"🌡  Väderdata: {len(weather_records)} mätpunkt(er) i området")
    return weather_records


def fetch_road_conditions(now: datetime) -> list[dict]:
    """Hämtar väglagsdata för sträckan."""
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

    road_records = []
    for cond in conditions:
        record = {
            "type": "road_condition",
            "timestamp": now.isoformat(),
            "id": cond.get("Id", ""),
            "location": cond.get("LocationText", ""),
            "condition_text": cond.get("ConditionText", ""),
            "condition_info": cond.get("ConditionInfo", []),
            "condition_code": cond.get("ConditionCode"),
            "warning": cond.get("Warning", False),
            "road_number": cond.get("RoadNumber", ""),
            "start_time": cond.get("StartTime", ""),
        }
        road_records.append(record)

    logger.info(f"🛣  Väglag: {len(road_records)} poster (Stockholms län)")
    return road_records


def _is_e4_road(road_number: str) -> bool:
    """Check if road number is E4 or E20 (our target corridor)."""
    if not road_number:
        return False
    rn = road_number.strip().upper()
    return rn in ("E 4", "E4", "E 20", "E20", "E4/E20", "E 4/E 20")


def _point_in_bbox(wgs84: str) -> bool:
    """Check if a WGS84 POINT geometry falls within our bounding box."""
    if not wgs84 or "POINT" not in wgs84:
        return False
    try:
        # Format: "POINT (lng lat)"
        coords = wgs84.split("(")[1].split(")")[0].strip().split()
        lng, lat = float(coords[0]), float(coords[1])
        return (
            BBOX["min_lat"] <= lat <= BBOX["max_lat"]
            and BBOX["min_lng"] <= lng <= BBOX["max_lng"]
        )
    except (IndexError, ValueError):
        return False


def fetch_situations(now: datetime) -> list[dict]:
    """Hämtar trafikhändelser på E4/E20 Södertälje–Stockholm."""
    xml_query = f"""
    <REQUEST>
        <LOGIN authenticationkey="{API_KEY}" />
        <QUERY objecttype="Situation" schemaversion="1">
            <FILTER>
                <ELEMENTMATCH>
                    <EQ name="Deviation.CountyNo" value="1" />
                </ELEMENTMATCH>
            </FILTER>
        </QUERY>
    </REQUEST>
    """

    data = api_request(xml_query)
    if not data:
        return []

    results = data.get("RESPONSE", {}).get("RESULT", [])
    situations = results[0].get("Situation", []) if results else []

    situation_records = []
    skipped = 0
    for sit in situations:
        for dev in sit.get("Deviation", []):
            road = dev.get("RoadNumber", "")
            geom = dev.get("Geometry", {})
            wgs84 = geom.get("WGS84", "")

            # Only keep E4/E20 incidents within our bounding box
            if not _is_e4_road(road) and not _point_in_bbox(wgs84):
                skipped += 1
                continue

            record = {
                "type": "situation",
                "timestamp": now.isoformat(),
                "situation_id": sit.get("Id", ""),
                "deviation_id": dev.get("Id", ""),
                "message_type": dev.get("MessageType", ""),
                "message_code": dev.get("MessageCode", ""),
                "header": dev.get("Header", ""),
                "location": dev.get("LocationDescriptor", ""),
                "road_number": road,
                "severity_code": dev.get("SeverityCode"),
                "severity_text": dev.get("SeverityText", ""),
                "start_time": dev.get("StartTime", ""),
                "end_time": dev.get("EndTime", ""),
                "message": dev.get("Message", ""),
                "icon": dev.get("IconId", ""),
                "managed_cause": dev.get("ManagedCause", False),
                "geometry_wgs84": wgs84,
                "lanes_restricted": dev.get("NumberOfLanesRestricted"),
                "traffic_restriction": dev.get("TrafficRestrictionType", ""),
            }
            situation_records.append(record)

    # Count by type for logging
    type_counts = {}
    for r in situation_records:
        mt = r["message_type"]
        type_counts[mt] = type_counts.get(mt, 0) + 1
    type_str = ", ".join(f"{t}: {c}" for t, c in sorted(type_counts.items()))

    logger.info(
        f"🚨 Situationer: {len(situation_records)} på E4/E20 "
        f"({type_str}) — {skipped} utanför sträckan filtrerade bort"
    )
    return situation_records


def _val(d: dict, key: str):
    """Get nested .Value from a sensor reading like {'Temperature': {'Value': -9.1}}."""
    sub = d.get(key, {})
    return sub.get("Value") if isinstance(sub, dict) else None


def _nested(d: dict, *keys):
    for k in keys:
        if isinstance(d, dict):
            d = d.get(k)
        else:
            return None
    return d


def append_jsonl(records: list[dict], filepath: str) -> None:
    with open(filepath, "a", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def _count_by_key(records: list[dict], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for r in records:
        val = r.get(key, "?")
        counts[val] = counts.get(val, 0) + 1
    return counts


# ---------------------------------------------------------------------------
# Main collection cycle
# ---------------------------------------------------------------------------
def collect_once(camera_ids: list[str]) -> None:
    now = datetime.now()
    day_dir = get_today_dir()
    jsonl_path = os.path.join(day_dir, "sensor_data.jsonl")

    logger.info(f"{'='*60}")
    logger.info(f"⏰ Insamlingscykel: {now.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"{'='*60}")

    all_records = []

    # 1. Camera vision pipeline (in-memory, no standard disk writes)
    vision_records, capacity_states = process_cameras(camera_ids, now)
    all_records.extend(vision_records)

    # 2. Väderdata
    weather_records = fetch_weather_data(now)
    all_records.extend(weather_records)

    # 3. Väglag
    road_records = fetch_road_conditions(now)
    all_records.extend(road_records)

    # 4. Situationer (olyckor, vägarbeten — ground truth)
    situation_records = fetch_situations(now)
    all_records.extend(situation_records)

    # Spara metadata
    if all_records:
        append_jsonl(all_records, jsonl_path)
        logger.info(
            f"✅ Sparade {len(all_records)} poster → {os.path.basename(jsonl_path)}"
        )
    else:
        logger.warning("⚠️  Inga poster att spara denna cykel")

    # Save latest vision state for dashboard
    _save_vision_state(capacity_states, now)

    # Statistik
    ok = sum(1 for r in vision_records if r.get("status") == "ok")
    failed = len(vision_records) - ok
    anomalies = sum(1 for r in vision_records if r.get("is_anomaly"))
    retained = sum(1 for r in vision_records if r.get("retained_path"))
    total_vehicles = sum(r.get("vehicle_count", 0) for r in vision_records)
    logger.info(
        f"📊 Vision: {ok} OK, {failed} misslyckade, "
        f"{total_vehicles} fordon, {anomalies} anomalier, {retained} sparade"
    )

    # Update status.json for dashboard
    _update_status(camera_ids, vision_records, weather_records, road_records, situation_records, now)


# Track errors across cycles
_error_log: list[dict] = []
_cycle_count = 0
_start_time: str | None = None


def _save_vision_state(
    capacity_states: list[CapacityState], now: datetime,
) -> None:
    """Write latest vision results for dashboard consumption."""
    state_path = os.path.join(DATA_DIR, "vision_state.json")
    try:
        data = {
            "timestamp": now.isoformat(),
            "cameras": [
                s.model_dump(mode="json") for s in capacity_states
            ],
        }
        os.makedirs(DATA_DIR, exist_ok=True)
        with open(state_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"Kunde inte skriva vision_state.json: {e}")


def _update_status(
    camera_ids: list[str],
    vision_records: list[dict],
    weather_records: list[dict],
    road_records: list[dict],
    situation_records: list[dict],
    now: datetime,
) -> None:
    """Write status.json for the dashboard to read."""
    global _cycle_count, _start_time
    _cycle_count += 1
    if not _start_time:
        _start_time = now.isoformat()

    # Disk usage (storage dirs only)
    total_bytes = 0
    retained_images = 0
    for scan_dir in [DATA_DIR, os.path.join(os.path.dirname(__file__), "storage")]:
        try:
            for root, _, files in os.walk(scan_dir):
                for f in files:
                    fp = os.path.join(root, f)
                    total_bytes += os.path.getsize(fp)
                    if f.endswith(".jpg"):
                        retained_images += 1
        except Exception:
            pass

    status = {
        "running": True,
        "start_time": _start_time,
        "last_update": now.isoformat(),
        "cycle_count": _cycle_count,
        "interval_seconds": INTERVAL_SECONDS,
        "camera_ids": camera_ids,
        "last_cycle": {
            "cameras_ok": sum(1 for r in vision_records if r.get("status") == "ok"),
            "cameras_failed": sum(1 for r in vision_records if r.get("status") != "ok"),
            "total_vehicles": sum(r.get("vehicle_count", 0) for r in vision_records),
            "anomalies": sum(1 for r in vision_records if r.get("is_anomaly")),
            "images_retained": sum(1 for r in vision_records if r.get("retained_path")),
            "weather_stations": len(weather_records),
            "road_conditions": len(road_records),
            "cameras": [
                {
                    "id": r.get("camera_id"),
                    "name": r.get("camera_name"),
                    "status": r.get("status"),
                    "vehicle_count": r.get("vehicle_count", 0),
                    "capacity_vph": r.get("capacity_vph", 0.0),
                    "is_anomaly": r.get("is_anomaly", False),
                    "anomaly_reason": r.get("anomaly_reason"),
                    "confidence": r.get("confidence", 0.0),
                    "photo_time": r.get("photo_time", ""),
                    "retained_path": r.get("retained_path"),
                }
                for r in vision_records
            ],
            "situations": len(situation_records),
            "situation_types": _count_by_key(situation_records, "message_type"),
        },
        "totals": {
            "retained_images": retained_images,
            "disk_usage_mb": round(total_bytes / (1024 * 1024), 1),
        },
        "recent_errors": _error_log[-20:],
    }

    status_path = os.path.join(DATA_DIR, "status.json")
    try:
        with open(status_path, "w", encoding="utf-8") as f:
            json.dump(status, f, ensure_ascii=False, default=str, indent=2)
    except Exception as e:
        logger.error(f"Kunde inte skriva status.json: {e}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Samla kamerabilder och sensordata från Trafikverket"
    )
    parser.add_argument("--once", action="store_true", help="Kör en enda cykel")
    parser.add_argument("--discover", action="store_true", help="Auto-discover kameror")
    parser.add_argument("--interval", type=int, default=INTERVAL_SECONDS, help="Sekunder mellan cykler")
    args = parser.parse_args()

    if not API_KEY:
        print("❌ Saknar API-nyckel. Sätt TRAFIKVERKET_API_KEY i .env")
        sys.exit(1)

    setup_file_logger(DATA_DIR)

    # Determine camera IDs
    all_camera_ids = list(CAMERA_IDS)
    if args.discover or not all_camera_ids:
        if not all_camera_ids:
            logger.info("Inga kameror konfigurerade — kör auto-discover...")
        all_camera_ids = auto_discover_cameras(max_cameras=4)
        if not all_camera_ids:
            logger.error("❌ Hittade inga kameror. Kontrollera bounding box i config.py")
            sys.exit(1)

    # Filter out excluded cameras (can be updated via dashboard)
    def _get_active_cameras() -> list[str]:
        excluded_file = os.path.join(DATA_DIR, "excluded_cameras.json")
        try:
            with open(excluded_file, "r", encoding="utf-8") as f:
                excluded = set(json.load(f))
        except (FileNotFoundError, json.JSONDecodeError):
            excluded = set()
        active = [c for c in all_camera_ids if c not in excluded]
        if excluded:
            logger.info(f"📋 {len(excluded)} kameror exkluderade, {len(active)} aktiva")
        return active

    camera_ids = _get_active_cameras()
    logger.info(f"🎯 Övervakar {len(camera_ids)} kameror: {camera_ids}")
    logger.info(f"📂 Datakatalog: {DATA_DIR}")
    logger.info(f"⏱  Intervall: {args.interval}s")

    if args.once:
        logger.info("🔂 Kör en enda cykel (--once)")
        collect_once(camera_ids)
        return

    logger.info("🚀 Startar kontinuerlig insamling... (Ctrl+C för att stoppa)")

    cycle = 0
    while not _shutdown:
        cycle += 1
        # Re-read exclusions each cycle (dashboard may have changed them)
        camera_ids = _get_active_cameras()
        logger.info(f"\n🔄 Cykel #{cycle} — {len(camera_ids)} kameror")
        try:
            collect_once(camera_ids)
        except Exception as e:
            logger.error(f"💥 Oväntat fel i cykel #{cycle}: {e}", exc_info=True)
            _error_log.append({
                "time": datetime.now().isoformat(),
                "cycle": cycle,
                "error": str(e),
            })

        for _ in range(args.interval):
            if _shutdown:
                break
            time.sleep(1)

    logger.info("👋 Insamlingen avslutad. Tack och hej!")

    # Summary
    try:
        today = datetime.now().strftime("%Y-%m-%d")
        img_dir = os.path.join(DATA_DIR, today, "images")
        if os.path.exists(img_dir):
            files = [f for f in os.listdir(img_dir) if f.endswith(".jpg")]
            total_mb = sum(os.path.getsize(os.path.join(img_dir, f)) for f in files) / (1024 * 1024)
            logger.info(f"📊 Idag: {len(files)} bilder, {total_mb:.1f} MB")
    except Exception:
        pass


if __name__ == "__main__":
    main()
