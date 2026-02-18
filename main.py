#!/usr/bin/env python3
"""
Unified PTRE Entry Point — FastAPI + Tick Loop.

Runs the Operator Decision Support API (FastAPI/uvicorn) and the 60-second
tick-based main loop in a single process.  The tick loop executes in a
background thread via ``asyncio.to_thread`` so it never blocks the async
event loop serving API requests.

Each tick's output (CapacityState, QueuePrediction, VMSRecommendation) is
injected into the Operator API's in-memory state so all ``/api/v1/operator/*``
endpoints serve **live** data.

The Camera-to-Camera Prophecy evaluator (``EvaluationLogger``) is also wired
in here — it records predictions and evaluates them against subsequent ticks.

Usage
-----
    python main.py              # continuous (default)
    python main.py --once       # single tick then exit
    python main.py --port 8081  # custom API port
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
import logging
import os
import signal
import sys
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any

import numpy as np
import requests
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from config import (
    API_KEY, API_URL, CAMERA_COORDS, CAMERA_IDS, DATA_DIR,
    INTERVAL_SECONDS, SENSOR_COORDS,
)
from main_loop import (
    build_camera_chainage_map, build_node_inflows,
    setup_file_logger, tick_once,
)
from src.evaluation_logger import EvaluationLogger
from src.incident_builder import build_incident_reports
from src.anomaly_store import get_anomalies, get_total_count
from src.models import CalibrationSnapshot, SensorReading, TravelTimeReading
from src.operator_api import (
    app as operator_app,
    set_pipeline_snapshot,
)

logger = logging.getLogger("ptre.main")

# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------

_shutdown_event = asyncio.Event()
_eval_logger: EvaluationLogger | None = None

# In-memory state for dashboard pages
_latest_sensor_readings: list[SensorReading] = []
_latest_travel_times: list[TravelTimeReading] = []
_latest_camera_states: list[dict] = []
_latest_calibration: CalibrationSnapshot | None = None
_latest_timestamp: str | None = None

# Jinja2 templates
_TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "templates")
templates = Jinja2Templates(directory=_TEMPLATE_DIR)


# ---------------------------------------------------------------------------
# Camera ID resolution (respects exclusions)
# ---------------------------------------------------------------------------


def _resolve_camera_ids() -> list[str]:
    """Return active camera IDs, excluding any in excluded_cameras.json."""
    camera_ids = list(CAMERA_IDS)
    excluded_file = os.path.join(DATA_DIR, "excluded_cameras.json")
    try:
        with open(excluded_file, "r", encoding="utf-8") as f:
            excluded = set(json.load(f))
        camera_ids = [c for c in camera_ids if c not in excluded]
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    return camera_ids


# ---------------------------------------------------------------------------
# Tick loop (runs in background thread via asyncio.to_thread)
# ---------------------------------------------------------------------------


async def _tick_loop_background(
    *,
    run_once: bool = False,
    interval: int = INTERVAL_SECONDS,
) -> None:
    """Run tick_once in a thread and inject results into the API + evaluator."""
    global _eval_logger

    chainage_map = build_camera_chainage_map()
    _eval_logger = EvaluationLogger(
        chainage_map=chainage_map,
        data_dir=DATA_DIR,
    )

    logger.info(f"🚀 Tick loop started (interval={interval}s, once={run_once})")

    while not _shutdown_event.is_set():
        camera_ids = _resolve_camera_ids()

        try:
            # Run the synchronous tick in a thread so we don't block uvicorn
            result = await asyncio.to_thread(tick_once, camera_ids)

            # --- Inject into Operator API state (single atomic snapshot) ---
            incidents = build_incident_reports(
                result.capacity_states,
                camera_coords=CAMERA_COORDS,
            )
            set_pipeline_snapshot(
                incidents=incidents,
                predictions=result.queue_predictions,
                vms_statuses=result.vms_statuses,
                recommendations=result.vms_recommendations,
                last_tick_time=result.timestamp,
            )

            # --- Store state for dashboard pages ---
            global _latest_sensor_readings, _latest_travel_times
            global _latest_camera_states, _latest_timestamp
            _latest_sensor_readings = list(result.sensor_readings)
            _latest_travel_times = list(result.travel_time_readings)
            global _latest_calibration
            _latest_calibration = result.calibration
            _latest_camera_states = [
                {
                    "camera_id": s.camera_id,
                    "name": s.camera_id.split("_")[-1] if "_" in s.camera_id else s.camera_id,
                    "status": "ok",
                    "vehicle_count": s.vehicle_count,
                    "estimated_vph": s.estimated_capacity_vph,
                    "capacity_drop": round(
                        (s.blocked_lanes / s.total_lanes * 100)
                        if s.is_anomaly and s.total_lanes > 0 else 0,
                        1,
                    ),
                    "is_anomaly": s.is_anomaly,
                    "lat": CAMERA_COORDS.get(s.camera_id, (None, None))[0],
                    "lng": CAMERA_COORDS.get(s.camera_id, (None, None))[1],
                }
                for s in result.capacity_states
            ]
            _latest_timestamp = result.timestamp.isoformat()

            # --- Evaluation Logger ---
            _eval_logger.evaluate_pending(
                result.capacity_states, result.timestamp
            )
            _eval_logger.record_prophecies(
                result.queue_predictions, result.timestamp
            )

            stats = _eval_logger.get_stats()
            logger.info(
                f"🔮 Prophecies: {stats['pending']} pending, "
                f"{stats['verified_success']} verified, "
                f"{stats['failed']} failed, "
                f"hit_rate={stats['hit_rate']}"
            )

        except Exception as e:
            logger.error(f"💥 Tick error: {e}", exc_info=True)

        if run_once:
            break

        # Sleep with shutdown check (1-second granularity)
        try:
            await asyncio.wait_for(
                _shutdown_event.wait(), timeout=interval
            )
            break  # shutdown_event was set
        except asyncio.TimeoutError:
            pass  # Normal — interval elapsed, run next tick


# ---------------------------------------------------------------------------
# FastAPI lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start the tick loop as a background task alongside the API server."""
    setup_file_logger(DATA_DIR)

    # Parse CLI args (uvicorn may add its own — we only look at ours)
    run_once = "--once" in sys.argv

    task = asyncio.create_task(
        _tick_loop_background(run_once=run_once)
    )

    yield

    # Shutdown
    logger.info("🛑 Shutting down tick loop...")
    _shutdown_event.set()
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    logger.info("👋 Tick loop stopped.")


# ---------------------------------------------------------------------------
# Application assembly
# ---------------------------------------------------------------------------

# Attach lifespan to the existing operator API app
operator_app.router.lifespan_context = lifespan


# --- Additional endpoint: evaluation stats ---


@operator_app.get("/api/v1/evaluation/stats")
async def evaluation_stats() -> dict[str, Any]:
    """Return Camera-to-Camera Prophecy accuracy statistics."""
    if _eval_logger is None:
        return {
            "status": "not_initialized",
            "message": "Tick loop has not started yet.",
        }
    return _eval_logger.get_stats()


@operator_app.get("/api/v1/evaluation/log")
async def evaluation_log(limit: int = 50) -> dict[str, Any]:
    """Return the prophecy event log for the dashboard feed."""
    if _eval_logger is None:
        return {"entries": [], "stats": {}}
    return {
        "entries": _eval_logger.get_log(limit=limit),
        "stats": _eval_logger.get_stats(),
    }


# --- Static files ---
_STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

operator_app.mount(
    "/static", StaticFiles(directory=_STATIC_DIR), name="static"
)


# --- Page routes (Jinja2 templates) ---


@operator_app.get("/", include_in_schema=False, response_class=HTMLResponse)
async def page_tmc(request: Request):
    """TMC overview (incidents, VMS, prophecy)."""
    return templates.TemplateResponse(request, "tmc.html")


@operator_app.get("/cameras", include_in_schema=False, response_class=HTMLResponse)
async def page_cameras(request: Request):
    """Camera grid page."""
    return templates.TemplateResponse(request, "cameras.html")


@operator_app.get("/sensors", include_in_schema=False, response_class=HTMLResponse)
async def page_sensors(request: Request):
    """Sensor data page."""
    return templates.TemplateResponse(request, "sensors.html")


@operator_app.get("/logs", include_in_schema=False, response_class=HTMLResponse)
async def page_logs(request: Request):
    """System log page."""
    return templates.TemplateResponse(request, "logs.html")


@operator_app.get("/system", include_in_schema=False, response_class=HTMLResponse)
async def page_system(request: Request):
    """System health page."""
    return templates.TemplateResponse(request, "system.html")


@operator_app.get("/anomalies", include_in_schema=False, response_class=HTMLResponse)
async def page_anomalies(request: Request):
    """Anomaly event log page."""
    return templates.TemplateResponse(request, "anomalies.html")


@operator_app.get("/map", include_in_schema=False, response_class=HTMLResponse)
async def page_map(request: Request):
    """Interactive corridor map."""
    return templates.TemplateResponse(request, "map.html")


@operator_app.get("/travel-times", include_in_schema=False, response_class=HTMLResponse)
async def page_travel_times(request: Request):
    """E4/E20 corridor travel times page."""
    return templates.TemplateResponse(request, "travel_times.html")

# --- Data API endpoints (for dashboard pages) ---


@operator_app.get("/api/v1/cameras")
async def api_cameras() -> dict[str, Any]:
    """Per-camera status from latest tick."""
    if not _latest_camera_states:
        # Fallback: read from vision_state.json
        state_path = os.path.join(DATA_DIR, "vision_state.json")
        if os.path.exists(state_path):
            with open(state_path, "r") as f:
                data = json.load(f)
            cameras = []
            for cam in data.get("cameras", []):
                cam_id = cam.get("camera_id", "")
                coords = CAMERA_COORDS.get(cam_id, (None, None))
                cameras.append({
                    "camera_id": cam_id,
                    "name": cam.get("camera_name", cam_id),
                    "status": "ok",
                    "vehicle_count": cam.get("vehicle_count", 0),
                    "estimated_vph": cam.get("estimated_capacity_vph", 0),
                    "capacity_drop": cam.get("capacity_drop_percentage", 0),
                    "is_anomaly": cam.get("is_anomaly", False),
                    "lat": coords[0],
                    "lng": coords[1],
                })
            return {"cameras": cameras, "timestamp": data.get("timestamp")}
        return {"cameras": [], "timestamp": None}
    return {"cameras": _latest_camera_states, "timestamp": _latest_timestamp}


@operator_app.get("/api/v1/sensors")
async def api_sensors() -> dict[str, Any]:
    """Latest sensor readings from current tick."""
    readings = []
    node_inflows = build_node_inflows(_latest_sensor_readings)

    # Build reverse map: camera_id -> list of site_ids
    cam_to_sites: dict[str, list[int]] = {}
    for r in _latest_sensor_readings:
        if r.site_id is None:
            continue
        sensor_pos = SENSOR_COORDS.get(r.site_id)
        if sensor_pos is None:
            continue
        # Find nearest camera (same logic as build_node_inflows)
        best_cam = min(
            CAMERA_COORDS.items(),
            key=lambda c: abs(c[1][0] - sensor_pos[0]),
        )[0]
        cam_to_sites.setdefault(best_cam, []).append(r.site_id)

    # Build a site -> camera map
    site_to_cam: dict[int, str] = {}
    for cam_id, sites in cam_to_sites.items():
        # Shorten camera name
        name = cam_id.split("_")[-1] if "_" in cam_id else cam_id
        for sid in sites:
            site_to_cam[sid] = name

    for r in _latest_sensor_readings:
        sensor_pos = SENSOR_COORDS.get(r.site_id) if r.site_id else None
        readings.append({
            "site_id": r.site_id,
            "lat": sensor_pos[0] if sensor_pos else None,
            "volume_vph": r.inflow_volume_vph,
            "speed_kmh": r.average_speed_kmh,
            "mapped_camera": site_to_cam.get(r.site_id) if r.site_id else None,
        })

    # Sort by latitude (north to south)
    readings.sort(key=lambda x: -(x["lat"] or 0))

    return {
        "readings": readings,
        "mapped_cameras": len(node_inflows),
        "timestamp": _latest_timestamp,
    }


@operator_app.get("/api/v1/travel-times")
async def api_travel_times() -> dict[str, Any]:
    """Latest travel time readings from TravelTimeRoute API.

    Returns per-route-segment travel times with corridor-level summary.
    """
    routes: list[dict] = []
    northbound: list[dict] = []
    southbound: list[dict] = []

    for t in _latest_travel_times:
        entry = {
            "route_id": t.route_id,
            "name": t.name,
            "travel_time_seconds": round(t.travel_time_seconds, 1),
            "free_flow_seconds": round(t.free_flow_seconds, 1),
            "delay_seconds": round(t.delay_seconds, 1),
            "speed_kmh": round(t.speed_kmh, 1),
            "length_meters": round(t.length_meters, 0),
            "traffic_status": t.traffic_status,
        }
        routes.append(entry)

        # Classify direction by route name
        name_upper = t.name.upper()
        if " N " in name_upper or name_upper.startswith("E4 N") or name_upper.startswith("E4/E20 N"):
            northbound.append(entry)
        elif " S " in name_upper or name_upper.startswith("E4 S") or name_upper.startswith("E4/E20 S"):
            southbound.append(entry)

    # Corridor summary
    total_tt = sum(t.travel_time_seconds for t in _latest_travel_times)
    total_ff = sum(t.free_flow_seconds for t in _latest_travel_times)
    total_delay = sum(t.delay_seconds for t in _latest_travel_times)
    slow_count = sum(1 for t in _latest_travel_times if t.traffic_status != "freeflow")

    # Overall congestion status
    if slow_count > len(_latest_travel_times) * 0.5:
        corridor_status = "congested"
    elif slow_count > 0:
        corridor_status = "degraded"
    else:
        corridor_status = "freeflow"

    # Format corridor TT as mm:ss
    corridor_tt_min = int(total_tt // 60)
    corridor_tt_sec = int(total_tt % 60)
    corridor_ff_min = int(total_ff // 60)
    corridor_ff_sec = int(total_ff % 60)

    return {
        "routes": routes,
        "northbound_count": len(northbound),
        "southbound_count": len(southbound),
        "summary": {
            "total_travel_time_seconds": round(total_tt, 1),
            "total_free_flow_seconds": round(total_ff, 1),
            "total_delay_seconds": round(total_delay, 1),
            "corridor_travel_time": f"{corridor_tt_min}:{corridor_tt_sec:02d}",
            "corridor_free_flow": f"{corridor_ff_min}:{corridor_ff_sec:02d}",
            "slow_routes": slow_count,
            "total_routes": len(_latest_travel_times),
            "corridor_status": corridor_status,
        },
        "timestamp": _latest_timestamp,
    }


@operator_app.get("/api/v1/calibration/status")
async def api_calibration_status() -> dict[str, Any]:
    """Current physics calibration state from TravelTimeRoute data."""
    if _latest_calibration is None:
        return {
            "status": "no_data",
            "message": "No calibration data yet — waiting for first tick",
        }

    cal = _latest_calibration
    return {
        "status": "active",
        "adapted_free_flow_speed": cal.adapted_free_flow_speed,
        "correction_factor": cal.correction_factor,
        "measured_free_flow_speed": cal.measured_free_flow_speed,
        "freeflow_segment_count": cal.freeflow_segment_count,
        "congested_segment_count": cal.congested_segment_count,
        "accuracy_hit_rate": cal.accuracy_hit_rate,
        "confidence": cal.confidence,
        "timestamp": _latest_timestamp,
    }

@operator_app.get("/api/v1/logs")
async def api_logs(lines: int = 200) -> dict[str, Any]:
    """Tail mainloop.log."""
    log_path = os.path.join(DATA_DIR, "mainloop.log")
    if not os.path.exists(log_path):
        return {"lines": [], "total": 0}
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            all_lines = f.readlines()
        recent = all_lines[-lines:] if len(all_lines) > lines else all_lines
        parsed = []
        for line in recent:
            line = line.strip()
            if not line:
                continue
            level = "info"
            if "[ERROR]" in line or "[CRITICAL]" in line:
                level = "error"
            elif "[WARNING]" in line:
                level = "warning"
            elif "[DEBUG]" in line:
                level = "debug"
            parsed.append({"text": line, "level": level})
        return {"lines": parsed, "total": len(all_lines)}
    except Exception as e:
        return {"lines": [], "total": 0, "error": str(e)}


@operator_app.get("/api/v1/status")
async def api_status() -> dict[str, Any]:
    """System status from status.json."""
    status_path = os.path.join(DATA_DIR, "status.json")
    if not os.path.exists(status_path):
        return {"running": False, "message": "No status file yet"}
    try:
        with open(status_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        return {"running": False, "error": str(e)}


# ---- Camera info cache (photo URLs from Trafikverket API) ----
_camera_info_cache: dict = {"data": {}, "ts": 0}
_CAMERA_INFO_TTL = 300  # 5 minutes
_CAMERA_INFO_FILE = os.path.join(DATA_DIR, "camera_info_cache.json")


def _get_camera_info() -> dict[str, dict]:
    """Fetch camera metadata from Trafikverket API with caching."""
    now = time.time()
    if _camera_info_cache["data"] and (now - _camera_info_cache["ts"]) < _CAMERA_INFO_TTL:
        return _camera_info_cache["data"]

    try:
        xml = f"""
        <REQUEST>
            <LOGIN authenticationkey="{API_KEY}" />
            <QUERY objecttype="Camera" schemaversion="1">
                <FILTER>
                    <EQ name="Active" value="true" />
                    <EQ name="HasFullSizePhoto" value="true" />
                </FILTER>
                <INCLUDE>Id</INCLUDE>
                <INCLUDE>Name</INCLUDE>
                <INCLUDE>Description</INCLUDE>
                <INCLUDE>PhotoUrl</INCLUDE>
            </QUERY>
        </REQUEST>
        """
        r = requests.post(API_URL, data=xml,
                          headers={"Content-Type": "text/xml"}, timeout=10)
        r.raise_for_status()
        result = r.json()
        cameras = result.get("RESPONSE", {}).get("RESULT", [{}])[0].get("Camera", [])

        info = {}
        for cam in cameras:
            cam_id = cam.get("Id", "")
            if cam_id in set(CAMERA_IDS):
                info[cam_id] = {
                    "name": cam.get("Name", ""),
                    "description": cam.get("Description", ""),
                    "photo_url": cam.get("PhotoUrl", ""),
                }

        _camera_info_cache["data"] = info
        _camera_info_cache["ts"] = now

        # Persist to disk
        try:
            os.makedirs(DATA_DIR, exist_ok=True)
            with open(_CAMERA_INFO_FILE, "w", encoding="utf-8") as f:
                json.dump(info, f, ensure_ascii=False)
        except Exception:
            pass

        return info
    except Exception:
        # Fall back to disk cache
        try:
            with open(_CAMERA_INFO_FILE, "r", encoding="utf-8") as f:
                info = json.load(f)
            _camera_info_cache["data"] = info
            _camera_info_cache["ts"] = now - _CAMERA_INFO_TTL + 60
            return info
        except Exception:
            return {}


@operator_app.get("/api/v1/camera-config")
async def api_camera_config() -> dict[str, Any]:
    """Serve ROI polygon config from camera_config.json."""
    config_path = os.path.join(os.path.dirname(__file__), "camera_config.json")
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {"cameras": {}}


@operator_app.get("/api/v1/camera-image/{camera_id}")
async def api_camera_image(camera_id: str):
    """Proxy live camera image from Trafikverket."""
    cam_info = _get_camera_info()
    info = cam_info.get(camera_id)
    if not info or not info.get("photo_url"):
        raise HTTPException(status_code=404, detail="Camera photo URL not found")

    photo_url = info["photo_url"]
    if "?" not in photo_url:
        photo_url += "?type=fullsize"

    try:
        resp = requests.get(photo_url, timeout=15)
        resp.raise_for_status()
        return Response(
            content=resp.content,
            media_type="image/jpeg",
            headers={"Cache-Control": "no-cache, max-age=0"},
        )
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Failed to fetch image: {e}")


# -- Lazy-loaded YOLO for detection overlay --
_vision_engine_singleton = None


def _get_vision_engine():
    global _vision_engine_singleton
    if _vision_engine_singleton is None:
        from src.vision_engine import VisionEngine
        _vision_engine_singleton = VisionEngine()
    return _vision_engine_singleton


@operator_app.get("/api/v1/camera-detections/{camera_id}")
async def api_camera_detections(camera_id: str) -> dict:
    """Run YOLO on a live camera image and return bounding boxes.

    Returns detections with ROI classification so the frontend can
    distinguish in-zone vehicles from total frame detections.
    """
    import cv2 as _cv2
    from src.roi_mapper import ROIMapper

    # Fetch image bytes
    cam_info = _get_camera_info()
    info = cam_info.get(camera_id)
    if not info or not info.get("photo_url"):
        raise HTTPException(status_code=404, detail="Camera not found")

    photo_url = info["photo_url"]
    if "?" not in photo_url:
        photo_url += "?type=fullsize"

    try:
        resp = requests.get(photo_url, timeout=15)
        resp.raise_for_status()
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Image fetch failed: {e}")

    # Decode to numpy array
    img_array = np.frombuffer(resp.content, dtype=np.uint8)
    frame = _cv2.imdecode(img_array, _cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=502, detail="Failed to decode image")

    # Run YOLO inference in a thread to avoid blocking the event loop
    def _run_yolo():
        engine = _get_vision_engine()
        return engine._detect_vehicles(frame, roi_polygon=None)

    detections = await asyncio.to_thread(_run_yolo)

    # Filter out detections inside exclusion zones (static false positives)
    config_path = os.path.join(os.path.dirname(__file__), "camera_config.json")
    exclusion_zones: list[list[int]] = []
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            cam_cfg = json.load(f).get("cameras", {}).get(camera_id, {})
            exclusion_zones = cam_cfg.get("exclusion_zones", [])
    except (FileNotFoundError, json.JSONDecodeError):
        pass

    def _in_exclusion_zone(det: dict) -> bool:
        x1, y1, x2, y2 = det["xyxy"]
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        for ez in exclusion_zones:
            if len(ez) == 4 and ez[0] <= cx <= ez[2] and ez[1] <= cy <= ez[3]:
                return True
        return False

    filtered = [d for d in detections if not _in_exclusion_zone(d)]

    # --- Classify detections into ROI segments ---
    roi_mapper = ROIMapper(config_path)
    segment_counts: dict[str, int] = {}
    det_results = []

    for d in filtered:
        x1, y1, x2, y2 = d["xyxy"]
        # Bottom-center = tire contact point (same convention as vision_engine)
        bx = (x1 + x2) / 2
        by = y2
        region = roi_mapper.classify_detection(camera_id, bx, by)
        in_roi = region is not None
        seg_name = region.road_id if region else None

        if seg_name:
            segment_counts[seg_name] = segment_counts.get(seg_name, 0) + 1

        det_results.append({
            "xyxy": d["xyxy"],
            "class_name": d["class_name"],
            "confidence": round(d["confidence"], 3),
            "in_roi": in_roi,
            "segment": seg_name,
        })

    in_roi_count = sum(1 for d in det_results if d["in_roi"])

    return {
        "detections": det_results,
        "total_count": len(filtered),
        "in_roi_count": in_roi_count,
        "segments": segment_counts,
        "excluded_count": len(detections) - len(filtered),
        "exclusion_zones": exclusion_zones,
        "image_width": frame.shape[1],
        "image_height": frame.shape[0],
        "timestamp": datetime.now().isoformat(),
    }


@operator_app.get("/api/v1/anomalies")
async def api_anomalies(
    limit: int = 100,
    camera_id: str | None = None,
) -> dict[str, Any]:
    """Return anomaly event log (most recent first)."""
    events = get_anomalies(DATA_DIR, limit=limit, camera_id=camera_id)
    total = get_total_count(DATA_DIR)
    return {
        "events": events,
        "total": total,
        "timestamp": _latest_timestamp,
    }


@operator_app.get("/api/v1/anomaly-image/{date}/{filename}")
async def api_anomaly_image(date: str, filename: str):
    """Serve a saved annotated anomaly image."""
    # Sanitize inputs
    if "/" in filename or ".." in filename or ".." in date:
        raise HTTPException(status_code=400, detail="Invalid path")
    path = os.path.join("storage", "anomalies", date, filename)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(
        path,
        media_type="image/jpeg",
        headers={"Cache-Control": "public, max-age=86400"},
    )


# ---------------------------------------------------------------------------
# Signal handling
# ---------------------------------------------------------------------------


def _handle_signal(sig, frame):
    logger.info(f"Received signal {sig}, shutting down...")
    _shutdown_event.set()


signal.signal(signal.SIGINT, _handle_signal)
signal.signal(signal.SIGTERM, _handle_signal)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PTRE — Unified operator API + tick loop"
    )
    parser.add_argument(
        "--once", action="store_true", help="Run one tick only then exit"
    )
    parser.add_argument(
        "--port", type=int, default=8081, help="API server port (default: 8081)"
    )
    parser.add_argument(
        "--host", type=str, default="0.0.0.0", help="API server host"
    )
    args = parser.parse_args()

    if not API_KEY:
        print("❌ Missing API key. Set TRAFIKVERKET_API_KEY in .env")
        sys.exit(1)

    uvicorn.run(
        operator_app,
        host=args.host,
        port=args.port,
        log_level="info",
    )
