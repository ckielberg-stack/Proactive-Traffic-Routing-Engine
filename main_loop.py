#!/usr/bin/env python3
"""CLI wrapper and compatibility shims for the PTRE tick loop.

TRAFIK-022 transitional note: hot-path implementation now lives in focused
``src`` modules. Re-exported helpers below keep older tests and ad hoc scripts
working while callers migrate to the owning modules.
"""

from __future__ import annotations

import argparse
import json
import logging
import logging.handlers
import os
import signal
import sys
import time

from dotenv import load_dotenv

from config import API_KEY, CAMERA_COORDS, CAMERA_IDS, DATA_DIR, INTERVAL_SECONDS, SENSOR_COORDS
from src import camera_pipeline as _camera_pipeline
from src import fusion_pipeline as _fusion_pipeline
from src import tick_orchestrator as _tick_orchestrator
from src import tick_persistence as _tick_persistence
from src import trafikverket_client as _trafikverket_client
from src import trafikverket_sources as _trafikverket_sources
from src.camera_pipeline import (
    _camera_failure_record,
    _camera_worker_count,
    _draw_annotated_frame,
    _get_camera_worker_vision_engine,
    _get_retention_policy,
    _get_roi_mapper,
    _process_camera,
    _save_annotated_image,
    fetch_cameras,
)
from src.fusion_pipeline import (
    _aggregate_multi_roi_capacity,
    _apply_fused_capacity_derivation,
    _apply_stopped_vehicle_detection,
    _derive_capacity_from_fused_state,
    _find_nearest_camera,
    _northbound_detections_for_persistence,
    _traffic_direction_from_road_id,
    apply_situation_capacity_impacts,
    build_camera_chainage_map,
    build_node_inflows,
    build_node_traffic_states,
    build_travel_time_speed_states,
    detect_sensor_anomalies,
)
from src.trafikverket_client import api_request, decode_frame, fetch_image_bytes
from src.tick_orchestrator import _get_vms_orchestrator
from src.trafikverket_sources import (
    _situation_capacity_factor,
    fetch_road_conditions,
    fetch_sensor_data,
    fetch_situation_deviations,
    fetch_smhi_forecast,
    fetch_travel_times,
    fetch_vms_status,
    fetch_weather_data,
    get_deviation_wgs84,
    in_bbox,
    parse_point_wgs84,
    project_e4_northbound_chainage,
)
from src.traffic_constants import FREE_FLOW_SPEED_KMH, K_CRITICAL_VEH_KM_LANE, Q_CAP_VPH_PER_LANE

load_dotenv()

LOG_DATE_FORMAT = "%H:%M:%S"

logger = logging.getLogger("mainloop")
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt=LOG_DATE_FORMAT))
    logger.addHandler(ch)


def setup_file_logger(data_dir: str) -> None:
    os.makedirs(data_dir, exist_ok=True)
    fh = logging.handlers.RotatingFileHandler(
        os.path.join(data_dir, "mainloop.log"),
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt=LOG_DATE_FORMAT))
    logger.addHandler(fh)


_shutdown = False
_tick_count = 0
_start_time: str | None = None
_camera_chainage_map: dict[str, float] | None = None
_vision_engine = None
_retention_policy = None
_roi_mapper = None
_physics_engine = None
_vms_orchestrator = None
_density_smoother = None
_track_persistence = None
_travel_time_calibrator = None
_weather_adapter = None
_smhi_forecast_source = None
_build_camera_chainage_map = build_camera_chainage_map
_inside_tick = False


def _signal_handler(sig, frame):
    global _shutdown
    _shutdown = True
    logger.info("Avslutningssignal mottagen - stänger efter denna tick")


signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


def _sync_compat_state_to_modules() -> None:
    """Apply transitional main_loop monkeypatches to the new owning modules."""
    _trafikverket_client.api_request = api_request
    _trafikverket_client.fetch_image_bytes = fetch_image_bytes
    _trafikverket_client.decode_frame = decode_frame

    _camera_pipeline.api_request = api_request
    _camera_pipeline.fetch_image_bytes = fetch_image_bytes
    _camera_pipeline.decode_frame = decode_frame
    _camera_pipeline._get_camera_worker_vision_engine = _get_camera_worker_vision_engine
    _camera_pipeline._get_retention_policy = _get_retention_policy
    _camera_pipeline._get_roi_mapper = _get_roi_mapper
    _camera_pipeline._camera_worker_count = _camera_worker_count
    _camera_pipeline.DATA_DIR = DATA_DIR
    _camera_pipeline.CAMERA_COORDS = CAMERA_COORDS

    _trafikverket_sources.api_request = api_request
    _trafikverket_sources._find_nearest_camera = _find_nearest_camera
    _trafikverket_sources._smhi_forecast_source = _smhi_forecast_source
    _trafikverket_sources._vms_orchestrator = _vms_orchestrator

    _fusion_pipeline.DATA_DIR = DATA_DIR
    _fusion_pipeline.CAMERA_COORDS = CAMERA_COORDS
    _fusion_pipeline.SENSOR_COORDS = SENSOR_COORDS
    _fusion_pipeline._track_persistence = _track_persistence
    _tick_persistence.DATA_DIR = DATA_DIR

    _tick_orchestrator._tick_count = _tick_count
    _tick_orchestrator._start_time = _start_time
    _tick_orchestrator._camera_chainage_map = _camera_chainage_map
    _tick_orchestrator._physics_engine = _physics_engine
    _tick_orchestrator._vms_orchestrator = _vms_orchestrator
    _tick_orchestrator._density_smoother = _density_smoother
    _tick_orchestrator._travel_time_calibrator = _travel_time_calibrator
    _tick_orchestrator._weather_adapter = _weather_adapter
    _tick_orchestrator.fetch_cameras = fetch_cameras
    _tick_orchestrator.fetch_sensor_data = fetch_sensor_data
    _tick_orchestrator.fetch_vms_status = fetch_vms_status
    _tick_orchestrator.fetch_travel_times = fetch_travel_times
    _tick_orchestrator.fetch_weather_data = fetch_weather_data
    _tick_orchestrator.fetch_road_conditions = fetch_road_conditions
    _tick_orchestrator.fetch_situation_deviations = fetch_situation_deviations
    _tick_orchestrator.fetch_smhi_forecast = fetch_smhi_forecast
    _tick_orchestrator.CAMERA_COORDS = CAMERA_COORDS
    _tick_orchestrator.build_camera_chainage_map = _build_camera_chainage_map
    _tick_orchestrator._build_camera_chainage_map = _build_camera_chainage_map
    _tick_orchestrator._get_vms_orchestrator = _get_vms_orchestrator


def _sync_compat_state_from_modules() -> None:
    global _tick_count, _start_time, _camera_chainage_map
    global _physics_engine, _vms_orchestrator, _density_smoother
    global _track_persistence, _travel_time_calibrator, _weather_adapter
    global _smhi_forecast_source

    _tick_count = _tick_orchestrator._tick_count
    _start_time = _tick_orchestrator._start_time
    _camera_chainage_map = _tick_orchestrator._camera_chainage_map
    _physics_engine = _tick_orchestrator._physics_engine
    _vms_orchestrator = _tick_orchestrator._vms_orchestrator
    _density_smoother = _tick_orchestrator._density_smoother
    _track_persistence = _fusion_pipeline._track_persistence
    _travel_time_calibrator = _tick_orchestrator._travel_time_calibrator
    _weather_adapter = _tick_orchestrator._weather_adapter
    _smhi_forecast_source = _trafikverket_sources._smhi_forecast_source


def fetch_cameras(camera_ids: list[str], now):
    if _inside_tick:
        return _camera_pipeline.fetch_cameras(camera_ids, now)
    _sync_compat_state_to_modules()
    result = _camera_pipeline.fetch_cameras(camera_ids, now)
    _sync_compat_state_from_modules()
    return result


def fetch_situation_deviations(now):
    if _inside_tick:
        return _trafikverket_sources.fetch_situation_deviations(now)
    _sync_compat_state_to_modules()
    return _trafikverket_sources.fetch_situation_deviations(now)


def fetch_vms_status(now):
    if _inside_tick:
        return _trafikverket_sources.fetch_vms_status(now)
    _sync_compat_state_to_modules()
    return _trafikverket_sources.fetch_vms_status(now)


def tick_once(camera_ids: list[str]):
    global _inside_tick
    _sync_compat_state_to_modules()
    _inside_tick = True
    try:
        result = _tick_orchestrator.tick_once(camera_ids)
    finally:
        _inside_tick = False
        _sync_compat_state_from_modules()
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Tick-based main loop for PTRE - discrete 60s polling"
    )
    parser.add_argument("--once", action="store_true", help="Run one tick only")
    parser.add_argument("--interval", type=int, default=INTERVAL_SECONDS, help="Seconds between ticks")
    args = parser.parse_args()

    if not API_KEY:
        print("Missing API key. Set TRAFIKVERKET_API_KEY in .env")
        sys.exit(1)

    setup_file_logger(DATA_DIR)
    camera_ids = list(CAMERA_IDS)
    excluded_file = os.path.join(DATA_DIR, "excluded_cameras.json")
    try:
        with open(excluded_file, "r", encoding="utf-8") as f:
            excluded = set(json.load(f))
        camera_ids = [c for c in camera_ids if c not in excluded]
    except (FileNotFoundError, json.JSONDecodeError):
        pass

    logger.info("Monitoring %s cameras", len(camera_ids))
    logger.info("Data dir: %s", DATA_DIR)
    logger.info("Interval: %ss (tick-based architecture)", args.interval)
    logger.info("Physics engine: LWR kinematic wave model")
    logger.info("VMS orchestrator: predictive queue tail modeling")

    if args.once:
        logger.info("Running one tick (--once)")
        tick_once(camera_ids)
        return

    logger.info("Starting tick-based main loop... (Ctrl+C to stop)")
    while not _shutdown:
        try:
            with open(excluded_file, "r", encoding="utf-8") as f:
                excluded = set(json.load(f))
            camera_ids = [c for c in CAMERA_IDS if c not in excluded]
        except (FileNotFoundError, json.JSONDecodeError):
            camera_ids = list(CAMERA_IDS)

        try:
            tick_once(camera_ids)
        except Exception as e:
            logger.error("Tick error: %s", e, exc_info=True)

        for _ in range(args.interval):
            if _shutdown:
                break
            time.sleep(1)

    logger.info("Main loop stopped. Goodbye!")


if __name__ == "__main__":
    main()
