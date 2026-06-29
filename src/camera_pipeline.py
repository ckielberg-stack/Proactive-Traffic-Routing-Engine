
"""Camera fetch, inference, retention, and vision-record processing."""

from __future__ import annotations

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from threading import Lock, local
from typing import Callable

import cv2
import numpy as np

from config import API_KEY, CAMERA_COORDS, DATA_DIR
from retention import RetentionPolicy
from src.anomaly_store import record_anomaly
from src.fusion_pipeline import _aggregate_multi_roi_capacity, _northbound_detections_for_persistence
from src.models import CameraMetadata, CapacityState
from src.roi_mapper import ROIMapper
from src.trafikverket_client import api_request, decode_frame, fetch_image_bytes
from src.vision_engine import VisionEngine

logger = logging.getLogger("mainloop")
_camera_worker_local = local()
_retention_lock = Lock()
_retention_policy: RetentionPolicy | None = None
_roi_mapper: ROIMapper | None = None


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
        _retention_policy = RetentionPolicy(base_dir=os.path.dirname(os.path.dirname(__file__)))
    return _retention_policy


def _get_roi_mapper() -> ROIMapper:
    global _roi_mapper
    if _roi_mapper is None:
        config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "camera_config.json")
        _roi_mapper = ROIMapper(config_path)
    return _roi_mapper


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
