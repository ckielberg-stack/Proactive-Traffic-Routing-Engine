
"""Persistence for tick results and dashboard status files."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime

from config import DATA_DIR, INTERVAL_SECONDS
from src.anomaly_store import get_total_count
from src.models import TickResult

logger = logging.getLogger("mainloop")
_start_time: str | None = None
_tick_count = 0


def set_status_context(*, start_time: str | None, tick_count: int) -> None:
    global _start_time, _tick_count
    _start_time = start_time
    _tick_count = tick_count


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

    # Weather and road-condition records keep the JSONL shape used by dashboard/API consumers.
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
            "residual_correction_enabled": qp.residual_correction_enabled,
            "residual_correction_minutes": qp.residual_correction_minutes,
            "residual_sample_count": qp.residual_sample_count,
            "residual_bucket": qp.residual_bucket,
            "residual_confidence": qp.residual_confidence,
            "residual_disabled_reason": qp.residual_disabled_reason,
            "base_eta_minutes_by_target": qp.base_eta_minutes_by_target,
            "corrected_eta_minutes_by_target": qp.corrected_eta_minutes_by_target,
            "corrected_eta_lower_minutes_by_target": (
                qp.corrected_eta_lower_minutes_by_target
            ),
            "corrected_eta_upper_minutes_by_target": (
                qp.corrected_eta_upper_minutes_by_target
            ),
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
            "base_eta_minutes": rec.base_eta_minutes,
            "corrected_eta_minutes": rec.corrected_eta_minutes,
            "eta_lower_minutes": rec.eta_lower_minutes,
            "eta_upper_minutes": rec.eta_upper_minutes,
            "confidence": rec.confidence,
            "uncertainty_level": rec.uncertainty_level,
            "residual_correction_enabled": rec.residual_correction_enabled,
            "residual_correction_minutes": rec.residual_correction_minutes,
            "residual_sample_count": rec.residual_sample_count,
            "residual_bucket": rec.residual_bucket,
            "residual_confidence": rec.residual_confidence,
            "residual_disabled_reason": rec.residual_disabled_reason,
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
            "residual_pending_count": cal.residual_pending_count,
            "residual_bucket_count": cal.residual_bucket_count,
            "residual_min_samples": cal.residual_min_samples,
            "residual_max_correction_minutes": cal.residual_max_correction_minutes,
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
            "forecast_state": adj.forecast_state,
            "forecast_lead_minutes": adj.forecast_lead_minutes,
            "proactive_halka": adj.proactive_halka,
        })

    if result.weather_forecast:
        fc = result.weather_forecast
        all_records.append({
            "type": "weather_forecast",
            "timestamp": now.isoformat(),
            "source": "smhi_metfcst",
            "surface_state": fc.surface_state,
            "onset_minutes": fc.onset_minutes,
            "confidence": fc.confidence,
            "reason": fc.reason,
            "reference_time": (
                fc.reference_time.isoformat() if fc.reference_time else None
            ),
            "valid_until": fc.valid_until.isoformat() if fc.valid_until else None,
            "lookahead_minutes": fc.lookahead_minutes,
            "sample_count": fc.sample_count,
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
            "forecast_surface_state": (
                result.weather_forecast.surface_state
                if result.weather_forecast
                else "unknown"
            ),
            "forecast_onset_minutes": (
                result.weather_forecast.onset_minutes
                if result.weather_forecast
                else None
            ),
            "proactive_halka": (
                result.weather_adjustment.proactive_halka
                if result.weather_adjustment
                else False
            ),
        },
    }

    status_path = os.path.join(DATA_DIR, "status.json")
    try:
        with open(status_path, "w", encoding="utf-8") as f:
            json.dump(status, f, ensure_ascii=False, default=str, indent=2)
    except Exception as e:
        logger.error(f"Could not write status.json: {e}")
