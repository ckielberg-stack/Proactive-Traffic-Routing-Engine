
"""Thin tick orchestration for the PTRE hot path."""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

from config import CAMERA_COORDS
from src.camera_pipeline import fetch_cameras
from src.density_smoother import DensitySmoother
from src.fusion_pipeline import (
    _apply_fused_capacity_derivation,
    _apply_stopped_vehicle_detection,
    build_camera_chainage_map,
    build_node_traffic_states,
    apply_situation_capacity_impacts,
    detect_sensor_anomalies,
)
from src.models import CalibrationSnapshot, CapacityState, SensorReading, TickResult, VMSStatusSnapshot, TravelTimeReading, SituationDeviation
from src.physics_engine import PhysicsEngine
from src.smhi_forecast import WeatherForecast
from src.tick_persistence import _persist_tick, set_status_context
from src.traffic_constants import FREE_FLOW_SPEED_KMH, K_CRITICAL_VEH_KM_LANE
from src.travel_time_calibrator import TravelTimeCalibrator
from src.trafikverket_sources import (
    fetch_road_conditions,
    fetch_sensor_data,
    fetch_situation_deviations,
    fetch_smhi_forecast,
    fetch_travel_times,
    fetch_vms_status,
    fetch_weather_data,
)
from src.vms_orchestrator import VMSOrchestrator
from src.weather_adapter import WeatherAdapter

logger = logging.getLogger("mainloop")
_tick_count = 0
_start_time: str | None = None
_camera_chainage_map: dict[str, float] | None = None
_physics_engine: PhysicsEngine | None = None
_vms_orchestrator: VMSOrchestrator | None = None
_density_smoother: DensitySmoother | None = None
_travel_time_calibrator: TravelTimeCalibrator | None = None
_weather_adapter: WeatherAdapter | None = None
_build_camera_chainage_map = build_camera_chainage_map


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
    weather_forecast: WeatherForecast | None = None
    situation_deviations: list[SituationDeviation] = []

    fetch_started = time.monotonic()
    with ThreadPoolExecutor(max_workers=8) as executor:
        future_cameras = executor.submit(fetch_cameras, camera_ids, now)
        future_sensors = executor.submit(fetch_sensor_data, now)
        future_vms = executor.submit(fetch_vms_status, now)
        future_tt = executor.submit(fetch_travel_times, now)
        future_weather = executor.submit(fetch_weather_data, now)
        future_road_conditions = executor.submit(fetch_road_conditions, now)
        future_situations = executor.submit(fetch_situation_deviations, now)
        future_forecast = executor.submit(fetch_smhi_forecast, now)

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

        try:
            weather_forecast = future_forecast.result(timeout=12)
        except Exception as e:
            logger.error(f"SMHI forecast fetch failed: {e}", exc_info=True)
    fetch_duration_ms = int((time.monotonic() - fetch_started) * 1000)
    logger.info("⏱️  Data fetch phase completed in %sms", fetch_duration_ms)

    weather_adjustment = _get_weather_adapter().compute(
        weather_records=weather_records,
        road_condition_records=road_condition_records,
        now=now,
        forecast=weather_forecast,
    )
    logger.info(
        "🌦  Surface adjustment: %s free_flow=%.2f capacity=%.2f confidence=%s (%s)",
        weather_adjustment.surface_state,
        weather_adjustment.free_flow_factor,
        weather_adjustment.capacity_factor,
        weather_adjustment.confidence,
        weather_adjustment.reason,
    )
    if weather_adjustment.proactive_halka:
        logger.info(
            "🔮 Proactive HALKA pre-stage: %s forecast escalates surface ahead of "
            "observation (onset %s)",
            weather_adjustment.forecast_state,
            (
                f"~{weather_adjustment.forecast_lead_minutes:.0f}min"
                if weather_adjustment.forecast_lead_minutes is not None
                else "n/a"
            ),
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
    if travel_time_readings:
        calibrator.apply_residual_corrections(
            readings=travel_time_readings,
            predictions=queue_predictions,
            now=now,
        )
        if calibration_snapshot:
            residual_state = calibrator.get_state()
            calibration_snapshot.residual_pending_count = int(
                residual_state["residual_pending_count"]
            )
            calibration_snapshot.residual_bucket_count = int(
                residual_state["residual_bucket_count"]
            )
            calibration_snapshot.residual_min_samples = int(
                residual_state["residual_min_samples"]
            )
            calibration_snapshot.residual_max_correction_minutes = float(
                residual_state["residual_max_correction_minutes"]
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
        weather_adjustment=weather_adjustment,
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
        weather_forecast=weather_forecast,
        situation_deviations=situation_deviations,
    )

    set_status_context(start_time=_start_time, tick_count=_tick_count)
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
