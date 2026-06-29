
"""Traffic fusion helpers used by tick orchestration."""

from __future__ import annotations

import logging
from datetime import datetime

import numpy as np

from config import (
    CAMERA_COORDS,
    DATA_DIR,
    DEFAULT_ROAD_SPEED_LIMIT,
    E4_NORTHBOUND_CORRIDOR_LENGTH_KM,
    E4_NORTHBOUND_ROUTE_POINTS,
    E4_NORTHBOUND_TRAVEL_TIME_ROUTE_IDS,
    SENSOR_COORDS,
    SENSOR_ROAD_SPEED_LIMITS,
    SENSOR_SEVERE_DROP_RATIO,
    SENSOR_SPEED_DROP_RATIO,
)
from src.anomaly_store import record_anomaly
from src.models import (
    CameraMetadata,
    CapacityState,
    MultiSegmentCapacity,
    SensorAnomaly,
    SensorReading,
    SegmentTrafficState,
    SituationDeviation,
    TravelTimeReading,
)
from src.route_chainage import build_route_chainage_map, find_nearest_by_chainage
from src.track_persistence import TrackPersistence
from src.traffic_constants import FREE_FLOW_SPEED_KMH, K_CRITICAL_VEH_KM_LANE, Q_CAP_VPH_PER_LANE

logger = logging.getLogger("mainloop")
_track_persistence: TrackPersistence | None = None


def _get_track_persistence() -> TrackPersistence:
    global _track_persistence
    if _track_persistence is None:
        _track_persistence = TrackPersistence(free_flow_speed_kmh=FREE_FLOW_SPEED_KMH)
    return _track_persistence


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


# Backward-compatible alias for older callers/tests.
_build_camera_chainage_map = build_camera_chainage_map
