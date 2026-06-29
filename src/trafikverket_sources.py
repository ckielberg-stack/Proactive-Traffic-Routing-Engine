
"""Trafikverket data-source fetchers and parsers for the tick pipeline."""

from __future__ import annotations

import logging
import re
from datetime import datetime

from config import (
    API_KEY,
    BBOX,
    E4_NORTHBOUND_CORRIDOR_LENGTH_KM,
    E4_NORTHBOUND_ROUTE_POINTS,
    E4_TRAVEL_TIME_ROUTE_IDS,
    SMHI_FORECAST_ENABLED,
    SMHI_FORECAST_LOOKAHEAD_MINUTES,
    SMHI_FORECAST_POLL_INTERVAL_MINUTES,
    SMHI_FORECAST_REFERENCE_POINT,
)
from src.fusion_pipeline import _find_nearest_camera
from src.models import SensorReading, SituationDeviation, TravelTimeReading, VMSStatusSnapshot
from src.route_chainage import RouteProjector
from src.smhi_forecast import SMHIForecastSource, WeatherForecast
from src.trafikverket_client import _nested, _val, api_request
from src.vms_orchestrator import VMSOrchestrator

logger = logging.getLogger("mainloop")
_smhi_forecast_source: SMHIForecastSource | None = None
_vms_orchestrator: VMSOrchestrator | None = None


def _get_smhi_forecast_source() -> SMHIForecastSource:
    """Poll-throttled SMHI forecast source — caches across ticks (~30 min)."""
    global _smhi_forecast_source
    if _smhi_forecast_source is None:
        lat, lon = SMHI_FORECAST_REFERENCE_POINT
        _smhi_forecast_source = SMHIForecastSource(
            lat=lat,
            lon=lon,
            poll_interval_minutes=SMHI_FORECAST_POLL_INTERVAL_MINUTES,
            lookahead_minutes=SMHI_FORECAST_LOOKAHEAD_MINUTES,
        )
    return _smhi_forecast_source


def _get_vms_orchestrator() -> VMSOrchestrator:
    global _vms_orchestrator
    if _vms_orchestrator is None:
        _vms_orchestrator = VMSOrchestrator()
    return _vms_orchestrator


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


def fetch_smhi_forecast(now: datetime) -> WeatherForecast | None:
    """Return the cached SMHI corridor forecast, refreshing every ~30 min.

    Fail-safe: any error yields ``None`` (or the last good forecast cached by
    the source), so the WeatherAdapter simply falls back to observed-only
    behaviour and never loses conservatism.
    """
    if not SMHI_FORECAST_ENABLED:
        return None
    try:
        forecast = _get_smhi_forecast_source().get_forecast(now)
    except Exception as e:  # pragma: no cover - defensive, get_forecast is fail-safe
        logger.error(f"SMHI forecast fetch failed: {e}", exc_info=True)
        return None
    if forecast is not None:
        logger.info(
            "🔭 SMHI forecast: %s within %smin (onset %s, %s)",
            forecast.surface_state,
            forecast.lookahead_minutes,
            (
                f"~{forecast.onset_minutes:.0f}min"
                if forecast.onset_minutes is not None
                else "n/a"
            ),
            forecast.reason,
        )
    return forecast


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
