"""
Pydantic domain models for the Proactive Traffic Routing Engine (PTRE).

These models define the data contracts passed between the Vision, Physics,
and Routing modules.
"""

from datetime import datetime

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Input models
# ---------------------------------------------------------------------------


class SensorReading(BaseModel):
    """Upstream radar / loop-detector sensor data."""

    timestamp: datetime
    site_id: int | None = Field(
        default=None, description="TrafficFlow SiteId (for per-node mapping)"
    )
    inflow_volume_vph: float = Field(
        ge=0, description="Vehicles per hour measured upstream of the camera"
    )
    average_speed_kmh: float = Field(
        ge=0, description="Mean speed (km/h) measured upstream of the camera"
    )


class CameraMetadata(BaseModel):
    """Static metadata mapping a camera to the road network."""

    camera_id: str
    name: str
    lat: float
    lng: float
    num_lanes: int = Field(default=2, ge=1)
    road: str = "E4"


# ---------------------------------------------------------------------------
# Output models
# ---------------------------------------------------------------------------


class CapacityState(BaseModel):
    """Output of the vision engine for one analysed frame.

    Downstream consumers (physics engine, geo-mapper) use this to compute
    shockwave propagation and routing penalties.
    """

    timestamp: datetime
    camera_id: str
    vehicle_count: int = Field(ge=0, description="Vehicles detected in ROI")
    blocked_lanes: int = Field(ge=0, description="Lanes blocked by anomaly")
    total_lanes: int = Field(ge=1, description="Total drivable lanes")
    estimated_capacity_vph: float = Field(
        ge=0, description="Estimated throughput at the bottleneck"
    )
    observed_density_veh_km_lane: float = Field(
        ge=0.0,
        default=0.0,
        description="Visual density in veh/km/lane from YOLO detections",
    )
    road_id: str | None = Field(
        default=None,
        description="Road/corridor identifier used to derive this camera-level state",
    )
    traffic_direction: str | None = Field(
        default=None,
        description="Traffic direction for physics iteration, e.g. northbound/southbound",
    )
    is_anomaly: bool = False
    anomaly_reason: str | None = None
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        default=0.0,
        description="Model confidence (mean of detection confidences)",
    )


class RoadSegmentState(BaseModel):
    """Per-road-segment capacity output from multi-ROI analysis."""

    road_id: str
    direction: str = Field(description="'towards' or 'away' relative to camera")
    vehicle_count: int = Field(ge=0)
    capacity_vph: float = Field(ge=0)
    observed_density_veh_km_lane: float = Field(
        ge=0.0,
        default=0.0,
        description="Visual density in veh/km/lane for this road segment",
    )
    num_lanes: int = Field(ge=1, default=2)
    is_anomaly: bool = False
    anomaly_reason: str | None = None
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        default=0.0,
        description="Mean detection confidence for this segment",
    )


class MultiSegmentCapacity(BaseModel):
    """Aggregated multi-ROI output for one camera frame.

    When a camera has ROI definitions, the vision engine produces one
    ``RoadSegmentState`` per defined region.  Detections outside all
    ROIs are counted in ``unmatched_detections`` and discarded from
    capacity calculations.
    """

    timestamp: datetime
    camera_id: str
    segments: list[RoadSegmentState]
    unmatched_detections: int = Field(
        ge=0,
        default=0,
        description="Detections outside all ROI polygons (discarded)",
    )


class SegmentTrafficState(BaseModel):
    """Per-camera traffic state used by the piecewise physics model."""

    local_inflow_vph: float | None = Field(
        default=None,
        ge=0,
        description="Measured or fallback inflow at the camera node",
    )
    local_speed_kmh: float | None = Field(
        default=None,
        ge=0,
        description="Measured or fallback upstream speed at the camera node",
    )
    inflow_source: str = Field(
        default="missing",
        description="Source of local_inflow_vph, e.g. traffic_flow or aggregate",
    )
    speed_source: str = Field(
        default="missing",
        description="Source of local_speed_kmh, e.g. traffic_flow or travel_time",
    )
    confidence: str = Field(
        default="low",
        description="'high', 'medium', or 'low' based on data locality",
    )


# ---------------------------------------------------------------------------
# Phase 3: Physics Engine output
# ---------------------------------------------------------------------------


class SegmentSpeed(BaseModel):
    """Per-segment shockwave speed in the piecewise LWR model.

    Each segment represents the stretch between two adjacent camera nodes.
    The wave speed varies per segment because local inflow differs (due to
    on-ramps, off-ramps, and merging traffic).
    """

    from_camera: str = Field(description="Downstream camera (closer to bottleneck)")
    to_camera: str = Field(description="Upstream camera (farther from bottleneck)")
    distance_km: float = Field(ge=0, description="Segment length in km")
    wave_speed_kmh: float = Field(
        description="LWR shockwave speed for this segment (km/h)"
    )
    local_inflow_vph: float = Field(
        ge=0, description="Measured inflow at the upstream end of this segment"
    )
    local_speed_kmh: float = Field(
        ge=0, description="Speed used for the upstream end of this segment"
    )
    inflow_source: str = Field(
        default="unknown", description="Source of local_inflow_vph"
    )
    speed_source: str = Field(
        default="unknown", description="Source of local_speed_kmh"
    )


class QueuePrediction(BaseModel):
    """Output of the Shockwave Prediction Engine (Phase 3).

    Describes the backward-propagating queue from a bottleneck.
    ``lengths_at_minutes`` maps future time offsets (minutes) to predicted
    queue length in kilometres.

    The piecewise model populates ``segment_speeds`` with per-segment
    wave speed data.  ``growth_speed_kmh`` is retained as a distance-
    weighted average for backward compatibility.
    """

    timestamp: datetime
    camera_id: str
    origin_lat: float = Field(description="Latitude of the bottleneck origin")
    origin_lng: float = Field(description="Longitude of the bottleneck origin")
    origin_chainage_km: float = Field(
        description="Linear reference (km) along the highway from a fixed datum"
    )
    growth_speed_kmh: float = Field(
        description="Distance-weighted average wave speed upstream (km/h)"
    )
    segment_speeds: list[SegmentSpeed] = Field(
        default_factory=list,
        description="Per-segment wave speeds from bottleneck upstream",
    )
    lengths_at_minutes: dict[int, float] = Field(
        description="Mapping of T+N minutes → predicted queue length in km"
    )
    local_data_segments: int = Field(
        ge=0,
        default=0,
        description="Segments using local TrafficFlow inflow and speed",
    )
    fallback_data_segments: int = Field(
        ge=0,
        default=0,
        description="Segments using at least one fallback data source",
    )
    missing_data_segments: int = Field(
        ge=0,
        default=0,
        description="Segments where missing data halted or degraded physics",
    )
    data_confidence: str = Field(
        default="low",
        description="'high', 'medium', or 'low' based on segment data locality",
    )


# ---------------------------------------------------------------------------
# VMS Status Polling (ground-truth log)
# ---------------------------------------------------------------------------


class VMSStatusSnapshot(BaseModel):
    """Polled VMS sign status — records what the human operator has set.

    These snapshots are persisted to JSONL so we can build a historical
    ground-truth log of when operators actually activated VMS signs.
    """

    timestamp: datetime
    vms_id: str
    vms_name: str
    is_active: bool = Field(
        description="True if the sign is currently displaying a message"
    )
    displayed_message: str | None = Field(
        default=None,
        description="Current text on the sign, e.g. 'KÖVARNING 70'",
    )
    speed_limit: int | None = Field(
        default=None,
        description="Advisory or mandatory speed limit shown (km/h)",
    )
    road_number: str | None = Field(
        default=None,
        description="Road number from Situation.Deviation.RoadNumber, e.g. 'E4'",
    )
    geometry_wgs84: str | None = Field(
        default=None,
        description="Raw Situation.Deviation.Geometry.WGS84 geometry",
    )
    lat: float | None = Field(
        default=None,
        description="Latitude parsed from Geometry.WGS84 when available",
    )
    lng: float | None = Field(
        default=None,
        description="Longitude parsed from Geometry.WGS84 when available",
    )
    chainage_km: float | None = Field(
        default=None,
        description="Projected route-linear chainage in km when geometry is available",
    )


# ---------------------------------------------------------------------------
# TravelTimeRoute (measured corridor travel times)
# ---------------------------------------------------------------------------


class TravelTimeReading(BaseModel):
    """Measured travel time from Trafikverket's TravelTimeRoute API.

    Each reading represents a road segment with Bluetooth/ANPR-measured
    actual A→B travel time vs. free-flow baseline.
    """

    timestamp: datetime
    route_id: str = Field(description="Trafikverket route segment ID")
    name: str = Field(
        description="Route name, e.g. 'E4/E20 N Fittja (147) - Vårby (148)'"
    )
    travel_time_seconds: float = Field(
        ge=0, description="Current measured travel time (seconds)"
    )
    free_flow_seconds: float = Field(
        ge=0, description="Free-flow baseline travel time (seconds)"
    )
    speed_kmh: float = Field(
        ge=0, description="Current average speed on segment (km/h)"
    )
    length_meters: float = Field(
        ge=0, description="Segment length in metres"
    )
    traffic_status: str = Field(
        description="'freeflow', 'slow', 'heavy', or 'unknown'"
    )
    delay_seconds: float = Field(
        description="Delay vs free-flow (positive = slower than normal)"
    )


# ---------------------------------------------------------------------------
# Phase 5: VMS Orchestrator models
# ---------------------------------------------------------------------------


class VMSGantry(BaseModel):
    """Static configuration for a single Variable Message Sign gantry."""

    vms_id: str
    name: str
    lat: float
    lng: float
    road: str = "E4"
    direction: str = Field(
        default="northbound",
        description="Traffic direction this VMS serves",
    )
    chainage_km: float = Field(
        description="Linear reference (km) along the highway from a fixed datum"
    )


class VMSRecommendation(BaseModel):
    """A recommendation to activate a specific VMS gantry."""

    timestamp: datetime
    vms_id: str
    vms_name: str
    recommended_message: str = Field(
        description="Message text, e.g. 'KÖVARNING 70 km/h'"
    )
    urgency: str = Field(
        description="'immediate', 'soon', or 'advisory'",
    )
    queue_growth_speed_kmh: float
    distance_queue_tail_to_vms_km: float = Field(
        description="Current distance from predicted queue tail to this VMS"
    )
    estimated_activation_minutes: float = Field(
        description="Minutes until queue tail reaches this VMS position"
    )
    triggering_camera_id: str
    current_vms_status: str | None = Field(
        default=None,
        description="Current real-world sign state, e.g. 'OFF', '70 km/h'",
    )
    summary: str = Field(
        description="Human-readable narrative for the operator"
    )


# ---------------------------------------------------------------------------
# Phase 6: Operator API models
# ---------------------------------------------------------------------------


class IncidentReport(BaseModel):
    """AI-verified incident report for operator decision support."""

    timestamp: datetime
    camera_id: str
    incident_type: str = Field(
        description="e.g. 'vehicle_stopped', 'accident', 'debris', 'congestion'"
    )
    lanes_affected: int = Field(ge=0)
    total_lanes: int = Field(ge=1)
    capacity_drop_percentage: float = Field(
        ge=0.0,
        le=100.0,
        description="Percentage drop in capacity vs. free-flow",
    )
    thumbnail_base64: str | None = Field(
        default=None,
        description="Base64-encoded JPEG with YOLO bounding boxes drawn",
    )
    confidence: float = Field(ge=0.0, le=1.0, default=0.0)
    lat: float | None = None
    lng: float | None = None


# ---------------------------------------------------------------------------
# Physics calibration snapshot
# ---------------------------------------------------------------------------


class CalibrationSnapshot(BaseModel):
    """Snapshot of the TravelTime-based physics calibration state.

    Tracks how the measured corridor free-flow speed compares to the
    physics engine's assumption, and scores prediction accuracy.
    """

    adapted_free_flow_speed: float = Field(
        description="EMA-smoothed free-flow speed from measurements (km/h)"
    )
    correction_factor: float = Field(
        description="Ratio: adapted / model free_flow_speed"
    )
    measured_free_flow_speed: float | None = Field(
        default=None,
        description="Raw weighted-average speed this tick (km/h)",
    )
    freeflow_segment_count: int = Field(
        ge=0, description="Number of freeflow TravelTimeRoute segments"
    )
    congested_segment_count: int = Field(
        ge=0, description="Number of congested TravelTimeRoute segments"
    )
    accuracy_hit_rate: float | None = Field(
        default=None,
        description="Fraction of congested segments with matching predictions",
    )
    confidence: str = Field(
        description="'high', 'medium', or 'low' based on segment count"
    )


# ---------------------------------------------------------------------------
# Sensor-based anomaly detection
# ---------------------------------------------------------------------------


class SensorAnomaly(BaseModel):
    """Anomaly detected from sensor speed data (no camera required).

    Generated when a sensor station reports a speed significantly below
    the road's posted speed limit.  This closes the detection gap for
    congestion events that are invisible to the camera-based pipeline
    (e.g. night, fog, stations without mapped cameras).
    """

    timestamp: datetime
    site_id: int = Field(description="TrafficFlow SiteId")
    measured_speed_kmh: float = Field(
        ge=0, description="Actual measured speed at the station"
    )
    road_speed_limit_kmh: int = Field(
        description="Posted speed limit for this road segment"
    )
    speed_ratio: float = Field(
        description="measured / limit, e.g. 0.50 = 50% of posted limit"
    )
    volume_vph: float = Field(ge=0, description="Traffic volume at station")
    severity: str = Field(
        description="'warning' (speed < 50% of limit) or 'severe' (< 35%)"
    )
    nearest_camera_id: str | None = Field(
        default=None,
        description="Closest mapped camera for cross-referencing",
    )
    lat: float = Field(description="Station latitude")
    lng: float = Field(description="Station longitude")


# ---------------------------------------------------------------------------
# Tick-based orchestration output
# ---------------------------------------------------------------------------


class TickResult(BaseModel):
    """Aggregated output of one 60-second tick cycle.

    Serves as the data contract between the main loop and downstream
    consumers (dashboard, operator API, JSONL persistence).
    """

    tick_number: int = Field(ge=0)
    timestamp: datetime
    capacity_states: list[CapacityState] = []
    sensor_readings: list[SensorReading] = []
    sensor_anomalies: list[SensorAnomaly] = []
    vms_statuses: list[VMSStatusSnapshot] = []
    queue_predictions: list[QueuePrediction] = []
    vms_recommendations: list[VMSRecommendation] = []
    travel_time_readings: list[TravelTimeReading] = []
    calibration: CalibrationSnapshot | None = None
