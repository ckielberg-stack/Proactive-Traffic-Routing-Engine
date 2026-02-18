"""
Operator Decision Support API (Phase 6).

FastAPI endpoints for the Trafikverket / Trafik Stockholm Traffic
Management Center control room frontend.  Provides instant, AI-verified
telemetry to reduce operator cognitive load.

Endpoints
---------
- ``GET /api/v1/operator/active-incidents``
    Active incidents verified by AI with YOLO thumbnails.
- ``GET /api/v1/operator/vms-recommendations``
    Active VMS activation recommendations with queue growth data
    and ``proxy_ground_truth_active`` flag from the Situation API.
- ``GET /api/v1/export/datex2``
    DATEX II XML export for National Traffic Management System (NTS).
- ``GET /health``
    Service health check.

Pipeline Integration
--------------------
The API reads from shared state populated by ``main_loop.py`` via the
``set_*`` functions.  Each tick pushes the latest TickResult into the
API state so endpoints always return the most recent data.
"""

from __future__ import annotations

import base64
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import RLock
from typing import Any
from xml.sax.saxutils import escape as xml_escape

from fastapi import FastAPI, Response
from pydantic import BaseModel, Field

from src.models import (
    IncidentReport,
    QueuePrediction,
    VMSRecommendation,
    VMSStatusSnapshot,
)
from src.vms_orchestrator import VMSOrchestrator

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Application setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="PTRE Operator Decision Support",
    description=(
        "Automated Incident Verification and Predictive VMS Copilot "
        "for Trafikverket / Trafik Stockholm traffic management operators."
    ),
    version="0.2.0",
)

# ---------------------------------------------------------------------------
# Shared pipeline state
# ---------------------------------------------------------------------------

_vms_orchestrator = VMSOrchestrator()


@dataclass
class _PipelineState:
    incidents: list[IncidentReport] = field(default_factory=list)
    predictions: list[QueuePrediction] = field(default_factory=list)
    vms_statuses: list[VMSStatusSnapshot] = field(default_factory=list)
    recommendations: list[VMSRecommendation] = field(default_factory=list)
    last_tick_time: datetime | None = None


_pipeline_state = _PipelineState()
_state_lock = RLock()


def _get_state_snapshot() -> _PipelineState:
    with _state_lock:
        return _PipelineState(
            incidents=list(_pipeline_state.incidents),
            predictions=list(_pipeline_state.predictions),
            vms_statuses=list(_pipeline_state.vms_statuses),
            recommendations=list(_pipeline_state.recommendations),
            last_tick_time=_pipeline_state.last_tick_time,
        )


# ---------------------------------------------------------------------------
# Response schemas (explicit for OpenAPI docs)
# ---------------------------------------------------------------------------


class IncidentListResponse(BaseModel):
    """Wrapper for the active-incidents endpoint."""
    count: int
    incidents: list[IncidentReport]


class VMSRecommendationWithGroundTruth(BaseModel):
    """A VMS recommendation enriched with Situation API proxy ground truth."""
    recommendation: VMSRecommendation
    proxy_ground_truth_active: bool = Field(
        description=(
            "True if a SPEEDMANAGEMENTID deviation already covers this "
            "road segment — meaning the human operator has already acted."
        )
    )
    proxy_speed_limit: int | None = Field(
        default=None,
        description="Speed limit from the Situation API proxy (if active)",
    )
    proxy_deviation_id: str | None = Field(
        default=None,
        description="Trafikverket Deviation ID of the matching proxy record",
    )


class VMSRecommendationResponse(BaseModel):
    """Wrapper for the vms-recommendations endpoint."""
    count: int
    ai_prediction_timestamp: datetime | None = None
    recommendations: list[VMSRecommendationWithGroundTruth]


# ---------------------------------------------------------------------------
# State management (called by main_loop.py each tick)
# ---------------------------------------------------------------------------


def set_active_incidents(incidents: list[IncidentReport]) -> None:
    """Inject incidents from the vision pipeline."""
    with _state_lock:
        _pipeline_state.incidents = list(incidents)


def set_active_predictions(predictions: list[QueuePrediction]) -> None:
    """Inject predictions from the physics pipeline."""
    with _state_lock:
        _pipeline_state.predictions = list(predictions)


def set_active_vms_statuses(statuses: list[VMSStatusSnapshot]) -> None:
    """Inject polled VMS proxy statuses from the Situation API."""
    with _state_lock:
        _pipeline_state.vms_statuses = list(statuses)


def set_active_recommendations(recommendations: list[VMSRecommendation]) -> None:
    """Inject VMS recommendations from the orchestrator."""
    with _state_lock:
        _pipeline_state.recommendations = list(recommendations)


def set_pipeline_snapshot(
    *,
    incidents: list[IncidentReport],
    predictions: list[QueuePrediction],
    vms_statuses: list[VMSStatusSnapshot],
    recommendations: list[VMSRecommendation],
    last_tick_time: datetime | None,
) -> None:
    """Atomically replace all live pipeline fields for one tick."""
    with _state_lock:
        _pipeline_state.incidents = list(incidents)
        _pipeline_state.predictions = list(predictions)
        _pipeline_state.vms_statuses = list(vms_statuses)
        _pipeline_state.recommendations = list(recommendations)
        _pipeline_state.last_tick_time = last_tick_time


def set_vms_orchestrator(orchestrator: VMSOrchestrator) -> None:
    """Replace the default VMS orchestrator (useful in tests)."""
    global _vms_orchestrator
    _vms_orchestrator = orchestrator


def set_last_tick_time(t: datetime | None) -> None:
    """Record the timestamp of the most recent tick."""
    with _state_lock:
        _pipeline_state.last_tick_time = t


# ---------------------------------------------------------------------------
# Ground-truth proxy matching
# ---------------------------------------------------------------------------


def _match_proxy_ground_truth(
    recommendation: VMSRecommendation,
    vms_statuses: list[VMSStatusSnapshot],
) -> tuple[bool, int | None, str | None]:
    """Check if a SPEEDMANAGEMENTID deviation covers this recommendation's road.

    Returns (is_active, speed_limit, deviation_id).
    """
    for status in vms_statuses:
        if not status.is_active:
            continue

        # Match by VMS ID (if the proxy deviation was mapped to a gantry ID)
        if status.vms_id == recommendation.vms_id:
            return True, status.speed_limit, status.vms_id

        # Match by road name (E4 in the proxy vms_name)
        # The proxy vms_name format is "E4 — location..."
        proxy_road = status.vms_name.split(" —")[0].strip()
        # The recommendation's VMS name may contain the road
        if proxy_road and proxy_road in recommendation.vms_name:
            return True, status.speed_limit, status.vms_id

    return False, None, None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/api/v1/operator/active-incidents", response_model=IncidentListResponse)
async def get_active_incidents() -> IncidentListResponse:
    """Return active incidents verified by AI.

    Each incident includes ``incident_type``, ``lanes_affected``,
    ``capacity_drop_percentage``, and a base64-encoded JPEG thumbnail
    with YOLO bounding boxes drawn for instant human verification.
    """
    state = _get_state_snapshot()
    return IncidentListResponse(
        count=len(state.incidents),
        incidents=state.incidents,
    )


@app.get("/api/v1/operator/vms-recommendations", response_model=VMSRecommendationResponse)
async def get_vms_recommendations() -> VMSRecommendationResponse:
    """Return active VMS recommendations with proxy ground-truth check.

    Each recommendation includes a ``proxy_ground_truth_active`` boolean
    that indicates whether the human operator has already activated a
    speed advisory on this road segment (detected via the Situation API
    SPEEDMANAGEMENTID proxy).

    This allows the control room to see:
    - AI predicted activation was needed at T₁
    - Human operator actually acted at T₂
    - Delta = T₂ - T₁ (our value proposition)
    """
    state = _get_state_snapshot()
    vms_statuses = state.vms_statuses
    active_predictions = state.predictions
    last_tick_time = state.last_tick_time

    # If we have pre-computed recommendations from the tick, use those
    recs_to_use = list(state.recommendations)

    # Fallback: compute on-the-fly if the tick hasn't pushed recs yet
    if not recs_to_use and active_predictions:
        for pred in active_predictions:
            recs = _vms_orchestrator.generate_recommendations(
                pred, vms_statuses=vms_statuses,
            )
            recs_to_use.extend(recs)

    enriched: list[VMSRecommendationWithGroundTruth] = []
    for rec in recs_to_use:
        gt_active, gt_speed, gt_dev_id = _match_proxy_ground_truth(
            rec, vms_statuses,
        )
        enriched.append(VMSRecommendationWithGroundTruth(
            recommendation=rec,
            proxy_ground_truth_active=gt_active,
            proxy_speed_limit=gt_speed,
            proxy_deviation_id=gt_dev_id,
        ))

    return VMSRecommendationResponse(
        count=len(enriched),
        ai_prediction_timestamp=last_tick_time,
        recommendations=enriched,
    )


@app.get("/api/v1/export/datex2")
async def export_datex2() -> Response:
    """Format active incidents and VMS recommendations into DATEX II XML.

    DATEX II is the European standard for traffic event data exchange.
    This produces a valid XML document with:
    - ``SituationPublication`` for AI-verified incidents
    - ``SpeedManagement`` records for VMS recommendations

    The output can be ingested by the National Traffic Management System
    (NTS) without manual intervention by operators.
    """
    state = _get_state_snapshot()
    xml_content = _build_datex2_xml(
        incidents=state.incidents,
        recommendations=state.recommendations or [],
        vms_statuses=state.vms_statuses,
    )
    return Response(
        content=xml_content,
        media_type="application/xml",
        headers={"Content-Disposition": "inline; filename=datex2_export.xml"},
    )


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


@app.get("/health")
async def health() -> dict[str, Any]:
    state = _get_state_snapshot()
    return {
        "status": "ok",
        "service": "operator-api",
        "last_tick": (
            state.last_tick_time.isoformat() if state.last_tick_time else None
        ),
        "active_incidents": len(state.incidents),
        "active_recommendations": len(state.recommendations),
        "proxy_statuses_polled": len(state.vms_statuses),
    }


# ---------------------------------------------------------------------------
# DATEX II XML builder
# ---------------------------------------------------------------------------

# DATEX II v3.4 namespace
_DATEX2_NS = "http://datex2.eu/schema/3/d2Payload"
_DATEX2_SITUATION_NS = "http://datex2.eu/schema/3/situation"
_XSI_NS = "http://www.w3.org/2001/XMLSchema-instance"
_XML_ATTR_ESCAPE_MAP = {'"': "&quot;", "'": "&apos;"}


def _xml_text(value: Any) -> str:
    return xml_escape(str(value))


def _xml_attr(value: Any) -> str:
    return xml_escape(str(value), _XML_ATTR_ESCAPE_MAP)


def _build_datex2_xml(
    incidents: list[IncidentReport],
    recommendations: list[VMSRecommendation],
    vms_statuses: list[VMSStatusSnapshot],
) -> str:
    """Build a DATEX II compliant XML document.

    Includes two record types:
    1. ``situationRecord`` for each AI-verified incident
    2. ``speedManagement`` for each VMS recommendation (PTRE extension)

    Uses string templating for the MVP.  A production version would
    use the full DATEX II XSD schemas with lxml.
    """
    situations = []

    # --- AI-verified incidents ---
    for i, inc in enumerate(incidents):
        situation_id = f"PTRE-INC-{inc.camera_id}-{i:04d}"
        lat_block = ""
        if inc.lat is not None and inc.lng is not None:
            lat_block = f"""
        <locationReference>
          <pointByCoordinates>
            <latitude>{inc.lat:.6f}</latitude>
            <longitude>{inc.lng:.6f}</longitude>
          </pointByCoordinates>
        </locationReference>"""

        situations.append(
            f"""    <situation id="{_xml_attr(situation_id)}">
      <headerInformation>
        <areaOfInterest>national</areaOfInterest>
        <confidentiality>restrictedToAuthoritiesTrafficOperatorsAndPublishers</confidentiality>
        <informationStatus>real</informationStatus>
      </headerInformation>
      <situationRecord>
        <situationRecordCreationTime>{_xml_text(inc.timestamp.isoformat())}</situationRecordCreationTime>
        <situationRecordVersionTime>{_xml_text(inc.timestamp.isoformat())}</situationRecordVersionTime>
        <probabilityOfOccurrence>certain</probabilityOfOccurrence>
        <severity>high</severity>
        <validity>
          <validityStatus>active</validityStatus>
        </validity>
        <operatorActionStatus>requested</operatorActionStatus>
        <typeOfIncident>{_xml_text(inc.incident_type)}</typeOfIncident>
        <lanesAffected>{inc.lanes_affected}</lanesAffected>
        <totalLanes>{inc.total_lanes}</totalLanes>
        <capacityDropPercentage>{inc.capacity_drop_percentage:.1f}</capacityDropPercentage>
        <sourceCamera>{_xml_text(inc.camera_id)}</sourceCamera>
        <confidence>{inc.confidence:.2f}</confidence>{lat_block}
      </situationRecord>
    </situation>"""
        )

    # --- VMS Speed Management Recommendations (PTRE extension) ---
    for j, rec in enumerate(recommendations):
        sm_id = f"PTRE-VMS-{rec.vms_id}-{j:04d}"

        # Check if human operator already acted (proxy ground truth)
        gt_active, gt_speed, gt_dev_id = _match_proxy_ground_truth(
            rec, vms_statuses,
        )
        operator_status = "implemented" if gt_active else "requested"

        situations.append(
            f"""    <situation id="{_xml_attr(sm_id)}">
      <headerInformation>
        <areaOfInterest>national</areaOfInterest>
        <confidentiality>restrictedToAuthoritiesTrafficOperatorsAndPublishers</confidentiality>
        <informationStatus>real</informationStatus>
      </headerInformation>
      <speedManagement>
        <creationTime>{_xml_text(rec.timestamp.isoformat())}</creationTime>
        <vmsId>{_xml_text(rec.vms_id)}</vmsId>
        <vmsName>{_xml_text(rec.vms_name)}</vmsName>
        <recommendedMessage>{_xml_text(rec.recommended_message)}</recommendedMessage>
        <urgency>{_xml_text(rec.urgency)}</urgency>
        <queueGrowthSpeedKmh>{rec.queue_growth_speed_kmh:.1f}</queueGrowthSpeedKmh>
        <estimatedActivationMinutes>{rec.estimated_activation_minutes:.1f}</estimatedActivationMinutes>
        <operatorActionStatus>{_xml_text(operator_status)}</operatorActionStatus>
        <triggeringCamera>{_xml_text(rec.triggering_camera_id)}</triggeringCamera>
        <narrative>{_xml_text(rec.summary)}</narrative>
      </speedManagement>
    </situation>"""
        )

    situations_xml = "\n".join(situations) if situations else ""

    return f"""<?xml version="1.0" encoding="UTF-8"?>
<d2LogicalModel xmlns="{_DATEX2_NS}"
                xmlns:sit="{_DATEX2_SITUATION_NS}"
                xmlns:xsi="{_XSI_NS}"
                modelBaseVersion="3">
  <exchange>
    <supplierIdentification>
      <country>se</country>
      <nationalIdentifier>PTRE-AI-Copilot</nationalIdentifier>
    </supplierIdentification>
  </exchange>
  <payloadPublication type="SituationPublication">
    <publicationTime>{datetime.now(tz=timezone.utc).isoformat()}</publicationTime>
    <publicationCreator>
      <country>se</country>
      <nationalIdentifier>PTRE-AI-Copilot</nationalIdentifier>
    </publicationCreator>
{situations_xml}
  </payloadPublication>
</d2LogicalModel>
"""


# ---------------------------------------------------------------------------
# Demo data loader (for development / testing)
# ---------------------------------------------------------------------------


def load_demo_data() -> None:
    """Populate the API with realistic demo data for development."""
    # Example 1-pixel white JPEG as placeholder thumbnail
    _placeholder_thumb = base64.b64encode(
        b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
        b"\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t"
        b"\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a"
        b"\x1f\x1e\x1d\x1a\x1c\x1c $.\' \",#\x1c\x1c(7),01444\x1f\'9=82<.342"
        b"\xff\xc0\x00\x0b\x08\x00\x01\x00\x01\x01\x01\x11\x00"
        b"\xff\xc4\x00\x1f\x00\x00\x01\x05\x01\x01\x01\x01\x01\x01\x00"
        b"\x00\x00\x00\x00\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b"
        b"\xff\xda\x00\x08\x01\x01\x00\x00?\x00T\xdb\x9e\xa7\x13\xa2\x80"
        b"\xff\xd9"
    ).decode("ascii")

    now = datetime.now()

    set_active_incidents([
        IncidentReport(
            timestamp=now,
            camera_id="SE_STA_CAMERA_0_50438756",
            incident_type="vehicle_stopped",
            lanes_affected=1,
            total_lanes=3,
            capacity_drop_percentage=33.3,
            thumbnail_base64=_placeholder_thumb,
            confidence=0.87,
            lat=59.2960,
            lng=18.0041,
        ),
        IncidentReport(
            timestamp=now,
            camera_id="SE_STA_CAMERA_0_50438726",
            incident_type="congestion",
            lanes_affected=2,
            total_lanes=3,
            capacity_drop_percentage=66.7,
            thumbnail_base64=_placeholder_thumb,
            confidence=0.93,
            lat=59.3150,
            lng=18.0033,
        ),
    ])

    demo_prediction = QueuePrediction(
        timestamp=now,
        camera_id="SE_STA_CAMERA_0_50438756",
        origin_lat=59.2960,
        origin_lng=18.0041,
        origin_chainage_km=8.6,
        growth_speed_kmh=8.0,
        lengths_at_minutes={1: 0.133, 3: 0.4, 5: 0.667},
    )
    set_active_predictions([demo_prediction])

    set_active_vms_statuses([
        VMSStatusSnapshot(
            timestamp=now,
            vms_id="SE_STA_SPEEDMANAGEMENTID_1_DEMO",
            vms_name="E4 — Hallunda-Kungens Kurva (demo)",
            is_active=True,
            displayed_message="Rekommenderad hastighet: 70km/h",
            speed_limit=70,
        ),
    ])

    recs = _vms_orchestrator.generate_recommendations(demo_prediction)
    set_active_recommendations(recs)
    set_last_tick_time(now)


# ---------------------------------------------------------------------------
# Standalone server
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    load_demo_data()
    uvicorn.run(app, host="0.0.0.0", port=8081)
