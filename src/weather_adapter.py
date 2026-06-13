"""Weather and road-surface adapter for safety-conservative physics tuning."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field

from config import WEATHER_SURFACE_FACTORS


SURFACE_RANK: dict[str, int] = {
    "dry": 0,
    "wet": 1,
    "snow": 2,
    "ice": 3,
}

DEFAULT_STALE_MINUTES = 30


class WeatherAdjustment(BaseModel):
    """Per-tick physics adjustment derived from weather/road-condition data."""

    surface_state: str = Field(description="'dry', 'wet', 'snow', or 'ice'")
    free_flow_factor: float = Field(ge=0.0, le=1.0)
    capacity_factor: float = Field(ge=0.0, le=1.0)
    confidence: str = Field(description="'high', 'medium', or 'low'")
    reason: str
    warning_records: list[dict[str, Any]] = Field(default_factory=list)


class WeatherAdapter:
    """Classify corridor surface state from Trafikverket weather feeds."""

    def __init__(
        self,
        surface_factors: dict[str, tuple[float, float]] | None = None,
        stale_minutes: int = DEFAULT_STALE_MINUTES,
    ) -> None:
        self.surface_factors = surface_factors or WEATHER_SURFACE_FACTORS
        self.stale_minutes = stale_minutes

    def compute(
        self,
        weather_records: list[dict[str, Any]],
        road_condition_records: list[dict[str, Any]],
        now: datetime | None = None,
    ) -> WeatherAdjustment:
        """Return the worst observed corridor surface adjustment.

        RoadCondition records are treated as authoritative active feed records.
        WeatherMeasurepoint observations are ignored when all samples are stale.
        Missing or stale data falls back to dry/low confidence and never raises
        model capacity above the configured dry baseline.
        """
        now = now or datetime.now()

        road_state, road_reason, warning_records = self._classify_road_conditions(
            road_condition_records
        )
        if warning_records:
            return self._build(
                road_state,
                "high",
                road_reason,
                warning_records,
            )

        weather_state, weather_reason, weather_confidence = self._classify_weather(
            weather_records,
            now,
        )

        state = self._worst_state(road_state, weather_state)
        if state == "dry" and not road_condition_records and weather_confidence == "low":
            return self._build("dry", "low", weather_reason, [])

        confidence = "medium" if state != "dry" else weather_confidence
        reason = road_reason if SURFACE_RANK[road_state] >= SURFACE_RANK[weather_state] else weather_reason
        return self._build(state, confidence, reason, [])

    def _classify_road_conditions(
        self,
        records: list[dict[str, Any]],
    ) -> tuple[str, str, list[dict[str, Any]]]:
        state = "dry"
        reason = "no active road-condition warning"
        warning_records: list[dict[str, Any]] = []

        for record in records:
            record_state = _classify_surface_text(
                " ".join(
                    str(value or "")
                    for value in (
                        record.get("condition_code"),
                        record.get("condition_text"),
                        record.get("condition_info"),
                    )
                )
            )
            state = self._worst_state(state, record_state)
            if record.get("warning") is True:
                warning_records.append(record)

        if warning_records:
            warning_state = "dry"
            for record in warning_records:
                warning_state = self._worst_state(
                    warning_state,
                    _classify_surface_text(
                        " ".join(
                            str(value or "")
                            for value in (
                                record.get("condition_code"),
                                record.get("condition_text"),
                                record.get("condition_info"),
                            )
                        )
                    ),
                )
            state = self._worst_state(state, warning_state)
            if state == "dry":
                state = "ice"
            reason = f"{len(warning_records)} RoadCondition warning record(s)"
        elif state != "dry":
            reason = "RoadCondition surface state"

        return state, reason, warning_records

    def _classify_weather(
        self,
        records: list[dict[str, Any]],
        now: datetime,
    ) -> tuple[str, str, str]:
        if not records:
            return "dry", "missing weather/road-condition data", "low"

        fresh_records = [
            record
            for record in records
            if not self._is_stale(record.get("sample_time"), now)
        ]
        if not fresh_records:
            return "dry", "stale WeatherMeasurepoint data", "low"

        state = "dry"
        reasons: list[str] = []
        for record in fresh_records:
            precipitation = str(record.get("precipitation") or "").lower()
            rain_sum = _as_float(record.get("precip_rain_sum"))
            snow_sum = _as_float(record.get("precip_snow_water_eq"))
            surface_temp = _as_float(record.get("surface_temp_c"))

            record_state = "dry"
            if snow_sum and snow_sum > 0:
                record_state = "snow"
                reasons.append("snow water-equivalent observed")
            elif "snö" in precipitation or "snow" in precipitation:
                record_state = "snow"
                reasons.append("snow precipitation observed")
            elif surface_temp is not None and surface_temp <= 0 and (
                (rain_sum and rain_sum > 0) or precipitation
            ):
                record_state = "ice"
                reasons.append("surface temp <= 0C with precipitation")
            elif (rain_sum and rain_sum > 0) or "regn" in precipitation or "rain" in precipitation:
                record_state = "wet"
                reasons.append("rain/wet precipitation observed")

            state = self._worst_state(state, record_state)

        reason = reasons[0] if reasons else f"{len(fresh_records)} fresh weather record(s)"
        return state, reason, "medium"

    def _is_stale(self, sample_time: Any, now: datetime) -> bool:
        sample_dt = _parse_datetime(sample_time)
        if sample_dt is None:
            return False
        compare_now = now
        if sample_dt.tzinfo is not None and compare_now.tzinfo is None:
            compare_now = compare_now.replace(tzinfo=timezone.utc)
        if sample_dt.tzinfo is None and compare_now.tzinfo is not None:
            sample_dt = sample_dt.replace(tzinfo=compare_now.tzinfo)
        age_seconds = (compare_now - sample_dt).total_seconds()
        return age_seconds > self.stale_minutes * 60

    @staticmethod
    def _worst_state(left: str, right: str) -> str:
        return left if SURFACE_RANK[left] >= SURFACE_RANK[right] else right

    def _build(
        self,
        state: str,
        confidence: str,
        reason: str,
        warning_records: list[dict[str, Any]],
    ) -> WeatherAdjustment:
        free_flow_factor, capacity_factor = self.surface_factors.get(
            state,
            self.surface_factors["dry"],
        )
        return WeatherAdjustment(
            surface_state=state,
            free_flow_factor=min(float(free_flow_factor), 1.0),
            capacity_factor=min(float(capacity_factor), 1.0),
            confidence=confidence,
            reason=reason,
            warning_records=warning_records,
        )


def _classify_surface_text(text: str) -> str:
    normalized = text.lower()
    if any(token in normalized for token in ("is", "ice", "halka", "halt", "frost")):
        return "ice"
    if any(token in normalized for token in ("snö", "snow", "modd", "slask")):
        return "snow"
    if any(token in normalized for token in ("våt", "wet", "fukt", "regn", "rain")):
        return "wet"
    return "dry"


def _as_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_datetime(value: Any) -> datetime | None:
    if not value:
        return None
    if isinstance(value, datetime):
        return value
    if not isinstance(value, str):
        return None
    try:
        normalized = value.replace("Z", "+00:00")
        return datetime.fromisoformat(normalized)
    except ValueError:
        return None
