"""Weather and road-surface adapter for safety-conservative physics tuning."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from config import WEATHER_SURFACE_FACTORS

if TYPE_CHECKING:  # avoid a runtime import cycle (smhi_forecast imports SURFACE_RANK)
    from src.smhi_forecast import WeatherForecast


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
    forecast_state: str = Field(
        default="dry",
        description="Worst SMHI forecast surface state within the look-ahead window",
    )
    forecast_lead_minutes: float | None = Field(
        default=None,
        description="Lead time (minutes) to the forecast onset; None when dry",
    )
    forecast_reason: str = ""
    proactive_halka: bool = Field(
        default=False,
        description="Forecast warrants a pre-staged HALKA advisory before friction drops",
    )


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
        forecast: "WeatherForecast | None" = None,
    ) -> WeatherAdjustment:
        """Return the worst corridor surface adjustment.

        RoadCondition records are treated as authoritative active feed records.
        WeatherMeasurepoint observations are ignored when all samples are stale.
        An optional SMHI ``forecast`` *escalates* the surface state when it
        predicts a worse upcoming surface than is currently observed — it can
        never relax the adjustment. Missing or stale data falls back to dry/low
        confidence and never raises model capacity above the dry baseline.
        """
        now = now or datetime.now()

        road_state, road_reason, warning_records = self._classify_road_conditions(
            road_condition_records
        )
        weather_state, weather_reason, weather_confidence = self._classify_weather(
            weather_records,
            now,
        )
        forecast_state, forecast_reason, forecast_lead = self._classify_forecast(forecast)

        observed_state = self._worst_state(road_state, weather_state)
        state = self._worst_state(observed_state, forecast_state)

        # Forecast-driven HALKA pre-staging: snow/ice is forecast that the live
        # road/weather feeds do not yet show, and no authoritative warning record
        # already covers it. This is the "anticipate, don't observe" path.
        proactive_halka = (
            forecast_state in {"snow", "ice"}
            and SURFACE_RANK[forecast_state] > SURFACE_RANK[observed_state]
            and not warning_records
        )

        forecast_meta = {
            "forecast_state": forecast_state,
            "forecast_lead_minutes": forecast_lead,
            "forecast_reason": forecast_reason,
            "proactive_halka": proactive_halka,
        }

        if warning_records:
            return self._build(state, "high", road_reason, warning_records, **forecast_meta)

        if state == "dry" and not road_condition_records and weather_confidence == "low":
            return self._build("dry", "low", weather_reason, [], **forecast_meta)

        confidence = "medium" if state != "dry" else weather_confidence
        reason = self._dominant_reason(
            road_state,
            road_reason,
            weather_state,
            weather_reason,
            forecast_state,
            forecast_reason,
        )
        return self._build(state, confidence, reason, [], **forecast_meta)

    def _classify_forecast(
        self,
        forecast: "WeatherForecast | None",
    ) -> tuple[str, str, float | None]:
        if forecast is None:
            return "dry", "", None
        state = forecast.surface_state if forecast.surface_state in SURFACE_RANK else "dry"
        return state, forecast.reason or "", forecast.onset_minutes

    @staticmethod
    def _dominant_reason(
        road_state: str,
        road_reason: str,
        weather_state: str,
        weather_reason: str,
        forecast_state: str,
        forecast_reason: str,
    ) -> str:
        """Return the reason from the source that produced the worst surface.

        Ties favour observed sources (road > weather) over the forecast, so a
        forecast only owns the reason string when it is the dominant signal.
        """
        candidates = [
            (SURFACE_RANK[road_state], 3, road_reason),
            (SURFACE_RANK[weather_state], 2, weather_reason),
            (SURFACE_RANK[forecast_state], 1, forecast_reason or "SMHI forecast"),
        ]
        candidates.sort(key=lambda candidate: (candidate[0], candidate[1]), reverse=True)
        return candidates[0][2]

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
        *,
        forecast_state: str = "dry",
        forecast_lead_minutes: float | None = None,
        forecast_reason: str = "",
        proactive_halka: bool = False,
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
            forecast_state=forecast_state,
            forecast_lead_minutes=forecast_lead_minutes,
            forecast_reason=forecast_reason,
            proactive_halka=proactive_halka,
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
