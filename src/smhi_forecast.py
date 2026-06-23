"""SMHI open-data point-forecast source for proactive weather adjustment.

This is the forecast half of TRAFIK-032 (proposal P5). The P1 ``WeatherAdapter``
classifies the road surface from *current* Trafikverket observations; this module
adds an *anticipatory* signal by polling the free SMHI metfcst (pmp3g v2) point
forecast for a single corridor reference point and classifying the worst upcoming
road-surface state within a short look-ahead window. Feeding that into the adapter
lets PTRE pre-degrade physics thresholds and pre-stage HALKA advisories *before*
friction actually drops.

Design rules (mirror the P1 ``WeatherAdapter`` fail-safe philosophy):

- **Conservative-only:** the forecast may only escalate conservatism downstream.
  A fetch or parse failure yields ``None`` (no forecast) and never relaxes the
  model — the adapter treats ``None`` as "no escalation", keeping observed-only
  behaviour.
- **Throttled refresh:** forecasts change slowly, so network calls are throttled
  to ``poll_interval_minutes``. They are *not* re-fetched every 60-second tick;
  the source caches the last good forecast and serves it until it ages out.
- **Testable:** the network fetch and the clock are injectable, so the parsing,
  classification, time-window, and caching logic are unit-testable with no
  network access.
- **Timezone-correct:** SMHI ``validTime`` is UTC; the caller's ``now`` may be
  naive local time. All comparisons happen in UTC (see :func:`_to_utc`).
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

import requests
from pydantic import BaseModel, Field

from src.weather_adapter import SURFACE_RANK

logger = logging.getLogger(__name__)

#: SMHI metfcst (pmp3g v2) point-forecast endpoint — free, no API key.
SMHI_FORECAST_URL = (
    "https://opendata-download-metfcst.smhi.se/api/category/pmp3g/version/2"
    "/geotype/point/lon/{lon:.6f}/lat/{lat:.6f}/data.json"
)

DEFAULT_POLL_INTERVAL_MINUTES = 30
DEFAULT_LOOKAHEAD_MINUTES = 60
DEFAULT_TIMEOUT_SECONDS = 10

# SMHI precipitation category (parameter ``pcat``) → surface state.
#   0 none · 1 snow · 2 snow+rain · 3 rain · 4 drizzle · 5 freezing rain · 6 freezing drizzle
_PCAT_SNOW: frozenset[int] = frozenset({1, 2})
_PCAT_RAIN: frozenset[int] = frozenset({3, 4})
_PCAT_FREEZING: frozenset[int] = frozenset({5, 6})


class WeatherForecast(BaseModel):
    """Worst upcoming road-surface state within the look-ahead window."""

    surface_state: str = Field(
        description="Worst upcoming 'dry', 'wet', 'snow', or 'ice' in the window"
    )
    onset_minutes: float | None = Field(
        default=None,
        description="Lead time (minutes) to the first step reaching surface_state; "
        "None when the forecast window is dry",
    )
    confidence: str = Field(
        default="medium",
        description="'medium' or 'low' — forecasts are predicted, never authoritative",
    )
    reason: str = ""
    reference_time: datetime | None = Field(
        default=None, description="SMHI approvedTime / referenceTime of the forecast"
    )
    valid_until: datetime | None = Field(
        default=None, description="validTime of the last forecast step considered"
    )
    lookahead_minutes: int = 0
    sample_count: int = Field(
        default=0, description="Number of forecast steps inside the look-ahead window"
    )


class SMHIForecastSource:
    """Poll-throttled SMHI point-forecast source for one corridor reference point.

    Parameters
    ----------
    lat, lon:
        Corridor reference point (decimal degrees, WGS84).
    poll_interval_minutes:
        Minimum age before the cached forecast is re-fetched from the network.
    lookahead_minutes:
        Forward window over which the worst surface state is classified.
    fetch_fn:
        Optional ``(lat, lon) -> dict | None`` override used for testing. When
        omitted, the live SMHI endpoint is queried.
    timeout_seconds:
        HTTP timeout for the live fetch.
    """

    def __init__(
        self,
        lat: float,
        lon: float,
        *,
        poll_interval_minutes: int = DEFAULT_POLL_INTERVAL_MINUTES,
        lookahead_minutes: int = DEFAULT_LOOKAHEAD_MINUTES,
        fetch_fn: Callable[[float, float], dict[str, Any] | None] | None = None,
        timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self.lat = float(lat)
        self.lon = float(lon)
        self.poll_interval_minutes = poll_interval_minutes
        self.lookahead_minutes = lookahead_minutes
        self.timeout_seconds = timeout_seconds
        self._fetch_fn = fetch_fn or self._default_fetch
        self._cached: WeatherForecast | None = None
        self._fetched_at_utc: datetime | None = None

    def get_forecast(self, now: datetime | None = None) -> WeatherForecast | None:
        """Return the current forecast, refreshing only when the cache is stale.

        On any fetch/parse failure the last good forecast is returned (or ``None``
        if none was ever obtained) so the adapter never loses conservatism.
        """
        now_utc = _to_utc(now or datetime.now(timezone.utc))

        if self._is_cache_fresh(now_utc):
            return self._cached

        raw: dict[str, Any] | None = None
        try:
            raw = self._fetch_fn(self.lat, self.lon)
        except Exception as exc:  # pragma: no cover - defensive, network errors
            logger.warning("SMHI forecast fetch failed: %s", exc)

        if not raw:
            # Fail-safe: keep serving the last good forecast (if any).
            return self._cached

        forecast = self.parse(
            raw, now=now_utc, lookahead_minutes=self.lookahead_minutes
        )
        if forecast is not None:
            self._cached = forecast
            self._fetched_at_utc = now_utc
        return self._cached

    def _is_cache_fresh(self, now_utc: datetime) -> bool:
        if self._cached is None or self._fetched_at_utc is None:
            return False
        age_seconds = (now_utc - self._fetched_at_utc).total_seconds()
        return age_seconds < self.poll_interval_minutes * 60

    def _default_fetch(self, lat: float, lon: float) -> dict[str, Any] | None:
        url = SMHI_FORECAST_URL.format(lon=lon, lat=lat)
        resp = requests.get(url, timeout=self.timeout_seconds)
        resp.raise_for_status()
        return resp.json()

    @classmethod
    def parse(
        cls,
        raw: dict[str, Any],
        *,
        now: datetime,
        lookahead_minutes: int,
    ) -> WeatherForecast | None:
        """Classify the worst surface state in ``[now, now + lookahead]``.

        Returns ``None`` when the payload has no usable steps in the window, so
        the adapter treats it as "no forecast signal" rather than a dry override.
        """
        if not isinstance(raw, dict):
            return None
        series = raw.get("timeSeries")
        if not isinstance(series, list) or not series:
            return None

        now_utc = _to_utc(now)
        window_end = now_utc + timedelta(minutes=lookahead_minutes)
        reference_time = _parse_iso(raw.get("approvedTime") or raw.get("referenceTime"))

        steps: list[tuple[float, str, datetime]] = []  # (lead_minutes, state, valid)
        for step in series:
            if not isinstance(step, dict):
                continue
            valid = _parse_iso(step.get("validTime"))
            if valid is None:
                continue
            if valid < now_utc:
                continue
            if valid > window_end:
                break  # timeSeries is chronological — past the window
            lead_minutes = (valid - now_utc).total_seconds() / 60.0
            state = _classify_step(step.get("parameters", []))
            steps.append((max(lead_minutes, 0.0), state, valid))

        if not steps:
            return None

        worst_state = "dry"
        for _, state, _ in steps:
            if SURFACE_RANK[state] > SURFACE_RANK[worst_state]:
                worst_state = state

        onset_minutes: float | None = None
        if worst_state != "dry":
            onset_minutes = round(
                min(lead for lead, state, _ in steps if state == worst_state), 1
            )

        if worst_state == "dry":
            reason = f"dry forecast across next {lookahead_minutes} min"
        else:
            reason = (
                f"{worst_state} forecast within {lookahead_minutes} min "
                f"(onset ~{onset_minutes:.0f} min)"
            )

        return WeatherForecast(
            surface_state=worst_state,
            onset_minutes=onset_minutes,
            confidence="medium",
            reason=reason,
            reference_time=reference_time,
            valid_until=steps[-1][2],
            lookahead_minutes=lookahead_minutes,
            sample_count=len(steps),
        )


def _classify_step(parameters: Any) -> str:
    """Map one SMHI forecast step's parameters to a road-surface state."""
    pcat = _param_value(parameters, "pcat")
    if pcat is None:
        return "dry"
    category = int(round(pcat))
    if category in _PCAT_FREEZING:
        return "ice"
    if category in _PCAT_SNOW:
        return "snow"
    if category in _PCAT_RAIN:
        temp = _param_value(parameters, "t")
        # Rain onto a sub-zero road surface freezes — treat as ice, not wet.
        if temp is not None and temp <= 0.0:
            return "ice"
        return "wet"
    return "dry"


def _param_value(parameters: Any, name: str) -> float | None:
    if not isinstance(parameters, list):
        return None
    for param in parameters:
        if isinstance(param, dict) and param.get("name") == name:
            values = param.get("values")
            if isinstance(values, list) and values:
                try:
                    return float(values[0])
                except (TypeError, ValueError):
                    return None
    return None


def _to_utc(value: datetime) -> datetime:
    """Return ``value`` as an aware UTC datetime.

    A naive datetime is assumed to be system-local time (matching the tick
    loop's ``datetime.now()``) and converted; an aware datetime is converted
    directly.
    """
    if value.tzinfo is None:
        value = value.astimezone()
    return value.astimezone(timezone.utc)


def _parse_iso(value: Any) -> datetime | None:
    if not value or not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
