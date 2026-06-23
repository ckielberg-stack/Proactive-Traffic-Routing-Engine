"""Tests for the SMHI open-data forecast source (TRAFIK-032)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.smhi_forecast import SMHIForecastSource, WeatherForecast

NOW = datetime(2026, 6, 23, 20, 0, 0, tzinfo=timezone.utc)


def _step(valid_time: str, *, pcat: int = 0, t: float = 5.0) -> dict:
    return {
        "validTime": valid_time,
        "parameters": [
            {"name": "t", "levelType": "hl", "level": 2, "unit": "Cel", "values": [t]},
            {"name": "pcat", "levelType": "hl", "level": 0, "unit": "category", "values": [pcat]},
        ],
    }


def _payload(steps: list[dict], approved: str = "2026-06-23T20:00:00Z") -> dict:
    return {
        "approvedTime": approved,
        "referenceTime": approved,
        "geometry": {"type": "Point", "coordinates": [[18.0, 59.3]]},
        "timeSeries": steps,
    }


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("pcat", "temp", "expected"),
    [
        (0, 5.0, "dry"),
        (1, -1.0, "snow"),     # snow
        (2, 0.0, "snow"),      # snow and rain
        (3, 5.0, "wet"),       # rain, above freezing
        (4, 5.0, "wet"),       # drizzle
        (3, -2.0, "ice"),      # rain onto a sub-zero road → freezes
        (5, 1.0, "ice"),       # freezing rain
        (6, 1.0, "ice"),       # freezing drizzle
    ],
)
def test_step_classification(pcat: int, temp: float, expected: str) -> None:
    forecast = SMHIForecastSource.parse(
        _payload([_step("2026-06-23T20:30:00Z", pcat=pcat, t=temp)]),
        now=NOW,
        lookahead_minutes=60,
    )
    assert forecast is not None
    assert forecast.surface_state == expected


def test_dry_forecast_reports_no_onset() -> None:
    forecast = SMHIForecastSource.parse(
        _payload([_step("2026-06-23T20:30:00Z", pcat=0, t=8.0)]),
        now=NOW,
        lookahead_minutes=60,
    )
    assert forecast is not None
    assert forecast.surface_state == "dry"
    assert forecast.onset_minutes is None
    assert "dry forecast" in forecast.reason


def test_onset_tracks_the_worst_state_not_the_first_degraded() -> None:
    forecast = SMHIForecastSource.parse(
        _payload(
            [
                _step("2026-06-23T20:15:00Z", pcat=3, t=5.0),   # wet at +15
                _step("2026-06-23T20:45:00Z", pcat=1, t=-1.0),  # snow at +45
            ]
        ),
        now=NOW,
        lookahead_minutes=60,
    )
    assert forecast is not None
    assert forecast.surface_state == "snow"
    assert forecast.onset_minutes == 45.0  # lead to the worst state, not the wet step
    assert forecast.sample_count == 2


def test_steps_outside_lookahead_window_are_ignored() -> None:
    forecast = SMHIForecastSource.parse(
        _payload(
            [
                _step("2026-06-23T20:30:00Z", pcat=0, t=6.0),   # dry, in window
                _step("2026-06-23T22:00:00Z", pcat=1, t=-3.0),  # snow, 2h out
            ]
        ),
        now=NOW,
        lookahead_minutes=60,
    )
    assert forecast is not None
    assert forecast.surface_state == "dry"  # the snow step is past the window
    assert forecast.sample_count == 1


def test_no_usable_steps_in_window_returns_none() -> None:
    # Only a step well past the look-ahead → no signal (not a dry override).
    forecast = SMHIForecastSource.parse(
        _payload([_step("2026-06-23T23:00:00Z", pcat=1, t=-2.0)]),
        now=NOW,
        lookahead_minutes=60,
    )
    assert forecast is None


def test_empty_or_malformed_payload_returns_none() -> None:
    assert SMHIForecastSource.parse({}, now=NOW, lookahead_minutes=60) is None
    assert SMHIForecastSource.parse(
        {"timeSeries": []}, now=NOW, lookahead_minutes=60
    ) is None
    assert SMHIForecastSource.parse(
        {"timeSeries": "nope"}, now=NOW, lookahead_minutes=60
    ) is None


def test_utc_normalisation_is_timezone_correct() -> None:
    """A naive-UTC vs offset 'now' for the same instant yields the same onset."""
    payload = _payload([_step("2026-06-23T20:30:00Z", pcat=1, t=-1.0)])
    forecast_utc = SMHIForecastSource.parse(payload, now=NOW, lookahead_minutes=60)
    forecast_offset = SMHIForecastSource.parse(
        payload,
        now=datetime(2026, 6, 23, 22, 0, 0, tzinfo=timezone(timedelta(hours=2))),
        lookahead_minutes=60,
    )
    assert forecast_utc is not None and forecast_offset is not None
    assert forecast_utc.onset_minutes == forecast_offset.onset_minutes == 30.0


# ---------------------------------------------------------------------------
# Poll-throttled caching and fail-safe behaviour
# ---------------------------------------------------------------------------


def test_forecast_is_cached_within_poll_interval_then_refetched() -> None:
    calls = {"n": 0}

    def fetch(_lat: float, _lon: float) -> dict:
        calls["n"] += 1
        return _payload([_step("2026-06-23T21:30:00Z", pcat=1, t=-1.0)])

    source = SMHIForecastSource(
        59.3, 18.0, poll_interval_minutes=30, lookahead_minutes=120, fetch_fn=fetch
    )

    first = source.get_forecast(now=NOW)
    assert first is not None and first.surface_state == "snow"
    assert calls["n"] == 1

    # 10 min later — cache is fresh, no extra fetch, same object returned.
    cached = source.get_forecast(now=NOW + timedelta(minutes=10))
    assert cached is first
    assert calls["n"] == 1

    # 31 min later — cache is stale, refetched.
    refreshed = source.get_forecast(now=NOW + timedelta(minutes=31))
    assert refreshed is not None and refreshed.surface_state == "snow"
    assert calls["n"] == 2


def test_failed_fetch_keeps_last_good_forecast() -> None:
    state = {"healthy": True}

    def fetch(_lat: float, _lon: float) -> dict | None:
        if state["healthy"]:
            return _payload([_step("2026-06-23T21:30:00Z", pcat=1, t=-1.0)])
        return None

    source = SMHIForecastSource(
        59.3, 18.0, poll_interval_minutes=30, lookahead_minutes=120, fetch_fn=fetch
    )

    good = source.get_forecast(now=NOW)
    assert good is not None and good.surface_state == "snow"

    state["healthy"] = False
    # Cache stale, fetch now fails → keep last good forecast (fail-safe).
    kept = source.get_forecast(now=NOW + timedelta(minutes=31))
    assert kept is good


def test_fetch_exception_yields_none_when_never_cached() -> None:
    def boom(_lat: float, _lon: float) -> dict:
        raise RuntimeError("network down")

    source = SMHIForecastSource(59.3, 18.0, fetch_fn=boom)
    assert source.get_forecast(now=NOW) is None


def test_returns_weatherforecast_type() -> None:
    source = SMHIForecastSource(
        59.3,
        18.0,
        fetch_fn=lambda _lat, _lon: _payload(
            [_step("2026-06-23T20:30:00Z", pcat=5, t=0.5)]
        ),
    )
    forecast = source.get_forecast(now=NOW)
    assert isinstance(forecast, WeatherForecast)
    assert forecast.surface_state == "ice"
    assert forecast.confidence == "medium"
    assert forecast.lookahead_minutes == 60
