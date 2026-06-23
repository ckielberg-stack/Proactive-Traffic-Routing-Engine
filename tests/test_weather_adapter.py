"""Tests for weather/road-condition physics adjustment."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from src.smhi_forecast import WeatherForecast
from src.weather_adapter import WeatherAdapter


def _forecast(state: str, onset: float | None = 30.0) -> WeatherForecast:
    return WeatherForecast(
        surface_state=state,
        onset_minutes=onset,
        confidence="medium",
        reason=f"{state} forecast within 60 min",
        lookahead_minutes=60,
        sample_count=2,
    )


def test_missing_data_falls_back_to_dry_low_confidence() -> None:
    adj = WeatherAdapter().compute([], [], now=datetime(2026, 6, 13, 12, 0, 0))

    assert adj.surface_state == "dry"
    assert adj.free_flow_factor == 1.0
    assert adj.capacity_factor == 1.0
    assert adj.confidence == "low"


def test_wet_weather_degrades_factors() -> None:
    now = datetime(2026, 6, 13, 12, 0, 0)
    adj = WeatherAdapter().compute(
        weather_records=[
            {
                "sample_time": now.isoformat(),
                "surface_temp_c": 8.0,
                "precipitation": "Regn",
                "precip_rain_sum": 0.4,
            }
        ],
        road_condition_records=[],
        now=now,
    )

    assert adj.surface_state == "wet"
    assert adj.free_flow_factor == pytest.approx(0.92)
    assert adj.capacity_factor == pytest.approx(0.90)
    assert adj.confidence == "medium"


def test_snow_weather_degrades_more_than_wet() -> None:
    now = datetime(2026, 6, 13, 12, 0, 0)
    adj = WeatherAdapter().compute(
        weather_records=[
            {
                "sample_time": now.isoformat(),
                "surface_temp_c": -1.0,
                "precipitation": "Snö",
                "precip_snow_water_eq": 0.2,
            }
        ],
        road_condition_records=[],
        now=now,
    )

    assert adj.surface_state == "snow"
    assert adj.free_flow_factor == pytest.approx(0.85)
    assert adj.capacity_factor == pytest.approx(0.75)


def test_ice_road_condition_warning_takes_precedence() -> None:
    now = datetime(2026, 6, 13, 12, 0, 0)
    adj = WeatherAdapter().compute(
        weather_records=[
            {
                "sample_time": now.isoformat(),
                "surface_temp_c": 8.0,
                "precipitation": "Regn",
                "precip_rain_sum": 0.4,
            }
        ],
        road_condition_records=[
            {
                "id": "RC-1",
                "warning": True,
                "condition_text": "Halka, isfläckar",
                "condition_code": "ice",
            }
        ],
        now=now,
    )

    assert adj.surface_state == "ice"
    assert adj.free_flow_factor == pytest.approx(0.75)
    assert adj.capacity_factor == pytest.approx(0.65)
    assert adj.confidence == "high"
    assert [record["id"] for record in adj.warning_records] == ["RC-1"]


def test_stale_weather_falls_back_to_dry_low_confidence() -> None:
    now = datetime(2026, 6, 13, 12, 0, 0)
    adj = WeatherAdapter().compute(
        weather_records=[
            {
                "sample_time": (now - timedelta(hours=2)).isoformat(),
                "surface_temp_c": -2.0,
                "precipitation": "Snö",
                "precip_snow_water_eq": 0.5,
            }
        ],
        road_condition_records=[],
        now=now,
    )

    assert adj.surface_state == "dry"
    assert adj.confidence == "low"
    assert "stale" in adj.reason


def test_custom_factor_lookup_is_used_and_capped() -> None:
    adapter = WeatherAdapter(
        surface_factors={
            "dry": (1.0, 1.0),
            "wet": (1.2, 1.1),
            "snow": (0.8, 0.7),
            "ice": (0.7, 0.6),
        }
    )
    now = datetime(2026, 6, 13, 12, 0, 0)

    adj = adapter.compute(
        weather_records=[
            {
                "sample_time": now.isoformat(),
                "surface_temp_c": 6.0,
                "precipitation": "Regn",
                "precip_rain_sum": 0.2,
            }
        ],
        road_condition_records=[],
        now=now,
    )

    assert adj.surface_state == "wet"
    assert adj.free_flow_factor == 1.0
    assert adj.capacity_factor == 1.0


# ---------------------------------------------------------------------------
# SMHI forecast escalation (TRAFIK-032)
# ---------------------------------------------------------------------------


def test_forecast_escalates_dry_observation_and_flags_proactive_halka() -> None:
    now = datetime(2026, 6, 13, 12, 0, 0)
    adj = WeatherAdapter().compute(
        weather_records=[
            {"sample_time": now.isoformat(), "surface_temp_c": 5.0, "precipitation": None}
        ],
        road_condition_records=[],
        now=now,
        forecast=_forecast("snow", onset=40.0),
    )

    assert adj.surface_state == "snow"
    assert adj.free_flow_factor == pytest.approx(0.85)
    assert adj.capacity_factor == pytest.approx(0.75)
    assert adj.confidence == "medium"
    assert adj.proactive_halka is True
    assert adj.forecast_state == "snow"
    assert adj.forecast_lead_minutes == 40.0
    assert "snow forecast" in adj.reason


def test_forecast_never_downgrades_an_observed_warning() -> None:
    now = datetime(2026, 6, 13, 12, 0, 0)
    adj = WeatherAdapter().compute(
        weather_records=[],
        road_condition_records=[
            {
                "id": "RC-1",
                "warning": True,
                "condition_text": "Halka, isfläckar",
                "condition_code": "ice",
            }
        ],
        now=now,
        forecast=_forecast("wet", onset=20.0),
    )

    # A milder forecast cannot relax an authoritative ice warning.
    assert adj.surface_state == "ice"
    assert adj.confidence == "high"
    assert adj.proactive_halka is False
    assert adj.forecast_state == "wet"


def test_forecast_no_worse_than_observation_does_not_pre_stage() -> None:
    now = datetime(2026, 6, 13, 12, 0, 0)
    adj = WeatherAdapter().compute(
        weather_records=[
            {
                "sample_time": now.isoformat(),
                "surface_temp_c": -1.0,
                "precipitation": "Snö",
                "precip_snow_water_eq": 0.2,
            }
        ],
        road_condition_records=[],
        now=now,
        forecast=_forecast("wet", onset=15.0),
    )

    assert adj.surface_state == "snow"
    assert adj.proactive_halka is False
    assert adj.forecast_state == "wet"


def test_no_forecast_keeps_observed_only_behaviour() -> None:
    now = datetime(2026, 6, 13, 12, 0, 0)
    adj = WeatherAdapter().compute([], [], now=now, forecast=None)

    assert adj.surface_state == "dry"
    assert adj.confidence == "low"
    assert adj.proactive_halka is False
    assert adj.forecast_state == "dry"
    assert adj.forecast_lead_minutes is None
