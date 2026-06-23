# 2026-06-23 - SMHI Forecast Proactive Weather Adjustment

## Context

TRAFIK-032 (proposal P5) added the SMHI open-data point forecast (metfcst pmp3g v2,
free, no API key) as an *anticipatory* weather signal on top of the P1 observed feeds
(`WeatherMeasurepoint`/`RoadCondition`). The goal is "anticipate, don't observe":
pre-degrade physics thresholds and pre-stage HALKA advisories *before* friction drops.

## What I Learned

A forecast is only safe as an *escalation-only* input. The adapter folds it in with the
same worst-of rule used for road/weather (`state = worst(observed, forecast)`), so a wrong
or stale forecast can only make PTRE more conservative, never less. The forecast owns a new
output channel — `proactive_halka` — which fires *only* when the forecast is strictly worse
than what is currently observed (the genuinely anticipatory case).

Timezones bite here. The tick loop runs on naive *local* time (`datetime.now()`), but SMHI
`validTime` is always UTC (`...Z`). The forecast source normalizes both to aware-UTC before
windowing; skipping that slides the look-ahead window by 1–2 hours.

Forecasts change slowly, so the source is poll-throttled (cache refreshed every ~30 min, not
every 60-second tick) — cross-tick state held in a lazy singleton, like `TravelTimeCalibrator`.
The SMHI fetch uses its own HTTP client, *not* `api_request`, so tests that mock `api_request`
do not cover it: the tick test fixture must stub `fetch_smhi_forecast` to stay offline.

## Reuse Rules

- A forecast may only escalate the surface state; never let it relax road/weather observations.
- Classify surface from SMHI `pcat` (1/2 → snow, 5/6 → ice, 3/4 → wet; rain with `t` ≤ 0 → ice).
- Normalize forecast `validTime` (UTC) and the tick `now` (naive local) to UTC before comparing.
- Keep network off the per-tick critical path: poll-throttle and serve the last good forecast
  on fetch failure (return `None`/cached — never a dry override).
- Pre-staged advisories use a distinct message (`HALKRISK`) and the southernmost E4 gantry,
  since a corridor forecast has no specific location.
- Stub `fetch_smhi_forecast` (not just `api_request`) to keep tick tests offline.

## Failure Signals

- Look-ahead window is shifted by hours → forecast onset wrong (UTC/local mix-up).
- A milder forecast lowers an active `RoadCondition` ice warning (escalation-only violated).
- Tests hit the live SMHI endpoint (slow/flaky offline) because `fetch_smhi_forecast` is unstubbed.
- The tick fetches SMHI every 60 s instead of every ~30 min (poll throttle bypassed / singleton reset).
- `proactive_halka` fires when observed conditions already match or exceed the forecast.

## Next Checklist

- Forecast confidence stays "medium" — it is predicted, never authoritative like a warning.
- DATEX II `slipperyRoad` export (P7) should consume the forecast escalation, not only observed warnings.
- If multiple corridor reference points are added, keep worst-of aggregation across them.
- Revisit `onset_minutes` drift: it is computed at fetch time and served for up to the poll interval.
