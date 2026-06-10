# PTRE Technical Audit & Improvement Plan

> Date: 2026-06-10 · Scope: full repository at commit `cb62bc3` · Method: full read of the pipeline core (`main.py`, `main_loop.py`, all of `src/`, configs, tests), lighter review of `roi_helper.py`, `collect.py`, `dashboard.py`, and `static/`. The unit test suite was executed during the audit (223 passed, ~5s, Python 3.11). No code was modified.

---

## 1. Executive Summary

**Overall health: C+.** PTRE is an unusually well-documented, well-tested *prototype* — 223 fast, behavior-asserting unit tests, clean Pydantic data contracts, an honest README, and ADR/handoff docs most production services lack. However, the audit found that several of the system's headline behaviors are broken or inert *in the wired-together production path*, precisely in the orchestration seam that has no test coverage, and there is no CI to catch regressions.

**Top 3 risks:**
1. **The headline metric is structurally broken**: `proxy_ground_truth_active` (the T₂−T₁ "AI beat the human" value proposition) can never be `True` with the shipped `vms_config.json` — the matching logic only works on test-fixture names (§3-C1).
2. **The predictive chain is desensitized**: sensor data is never passed into the vision engine, so estimated bottleneck capacity essentially never drops under congestion, and the physics trigger requires measured inflow to exceed *theoretical maximum* road capacity (§3-C2).
3. **No CI, no auth, unpinned deps**: nothing enforces the test suite; all API endpoints (including log disclosure and on-demand YOLO) are unauthenticated on `0.0.0.0` (§3-S1, §3-O1).

**Top 3 opportunities:**
1. A 1-day CI + config-integration-test investment would have caught every Critical/High correctness finding here, and will catch the next ones.
2. The replay evaluator (`src/replay_evaluator.py`) is the right foundation — extending replay fixtures to cover the orchestration path gives offline end-to-end verification without API keys.
3. Deleting/quarantining the legacy `collect.py`/`dashboard.py` path removes ~1,250 duplicated lines and halves the Docker deployment's API/compute load.

---

## 2. Repo Map

**Purpose.** Open-source research prototype for the E4/E20 Stockholm corridor: YOLO vehicle detection on 46 public traffic cameras every 60s → LWR kinematic-wave queue-tail prediction → preemptive VMS (variable message sign) recommendations, plus an operator dashboard and DATEX II export. Intended users: traffic-ops researchers and contributors; explicitly not production-certified (README is honest about this).

**Stack.** Python 3.12 (Dockerfile; runs on 3.11), FastAPI/uvicorn, Ultralytics YOLOv8n, OpenCV, Shapely, Pydantic v2, pytest. Vanilla JS + Jinja2 dashboard. Docker Compose deployment. License AGPL-3.0 (consistent with the Ultralytics AGPL dependency).

**Architecture.** A 60-second tick (`main_loop.tick_once`, main_loop.py:1148) concurrently fetches cameras/sensors/VMS-proxy/travel-times via `ThreadPoolExecutor`, runs YOLO per camera, smooths density (EMA), runs the LWR physics engine, generates VMS recommendations, and persists JSONL. `main.py` wraps this loop in an asyncio task alongside the FastAPI app (`src/operator_api.py`), injecting each tick's results into locked in-memory state.

| Area | Contents |
|---|---|
| `main.py` (822 ln) | Canonical entry point: FastAPI assembly, dashboard data endpoints, tick-loop driver |
| `main_loop.py` (1,590 ln) | Tick orchestration: API clients, fetchers, sensor fusion, persistence |
| `config.py` | Camera/sensor/route IDs, coordinates, thresholds (hand-curated corridor data) |
| `src/` | Pipeline modules: vision, ROI mapping, density smoothing, physics, calibration, VMS orchestration, evaluation, operator API, Pydantic models |
| `collect.py`, `dashboard.py` | **Legacy** duplicated pipeline + dashboard — still wired into Docker Compose |
| `roi_helper.py` (1,240 ln) | Interactive OpenCV ROI/homography calibration tool (desktop-only) |
| `tests/` (16 modules, 223 tests) | Unit tests for all `src/` modules + pure helpers of `main_loop` |
| `camera_config.json` | 46 cameras with ROI polygons (48 ROIs: 46 NB / 2 SB); **0 homography matrices configured** |
| `vms_config.json` | 8 VMS gantries with chainage |
| `docs/` | Handoff, roadmap, ADRs, issue folders, learning notes, replay baselines — exemplary |
| `static/`, `templates/` | Dashboard pages (8 pages) |

**Surprises found during mapping:**
- No CI configuration of any kind (`.github/` does not exist).
- `docker-compose.yml` runs *both* the legacy collector and the new unified loop — two containers each fetching all 46 cameras and running YOLO every 60s.
- BEV homography ("Expert Audit Fix 2") is implemented but unused: zero cameras have a `homography_matrix` (verified against `camera_config.json`).
- Committed artifacts: `static/index.html.bak`, `discovered_cameras.json` (139 KB generated dump); `AGENTS.md:3` references a personal machine path (`/Users/chips/.codex/agents.md`).

---

## 3. Audit Report

Severity scale: **Critical** = headline behavior wrong; **High** = materially undermines correctness/safety/operability; **Medium** = real cost, contained; **Low** = hygiene. Each finding is labeled **[fact]** (verifiable in code) or **[judgment]**.

### 3.1 Correctness & code quality — *the ugly parts*

**C1 · Critical · [fact] — Proxy ground-truth matching can never succeed in production.**
`_match_proxy_ground_truth` (src/operator_api.py:194-217) matches a recommendation to an active speed advisory by (a) `status.vms_id == recommendation.vms_id` or (b) `proxy_road in recommendation.vms_name`. Neither can hold with shipped data:
- Active proxy statuses carry Trafikverket deviation IDs (`vms_id=dev_id`, main_loop.py:880-887, e.g. `SE_STA_SPEEDMANAGEMENTID_…`), while recommendations carry gantry IDs (`VMS-4001`…`VMS-4008`, vms_config.json). Never equal.
- `proxy_road` is `"E4"`, but the real gantry names are `"Hallunda södra"`, `"Fittja"`, `"Kungens Kurva"`, … — none contain a road string. The unit test passes only because its fixture invents `vms_name="Kungens Kurva E4"` (tests/test_operator_api.py:91).
**Consequence:** `proxy_ground_truth_active` — the project's central T₂−T₁ evaluation flag (README §VMS Ground-Truth Strategy) — is always `False`; the DATEX II `operatorActionStatus` is always `"requested"` (operator_api.py:416). The comparison dataset the project is building is silently empty of matches.

**C2 · High · [fact + judgment] — Sensor data never reaches the vision engine, desensitizing the whole predictive chain.**
`fetch_cameras` calls `engine.analyze_multi_roi(frame, meta, roi_mapper)` (main_loop.py:685) and `engine.analyze_array(frame, meta)` (main_loop.py:690) without the `sensor` argument (structurally unavoidable as written: sensors are fetched concurrently in a sibling future). Effects:
1. Speed defaults to 110 km/h (vision_engine.py:231, :336). The congested-capacity branch `capacity = density × lanes × speed` then exceeds and clamps to full `Q_cap` for any density > ~18 veh/km/lane (vision_engine.py:232-236, :355-359) — so `estimated_capacity_vph` **never drops under congestion** (only the aspect-ratio `blocked_lanes` heuristic or a black frame can lower it).
2. The physics gate `capacity_drop = inflow − capacity ≥ 200` (physics_engine.py:220-226) therefore requires measured inflow to exceed *theoretical maximum capacity* + 200 vph (e.g. > 6,200 vph on 3 lanes) — queue predictions are far rarer than the model intends. [judgment on magnitude; the wiring gap is fact]
3. The black-image sensor-fusion path (vision_engine.py:418-433) and anomaly cases 2–3 (vision_engine.py:557-573) are dead code in the live pipeline.

**C3 · High · [fact] — Travel direction is hardcoded northbound; southbound ROIs can trigger northbound predictions.**
physics_engine.py:241: `direction = "northbound"  # TODO: derive from CapacityState/ROI road_id`. Meanwhile `_aggregate_multi_roi_capacity` takes the **max density across all ROIs** including the two `E4_Southbound` ROIs (main_loop.py:588-591). Southbound congestion at those cameras produces a *northbound* queue prediction and recommends VMS activation on the wrong carriageway.

**C4 · Medium · [fact] — Density smoothing happens after capacity/anomaly were computed from raw density.**
The EMA smoother overwrites `observed_density_veh_km_lane` post-hoc (main_loop.py:1205-1210), but `estimated_capacity_vph`, `is_anomaly`, and `anomaly_reason` were already derived from the *raw* density inside the vision engine. A transient occlusion still flags `is_anomaly` → incident report + annotated image, defeating the stated intent of "Expert Audit Fix 3" for everything except the physics trigger.

**C5 · Medium · [fact] — Inconsistent free-flow constants produce phantom capacity drops.**
Baseline 2,200 vph/lane in src/evaluation_logger.py:60 and src/incident_builder.py:12 vs. `Q_CAP` 2,000 vph/lane in src/vision_engine.py:87. Every `density_exceeds_k_critical` incident therefore reports a fixed ≥9.1% "capacity drop" even when the engine computed zero drop, and prophecy verification (evaluation_logger.py:249-258) effectively reduces to `is_anomaly` because the capacity ratio can rarely fall below 0.5.

**C6 · Medium · [fact] — `python main.py --once` does not exit.**
The lifespan reads `--once` (main.py:213) and the tick task breaks after one tick (main.py:189-190), but nothing stops uvicorn. README Quick Start step 3 ("run a single tick and exit", README:71-72) is wrong; `main_loop.py --once` behaves correctly.

**C7 · Low · [fact] — Silent exception swallowing in fallback paths.** e.g. `except Exception: pass` around cache persistence (main.py:587-588) and the cascading bare excepts in `_get_camera_info` (main.py:591-600); anomaly count fallback (src/anomaly_store.py:113-114). Acceptable for a prototype, but persistent failures become invisible.

**C8 · Low · [fact] — Duplicate logic.** ROI-mask filtering implemented twice in vision_engine (analyze_array lines 194-208 vs. `_detect_vehicles` lines 479-505); exclusion-zone check reimplemented in main.py:699-707 despite `ROIMapper.is_excluded` (roi_mapper.py:201-206); camera-fetch XML built in three places (main.py:548-561, main_loop.py:627-637, collect.py).

### 3.2 Architecture & design

**A1 · High · [fact] — Legacy pipeline still deployed in parallel.**
`docker-compose.yml` runs `collector` (= `collect.py`, the Dockerfile `ENTRYPOINT`) *and* `dashboard` (= `main.py`, which runs the full tick loop). Both fetch all 46 cameras and run YOLO every 60s → double Trafikverket API load, double CPU, two writers into the shared `./data` volume. `dashboard.py` (358 ln) duplicates endpoints. README correctly labels these "legacy" yet they remain the default deployment. The Dockerfile `HEALTHCHECK` checks `collector.log` and is inherited by the dashboard container, where it is meaningless.

**A2 · Medium · [fact] — App assembly by cross-module mutation.**
`main.py` mutates the imported `operator_app` (lifespan injected at main.py:237; routes registered across two files; module-level mutable state at main.py:69-77 and main_loop.py:207-213, 1143-1145). Import order becomes load-bearing; the assembled app cannot be constructed twice (e.g. in tests).

**A3 · Medium · [fact + judgment] — `main_loop.py` is a 1,590-line god module** mixing HTTP clients, XML query construction, geometry, anomaly detection, image annotation, persistence, and orchestration at repo root. Its pure helpers are tested; its orchestration (`tick_once`, `fetch_*`, `_persist_tick`, `_aggregate_multi_roi_capacity`) is not — and that is exactly where C1–C4 live.

**A4 · Low · [fact] — Leaky layering:** `src/vms_orchestrator.py:28` imports root-level `config`, while root modules import `src/` — the `src` package is not self-contained.

### 3.3 Security

**S1 · High · [fact] — No authentication on any endpoint; default bind `0.0.0.0`** (main.py:809; compose publishes 8080). Includes `/api/v1/logs` (internal log disclosure, main.py:494), `/api/v1/camera-detections/{id}` (unauthenticated requests trigger YOLO inference + outbound fetch — cheap DoS, main.py:650), and the operator/DATEX surfaces. For the stated VPS-deployment goal (docs/project_state.md:107) this is the top gap. *Calibrated note: acceptable for localhost research use; not for any networked deployment.*

**S2 · Low · [fact] — Secrets hygiene is good.** `.env` never appears in git history; `.env.example` placeholder only; API key sourced from env (config.py:13); `.gitignore` covers `data/`, `storage/`, `*.pt`. Path traversal on the anomaly-image endpoint is mitigated (main.py:768-770). SECURITY.md exists and is sensible.

**S3 · Medium · [fact] — Unpinned dependencies, no lockfile** (`requirements.txt` uses only `>=`). Builds are non-reproducible and silently absorb future breaking changes/CVEs in heavy, fast-moving deps (ultralytics, opencv). CVE posture cannot be asserted without pinning.

### 3.4 Testing

**T1 · High · [fact] — Zero coverage of the orchestration seam.** No test imports `main.py`; `main_loop` coverage is limited to pure helpers (tests/test_route_linear_mapping.py, tests/test_sensor_anomaly.py). Every Critical/High correctness finding above lives in untested glue.

**T2 · Medium · [fact] — Fixtures diverge from shipped config.** tests/test_operator_api.py:91 invents gantry name `"Kungens Kurva E4"`; no test exercises `_match_proxy_ground_truth` against the real `vms_config.json` (which is why C1 survived).

**T3 · Low · [fact] — Live-API test collected by default.** `tests/smoke_test.py` matches pytest's `*_test.py` pattern, so bare `pytest` hits the Trafikverket API; exclusion relies on remembering `--ignore` (README:260). No pytest config or markers exist.

**Strength:** the 223 unit tests are genuinely good — fast (5.4s), deterministic, and they assert behavior (e.g. monotonicity of wave speeds vs. inflow, tests/test_physics_engine.py:209-238), not just execution.

### 3.5 Performance

**P1 · High · [fact] — Blocking HTTP inside async endpoints.** `requests.post/get` run directly in `async def` handlers: `_get_camera_info` (main.py:563) called from async paths, camera-image proxy (main.py:627, 15s timeout), detection endpoint image fetch (main.py:671). Each call freezes the entire event loop — the whole dashboard and operator API stall for up to 15s per slow upstream request.

**P2 · High · [fact] — Sequential fetch + YOLO for 46 cameras inside a 55s budget.** `fetch_cameras` loops camera-by-camera (main_loop.py:653-741); each image fetch alone permits 30s × 3 retries (main_loop.py:183-195). On overrun, `future_cameras.result(timeout=55)` (main_loop.py:1185) raises — but the `with ThreadPoolExecutor` exit then blocks until the work finishes anyway, so ticks silently stretch beyond 60s and the "60-second cadence" becomes aspirational under load.

**P3 · Medium · [fact] — Unbounded data growth.** Daily `sensor_data.jsonl`, `anomaly_log.jsonl`, `evaluation_metrics.jsonl` are append-only with no pruning (`retention.py` governs images only). Months of operation will exhaust disk.

**P4 · Low · [fact] — BEV classification recomputes polygon transforms per detection** (roi_mapper.py:291-297) — O(detections × ROIs) `perspectiveTransform` calls; currently moot (0 homographies configured).

### 3.6 Dependencies

**D1 · Medium · [fact] — Dead and oversized deps.** `schedule` and `lxml` are declared (requirements.txt:3, :21) but never imported anywhere (grep-verified); `opencv-python` (GUI build) is used where `opencv-python-headless` suffices in Docker — the Dockerfile installs X libs to compensate. `httpx` is present (test-only) and would also be the right replacement for blocking `requests` in async paths (P1).

**D2 · Low · [fact] — Python version drift:** Dockerfile `python:3.12-slim`, README "Python 3.12", audit environment ran the suite on 3.11 without issue. Harmless but unstated.

### 3.7 DevEx & operations

**O1 · High · [fact] — No CI.** No `.github/`, no workflows. The excellent test suite enforces nothing.

**O2 · Medium · [fact] — No lint/format/type-check configuration** (no `pyproject.toml`, ruff/mypy/flake8 config) despite the codebase being fully type-annotated — free value left unclaimed.

**O3 · Low · [fact] — Committed clutter / broken references:** `static/index.html.bak`; `discovered_cameras.json` (139 KB generated artifact); `AGENTS.md:3` points at a personal absolute path.

### 3.8 Documentation

**DOC1 · Medium · [fact] — README documents endpoints that don't exist in the canonical entry point.** README:143 claims camera exclusions are "toggled via `DELETE /api/cameras/{id}` and `POST /api/cameras/{id}/restore` from the camera page" — those routes exist only in legacy `dashboard.py:193-205`; `main.py` doesn't define them and `static/cameras.js` never calls them. The exclusion feature is unreachable from the shipped UI.

**DOC2 · Low · [fact] — Stale counts/status:** docs/project_state.md says "150 tests" (actual: 223) and lists "Wire main_loop into operator_api" as future work (done in main.py); handoff.md and config.py:25 say 53 cameras (46 configured).

**Strength:** Documentation is otherwise a model for prototypes — honest maturity framing, handoff doc, ADR table, per-issue folders, learning notes, replay baselines.

### 3.9 What the repo does well (preserve these)

1. **Test culture in `src/`** — behavior-asserting, fast, well-organized; physics tests encode real domain invariants.
2. **Pydantic data contracts** (src/models.py) — every inter-module boundary is typed and documented; provenance fields (`inflow_source`, `data_confidence`) are thoughtful.
3. **Honest prototype framing** — README/SECURITY.md/CONTRIBUTING.md set correct expectations; AGPL licensing handled correctly w.r.t. Ultralytics.
4. **Replay evaluator + fixtures** (src/replay_evaluator.py, tests/fixtures/) — offline, versioned metrics without API keys; the right seed for evaluation discipline.
5. **Atomic state snapshotting** with `RLock` (operator_api.py:160-175) and graceful-degradation patterns (ROI fallback, disk-cache fallback).
6. **Smart image retention** (retention.py) — deliberate bounded disk policy for frames.

---

## 4. Improvement Strategy

### Theme 1 — The production seam is unverified (explains C1–C4, T1, T2, DOC1)
Unit tests verify modules in isolation against invented fixtures; nothing verifies the *wired* system against the *shipped configs*. **Target state:** an offline integration test that runs `tick_once` (with HTTP mocked/replayed) over real `camera_config.json`/`vms_config.json` and asserts end-to-end invariants (a southbound-only congestion produces no northbound rec; a synthetic active proxy on a gantry's segment flips `proxy_ground_truth_active`). **Principle:** test the configuration as part of the system, because here the config *is* the product.

### Theme 2 — Nothing is enforced (explains O1, O2, T3, S3/D2)
**Target state:** CI runs `pytest` (smoke test excluded via marker), `ruff check` + `ruff format --check`, on every PR; dependencies pinned via a lockfile (`pip-compile` or `uv`). **Done means:** a PR with a failing test or lint error cannot merge; `pip install -r requirements.lock` is reproducible.

### Theme 3 — The live pipeline diverged from the modeled design (explains C2, C4, C5)
The "Expert Audit Fixes" were implemented module-locally but the orchestration starves them of inputs. **Target state:** capacity/anomaly derivation moves *after* fusion — the vision engine outputs only detections + density; capacity, anomaly, and smoothing are computed in one place (physics layer or a new fusion step) with per-node speed from `node_traffic_states`. One free-flow constant, defined once. **Principle:** single source of truth for traffic-engineering constants and for the density→capacity decision.

### Theme 4 — Legacy duplication doubles cost (explains A1, DOC1, parts of C8)
**Target state:** `collect.py`/`dashboard.py` removed (or moved to `legacy/` with a deprecation note), Docker Compose runs one service, Dockerfile entrypoint is `main.py`, README matches. **Done means:** one process fetches cameras; exclusion endpoints exist in `main.py` or the README claim is removed.

### Theme 5 — Single-process responsiveness limits (explains P1, P2, S1)
**Target state:** camera fetches parallelized (bounded `ThreadPoolExecutor.map` or httpx async) with a per-camera time budget; no blocking HTTP on the event loop; optional bearer-token auth (env var) for non-localhost deployments. **Done means:** tick wall-time < 60s with 46 cameras under realistic latency; dashboard stays responsive during image proxying; deployment docs state the auth posture.

### Explicitly NOT recommended now (effort vs. payoff at prototype maturity)
- **BEV homography rollout** — implemented, needs per-camera field calibration, not engineering work.
- **Multi-corridor / bidirectional generalization** — fix the southbound *mis-trigger* (C3) cheaply; full direction-aware modeling is a research milestone, not a refactor.
- **Database instead of JSONL** — JSONL + replay is appropriate; add pruning, not Postgres.
- **DATEX II XSD validation / NTS conformance** — only worth it when a real downstream consumer exists.
- **Microservice split / message bus** — single process is right for 46 cameras.

---

## 5. Task Plan

Effort: S < 2h · M = half-day · L = 1–2 days · XL = needs breakdown.

### Milestone 0 — Safety net (before touching behavior)

| # | Task | Files | Acceptance criteria | Effort | Risk | Deps |
|---|---|---|---|---|---|---|
| 0.1 | **Add CI workflow** running pytest (excl. smoke) on push/PR | `.github/workflows/ci.yml` | PR with failing test blocks; runs < 5 min | S | None | — |
| 0.2 | **Mark smoke test** with `@pytest.mark.live` + `pyproject.toml` pytest config so bare `pytest` is safe | `tests/smoke_test.py`, `pyproject.toml` | `pytest` runs 223 tests, no network | S | None | — |
| 0.3 | **Config-integration tests**: run `_match_proxy_ground_truth`, `VMSOrchestrator`, and `_aggregate_multi_roi_capacity` against the real `vms_config.json` / `camera_config.json` | `tests/test_config_integration.py` | Test reproducing C1 exists and (initially) fails; C3 southbound case encoded | M | None | 0.1 |
| 0.4 | **Offline `tick_once` integration test** with mocked HTTP (responses fixture) over shipped configs | `tests/test_tick_integration.py` | One green end-to-end tick without network; asserts persistence + snapshot injection | L | Low | 0.3 |
| 0.5 | **Pin dependencies** (lockfile) + ruff config + `ruff check` in CI | `requirements.lock`/`pyproject.toml` | Reproducible install; CI fails on lint | S | Low | 0.1 |

### Milestone 1 — Critical & High correctness fixes

| # | Task | Files | Acceptance criteria | Effort | Risk | Deps |
|---|---|---|---|---|---|---|
| 1.1 | **Fix proxy ground-truth matching (C1)**: match by route-projected chainage/geometry of the deviation (`Deviation.Geometry.WGS84` is already fetched, main_loop.py:853) against gantry chainage, with road-number guard | `src/operator_api.py`, `main_loop.py` (carry geometry into `VMSStatusSnapshot`), `src/models.py` | 0.3's failing test passes; a real SPEEDMANAGEMENT deviation within X km of a gantry flips the flag | L | Medium (touches the metric) | 0.3 |
| 1.2 | **Re-wire fusion before capacity (C2/C4/C5)**: vision outputs detections+density only; capacity/anomaly/smoothing computed post-fetch using `node_traffic_states` speeds; unify 2000/2200 constants in one module | `src/vision_engine.py`, `main_loop.py`, `src/incident_builder.py`, `src/evaluation_logger.py` | Congested k=60, v=30 km/h yields capacity < Q_cap; smoothed density drives `is_anomaly`; single constant source | XL → break down | High (core math) | 0.4 |
| 1.3 | **Stop southbound ROIs triggering northbound predictions (C3)**: filter aggregation to the camera's monitored direction; tag `CapacityState` with direction | `main_loop.py:575-618`, `src/models.py`, `src/physics_engine.py:241` | 0.3's southbound test passes; SB density excluded from NB max | M | Medium | 0.3 |
| 1.4 | **Auth option for non-local deployment (S1)**: optional `PTRE_API_TOKEN` env var enforced by FastAPI dependency; document posture | `src/operator_api.py`, `main.py`, README | With token set, unauthenticated requests get 401; without, behavior unchanged | M | Low | — |
| 1.5 | **Fix `--once` exit (C6)** or correct the README | `main.py` lifespan / README:71 | `python main.py --once` terminates after one tick | S | Low | — |

### Milestone 2 — High-leverage improvements

| # | Task | Files | Acceptance criteria | Effort | Risk | Deps |
|---|---|---|---|---|---|---|
| 2.1 | **Parallelize camera fetch+inference (P2)** with bounded workers and per-camera deadline; log tick wall-time | `main_loop.py:621-742, 1178-1202` | Tick < 60s with simulated 1s/image latency × 46 cameras; overruns logged | L | Medium | 0.4 |
| 2.2 | **Unblock the event loop (P1)**: replace `requests` in async handlers with `httpx.AsyncClient` (already a dependency) or `asyncio.to_thread` | `main.py:541-600, 614-635, 650-746` | No synchronous HTTP on the loop; dashboard responsive during image proxy | M | Low | — |
| 2.3 | **Retire legacy pipeline (A1, DOC1)**: remove/quarantine `collect.py` + `dashboard.py`; compose runs single service; Dockerfile entrypoint `main.py`; port exclusion endpoints into `main.py` (or drop README claim); fix HEALTHCHECK | `docker-compose.yml`, `Dockerfile`, `collect.py`, `dashboard.py`, `main.py`, README | One container; one camera fetcher; README endpoints all exist | L | Medium (deployment change) | 0.4 |
| 2.4 | **Extract orchestration from `main_loop.py` (A3)**: split API clients (`src/trafikverket_client.py`), fusion, persistence; keep `tick_once` thin | `main_loop.py` → `src/` | No file > ~500 ln in the hot path; 0.4 test still green; imports one-directional (`src` self-contained, fixes A4) | XL → break down | Medium | 0.4, 1.2 |

### Milestone 3 — Quality & polish

| # | Task | Files | Acceptance criteria | Effort | Risk | Deps |
|---|---|---|---|---|---|---|
| 3.1 | JSONL retention/pruning policy (P3) | `retention.py`, `main_loop.py` | Files older than N days pruned; configurable | S | Low | — |
| 3.2 | Doc refresh (DOC2): test counts, camera counts (53→46, config.py:25), project_state "What's Next" | `docs/`, `config.py`, README | Docs match code | S | None | — |
| 3.3 | Remove committed clutter (O3): `index.html.bak`, relocate `discovered_cameras.json` to docs/fixtures or ignore, fix `AGENTS.md` path | repo root | Clean tree | S | None | — |
| 3.4 | Deduplicate ROI-mask/exclusion-zone logic (C8) into `ROIMapper` | `src/vision_engine.py`, `main.py`, `src/roi_mapper.py` | One implementation each; tests green | M | Low | 0.4 |
| 3.5 | Remove unused deps (`schedule`, `lxml`), switch to `opencv-python-headless` in Docker (D1) | `requirements.txt`, Dockerfile | Image builds & runs; smaller image | S | Low | 0.5 |
| 3.6 | Type-safety pass: `Literal`/enums for `urgency`, `severity`, `confidence`, prophecy `status` strings | `src/models.py` + consumers | mypy/ruff clean on `src/` | M | Low | 0.5 |

### Quick wins (do immediately — all S effort, high impact)
- **0.1 CI workflow** — converts the existing 223 tests into a guarantee.
- **0.2 smoke-test marker** — removes the "bare pytest hits live API" trap.
- **0.5 pin deps + ruff** — reproducibility for near-zero cost.
- **1.5 `--once` fix** — restores the documented smoke-test workflow.
- **3.3 clutter removal / 3.2 doc counts** — credibility polish for an open-source repo.

### Implementation sketches — top 3 tasks

**1.1 Proxy ground-truth matching.**
Approach: stop matching on names entirely. In `fetch_vms_status`, parse `Deviation.Geometry.WGS84` (already in the INCLUDE list, main_loop.py:853) and project it to corridor chainage with the existing `RouteProjector`; store `chainage_km: float | None` and `road_number` on `VMSStatusSnapshot`. In `_match_proxy_ground_truth`, a recommendation matches an active proxy when the proxy's chainage lies within a window (e.g. ±2 km) of the gantry chainage and the road number matches. Gotchas: deviations outside the corridor project to clamped endpoints — require projection distance-to-route below a threshold before trusting chainage (extend `RouteProjector` to also return offset distance); keep the old vms_id equality as a fast path for a future real TMC feed; update test fixtures to use the *real* config names so this can't regress.

**1.2 Fusion-before-capacity rewire.**
Approach: change `VisionEngine.analyze_multi_roi` to return detections + per-ROI density only (keep API by deprecating capacity fields). Add a `derive_capacity_states(multi_states, node_traffic_states, smoother)` step in the tick after both futures resolve: smooth density first, then decide congestion (`k > k_critical`) on the smoothed value, then compute capacity using the *node-local* speed (TrafficFlow → TravelTime fallback → free-flow), then anomaly flags. Move `Q_CAP`, `K_CRITICAL`, `JAM_DENSITY`, free-flow baseline into a single `src/traffic_constants.py`. Key steps: (1) constants module + mechanical import swap; (2) new fusion function with unit tests for the k=60/v=30 case; (3) switch `tick_once` over; (4) delete capacity math from vision engine. Gotchas: `analyze_array` single-ROI path and `collect.py` also call the old API (less of a problem if 2.3 lands first); the evaluation logger's verification criterion must be re-derived once capacity is meaningful — expect prophecy hit-rates to shift, so snapshot replay metrics (docs/baselines) before/after.

**0.4 Offline tick integration test.**
Approach: monkeypatch `main_loop.api_request` and `fetch_image_bytes` to return canned Trafikverket JSON (camera list, TrafficFlow, Situation, TravelTimeRoute) and a small synthetic JPEG with a known number of car-like rectangles — or stub `VisionEngine._detect_vehicles` to return deterministic detections, avoiding model download in CI. Run `tick_once` against real configs into a `tmp_path` DATA_DIR; assert: JSONL written with expected record types, `set_pipeline_snapshot` received states for all mocked cameras, no exceptions in logs. Gotchas: module-level globals (`_tick_count`, singletons, `DATA_DIR` baked at import) need reset fixtures — patch `main_loop.DATA_DIR` and reset `_vision_engine`/`_density_smoother` globals between tests; keep YOLO out of CI by stubbing at `_detect_vehicles`, not at Ultralytics level.

---

## 6. Open Questions (need a human decision)

1. **Is the legacy collector still collecting anything you need?** `collect.py` gathers weather/road data the new loop doesn't. If that data matters, port the fetcher into the tick before deleting (affects task 2.3 scope).
2. **Deprecation of `dashboard.py` + the camera-exclusion UI**: should exclusion toggling be a supported feature of the canonical app (port endpoints + add UI), or is editing `data/excluded_cameras.json` by hand acceptable for the prototype?
3. **Proxy ground-truth matching tolerance**: what chainage window (±1 km? ±3 km?) counts as "the human acted on the same segment"? This directly defines the headline T₂−T₁ metric — a domain owner should set it.
4. **Deployment posture**: is the VPS deployment (docs/project_state.md) still planned, and will it be network-exposed? That decides whether task 1.4 (auth) is Milestone-1 or can slip.
5. **Tick budget under load**: is occasional >60s tick acceptable (skip-and-log), or must cadence be hard (drop unfinished cameras at the deadline)? Determines the design of task 2.1.
6. **Direction support**: is southbound modeling on the roadmap, or should the two `E4_Southbound` ROIs simply be excluded from physics (cheapest C3 fix)?
