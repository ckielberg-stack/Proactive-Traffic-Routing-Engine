#!/usr/bin/env python3
"""
Unified PTRE Entry Point — FastAPI + Tick Loop.

Runs the Operator Decision Support API (FastAPI/uvicorn) and the 60-second
tick-based main loop in a single process.  The tick loop executes in a
background thread via ``asyncio.to_thread`` so it never blocks the async
event loop serving API requests.

Each tick's output (CapacityState, QueuePrediction, VMSRecommendation) is
injected into the Operator API's in-memory state so all ``/api/v1/operator/*``
endpoints serve **live** data.

The Camera-to-Camera Prophecy evaluator (``EvaluationLogger``) is also wired
in here — it records predictions and evaluates them against subsequent ticks.

Usage
-----
    python main.py              # continuous (default)
    python main.py --once       # single tick then exit
    python main.py --port 8081  # custom API port
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import signal
import sys
from contextlib import asynccontextmanager
from typing import Any

import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from config import API_KEY, CAMERA_COORDS, CAMERA_IDS, DATA_DIR, INTERVAL_SECONDS
from main_loop import build_camera_chainage_map, setup_file_logger, tick_once
from src.evaluation_logger import EvaluationLogger
from src.incident_builder import build_incident_reports
from src.operator_api import (
    app as operator_app,
    set_pipeline_snapshot,
)

logger = logging.getLogger("ptre.main")

# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------

_shutdown_event = asyncio.Event()
_eval_logger: EvaluationLogger | None = None


# ---------------------------------------------------------------------------
# Camera ID resolution (respects exclusions)
# ---------------------------------------------------------------------------


def _resolve_camera_ids() -> list[str]:
    """Return active camera IDs, excluding any in excluded_cameras.json."""
    camera_ids = list(CAMERA_IDS)
    excluded_file = os.path.join(DATA_DIR, "excluded_cameras.json")
    try:
        with open(excluded_file, "r", encoding="utf-8") as f:
            excluded = set(json.load(f))
        camera_ids = [c for c in camera_ids if c not in excluded]
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    return camera_ids


# ---------------------------------------------------------------------------
# Tick loop (runs in background thread via asyncio.to_thread)
# ---------------------------------------------------------------------------


async def _tick_loop_background(
    *,
    run_once: bool = False,
    interval: int = INTERVAL_SECONDS,
) -> None:
    """Run tick_once in a thread and inject results into the API + evaluator."""
    global _eval_logger

    chainage_map = build_camera_chainage_map()
    _eval_logger = EvaluationLogger(
        chainage_map=chainage_map,
        data_dir=DATA_DIR,
    )

    logger.info(f"🚀 Tick loop started (interval={interval}s, once={run_once})")

    while not _shutdown_event.is_set():
        camera_ids = _resolve_camera_ids()

        try:
            # Run the synchronous tick in a thread so we don't block uvicorn
            result = await asyncio.to_thread(tick_once, camera_ids)

            # --- Inject into Operator API state (single atomic snapshot) ---
            incidents = build_incident_reports(
                result.capacity_states,
                camera_coords=CAMERA_COORDS,
            )
            set_pipeline_snapshot(
                incidents=incidents,
                predictions=result.queue_predictions,
                vms_statuses=result.vms_statuses,
                recommendations=result.vms_recommendations,
                last_tick_time=result.timestamp,
            )

            # --- Evaluation Logger ---
            _eval_logger.evaluate_pending(
                result.capacity_states, result.timestamp
            )
            _eval_logger.record_prophecies(
                result.queue_predictions, result.timestamp
            )

            stats = _eval_logger.get_stats()
            logger.info(
                f"🔮 Prophecies: {stats['pending']} pending, "
                f"{stats['verified_success']} verified, "
                f"{stats['failed']} failed, "
                f"hit_rate={stats['hit_rate']}"
            )

        except Exception as e:
            logger.error(f"💥 Tick error: {e}", exc_info=True)

        if run_once:
            break

        # Sleep with shutdown check (1-second granularity)
        try:
            await asyncio.wait_for(
                _shutdown_event.wait(), timeout=interval
            )
            break  # shutdown_event was set
        except asyncio.TimeoutError:
            pass  # Normal — interval elapsed, run next tick


# ---------------------------------------------------------------------------
# FastAPI lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start the tick loop as a background task alongside the API server."""
    setup_file_logger(DATA_DIR)

    # Parse CLI args (uvicorn may add its own — we only look at ours)
    run_once = "--once" in sys.argv

    task = asyncio.create_task(
        _tick_loop_background(run_once=run_once)
    )

    yield

    # Shutdown
    logger.info("🛑 Shutting down tick loop...")
    _shutdown_event.set()
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    logger.info("👋 Tick loop stopped.")


# ---------------------------------------------------------------------------
# Application assembly
# ---------------------------------------------------------------------------

# Attach lifespan to the existing operator API app
operator_app.router.lifespan_context = lifespan


# --- Additional endpoint: evaluation stats ---


@operator_app.get("/api/v1/evaluation/stats")
async def evaluation_stats() -> dict[str, Any]:
    """Return Camera-to-Camera Prophecy accuracy statistics."""
    if _eval_logger is None:
        return {
            "status": "not_initialized",
            "message": "Tick loop has not started yet.",
        }
    return _eval_logger.get_stats()


@operator_app.get("/api/v1/evaluation/log")
async def evaluation_log(limit: int = 50) -> dict[str, Any]:
    """Return the prophecy event log for the dashboard feed."""
    if _eval_logger is None:
        return {"entries": [], "stats": {}}
    return {
        "entries": _eval_logger.get_log(limit=limit),
        "stats": _eval_logger.get_stats(),
    }


# --- Static files (TMC Dashboard) ---
_STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")


@operator_app.get("/", include_in_schema=False)
async def dashboard_root():
    """Serve the TMC control room dashboard."""
    return FileResponse(os.path.join(_STATIC_DIR, "index.html"))


operator_app.mount(
    "/static", StaticFiles(directory=_STATIC_DIR), name="static"
)


# ---------------------------------------------------------------------------
# Signal handling
# ---------------------------------------------------------------------------


def _handle_signal(sig, frame):
    logger.info(f"Received signal {sig}, shutting down...")
    _shutdown_event.set()


signal.signal(signal.SIGINT, _handle_signal)
signal.signal(signal.SIGTERM, _handle_signal)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PTRE — Unified operator API + tick loop"
    )
    parser.add_argument(
        "--once", action="store_true", help="Run one tick only then exit"
    )
    parser.add_argument(
        "--port", type=int, default=8081, help="API server port (default: 8081)"
    )
    parser.add_argument(
        "--host", type=str, default="0.0.0.0", help="API server host"
    )
    args = parser.parse_args()

    if not API_KEY:
        print("❌ Missing API key. Set TRAFIKVERKET_API_KEY in .env")
        sys.exit(1)

    uvicorn.run(
        operator_app,
        host=args.host,
        port=args.port,
        log_level="info",
    )
