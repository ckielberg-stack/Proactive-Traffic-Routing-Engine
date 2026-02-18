"""
Anomaly Event Store — append-only JSONL persistence for anomaly events.

Each anomaly is stored as one JSON line in ``data/anomaly_log.jsonl`` with
the annotated image path, camera info, reason, and confidence.  This file
survives restarts so the dashboard can always show the full history.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)

# In-memory counter that survives across ticks (but not process restarts)
_total_anomalies: int = 0


def _log_path(data_dir: str) -> str:
    return os.path.join(data_dir, "anomaly_log.jsonl")


def record_anomaly(
    data_dir: str,
    *,
    timestamp: datetime,
    camera_id: str,
    camera_name: str,
    anomaly_reason: str | None,
    confidence: float,
    vehicle_count: int,
    capacity_vph: float,
    image_path: str | None,
) -> None:
    """Append one anomaly event to the JSONL log."""
    global _total_anomalies

    event = {
        "timestamp": timestamp.isoformat(),
        "camera_id": camera_id,
        "camera_name": camera_name,
        "anomaly_reason": anomaly_reason or "unknown",
        "confidence": round(confidence, 3),
        "vehicle_count": vehicle_count,
        "capacity_vph": round(capacity_vph, 1),
        "image_path": image_path,
    }

    path = _log_path(data_dir)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
        _total_anomalies += 1
        logger.info(
            "🚨 Anomaly logged: %s — %s (conf=%.2f, img=%s)",
            camera_id, anomaly_reason, confidence,
            os.path.basename(image_path) if image_path else "none",
        )
    except Exception as e:
        logger.error("Failed to write anomaly log: %s", e)


def get_anomalies(
    data_dir: str,
    *,
    limit: int = 100,
    camera_id: str | None = None,
) -> list[dict[str, Any]]:
    """Read anomaly events from JSONL (most recent first)."""
    path = _log_path(data_dir)
    if not os.path.exists(path):
        return []

    events: list[dict] = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    evt = json.loads(line)
                    if camera_id and evt.get("camera_id") != camera_id:
                        continue
                    events.append(evt)
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        logger.error("Failed to read anomaly log: %s", e)

    # Most recent first, limited
    events.reverse()
    return events[:limit]


def get_total_count(data_dir: str) -> int:
    """Return total anomaly count (fast in-memory counter + file fallback)."""
    global _total_anomalies
    if _total_anomalies > 0:
        return _total_anomalies

    # On first call after restart, count lines in file
    path = _log_path(data_dir)
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                _total_anomalies = sum(1 for line in f if line.strip())
        except Exception:
            pass

    return _total_anomalies
