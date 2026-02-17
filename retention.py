"""
Smart Retention Policy for the in-memory image processing pipeline.

Saves camera frames to disk ONLY when:
  1. An anomaly is detected (for human debugging of false positives).
  2. A training sample is due (1 random frame per camera every 4 hours).

All other images are processed in RAM and immediately discarded.
"""

from __future__ import annotations

import json
import logging
import os
import random
from datetime import datetime

from src.models import CapacityState

logger = logging.getLogger(__name__)


class RetentionPolicy:
    """Decide whether to persist a camera frame to disk."""

    ANOMALY_DIR = "storage/anomalies"
    TRAINING_DIR = "storage/training"
    TRAINING_INTERVAL_HOURS = 4

    def __init__(self, base_dir: str = ".") -> None:
        self._base = base_dir
        self._last_training_save: dict[str, datetime] = {}
        self._schedule_file = os.path.join(base_dir, "data", "training_schedule.json")
        self._load_schedule()

    # -- public API -----------------------------------------------------------

    def maybe_retain(
        self,
        raw_bytes: bytes,
        camera_id: str,
        timestamp: datetime,
        capacity_state: CapacityState,
    ) -> str | None:
        """Save image to disk ONLY if retention criteria are met.

        Returns the save path if retained, else None.
        """
        # Rule 1: Anomaly detected → always save
        if capacity_state.is_anomaly:
            path = self._save(
                raw_bytes, camera_id, timestamp, self.ANOMALY_DIR,
                tag=capacity_state.anomaly_reason or "anomaly",
            )
            logger.info(
                "🚨 Anomaly image retained: %s (%s)",
                camera_id, capacity_state.anomaly_reason,
            )
            return path

        # Rule 2: Training sample (1 per camera per 4 hours)
        if self._is_training_sample_due(camera_id, timestamp):
            path = self._save(
                raw_bytes, camera_id, timestamp, self.TRAINING_DIR,
                tag="training",
            )
            self._last_training_save[camera_id] = timestamp
            self._persist_schedule()
            logger.info("📸 Training sample retained: %s", camera_id)
            return path

        return None  # image discarded — normal behavior

    # -- internal helpers -----------------------------------------------------

    def _is_training_sample_due(self, camera_id: str, ts: datetime) -> bool:
        """Check if TRAINING_INTERVAL_HOURS have elapsed since last save."""
        last = self._last_training_save.get(camera_id)
        if last is None:
            # First run: randomize the start offset so all cameras don't
            # trigger at the same time
            if random.random() < 0.1:  # ~10 % chance on first encounter
                return True
            # Register a "virtual" last-save so we don't keep rolling dice
            self._last_training_save[camera_id] = ts
            return False

        elapsed_hours = (ts - last).total_seconds() / 3600
        return elapsed_hours >= self.TRAINING_INTERVAL_HOURS

    def _save(
        self,
        raw: bytes,
        camera_id: str,
        ts: datetime,
        sub_dir: str,
        tag: str = "",
    ) -> str:
        """Write JPEG bytes to a date-organized directory."""
        date_str = ts.strftime("%Y-%m-%d")
        time_str = ts.strftime("%H-%M-%S")
        safe_id = camera_id.replace("/", "_")

        out_dir = os.path.join(self._base, sub_dir, date_str)
        os.makedirs(out_dir, exist_ok=True)

        filename = f"{safe_id}_{time_str}.jpg"
        path = os.path.join(out_dir, filename)

        with open(path, "wb") as f:
            f.write(raw)

        size_kb = len(raw) / 1024
        logger.debug("Retained %s (%.0f KB, %s)", path, size_kb, tag)
        return path

    # -- schedule persistence (best-effort) -----------------------------------

    def _load_schedule(self) -> None:
        """Load training schedule from disk (if available)."""
        try:
            with open(self._schedule_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            for cam_id, iso_str in data.items():
                self._last_training_save[cam_id] = datetime.fromisoformat(iso_str)
            logger.debug("Loaded training schedule: %d cameras", len(data))
        except (FileNotFoundError, json.JSONDecodeError, ValueError):
            pass

    def _persist_schedule(self) -> None:
        """Best-effort save of training schedule to disk."""
        try:
            os.makedirs(os.path.dirname(self._schedule_file), exist_ok=True)
            data = {
                cam_id: ts.isoformat()
                for cam_id, ts in self._last_training_save.items()
            }
            with open(self._schedule_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass  # non-critical
