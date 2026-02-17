"""
Camera-to-Camera Prophecy — Visual Cross-Validation (Phase 7).

Creates testable predictions by using the Physics Engine output to forecast
when a queue tail will reach the next upstream camera.  On subsequent ticks,
the Vision Engine's CapacityState for that camera is checked to verify or
refute the prediction.

This produces a **closed-loop, self-validating** accuracy metric using only
data we control — no external radar or ground-truth feed required.

Flow
----
1. Physics Engine emits ``QueuePrediction`` for Camera N (bottleneck).
2. ``record_prophecies()`` finds Camera N-1 (next upstream by chainage),
   computes ETA, and logs a ``Prophecy`` with ``status = "pending"``.
3. On later ticks, ``evaluate_pending()`` checks if the time has arrived
   and inspects Camera N-1's ``CapacityState``.  If it shows a capacity
   drop, the Prophecy is ``VERIFIED_SUCCESS``; otherwise ``FAILED``.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from bisect import bisect_left
from datetime import datetime, timedelta

from pydantic import BaseModel, Field

from src.models import CapacityState, QueuePrediction

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Minimum ETA (minutes) for a prophecy to be worth tracking.
MIN_ETA_MINUTES: float = 1.0

#: Maximum ETA (minutes).  Beyond this the queue might dissolve.
MAX_ETA_MINUTES: float = 30.0

#: Tolerance window (seconds) around the predicted impact time.
#: Since ticks are 60 s, we allow ±90 s to catch the right tick.
EVALUATION_TOLERANCE_SECONDS: float = 90.0

#: Expiry horizon — prophecies older than this past impact time are expired.
EXPIRY_MINUTES: float = 30.0

#: Capacity drop threshold (fraction of free-flow) to consider "queue arrived".
#: e.g. 0.50 means if capacity is < 50 % of the camera's lane-based free-flow
#: (num_lanes × 2200 VPH), we consider the queue present.
CAPACITY_DROP_FRACTION: float = 0.50

#: Free-flow per-lane capacity (VPH) used to derive camera baseline.
FREE_FLOW_PER_LANE_VPH: float = 2200.0


# ---------------------------------------------------------------------------
# Prophecy model
# ---------------------------------------------------------------------------


class Prophecy(BaseModel):
    """A single prediction that a queue will impact a specific upstream camera."""

    prophecy_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    created_at: datetime
    source_camera_id: str = Field(
        description="Camera where the bottleneck was detected"
    )
    target_camera_id: str = Field(
        description="Upstream camera where the queue tail is expected to arrive"
    )
    source_chainage_km: float
    target_chainage_km: float
    growth_speed_kmh: float
    predicted_eta_minutes: float
    predicted_impact_time: datetime
    status: str = Field(
        default="pending",
        description="pending | VERIFIED_SUCCESS | FAILED | EXPIRED",
    )
    evaluation_time: datetime | None = None
    target_capacity_vph: float | None = None
    target_is_anomaly: bool | None = None


# ---------------------------------------------------------------------------
# Evaluation Logger
# ---------------------------------------------------------------------------


class EvaluationLogger:
    """Manages Camera-to-Camera Prophecies and their evaluation.

    Parameters
    ----------
    chainage_map:
        Mapping of camera_id → chainage (km) along the corridor.
    data_dir:
        Directory where ``evaluation_metrics.jsonl`` is written.
    """

    def __init__(
        self,
        chainage_map: dict[str, float],
        data_dir: str = "data",
    ) -> None:
        self._chainage_map = chainage_map
        self._data_dir = data_dir

        # Pre-sorted list of (chainage, camera_id) for binary search
        self._sorted_cameras: list[tuple[float, float | str]] = sorted(
            (ch, cam_id) for cam_id, ch in chainage_map.items()
        )

        self._pending: list[Prophecy] = []
        self._history: list[dict] = []  # Rolling log for dashboard feed
        self._max_history = 100

        # Running stats
        self._total_created = 0
        self._total_verified = 0
        self._total_failed = 0
        self._total_expired = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_prophecies(
        self,
        predictions: list[QueuePrediction],
        now: datetime,
    ) -> list[Prophecy]:
        """Create prophecies from physics engine predictions.

        For each QueuePrediction, find the nearest upstream camera
        and compute when the queue tail will reach it.

        Returns the newly created prophecies.
        """
        new_prophecies: list[Prophecy] = []

        for pred in predictions:
            target = self._find_upstream_camera(pred.origin_chainage_km)
            if target is None:
                continue

            target_cam_id, target_chainage = target
            distance_km = pred.origin_chainage_km - target_chainage

            if distance_km <= 0 or pred.growth_speed_kmh <= 0:
                continue

            eta_minutes = (distance_km / pred.growth_speed_kmh) * 60.0

            if eta_minutes < MIN_ETA_MINUTES or eta_minutes > MAX_ETA_MINUTES:
                continue

            prophecy = Prophecy(
                created_at=now,
                source_camera_id=pred.camera_id,
                target_camera_id=target_cam_id,
                source_chainage_km=pred.origin_chainage_km,
                target_chainage_km=target_chainage,
                growth_speed_kmh=pred.growth_speed_kmh,
                predicted_eta_minutes=round(eta_minutes, 1),
                predicted_impact_time=now + timedelta(minutes=eta_minutes),
            )

            self._pending.append(prophecy)
            new_prophecies.append(prophecy)
            self._total_created += 1
            self._add_to_history(prophecy)

            logger.info(
                f"🔮 Prophecy: queue from {pred.camera_id} → "
                f"{target_cam_id} in {eta_minutes:.1f} min "
                f"(ETA {prophecy.predicted_impact_time.strftime('%H:%M:%S')})"
            )

        if new_prophecies:
            self._write_jsonl(new_prophecies)

        return new_prophecies

    def evaluate_pending(
        self,
        capacity_states: list[CapacityState],
        now: datetime,
    ) -> list[Prophecy]:
        """Evaluate pending prophecies against current vision data.

        Returns prophecies that were resolved this tick.
        """
        # Build a lookup of current capacity by camera_id
        capacity_by_cam: dict[str, CapacityState] = {
            cs.camera_id: cs for cs in capacity_states
        }

        resolved: list[Prophecy] = []
        still_pending: list[Prophecy] = []

        for p in self._pending:
            # Check if past the expiry window
            expiry_cutoff = p.predicted_impact_time + timedelta(
                minutes=EXPIRY_MINUTES
            )
            if now > expiry_cutoff:
                p.status = "EXPIRED"
                p.evaluation_time = now
                resolved.append(p)
                self._total_expired += 1
                logger.debug(f"🔮 Expired: {p.prophecy_id}")
                continue

            # Check if within the evaluation window
            delta_seconds = abs(
                (now - p.predicted_impact_time).total_seconds()
            )
            if delta_seconds > EVALUATION_TOLERANCE_SECONDS:
                # Not yet time — or already past but within expiry
                if now < p.predicted_impact_time:
                    still_pending.append(p)
                    continue
                # Past the tolerance but within expiry — still pending,
                # will be checked every tick until expiry
                still_pending.append(p)
                continue

            # Within the tolerance window — evaluate now
            target_state = capacity_by_cam.get(p.target_camera_id)
            if target_state is None:
                # Camera data not available this tick — keep pending
                still_pending.append(p)
                continue

            p.evaluation_time = now
            p.target_capacity_vph = target_state.estimated_capacity_vph
            p.target_is_anomaly = target_state.is_anomaly

            # Determine queue arrival
            baseline_capacity = (
                target_state.total_lanes * FREE_FLOW_PER_LANE_VPH
            )
            capacity_ratio = (
                target_state.estimated_capacity_vph / baseline_capacity
                if baseline_capacity > 0
                else 1.0
            )

            if target_state.is_anomaly or capacity_ratio < CAPACITY_DROP_FRACTION:
                p.status = "VERIFIED_SUCCESS"
                self._total_verified += 1
                logger.info(
                    f"✅ Prophecy VERIFIED: {p.source_camera_id} → "
                    f"{p.target_camera_id} | capacity={target_state.estimated_capacity_vph:.0f} VPH"
                )
            else:
                p.status = "FAILED"
                self._total_failed += 1
                logger.info(
                    f"❌ Prophecy FAILED: {p.source_camera_id} → "
                    f"{p.target_camera_id} | capacity={target_state.estimated_capacity_vph:.0f} VPH "
                    f"(ratio={capacity_ratio:.2f})"
                )

            self._add_to_history(p)
            resolved.append(p)

        self._pending = still_pending

        if resolved:
            self._write_jsonl(resolved)

        return resolved

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    def get_stats(self) -> dict:
        """Return summary statistics for the evaluation endpoint."""
        total_resolved = self._total_verified + self._total_failed
        hit_rate = (
            self._total_verified / total_resolved if total_resolved > 0 else None
        )
        return {
            "total_prophecies_created": self._total_created,
            "pending": len(self._pending),
            "verified_success": self._total_verified,
            "failed": self._total_failed,
            "expired": self._total_expired,
            "hit_rate": round(hit_rate, 3) if hit_rate is not None else None,
        }

    def get_log(self, limit: int = 50) -> list[dict]:
        """Return the most recent prophecy log entries for the dashboard."""
        return list(reversed(self._history[-limit:]))

    def _add_to_history(self, prophecy: Prophecy) -> None:
        """Add a prophecy event to the rolling history buffer."""
        entry = {
            "prophecy_id": prophecy.prophecy_id,
            "time": prophecy.created_at.strftime("%H:%M:%S"),
            "source": prophecy.source_camera_id.split("_")[-1],  # Short ID
            "target": prophecy.target_camera_id.split("_")[-1],
            "eta_min": prophecy.predicted_eta_minutes,
            "status": prophecy.status,
        }
        if prophecy.evaluation_time:
            entry["eval_time"] = prophecy.evaluation_time.strftime("%H:%M:%S")
        if prophecy.target_capacity_vph is not None:
            entry["capacity_vph"] = round(prophecy.target_capacity_vph, 0)
        self._history.append(entry)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _find_upstream_camera(
        self, origin_chainage_km: float
    ) -> tuple[str, float] | None:
        """Find the nearest camera *upstream* (lower chainage) of the origin.

        On the northbound E4, upstream = south = lower chainage value.
        Returns (camera_id, chainage_km) or None if no upstream camera
        exists.
        """
        if not self._sorted_cameras:
            return None

        chainages = [c[0] for c in self._sorted_cameras]
        idx = bisect_left(chainages, origin_chainage_km)

        # We want the camera just *below* the origin chainage
        if idx <= 0:
            return None  # Origin is at or below the southernmost camera

        upstream_chainage, upstream_cam = self._sorted_cameras[idx - 1]

        # Ensure it's actually a different camera (not the same chainage)
        if abs(upstream_chainage - origin_chainage_km) < 0.01:
            if idx - 2 >= 0:
                upstream_chainage, upstream_cam = self._sorted_cameras[idx - 2]
            else:
                return None

        return str(upstream_cam), float(upstream_chainage)

    def _write_jsonl(self, prophecies: list[Prophecy]) -> None:
        """Append prophecy records to the JSONL file."""
        os.makedirs(self._data_dir, exist_ok=True)
        jsonl_path = os.path.join(self._data_dir, "evaluation_metrics.jsonl")

        try:
            with open(jsonl_path, "a", encoding="utf-8") as f:
                for p in prophecies:
                    record = p.model_dump(mode="json")
                    f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        except OSError as e:
            logger.error(f"Could not write evaluation metrics: {e}")
