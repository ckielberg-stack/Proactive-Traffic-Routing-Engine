"""
Temporal density smoother for the PTRE tick-based architecture.

Expert Audit Fix 3: Introduces an Exponential Moving Average (EMA) filter
for per-camera visual density to prevent transient occlusions (e.g. a bus
blocking the camera for one frame) from triggering false congestion alerts.

This is the *only* intentional breach of the "stateless tick" ADR-005.
The smoother holds minimal cross-tick state (one float per camera).
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class DensitySmoother:
    """EMA smoother for per-camera visual density.

    Parameters
    ----------
    alpha : float
        Smoothing factor in (0, 1].  Higher → more responsive to new data,
        lower → more damping.  Default 0.4 means a single-frame spike
        contributes only 40% to the smoothed value.
    """

    def __init__(self, alpha: float = 0.4) -> None:
        if not 0.0 < alpha <= 1.0:
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        self.alpha = alpha
        self._state: dict[str, float] = {}

    def update(self, camera_id: str, raw_density: float) -> float:
        """Update EMA and return the smoothed density for *camera_id*.

        On first observation for a camera, the smoothed value is set to
        the raw density (no history to blend with).
        """
        if camera_id not in self._state:
            self._state[camera_id] = raw_density
        else:
            self._state[camera_id] = (
                self.alpha * raw_density
                + (1.0 - self.alpha) * self._state[camera_id]
            )
        return self._state[camera_id]

    def get(self, camera_id: str) -> float | None:
        """Return the last smoothed density for *camera_id*, or None."""
        return self._state.get(camera_id)

    def reset(self, camera_id: str | None = None) -> None:
        """Reset state for one camera (or all cameras if *camera_id* is None)."""
        if camera_id is not None:
            self._state.pop(camera_id, None)
        else:
            self._state.clear()

    @property
    def camera_ids(self) -> list[str]:
        """List of cameras currently tracked."""
        return list(self._state.keys())
