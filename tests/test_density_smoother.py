"""
Unit tests for the DensitySmoother (Expert Audit Fix 3).
"""

import pytest

from src.density_smoother import DensitySmoother


# ======================================================================
# Construction & validation
# ======================================================================


class TestDensitySmootherInit:
    def test_default_alpha(self) -> None:
        s = DensitySmoother()
        assert s.alpha == 0.4

    def test_custom_alpha(self) -> None:
        s = DensitySmoother(alpha=0.7)
        assert s.alpha == 0.7

    def test_alpha_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="alpha"):
            DensitySmoother(alpha=0.0)

    def test_alpha_must_not_exceed_one(self) -> None:
        with pytest.raises(ValueError, match="alpha"):
            DensitySmoother(alpha=1.5)

    def test_alpha_one_is_passthrough(self) -> None:
        s = DensitySmoother(alpha=1.0)
        assert s.update("cam1", 50.0) == 50.0
        assert s.update("cam1", 100.0) == 100.0


# ======================================================================
# EMA behavior
# ======================================================================


class TestEMABehavior:
    def test_first_observation_is_raw(self) -> None:
        s = DensitySmoother(alpha=0.4)
        result = s.update("cam1", 30.0)
        assert result == 30.0

    def test_second_observation_blends(self) -> None:
        s = DensitySmoother(alpha=0.4)
        s.update("cam1", 30.0)  # smoothed = 30.0
        result = s.update("cam1", 50.0)
        # Expected: 0.4*50 + 0.6*30 = 20 + 18 = 38.0
        assert abs(result - 38.0) < 0.01

    def test_transient_spike_dampened(self) -> None:
        """A bus blocks camera for 1 frame → shouldn't trigger k_critical."""
        s = DensitySmoother(alpha=0.4)
        s.update("cam1", 10.0)   # Normal density
        s.update("cam1", 10.0)   # Normal density
        # Bus blocks camera → YOLO sees 0 vehicles + bus = high density
        result = s.update("cam1", 80.0)
        # Should NOT jump to 80.  Expected: 0.4*80 + 0.6*10 = 38.0
        assert result < 45.0  # Still below k_critical (45)

    def test_genuine_congestion_builds(self) -> None:
        """Sustained high density over 4 ticks should cross k_critical."""
        s = DensitySmoother(alpha=0.4)
        s.update("cam1", 10.0)      # Tick 1: normal (smoothed=10)
        s.update("cam1", 60.0)      # Tick 2: congestion starts (smoothed=30)
        s.update("cam1", 60.0)      # Tick 3: congestion persists (smoothed=42)
        r4 = s.update("cam1", 60.0)  # Tick 4: still congested (smoothed=49.2)
        # After 4 ticks of high density, smoothed should be > 45
        assert r4 > 45.0

    def test_convergence(self) -> None:
        """After many identical observations, smoothed == raw."""
        s = DensitySmoother(alpha=0.4)
        for _ in range(50):
            result = s.update("cam1", 42.0)
        assert abs(result - 42.0) < 0.01


# ======================================================================
# Multi-camera independence
# ======================================================================


class TestMultiCamera:
    def test_cameras_are_independent(self) -> None:
        s = DensitySmoother(alpha=0.4)
        s.update("cam1", 10.0)
        s.update("cam2", 50.0)
        assert s.get("cam1") == 10.0
        assert s.get("cam2") == 50.0

    def test_update_one_camera_doesnt_affect_other(self) -> None:
        s = DensitySmoother(alpha=0.4)
        s.update("cam1", 10.0)
        s.update("cam2", 50.0)
        s.update("cam1", 20.0)  # Only cam1 changes
        assert s.get("cam2") == 50.0


# ======================================================================
# Reset & query
# ======================================================================


class TestResetAndQuery:
    def test_get_unknown_camera_returns_none(self) -> None:
        s = DensitySmoother()
        assert s.get("unknown") is None

    def test_reset_single_camera(self) -> None:
        s = DensitySmoother()
        s.update("cam1", 10.0)
        s.update("cam2", 20.0)
        s.reset("cam1")
        assert s.get("cam1") is None
        assert s.get("cam2") == 20.0

    def test_reset_all(self) -> None:
        s = DensitySmoother()
        s.update("cam1", 10.0)
        s.update("cam2", 20.0)
        s.reset()
        assert s.get("cam1") is None
        assert s.get("cam2") is None

    def test_camera_ids(self) -> None:
        s = DensitySmoother()
        s.update("cam_a", 10.0)
        s.update("cam_b", 20.0)
        assert sorted(s.camera_ids) == ["cam_a", "cam_b"]
