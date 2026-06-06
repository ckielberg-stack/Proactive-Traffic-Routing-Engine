"""ROI physical-length estimation and validation utilities.

The preferred calibration path is BEV homography from ``roi_helper.py``.
When a camera has only pixel-space ROI polygons, this module provides a
repeatable lane-width geometry estimate so density math no longer depends on
the silent 100 m default.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_ROI_LENGTH_METERS = 100.0
LANE_WIDTH_METERS = 3.5
MIN_REASONABLE_LENGTH_METERS = 5.0
MAX_REASONABLE_LENGTH_METERS = 250.0
ESTIMATE_SOURCE = "lane_width_geometry_estimate"


@dataclass(frozen=True)
class ROIValidationIssue:
    camera_id: str
    roi_index: int
    road_id: str
    severity: str
    message: str


def estimate_roi_length_meters(
    polygon: list[list[int | float]],
    num_lanes: int,
    lane_width_meters: float = LANE_WIDTH_METERS,
) -> float:
    """Estimate visible road length from ROI geometry and lane width.

    The polygon's major axis is treated as road direction and its minor axis
    as visible paved width.  ``num_lanes * lane_width_meters`` provides the
    physical scale for converting major-axis pixels into meters.
    """
    if len(polygon) < 3:
        raise ValueError("ROI polygon needs at least 3 vertices")
    if num_lanes < 1:
        raise ValueError("num_lanes must be >= 1")

    points = np.array(polygon, dtype=np.float64)
    centered = points - points.mean(axis=0)
    covariance = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    axes = eigenvectors[:, order]
    projected = centered @ axes

    major_axis_px = float(projected[:, 0].max() - projected[:, 0].min())
    minor_axis_px = float(projected[:, 1].max() - projected[:, 1].min())
    if minor_axis_px <= 0:
        raise ValueError("ROI polygon has zero minor-axis width")

    road_width_meters = num_lanes * lane_width_meters
    length_meters = major_axis_px * (road_width_meters / minor_axis_px)
    return round(max(length_meters, 1.0), 1)


def load_config(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_config(path: str | Path, config: dict[str, Any]) -> None:
    Path(path).write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def apply_estimated_lengths(
    config: dict[str, Any],
    *,
    force: bool = False,
) -> tuple[int, list[ROIValidationIssue]]:
    """Add estimated ``roi_length_meters`` values to ROI definitions.

    Existing non-default lengths are preserved unless ``force`` is true.
    Returns ``(updated_count, issues)``.
    """
    updated = 0
    issues: list[ROIValidationIssue] = []
    cameras = config.get("cameras", {})

    for camera_id, camera_data in cameras.items():
        for roi_index, roi in enumerate(camera_data.get("rois", [])):
            road_id = str(roi.get("road_id", "<unknown>"))
            current = roi.get("roi_length_meters")
            should_update = force or current is None or current == DEFAULT_ROI_LENGTH_METERS
            if not should_update:
                continue

            try:
                length_meters = estimate_roi_length_meters(
                    polygon=roi["polygon"],
                    num_lanes=int(roi.get("num_lanes", 2)),
                )
            except (KeyError, TypeError, ValueError) as exc:
                issues.append(
                    ROIValidationIssue(
                        camera_id=camera_id,
                        roi_index=roi_index,
                        road_id=road_id,
                        severity="error",
                        message=f"could not estimate roi_length_meters: {exc}",
                    )
                )
                continue

            roi["roi_length_meters"] = length_meters
            roi["roi_length_source"] = ESTIMATE_SOURCE
            roi["roi_length_confidence"] = "estimated"
            updated += 1

    if updated:
        config["roi_length_calibration"] = {
            "source": ESTIMATE_SOURCE,
            "lane_width_meters": LANE_WIDTH_METERS,
            "method": "PCA major/minor axis scaled by num_lanes * lane_width_meters",
            "updated_rois": updated,
        }

    return updated, issues


def validate_roi_lengths(config: dict[str, Any]) -> list[ROIValidationIssue]:
    """Return missing/default/suspicious ROI length issues."""
    issues: list[ROIValidationIssue] = []
    cameras = config.get("cameras", {})

    for camera_id, camera_data in cameras.items():
        for roi_index, roi in enumerate(camera_data.get("rois", [])):
            road_id = str(roi.get("road_id", "<unknown>"))
            length = roi.get("roi_length_meters")
            if length is None:
                issues.append(
                    ROIValidationIssue(
                        camera_id=camera_id,
                        roi_index=roi_index,
                        road_id=road_id,
                        severity="error",
                        message="missing roi_length_meters",
                    )
                )
                continue

            try:
                length_float = float(length)
            except (TypeError, ValueError):
                issues.append(
                    ROIValidationIssue(
                        camera_id=camera_id,
                        roi_index=roi_index,
                        road_id=road_id,
                        severity="error",
                        message=f"roi_length_meters is not numeric: {length!r}",
                    )
                )
                continue

            if length_float == DEFAULT_ROI_LENGTH_METERS:
                issues.append(
                    ROIValidationIssue(
                        camera_id=camera_id,
                        roi_index=roi_index,
                        road_id=road_id,
                        severity="error",
                        message="roi_length_meters is still the 100 m fallback",
                    )
                )
            elif length_float < MIN_REASONABLE_LENGTH_METERS:
                issues.append(
                    ROIValidationIssue(
                        camera_id=camera_id,
                        roi_index=roi_index,
                        road_id=road_id,
                        severity="warning",
                        message=(
                            f"roi_length_meters={length_float:.1f} m is unusually short"
                        ),
                    )
                )
            elif length_float > MAX_REASONABLE_LENGTH_METERS:
                issues.append(
                    ROIValidationIssue(
                        camera_id=camera_id,
                        roi_index=roi_index,
                        road_id=road_id,
                        severity="warning",
                        message=(
                            f"roi_length_meters={length_float:.1f} m is unusually long"
                        ),
                    )
                )

    return issues


def _print_issues(issues: list[ROIValidationIssue]) -> None:
    for issue in issues:
        print(
            f"{issue.severity.upper()}: {issue.camera_id} "
            f"roi[{issue.roi_index}] {issue.road_id}: {issue.message}"
        )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Estimate or validate ROI physical road lengths.",
    )
    parser.add_argument(
        "command",
        choices=("calibrate-estimates", "validate"),
    )
    parser.add_argument(
        "--config",
        default="camera_config.json",
        help="Path to camera_config.json",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Write estimated lengths back to the config file",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute lengths even when a non-default length exists",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    config_path = Path(args.config)
    config = load_config(config_path)

    if args.command == "calibrate-estimates":
        updated, estimation_issues = apply_estimated_lengths(
            config,
            force=args.force,
        )
        validation_issues = validate_roi_lengths(config)
        issues = estimation_issues + validation_issues
        print(f"ROI length estimate update: {updated} ROI(s)")
        _print_issues(issues)
        if args.write:
            save_config(config_path, config)
            print(f"Wrote {config_path}")
        elif updated:
            print("Dry run only. Re-run with --write to update the config.")
        return 1 if any(issue.severity == "error" for issue in issues) else 0

    issues = validate_roi_lengths(config)
    _print_issues(issues)
    errors = [issue for issue in issues if issue.severity == "error"]
    warnings = [issue for issue in issues if issue.severity == "warning"]
    if errors:
        print(f"ROI length validation failed: {len(errors)} error(s)")
        return 1
    print(f"ROI length validation passed: {len(warnings)} warning(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
