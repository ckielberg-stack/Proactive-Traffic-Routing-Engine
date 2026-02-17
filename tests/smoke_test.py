#!/usr/bin/env python3
"""
Smoke test – runs the vision engine against real collected camera images.

Usage:
    python tests/smoke_test.py                          # today's images
    python tests/smoke_test.py --date 2026-02-15       # specific date
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# Ensure project root is on sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.models import CameraMetadata, SensorReading  # noqa: E402
from src.vision_engine import VisionEngine  # noqa: E402

DATA_DIR = ROOT / "data"
METADATA_PATH = DATA_DIR / "metadata.json"


def load_camera_metadata() -> dict[str, CameraMetadata]:
    """Load camera metadata from data/metadata.json."""
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {m["camera_id"]: CameraMetadata(**m) for m in raw}


def run(date_str: str) -> None:
    day_dir = DATA_DIR / date_str / "images"
    if not day_dir.exists():
        print(f"❌  No images found at {day_dir}")
        sys.exit(1)

    metadata = load_camera_metadata()
    engine = VisionEngine()

    images = sorted(day_dir.glob("*.jpg"))
    print(f"\n🔍  Analysing {len(images)} images from {date_str}\n")
    print(f"{'Image':<55} {'Vehicles':>8} {'Capacity':>10} {'Anomaly':>8} {'Conf':>6}")
    print("─" * 95)

    total_vehicles = 0
    anomaly_count = 0

    for img in images:
        # Try to match camera metadata from filename
        meta = _match_metadata(img.name, metadata)

        state = engine.analyze_frame(
            image_path=str(img),
            camera_meta=meta,
            sensor=None,  # no live sensor pairing for smoke test
        )

        flag = "⚠️" if state.is_anomaly else "  "
        print(
            f"{img.name:<55} {state.vehicle_count:>8} "
            f"{state.estimated_capacity_vph:>10.1f} "
            f"{flag:>8} {state.confidence:>6.3f}"
        )
        total_vehicles += state.vehicle_count
        anomaly_count += 1 if state.is_anomaly else 0

    print("─" * 95)
    print(
        f"Total: {len(images)} frames, {total_vehicles} vehicles detected, "
        f"{anomaly_count} anomalies\n"
    )


def _match_metadata(
    filename: str, metadata: dict[str, CameraMetadata]
) -> CameraMetadata:
    """Best-effort match of a filename to a CameraMetadata entry.

    Filename pattern: cam_<name>_<timestamp>.jpg
    """
    lower = filename.lower()
    for cam_id, meta in metadata.items():
        if meta.name.lower().replace(" ", "_") in lower:
            return meta

    # Fallback: generic 2-lane camera
    return CameraMetadata(
        camera_id="unknown",
        name="Unknown",
        lat=59.25,
        lng=17.85,
        num_lanes=2,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vision engine smoke test")
    parser.add_argument(
        "--date",
        default=datetime.now().strftime("%Y-%m-%d"),
        help="Date folder to process (default: today)",
    )
    args = parser.parse_args()
    run(args.date)
