import json
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

from src.replay_evaluator import (
    METRICS_VERSION,
    evaluate_replay_paths,
    evaluate_replay_records,
)


ROUTE_IDS = ["R1", "R2"]


def _record(
    timestamp: datetime,
    record_type: str,
    **fields,
) -> dict:
    return {
        "type": record_type,
        "timestamp": timestamp.isoformat(),
        **fields,
    }


def _travel_time(
    timestamp: datetime,
    route_id: str,
    status: str,
) -> dict:
    return _record(
        timestamp,
        "travel_time",
        route_id=route_id,
        name=route_id,
        travel_time_seconds=120.0,
        free_flow_seconds=60.0,
        delay_seconds=60.0 if status in {"slow", "heavy"} else 0.0,
        speed_kmh=30.0 if status in {"slow", "heavy"} else 80.0,
        traffic_status=status,
        length_meters=5000.0,
    )


def _prediction(
    timestamp: datetime,
    prediction_id: str,
    origin_chainage_km: float = 10.0,
    growth_speed_kmh: float = 60.0,
    lengths_at_minutes: dict[int, float] | None = None,
) -> dict:
    return _record(
        timestamp,
        "queue_prediction",
        prediction_id=prediction_id,
        camera_id="CAM_X",
        origin_chainage_km=origin_chainage_km,
        growth_speed_kmh=growth_speed_kmh,
        lengths_at_minutes=lengths_at_minutes or {1: 1.0, 5: 5.0, 10: 10.0},
    )


def test_successful_hit_reports_eta_and_distance_accuracy() -> None:
    start = datetime(2026, 6, 6, 12, 0, 0)
    records = [
        _travel_time(start, "R1", "freeflow"),
        _travel_time(start, "R2", "freeflow"),
        _prediction(start, "p1"),
        _travel_time(start + timedelta(minutes=2, seconds=30), "R2", "slow"),
        _record(
            start + timedelta(minutes=3),
            "vms_status",
            vms_id="VMS-1",
            is_active=True,
        ),
    ]

    metrics = evaluate_replay_records(records, route_ids=ROUTE_IDS, corridor_length_km=10.0)

    assert metrics["version"] == METRICS_VERSION
    assert metrics["corridor"]["prediction_count"] == 1
    assert metrics["corridor"]["matched_congestion_count"] == 1
    assert metrics["corridor"]["precision"] == 1.0
    assert metrics["corridor"]["recall"] == 1.0
    assert metrics["corridor"]["mean_eta_error_minutes"] == 0.0
    assert metrics["corridor"]["mean_distance_error_km"] == 0.0
    assert metrics["corridor"]["mean_vms_lead_time_minutes"] == 2.5
    assert metrics["segments"]["R2"]["hit_count"] == 1


def test_late_hit_has_negative_lead_time() -> None:
    start = datetime(2026, 6, 6, 12, 0, 0)
    records = [
        _travel_time(start, "R1", "freeflow"),
        _travel_time(start, "R2", "freeflow"),
        _travel_time(start + timedelta(minutes=5), "R2", "heavy"),
        _prediction(
            start + timedelta(minutes=7),
            "late",
            origin_chainage_km=7.5,
            lengths_at_minutes={1: 1.0},
        ),
    ]

    metrics = evaluate_replay_records(records, route_ids=ROUTE_IDS, corridor_length_km=10.0)

    assert metrics["corridor"]["matched_congestion_count"] == 1
    assert metrics["matches"][0]["status"] == "late_hit"
    assert metrics["matches"][0]["lead_time_minutes"] == -2.0
    assert metrics["corridor"]["mean_lead_time_minutes"] == -2.0


def test_early_false_positive_is_counted_and_expired() -> None:
    start = datetime(2026, 6, 6, 12, 0, 0)
    records = [
        _travel_time(start, "R1", "freeflow"),
        _travel_time(start, "R2", "freeflow"),
        _prediction(start, "false-positive"),
        _travel_time(start + timedelta(minutes=40), "R1", "freeflow"),
    ]

    metrics = evaluate_replay_records(records, route_ids=ROUTE_IDS, corridor_length_km=10.0)

    assert metrics["corridor"]["false_positive_count"] == 1
    assert metrics["corridor"]["expired_prediction_count"] == 1
    assert metrics["false_positives"][0]["status"] == "expired_prediction"
    assert metrics["corridor"]["precision"] == 0.0
    assert metrics["corridor"]["recall"] is None


def test_missed_congestion_is_counted_per_segment() -> None:
    start = datetime(2026, 6, 6, 12, 0, 0)
    records = [
        _travel_time(start, "R1", "freeflow"),
        _travel_time(start, "R2", "freeflow"),
        _travel_time(start + timedelta(minutes=5), "R1", "slow"),
    ]

    metrics = evaluate_replay_records(records, route_ids=ROUTE_IDS, corridor_length_km=10.0)

    assert metrics["corridor"]["missed_congestion_count"] == 1
    assert metrics["corridor"]["recall"] == 0.0
    assert metrics["segments"]["R1"]["missed_count"] == 1
    assert metrics["misses"][0]["status"] == "missed_congestion"


def test_expired_prediction_does_not_match_late_congestion() -> None:
    start = datetime(2026, 6, 6, 12, 0, 0)
    records = [
        _travel_time(start, "R1", "freeflow"),
        _travel_time(start, "R2", "freeflow"),
        _prediction(start, "expired"),
        _travel_time(start + timedelta(minutes=40), "R1", "slow"),
    ]

    metrics = evaluate_replay_records(records, route_ids=ROUTE_IDS, corridor_length_km=10.0)

    assert metrics["corridor"]["matched_congestion_count"] == 0
    assert metrics["corridor"]["missed_congestion_count"] == 1
    assert metrics["corridor"]["false_positive_count"] == 1
    assert metrics["corridor"]["expired_prediction_count"] == 1


def test_replay_command_writes_metrics_artifact(tmp_path: Path) -> None:
    fixture = Path("tests/fixtures/replay_sample.jsonl")
    output = tmp_path / "metrics.json"

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.replay_evaluator",
            str(fixture),
            "--output",
            str(output),
            "--corridor-length-km",
            "10.0",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    stdout_metrics = json.loads(completed.stdout)
    artifact_metrics = json.loads(output.read_text())
    assert stdout_metrics["version"] == METRICS_VERSION
    assert artifact_metrics == stdout_metrics
    assert artifact_metrics["corridor"]["prediction_count"] == 2


def test_evaluate_replay_paths_accepts_direct_jsonl_file(tmp_path: Path) -> None:
    path = tmp_path / "sensor_data.jsonl"
    start = datetime(2026, 6, 6, 12, 0, 0)
    lines = [
        json.dumps(_travel_time(start, "R1", "freeflow")),
        json.dumps(_travel_time(start, "R2", "freeflow")),
        json.dumps(_prediction(start, "p1")),
        json.dumps(_travel_time(start + timedelta(minutes=5), "R1", "slow")),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    metrics = evaluate_replay_paths(
        [path],
        route_ids=ROUTE_IDS,
        corridor_length_km=10.0,
    )

    assert metrics["source_files"] == [str(path)]
    assert metrics["corridor"]["matched_congestion_count"] == 1
