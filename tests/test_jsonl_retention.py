from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

from src.jsonl_retention import prune_jsonl_retention


NOW = datetime(2026, 6, 29, 12, 0, 0)


def _write_jsonl(path: Path, records: list[dict | str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        record if isinstance(record, str) else json.dumps(record)
        for record in records
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_prunes_old_dated_sensor_logs_and_keeps_cutoff_and_newer(
    tmp_path: Path,
) -> None:
    old_dir = tmp_path / "2026-05-29"
    cutoff_dir = tmp_path / "2026-05-30"
    newer_dir = tmp_path / "2026-06-29"
    for day_dir in (old_dir, cutoff_dir, newer_dir):
        _write_jsonl(day_dir / "sensor_data.jsonl", [{"timestamp": NOW.isoformat()}])

    result = prune_jsonl_retention(tmp_path, now=NOW, retention_days=30)

    assert not old_dir.exists()
    assert (cutoff_dir / "sensor_data.jsonl").exists()
    assert (newer_dir / "sensor_data.jsonl").exists()
    assert result.deleted_files == 1
    assert result.removed_empty_dirs == 1


def test_prunes_sensor_log_but_keeps_non_empty_date_directory(
    tmp_path: Path,
) -> None:
    old_dir = tmp_path / "2026-05-01"
    _write_jsonl(old_dir / "sensor_data.jsonl", [{"timestamp": "2026-05-01T00:00:00"}])
    (old_dir / "keep.txt").write_text("operator note", encoding="utf-8")

    result = prune_jsonl_retention(tmp_path, now=NOW, retention_days=30)

    assert old_dir.exists()
    assert not (old_dir / "sensor_data.jsonl").exists()
    assert (old_dir / "keep.txt").exists()
    assert result.deleted_files == 1
    assert result.removed_empty_dirs == 0


def test_compacts_anomaly_log_by_timestamp(tmp_path: Path) -> None:
    path = tmp_path / "anomaly_log.jsonl"
    _write_jsonl(
        path,
        [
            {"timestamp": "2026-05-01T00:00:00", "camera_id": "OLD"},
            {"timestamp": "2026-06-01T00:00:00", "camera_id": "NEW"},
        ],
    )

    result = prune_jsonl_retention(tmp_path, now=NOW, retention_days=30)

    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [record["camera_id"] for record in records] == ["NEW"]
    assert result.compacted_files == 1
    assert result.removed_lines == 1
    assert result.retained_lines == 1


def test_compacts_evaluation_metrics_by_timestamp_fallback_order(
    tmp_path: Path,
) -> None:
    path = tmp_path / "evaluation_metrics.jsonl"
    _write_jsonl(
        path,
        [
            {"created_at": "2026-05-01T00:00:00", "prophecy_id": "old-created"},
            {"evaluation_time": "2026-05-01T00:00:00", "prophecy_id": "old-eval"},
            {
                "predicted_impact_time": "2026-05-01T00:00:00",
                "prophecy_id": "old-impact",
            },
            {"created_at": "2026-06-01T00:00:00", "prophecy_id": "new-created"},
            {"evaluation_time": "2026-06-01T00:00:00", "prophecy_id": "new-eval"},
            {
                "predicted_impact_time": "2026-06-01T00:00:00",
                "prophecy_id": "new-impact",
            },
        ],
    )

    result = prune_jsonl_retention(tmp_path, now=NOW, retention_days=30)

    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [record["prophecy_id"] for record in records] == [
        "new-created",
        "new-eval",
        "new-impact",
    ]
    assert result.compacted_files == 1
    assert result.removed_lines == 3
    assert result.retained_lines == 3


def test_preserves_malformed_and_undated_root_jsonl_lines(tmp_path: Path) -> None:
    path = tmp_path / "anomaly_log.jsonl"
    malformed = "{not-json"
    undated = {"camera_id": "NO_TIME"}
    _write_jsonl(
        path,
        [
            {"timestamp": "2026-05-01T00:00:00", "camera_id": "OLD"},
            malformed,
            undated,
            {"timestamp": "2026-06-01T00:00:00", "camera_id": "NEW"},
        ],
    )

    result = prune_jsonl_retention(tmp_path, now=NOW, retention_days=30)

    lines = path.read_text(encoding="utf-8").splitlines()
    assert malformed in lines
    assert any('"NO_TIME"' in line for line in lines)
    assert any('"NEW"' in line for line in lines)
    assert not any('"OLD"' in line for line in lines)
    assert result.removed_lines == 1
    assert result.preserved_lines == 2
    assert result.retained_lines == 1


def test_disabled_retention_performs_no_mutations(tmp_path: Path) -> None:
    old_dir = tmp_path / "2026-05-01"
    _write_jsonl(old_dir / "sensor_data.jsonl", [{"timestamp": "2026-05-01T00:00:00"}])
    root_log = tmp_path / "anomaly_log.jsonl"
    _write_jsonl(root_log, [{"timestamp": "2026-05-01T00:00:00"}])
    before = root_log.read_text(encoding="utf-8")

    result = prune_jsonl_retention(
        tmp_path,
        now=NOW,
        retention_days=30,
        enabled=False,
    )

    assert (old_dir / "sensor_data.jsonl").exists()
    assert root_log.read_text(encoding="utf-8") == before
    assert result.deleted_files == 0
    assert result.removed_lines == 0


def test_timezone_z_timestamps_are_parsed(tmp_path: Path) -> None:
    path = tmp_path / "anomaly_log.jsonl"
    old = NOW - timedelta(days=90)
    _write_jsonl(
        path,
        [
            {"timestamp": old.strftime("%Y-%m-%dT%H:%M:%SZ"), "camera_id": "OLD"},
            {"timestamp": "2026-06-01T00:00:00Z", "camera_id": "NEW"},
        ],
    )

    result = prune_jsonl_retention(tmp_path, now=NOW, retention_days=30)

    assert '"OLD"' not in path.read_text(encoding="utf-8")
    assert result.removed_lines == 1
