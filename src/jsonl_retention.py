"""Retention policy for append-only JSONL runtime data."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Iterable

logger = logging.getLogger("mainloop")

_DATE_DIR_FORMAT = "%Y-%m-%d"
_ROOT_JSONL_TIMESTAMP_FIELDS = {
    "anomaly_log.jsonl": ("timestamp",),
    "evaluation_metrics.jsonl": (
        "created_at",
        "evaluation_time",
        "predicted_impact_time",
    ),
}


@dataclass(frozen=True)
class JsonlRetentionResult:
    """Summary of one JSONL retention pass."""

    deleted_files: int = 0
    removed_empty_dirs: int = 0
    compacted_files: int = 0
    removed_lines: int = 0
    retained_lines: int = 0
    preserved_lines: int = 0
    skipped_files: int = 0

    @property
    def changed(self) -> bool:
        return (
            self.deleted_files > 0
            or self.removed_empty_dirs > 0
            or self.compacted_files > 0
            or self.removed_lines > 0
        )


def prune_jsonl_retention(
    data_dir: str | os.PathLike[str],
    *,
    now: datetime,
    retention_days: int,
    enabled: bool = True,
) -> JsonlRetentionResult:
    """Prune JSONL runtime data older than the configured retention window."""
    if not enabled:
        return JsonlRetentionResult()

    root = Path(data_dir)
    if retention_days < 0:
        logger.warning("JSONL retention skipped: retention_days=%s", retention_days)
        return JsonlRetentionResult(skipped_files=1)
    if not root.exists():
        return JsonlRetentionResult()

    cutoff = now.date() - timedelta(days=retention_days)
    result = _prune_dated_sensor_logs(root, cutoff)
    for filename, timestamp_fields in _ROOT_JSONL_TIMESTAMP_FIELDS.items():
        compacted = _compact_root_jsonl(root / filename, cutoff, timestamp_fields)
        result = _merge_results(result, compacted)

    if result.changed:
        logger.info(
            "JSONL retention: deleted_files=%s removed_lines=%s "
            "preserved_lines=%s compacted_files=%s",
            result.deleted_files,
            result.removed_lines,
            result.preserved_lines,
            result.compacted_files,
        )
    return result


def _prune_dated_sensor_logs(root: Path, cutoff: date) -> JsonlRetentionResult:
    deleted_files = 0
    removed_empty_dirs = 0
    skipped_files = 0

    for child in root.iterdir():
        if not child.is_dir():
            continue
        try:
            day = datetime.strptime(child.name, _DATE_DIR_FORMAT).date()
        except ValueError:
            continue
        if day >= cutoff:
            continue

        sensor_log = child / "sensor_data.jsonl"
        if not sensor_log.exists():
            skipped_files += 1
            continue
        try:
            sensor_log.unlink()
            deleted_files += 1
            child.rmdir()
            removed_empty_dirs += 1
        except OSError:
            if sensor_log.exists():
                skipped_files += 1

    return JsonlRetentionResult(
        deleted_files=deleted_files,
        removed_empty_dirs=removed_empty_dirs,
        skipped_files=skipped_files,
    )


def _compact_root_jsonl(
    path: Path,
    cutoff: date,
    timestamp_fields: Iterable[str],
) -> JsonlRetentionResult:
    if not path.exists():
        return JsonlRetentionResult()

    retained_lines: list[str] = []
    removed_lines = 0
    preserved_lines = 0
    total_retained = 0

    try:
        with path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                if not raw_line.strip():
                    preserved_lines += 1
                    retained_lines.append(raw_line)
                    continue
                record_date = _record_date(raw_line, timestamp_fields)
                if record_date is None:
                    preserved_lines += 1
                    retained_lines.append(raw_line)
                    continue
                if record_date < cutoff:
                    removed_lines += 1
                    continue
                total_retained += 1
                retained_lines.append(raw_line)
    except OSError:
        return JsonlRetentionResult(skipped_files=1)

    if removed_lines == 0:
        return JsonlRetentionResult(
            retained_lines=total_retained,
            preserved_lines=preserved_lines,
        )

    try:
        _atomic_write_lines(path, retained_lines)
    except OSError:
        return JsonlRetentionResult(
            retained_lines=total_retained,
            preserved_lines=preserved_lines,
            skipped_files=1,
        )

    return JsonlRetentionResult(
        compacted_files=1,
        removed_lines=removed_lines,
        retained_lines=total_retained,
        preserved_lines=preserved_lines,
    )


def _record_date(line: str, timestamp_fields: Iterable[str]) -> date | None:
    try:
        record = json.loads(line)
    except json.JSONDecodeError:
        return None
    if not isinstance(record, dict):
        return None

    for field in timestamp_fields:
        value = record.get(field)
        parsed = _parse_datetime(value)
        if parsed is not None:
            return parsed.date()
    return None


def _parse_datetime(value: object) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    normalized = value.strip()
    if normalized.endswith("Z"):
        normalized = f"{normalized[:-1]}+00:00"
    try:
        return datetime.fromisoformat(normalized)
    except ValueError:
        return None


def _atomic_write_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        delete=False,
        prefix=f".{path.name}.",
        suffix=".tmp",
    ) as tmp:
        tmp.writelines(lines)
        tmp_path = Path(tmp.name)
    try:
        tmp_path.replace(path)
    except OSError:
        tmp_path.unlink(missing_ok=True)
        raise


def _merge_results(
    left: JsonlRetentionResult,
    right: JsonlRetentionResult,
) -> JsonlRetentionResult:
    return JsonlRetentionResult(
        deleted_files=left.deleted_files + right.deleted_files,
        removed_empty_dirs=left.removed_empty_dirs + right.removed_empty_dirs,
        compacted_files=left.compacted_files + right.compacted_files,
        removed_lines=left.removed_lines + right.removed_lines,
        retained_lines=left.retained_lines + right.retained_lines,
        preserved_lines=left.preserved_lines + right.preserved_lines,
        skipped_files=left.skipped_files + right.skipped_files,
    )
