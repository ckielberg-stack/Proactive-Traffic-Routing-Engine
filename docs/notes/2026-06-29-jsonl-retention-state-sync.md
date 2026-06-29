# 2026-06-29 - JSONL Retention State Sync

## Context

TRAFIK-023 added a configurable retention pass for append-only JSONL runtime
files. The same data feeds replay, dashboard anomaly counts, and operator
diagnostics.

## What I Learned

Pruning append-only logs is not just a filesystem cleanup. Any in-memory summary
derived from the pruned log must be invalidated or recomputed in the same flow,
otherwise the dashboard can report pre-prune totals after old records have been
removed.

## Reuse Rules

- Prune JSONL after the current tick has written its records, then write status
  snapshots from the post-prune state.
- Preserve malformed or undated JSONL lines during compaction so manual recovery
  stays possible.
- Return structured retention counts from pruning helpers so tests can assert
  deletions and compactions without scraping logs.
- Refresh cached counters derived from compacted logs before exposing dashboard
  status.

## Failure Signals

- `status.json` shows an anomaly total larger than the compacted
  `anomaly_log.jsonl`.
- A retention pass deletes old replay files but leaves root append-only JSONL
  unbounded.
- Tests only assert that pruning ran, not which files or lines changed.

## Next Checklist

- Seed old dated `sensor_data.jsonl` files and root JSONL logs in `tmp_path`.
- Assert cutoff-day records are retained and older records are removed.
- Include malformed and undated lines in compaction tests.
- Verify dashboard/status counters after pruning, not only filesystem changes.
