# 2026-06-10 - AI Queue Triage Synchronization

## Context

The 2026-06-09 traffic-safety proposals and the 2026-06-10 technical audit added substantial new planning work, but the canonical `.ai/tasks.json` queue still pointed at the older `TRAFIK-006` task from 2026-06-06.

## What I Learned

Planning documents do not become executable state until they are synchronized into the `main` checkout's `.ai/` files. When new audit or proposal docs overlap existing issue docs, duplicate task creation makes the queue noisy; it is better to promote the existing task in queue order and record the newer source on that task.

## Reuse Rules

- Before continuing work, compare `docs/audit/`, `docs/proposals/`, and `docs/issues/` against `.ai/state.json` and `.ai/tasks.json`.
- Treat `.ai/state.json.next` as the executable pointer, not the newest markdown document.
- When a new proposal maps to an existing issue, update the existing `.ai` task ordering, dependencies, priority, or source metadata instead of creating a duplicate.
- Put safety-net and critical correctness tasks ahead of model or product enhancements when an audit identifies broken headline behavior.

## Failure Signals

- `.ai/state.json.next` points to an older task even though newer audit findings are critical or blocking.
- Proposal docs describe priorities such as P1/P2, but no corresponding `.ai` task exists.
- Existing issues and new proposals describe the same work under different names.
- `continue` resumes a lower-priority enhancement while audit P0/P1 correctness work remains unscheduled.

## Next Checklist

- Verify the current branch is `main` before reading or writing `.ai/`.
- Validate `.ai/state.json` and `.ai/tasks.json` as JSON after edits.
- Confirm the first `todo` task matches `.ai/state.json.next`.
- Append a triage event to `.ai/ledger.jsonl` whenever queue order changes.
