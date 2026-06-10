# 2026-06-10 - Camera Worker Concurrency State

## Context

`fetch_cameras` was parallelized so each tick can overlap camera image fetch,
decode, and inference. The first implementation shared one `VisionEngine` and
one `RetentionPolicy` across worker futures.

## What I Learned

Camera parallelism must separate stateless work from stateful library and
persistence objects. A shared Ultralytics predictor can serialize inference
behind its internal lock, so a shared `VisionEngine` may only parallelize
fetch/decode. `RetentionPolicy` also mutates the training schedule and writes
`training_schedule.json`, so concurrent calls can lose updates without a lock.

## Reuse Rules

- Use thread-local or otherwise isolated inference engines for concurrent
  camera workers.
- Serialize calls into mutable retention/persistence policy objects.
- Keep output ordering deterministic by collecting futures into original
  camera metadata order before returning records or capacity states.
- Test both soft per-camera failures (`None` fetch/decode) and hard exceptions
  in the same batch.

## Failure Signals

- Camera batch duration improves but YOLO inference still appears serial.
- Training sample schedule entries disappear or `training_schedule.json` is
  inconsistently updated.
- Mocked camera tests become order-dependent after parallelization.
- One camera exception aborts a whole tick or drops later camera records.

## Next Checklist

- Confirm worker count is bounded before adding more cameras.
- Confirm the worker path does not share a stateful predictor instance.
- Guard any mutable schedule, JSONL, or file persistence object used by workers.
- Assert deterministic record and `CapacityState` ordering in tests.
