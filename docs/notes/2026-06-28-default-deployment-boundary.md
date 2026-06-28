# 2026-06-28 - Default Deployment Boundary

## Context

TRAFIK-021 retired the legacy `collect.py` and `dashboard.py` path from the
default Docker deployment while preserving the files under `legacy/` for
historical reference.

## What I Learned

Quarantining legacy entry points is not enough if the Docker image still copies
root files with a broad wildcard or if Compose tests only exercise runtime
behavior. The deployment boundary needs to exclude legacy files structurally,
and a small regression test should assert the intended service shape.

## Reuse Rules

- Prefer explicit Docker `COPY` entries for canonical runtime files when legacy
  or reference entry points remain in the repo.
- Add a deployment-shape test for Compose changes that assert service count,
  entrypoint/command targets, and healthcheck target.
- Healthchecks should probe the canonical serving process, not a side-effect
  file owned by a retired process.

## Failure Signals

- `docker-compose.yml` defines more than one default service for one canonical
  runtime.
- `Dockerfile` copies all root `*.py` files after legacy entry points move into
  a reference directory.
- Healthchecks reference logs or files from a retired service.

## Next Checklist

- Run `docker compose config --quiet` after Compose edits.
- Search deployment files for retired entrypoint names.
- Keep `legacy/` out of the Docker image unless explicitly building a reference
  image.
