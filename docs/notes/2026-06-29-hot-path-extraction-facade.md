# 2026-06-29 - Hot-Path Extraction Facade

## Context

TRAFIK-022 extracted tick orchestration, Trafikverket fetchers, camera work,
fusion helpers, and persistence out of `main_loop.py` into focused `src/`
modules. Existing integration tests and ad hoc callers still patched
`main_loop` globals directly.

## What I Learned

Large hot-path extractions are safer when the old entry point becomes an
explicit transitional facade instead of disappearing immediately. Direct-module
tests should cover the new owners, while one integration path should keep
exercising the facade so legacy monkeypatch and script behavior cannot drift
silently.

## Reuse Rules

- Move behavior to the new owner first, then keep `main_loop` re-exports thin.
- If old tests patch facade globals, synchronize those patched values into the
  owning modules before delegating.
- Guard nested facade calls during `tick_once` so compatibility sync does not
  reset in-flight orchestrator state.
- Migrate pure helper tests to new modules; leave only compatibility-focused
  tests on the facade.

## Failure Signals

- `TickResult.tick_number` or cached chainage resets during one tick.
- Offline integration tests accidentally hit the live Trafikverket API after an
  extraction.
- Monkeypatched camera workers, data dirs, or source fetchers stop affecting the
  tick path.
- `main_loop.py` starts gaining new behavior instead of delegating to `src`.

## Next Checklist

- Check `rg "import main_loop|from main_loop"` before and after extraction.
- Keep compatibility shims named and documented as transitional.
- Run targeted facade integration tests and direct owner tests separately.
- Remove facade shims only in a later deliberate cleanup after callers migrate.
