# Contributing

PTRE is published as an open-source prototype. Contributions are welcome when they make the system easier to understand, validate, replay, port, or operate safely.

## Good Contributions

- Small, testable changes with clear motivation.
- Replay fixtures or synthetic data that do not expose credentials, private data, captured camera frames, or operational logs.
- Improvements to camera calibration, confidence scoring, test coverage, DATEX II export, dashboard usability, or documentation.
- Notes about assumptions, limitations, and failure modes.

## Local Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Set `TRAFIKVERKET_API_KEY` in `.env` before running live API paths.

## Before Opening a Pull Request

Run:

```bash
pytest tests/ -v --ignore=tests/smoke_test.py
```

Keep generated runtime artifacts out of Git:

- `.env`
- `data/`
- `storage/`
- `*.pt`
- `.DS_Store`

If your change depends on live Trafikverket data, document that clearly and provide a non-live fallback or test where practical.
