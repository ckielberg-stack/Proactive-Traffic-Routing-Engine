# Security Policy

PTRE is a prototype and should not be used as safety-critical traffic-control software without independent validation.

## Reporting Security Issues

Please do not open a public issue for suspected credential leaks, unsafe defaults, data exposure, or vulnerabilities that could affect real systems.

Report privately to the repository owner through GitHub security advisories or another private contact channel listed on the repository profile.

## Sensitive Data Rules

Do not commit:

- Trafikverket API keys or any other credentials.
- Local `.env` files.
- Captured traffic-camera images, anomaly frames, or training frames.
- Runtime logs, JSONL data, cached API responses, or local operator state.
- Model weight files such as `*.pt`.

If a secret is accidentally committed, rotate it immediately and rewrite repository history before publishing or sharing the repository.

## Supported Versions

This repository currently has no stable release line. Security fixes should target `main`.
