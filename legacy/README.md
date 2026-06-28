# Legacy Entry Points

`collect.py` and `dashboard.py` are historical reference implementations from
the pre-unified runtime. They are not part of the default Docker image or
`docker compose up` deployment.

Use `main.py` for the canonical FastAPI dashboard, operator API, and tick loop.
