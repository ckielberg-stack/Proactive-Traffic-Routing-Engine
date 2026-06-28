"""Regression tests for the default Docker deployment shape."""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _top_level_compose_services(compose_text: str) -> list[str]:
    services: list[str] = []
    in_services = False
    for line in compose_text.splitlines():
        if line == "services:":
            in_services = True
            continue
        if not in_services:
            continue
        if line and not line.startswith(" "):
            break
        if line.startswith("  ") and not line.startswith("    "):
            services.append(line.strip().removesuffix(":"))
    return services


def test_compose_defines_one_canonical_service() -> None:
    compose_text = (ROOT / "docker-compose.yml").read_text()
    services = _top_level_compose_services(compose_text)

    assert services == ["trafik"]
    assert "depends_on" not in compose_text
    assert "collector" not in compose_text
    assert "collect.py" not in compose_text
    assert "dashboard.py" not in compose_text


def test_dockerfile_uses_canonical_entrypoint_and_healthcheck() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text()

    assert "collector.log" not in dockerfile
    assert "collect.py" not in dockerfile
    assert "dashboard.py" not in dockerfile
    assert "127.0.0.1:8080/health" in dockerfile

    entrypoint_line = next(
        line for line in dockerfile.splitlines() if line.startswith("ENTRYPOINT ")
    )
    cmd_line = next(line for line in dockerfile.splitlines() if line.startswith("CMD "))

    assert ast.literal_eval(entrypoint_line.removeprefix("ENTRYPOINT ")) == [
        "python",
        "-u",
        "main.py",
    ]
    assert ast.literal_eval(cmd_line.removeprefix("CMD ")) == [
        "--host",
        "0.0.0.0",
        "--port",
        "8080",
    ]
