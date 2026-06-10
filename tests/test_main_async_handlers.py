"""Regression tests for async dashboard handlers in main.py."""

import asyncio

from fastapi.testclient import TestClient

import main


class _FakeResponse:
    content = b"jpeg-bytes"

    def raise_for_status(self) -> None:
        return None


def _assert_not_in_event_loop() -> None:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return
    raise AssertionError("blocking request helper ran on the event-loop thread")


def test_camera_image_fetches_blocking_work_off_event_loop(monkeypatch) -> None:
    monkeypatch.delenv("PTRE_API_TOKEN", raising=False)

    def fake_get_camera_info() -> dict[str, dict]:
        _assert_not_in_event_loop()
        return {
            "CAM_TEST": {
                "name": "Test camera",
                "description": "Offline test camera",
                "photo_url": "https://example.invalid/camera.jpg",
            }
        }

    def fake_get(url: str, timeout: int) -> _FakeResponse:
        _assert_not_in_event_loop()
        assert url == "https://example.invalid/camera.jpg?type=fullsize"
        assert timeout == 15
        return _FakeResponse()

    monkeypatch.setattr(main, "_get_camera_info", fake_get_camera_info)
    monkeypatch.setattr(main.requests, "get", fake_get)

    client = TestClient(main.operator_app)
    resp = client.get("/api/v1/camera-image/CAM_TEST")

    assert resp.status_code == 200
    assert resp.content == b"jpeg-bytes"
    assert resp.headers["content-type"] == "image/jpeg"
