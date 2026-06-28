"""Regression tests for async dashboard handlers in main.py."""

import asyncio
import json

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


def _use_excluded_file(monkeypatch, tmp_path):
    excluded_file = tmp_path / "excluded_cameras.json"
    monkeypatch.setattr(main, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(main, "EXCLUDED_CAMERAS_FILE", str(excluded_file))
    return excluded_file


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


def test_excluded_cameras_missing_file_returns_empty_list(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.delenv("PTRE_API_TOKEN", raising=False)
    _use_excluded_file(monkeypatch, tmp_path)

    client = TestClient(main.operator_app)
    resp = client.get("/api/cameras/excluded")

    assert resp.status_code == 200
    assert resp.json() == {"excluded": []}


def test_exclude_camera_creates_file_and_is_idempotent(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("PTRE_API_TOKEN", raising=False)
    excluded_file = _use_excluded_file(monkeypatch, tmp_path)
    camera_id = main.CAMERA_IDS[0]

    client = TestClient(main.operator_app)
    first = client.delete(f"/api/cameras/{camera_id}")
    second = client.delete(f"/api/cameras/{camera_id}")

    assert first.status_code == 200
    assert first.json() == {"ok": True, "excluded_count": 1}
    assert second.status_code == 200
    assert second.json() == {"ok": True, "excluded_count": 1}
    assert json.loads(excluded_file.read_text()) == [camera_id]


def test_excluded_cameras_returns_legacy_payload(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("PTRE_API_TOKEN", raising=False)
    excluded_file = _use_excluded_file(monkeypatch, tmp_path)
    camera_id = main.CAMERA_IDS[0]
    excluded_file.write_text(json.dumps([camera_id]))

    client = TestClient(main.operator_app)
    resp = client.get("/api/cameras/excluded")

    coords = main.CAMERA_COORDS.get(camera_id)
    assert resp.status_code == 200
    assert resp.json() == {
        "excluded": [
            {
                "id": camera_id,
                "name": camera_id.split("_")[-1],
                "lat": coords[0] if coords else None,
                "lng": coords[1] if coords else None,
            }
        ]
    }


def test_restore_camera_removes_exclusion(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("PTRE_API_TOKEN", raising=False)
    excluded_file = _use_excluded_file(monkeypatch, tmp_path)
    camera_id = main.CAMERA_IDS[0]
    other_camera_id = main.CAMERA_IDS[1]
    excluded_file.write_text(json.dumps([camera_id, other_camera_id]))

    client = TestClient(main.operator_app)
    resp = client.post(f"/api/cameras/{camera_id}/restore")

    assert resp.status_code == 200
    assert resp.json() == {"ok": True, "excluded_count": 1}
    assert json.loads(excluded_file.read_text()) == [other_camera_id]


def test_camera_exclusion_rejects_unknown_camera(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("PTRE_API_TOKEN", raising=False)
    _use_excluded_file(monkeypatch, tmp_path)

    client = TestClient(main.operator_app)
    delete_resp = client.delete("/api/cameras/not-a-camera")
    restore_resp = client.post("/api/cameras/not-a-camera/restore")

    assert delete_resp.status_code == 404
    assert restore_resp.status_code == 404
