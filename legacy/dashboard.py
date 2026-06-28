#!/usr/bin/env python3
"""
Dashboard API för Trafikverket-insamlaren.

Endpoints:
    GET /                   → Dashboard (static HTML)
    GET /api/status         → Collector status
    GET /api/cameras        → Camera IDs, names, and coordinates
    GET /api/logs           → Recent log lines
    GET /api/sensor-data    → Latest weather + road data
    GET /api/images/{date}/{filename} → Serve camera image
"""
import json
import os
import time
from datetime import datetime
from pathlib import Path

import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from config import API_KEY, API_URL, CAMERA_COORDS, CAMERA_IDS

DATA_DIR = os.environ.get("DATA_DIR", os.path.join(os.path.dirname(__file__), "data"))
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

app = FastAPI(title="Trafik Collector Dashboard")

# Serve static files
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", response_class=HTMLResponse)
async def index():
    index_path = os.path.join(STATIC_DIR, "index.html")
    with open(index_path, "r", encoding="utf-8") as f:
        return f.read()


@app.get("/api/status")
async def get_status():
    status_path = os.path.join(DATA_DIR, "status.json")
    if not os.path.exists(status_path):
        return {
            "running": False,
            "message": "Collector har inte startats ännu. Kör: docker compose up -d",
        }
    try:
        with open(status_path, "r", encoding="utf-8") as f:
            status = json.load(f)

        # Check if collector is stale (no update in 3 minutes)
        last = datetime.fromisoformat(status.get("last_update", "2000-01-01"))
        age_s = (datetime.now() - last).total_seconds()
        status["stale"] = age_s > 180
        status["seconds_since_update"] = int(age_s)

        return status
    except Exception as e:
        return {"running": False, "error": str(e)}


# ---- Camera exclusion helpers ----
EXCLUDED_FILE = os.path.join(DATA_DIR, "excluded_cameras.json")


def _load_excluded() -> list[str]:
    try:
        with open(EXCLUDED_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def _save_excluded(ids: list[str]) -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(EXCLUDED_FILE, "w", encoding="utf-8") as f:
        json.dump(sorted(set(ids)), f, indent=2)


# ---- Camera info cache (photo URLs, names from Trafikverket API) ----
_camera_info_cache: dict = {"data": {}, "ts": 0}
CAMERA_INFO_TTL = 300  # 5 minutes
CAMERA_INFO_FILE = os.path.join(DATA_DIR, "camera_info_cache.json")


def _get_camera_info() -> dict[str, dict]:
    """Fetch camera metadata from Trafikverket API with caching."""
    now = time.time()
    if _camera_info_cache["data"] and (now - _camera_info_cache["ts"]) < CAMERA_INFO_TTL:
        return _camera_info_cache["data"]

    # Try API fetch
    try:
        xml = f"""
        <REQUEST>
            <LOGIN authenticationkey="{API_KEY}" />
            <QUERY objecttype="Camera" schemaversion="1">
                <FILTER>
                    <EQ name="Active" value="true" />
                    <EQ name="HasFullSizePhoto" value="true" />
                </FILTER>
                <INCLUDE>Id</INCLUDE>
                <INCLUDE>Name</INCLUDE>
                <INCLUDE>Description</INCLUDE>
                <INCLUDE>PhotoUrl</INCLUDE>
            </QUERY>
        </REQUEST>
        """
        r = requests.post(API_URL, data=xml,
                          headers={"Content-Type": "text/xml"}, timeout=10)
        r.raise_for_status()
        result = r.json()
        cameras = result.get("RESPONSE", {}).get("RESULT", [{}])[0].get("Camera", [])

        info = {}
        for cam in cameras:
            cam_id = cam.get("Id", "")
            if cam_id in {c for c in CAMERA_IDS}:
                info[cam_id] = {
                    "name": cam.get("Name", ""),
                    "description": cam.get("Description", ""),
                    "photo_url": cam.get("PhotoUrl", ""),
                }

        _camera_info_cache["data"] = info
        _camera_info_cache["ts"] = now

        # Persist to disk
        try:
            os.makedirs(DATA_DIR, exist_ok=True)
            with open(CAMERA_INFO_FILE, "w", encoding="utf-8") as f:
                json.dump(info, f, ensure_ascii=False)
        except Exception:
            pass

        return info
    except Exception as e:
        # Fall back to disk cache
        try:
            with open(CAMERA_INFO_FILE, "r", encoding="utf-8") as f:
                info = json.load(f)
            _camera_info_cache["data"] = info
            _camera_info_cache["ts"] = now - CAMERA_INFO_TTL + 60  # retry in 1 min
            return info
        except Exception:
            return {}


@app.get("/api/cameras")
async def get_cameras():
    """Return active cameras (minus excluded) with coordinates and photo URLs."""
    excluded = set(_load_excluded())

    # Get camera info (photo URLs, names) from cache or API
    cam_info = _get_camera_info()

    cameras = []
    for cam_id in CAMERA_IDS:
        if cam_id in excluded:
            continue
        coords = CAMERA_COORDS.get(cam_id)
        info = cam_info.get(cam_id, {})
        cameras.append({
            "id": cam_id,
            "name": info.get("name", cam_id.split("_")[-1]),
            "description": info.get("description", ""),
            "photo_url": info.get("photo_url", ""),
            "lat": coords[0] if coords else None,
            "lng": coords[1] if coords else None,
        })
    return {"cameras": cameras, "total": len(CAMERA_IDS), "excluded": len(excluded)}


@app.get("/api/cameras/excluded")
async def get_excluded_cameras():
    """Return list of excluded camera IDs."""
    excluded = _load_excluded()
    cameras = []
    for cam_id in excluded:
        coords = CAMERA_COORDS.get(cam_id)
        cameras.append({
            "id": cam_id,
            "name": cam_id.split("_")[-1],
            "lat": coords[0] if coords else None,
            "lng": coords[1] if coords else None,
        })
    return {"excluded": cameras}


@app.delete("/api/cameras/{camera_id}")
async def exclude_camera(camera_id: str):
    """Exclude a camera from collection and map."""
    if camera_id not in CAMERA_IDS:
        raise HTTPException(404, f"Camera {camera_id} not found in config")
    excluded = _load_excluded()
    if camera_id not in excluded:
        excluded.append(camera_id)
        _save_excluded(excluded)
    return {"ok": True, "excluded_count": len(excluded)}


@app.post("/api/cameras/{camera_id}/restore")
async def restore_camera(camera_id: str):
    """Restore a previously excluded camera."""
    excluded = _load_excluded()
    excluded = [c for c in excluded if c != camera_id]
    _save_excluded(excluded)
    return {"ok": True, "excluded_count": len(excluded)}


@app.get("/api/logs")
async def get_logs(lines: int = 80):
    log_path = os.path.join(DATA_DIR, "collector.log")
    if not os.path.exists(log_path):
        return {"lines": [], "total": 0}

    try:
        with open(log_path, "r", encoding="utf-8") as f:
            all_lines = f.readlines()

        recent = all_lines[-lines:] if len(all_lines) > lines else all_lines

        parsed = []
        for line in recent:
            line = line.strip()
            if not line:
                continue
            # Detect level
            level = "info"
            if "[ERROR]" in line or "[CRITICAL]" in line:
                level = "error"
            elif "[WARNING]" in line:
                level = "warning"
            elif "[DEBUG]" in line:
                level = "debug"
            parsed.append({"text": line, "level": level})

        return {"lines": parsed, "total": len(all_lines)}
    except Exception as e:
        return {"lines": [], "error": str(e)}


@app.get("/api/sensor-data")
async def get_sensor_data():
    """Return latest weather + road condition records from today's JSONL."""
    today = datetime.now().strftime("%Y-%m-%d")
    jsonl_path = os.path.join(DATA_DIR, today, "sensor_data.jsonl")
    if not os.path.exists(jsonl_path):
        return {"weather": [], "road_conditions": [], "date": today}

    try:
        weather = []
        road = []
        situations = []
        # Read last 300 lines (most recent data)
        with open(jsonl_path, "r", encoding="utf-8") as f:
            all_lines = f.readlines()

        for line in all_lines[-300:]:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                rtype = record.get("type")
                if rtype == "weather":
                    weather.append(record)
                elif rtype == "road_condition":
                    road.append(record)
                elif rtype == "situation":
                    situations.append(record)
            except json.JSONDecodeError:
                continue

        # Dedupe weather by station (keep latest)
        seen = {}
        for w in weather:
            seen[w.get("station_id")] = w
        weather = list(seen.values())

        # Dedupe road conditions by id (keep latest)
        seen_road = {}
        for r in road:
            seen_road[r.get("id")] = r
        road = list(seen_road.values())

        # Dedupe situations by deviation_id (keep latest)
        seen_sit = {}
        for s in situations:
            seen_sit[s.get("deviation_id")] = s
        situations = list(seen_sit.values())

        return {
            "weather": weather,
            "road_conditions": road,
            "situations": situations,
            "date": today,
        }
    except Exception as e:
        return {"weather": [], "road_conditions": [], "situations": [], "error": str(e)}


@app.get("/api/images/{date}/{filename}")
async def get_image(date: str, filename: str):
    """Serve a camera image."""
    # Validate path to prevent directory traversal
    if ".." in date or ".." in filename or "/" in filename:
        raise HTTPException(status_code=400, detail="Invalid path")
    
    img_path = os.path.join(DATA_DIR, date, "images", filename)
    if not os.path.exists(img_path):
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(img_path, media_type="image/jpeg")


@app.get("/api/images/latest")
async def get_latest_images():
    """Return URLs to the most recent image for each camera."""
    today = datetime.now().strftime("%Y-%m-%d")
    img_dir = os.path.join(DATA_DIR, today, "images")
    if not os.path.exists(img_dir):
        return {"images": [], "date": today}

    files = sorted(
        [f for f in os.listdir(img_dir) if f.endswith(".jpg")],
        reverse=True,
    )

    # Group by camera name (cam_<name>_<timestamp>.jpg)
    latest = {}
    for f in files:
        # Extract camera name: everything between cam_ and _YYYY-MM-DD
        parts = f.split("_")
        if len(parts) >= 3:
            # Find the date part index
            cam_parts = []
            for p in parts[1:]:
                if p.startswith("202"):
                    break
                cam_parts.append(p)
            cam_name = "_".join(cam_parts) if cam_parts else parts[1]
            if cam_name not in latest:
                latest[cam_name] = {
                    "camera": cam_name,
                    "filename": f,
                    "url": f"/api/images/{today}/{f}",
                    "size_bytes": os.path.getsize(os.path.join(img_dir, f)),
                }

    return {"images": list(latest.values()), "date": today}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
