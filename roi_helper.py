#!/usr/bin/env python3
"""
ROI Helper Tool — Interactive polygon drawing for camera ROI calibration.

Opens each camera's latest image in an OpenCV window. Click to define
polygon corners for each road segment, then saves coordinates directly
to camera_config.json.

Controls:
─────────────────────────────────────────────────────────
  LEFT CLICK       Add polygon vertex at cursor position
  RIGHT CLICK      Undo last vertex
  ENTER / SPACE    Finish current polygon → name it
  S                Skip to next camera (no changes)
  R                Restart current polygon (clear vertices)
  D                Delete all ROIs for current camera
  Q / ESC          Quit and save

Usage:
    python roi_helper.py                  # All cameras
    python roi_helper.py --camera CAM_ID  # Single camera
    python roi_helper.py --list           # List all camera IDs
"""

import argparse
import json
import os
import sys
import time

import cv2
import numpy as np
import requests
from PIL import ImageFont, ImageDraw, Image

from config import API_KEY, API_URL, CAMERA_IDS, CAMERA_COORDS, MAX_RETRIES, RETRY_BACKOFF

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "camera_config.json")
WINDOW_NAME = "ROI Helper — Click to draw polygon"
COLORS = [
    (0, 255, 0),    # Green
    (255, 165, 0),  # Orange (BGR)
    (0, 0, 255),    # Red
    (255, 255, 0),  # Cyan
    (255, 0, 255),  # Magenta
    (128, 255, 128),# Light green
]
FONT = cv2.FONT_HERSHEY_SIMPLEX

# Load a TrueType font for Unicode rendering (PIL)
try:
    _PIL_FONT_BASE = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
except OSError:
    _PIL_FONT_BASE = ImageFont.load_default()

_pil_font_cache: dict[int, ImageFont.FreeTypeFont] = {}

def _get_pil_font(size: int) -> ImageFont.FreeTypeFont:
    """Get a cached PIL font at the given pixel size."""
    if size not in _pil_font_cache:
        try:
            _pil_font_cache[size] = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", size)
        except OSError:
            _pil_font_cache[size] = ImageFont.load_default()
    return _pil_font_cache[size]

def pil_puttext(img: np.ndarray, text: str, org: tuple[int, int],
                color_bgr: tuple[int, int, int], scale: float = 0.5,
                thickness: int = 1) -> np.ndarray:
    """Draw Unicode text on an OpenCV image using PIL.
    
    `scale` maps to approximate font size: size = int(scale * 28 + 6).
    `thickness` > 1 renders a bold effect via slight offsets.
    """
    font_size = int(scale * 28 + 6)
    font = _get_pil_font(font_size)
    # Convert BGR → RGB for PIL
    color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    x, y = org
    # Adjust y: cv2.putText uses baseline, PIL uses top-left
    y_adjusted = y - font_size + 2
    if thickness > 1:
        # Simulate bold by drawing at slight offsets
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                draw.text((x + dx, y_adjusted + dy), text, font=font, fill=color_rgb)
    else:
        draw.text((x, y_adjusted), text, font=font, fill=color_rgb)
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


# ---------------------------------------------------------------------------
# Config I/O
# ---------------------------------------------------------------------------
def load_config() -> dict:
    """Load existing camera_config.json or create empty structure."""
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"_comment": "ROI polygons per camera. Pixel [x, y] in native resolution.", "cameras": {}}


def save_config(config: dict) -> None:
    """Save config to camera_config.json with nice formatting."""
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"  💾 Saved → {CONFIG_PATH}")


# ---------------------------------------------------------------------------
# Image fetching
# ---------------------------------------------------------------------------
def fetch_camera_image(camera_id: str) -> tuple[np.ndarray | None, dict]:
    """Fetch the latest image and metadata for a camera from Trafikverket API.

    Returns (frame, metadata) where metadata contains:
        name, description, direction (compass bearing).
    """
    empty_meta = {"name": camera_id, "description": "", "direction": None}

    xml_query = f"""
    <REQUEST>
        <LOGIN authenticationkey="{API_KEY}" />
        <QUERY objecttype="Camera" schemaversion="1">
            <FILTER>
                <EQ name="Id" value="{camera_id}" />
            </FILTER>
        </QUERY>
    </REQUEST>
    """
    try:
        resp = requests.post(
            API_URL,
            data=xml_query.encode("utf-8"),
            headers={"Content-Type": "text/xml; charset=utf-8"},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"  ❌ API error: {e}")
        return None, empty_meta

    results = data.get("RESPONSE", {}).get("RESULT", [])
    cameras = results[0].get("Camera", []) if results else []
    if not cameras:
        print(f"  ❌ Camera {camera_id} not found in API")
        return None, empty_meta

    cam = cameras[0]
    meta = {
        "name": cam.get("Name", camera_id),
        "description": cam.get("Description", ""),
        "direction": cam.get("Direction"),
    }

    photo_url = cam.get("PhotoUrl", "")
    if not photo_url:
        print(f"  ❌ No PhotoUrl for {camera_id}")
        return None, meta

    if cam.get("HasFullSizePhoto"):
        photo_url += "?type=fullsize"

    print(f"  📷 Fetching image: {meta['name']}...")
    if meta["description"]:
        print(f"  📍 {meta['description']}")
    if meta["direction"] is not None:
        print(f"  🧭 Compass bearing: {meta['direction']}°")

    try:
        img_resp = requests.get(photo_url, timeout=30)
        img_resp.raise_for_status()
        arr = np.frombuffer(img_resp.content, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is not None:
            print(f"  ✅ Image loaded: {frame.shape[1]}×{frame.shape[0]}")
        return frame, meta
    except Exception as e:
        print(f"  ❌ Image fetch error: {e}")
        return None, meta


# ---------------------------------------------------------------------------
# Interactive polygon drawing
# ---------------------------------------------------------------------------

# States for the in-window state machine
STATE_DRAWING = "drawing"
STATE_SELECT_DIRECTION = "select_direction"
STATE_SELECT_LANES = "select_lanes"


class PolygonDrawer:
    """Interactive OpenCV polygon drawing with keyboard-only controls.

    No terminal input() calls — everything happens in the OpenCV window.
    After completing a polygon (Enter), press 1 or 2 to tag direction,
    then 1-4 to set lane count. All metadata is auto-filled.
    """

    def __init__(self, frame: np.ndarray, camera_id: str, existing_rois: list, meta: dict | None = None):
        self.original = frame.copy()
        self.camera_id = camera_id
        self.h, self.w = frame.shape[:2]
        self.rois = list(existing_rois)
        self.current_points: list[list[int]] = []
        self.mouse_pos = (0, 0)
        self.meta = meta or {}
        self.state = STATE_DRAWING
        self._pending_roi: dict | None = None  # Partially built ROI
        self._delete_armed = False  # D pressed once
        self._status_msg = ""  # Temporary status message
        self._status_until = 0.0  # When to clear status msg

    def mouse_callback(self, event, x, y, flags, param):
        self.mouse_pos = (x, y)
        if self.state != STATE_DRAWING:
            return  # Ignore clicks during selection states
        if event == cv2.EVENT_LBUTTONDOWN:
            self.current_points.append([x, y])
            self._delete_armed = False
        elif event == cv2.EVENT_RBUTTONDOWN:
            if self.current_points:
                self.current_points.pop()

    def _set_status(self, msg: str, duration: float = 3.0):
        self._status_msg = msg
        self._status_until = time.time() + duration

    def _get_camera_direction_text(self) -> str:
        """Extract direction from camera description."""
        description = self.meta.get("description", "")
        if not description:
            return ""
        lower = description.lower()
        if "mot stockholm" in lower:
            return "MOT STOCKHOLM"
        if "mot södertälje" in lower:
            return "MOT SÖDERTÄLJE"
        if "riktad mot" in lower:
            idx = lower.index("riktad mot")
            tail = description[idx + len("riktad mot"):].strip().rstrip(".")
            return f"MOT {tail.upper()}"
        return ""

    def try_finish_polygon(self) -> bool:
        """Called on Enter — switches to direction selection state if polygon is valid."""
        if len(self.current_points) < 3:
            self._set_status("Need at least 3 vertices!", 3.0)
            return False

        self._pending_roi = {
            "polygon": [list(p) for p in self.current_points],
        }
        self.state = STATE_SELECT_DIRECTION
        return True

    def select_direction(self, key: int) -> bool:
        """Handle direction selection (1 = Stockholm/Northbound, 2 = Södertälje/Southbound)."""
        if key == ord("1"):
            self._pending_roi["road_id"] = "E4_Northbound"
            self._pending_roi["direction_relative_to_camera"] = "away"
            self._pending_roi["capacity_vph"] = 6000  # 3 lanes × 2000
        elif key == ord("2"):
            self._pending_roi["road_id"] = "E4_Southbound"
            self._pending_roi["direction_relative_to_camera"] = "towards"
            self._pending_roi["capacity_vph"] = 6000
        elif key == 27:  # ESC to cancel
            self.state = STATE_DRAWING
            self._pending_roi = None
            self._set_status("Cancelled — polygon kept, press Enter to retry")
            return False
        else:
            return False  # Ignore other keys

        self.state = STATE_SELECT_LANES
        return True

    def select_lanes(self, key: int) -> bool:
        """Handle lane count selection (1-4)."""
        if key in (ord("1"), ord("2"), ord("3"), ord("4")):
            num_lanes = int(chr(key))
            self._pending_roi["num_lanes"] = num_lanes
            self._pending_roi["capacity_vph"] = num_lanes * 2000

            # Finalize ROI
            roi = self._pending_roi
            self.rois.append(roi)
            self.current_points = []
            self._pending_roi = None
            self.state = STATE_DRAWING

            road_id = roi["road_id"]
            self._set_status(
                f"ROI '{road_id}' saved ({num_lanes} lanes, {roi['capacity_vph']} VPH)",
                4.0,
            )
            print(f"  ✅ ROI '{road_id}' added ({len(roi['polygon'])} vertices, {num_lanes} lanes)")
            return True
        elif key == 27:  # ESC to go back
            self.state = STATE_SELECT_DIRECTION
            return False
        return False

    def render(self) -> np.ndarray:
        """Draw current state on the frame."""
        display = self.original.copy()

        # Draw existing (saved) ROIs with semi-transparent fill
        for i, roi in enumerate(self.rois):
            color = COLORS[i % len(COLORS)]
            pts = np.array(roi["polygon"], dtype=np.int32)
            # Semi-transparent fill
            overlay = display.copy()
            cv2.fillPoly(overlay, [pts], color)
            cv2.addWeighted(overlay, 0.2, display, 0.8, 0, display)
            cv2.polylines(display, [pts], isClosed=True, color=color, thickness=2)

            # Label
            cx = int(np.mean(pts[:, 0]))
            cy = int(np.mean(pts[:, 1]))
            label = f'{roi["road_id"]} ({roi["num_lanes"]}L)'
            display = pil_puttext(display, label, (cx - 60, cy), color, 0.55, 2)

        # Draw current in-progress polygon
        if self.current_points:
            pts = np.array(self.current_points, dtype=np.int32)
            for j in range(len(pts) - 1):
                cv2.line(display, tuple(pts[j]), tuple(pts[j + 1]), (0, 255, 255), 2)

            last = tuple(pts[-1])
            cv2.line(display, last, self.mouse_pos, (0, 255, 255), 1)

            if len(pts) >= 3:
                cv2.line(display, self.mouse_pos, tuple(pts[0]), (0, 200, 200), 1)
                # Semi-transparent preview fill
                preview_pts = np.array(
                    self.current_points + [list(self.mouse_pos)], dtype=np.int32
                )
                overlay = display.copy()
                cv2.fillPoly(overlay, [preview_pts], (0, 255, 255))
                cv2.addWeighted(overlay, 0.1, display, 0.9, 0, display)

            for pt in pts:
                cv2.circle(display, tuple(pt), 5, (0, 255, 255), -1)
                cv2.circle(display, tuple(pt), 5, (0, 0, 0), 1)

        # ── Build HUD ──
        cam_name = self.meta.get("name", self.camera_id)
        cam_dir = self._get_camera_direction_text()
        bearing = self.meta.get("direction")
        bearing_text = f" | {bearing}°" if bearing is not None else ""

        hud_lines: list[tuple[str, tuple[int, int, int], float, int]] = []

        # Line 1: Camera name
        hud_lines.append((
            f"Camera: {cam_name} ({self.camera_id})",
            (255, 255, 255), 0.6, 1,
        ))

        # Line 2: Direction (if known)
        if cam_dir:
            hud_lines.append((
                f"→ {cam_dir}{bearing_text}",
                (0, 255, 200), 0.6, 2,
            ))

        # State-specific HUD lines
        if self.state == STATE_DRAWING:
            hud_lines.append((
                f"Saved ROIs: {len(self.rois)}  |  Vertices: {len(self.current_points)}",
                (200, 200, 200), 0.5, 1,
            ))
            hud_lines.append((
                "CLICK: vertex | RIGHT-CLICK: undo | ENTER: finish polygon",
                (150, 200, 255), 0.45, 1,
            ))
            hud_lines.append((
                "R: restart | D+D: delete all | S: skip | Q/ESC: quit & save",
                (150, 200, 255), 0.45, 1,
            ))

        elif self.state == STATE_SELECT_DIRECTION:
            hud_lines.append((
                f"Polygon complete ({len(self._pending_roi['polygon'])} vertices) — select direction:",
                (255, 255, 100), 0.55, 1,
            ))
            hud_lines.append((
                ">>> 1: MOT STOCKHOLM (Northbound)    2: MOT SODERTALJE (Southbound) <<<",
                (0, 255, 150), 0.7, 2,
            ))
            hud_lines.append((
                "ESC: cancel",
                (150, 150, 150), 0.4, 1,
            ))

        elif self.state == STATE_SELECT_LANES:
            road = self._pending_roi.get("road_id", "")
            hud_lines.append((
                f"{road} — how many lanes in this zone?",
                (255, 255, 100), 0.55, 1,
            ))
            hud_lines.append((
                ">>> 1: 1 lane    2: 2 lanes    3: 3 lanes    4: 4 lanes <<<",
                (0, 255, 150), 0.7, 2,
            ))
            hud_lines.append((
                "ESC: go back",
                (150, 150, 150), 0.4, 1,
            ))

        # Status message (temporary)
        if self._status_msg and time.time() < self._status_until:
            hud_lines.append((self._status_msg, (100, 255, 255), 0.5, 1))

        # Mouse position
        hud_lines.append((
            f"Mouse: ({self.mouse_pos[0]}, {self.mouse_pos[1]})",
            (180, 180, 180), 0.4, 1,
        ))

        # Render HUD
        hud_height = 22 * len(hud_lines) + 15
        cv2.rectangle(display, (0, 0), (self.w, hud_height), (0, 0, 0), -1)
        hud_y = 22
        for text, color, scale, thickness in hud_lines:
            display = pil_puttext(display, text, (10, hud_y), color, scale, thickness)
            hud_y += 22

        return display


# ---------------------------------------------------------------------------
# Main interactive loop
# ---------------------------------------------------------------------------
def calibrate_camera(camera_id: str, config: dict) -> bool:
    """Interactive ROI calibration for one camera. Returns True if user wants to quit."""
    print(f"\n{'='*60}")
    print(f"  📷 Camera: {camera_id}")
    print(f"{'='*60}")

    # Fetch image + metadata
    frame, meta = fetch_camera_image(camera_id)
    if frame is None:
        print("  ⏭  Skipping — could not load image")
        return False

    # Load existing ROIs
    cam_config = config.get("cameras", {}).get(camera_id, {})
    existing_rois = cam_config.get("rois", [])

    if existing_rois:
        print(f"  ℹ️  {len(existing_rois)} existing ROI(s) loaded")

    drawer = PolygonDrawer(frame, camera_id, existing_rois, meta=meta)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, min(frame.shape[1], 1400), min(frame.shape[0], 900))
    cv2.setMouseCallback(WINDOW_NAME, drawer.mouse_callback)

    quit_all = False

    while True:
        display = drawer.render()
        cv2.imshow(WINDOW_NAME, display)
        key = cv2.waitKey(30) & 0xFF

        if key == 255:
            continue  # No key pressed

        # ── Direction selection state ──
        if drawer.state == STATE_SELECT_DIRECTION:
            drawer.select_direction(key)
            continue

        # ── Lane count selection state ──
        if drawer.state == STATE_SELECT_LANES:
            drawer.select_lanes(key)
            continue

        # ── Normal drawing state ──
        if key == 27 or key == ord("q"):  # ESC or Q → quit
            quit_all = True
            break

        elif key == ord("s"):  # Skip camera
            print("  ⏭  Skipped")
            break

        elif key == 13 or key == 32:  # ENTER or SPACE → finish polygon
            drawer.try_finish_polygon()

        elif key == ord("r"):  # Restart current polygon
            drawer.current_points = []
            drawer._set_status("Polygon cleared")
            print("  🔄 Polygon cleared")

        elif key == ord("d"):  # Delete all ROIs (double-press)
            if not drawer.rois:
                drawer._set_status("No ROIs to delete")
            elif drawer._delete_armed:
                drawer.rois = []
                drawer._delete_armed = False
                drawer._set_status("All ROIs deleted!")
                print("  🗑  All ROIs deleted")
            else:
                drawer._delete_armed = True
                drawer._set_status("Press D again to confirm delete all ROIs", 3.0)

    cv2.destroyAllWindows()

    # Save ROIs to config
    if drawer.rois:
        if "cameras" not in config:
            config["cameras"] = {}
        config["cameras"][camera_id] = {"rois": drawer.rois}
        save_config(config)
    elif camera_id in config.get("cameras", {}):
        # ROIs were deleted
        del config["cameras"][camera_id]
        save_config(config)

    return quit_all


def main():
    parser = argparse.ArgumentParser(
        description="Interactive ROI Helper — draw polygon zones on camera images"
    )
    parser.add_argument("--camera", type=str, help="Calibrate a single camera by ID")
    parser.add_argument("--list", action="store_true", help="List all camera IDs and exit")
    parser.add_argument("--skip-configured", action="store_true", help="Skip cameras that already have ROIs")
    args = parser.parse_args()

    if not API_KEY:
        print("❌ Missing API key. Set TRAFIKVERKET_API_KEY in .env")
        sys.exit(1)

    config = load_config()

    if args.list:
        configured = set(config.get("cameras", {}).keys())
        print(f"\n  Total cameras: {len(CAMERA_IDS)}")
        print(f"  Configured:    {len(configured)}")
        print(f"  Remaining:     {len(CAMERA_IDS) - len(configured)}\n")
        for cid in CAMERA_IDS:
            status = "✅" if cid in configured else "⬜"
            rois = len(config.get("cameras", {}).get(cid, {}).get("rois", []))
            roi_info = f" — {rois} ROI(s)" if rois else ""
            print(f"  {status} {cid}{roi_info}")
        return

    if args.camera:
        if args.camera not in CAMERA_IDS:
            print(f"❌ Unknown camera: {args.camera}")
            print(f"   Use --list to see all camera IDs")
            sys.exit(1)
        calibrate_camera(args.camera, config)
        return

    # All cameras
    configured = set(config.get("cameras", {}).keys())
    cameras = [c for c in CAMERA_IDS if c not in configured] if args.skip_configured else list(CAMERA_IDS)

    print(f"\n🎯 ROI Helper Tool — {len(cameras)} cameras to calibrate")
    print(f"   Already configured: {len(configured)}")
    if args.skip_configured:
        print(f"   Skipping configured cameras")
    print()

    for i, cam_id in enumerate(cameras, 1):
        print(f"\n[{i}/{len(cameras)}]", end="")
        quit_all = calibrate_camera(cam_id, config)
        if quit_all:
            print(f"\n👋 Quit after {i} camera(s). Progress saved.")
            break
    else:
        print(f"\n🎉 All {len(cameras)} cameras calibrated!")

    # Summary
    configured = set(config.get("cameras", {}).keys())
    total_rois = sum(
        len(config["cameras"][c].get("rois", []))
        for c in configured
    )
    print(f"\n📊 Summary: {len(configured)}/{len(CAMERA_IDS)} cameras configured, {total_rois} total ROIs")


if __name__ == "__main__":
    main()
