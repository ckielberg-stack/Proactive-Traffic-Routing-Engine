#!/usr/bin/env python3
"""
ROI Helper Tool — Interactive polygon drawing for camera ROI calibration.

Opens each camera's latest image in an OpenCV window. Click to define
polygon corners for each road segment, then saves coordinates directly
to camera_config.json.

Controls:
─────────────────────────────────────────────────────────
  EXCLUSION ZONE DRAWING:
    X                Enter exclusion zone mode (draw rectangle)
    LEFT CLICK       Set corner 1 (top-left), then corner 2 (bottom-right)
    RIGHT CLICK      Undo last corner
    ENTER / SPACE    Confirm zone
    Z                Delete last exclusion zone (from draw mode)
    ESC              Cancel back to draw mode
  DRAW MODE (default):
    LEFT CLICK       Add polygon vertex at cursor position
    RIGHT CLICK      Undo last vertex
    ENTER / SPACE    Finish current polygon → name it
    E                Switch to Edit mode
    S                Skip to next camera (no changes)
    R                Restart current polygon (clear vertices)
    D                Delete all ROIs for current camera
    Q / ESC          Quit and save

  EDIT MODE:
    LEFT CLICK       Grab nearest vertex (≤12px) or insert on edge (≤8px)
    DRAG             Move grabbed vertex
    RIGHT CLICK      Delete vertex under cursor (min 3 vertices)
    E                Switch back to Draw mode
    S                Skip to next camera
    Q / ESC          Quit and save

Usage:
    python roi_helper.py                  # All cameras
    python roi_helper.py --camera CAM_ID  # Single camera
    python roi_helper.py --list           # List all camera IDs
"""

import argparse
import json
import math
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
HUD_HEIGHT = 160  # Fixed height for HUD area above the image
COLORS = [
    (0, 255, 0),    # Green
    (255, 165, 0),  # Orange (BGR)
    (0, 0, 255),    # Red
    (255, 255, 0),  # Cyan
    (255, 0, 255),  # Magenta
    (128, 255, 128),# Light green
]
FONT = cv2.FONT_HERSHEY_SIMPLEX

# Edit mode constants
VERTEX_GRAB_RADIUS = 12  # pixels — how close to grab a vertex
EDGE_INSERT_RADIUS = 8   # pixels — how close to insert on edge
VERTEX_HANDLE_RADIUS = 6 # drawn handle size

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
# Geometry helpers
# ---------------------------------------------------------------------------

def _point_to_segment_dist(
    px: float, py: float,
    ax: float, ay: float,
    bx: float, by: float,
) -> tuple[float, float, float]:
    """Distance from point (px,py) to line segment (ax,ay)-(bx,by).

    Returns (distance, proj_x, proj_y) where proj is the closest point on segment.
    """
    dx, dy = bx - ax, by - ay
    if dx == 0 and dy == 0:
        return math.hypot(px - ax, py - ay), ax, ay
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)))
    proj_x = ax + t * dx
    proj_y = ay + t * dy
    return math.hypot(px - proj_x, py - proj_y), proj_x, proj_y


# ---------------------------------------------------------------------------
# Interactive polygon drawing
# ---------------------------------------------------------------------------

# States for the in-window state machine
STATE_DRAWING = "drawing"
STATE_EDITING = "editing"
STATE_SELECT_DIRECTION = "select_direction"
STATE_SELECT_LANES = "select_lanes"
STATE_PERSPECTIVE_RULER = "perspective_ruler"
STATE_BEV_CALIBRATION = "bev_calibration"  # Expert Audit Fix 2
STATE_EXCLUSION_ZONE = "exclusion_zone"

# Road marking reference: Swedish dashed centre line = 3 m dash + 9 m gap = 12 m
RULER_DASH_SPACING_M = 12.0

# Default BEV reference rectangle dimensions (meters)
DEFAULT_BEV_WIDTH_M = 3.5   # One lane width
DEFAULT_BEV_LENGTH_M = 12.0  # One dash cycle


class PolygonDrawer:
    """Interactive OpenCV polygon drawing with keyboard-only controls.

    No terminal input() calls — everything happens in the OpenCV window.
    After completing a polygon (Enter), press 1 or 2 to tag direction,
    then 1-4 to set lane count. All metadata is auto-filled.

    The HUD is rendered ABOVE the image by expanding the canvas. Mouse
    coordinates are offset so clicks map to actual image pixel positions.
    """

    def __init__(self, frame: np.ndarray, camera_id: str, existing_rois: list,
                 meta: dict | None = None, existing_exclusion_zones: list | None = None):
        self.original = frame.copy()
        self.camera_id = camera_id
        self.h, self.w = frame.shape[:2]
        self.rois = list(existing_rois)
        self.exclusion_zones: list[list[int]] = list(existing_exclusion_zones or [])
        self.current_points: list[list[int]] = []
        self.mouse_pos = (0, 0)          # In IMAGE coordinate space
        self.mouse_pos_raw = (0, 0)      # Raw canvas coordinates
        self.meta = meta or {}
        self.state = STATE_DRAWING
        self._pending_roi: dict | None = None  # Partially built ROI
        self._delete_armed = False  # D pressed once
        self._status_msg = ""  # Temporary status message
        self._status_until = 0.0  # When to clear status msg

        # Exclusion zone drawing state
        self._excl_corner1: tuple[int, int] | None = None

        # Perspective ruler state
        self._ruler_points: list[tuple[int, int]] = []  # (x, y) image coords
        self._perspective_poly: np.poly1d | None = None

        # Edit mode state
        self._dragging = False
        self._drag_roi_idx: int = -1
        self._drag_vert_idx: int = -1
        self._hover_roi_idx: int = -1    # Which ROI has nearest vertex/edge
        self._hover_vert_idx: int = -1   # Nearest vertex (-1 if edge is closer)
        self._hover_edge_idx: int = -1   # Nearest edge
        self._hover_proj: tuple[float, float] = (0, 0)  # Projection point on edge

    def mouse_callback(self, event, x, y, flags, param):
        # Convert from canvas coords to image coords (subtract HUD offset)
        img_x = x
        img_y = y - HUD_HEIGHT
        self.mouse_pos_raw = (x, y)
        self.mouse_pos = (img_x, max(0, img_y))

        # ── Edit mode mouse handling ──
        if self.state == STATE_EDITING:
            self._handle_edit_mouse(event, img_x, img_y, flags)
            return

        # ── Perspective ruler mode (clicks add ruler points) ──
        # ── Exclusion zone mode ──
        if self.state == STATE_EXCLUSION_ZONE:
            if img_y < 0:
                return
            if event == cv2.EVENT_LBUTTONDOWN:
                if self._excl_corner1 is None:
                    self._excl_corner1 = (img_x, img_y)
                    self._set_status("Corner 1 set — click bottom-right corner")
                else:
                    # Build rectangle from the two clicks
                    x1 = min(self._excl_corner1[0], img_x)
                    y1 = min(self._excl_corner1[1], img_y)
                    x2 = max(self._excl_corner1[0], img_x)
                    y2 = max(self._excl_corner1[1], img_y)
                    self.exclusion_zones.append([x1, y1, x2, y2])
                    self._excl_corner1 = None
                    self.state = STATE_DRAWING
                    self._set_status(
                        f"Exclusion zone added: [{x1}, {y1}, {x2}, {y2}] "
                        f"({len(self.exclusion_zones)} total)", 4.0
                    )
                    print(f"  🚫 Exclusion zone added: [{x1}, {y1}, {x2}, {y2}]")
            elif event == cv2.EVENT_RBUTTONDOWN:
                if self._excl_corner1 is not None:
                    self._excl_corner1 = None
                    self._set_status("Corner 1 undone — click top-left again")
            return

        if self.state == STATE_PERSPECTIVE_RULER:
            if img_y < 0:
                return
            if event == cv2.EVENT_LBUTTONDOWN:
                self._ruler_points.append((img_x, img_y))
                n = len(self._ruler_points)
                print(f"    📍 Ruler point #{n}: Y={img_y}px → {(n-1) * RULER_DASH_SPACING_M:.0f} m")
            elif event == cv2.EVENT_RBUTTONDOWN:
                if self._ruler_points:
                    self._ruler_points.pop()
                    self._set_status(f"Removed last point ({len(self._ruler_points)} remaining)")

        # ── BEV calibration mode (clicks add rectangle corners) ──
        if self.state == STATE_BEV_CALIBRATION:
            if img_y < 0:
                return
            if event == cv2.EVENT_LBUTTONDOWN:
                if len(self._bev_points) < 4:
                    self._bev_points.append((img_x, img_y))
                    n = len(self._bev_points)
                    labels = ["top-left", "top-right", "bottom-right", "bottom-left"]
                    print(f"    🔲 BEV corner #{n} ({labels[n-1]}): ({img_x}, {img_y})")
                    if n == 4:
                        self._set_status("4 points set — press ENTER to compute H")
                else:
                    self._set_status("Already have 4 points — press ENTER")
            elif event == cv2.EVENT_RBUTTONDOWN:
                if self._bev_points:
                    self._bev_points.pop()
                    self._set_status(f"Undid — {len(self._bev_points)} points")
            return

        # ── Draw mode (ignore clicks in HUD area) ──
        if img_y < 0:
            return
        if self.state != STATE_DRAWING:
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            self.current_points.append([img_x, img_y])
            self._delete_armed = False
        elif event == cv2.EVENT_RBUTTONDOWN:
            if self.current_points:
                self.current_points.pop()

    def _handle_edit_mouse(self, event, img_x: int, img_y: int, flags):
        """Handle mouse events in edit mode."""
        if event == cv2.EVENT_LBUTTONDOWN and img_y >= 0:
            # Try to grab a vertex first
            roi_idx, vert_idx, dist = self._find_nearest_vertex(img_x, img_y)
            if dist <= VERTEX_GRAB_RADIUS:
                self._dragging = True
                self._drag_roi_idx = roi_idx
                self._drag_vert_idx = vert_idx
                return

            # Try to insert on an edge
            roi_idx, edge_idx, dist, proj = self._find_nearest_edge(img_x, img_y)
            if dist <= EDGE_INSERT_RADIUS:
                # Insert new vertex after edge_idx
                poly = self.rois[roi_idx]["polygon"]
                insert_at = edge_idx + 1
                poly.insert(insert_at, [int(proj[0]), int(proj[1])])
                # Start dragging the new vertex immediately
                self._dragging = True
                self._drag_roi_idx = roi_idx
                self._drag_vert_idx = insert_at
                self._set_status(f"Inserted vertex at ({int(proj[0])}, {int(proj[1])})")
                return

        elif event == cv2.EVENT_MOUSEMOVE:
            if self._dragging and img_y >= 0:
                # Clamp to image bounds
                cx = max(0, min(img_x, self.w - 1))
                cy = max(0, min(img_y, self.h - 1))
                self.rois[self._drag_roi_idx]["polygon"][self._drag_vert_idx] = [cx, cy]
            else:
                # Update hover state for visual feedback
                self._update_hover(img_x, img_y)

        elif event == cv2.EVENT_LBUTTONUP:
            if self._dragging:
                self._dragging = False
                self._set_status("Vertex moved")

        elif event == cv2.EVENT_RBUTTONDOWN and img_y >= 0:
            # Delete vertex
            roi_idx, vert_idx, dist = self._find_nearest_vertex(img_x, img_y)
            if dist <= VERTEX_GRAB_RADIUS:
                poly = self.rois[roi_idx]["polygon"]
                if len(poly) <= 3:
                    self._set_status("Can't delete — polygon needs ≥ 3 vertices")
                else:
                    removed = poly.pop(vert_idx)
                    self._set_status(f"Deleted vertex ({removed[0]}, {removed[1]})")

    def _find_nearest_vertex(self, x: int, y: int) -> tuple[int, int, float]:
        """Find closest vertex across all ROIs. Returns (roi_idx, vert_idx, distance)."""
        best_dist = float("inf")
        best_roi = -1
        best_vert = -1
        for ri, roi in enumerate(self.rois):
            for vi, pt in enumerate(roi["polygon"]):
                d = math.hypot(x - pt[0], y - pt[1])
                if d < best_dist:
                    best_dist = d
                    best_roi = ri
                    best_vert = vi
        return best_roi, best_vert, best_dist

    def _find_nearest_edge(self, x: int, y: int) -> tuple[int, int, float, tuple[float, float]]:
        """Find closest edge across all ROIs. Returns (roi_idx, edge_idx, distance, projection_point)."""
        best_dist = float("inf")
        best_roi = -1
        best_edge = -1
        best_proj = (0.0, 0.0)
        for ri, roi in enumerate(self.rois):
            poly = roi["polygon"]
            n = len(poly)
            for ei in range(n):
                ax, ay = poly[ei]
                bx, by = poly[(ei + 1) % n]
                d, px, py = _point_to_segment_dist(x, y, ax, ay, bx, by)
                if d < best_dist:
                    best_dist = d
                    best_roi = ri
                    best_edge = ei
                    best_proj = (px, py)
        return best_roi, best_edge, best_dist, best_proj

    def _update_hover(self, x: int, y: int):
        """Update hover highlights for edit mode."""
        if not self.rois:
            self._hover_roi_idx = -1
            self._hover_vert_idx = -1
            self._hover_edge_idx = -1
            return

        roi_v, vert_v, dist_v = self._find_nearest_vertex(x, y)
        roi_e, edge_e, dist_e, proj_e = self._find_nearest_edge(x, y)

        if dist_v <= VERTEX_GRAB_RADIUS:
            self._hover_roi_idx = roi_v
            self._hover_vert_idx = vert_v
            self._hover_edge_idx = -1
        elif dist_e <= EDGE_INSERT_RADIUS:
            self._hover_roi_idx = roi_e
            self._hover_vert_idx = -1
            self._hover_edge_idx = edge_e
            self._hover_proj = proj_e
        else:
            self._hover_roi_idx = -1
            self._hover_vert_idx = -1
            self._hover_edge_idx = -1

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
        """Handle lane count selection (1-4). Finalizes ROI immediately."""
        if key in (ord("1"), ord("2"), ord("3"), ord("4")):
            num_lanes = int(chr(key))
            self._pending_roi["num_lanes"] = num_lanes
            self._pending_roi["capacity_vph"] = num_lanes * 2000

            # Finalize ROI (roi_length_meters set later by perspective ruler)
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

    # -- Perspective ruler --------------------------------------------------

    def enter_ruler_mode(self) -> None:
        """Transition to perspective ruler state."""
        self._ruler_points = []
        self._perspective_poly = None
        self.state = STATE_PERSPECTIVE_RULER
        print("\n  📐 PERSPECTIVE RULER")
        print("  Click the START of consecutive dashed road markings,")
        print("  from BOTTOM of image moving UP. Min 4 clicks.")
        print("  Each dash start = 12 m apart (3 m dash + 9 m gap).")
        print("  ENTER to confirm | RIGHT-CLICK to undo | ESC to cancel\n")

    def compute_perspective_lengths(self) -> bool:
        """Fit polynomial and set roi_length_meters on all ROIs.

        Returns True if successful, False if not enough points.
        """
        n = len(self._ruler_points)
        if n < 4:
            self._set_status(f"Need at least 4 points (have {n})", 3.0)
            return False

        # Extract Y-pixel coordinates (bottom-to-top order)
        y_points = np.array([p[1] for p in self._ruler_points], dtype=float)
        d_points = np.array([i * RULER_DASH_SPACING_M for i in range(n)], dtype=float)

        # Fit 2nd-degree polynomial: Y-pixel → physical meters
        z_coeffs = np.polyfit(y_points, d_points, 2)
        p = np.poly1d(z_coeffs)
        self._perspective_poly = p

        print(f"\n  📈 Polynomial fit: z = {z_coeffs[0]:.6f}·y² + {z_coeffs[1]:.4f}·y + {z_coeffs[2]:.2f}")

        # Calculate roi_length_meters for each ROI from its polygon Y-bounds
        for roi in self.rois:
            y_coords = [pt[1] for pt in roi["polygon"]]
            y_bottom = max(y_coords)  # max Y = bottom of image
            y_top = min(y_coords)     # min Y = top of image

            dist_bottom = p(y_bottom)
            dist_top = p(y_top)
            length_m = round(abs(dist_top - dist_bottom), 1)

            roi["roi_length_meters"] = max(length_m, 1.0)  # Floor at 1 m

            road_id = roi.get("road_id", "?")

            # Expert Audit Fix 4: Warn if ROI depth exceeds YOLOv8n effective range
            MAX_ROI_DEPTH_M = 150.0
            if length_m > MAX_ROI_DEPTH_M:
                print(
                    f"  ⚠️  WARNING: ROI '{road_id}' depth {length_m:.0f}m exceeds "
                    f"recommended max {MAX_ROI_DEPTH_M:.0f}m.\n"
                    f"      YOLOv8n detection degrades past ~150m — "
                    f"density will be underestimated."
                )
                self._set_status(
                    f"⚠️ ROI '{road_id}' depth {length_m:.0f}m > {MAX_ROI_DEPTH_M:.0f}m limit!",
                    5.0,
                )

            print(f"  📏 {road_id}: Y=[{y_top}..{y_bottom}]px → {length_m:.1f} m")

        self._set_status(
            f"Perspective calibration done — {len(self.rois)} ROI(s) measured",
            5.0,
        )
        return True

    # -- BEV calibration (Expert Audit Fix 2) ----------------------------------

    def enter_bev_mode(self) -> None:
        """Switch to BEV calibration: click 4 corners of a known rectangle."""
        self._bev_points: list[tuple[int, int]] = []
        self._bev_width_m: float = DEFAULT_BEV_WIDTH_M
        self._bev_length_m: float = DEFAULT_BEV_LENGTH_M
        self._homography_matrix: np.ndarray | None = None
        self.state = STATE_BEV_CALIBRATION
        print("\n  🔲 BEV CALIBRATION (Expert Audit Fix 2)")
        print("  Click exactly 4 corners of a known rectangle on the road.")
        print("  Order: top-left → top-right → bottom-right → bottom-left")
        print(f"  Default: {DEFAULT_BEV_WIDTH_M}m × {DEFAULT_BEV_LENGTH_M}m lane segment")
        print("  RIGHT-CLICK to undo | ENTER to confirm | ESC to cancel\n")

    def compute_bev_homography(self) -> bool:
        """Compute 3×3 homography from 4 pixel points → physical rectangle.

        Returns True if successful.
        """
        if len(self._bev_points) != 4:
            self._set_status(f"Need exactly 4 points (have {len(self._bev_points)})")
            return False

        src = np.array(self._bev_points, dtype=np.float32)
        # Destination: flat rectangle in meters
        w, l = self._bev_width_m, self._bev_length_m
        dst = np.array([[0, 0], [w, 0], [w, l], [0, l]], dtype=np.float32)

        H = cv2.getPerspectiveTransform(src, dst)
        self._homography_matrix = H

        print(f"\n  ✅ Homography computed ({self._bev_width_m}m × {self._bev_length_m}m):")
        for row in H:
            print(f"     [{row[0]:12.6f}, {row[1]:12.6f}, {row[2]:12.6f}]")

        # Also compute roi_length_meters for each ROI using the homography
        for roi in self.rois:
            y_coords = [pt[1] for pt in roi["polygon"]]
            x_coords = [pt[0] for pt in roi["polygon"]]
            cx = sum(x_coords) / len(x_coords)  # centroid X
            y_top = min(y_coords)
            y_bottom = max(y_coords)

            # Transform top and bottom points through homography
            pts_px = np.array([[[cx, y_top]], [[cx, y_bottom]]], dtype=np.float64)
            pts_bev = cv2.perspectiveTransform(pts_px, H)
            bev_top = pts_bev[0][0]
            bev_bottom = pts_bev[1][0]
            length_m = float(np.linalg.norm(bev_bottom - bev_top))
            roi["roi_length_meters"] = round(max(length_m, 1.0), 1)

            road_id = roi.get("road_id", "?")
            print(f"  📏 {road_id}: BEV length = {length_m:.1f} m")

            # Horizon hardcap (Fix 4)
            MAX_ROI_DEPTH_M = 150.0
            if length_m > MAX_ROI_DEPTH_M:
                print(
                    f"  ⚠️  WARNING: ROI '{road_id}' depth {length_m:.0f}m exceeds "
                    f"recommended max {MAX_ROI_DEPTH_M:.0f}m."
                )

        self._set_status("BEV calibration done — H saved", 5.0)
        return True

    def render(self) -> np.ndarray:
        """Draw current state on a canvas with HUD above the image."""
        # --- Build the image portion ---
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

            # In edit mode, draw vertex handles
            if self.state == STATE_EDITING:
                for vi, pt in enumerate(roi["polygon"]):
                    handle_color = color
                    radius = VERTEX_HANDLE_RADIUS
                    # Highlight hovered vertex
                    if (self._hover_roi_idx == i and self._hover_vert_idx == vi):
                        handle_color = (0, 255, 255)  # Yellow
                        radius = VERTEX_HANDLE_RADIUS + 3
                    cv2.circle(display, (pt[0], pt[1]), radius, handle_color, -1)
                    cv2.circle(display, (pt[0], pt[1]), radius, (0, 0, 0), 1)

                # Highlight hovered edge with insertion preview
                if (self._hover_roi_idx == i and self._hover_edge_idx >= 0
                        and self._hover_vert_idx < 0 and not self._dragging):
                    poly = roi["polygon"]
                    ei = self._hover_edge_idx
                    a = poly[ei]
                    b = poly[(ei + 1) % len(poly)]
                    cv2.line(display, tuple(a), tuple(b), (0, 255, 255), 3)
                    # Draw insertion preview dot
                    px, py = int(self._hover_proj[0]), int(self._hover_proj[1])
                    cv2.circle(display, (px, py), 7, (0, 255, 255), -1)
                    cv2.circle(display, (px, py), 7, (255, 255, 255), 2)

            # Label
            cx = int(np.mean(pts[:, 0]))
            cy = int(np.mean(pts[:, 1]))
            length_str = f' {roi["roi_length_meters"]:.0f}m' if "roi_length_meters" in roi else ""
            label = f'{roi["road_id"]} ({roi["num_lanes"]}L{length_str})'
            display = pil_puttext(display, label, (cx - 60, cy), color, 0.55, 2)

        # Draw exclusion zones as red dashed rectangles
        for idx, ez in enumerate(self.exclusion_zones):
            x1, y1, x2, y2 = ez
            # Dashed rectangle using line segments
            dash_len = 10
            red = (0, 0, 255)  # BGR
            for edge in [
                ((x1, y1), (x2, y1)),  # top
                ((x2, y1), (x2, y2)),  # right
                ((x2, y2), (x1, y2)),  # bottom
                ((x1, y2), (x1, y1)),  # left
            ]:
                pt_a, pt_b = edge
                dx = pt_b[0] - pt_a[0]
                dy = pt_b[1] - pt_a[1]
                length = math.hypot(dx, dy)
                if length == 0:
                    continue
                num_dashes = max(1, int(length / dash_len))
                for d in range(0, num_dashes, 2):
                    t0 = d / num_dashes
                    t1 = min((d + 1) / num_dashes, 1.0)
                    p0 = (int(pt_a[0] + dx * t0), int(pt_a[1] + dy * t0))
                    p1 = (int(pt_a[0] + dx * t1), int(pt_a[1] + dy * t1))
                    cv2.line(display, p0, p1, red, 2)
            # Semi-transparent red fill
            overlay = display.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), red, -1)
            cv2.addWeighted(overlay, 0.1, display, 0.9, 0, display)
            # Label
            display = pil_puttext(display, f"EXCL {idx+1}", (x1 + 4, y1 + 16), red, 0.45, 2)

        # Draw in-progress exclusion zone corner + live preview rectangle
        if self.state == STATE_EXCLUSION_ZONE and self._excl_corner1 is not None:
            c1x, c1y = self._excl_corner1
            mx, my = self.mouse_pos
            cv2.circle(display, (c1x, c1y), 6, (0, 0, 255), -1)
            cv2.rectangle(display, (c1x, c1y), (mx, my), (0, 0, 255), 1)

        # Draw current in-progress polygon (draw mode only)
        if self.current_points and self.state == STATE_DRAWING:
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

        # Draw perspective ruler points
        if self._ruler_points:
            for i, (rx, ry) in enumerate(self._ruler_points):
                # Magenta circle with index label
                cv2.circle(display, (rx, ry), 8, (255, 0, 255), -1)
                cv2.circle(display, (rx, ry), 8, (255, 255, 255), 2)
                cv2.putText(
                    display, str(i + 1), (rx + 12, ry + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2,
                )
                # Connect consecutive points
                if i > 0:
                    px, py = self._ruler_points[i - 1]
                    cv2.line(display, (px, py), (rx, ry), (255, 0, 255), 1)
                cv2.circle(display, tuple(pt), 5, (0, 0, 0), 1)

        # --- Build HUD above the image ---
        hud = np.zeros((HUD_HEIGHT, self.w, 3), dtype=np.uint8)

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
            excl_count = len(self.exclusion_zones)
            excl_info = f"  |  Excl. zones: {excl_count}" if excl_count else ""
            hud_lines.append((
                f"MODE: DRAW  |  Saved ROIs: {len(self.rois)}  |  Vertices: {len(self.current_points)}{excl_info}",
                (200, 200, 200), 0.5, 1,
            ))
            hud_lines.append((
                "CLICK: vertex | RIGHT-CLICK: undo | ENTER: finish polygon",
                (150, 200, 255), 0.45, 1,
            ))
            hud_lines.append((
                "E: edit | X: excl. zone | Z: del excl. | D+D: delete all | S: skip | Q/ESC: quit",
                (150, 200, 255), 0.45, 1,
            ))
            hud_lines.append((
                "W: save & calibrate (perspective ruler)",
                (255, 200, 100), 0.5, 1,
            ))

        elif self.state == STATE_EDITING:
            hud_lines.append((
                f"MODE: EDIT  |  ROIs: {len(self.rois)}  |  "
                + ("DRAGGING vertex" if self._dragging else "Hover to select"),
                (255, 200, 100), 0.5, 1,
            ))
            hud_lines.append((
                "DRAG: move vertex | CLICK edge: insert vertex | RIGHT-CLICK: delete vertex",
                (150, 200, 255), 0.45, 1,
            ))
            hud_lines.append((
                "E: draw mode | S: skip | Q/ESC: quit & save",
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

        elif self.state == STATE_EXCLUSION_ZONE:
            corner_status = "Click TOP-LEFT corner" if self._excl_corner1 is None else "Click BOTTOM-RIGHT corner"
            hud_lines.append((
                f"EXCLUSION ZONE — {corner_status}",
                (0, 100, 255), 0.6, 2,
            ))
            hud_lines.append((
                "Draw a rectangle over fixed objects (lamps, signs) that cause false detections",
                (200, 200, 200), 0.45, 1,
            ))
            hud_lines.append((
                "RIGHT-CLICK: undo corner | ESC: cancel",
                (150, 200, 255), 0.45, 1,
            ))

        elif self.state == STATE_PERSPECTIVE_RULER:
            n = len(self._ruler_points)
            hud_lines.append((
                f"PERSPECTIVE RULER — {n} point(s) clicked",
                (255, 100, 255), 0.6, 2,
            ))
            hud_lines.append((
                "LEFT-CLICK: mark start of next road dash (bottom → top)",
                (255, 255, 100), 0.5, 1,
            ))
            hud_lines.append((
                "Each click = 12 m further (3 m dash + 9 m gap)",
                (100, 200, 255), 0.45, 1,
            ))
            status = f"Need {max(0, 4 - n)} more" if n < 4 else "✅ Ready"
            hud_lines.append((
                f"RIGHT-CLICK: undo | ENTER: confirm ({status}) | ESC: cancel",
                (150, 200, 255), 0.45, 1,
            ))

        # Status message (temporary)
        if self._status_msg and time.time() < self._status_until:
            hud_lines.append((self._status_msg, (100, 255, 255), 0.5, 1))

        # Mouse position
        hud_lines.append((
            f"Mouse: ({self.mouse_pos[0]}, {self.mouse_pos[1]})",
            (180, 180, 180), 0.4, 1,
        ))

        # Render HUD text
        hud_y = 22
        for text, color, scale, thickness in hud_lines:
            hud = pil_puttext(hud, text, (10, hud_y), color, scale, thickness)
            hud_y += 22

        # --- Combine: HUD on top, image below ---
        canvas = np.vstack([hud, display])
        return canvas


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

    # Load existing ROIs and exclusion zones
    cam_config = config.get("cameras", {}).get(camera_id, {})
    existing_rois = cam_config.get("rois", [])
    existing_excl = cam_config.get("exclusion_zones", [])

    if existing_rois:
        print(f"  ℹ️  {len(existing_rois)} existing ROI(s) loaded")
    if existing_excl:
        print(f"  🚫 {len(existing_excl)} existing exclusion zone(s) loaded")

    drawer = PolygonDrawer(frame, camera_id, existing_rois, meta=meta,
                           existing_exclusion_zones=existing_excl)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    # Expand window height to accommodate HUD above the image
    win_w = min(frame.shape[1], 1400)
    win_h = min(frame.shape[0] + HUD_HEIGHT, 1060)
    cv2.resizeWindow(WINDOW_NAME, win_w, win_h)
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

        # ── Perspective ruler state ──
        if drawer.state == STATE_PERSPECTIVE_RULER:
            if key == 13 or key == 32:  # ENTER/SPACE → compute
                if drawer.compute_perspective_lengths():
                    # Save and exit
                    break
            elif key == 27:  # ESC → cancel ruler, back to drawing
                drawer.state = STATE_DRAWING
                drawer._ruler_points = []
                drawer._set_status("Ruler cancelled")
            continue

        # ── BEV calibration state (Expert Audit Fix 2) ──
        if drawer.state == STATE_BEV_CALIBRATION:
            if key == 13 or key == 32:  # ENTER/SPACE → compute
                if drawer.compute_bev_homography():
                    break
            elif key == 27:  # ESC → cancel BEV, back to drawing
                drawer.state = STATE_DRAWING
                drawer._bev_points = []
                drawer._set_status("BEV calibration cancelled")
            continue

        # ── Exclusion zone state ──
        if drawer.state == STATE_EXCLUSION_ZONE:
            if key == 27:  # ESC → cancel, back to drawing
                drawer.state = STATE_DRAWING
                drawer._excl_corner1 = None
                drawer._set_status("Exclusion zone cancelled")
            continue

        # ── Common keys for both draw and edit modes ──
        if key == 27 or key == ord("q"):  # ESC or Q → quit
            quit_all = True
            break

        elif key == ord("s"):  # Skip camera
            print("  ⏭  Skipped")
            break

        elif key == ord("b"):  # BEV calibration (Expert Audit Fix 2)
            if not drawer.rois:
                drawer._set_status("No ROIs — draw at least one first")
            else:
                drawer.enter_bev_mode()

        elif key == ord("w"):  # Save & calibrate (legacy perspective ruler)
            if not drawer.rois:
                drawer._set_status("No ROIs to calibrate — draw at least one first")
            else:
                drawer.enter_ruler_mode()

        elif key == ord("x"):  # Enter exclusion zone drawing mode
            drawer.state = STATE_EXCLUSION_ZONE
            drawer._excl_corner1 = None
            drawer._set_status("Exclusion zone mode — click top-left corner")
            print("  🚫 Exclusion zone mode")

        elif key == ord("e"):  # Toggle edit/draw mode
            if drawer.state == STATE_DRAWING:
                if drawer.rois:
                    drawer.state = STATE_EDITING
                    drawer._set_status("Edit mode — drag vertices or click edges to insert")
                    print("  ✏️  Edit mode")
                else:
                    drawer._set_status("No ROIs to edit — draw one first")
            elif drawer.state == STATE_EDITING:
                drawer.state = STATE_DRAWING
                drawer._dragging = False
                drawer._set_status("Draw mode")
                print("  ✏️  Draw mode")

        # ── Draw mode specific keys ──
        elif drawer.state == STATE_DRAWING:
            if key == 13 or key == 32:  # ENTER or SPACE → finish polygon
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

            elif key == ord("z"):  # Delete last exclusion zone
                if drawer.exclusion_zones:
                    removed = drawer.exclusion_zones.pop()
                    drawer._set_status(
                        f"Removed exclusion zone {removed} ({len(drawer.exclusion_zones)} remaining)"
                    )
                    print(f"  🗑  Removed exclusion zone: {removed}")
                else:
                    drawer._set_status("No exclusion zones to remove")

    cv2.destroyAllWindows()

    # Save ROIs, exclusion zones, and homography to config
    has_data = drawer.rois or drawer.exclusion_zones
    if has_data:
        if "cameras" not in config:
            config["cameras"] = {}
        cam_data: dict = {"rois": drawer.rois}
        if drawer.exclusion_zones:
            cam_data["exclusion_zones"] = drawer.exclusion_zones
        # Expert Audit Fix 2: save homography matrix
        if hasattr(drawer, '_homography_matrix') and drawer._homography_matrix is not None:
            cam_data["homography_matrix"] = drawer._homography_matrix.tolist()
            cam_data["bev_rect_width_m"] = drawer._bev_width_m
            cam_data["bev_rect_length_m"] = drawer._bev_length_m
        config["cameras"][camera_id] = cam_data
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
