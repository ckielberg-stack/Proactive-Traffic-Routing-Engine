# Multi-Page Operator Dashboard — Design

## Context

The PTRE dashboard is a single-page TMC view (incidents, VMS, prophecy log) that provides no visibility into system operations — camera collection status, sensor data flow, logs, or tick health. The developer needs full operational awareness.

## Decisions Made

- **Primary user:** Developer/operator (not traffic management operators)
- **TMC page:** Keep as one page among siblings
- **Tech:** Jinja2 server-side templates (no JS framework, no build step)
- **Style:** Same dark control-room aesthetic, JetBrains Mono + Inter fonts

## Architecture

### Template Structure

```
templates/
  base.html          — Shared layout: head, nav bar, footer, CSS/JS imports
  tmc.html            — Current 3-column view (incidents, VMS, prophecy)
  cameras.html        — Camera grid with per-camera status
  sensors.html        — Sensor station readings table
  logs.html           — Terminal tail of mainloop.log
  system.html         — Tick stats, uptime, system health
static/
  app.js              — Keep existing TMC polling logic
  cameras.js          — Camera page polling
  sensors.js          — Sensor page polling
  logs.js             — Log page polling
  system.js           — System page polling
  style.css           — Shared styles (extracted from inline)
```

### Navigation Bar

Added to `base.html` below the header. 5 items:

| Label | Route | Description |
|-------|-------|-------------|
| TMC | `/` | Current operator view |
| Cameras | `/cameras` | Per-camera status grid |
| Sensors | `/sensors` | Sensor station data |
| Log | `/logs` | System log terminal |
| System | `/system` | Tick health & stats |

Active page highlighted with accent color.

### Backend Endpoints (new)

| Endpoint | Data source | Returns |
|----------|-------------|---------|
| `GET /api/v1/cameras` | `vision_state.json` + config | Per-camera: name, coords, status, vehicle count, VPH, anomaly flag |
| `GET /api/v1/sensors` | In-memory tick result | Per-station: site_id, volume, speed, mapped camera |
| `GET /api/v1/logs` | `data/mainloop.log` | Last N lines, color-coded by level |
| `GET /api/v1/status` | `data/status.json` | Tick count, uptime, last tick stats |

### Page Designs

**Cameras** — Grid of cards. Each card shows camera name, status badge (OK/FAIL), vehicle count, VPH, anomaly indicator. Cards sorted by chainage (north→south).

**Sensors** — Table with columns: Station ID, Volume (VPH), Speed (km/h), Mapped Camera. Sortable. Summary row at top with aggregates.

**Log** — Full-width terminal. Auto-scrolling. Lines color-coded: green=INFO, yellow=WARN, red=ERROR. Monospace font.

**System** — Stat cards: Tick #, Uptime, Cameras OK/Failed, Total Vehicles, Anomalies, Sensors, Predictions, VMS Recs. Auto-refreshing.
