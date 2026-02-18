/**
 * PTRE — Interactive Corridor Map
 *
 * Leaflet.js map showing:
 *   - Camera markers with direction arrows
 *   - ROI section labels
 *   - VMS gantry markers
 *   - Vehicle count + VPH in popups
 *   - E4 corridor polyline
 *   - Auto-refresh every 10s
 */

const MAP_POLL_INTERVAL = 10_000;

// ─── State ───────────────────────────────────────────────────────────
let map;
let cameraMarkers = {};   // camera_id → { marker, arrow }
let vmsMarkers = [];
let corridorLine = null;
let roiConfig = null;

// VMS gantry positions (loaded from vms_config.json via inline)
const VMS_GANTRIES = [
    { id: "VMS-4001", name: "Hallunda södra", lat: 59.2420, lng: 17.8370 },
    { id: "VMS-4002", name: "Fittja", lat: 59.2540, lng: 17.8610 },
    { id: "VMS-4003", name: "Kungens Kurva", lat: 59.2720, lng: 17.9140 },
    { id: "VMS-4004", name: "Västberga", lat: 59.2960, lng: 18.0040 },
    { id: "VMS-4005", name: "Nyboda", lat: 59.3010, lng: 18.0200 },
    { id: "VMS-4006", name: "Gröndal", lat: 59.3150, lng: 18.0033 },
    { id: "VMS-4007", name: "Essingen", lat: 59.3210, lng: 17.9970 },
    { id: "VMS-4008", name: "Kristineberg", lat: 59.3340, lng: 18.0100 },
];

// ─── Helpers ─────────────────────────────────────────────────────────

async function fetchJSON(path) {
    const res = await fetch(path);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return res.json();
}

function cameraShortName(cam) {
    if (cam.name && cam.name !== cam.camera_id) return cam.name;
    const parts = cam.camera_id.split('_');
    return parts[parts.length - 1];
}

/**
 * Compute bearing (degrees) from point A to point B.
 */
function bearing(lat1, lng1, lat2, lng2) {
    const toRad = d => d * Math.PI / 180;
    const toDeg = r => r * 180 / Math.PI;
    const dLng = toRad(lng2 - lng1);
    const y = Math.sin(dLng) * Math.cos(toRad(lat2));
    const x = Math.cos(toRad(lat1)) * Math.sin(toRad(lat2)) -
        Math.sin(toRad(lat1)) * Math.cos(toRad(lat2)) * Math.cos(dLng);
    return (toDeg(Math.atan2(y, x)) + 360) % 360;
}

/**
 * Create arrow polyline from a camera position + bearing.
 * Returns a Leaflet polyline (small wedge shape).
 */
function createArrow(lat, lng, bearingDeg, color) {
    const len = 0.0012;   // arrow length in degrees (~100m)
    const spread = 25;    // half-angle of wedge
    const toRad = d => d * Math.PI / 180;

    // Tip of arrow
    const tipLat = lat + len * Math.cos(toRad(bearingDeg));
    const tipLng = lng + len * Math.sin(toRad(bearingDeg)) / Math.cos(toRad(lat));

    // Left wing
    const leftAngle = bearingDeg + 180 - spread;
    const leftLat = tipLat + (len * 0.45) * Math.cos(toRad(leftAngle));
    const leftLng = tipLng + (len * 0.45) * Math.sin(toRad(leftAngle)) / Math.cos(toRad(lat));

    // Right wing
    const rightAngle = bearingDeg + 180 + spread;
    const rightLat = tipLat + (len * 0.45) * Math.cos(toRad(rightAngle));
    const rightLng = tipLng + (len * 0.45) * Math.sin(toRad(rightAngle)) / Math.cos(toRad(lat));

    return L.polygon([
        [tipLat, tipLng],
        [leftLat, leftLng],
        [lat, lng],
        [rightLat, rightLng],
    ], {
        color: color,
        fillColor: color,
        fillOpacity: 0.7,
        weight: 1,
        interactive: false,
    });
}

// ─── Map Initialization ──────────────────────────────────────────────

function initMap() {
    map = L.map('map', {
        center: [59.290, 17.960],
        zoom: 13,
        zoomControl: true,
    });

    // Dark-themed tile layer (CartoDB Dark Matter)
    L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
        attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OSM</a> &copy; <a href="https://carto.com/">CARTO</a>',
        subdomains: 'abcd',
        maxZoom: 19,
    }).addTo(map);

    // Add VMS gantry markers
    addVMSMarkers();
}

// ─── VMS Gantry Markers ──────────────────────────────────────────────

function addVMSMarkers() {
    const vmsIcon = L.divIcon({
        className: '',
        html: `<div style="
            width:14px; height:14px;
            background:#f59e0b;
            transform: rotate(45deg);
            border: 1px solid rgba(0,0,0,0.3);
            box-shadow: 0 0 6px rgba(245,158,11,0.5);
        "></div>`,
        iconSize: [14, 14],
        iconAnchor: [7, 7],
    });

    VMS_GANTRIES.forEach(g => {
        const marker = L.marker([g.lat, g.lng], { icon: vmsIcon })
            .addTo(map)
            .bindPopup(`
                <div class="map-popup-title">⬦ ${g.name}</div>
                <div class="map-popup-row">
                    <span class="map-popup-label">VMS ID</span>
                    <span class="map-popup-value">${g.id}</span>
                </div>
                <div class="map-popup-row">
                    <span class="map-popup-label">Type</span>
                    <span class="map-popup-value">Variable Message Sign</span>
                </div>
            `);
        vmsMarkers.push(marker);
    });
}

// ─── Camera Markers ──────────────────────────────────────────────────

function buildPopupContent(cam, rois) {
    const name = cameraShortName(cam);
    const statusClass = cam.is_anomaly ? 'anomaly' : 'ok';
    const statusText = cam.is_anomaly ? '⚠ ANOMALY' : '● OK';

    let roiHtml = '';
    if (rois && rois.length > 0) {
        roiHtml = '<div class="map-popup-divider"></div>';
        rois.forEach(roi => {
            const dir = roi.direction_relative_to_camera === 'towards' ? '↓ towards' : '↑ away';
            const lanes = roi.num_lanes || '?';
            const cap = roi.capacity_vph ? `${(roi.capacity_vph / 1000).toFixed(0)}k` : '—';
            roiHtml += `
                <div class="map-popup-roi">
                    <span class="map-popup-roi-name">${roi.road_id || 'ROI'}</span>
                    — ${dir} · ${lanes}L · ${cap} VPH
                </div>`;
        });
    }

    return `
        <div class="map-popup-title">📷 ${name}</div>
        <div class="map-popup-row">
            <span class="map-popup-label">Status</span>
            <span class="map-popup-value ${statusClass}">${statusText}</span>
        </div>
        <div class="map-popup-row">
            <span class="map-popup-label">Vehicles</span>
            <span class="map-popup-value">${cam.vehicle_count ?? '—'}</span>
        </div>
        <div class="map-popup-row">
            <span class="map-popup-label">VPH</span>
            <span class="map-popup-value">${cam.estimated_vph?.toLocaleString() ?? '—'}</span>
        </div>
        ${cam.capacity_drop > 0 ? `
        <div class="map-popup-row">
            <span class="map-popup-label">Cap Drop</span>
            <span class="map-popup-value anomaly">${cam.capacity_drop.toFixed(0)}%</span>
        </div>` : ''}
        <div class="map-popup-row">
            <span class="map-popup-label">Camera ID</span>
            <span class="map-popup-value" style="font-size:0.5rem;color:var(--tmc-muted);">${cam.camera_id}</span>
        </div>
        ${roiHtml}
    `;
}

function updateCameraMarkers(cameras) {
    // Sort cameras by latitude (south → north) for bearing computation
    const sorted = [...cameras].filter(c => c.lat && c.lng).sort((a, b) => a.lat - b.lat);

    // Compute bearings from each camera to the next (corridor direction)
    const bearings = {};
    for (let i = 0; i < sorted.length; i++) {
        const cam = sorted[i];
        let next = sorted[i + 1];
        let prev = sorted[i - 1];

        if (next) {
            bearings[cam.camera_id] = bearing(cam.lat, cam.lng, next.lat, next.lng);
        } else if (prev) {
            bearings[cam.camera_id] = bearing(prev.lat, prev.lng, cam.lat, cam.lng);
        }
    }

    // Get ROI data for each camera
    const roiData = roiConfig?.cameras || {};

    // Update or create markers
    sorted.forEach(cam => {
        const rois = roiData[cam.camera_id]?.rois || [];
        const isAnomaly = cam.is_anomaly;
        const color = isAnomaly ? '#ef4444' : '#22c55e';
        const popupContent = buildPopupContent(cam, rois);

        if (cameraMarkers[cam.camera_id]) {
            // Update existing marker
            const entry = cameraMarkers[cam.camera_id];
            entry.marker.setPopupContent(popupContent);
            entry.marker.setStyle({
                color: color,
                fillColor: color,
            });
            // Update arrow color
            if (entry.arrow) {
                entry.arrow.setStyle({ color: color, fillColor: color });
            }
        } else {
            // Create new marker
            const marker = L.circleMarker([cam.lat, cam.lng], {
                radius: 7,
                color: color,
                fillColor: color,
                fillOpacity: 0.8,
                weight: 2,
            }).addTo(map).bindPopup(popupContent);

            // Create direction arrow
            let arrow = null;
            if (bearings[cam.camera_id] !== undefined) {
                arrow = createArrow(cam.lat, cam.lng, bearings[cam.camera_id], color);
                arrow.addTo(map);
            }

            cameraMarkers[cam.camera_id] = { marker, arrow };
        }
    });

    // Draw corridor polyline (connecting all cameras south→north)
    if (!corridorLine && sorted.length > 1) {
        const latlngs = sorted.map(c => [c.lat, c.lng]);
        corridorLine = L.polyline(latlngs, {
            color: '#3b82f6',
            weight: 2,
            opacity: 0.4,
            dashArray: '8, 6',
            interactive: false,
        }).addTo(map);

        // Send corridor line to back
        corridorLine.bringToBack();
    }
}

// ─── Metrics ─────────────────────────────────────────────────────────

function updateMetrics(data) {
    const cameras = data.cameras || [];
    const anomalies = cameras.filter(c => c.is_anomaly).length;
    const totalVehicles = cameras.reduce((s, c) => s + (c.vehicle_count || 0), 0);

    const el = (id) => document.getElementById(id);
    el('map-cam-total').textContent = cameras.length;
    el('map-anomalies').textContent = anomalies;
    el('map-vehicles').textContent = totalVehicles.toLocaleString();
    if (data.timestamp) {
        const t = new Date(data.timestamp);
        el('map-updated').textContent = t.toLocaleTimeString('sv-SE');
    }
}

// ─── Polling ─────────────────────────────────────────────────────────

async function loadROIConfig() {
    if (roiConfig) return;
    try {
        roiConfig = await fetchJSON('/api/v1/camera-config');
    } catch (e) {
        console.warn('Failed to load ROI config:', e);
        roiConfig = { cameras: {} };
    }
}

async function pollMap() {
    try {
        await loadROIConfig();
        const data = await fetchJSON('/api/v1/cameras');
        updateCameraMarkers(data.cameras || []);
        updateMetrics(data);
    } catch (e) {
        console.error('Map poll error:', e);
    }
}

// ─── Init ────────────────────────────────────────────────────────────

initMap();
pollMap();
setInterval(pollMap, MAP_POLL_INTERVAL);
console.log('🗺️ Corridor map initialized — polling every 10s');
