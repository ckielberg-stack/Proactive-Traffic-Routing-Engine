/**
 * PTRE — Camera Grid Page
 * Polls /api/v1/cameras every 10s, renders camera card grid.
 * Click a card to open the live image with ROI polygon overlay.
 */

const POLL_INTERVAL = 10_000;

// ROI polygon config cache (loaded once)
let _roiConfig = null;

// Detection state
let _lastDetections = null;
let _showDetections = true;
let _currentDrawState = null;  // {canvas, img, cameraId, roiConfig}

const ROI_COLORS = [
    { fill: 'rgba(34, 197, 94, 0.25)', stroke: '#22c55e', label: '#22c55e' },  // Green
    { fill: 'rgba(59, 130, 246, 0.25)', stroke: '#3b82f6', label: '#3b82f6' },  // Blue
    { fill: 'rgba(245, 158, 11, 0.25)', stroke: '#f59e0b', label: '#f59e0b' },  // Amber
    { fill: 'rgba(168, 85, 247, 0.25)', stroke: '#a855f7', label: '#a855f7' },  // Purple
    { fill: 'rgba(236, 72, 153, 0.25)', stroke: '#ec4899', label: '#ec4899' },  // Pink
    { fill: 'rgba(14, 165, 233, 0.25)', stroke: '#0ea5e9', label: '#0ea5e9' },  // Sky
];

// Native image resolution used in camera_config.json
const NATIVE_W = 1280;
const NATIVE_H = 720;

async function fetchJSON(path) {
    const res = await fetch(path);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return res.json();
}

function cameraName(cam) {
    if (cam.name && cam.name !== cam.camera_id) return cam.name;
    const parts = cam.camera_id.split('_');
    return parts[parts.length - 1];
}

async function loadROIConfig() {
    if (_roiConfig) return _roiConfig;
    try {
        _roiConfig = await fetchJSON('/api/v1/camera-config');
    } catch (e) {
        console.warn('Failed to load ROI config:', e);
        _roiConfig = { cameras: {} };
    }
    return _roiConfig;
}

// ─── Camera Grid Rendering ───────────────────────────────────────────

function renderCameras(data) {
    const grid = document.getElementById('camera-grid');
    const cameras = data.cameras || [];

    // Update metrics
    const ok = cameras.filter(c => c.status === 'ok').length;
    const failed = cameras.filter(c => c.status !== 'ok').length;
    const anomalies = cameras.filter(c => c.is_anomaly).length;
    const totalVehicles = cameras.reduce((s, c) => s + (c.vehicle_count || 0), 0);

    document.getElementById('cam-total').textContent = cameras.length;
    document.getElementById('cam-ok').textContent = ok;
    document.getElementById('cam-failed').textContent = failed;
    document.getElementById('cam-anomalies').textContent = anomalies;
    document.getElementById('cam-vehicles').textContent = totalVehicles;
    if (data.timestamp) {
        const t = new Date(data.timestamp);
        document.getElementById('cam-updated').textContent = t.toLocaleTimeString('sv-SE');
    }

    if (cameras.length === 0) {
        grid.innerHTML = '<div class="empty-state" style="grid-column:1/-1;">No camera data available yet</div>';
        return;
    }

    grid.innerHTML = cameras.map(cam => {
        const isAnomaly = cam.is_anomaly;
        const statusBadge = cam.status === 'ok'
            ? '<span class="badge badge-ok">OK</span>'
            : '<span class="badge badge-fail">FAIL</span>';
        const anomalyBadge = isAnomaly
            ? ' <span class="badge badge-warn">ANOMALY</span>'
            : '';
        const capDrop = cam.capacity_drop != null
            ? `<span class="camera-stat-label">Cap Drop</span><span class="camera-stat-value" style="color:${cam.capacity_drop > 30 ? 'var(--urgency-immediate)' : 'var(--tmc-text)'}">${cam.capacity_drop.toFixed(0)}%</span>`
            : '';

        return `
            <div class="camera-card clickable ${isAnomaly ? 'anomaly' : ''} fade-in"
                 data-camera-id="${cam.camera_id}"
                 title="Click to view live image">
                <div class="camera-name">
                    ${statusBadge}${anomalyBadge}
                    <span style="margin-left: 0.25rem;">${cameraName(cam)}</span>
                    <span class="cam-view-icon">🔍</span>
                </div>
                <div class="camera-stats">
                    <span class="camera-stat-label">Vehicles</span>
                    <span class="camera-stat-value">${cam.vehicle_count ?? '—'}</span>
                    <span class="camera-stat-label">VPH</span>
                    <span class="camera-stat-value">${cam.estimated_vph?.toLocaleString() ?? '—'}</span>
                    ${capDrop}
                    <span class="camera-stat-label">Lat</span>
                    <span class="camera-stat-value">${cam.lat?.toFixed(4) ?? '—'}</span>
                </div>
            </div>`;
    }).join('');

    // Attach click handlers
    grid.querySelectorAll('.camera-card[data-camera-id]').forEach(card => {
        card.addEventListener('click', () => openCameraViewer(card.dataset.cameraId));
    });
}

// ─── Modal Viewer ────────────────────────────────────────────────────

function openCameraViewer(cameraId) {
    const modal = document.getElementById('cam-modal');
    const canvas = document.getElementById('cam-modal-canvas');
    const loading = document.getElementById('cam-modal-loading');
    const title = document.getElementById('cam-modal-title');
    const footer = document.getElementById('cam-modal-footer');

    // Show modal with loading
    modal.style.display = 'flex';
    canvas.style.display = 'none';
    loading.style.display = 'flex';
    footer.textContent = '';
    _lastDetections = null;
    _currentDrawState = null;

    // Set title from camera name
    const shortName = cameraId.split('_').pop();
    title.textContent = shortName;

    // Ensure toggle button exists
    let toggleBtn = document.getElementById('cam-detect-toggle');
    if (!toggleBtn) {
        toggleBtn = document.createElement('button');
        toggleBtn.id = 'cam-detect-toggle';
        toggleBtn.style.cssText =
            'position:absolute;top:12px;right:50px;z-index:20;' +
            'background:rgba(0,0,0,0.7);color:#22c55e;border:1px solid #22c55e;' +
            'padding:4px 10px;border-radius:4px;cursor:pointer;font-size:12px;' +
            'font-family:monospace;';
        toggleBtn.addEventListener('click', () => {
            _showDetections = !_showDetections;
            toggleBtn.textContent = _showDetections ? '🟢 YOLO ON' : '⚫ YOLO OFF';
            toggleBtn.style.color = _showDetections ? '#22c55e' : '#666';
            toggleBtn.style.borderColor = _showDetections ? '#22c55e' : '#666';
            if (_currentDrawState) {
                const { canvas, img, cameraId, roiConfig } = _currentDrawState;
                drawImageWithPolygons(canvas, img, cameraId, roiConfig);
            }
        });
        canvas.parentElement.appendChild(toggleBtn);
    }
    toggleBtn.textContent = _showDetections ? '🟢 YOLO ON' : '⚫ YOLO OFF';
    toggleBtn.style.display = 'block';

    // Load image + ROI config + detections in parallel
    const imgPromise = new Promise((resolve, reject) => {
        const img = new Image();
        img.crossOrigin = 'anonymous';
        img.onload = () => resolve(img);
        img.onerror = () => reject(new Error('Failed to load camera image'));
        img.src = `/api/v1/camera-image/${encodeURIComponent(cameraId)}?t=${Date.now()}`;
    });

    const roiPromise = loadROIConfig();

    const detectPromise = fetchJSON(`/api/v1/camera-detections/${encodeURIComponent(cameraId)}`)
        .catch(e => { console.warn('Detection fetch failed:', e); return null; });

    Promise.all([imgPromise, roiPromise, detectPromise])
        .then(([img, roiConfig, detData]) => {
            loading.style.display = 'none';
            canvas.style.display = 'block';
            _lastDetections = detData;
            _currentDrawState = { canvas, img, cameraId, roiConfig };
            drawImageWithPolygons(canvas, img, cameraId, roiConfig);
        })
        .catch(err => {
            loading.innerHTML = `<span style="color: var(--status-failed);">⚠ ${err.message}</span>`;
            console.error('Camera viewer error:', err);
        });
}

function closeCameraViewer() {
    document.getElementById('cam-modal').style.display = 'none';
    const toggleBtn = document.getElementById('cam-detect-toggle');
    if (toggleBtn) toggleBtn.style.display = 'none';
}

function drawImageWithPolygons(canvas, img, cameraId, roiConfig) {
    const ctx = canvas.getContext('2d');

    // Size the canvas to fit viewport while maintaining aspect ratio
    const maxW = window.innerWidth * 0.9;
    const maxH = window.innerHeight * 0.8;
    const scale = Math.min(maxW / img.naturalWidth, maxH / img.naturalHeight, 1);
    const drawW = Math.round(img.naturalWidth * scale);
    const drawH = Math.round(img.naturalHeight * scale);

    canvas.width = drawW;
    canvas.height = drawH;

    // Draw the image
    ctx.drawImage(img, 0, 0, drawW, drawH);

    // Scale factor from native 1280×720 to canvas size
    const scaleX = drawW / NATIVE_W;
    const scaleY = drawH / NATIVE_H;

    // Draw ROI polygons
    const camConfig = roiConfig?.cameras?.[cameraId];
    if (!camConfig || !camConfig.rois) {
        // No ROIs — show footer note
        document.getElementById('cam-modal-footer').textContent = 'No ROI polygons configured for this camera';
        return;
    }

    const rois = camConfig.rois;
    const legendParts = [];

    rois.forEach((roi, idx) => {
        const color = ROI_COLORS[idx % ROI_COLORS.length];
        const points = roi.polygon;
        if (!points || points.length < 3) return;

        // Draw filled polygon
        ctx.beginPath();
        ctx.moveTo(points[0][0] * scaleX, points[0][1] * scaleY);
        for (let i = 1; i < points.length; i++) {
            ctx.lineTo(points[i][0] * scaleX, points[i][1] * scaleY);
        }
        ctx.closePath();
        ctx.fillStyle = color.fill;
        ctx.fill();

        // Draw polygon outline
        ctx.strokeStyle = color.stroke;
        ctx.lineWidth = 2;
        ctx.stroke();

        // Calculate centroid for label
        let cx = 0, cy = 0;
        points.forEach(p => { cx += p[0]; cy += p[1]; });
        cx = (cx / points.length) * scaleX;
        cy = (cy / points.length) * scaleY;

        // Draw label background
        const label = roi.road_id || `ROI ${idx + 1}`;
        const dir = roi.direction_relative_to_camera === 'towards' ? '↓' : '↑';
        const labelText = `${label} ${dir}`;
        const fontSize = Math.max(11, Math.round(13 * scale));
        ctx.font = `bold ${fontSize}px 'JetBrains Mono', monospace`;
        const metrics = ctx.measureText(labelText);
        const pad = 4;
        const lw = metrics.width + pad * 2;
        const lh = fontSize + pad * 2;

        ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
        ctx.fillRect(cx - lw / 2, cy - lh / 2, lw, lh);

        // Draw label text
        ctx.fillStyle = color.label;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(labelText, cx, cy);

        // Build legend
        const lanes = roi.num_lanes || '?';
        const cap = roi.capacity_vph ? `${(roi.capacity_vph / 1000).toFixed(0)}k VPH` : '';
        legendParts.push(`${label} (${dir} ${lanes}L ${cap})`);
    });

    // Draw YOLO detection boxes (Expert Audit — visual verification)
    let detectCount = 0;
    if (_showDetections && _lastDetections && _lastDetections.detections) {
        const dets = _lastDetections.detections;
        const imgW = _lastDetections.image_width || NATIVE_W;
        const imgH = _lastDetections.image_height || NATIVE_H;
        const dScaleX = drawW / imgW;
        const dScaleY = drawH / imgH;
        detectCount = dets.length;

        const classColors = {
            car: '#00ff88',
            truck: '#ff6644',
            bus: '#ffaa00',
            motorcycle: '#aa66ff',
        };

        dets.forEach(det => {
            const [x1, y1, x2, y2] = det.xyxy;
            const dx = x1 * dScaleX;
            const dy = y1 * dScaleY;
            const dw = (x2 - x1) * dScaleX;
            const dh = (y2 - y1) * dScaleY;

            const color = classColors[det.class_name] || '#00ff88';

            // Box
            ctx.strokeStyle = color;
            ctx.lineWidth = 2;
            ctx.strokeRect(dx, dy, dw, dh);

            // Label background
            const label = `${det.class_name} ${(det.confidence * 100).toFixed(0)}%`;
            const fontSize = Math.max(10, Math.round(11 * scale));
            ctx.font = `bold ${fontSize}px monospace`;
            const tw = ctx.measureText(label).width + 6;
            ctx.fillStyle = 'rgba(0,0,0,0.75)';
            ctx.fillRect(dx, dy - fontSize - 4, tw, fontSize + 4);

            // Label text
            ctx.fillStyle = color;
            ctx.textAlign = 'left';
            ctx.textBaseline = 'top';
            ctx.fillText(label, dx + 3, dy - fontSize - 2);
        });
    }

    // Draw exclusion zones as red dashed rectangles
    if (_showDetections && _lastDetections && _lastDetections.exclusion_zones) {
        const imgW = _lastDetections.image_width || NATIVE_W;
        const imgH = _lastDetections.image_height || NATIVE_H;
        const eScaleX = drawW / imgW;
        const eScaleY = drawH / imgH;

        ctx.save();
        ctx.setLineDash([6, 4]);
        _lastDetections.exclusion_zones.forEach(ez => {
            const [x1, y1, x2, y2] = ez;
            ctx.strokeStyle = '#ff4444';
            ctx.lineWidth = 2;
            ctx.strokeRect(x1 * eScaleX, y1 * eScaleY,
                (x2 - x1) * eScaleX, (y2 - y1) * eScaleY);

            // Small "EXCL" label
            const fontSize = Math.max(9, Math.round(10 * scale));
            ctx.font = `bold ${fontSize}px monospace`;
            ctx.fillStyle = 'rgba(255,68,68,0.7)';
            ctx.textAlign = 'left';
            ctx.textBaseline = 'top';
            ctx.fillText('EXCL', x1 * eScaleX + 2, y1 * eScaleY + 2);
        });
        ctx.restore();
    }

    // Footer legend
    const excludedInfo = _showDetections && _lastDetections && _lastDetections.excluded_count > 0
        ? ` (${_lastDetections.excluded_count} excluded)`
        : '';
    const detectInfo = _showDetections && detectCount > 0
        ? `  │  🔍 ${detectCount} vehicles detected${excludedInfo}`
        : '';
    document.getElementById('cam-modal-footer').textContent =
        legendParts.join('  •  ') + detectInfo;
}

// ─── Event Listeners ─────────────────────────────────────────────────

document.getElementById('cam-modal-close').addEventListener('click', closeCameraViewer);
document.getElementById('cam-modal').addEventListener('click', (e) => {
    if (e.target.classList.contains('cam-modal-backdrop')) closeCameraViewer();
});
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') closeCameraViewer();
});

// ─── Polling ─────────────────────────────────────────────────────────

async function poll() {
    try {
        const data = await fetchJSON('/api/v1/cameras');
        renderCameras(data);
    } catch (e) {
        console.error('Camera poll error:', e);
    }
}

poll();
setInterval(poll, POLL_INTERVAL);
console.log('📷 Camera grid initialized — polling every 10s');
