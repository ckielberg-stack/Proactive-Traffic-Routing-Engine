/**
 * PTRE — Sensor Data Page
 * Polls /api/v1/sensors every 10s, renders sensor station table.
 * Mapped cameras are clickable — opens inline image popup.
 */

const POLL_INTERVAL = 10_000;

/* ── Camera image popup ─────────────────────────────────────── */
let _camPopup = null;

function closeCamPopup() {
    if (_camPopup) { _camPopup.remove(); _camPopup = null; }
}

function openCamPopup(cameraId, anchorEl) {
    closeCamPopup();
    const shortId = cameraId.split('_').pop();

    const popup = document.createElement('div');
    popup.id = 'sensor-cam-popup';
    popup.style.cssText =
        'position:fixed;z-index:9000;background:#1a1d26;border:1px solid #334;' +
        'border-radius:10px;box-shadow:0 12px 40px rgba(0,0,0,.6);' +
        'max-width:640px;width:90vw;overflow:hidden;';

    // Position near the clicked element
    const rect = anchorEl.getBoundingClientRect();
    const top = Math.min(rect.bottom + 8, window.innerHeight - 500);
    const left = Math.min(rect.left, window.innerWidth - 660);
    popup.style.top = Math.max(8, top) + 'px';
    popup.style.left = Math.max(8, left) + 'px';

    popup.innerHTML = `
        <div style="display:flex;justify-content:space-between;align-items:center;
                    padding:10px 14px;border-bottom:1px solid #334;">
            <span style="font-weight:600;color:#e0e0e0;font-size:13px;">
                📷 Camera ${shortId}
                <span style="color:#888;font-size:11px;margin-left:6px;">${cameraId}</span>
            </span>
            <button id="cam-popup-close" style="background:none;border:none;color:#888;
                    font-size:18px;cursor:pointer;padding:2px 6px;">✕</button>
        </div>
        <div id="cam-popup-body" style="padding:8px;text-align:center;min-height:200px;
                display:flex;align-items:center;justify-content:center;">
            <span style="color:#888;font-size:13px;">Loading image…</span>
        </div>
        <div style="padding:6px 14px 10px;border-top:1px solid #334;text-align:right;">
            <a href="/cameras" style="color:#5b9cf4;font-size:12px;text-decoration:none;"
               title="Open full camera grid">Open in Cameras →</a>
        </div>`;

    document.body.appendChild(popup);
    _camPopup = popup;

    // Close button
    document.getElementById('cam-popup-close').addEventListener('click', closeCamPopup);

    // Close on click outside
    setTimeout(() => {
        document.addEventListener('click', function handler(e) {
            if (_camPopup && !_camPopup.contains(e.target)) {
                closeCamPopup();
                document.removeEventListener('click', handler);
            }
        });
    }, 100);

    // Load image
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => {
        const body = document.getElementById('cam-popup-body');
        if (body) {
            body.innerHTML = '';
            img.style.cssText = 'width:100%;height:auto;border-radius:4px;display:block;';
            body.appendChild(img);
        }
    };
    img.onerror = () => {
        const body = document.getElementById('cam-popup-body');
        if (body) body.innerHTML = '<span style="color:#f44;">Failed to load image</span>';
    };
    img.src = `/api/v1/camera-image/${encodeURIComponent(cameraId)}?t=${Date.now()}`;
}

// Close popup on Escape
document.addEventListener('keydown', e => { if (e.key === 'Escape') closeCamPopup(); });

async function fetchJSON(path) {
    const res = await fetch(path);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return res.json();
}

function renderSensors(data) {
    const tbody = document.getElementById('sensor-tbody');
    const readings = data.readings || [];

    // Update metrics
    document.getElementById('sen-total').textContent = readings.length;
    document.getElementById('sen-mapped').textContent = data.mapped_cameras ?? '—';

    if (readings.length > 0) {
        const avgVol = readings.reduce((s, r) => s + r.volume_vph, 0) / readings.length;
        const avgSpd = readings.reduce((s, r) => s + r.speed_kmh, 0) / readings.length;
        document.getElementById('sen-avg-vol').textContent = Math.round(avgVol).toLocaleString() + ' VPH';
        document.getElementById('sen-avg-speed').textContent = avgSpd.toFixed(1) + ' km/h';
    }

    if (data.timestamp) {
        const t = new Date(data.timestamp);
        document.getElementById('sen-updated').textContent = t.toLocaleTimeString('sv-SE');
    }

    if (readings.length === 0) {
        tbody.innerHTML = '<tr><td colspan="5" class="empty-state">No sensor data available yet</td></tr>';
        return;
    }

    tbody.innerHTML = readings.map(r => {
        const volColor = r.volume_vph > 3000 ? 'color: var(--urgency-soon)' :
            r.volume_vph > 5000 ? 'color: var(--urgency-immediate)' : '';
        const spdColor = r.speed_kmh < 40 ? 'color: var(--urgency-immediate)' :
            r.speed_kmh < 70 ? 'color: var(--urgency-soon)' : '';

        let camCell;
        if (r.mapped_camera) {
            const shortId = r.mapped_camera.split('_').pop();
            camCell = `<a href="#" class="cam-link" data-cam="${r.mapped_camera}"
                          style="color:#5b9cf4;text-decoration:none;cursor:pointer;
                                 border-bottom:1px dashed #5b9cf4;"
                          title="Click to preview camera image">
                          📷 ${shortId}</a>`;
        } else {
            camCell = '<span style="color:#555;">—</span>';
        }

        return `
            <tr>
                <td>${r.site_id}</td>
                <td>${r.lat?.toFixed(4) ?? '—'}</td>
                <td style="${volColor}">${r.volume_vph.toLocaleString()}</td>
                <td style="${spdColor}">${r.speed_kmh.toFixed(1)}</td>
                <td>${camCell}</td>
            </tr>`;
    }).join('');
}

async function poll() {
    try {
        const data = await fetchJSON('/api/v1/sensors');
        renderSensors(data);
    } catch (e) {
        console.error('Sensor poll error:', e);
    }
}

poll();
setInterval(poll, POLL_INTERVAL);

// Event delegation for camera links (survives table re-renders)
document.getElementById('sensor-tbody').addEventListener('click', e => {
    const link = e.target.closest('.cam-link');
    if (link) {
        e.preventDefault();
        e.stopPropagation();
        openCamPopup(link.dataset.cam, link);
    }
});

console.log('🔢 Sensor table initialized — polling every 10s');
