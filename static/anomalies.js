/**
 * PTRE — Anomaly Log Page
 * Polls /api/v1/anomalies every 10s, renders the table + image modal.
 */

const POLL_INTERVAL = 10_000;

async function fetchJSON(path) {
    const res = await fetch(path);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return res.json();
}

function formatTime(isoStr) {
    if (!isoStr) return '—';
    const d = new Date(isoStr);
    return d.toLocaleTimeString('sv-SE');
}

function formatDate(isoStr) {
    if (!isoStr) return '—';
    const d = new Date(isoStr);
    return d.toLocaleDateString('sv-SE');
}

function getImageUrl(imagePath) {
    if (!imagePath) return null;
    // imagePath looks like: storage/anomalies/2026-02-18/CAM_ID_14-30-00_annotated.jpg
    // We need: /api/v1/anomaly-image/2026-02-18/CAM_ID_14-30-00_annotated.jpg
    const parts = imagePath.replace(/\\/g, '/').split('/');
    // Find "anomalies" index and take the rest
    const anomIdx = parts.indexOf('anomalies');
    if (anomIdx < 0 || anomIdx + 2 >= parts.length) return null;
    const date = parts[anomIdx + 1];
    const filename = parts[anomIdx + 2];
    return `/api/v1/anomaly-image/${date}/${filename}`;
}

function isToday(isoStr) {
    if (!isoStr) return false;
    const d = new Date(isoStr);
    const now = new Date();
    return d.toDateString() === now.toDateString();
}

function confidenceColor(conf) {
    if (conf >= 0.7) return 'var(--status-online)';
    if (conf >= 0.4) return 'var(--urgency-soon)';
    return 'var(--urgency-immediate)';
}

function reasonBadge(reason) {
    const cls = reason.includes('aspect') ? 'badge-warn' :
        reason.includes('zero') ? 'badge-fail' :
            reason.includes('black') ? 'badge-fail' : 'badge-warn';
    return `<span class="badge ${cls}">${reason}</span>`;
}

function renderTable(events) {
    const tbody = document.getElementById('anomaly-tbody');
    if (!events || events.length === 0) {
        tbody.innerHTML = '<tr><td colspan="7" class="empty-state">No anomalies recorded yet</td></tr>';
        return;
    }

    tbody.innerHTML = events.map(evt => {
        const hasImage = !!evt.image_path;
        const imgUrl = getImageUrl(evt.image_path);
        const rowClass = hasImage ? 'anomaly-row clickable' : 'anomaly-row';
        const cursor = hasImage ? 'cursor: pointer;' : '';
        const imgIcon = hasImage ? '🖼️' : '—';
        const dataAttr = imgUrl
            ? `data-img="${imgUrl}" data-camera="${evt.camera_name || evt.camera_id}" data-reason="${evt.anomaly_reason}" data-time="${evt.timestamp}"`
            : '';

        return `<tr class="${rowClass}" style="${cursor}" ${dataAttr}>
            <td>${formatTime(evt.timestamp)}<br><span style="font-size: 0.55rem; color: var(--tmc-muted);">${formatDate(evt.timestamp)}</span></td>
            <td>${evt.camera_name || evt.camera_id}</td>
            <td>${reasonBadge(evt.anomaly_reason || 'unknown')}</td>
            <td style="color: ${confidenceColor(evt.confidence)};">${(evt.confidence * 100).toFixed(1)}%</td>
            <td>${evt.vehicle_count}</td>
            <td>${evt.capacity_vph?.toLocaleString() || '—'}</td>
            <td style="text-align: center;">${imgIcon}</td>
        </tr>`;
    }).join('');

    // Attach click handlers
    tbody.querySelectorAll('.anomaly-row.clickable').forEach(row => {
        row.addEventListener('click', () => {
            const imgUrl = row.dataset.img;
            const camera = row.dataset.camera;
            const reason = row.dataset.reason;
            const time = row.dataset.time;
            openModal(imgUrl, camera, reason, time);
        });
    });
}

function renderSummary(data) {
    document.getElementById('anomaly-total').textContent = data.total ?? '0';

    const todayCount = (data.events || []).filter(e => isToday(e.timestamp)).length;
    document.getElementById('anomaly-today').textContent = todayCount;

    if (data.timestamp) {
        document.getElementById('anomaly-last-update').textContent = formatTime(data.timestamp);
    }
}

/* ---- Modal ---- */

const modal = document.getElementById('anomaly-modal');
const modalImg = document.getElementById('anomaly-modal-img');
const modalLoading = document.getElementById('anomaly-modal-loading');
const modalTitle = document.getElementById('anomaly-modal-title');
const modalFooter = document.getElementById('anomaly-modal-footer');

function openModal(imgUrl, camera, reason, time) {
    modal.style.display = 'flex';
    modalImg.style.display = 'none';
    modalLoading.style.display = 'flex';
    modalTitle.textContent = `${camera} — ${reason}`;
    modalFooter.textContent = `${formatTime(time)} ${formatDate(time)} | ${reason}`;

    modalImg.onload = () => {
        modalLoading.style.display = 'none';
        modalImg.style.display = 'block';
    };
    modalImg.onerror = () => {
        modalLoading.innerHTML = '<span style="color: var(--status-failed);">Failed to load image</span>';
    };
    modalImg.src = imgUrl;
}

function closeModal() {
    modal.style.display = 'none';
    modalImg.src = '';
}

document.getElementById('anomaly-modal-close').addEventListener('click', closeModal);
document.getElementById('anomaly-modal-backdrop').addEventListener('click', closeModal);
document.addEventListener('keydown', e => { if (e.key === 'Escape') closeModal(); });

/* ---- Polling ---- */

async function poll() {
    try {
        const data = await fetchJSON('/api/v1/anomalies?limit=200');
        renderTable(data.events);
        renderSummary(data);
    } catch (e) {
        console.error('Anomaly poll error:', e);
    }
}

poll();
setInterval(poll, POLL_INTERVAL);
console.log('🚨 Anomaly log initialized — polling every 10s');
