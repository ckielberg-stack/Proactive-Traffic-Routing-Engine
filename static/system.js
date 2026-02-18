/**
 * PTRE — System Health Page
 * Polls /api/v1/status every 10s, renders stat cards + raw JSON.
 */

const POLL_INTERVAL = 10_000;

async function fetchJSON(path) {
    const res = await fetch(path);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return res.json();
}

function formatUptime(startIso) {
    if (!startIso) return '—';
    const start = new Date(startIso);
    const now = new Date();
    const diff = Math.floor((now - start) / 1000);
    const h = Math.floor(diff / 3600);
    const m = Math.floor((diff % 3600) / 60);
    const s = diff % 60;
    if (h > 0) return `${h}h ${m}m`;
    if (m > 0) return `${m}m ${s}s`;
    return `${s}s`;
}

function renderStatus(data) {
    const tick = data.last_tick || {};

    document.getElementById('sys-tick').textContent = data.tick_count ?? '—';
    document.getElementById('sys-uptime').textContent = formatUptime(data.start_time);
    document.getElementById('sys-cameras-ok').textContent = tick.cameras_ok ?? '—';
    document.getElementById('sys-cameras-fail').textContent = tick.cameras_failed ?? '—';
    document.getElementById('sys-vehicles').textContent = tick.total_vehicles ?? '—';
    document.getElementById('sys-anomalies').textContent = data.total_anomalies ?? tick.anomalies ?? '—';
    document.getElementById('sys-sensors').textContent = tick.sensor_readings ?? '—';
    document.getElementById('sys-predictions').textContent = tick.queue_predictions ?? '—';
    document.getElementById('sys-vms-recs').textContent = tick.vms_recommendations ?? '—';
    document.getElementById('sys-vms-polled').textContent = tick.vms_statuses_polled ?? '—';
    document.getElementById('sys-interval').textContent = data.interval_seconds ?? '—';

    if (data.last_update) {
        const t = new Date(data.last_update);
        document.getElementById('sys-last-update').textContent = t.toLocaleTimeString('sv-SE');
    }

    // Raw JSON
    document.getElementById('sys-raw').textContent = JSON.stringify(data, null, 2);
}

async function poll() {
    try {
        const data = await fetchJSON('/api/v1/status');
        renderStatus(data);
    } catch (e) {
        console.error('Status poll error:', e);
    }
}

poll();
setInterval(poll, POLL_INTERVAL);
console.log('⚙️ System health initialized — polling every 10s');
