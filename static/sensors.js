/**
 * PTRE — Sensor Data Page
 * Polls /api/v1/sensors every 10s, renders sensor station table.
 */

const POLL_INTERVAL = 10_000;

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
        return `
            <tr>
                <td>${r.site_id}</td>
                <td>${r.lat?.toFixed(4) ?? '—'}</td>
                <td style="${volColor}">${r.volume_vph.toLocaleString()}</td>
                <td style="${spdColor}">${r.speed_kmh.toFixed(1)}</td>
                <td>${r.mapped_camera ?? '—'}</td>
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
console.log('🔢 Sensor table initialized — polling every 10s');
