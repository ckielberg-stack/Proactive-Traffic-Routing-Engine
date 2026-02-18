/**
 * travel_times.js — E4/E20 Corridor Travel Times Dashboard
 *
 * Polls /api/v1/travel-times every 15 seconds and renders:
 *  - Summary metrics (corridor TT, free flow, delay, status)
 *  - Northbound and Southbound segment tables
 */

const POLL_INTERVAL = 15_000;

// ── Helpers ──────────────────────────────────────────────

function fmtTime(seconds) {
    if (seconds == null || isNaN(seconds)) return '—';
    const m = Math.floor(seconds / 60);
    const s = Math.round(seconds % 60);
    return `${m}:${s.toString().padStart(2, '0')}`;
}

function fmtDelay(seconds) {
    if (seconds == null || isNaN(seconds)) return '—';
    const prefix = seconds > 0 ? '+' : '';
    return `${prefix}${Math.round(seconds)}s`;
}

function statusBadge(status) {
    const colors = {
        freeflow: 'var(--status-online)',
        slow: 'var(--urgency-soon)',
        heavy: 'var(--urgency-immediate)',
        unknown: 'var(--tmc-muted)',
    };
    const color = colors[status] || colors.unknown;
    const label = status === 'freeflow' ? 'Free' : status.charAt(0).toUpperCase() + status.slice(1);
    return `<span class="badge" style="background:${color}; color:#000; font-weight:600;">${label}</span>`;
}

function corridorStatusClass(status) {
    if (status === 'freeflow') return 'ok';
    if (status === 'degraded') return 'warn';
    if (status === 'congested') return 'error';
    return '';
}

function corridorStatusLabel(status) {
    if (status === 'freeflow') return '✅ Free Flow';
    if (status === 'degraded') return '⚠️ Degraded';
    if (status === 'congested') return '🔴 Congested';
    return '—';
}

function classifyDirection(name) {
    const upper = name.toUpperCase();
    if (upper.includes(' N ') || upper.startsWith('E4 N') || upper.startsWith('E4/E20 N')) return 'north';
    if (upper.includes(' S ') || upper.startsWith('E4 S') || upper.startsWith('E4/E20 S')) return 'south';
    return 'north'; // default to north if unclear
}

function delayClass(delay) {
    if (delay > 30) return 'color: var(--urgency-immediate); font-weight: 600;';
    if (delay > 10) return 'color: var(--urgency-soon); font-weight: 600;';
    if (delay > 0) return 'color: var(--status-verified);';
    return 'color: var(--tmc-muted);';
}

// ── Rendering ────────────────────────────────────────────

function renderRouteRow(route) {
    // Shorten segment name: strip "E4/E20 N " or "E4/E20 S " prefix
    let name = route.name || '—';
    name = name.replace(/^E4\/E20\s+[NS]\s+/i, '').replace(/^E4\s+[NS]\s+/i, '');

    const lengthKm = route.length_meters ? (route.length_meters / 1000).toFixed(1) : '?';

    return `<tr>
        <td title="${route.name}">${name} <span class="muted" style="font-size:0.625rem;">(${lengthKm}km)</span></td>
        <td style="font-weight:600; font-family:'JetBrains Mono',monospace;">${fmtTime(route.travel_time_seconds)}</td>
        <td style="color:var(--tmc-muted); font-family:'JetBrains Mono',monospace;">${fmtTime(route.free_flow_seconds)}</td>
        <td style="${delayClass(route.delay_seconds)} font-family:'JetBrains Mono',monospace;">${fmtDelay(route.delay_seconds)}</td>
        <td style="font-family:'JetBrains Mono',monospace;">${route.speed_kmh ? Math.round(route.speed_kmh) + ' km/h' : '—'}</td>
        <td>${statusBadge(route.traffic_status)}</td>
    </tr>`;
}

function updateTable(tbodyId, routes) {
    const tbody = document.getElementById(tbodyId);
    if (!tbody) return;

    if (!routes.length) {
        tbody.innerHTML = `<tr><td colspan="6" class="empty-state">No route data available</td></tr>`;
        return;
    }

    tbody.innerHTML = routes.map(renderRouteRow).join('');
}

// ── Polling ──────────────────────────────────────────────

async function refreshTravelTimes() {
    try {
        const res = await fetch('/api/v1/travel-times');
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();

        const { routes, summary, timestamp } = data;

        // Update metrics
        document.getElementById('tt-total').textContent = summary.total_routes || 0;
        document.getElementById('tt-corridor').textContent = summary.corridor_travel_time || '—';
        document.getElementById('tt-freeflow').textContent = summary.corridor_free_flow || '—';

        const delayEl = document.getElementById('tt-delay');
        const delayVal = summary.total_delay_seconds || 0;
        delayEl.textContent = fmtDelay(delayVal);
        delayEl.className = 'metric-value ' + (delayVal > 60 ? 'error' : delayVal > 0 ? 'warn' : 'ok');

        const statusEl = document.getElementById('tt-status');
        statusEl.textContent = corridorStatusLabel(summary.corridor_status);
        statusEl.className = 'metric-value ' + corridorStatusClass(summary.corridor_status);

        if (timestamp) {
            const t = new Date(timestamp);
            document.getElementById('tt-updated').textContent =
                t.toLocaleTimeString('sv-SE', { hour: '2-digit', minute: '2-digit', second: '2-digit' });
        }

        // Split routes by direction
        const northRoutes = routes.filter(r => classifyDirection(r.name) === 'north');
        const southRoutes = routes.filter(r => classifyDirection(r.name) === 'south');

        updateTable('tt-tbody-north', northRoutes);
        updateTable('tt-tbody-south', southRoutes);

    } catch (err) {
        console.error('Travel times fetch error:', err);
    }
}

// ── Init ─────────────────────────────────────────────────

refreshTravelTimes();
setInterval(refreshTravelTimes, POLL_INTERVAL);
