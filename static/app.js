/**
 * PTRE — TMC Control Room Dashboard
 * 
 * Polls the operator API every 10 seconds and updates the 3-column
 * dashboard: Incidents, VMS Recommendations, and Prophecy Log.
 */

const API_BASE = '';  // Same origin
const POLL_INTERVAL = 10_000; // 10 seconds

// ─────────────────────────────────────────────
// DOM references
// ─────────────────────────────────────────────

const $clock = document.getElementById('live-clock');
const $tickCounter = document.getElementById('tick-counter');
const $statusDot = document.getElementById('status-dot');
const $statusText = document.getElementById('status-text');

// Metrics bar
const $metricIncidents = document.getElementById('metric-incidents');
const $metricVms = document.getElementById('metric-vms');
const $metricProphecies = document.getElementById('metric-prophecies');
const $metricHitrate = document.getElementById('metric-hitrate');
const $metricLasttick = document.getElementById('metric-lasttick');

// Panels
const $incidents = document.getElementById('incidents-container');
const $vms = document.getElementById('vms-container');
const $terminal = document.getElementById('prophecy-terminal');

// Prophecy stats
const $logVerified = document.getElementById('log-verified');
const $logFailed = document.getElementById('log-failed');
const $logPending = document.getElementById('log-pending');
const $logExpired = document.getElementById('log-expired');

// DATEX II modal
const $modal = document.getElementById('datex2-modal');
const $modalClose = document.getElementById('modal-close');
const $backdrop = document.getElementById('modal-backdrop');
const $datexBtn = document.getElementById('btn-datex2');
const $datexContent = document.getElementById('datex2-content');


// ─────────────────────────────────────────────
// Live clock
// ─────────────────────────────────────────────

function updateClock() {
    const now = new Date();
    $clock.textContent = now.toLocaleTimeString('sv-SE', { hour12: false });
}
setInterval(updateClock, 1000);
updateClock();


// ─────────────────────────────────────────────
// API fetchers
// ─────────────────────────────────────────────

async function fetchJSON(path) {
    try {
        const res = await fetch(`${API_BASE}${path}`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return await res.json();
    } catch (err) {
        console.error(`Fetch ${path} failed:`, err);
        return null;
    }
}


// ─────────────────────────────────────────────
// Render: Health
// ─────────────────────────────────────────────

async function refreshHealth() {
    const data = await fetchJSON('/health');
    if (!data) {
        $statusDot.className = 'w-2 h-2 rounded-full bg-status-failed';
        $statusText.textContent = 'OFFLINE';
        $statusText.className = 'text-xs font-mono text-status-failed uppercase tracking-wider';
        return;
    }

    $statusDot.className = 'w-2 h-2 rounded-full bg-status-online pulse-online';
    $statusText.textContent = 'SYSTEM ONLINE';
    $statusText.className = 'text-xs font-mono text-status-online uppercase tracking-wider';

    if (data.last_tick) {
        const t = new Date(data.last_tick);
        $metricLasttick.textContent = t.toLocaleTimeString('sv-SE', { hour12: false });
    }
}


// ─────────────────────────────────────────────
// Render: Incidents
// ─────────────────────────────────────────────

function shortCamId(id) {
    // SE_STA_CAMERA_0_50438756 → 50438756
    const parts = id.split('_');
    return parts[parts.length - 1];
}

async function refreshIncidents() {
    const data = await fetchJSON('/api/v1/operator/active-incidents');
    if (!data) return;

    $metricIncidents.textContent = data.count;

    if (data.count === 0) {
        $incidents.innerHTML = `
            <div class="text-center text-tmc-muted text-xs font-mono py-12 opacity-60">
                No active incidents detected
            </div>`;
        return;
    }

    $incidents.innerHTML = data.incidents.map(inc => {
        const thumbHtml = inc.thumbnail_base64
            ? `<img src="data:image/jpeg;base64,${inc.thumbnail_base64}" 
                    class="w-full h-32 object-cover rounded-t-lg" alt="YOLO detection">`
            : '';

        const dropColor = inc.capacity_drop_percentage > 50 ? 'text-urgency-immediate' :
            inc.capacity_drop_percentage > 25 ? 'text-urgency-soon' : 'text-urgency-advisory';

        return `
            <div class="bg-tmc-panel rounded-lg border border-tmc-border panel-glow fade-in overflow-hidden">
                ${thumbHtml}
                <div class="p-3 space-y-2">
                    <div class="flex items-center justify-between">
                        <span class="text-[10px] font-mono text-tmc-muted uppercase">Camera</span>
                        <span class="text-xs font-mono font-semibold text-white">${shortCamId(inc.camera_id)}</span>
                    </div>
                    <div class="flex items-center justify-between">
                        <span class="text-[10px] font-mono text-tmc-muted uppercase">Type</span>
                        <span class="text-xs font-mono text-urgency-immediate">${inc.incident_type.replace('_', ' ')}</span>
                    </div>
                    <div class="flex items-center justify-between">
                        <span class="text-[10px] font-mono text-tmc-muted uppercase">Capacity Drop</span>
                        <span class="text-sm font-mono font-bold ${dropColor}">${inc.capacity_drop_percentage.toFixed(1)}%</span>
                    </div>
                    <div class="flex items-center justify-between">
                        <span class="text-[10px] font-mono text-tmc-muted uppercase">Lanes</span>
                        <span class="text-xs font-mono text-tmc-text">${inc.lanes_affected} / ${inc.total_lanes}</span>
                    </div>
                    <div class="flex items-center justify-between">
                        <span class="text-[10px] font-mono text-tmc-muted uppercase">Confidence</span>
                        <div class="flex items-center gap-1.5">
                            <div class="w-16 h-1.5 rounded-full bg-tmc-accent overflow-hidden">
                                <div class="h-full rounded-full bg-status-verified" 
                                     style="width: ${(inc.confidence * 100).toFixed(0)}%"></div>
                            </div>
                            <span class="text-[10px] font-mono text-tmc-muted">${(inc.confidence * 100).toFixed(0)}%</span>
                        </div>
                    </div>
                </div>
            </div>`;
    }).join('');
}


// ─────────────────────────────────────────────
// Render: VMS Recommendations
// ─────────────────────────────────────────────

function urgencyBorder(urgency) {
    switch (urgency) {
        case 'IMMEDIATE': return 'border-urgency-immediate urgency-flash-immediate';
        case 'SOON': return 'border-urgency-soon';
        case 'ADVISORY': return 'border-urgency-advisory';
        default: return 'border-tmc-border';
    }
}

function urgencyDot(urgency) {
    switch (urgency) {
        case 'IMMEDIATE': return 'bg-urgency-immediate';
        case 'SOON': return 'bg-urgency-soon';
        case 'ADVISORY': return 'bg-urgency-advisory';
        default: return 'bg-tmc-muted';
    }
}

function urgencyLabel(urgency) {
    switch (urgency) {
        case 'IMMEDIATE': return 'text-urgency-immediate';
        case 'SOON': return 'text-urgency-soon';
        case 'ADVISORY': return 'text-urgency-advisory';
        default: return 'text-tmc-muted';
    }
}

async function refreshVMS() {
    const data = await fetchJSON('/api/v1/operator/vms-recommendations');
    if (!data) return;

    $metricVms.textContent = data.count;

    if (data.count === 0) {
        $vms.innerHTML = `
            <div class="text-center text-tmc-muted text-xs font-mono py-12 opacity-60">
                No active VMS recommendations
            </div>`;
        return;
    }

    $vms.innerHTML = data.recommendations.map(item => {
        const rec = item.recommendation;
        const urgency = rec.urgency || 'ADVISORY';
        const borderClass = urgencyBorder(urgency);
        const dotClass = urgencyDot(urgency);
        const labelClass = urgencyLabel(urgency);

        const gtBadge = item.proxy_ground_truth_active
            ? `<span class="text-[10px] font-mono bg-status-verified/20 text-status-verified px-1.5 py-0.5 rounded">OPERATOR ACTED</span>`
            : `<span class="text-[10px] font-mono bg-urgency-soon/20 text-urgency-soon px-1.5 py-0.5 rounded">PENDING</span>`;

        return `
            <div class="bg-tmc-panel rounded-lg border-2 ${borderClass} panel-glow fade-in">
                <div class="p-3 space-y-2.5">
                    <!-- Header -->
                    <div class="flex items-center justify-between">
                        <div class="flex items-center gap-2">
                            <div class="w-2 h-2 rounded-full ${dotClass}"></div>
                            <span class="text-xs font-mono font-bold ${labelClass}">${urgency}</span>
                        </div>
                        ${gtBadge}
                    </div>

                    <!-- VMS Name -->
                    <div class="text-xs font-mono text-white/90 truncate">${rec.vms_name}</div>

                    <!-- Swedish message -->
                    <div class="bg-tmc-bg rounded px-3 py-2 border border-tmc-border">
                        <span class="text-sm font-mono font-bold text-urgency-soon">${rec.recommended_message}</span>
                    </div>

                    <!-- ETA -->
                    <div class="flex items-center justify-between">
                        <span class="text-[10px] font-mono text-tmc-muted uppercase">ETA to VMS</span>
                        <span class="text-lg font-mono font-bold ${labelClass}">${rec.estimated_activation_minutes.toFixed(1)} min</span>
                    </div>

                    <!-- Queue speed -->
                    <div class="flex items-center justify-between">
                        <span class="text-[10px] font-mono text-tmc-muted uppercase">Queue Growth</span>
                        <span class="text-xs font-mono text-tmc-text">${rec.queue_growth_speed_kmh.toFixed(1)} km/h</span>
                    </div>

                    <!-- Trigger cam -->
                    <div class="flex items-center justify-between">
                        <span class="text-[10px] font-mono text-tmc-muted uppercase">Trigger Camera</span>
                        <span class="text-xs font-mono text-tmc-muted">${shortCamId(rec.triggering_camera_id)}</span>
                    </div>

                    <!-- Summary -->
                    <p class="text-[11px] text-tmc-muted leading-relaxed border-t border-tmc-border pt-2">${rec.summary}</p>
                </div>
            </div>`;
    }).join('');
}


// ─────────────────────────────────────────────
// Render: Prophecy Log
// ─────────────────────────────────────────────

function statusIcon(status) {
    switch (status) {
        case 'VERIFIED_SUCCESS': return '<span class="text-status-verified">✅</span>';
        case 'FAILED': return '<span class="text-status-failed">❌</span>';
        case 'EXPIRED': return '<span class="text-status-expired">🗑</span>';
        case 'pending': return '<span class="text-status-pending">⏳</span>';
        default: return '🔮';
    }
}

function statusColor(status) {
    switch (status) {
        case 'VERIFIED_SUCCESS': return 'text-status-verified';
        case 'FAILED': return 'text-status-failed';
        case 'EXPIRED': return 'text-status-expired';
        case 'pending': return 'text-status-pending';
        default: return 'text-tmc-muted';
    }
}

async function refreshProphecyLog() {
    const data = await fetchJSON('/api/v1/evaluation/log');
    if (!data) return;

    // Update stats
    if (data.stats) {
        const s = data.stats;
        $logVerified.textContent = s.verified_success || 0;
        $logFailed.textContent = s.failed || 0;
        $logPending.textContent = s.pending || 0;
        $logExpired.textContent = s.expired || 0;
        $metricProphecies.textContent = s.total_prophecies_created || 0;
        $metricHitrate.textContent = s.hit_rate !== null ? `${(s.hit_rate * 100).toFixed(1)}%` : '—';
    }

    if (!data.entries || data.entries.length === 0) {
        $terminal.innerHTML = `
            <div class="text-tmc-muted opacity-60">
                $ awaiting prophecy data...<span class="animate-pulse">█</span>
            </div>`;
        return;
    }

    const lines = data.entries.map(e => {
        const icon = statusIcon(e.status);
        const color = statusColor(e.status);

        if (e.status === 'pending') {
            return `<div class="flex gap-2 py-0.5 opacity-80">
                <span class="text-tmc-muted shrink-0">[${e.time}]</span>
                ${icon}
                <span class="text-tmc-text">Predicted queue <span class="text-white font-semibold">${e.source}</span> → <span class="text-white font-semibold">${e.target}</span> in <span class="text-urgency-soon">${e.eta_min}m</span></span>
            </div>`;
        }

        const evalTime = e.eval_time ? `[${e.eval_time}]` : '';
        const capInfo = e.capacity_vph !== undefined ? ` | ${e.capacity_vph} VPH` : '';

        return `<div class="flex gap-2 py-0.5">
            <span class="text-tmc-muted shrink-0">[${e.time}]</span>
            ${icon}
            <span class="${color}">
                ${e.source} → ${e.target} (${e.eta_min}m) → 
                <span class="font-semibold">${e.status}</span>
                <span class="text-tmc-muted">${evalTime}${capInfo}</span>
            </span>
        </div>`;
    }).join('');

    $terminal.innerHTML = lines;
}


// ─────────────────────────────────────────────
// DATEX II Modal
// ─────────────────────────────────────────────

$datexBtn.addEventListener('click', async () => {
    $modal.classList.remove('hidden');
    $datexContent.textContent = 'Loading DATEX II export...';

    try {
        const res = await fetch(`${API_BASE}/api/v1/export/datex2`);
        const xml = await res.text();
        $datexContent.textContent = xml;
    } catch (err) {
        $datexContent.textContent = `Error: ${err.message}`;
    }
});

$modalClose.addEventListener('click', () => $modal.classList.add('hidden'));
$backdrop.addEventListener('click', () => $modal.classList.add('hidden'));
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') $modal.classList.add('hidden');
});


// ─────────────────────────────────────────────
// Main polling loop
// ─────────────────────────────────────────────

async function pollAll() {
    await Promise.all([
        refreshHealth(),
        refreshIncidents(),
        refreshVMS(),
        refreshProphecyLog(),
    ]);
}

// Initial fetch
pollAll();

// Poll every 10 seconds
setInterval(pollAll, POLL_INTERVAL);

console.log('🚀 PTRE TMC Dashboard initialized — polling every 10s');
