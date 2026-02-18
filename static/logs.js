/**
 * PTRE — System Log Page
 * Polls /api/v1/logs every 5s, renders terminal-style log viewer.
 */

const POLL_INTERVAL = 5_000;
let autoScroll = true;

async function fetchJSON(path) {
    const res = await fetch(path);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return res.json();
}

function levelClass(level) {
    switch (level) {
        case 'error': return 'error';
        case 'warning': return 'warning';
        case 'debug': return 'debug';
        default: return 'info';
    }
}

function renderLogs(data) {
    const container = document.getElementById('log-container');
    const lines = data.lines || [];

    // Update metrics
    document.getElementById('log-total').textContent = data.total ?? lines.length;

    let errors = 0, warnings = 0;
    lines.forEach(l => {
        if (l.level === 'error') errors++;
        if (l.level === 'warning') warnings++;
    });
    document.getElementById('log-errors').textContent = errors;
    document.getElementById('log-warnings').textContent = warnings;

    if (lines.length === 0) {
        container.innerHTML = '<div class="empty-state">No log entries yet</div>';
        return;
    }

    container.innerHTML = lines.map(l =>
        `<div class="log-line ${levelClass(l.level)}">${escapeHtml(l.text)}</div>`
    ).join('');

    if (autoScroll) {
        container.scrollTop = container.scrollHeight;
    }
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Toggle auto-scroll on click
document.getElementById('log-autoscroll').addEventListener('click', () => {
    autoScroll = !autoScroll;
    document.getElementById('log-autoscroll').textContent = autoScroll ? 'ON' : 'OFF';
    document.getElementById('log-autoscroll').style.color = autoScroll
        ? 'var(--status-online)' : 'var(--tmc-muted)';
});

async function poll() {
    try {
        const data = await fetchJSON('/api/v1/logs?lines=200');
        renderLogs(data);
    } catch (e) {
        console.error('Log poll error:', e);
    }
}

poll();
setInterval(poll, POLL_INTERVAL);
console.log('📋 Log viewer initialized — polling every 5s');
