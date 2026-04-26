/* ═══════════════════════════════════════════════════════════════════════════
   FraudShield AI — Dashboard Application
   Fetches API data, renders charts, tables, alerts, and timeline
   ═══════════════════════════════════════════════════════════════════════════ */

// ── Chart.js Global Defaults ────────────────────────────────────────────────
Chart.defaults.color = '#8b92a8';
Chart.defaults.borderColor = 'rgba(255,255,255,0.06)';
Chart.defaults.font.family = "'Inter', sans-serif";
Chart.defaults.font.size = 12;
Chart.defaults.plugins.legend.labels.usePointStyle = true;
Chart.defaults.plugins.legend.labels.pointStyleWidth = 10;
Chart.defaults.plugins.legend.labels.padding = 16;

// ── State ───────────────────────────────────────────────────────────────────
let dashboardData = null;
let currentTxnFilter = 'all';
let currentTxnPage = 1;
const TXN_PAGE_SIZE = 20;
const chartInstances = {};

// ── Color Palette ───────────────────────────────────────────────────────────
const COLORS = {
    blue: '#00d4ff',
    purple: '#7b2fff',
    pink: '#ff2fa0',
    green: '#00e68a',
    orange: '#ff9f43',
    red: '#ff4757',
    yellow: '#ffd32a',
    blueAlpha: 'rgba(0, 212, 255, 0.15)',
    purpleAlpha: 'rgba(123, 47, 255, 0.15)',
    redAlpha: 'rgba(255, 71, 87, 0.15)',
    greenAlpha: 'rgba(0, 230, 138, 0.15)',
};

const GRADIENT_COLORS = [
    '#00d4ff', '#3db8f5', '#7b9ceb', '#9b7fe0', '#7b2fff',
    '#9b2fe0', '#bb2fc0', '#db2fa0', '#ff2fa0', '#ff4757',
];

// ══════════════════════════════════════════════════════════════════════════
// INITIALIZATION
// ══════════════════════════════════════════════════════════════════════════

document.addEventListener('DOMContentLoaded', () => {
    initNavigation();
    updateClock();
    setInterval(updateClock, 1000);
    loadDashboard();
});

function initNavigation() {
    document.querySelectorAll('.nav-item').forEach(item => {
        item.addEventListener('click', () => {
            const section = item.dataset.section;
            if (!section) return;

            // Update active nav
            document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
            item.classList.add('active');

            // Show section
            document.querySelectorAll('.content-section').forEach(s => s.classList.remove('active'));
            const target = document.getElementById(`section-${section}`);
            if (target) target.classList.add('active');

            // Update title
            const titles = {
                overview: 'Executive Dashboard',
                transactions: 'Transaction Monitor',
                alerts: 'Alerts & Early Warning Signals',
                models: 'Model Performance Analytics',
                timeline: 'Implementation Timeline',
            };
            document.getElementById('page-title').textContent = titles[section] || 'Dashboard';

            // Close mobile sidebar
            document.getElementById('sidebar').classList.remove('open');
        });
    });

    // Mobile menu toggle
    document.getElementById('menu-toggle').addEventListener('click', () => {
        document.getElementById('sidebar').classList.toggle('open');
    });

    // Transaction filters
    document.querySelectorAll('.filter-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            currentTxnFilter = btn.dataset.filter;
            currentTxnPage = 1;
            loadTransactions();
        });
    });
}

function updateClock() {
    const now = new Date();
    document.getElementById('header-clock').textContent = now.toLocaleString('en-GB', {
        day: '2-digit', month: 'short', year: 'numeric',
        hour: '2-digit', minute: '2-digit', second: '2-digit',
    });
}

// ══════════════════════════════════════════════════════════════════════════
// DATA LOADING
// ══════════════════════════════════════════════════════════════════════════

async function loadDashboard() {
    try {
        const [statsRes, modelRes, timelineRes] = await Promise.all([
            fetch('/api/dashboard/stats'),
            fetch('/api/model/performance'),
            fetch('/api/timeline'),
        ]);

        dashboardData = {
            stats: await statsRes.json(),
            models: await modelRes.json(),
            timeline: await timelineRes.json(),
        };

        renderKPIs(dashboardData.stats);
        renderCharts(dashboardData.stats);
        renderModelPerformance(dashboardData.models);
        renderTimeline(dashboardData.timeline);
        loadTransactions();
        loadAlerts();

        // Update alert badge
        const alertCount = dashboardData.stats.alerts.open;
        document.getElementById('alert-badge').textContent = alertCount;
    } catch (err) {
        console.error('Failed to load dashboard:', err);
    }
}

// ══════════════════════════════════════════════════════════════════════════
// KPI CARDS
// ══════════════════════════════════════════════════════════════════════════

function renderKPIs(stats) {
    const s = stats.summary;
    animateValue('kpi-total-value', 0, s.total_transactions, 1000, formatNumber);
    animateValue('kpi-fraud-value', 0, s.fraud_detected, 1000, formatNumber);
    animateValue('kpi-loss-value', 0, s.blocked_amount, 1200, formatRupiah);
    document.getElementById('kpi-sla-value').textContent = `+${s.sla_improvement_pct}%`;
}

function animateValue(elementId, start, end, duration, formatter) {
    const el = document.getElementById(elementId);
    if (!el) return;
    const range = end - start;
    const startTime = performance.now();

    function update(now) {
        const elapsed = now - startTime;
        const progress = Math.min(elapsed / duration, 1);
        const eased = 1 - Math.pow(1 - progress, 3);
        const current = Math.floor(start + range * eased);
        el.textContent = formatter(current);
        if (progress < 1) requestAnimationFrame(update);
    }
    requestAnimationFrame(update);
}

// ══════════════════════════════════════════════════════════════════════════
// CHARTS
// ══════════════════════════════════════════════════════════════════════════

function renderCharts(stats) {
    renderMonthlyTrend(stats.charts);
    renderRiskDistribution(stats.charts);
    renderFraudByChannel(stats.charts);
    renderFraudByType(stats.charts);
    renderEWSSeverity(stats.ews);
}

function renderMonthlyTrend(charts) {
    const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
    const totalData = Object.values(charts.monthly_total);
    const fraudData = Object.values(charts.monthly_fraud);

    createChart('chart-monthly-trend', {
        type: 'line',
        data: {
            labels: months,
            datasets: [
                {
                    label: 'Total Transactions',
                    data: totalData,
                    borderColor: COLORS.blue,
                    backgroundColor: COLORS.blueAlpha,
                    fill: true,
                    tension: 0.4,
                    borderWidth: 2,
                    pointRadius: 4,
                    pointBackgroundColor: COLORS.blue,
                    pointHoverRadius: 6,
                },
                {
                    label: 'Fraud Detected',
                    data: fraudData,
                    borderColor: COLORS.red,
                    backgroundColor: COLORS.redAlpha,
                    fill: true,
                    tension: 0.4,
                    borderWidth: 2,
                    pointRadius: 4,
                    pointBackgroundColor: COLORS.red,
                    pointHoverRadius: 6,
                    yAxisID: 'y1',
                },
            ],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { mode: 'index', intersect: false },
            scales: {
                y: {
                    beginAtZero: true,
                    grid: { color: 'rgba(255,255,255,0.04)' },
                    ticks: { callback: v => formatNumber(v) },
                },
                y1: {
                    position: 'right',
                    beginAtZero: true,
                    grid: { drawOnChartArea: false },
                    ticks: { callback: v => formatNumber(v) },
                },
                x: { grid: { display: false } },
            },
            plugins: { legend: { position: 'top' } },
        },
    });
}

function renderRiskDistribution(charts) {
    const labels = Object.keys(charts.risk_score_distribution);
    const data = Object.values(charts.risk_score_distribution);

    const barColors = labels.map((_, i) => {
        const ratio = i / (labels.length - 1);
        const r = Math.round(0 + ratio * 255);
        const g = Math.round(212 - ratio * 141);
        const b = Math.round(255 - ratio * 168);
        return `rgba(${r}, ${g}, ${b}, 0.7)`;
    });

    createChart('chart-risk-dist', {
        type: 'bar',
        data: {
            labels,
            datasets: [{
                label: 'Transactions',
                data,
                backgroundColor: barColors,
                borderRadius: 6,
                borderSkipped: false,
                barPercentage: 0.7,
            }],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                y: {
                    beginAtZero: true,
                    grid: { color: 'rgba(255,255,255,0.04)' },
                    ticks: { callback: v => formatNumber(v) },
                },
                x: { grid: { display: false } },
            },
        },
    });
}

function renderFraudByChannel(charts) {
    const labels = Object.keys(charts.fraud_by_channel);
    const data = Object.values(charts.fraud_by_channel);
    const colors = [COLORS.blue, COLORS.purple, COLORS.pink, COLORS.green,
                    COLORS.orange, COLORS.red, COLORS.yellow];

    createChart('chart-fraud-channel', {
        type: 'doughnut',
        data: {
            labels,
            datasets: [{
                data,
                backgroundColor: colors.map(c => c + '99'),
                borderColor: colors,
                borderWidth: 2,
                hoverOffset: 8,
            }],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            cutout: '58%',
            plugins: {
                legend: { position: 'right', labels: { font: { size: 11 } } },
            },
        },
    });
}

function renderFraudByType(charts) {
    const labels = Object.keys(charts.fraud_by_type).map(l =>
        l.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())
    );
    const data = Object.values(charts.fraud_by_type);
    const colors = [COLORS.red, COLORS.orange, COLORS.purple, COLORS.blue, COLORS.pink];

    createChart('chart-fraud-type', {
        type: 'polarArea',
        data: {
            labels,
            datasets: [{
                data,
                backgroundColor: colors.map(c => c + '55'),
                borderColor: colors,
                borderWidth: 2,
            }],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { position: 'right', labels: { font: { size: 11 } } },
            },
            scales: {
                r: {
                    grid: { color: 'rgba(255,255,255,0.05)' },
                    ticks: { display: false },
                },
            },
        },
    });
}

function renderEWSSeverity(ews) {
    const labels = ['Green', 'Yellow', 'Orange', 'Red'];
    const data = [ews.by_severity.Green, ews.by_severity.Yellow, ews.by_severity.Orange, ews.by_severity.Red];
    const colors = [COLORS.green, COLORS.yellow, COLORS.orange, COLORS.red];

    createChart('chart-ews-severity', {
        type: 'doughnut',
        data: {
            labels,
            datasets: [{
                data,
                backgroundColor: colors.map(c => c + '88'),
                borderColor: colors,
                borderWidth: 2,
                hoverOffset: 8,
            }],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            cutout: '55%',
            plugins: {
                legend: { position: 'right', labels: { font: { size: 11 } } },
            },
        },
    });
}

// ══════════════════════════════════════════════════════════════════════════
// TRANSACTIONS TABLE
// ══════════════════════════════════════════════════════════════════════════

async function loadTransactions() {
    const params = new URLSearchParams({
        page: currentTxnPage,
        page_size: TXN_PAGE_SIZE,
    });
    if (currentTxnFilter !== 'all') params.set('risk_level', currentTxnFilter);

    try {
        const res = await fetch(`/api/transactions?${params}`);
        const data = await res.json();
        renderTransactionsTable(data.transactions);
        renderPagination(data.pagination);
    } catch (err) {
        console.error('Failed to load transactions:', err);
    }
}

function renderTransactionsTable(txns) {
    const tbody = document.getElementById('txn-tbody');
    tbody.innerHTML = txns.map(t => {
        const riskClass = t.risk_level === 'HIGH' ? 'high' : t.risk_level === 'MEDIUM' ? 'medium' : 'low';
        const decisionClass = t.decision === 'BLOCK' ? 'block' : t.decision === 'FLAG' ? 'flag' : 'approve';
        return `
            <tr>
                <td><span style="font-weight:600;color:var(--accent-blue)">${t.transaction_id}</span></td>
                <td>${t.customer_id}</td>
                <td>${formatTimestamp(t.timestamp)}</td>
                <td style="font-weight:600">${formatRupiah(t.amount)}</td>
                <td>${t.channel}</td>
                <td>${t.merchant_category}</td>
                <td><span class="risk-badge ${riskClass}">
                    <span class="dot ${riskClass === 'high' ? 'red' : riskClass === 'medium' ? 'orange' : 'green'}"></span>
                    ${t.risk_score}
                </span></td>
                <td><span class="decision-badge ${decisionClass}">${t.decision}</span></td>
            </tr>
        `;
    }).join('');
}

function renderPagination(pag) {
    const container = document.getElementById('txn-pagination');
    if (pag.total_pages <= 1) { container.innerHTML = ''; return; }

    let html = '';
    const maxVisible = 7;
    const start = Math.max(1, pag.page - Math.floor(maxVisible / 2));
    const end = Math.min(pag.total_pages, start + maxVisible - 1);

    if (pag.page > 1) {
        html += `<button class="page-btn" onclick="goToPage(${pag.page - 1})">‹</button>`;
    }

    for (let i = start; i <= end; i++) {
        html += `<button class="page-btn ${i === pag.page ? 'active' : ''}" onclick="goToPage(${i})">${i}</button>`;
    }

    if (pag.page < pag.total_pages) {
        html += `<button class="page-btn" onclick="goToPage(${pag.page + 1})">›</button>`;
    }

    container.innerHTML = html;
}

function goToPage(page) {
    currentTxnPage = page;
    loadTransactions();
}

// ══════════════════════════════════════════════════════════════════════════
// ALERTS
// ══════════════════════════════════════════════════════════════════════════

async function loadAlerts() {
    try {
        const res = await fetch('/api/alerts?page_size=50');
        const data = await res.json();

        const fraudAlerts = data.alerts.filter(a => a.alert_type === 'Fraud Detection');
        const ewsAlerts = data.alerts.filter(a => a.alert_type === 'Early Warning Signal');

        document.getElementById('fraud-alert-count').textContent = fraudAlerts.length;
        document.getElementById('ews-alert-count').textContent = ewsAlerts.length;

        renderAlertsList('fraud-alerts-list', fraudAlerts);
        renderAlertsList('ews-alerts-list', ewsAlerts);
    } catch (err) {
        console.error('Failed to load alerts:', err);
    }
}

function renderAlertsList(containerId, alerts) {
    const container = document.getElementById(containerId);
    container.innerHTML = alerts.map(a => {
        const sevClass = a.severity.toLowerCase();
        return `
            <div class="alert-item ${sevClass}">
                <div class="alert-item-header">
                    <span class="alert-item-id">${a.alert_id}</span>
                    <span class="alert-severity ${sevClass}">${a.severity}</span>
                </div>
                <div class="alert-description">${a.description}</div>
                <div class="alert-meta">
                    <span>${a.customer_id}</span>
                    <span>•</span>
                    <span>${a.status}</span>
                    <span>•</span>
                    <span class="alert-score">Score: ${a.risk_score}</span>
                </div>
            </div>
        `;
    }).join('');
}

// ══════════════════════════════════════════════════════════════════════════
// MODEL PERFORMANCE
// ══════════════════════════════════════════════════════════════════════════

function renderModelPerformance(models) {
    renderMetricsRow('fraud-metrics-row', models.fraud_model);
    renderMetricsRow('ews-metrics-row', models.ews_model);
    renderConfusionMatrix('chart-fraud-confusion', models.fraud_model.confusion_matrix, 'Fraud Model');
    renderConfusionMatrix('chart-ews-confusion', models.ews_model.confusion_matrix, 'EWS Model');
    renderFeatureImportance('chart-fraud-features', models.fraud_model.feature_importance);
    renderFeatureImportance('chart-ews-features', models.ews_model.feature_importance);
}

function renderMetricsRow(containerId, model) {
    const container = document.getElementById(containerId);
    const metrics = [
        { label: 'AUC-ROC', value: model.auc_roc },
        { label: 'Accuracy', value: model.accuracy },
        { label: 'Precision', value: model.precision },
        { label: 'Recall', value: model.recall },
    ];
    container.innerHTML = metrics.map(m => `
        <div class="metric-item">
            <div class="metric-value">${(m.value * 100).toFixed(1)}%</div>
            <div class="metric-label">${m.label}</div>
        </div>
    `).join('');
}

function renderConfusionMatrix(canvasId, cm, title) {
    const labels = ['True Neg', 'False Pos', 'False Neg', 'True Pos'];
    const data = [cm[0][0], cm[0][1], cm[1][0], cm[1][1]];
    const colors = [COLORS.green + '88', COLORS.orange + '88', COLORS.red + '88', COLORS.blue + '88'];

    createChart(canvasId, {
        type: 'bar',
        data: {
            labels,
            datasets: [{
                label: 'Count',
                data,
                backgroundColor: colors,
                borderColor: [COLORS.green, COLORS.orange, COLORS.red, COLORS.blue],
                borderWidth: 2,
                borderRadius: 6,
                barPercentage: 0.6,
            }],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            indexAxis: 'y',
            plugins: {
                legend: { display: false },
                title: { display: true, text: 'Confusion Matrix', color: '#8b92a8', font: { size: 12 } },
            },
            scales: {
                x: {
                    beginAtZero: true,
                    grid: { color: 'rgba(255,255,255,0.04)' },
                    ticks: { callback: v => formatNumber(v) },
                },
                y: { grid: { display: false } },
            },
        },
    });
}

function renderFeatureImportance(canvasId, importance) {
    const sorted = Object.entries(importance).sort((a, b) => b[1] - a[1]).slice(0, 10);
    const labels = sorted.map(([k]) => k.replace(/_/g, ' '));
    const data = sorted.map(([, v]) => (v * 100).toFixed(1));

    const bgColors = sorted.map((_, i) => {
        const ratio = i / (sorted.length - 1);
        return `rgba(${Math.round(0 + ratio * 123)}, ${Math.round(212 - ratio * 165)}, ${Math.round(255 - ratio * 96)}, 0.6)`;
    });

    createChart(canvasId, {
        type: 'bar',
        data: {
            labels,
            datasets: [{
                label: 'Importance (%)',
                data,
                backgroundColor: bgColors,
                borderColor: bgColors.map(c => c.replace('0.6', '1')),
                borderWidth: 1,
                borderRadius: 4,
                barPercentage: 0.65,
            }],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            indexAxis: 'y',
            plugins: { legend: { display: false } },
            scales: {
                x: {
                    beginAtZero: true,
                    grid: { color: 'rgba(255,255,255,0.04)' },
                    ticks: { callback: v => v + '%' },
                },
                y: {
                    grid: { display: false },
                    ticks: { font: { size: 11 } },
                },
            },
        },
    });
}

// ══════════════════════════════════════════════════════════════════════════
// TIMELINE
// ══════════════════════════════════════════════════════════════════════════

function renderTimeline(timeline) {
    const container = document.getElementById('timeline-container');
    container.innerHTML = timeline.phases.map((phase, idx) => {
        const statusLabel = {
            completed: '✓ Completed',
            in_progress: '◉ In Progress',
            planned: '○ Planned',
        };

        return `
            <div class="timeline-item" style="animation-delay: ${idx * 0.15}s">
                <div class="timeline-dot ${phase.status}"></div>
                <div class="timeline-card">
                    <div class="timeline-month">Bulan ${phase.month}</div>
                    <span class="timeline-status ${phase.status}">${statusLabel[phase.status]}</span>
                    <div class="timeline-title">${phase.title}</div>
                    <div class="timeline-desc">${phase.description}</div>
                    <div class="timeline-tasks">
                        ${phase.tasks.map(t => `<span class="timeline-task">${t}</span>`).join('')}
                    </div>
                </div>
            </div>
        `;
    }).join('');
}

// ══════════════════════════════════════════════════════════════════════════
// UTILITIES
// ══════════════════════════════════════════════════════════════════════════

function createChart(id, config) {
    if (chartInstances[id]) chartInstances[id].destroy();
    const ctx = document.getElementById(id);
    if (!ctx) return;
    chartInstances[id] = new Chart(ctx.getContext('2d'), config);
}

function formatNumber(n) {
    if (n >= 1_000_000) return (n / 1_000_000).toFixed(1) + 'M';
    if (n >= 1_000) return (n / 1_000).toFixed(n >= 10_000 ? 0 : 1) + 'K';
    return n.toLocaleString('id-ID');
}

function formatRupiah(amount) {
    if (amount >= 1_000_000_000_000) return 'Rp ' + (amount / 1_000_000_000_000).toFixed(1) + 'T';
    if (amount >= 1_000_000_000) return 'Rp ' + (amount / 1_000_000_000).toFixed(1) + 'B';
    if (amount >= 1_000_000) return 'Rp ' + (amount / 1_000_000).toFixed(1) + 'M';
    return 'Rp ' + amount.toLocaleString('id-ID');
}

function formatTimestamp(ts) {
    if (!ts) return '—';
    const d = new Date(ts);
    return d.toLocaleString('en-GB', {
        day: '2-digit', month: 'short',
        hour: '2-digit', minute: '2-digit',
    });
}
