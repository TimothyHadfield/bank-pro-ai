// ============================================================
// Bank AI Training Dashboard
// ============================================================

let scoreChart, policyLossChart, lossChart, evalChart, timeChart;
let lastBatch = -1;
let pollCount = 0;
let isRunning = false;
let serverMode = false;
let trainingStartTime = null;
let timerInterval = null;
let batchAtStart = 0;

// ==================== CHART SETUP ====================
let cumulativeTime = []; // tracks total elapsed time for the time chart top axis

function initCharts() {
    const hoverRadius = 20; // large hover detection radius
    const chartOpts = {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        interaction: { mode: "nearest", axis: "x", intersect: false },
        plugins: {
            legend: { labels: { color: "#8b8fa3", font: { size: 11 } } },
            tooltip: { mode: "nearest", intersect: false },
        },
        scales: {
            x: { title: { display: true, text: "Batch #", color: "#8b8fa3" }, ticks: { color: "#8b8fa3", maxTicksLimit: 10 }, grid: { color: "rgba(42,46,61,0.5)" } },
            y: { title: { display: true, text: "", color: "#8b8fa3" }, ticks: { color: "#8b8fa3" }, grid: { color: "rgba(42,46,61,0.5)" } },
        },
        elements: { point: { hitRadius: hoverRadius, hoverRadius: 6 } },
    };

    scoreChart = new Chart(document.getElementById("chart-score"), {
        type: "line",
        data: {
            labels: [],
            datasets: [
                { label: "Win Rate vs Bots", data: [], borderColor: "#34d399", borderWidth: 2, pointRadius: 0, tension: 0.2 },
                { label: "Avg Score", data: [], borderColor: "#60a5fa", borderWidth: 1.5, pointRadius: 0, tension: 0.2, yAxisID: "y1" },
            ],
        },
        options: {
            ...chartOpts,
            scales: {
                ...chartOpts.scales,
                y: { ...chartOpts.scales.y, title: { display: true, text: "Win Rate", color: "#34d399" }, min: 0, max: 1, ticks: { color: "#34d399", callback: v => (v*100).toFixed(0) + "%" } },
                y1: { position: "right", title: { display: true, text: "Score", color: "#60a5fa" }, ticks: { color: "#60a5fa" }, grid: { drawOnChartArea: false } },
            },
        },
    });

    policyLossChart = new Chart(document.getElementById("chart-ploss"), {
        type: "line",
        data: {
            labels: [],
            datasets: [
                { label: "Policy Loss", data: [], borderColor: "#60a5fa", borderWidth: 1.5, pointRadius: 0, tension: 0.2, fill: true, backgroundColor: "rgba(96,165,250,0.08)" },
            ],
        },
        options: {
            ...chartOpts,
            scales: {
                x: { ...chartOpts.scales.x },
                y: {
                    title: { display: true, text: "Policy Loss", color: "#8b8fa3" },
                    ticks: {
                        color: "#8b8fa3",
                        callback: function(v) {
                            if (Math.abs(v) < 0.01) return v.toExponential(1);
                            return v.toFixed(4);
                        }
                    },
                    grid: { color: "rgba(42,46,61,0.5)" },
                },
            },
        },
    });

    lossChart = new Chart(document.getElementById("chart-loss"), {
        type: "line",
        data: {
            labels: [],
            datasets: [
                { label: "Value Loss", data: [], borderColor: "#f87171", borderWidth: 1.5, pointRadius: 0, tension: 0.2 },
                { label: "Entropy", data: [], borderColor: "#34d399", borderWidth: 1.5, pointRadius: 0, tension: 0.2, yAxisID: "y1" },
            ],
        },
        options: {
            ...chartOpts,
            scales: {
                ...chartOpts.scales,
                y: { ...chartOpts.scales.y, title: { display: true, text: "Value Loss", color: "#8b8fa3" } },
                y1: { position: "right", title: { display: true, text: "Entropy", color: "#34d399" }, ticks: { color: "#34d399" }, grid: { drawOnChartArea: false } },
            },
        },
    });

    evalChart = new Chart(document.getElementById("chart-eval"), {
        type: "line",
        data: {
            labels: [],
            datasets: [
                { label: "vs Threshold(100)", data: [], borderColor: "#60a5fa", borderWidth: 2, pointRadius: 3, tension: 0.2 },
                { label: "vs Threshold(200)", data: [], borderColor: "#fbbf24", borderWidth: 2, pointRadius: 3, tension: 0.2 },
            ],
        },
        options: {
            ...chartOpts,
            scales: {
                ...chartOpts.scales,
                y: {
                    ...chartOpts.scales.y,
                    title: { display: true, text: "Win Rate", color: "#8b8fa3" },
                    min: 0, max: 1,
                    ticks: { ...chartOpts.scales.y.ticks, callback: v => (v * 100).toFixed(0) + "%" },
                },
            },
        },
    });

    timeChart = new Chart(document.getElementById("chart-time"), {
        type: "line",
        data: {
            labels: [],
            datasets: [
                { label: "Batch Time (s)", data: [], borderColor: "#c084fc", borderWidth: 1.5, pointRadius: 0, tension: 0.2, fill: true, backgroundColor: "rgba(192,132,252,0.08)" },
            ],
        },
        options: {
            ...chartOpts,
            scales: {
                x: { ...chartOpts.scales.x, position: "bottom" },
                x2: {
                    position: "top",
                    title: { display: true, text: "Total Time", color: "#8b8fa3" },
                    ticks: { color: "#8b8fa3", maxTicksLimit: 8, callback: function(val, idx) { return cumulativeTime[idx] || ""; } },
                    grid: { drawOnChartArea: false },
                },
                y: { ...chartOpts.scales.y, title: { display: true, text: "Seconds", color: "#8b8fa3" }, min: 0 },
            },
        },
    });
}

// ==================== TRAINING CONTROL ====================
async function startTraining() {
    const resume = document.getElementById("resume-select").value;
    // Always send resume if one is selected — never start fresh over existing data
    const body = resume ? { resume: resume } : {};

    const btn = document.getElementById("btn-start");
    btn.disabled = true;
    btn.textContent = "Starting...";

    try {
        const resp = await fetch("/api/start", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(body),
        });
        const text = await resp.text();
        let data;
        try { data = JSON.parse(text); } catch {
            alert("Server returned invalid response. Make sure you opened http://localhost:5001");
            return;
        }
        if (data.error) { alert(data.error); return; }

        trainingStartTime = Date.now();
        batchAtStart = lastBatch > 0 ? lastBatch : 0;
        startTimer();
        showBanner(true);
    } catch (e) {
        alert("Could not reach server. Make sure dashboard/server.py is running.");
    } finally {
        btn.disabled = false;
        btn.textContent = "Resume Training";
    }
}

async function pauseTraining() {
    document.getElementById("btn-pause").disabled = true;
    try {
        await fetch("/api/stop", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: "{}",
        });
    } catch (e) { console.error(e); }
    finally { document.getElementById("btn-pause").disabled = false; }
    stopTimer();
    showBanner(false);
}

async function resetTraining() {
    if (!confirm("This will stop training and clear all progress. Are you sure?")) return;

    try {
        await fetch("/api/reset", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: "{}",
        });
    } catch (e) { console.error(e); }

    stopTimer();
    showBanner(false);
    trainingStartTime = null;
    lastBatch = -1;

    // Clear charts
    [scoreChart, policyLossChart, lossChart, evalChart, timeChart].forEach(c => {
        c.data.labels = [];
        c.data.datasets.forEach(d => d.data = []);
        c.update("none");
    });

    // Clear metrics display
    ["m-winrate","m-score","m-ploss","m-vloss","m-entropy","m-wr100","m-wr200"].forEach(id => {
        setText(id, "\u2014");
    });
    document.getElementById("batch-counter").textContent = "Batch 0 / 100,000";
    document.getElementById("progress-bar").style.width = "0%";
}

async function saveTraining() {
    const name = prompt("Enter a name for this save:", "save");
    if (!name) return;

    try {
        const resp = await fetch("/api/save", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ name }),
        });
        const text = await resp.text();
        let data;
        try { data = JSON.parse(text); } catch { data = { error: text }; }
        if (data.error) {
            alert("Save failed: " + data.error);
        } else {
            alert(`Saved!\n\nName: ${name}\nBatch: ${data.batch_num}\nPath: ${data.path}`);
        }
    } catch (e) {
        alert("Save error: " + e);
    }
}

async function exportLatest() {
    try {
        const resp = await fetch("/api/checkpoints");
        const data = await resp.json();
        if (!data.checkpoints || data.checkpoints.length === 0) {
            alert("No checkpoints available. Train first.");
            return;
        }
        const latest = data.checkpoints[data.checkpoints.length - 1].filename;
        await exportCheckpoint(latest);
    } catch (e) { alert("Error: " + e); }
}

async function exportCheckpoint(filename) {
    try {
        const resp = await fetch("/api/export", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ checkpoint: filename }),
        });
        const text = await resp.text();
        let data;
        try { data = JSON.parse(text); } catch { data = { error: text }; }
        if (data.error) {
            alert("Export failed: " + data.error);
        } else {
            alert("Exported to play/weights.json!\n\n" + (data.output || ""));
        }
    } catch (e) {
        alert("Export error: " + e);
    }
}

// ==================== SETTINGS ====================
async function loadSettings() {
    try {
        const resp = await fetch("/api/overrides");
        if (!resp.ok) return;
        const text = await resp.text();
        let data;
        try { data = JSON.parse(text); } catch { return; }

        if (data.batch_size) document.getElementById("set-batch-size").value = data.batch_size;
        if (data.entropy_coefficient !== undefined) document.getElementById("set-entropy").value = data.entropy_coefficient;
        const wdCheck = document.getElementById("set-watchdog");
        const wdLabel = document.getElementById("wd-label");
        wdCheck.checked = !!data.watchdog_enabled;
        wdLabel.textContent = wdCheck.checked ? "ON" : "OFF";
    } catch (e) { /* ignore */ }
}

async function applySettings() {
    const batch_size = parseInt(document.getElementById("set-batch-size").value);
    const entropy_coefficient = parseFloat(document.getElementById("set-entropy").value);
    const watchdog_enabled = document.getElementById("set-watchdog").checked;
    document.getElementById("wd-label").textContent = watchdog_enabled ? "ON" : "OFF";

    if (isNaN(batch_size) || batch_size < 32) { alert("Games per batch must be at least 32"); return; }
    if (isNaN(entropy_coefficient) || entropy_coefficient < 0.001) { alert("Entropy coefficient must be >= 0.001"); return; }

    try {
        const resp = await fetch("/api/overrides", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ batch_size, entropy_coefficient, watchdog_enabled }),
        });
        const text = await resp.text();
        let data;
        try { data = JSON.parse(text); } catch { data = { error: text }; }
        if (data.error) {
            alert("Failed: " + data.error);
        } else {
            // Brief visual feedback
            const btn = document.querySelector(".btn-apply");
            btn.textContent = "Applied!";
            btn.style.background = "#34d399";
            setTimeout(() => { btn.textContent = "Apply Settings"; btn.style.background = ""; }, 1500);
        }
    } catch (e) { alert("Error: " + e); }
}

// ==================== TIMER ====================
function startTimer() {
    if (timerInterval) clearInterval(timerInterval);
    timerInterval = setInterval(updateTimer, 1000);
    updateTimer();
}

function stopTimer() {
    if (timerInterval) { clearInterval(timerInterval); timerInterval = null; }
    document.getElementById("elapsed-time").textContent = "";
}

function updateTimer() {
    if (!trainingStartTime) return;
    const elapsed = Math.floor((Date.now() - trainingStartTime) / 1000);
    const h = Math.floor(elapsed / 3600);
    const m = Math.floor((elapsed % 3600) / 60);
    const s = elapsed % 60;
    let str = "";
    if (h > 0) str += h + "h ";
    if (m > 0 || h > 0) str += m + "m ";
    str += s + "s";
    document.getElementById("elapsed-time").textContent = str;
}

// ==================== BANNER ====================
function showBanner(visible) {
    document.getElementById("training-banner").classList.toggle("hidden", !visible);
}

function updateBannerText(batch) {
    const detail = document.getElementById("banner-detail");
    if (!detail) return;
    if (batch > batchAtStart) {
        showBanner(false);
    } else {
        const elapsed = trainingStartTime ? Math.floor((Date.now() - trainingStartTime) / 1000) : 0;
        detail.textContent = `First results appear soon... (${elapsed}s elapsed)`;
    }
}

// ==================== POLLING ====================
async function poll() {
    pollCount++;
    try {
        const statusResp = await fetch("/api/status");
        if (statusResp.ok) {
            const text = await statusResp.text();
            try {
                const status = JSON.parse(text);
                updateStatusButtons(status);
                serverMode = true;
            } catch { serverMode = false; }
        }

        const metricsResp = await fetch("metrics.json?" + Date.now());
        if (metricsResp.ok) {
            const text = await metricsResp.text();
            try {
                const data = JSON.parse(text);
                updateDashboard(data);
            } catch { /* skip */ }
        }

        if (serverMode) {
            const logsResp = await fetch("/api/logs");
            if (logsResp.ok) {
                const text = await logsResp.text();
                try {
                    const logs = JSON.parse(text);
                    updateLogs(logs.lines);
                } catch { /* skip */ }
            }
        }

        if (pollCount % 5 === 0) { loadCheckpoints(); loadSaves(); }
        if (isRunning && trainingStartTime) updateBannerText(lastBatch);

    } catch (e) { /* ignore */ }
}

// ==================== UI UPDATES ====================
function updateStatusButtons(status) {
    const wasRunning = isRunning;
    isRunning = status.running;
    const btnStart = document.getElementById("btn-start");
    const btnPause = document.getElementById("btn-pause");

    if (status.running) {
        btnStart.classList.add("hidden");
        btnPause.classList.remove("hidden");
        if (!wasRunning && !trainingStartTime) {
            trainingStartTime = Date.now();
            batchAtStart = lastBatch > 0 ? lastBatch : 0;
            startTimer();
            showBanner(true);
        }
    } else {
        btnStart.classList.remove("hidden");
        btnPause.classList.add("hidden");
        if (wasRunning) {
            stopTimer();
            showBanner(false);
            trainingStartTime = null;
        }
    }

    const badge = document.getElementById("status-badge");
    if (status.halted) {
        badge.className = "status-badge status-halted";
        badge.textContent = "HALTED";
        showBanner(false);
        stopTimer();
    } else if (status.running) {
        badge.className = "status-badge status-running";
        badge.textContent = "TRAINING";
    }
}

function updateDashboard(data) {
    if (!data || data.error) return;

    const batch = data.current_batch || 0;
    const max = data.max_batches || 100000;
    const wd = data.watchdog || {};

    if (!isRunning && !wd.halted) {
        const badge = document.getElementById("status-badge");
        if (batch > 0) {
            badge.className = "status-badge status-paused";
            badge.textContent = "PAUSED";
        } else {
            badge.className = "status-badge status-waiting";
            badge.textContent = "WAITING";
        }
    }

    document.getElementById("batch-counter").textContent =
        `Batch ${batch.toLocaleString()} / ${max.toLocaleString()}`;
    document.getElementById("progress-bar").style.width = `${(batch / max) * 100}%`;

    const l = data.latest || {};
    setText("m-winrate", ((l.win_rate || 0) * 100).toFixed(1) + "%");
    setText("m-score", (l.avg_score || 0).toFixed(1));
    setText("m-ploss", (l.policy_loss || 0).toExponential(2));
    setText("m-vloss", (l.value_loss || 0).toFixed(4));
    setText("m-entropy", (l.entropy || 0).toFixed(4));

    const ev = data.evaluations || {};
    if (ev.win_rate_vs_100 && ev.win_rate_vs_100.length > 0)
        setText("m-wr100", (ev.win_rate_vs_100[ev.win_rate_vs_100.length - 1] * 100).toFixed(1) + "%");
    if (ev.win_rate_vs_200 && ev.win_rate_vs_200.length > 0)
        setText("m-wr200", (ev.win_rate_vs_200[ev.win_rate_vs_200.length - 1] * 100).toFixed(1) + "%");

    if (batch !== lastBatch) {
        lastBatch = batch;
        const h = data.history || {};
        if (h.batch_nums && h.batch_nums.length > 0) {
            scoreChart.data.labels = h.batch_nums;
            scoreChart.data.datasets[0].data = h.win_rate || [];
            scoreChart.data.datasets[1].data = h.avg_score;
            scoreChart.update("none");

            policyLossChart.data.labels = h.batch_nums;
            policyLossChart.data.datasets[0].data = h.policy_loss;
            policyLossChart.update("none");

            lossChart.data.labels = h.batch_nums;
            lossChart.data.datasets[0].data = h.value_loss;
            lossChart.data.datasets[1].data = h.entropy;
            lossChart.update("none");

            if (h.batch_time) {
                // Compute cumulative time labels for top axis
                let total = 0;
                cumulativeTime = h.batch_time.map(t => {
                    total += t;
                    if (total < 60) return Math.round(total) + "s";
                    if (total < 3600) return Math.round(total / 60) + "m";
                    return (total / 3600).toFixed(1) + "h";
                });
                timeChart.data.labels = h.batch_nums;
                timeChart.data.datasets[0].data = h.batch_time;
                timeChart.update("none");
            }
        }

        if (ev.batch_nums && ev.batch_nums.length > 0) {
            evalChart.data.labels = ev.batch_nums;
            evalChart.data.datasets[0].data = ev.win_rate_vs_100;
            evalChart.data.datasets[1].data = ev.win_rate_vs_200;
            evalChart.update("none");
        }
    }

    updateWatchdog(wd);
}

function updateWatchdog(wd) {
    const panel = document.getElementById("watchdog-panel");
    if (wd.halted) {
        panel.innerHTML = `<div class="wd-item halted"><span class="wd-icon">\u2717</span><span>HALTED: ${wd.halt_reason}</span></div>`;
        return;
    }
    let html = "";
    const peakOk = (wd.below_peak_count || 0) === 0;
    html += `<div class="wd-item ${peakOk ? "ok" : "warn"}"><span class="wd-icon">${peakOk ? "\u2713" : "\u26A0"}</span><span>Score: peak ${(wd.peak_score || 0).toFixed(0)}${!peakOk ? " (" + wd.below_peak_count + " below)" : ""}</span></div>`;
    const collapseOk = (wd.collapse_count || 0) === 0;
    html += `<div class="wd-item ${collapseOk ? "ok" : "warn"}"><span class="wd-icon">${collapseOk ? "\u2713" : "\u26A0"}</span><span>Policy diversity${!collapseOk ? " (" + wd.collapse_count + " repeats)" : " OK"}</span></div>`;
    html += `<div class="wd-item ok"><span class="wd-icon">\u2713</span><span>No NaN detected</span></div>`;
    panel.innerHTML = html;
}

function updateLogs(lines) {
    const el = document.getElementById("log-output");
    if (!el || lines.length === 0) return;
    const wasAtBottom = el.scrollTop + el.clientHeight >= el.scrollHeight - 20;
    el.innerHTML = lines.map(l => `<div class="log-line">${escapeHtml(l)}</div>`).join("");
    if (wasAtBottom) el.scrollTop = el.scrollHeight;
}

function escapeHtml(str) {
    return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

function setText(id, val) {
    document.getElementById(id).textContent = val;
}

// ==================== CHECKPOINTS ====================
async function loadCheckpoints() {
    try {
        const resp = await fetch("/api/checkpoints");
        if (!resp.ok) return;
        const text = await resp.text();
        let data;
        try { data = JSON.parse(text); } catch { return; }
        const list = document.getElementById("checkpoint-list");
        const select = document.getElementById("resume-select");

        if (!data.checkpoints || data.checkpoints.length === 0) {
            list.innerHTML = '<p class="text-dim">No checkpoints yet</p>';
            select.innerHTML = '<option value="">Fresh start</option>';
            return;
        }

        list.innerHTML = data.checkpoints.map(cp => `
            <div class="checkpoint-item">
                <div class="ckpt-info">
                    <span class="ckpt-name">${cp.filename}</span>
                    <span class="ckpt-meta">${cp.size_mb} MB &middot; ${cp.modified}</span>
                </div>
                <div class="ckpt-buttons">
                    <button class="btn btn-tiny" onclick="exportCheckpoint('${cp.filename}')">Export</button>
                    <button class="btn btn-tiny btn-del" onclick="deleteCheckpoint('${cp.filename}')">Del</button>
                </div>
            </div>
        `).join("");

        select.innerHTML = '<option value="">Fresh start</option>' +
            data.checkpoints.map(cp =>
                `<option value="${cp.filename}">${cp.filename}</option>`
            ).join("");
    } catch (e) { /* ignore */ }
}

async function deleteAllCheckpoints() {
    try {
        await fetch("/api/delete_all_checkpoints", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: "{}",
        });
        loadCheckpoints();
    } catch (e) { console.error(e); }
}

async function deleteCheckpoint(filename) {
    try {
        const resp = await fetch("/api/delete_checkpoint", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ filename }),
        });
        const text = await resp.text();
        try { const d = JSON.parse(text); if (d.error) { alert("Delete failed: " + d.error); return; } } catch {}
        loadCheckpoints();
    } catch (e) { alert("Delete error: " + e); }
}

async function deleteSave(name) {
    try {
        const resp = await fetch("/api/delete_save", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ name }),
        });
        const text = await resp.text();
        try { const d = JSON.parse(text); if (d.error) { alert("Delete failed: " + d.error); return; } } catch {}
        loadSaves();
    } catch (e) { alert("Delete error: " + e); }
}

async function newSession() {
    // Stop training if running
    if (isRunning) {
        await fetch("/api/stop", { method: "POST", headers: {"Content-Type":"application/json"}, body: "{}" });
    }
    // Clear metrics and reset state
    await fetch("/api/reset", { method: "POST", headers: {"Content-Type":"application/json"}, body: "{}" });

    // Reset overrides to defaults
    await fetch("/api/overrides", {
        method: "POST",
        headers: {"Content-Type":"application/json"},
        body: JSON.stringify({ batch_size: 256, entropy_coefficient: 0.03, watchdog_enabled: false }),
    });

    // Reset UI
    lastBatch = -1;
    trainingStartTime = null;
    stopTimer();
    showBanner(false);
    [scoreChart, policyLossChart, lossChart, evalChart, timeChart].forEach(c => {
        c.data.labels = [];
        c.data.datasets.forEach(d => d.data = []);
        c.update("none");
    });
    ["m-winrate","m-score","m-ploss","m-vloss","m-entropy","m-wr100","m-wr200"].forEach(id => setText(id, "\u2014"));
    document.getElementById("batch-counter").textContent = "Batch 0 / 100,000";
    document.getElementById("progress-bar").style.width = "0%";
    document.getElementById("resume-select").value = "";

    loadSettings();
    loadCheckpoints();
    loadSaves();
}

// ==================== SAVED SESSIONS ====================
async function loadSaves() {
    const list = document.getElementById("saves-list");
    if (!list) { console.error("saves-list element not found"); return; }
    try {
        const resp = await fetch("/api/saves");
        const text = await resp.text();
        console.log("saves response:", text.substring(0, 200));
        const data = JSON.parse(text);

        if (!data.saves || data.saves.length === 0) {
            list.innerHTML = '<p class="text-dim">No saved sessions</p>';
            return;
        }

        list.innerHTML = data.saves.map(s => `
            <div class="save-item">
                <div class="save-info">
                    <span class="save-name">${s.name}</span>
                    <span class="save-meta">Batch ${s.batch_num} &middot; ${s.modified}</span>
                </div>
                <div class="save-buttons">
                    <button class="btn btn-tiny btn-load" onclick="loadSave('${s.name}')">Load</button>
                    <button class="btn btn-tiny btn-del" onclick="deleteSave('${s.name}')">Del</button>
                </div>
            </div>
        `).join("");
        console.log("saves loaded:", data.saves.length);
    } catch (e) { console.error("loadSaves error:", e); list.innerHTML = '<p class="text-dim">Error: ' + e.message + '</p>'; }
}

async function loadSave(name) {
    try {
        const resp = await fetch("/api/load_save", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ name }),
        });
        const text = await resp.text();
        let data;
        try { data = JSON.parse(text); } catch { data = { error: text }; }
        if (data.error) {
            alert("Failed to load: " + data.error);
            return;
        }

        // Force refresh everything
        lastBatch = -1;
        await poll();
        await loadCheckpoints();

        // Set the resume dropdown to the restored checkpoint
        if (data.resume_checkpoint) {
            const sel = document.getElementById("resume-select");
            // Make sure the option exists
            let found = false;
            for (const opt of sel.options) {
                if (opt.value === data.resume_checkpoint) { found = true; break; }
            }
            if (!found) {
                const opt = document.createElement("option");
                opt.value = data.resume_checkpoint;
                opt.textContent = data.resume_checkpoint;
                sel.appendChild(opt);
            }
            sel.value = data.resume_checkpoint;
        }
    } catch (e) {
        alert("Load error: " + e);
    }
}

// ==================== INIT ====================
window.addEventListener("DOMContentLoaded", () => {
    initCharts();
    loadCheckpoints();
    loadSaves();
    loadSettings();
    poll();
    setInterval(poll, 2000);
});
