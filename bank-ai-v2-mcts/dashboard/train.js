// Training tab: start/stop/reset, save/load, polling for charts and log.

// jget, jpost and drawLineChart are globals from common.js (re-declaring them here
// with const stopped this whole file from loading).

const startBtn = document.getElementById("btn-start");
const stopBtn = document.getElementById("btn-stop");
const resetBtn = document.getElementById("btn-reset");
const saveBtn = document.getElementById("btn-save");
const loadBtn = document.getElementById("btn-load");
const elapsedSpan = document.getElementById("train-elapsed");
const logEl = document.getElementById("train-log");
const saveListEl = document.getElementById("save-list");

function getTrainArgs() {
    return {
        max_iterations: document.getElementById("opt-max-iter").value,
        games_per_iter: document.getElementById("opt-games-per-iter").value,
        steps_per_iter: document.getElementById("opt-steps-per-iter").value,
        sims_per_move: document.getElementById("opt-sims").value,
        batch_size: document.getElementById("opt-batch").value,
        workers: document.getElementById("opt-workers").value,
    };
}

startBtn.addEventListener("click", async () => {
    startBtn.disabled = true;
    const r = await jpost("/api/train/start", { args: getTrainArgs() });
    if (!r.ok) {
        alert("could not start: " + (r.error || "unknown"));
        startBtn.disabled = false;
        return;
    }
    setRunningUI(true);
});

stopBtn.addEventListener("click", async () => {
    stopBtn.disabled = true;
    const r = await jpost("/api/train/stop");
    if (!r.ok) {
        alert("could not stop: " + (r.error || "unknown"));
    }
    setRunningUI(false);
});

resetBtn.addEventListener("click", async () => {
    if (!confirm("Delete all checkpoints and metrics? This cannot be undone.")) return;
    const r = await jpost("/api/train/reset");
    if (!r.ok) alert("reset failed: " + (r.error || "unknown"));
    refreshCharts();
});

saveBtn.addEventListener("click", async () => {
    const name = document.getElementById("save-name").value.trim();
    if (!name) { alert("enter a save name"); return; }
    const r = await jpost("/api/save", { name });
    if (!r.ok) { alert("save failed: " + r.error); return; }
    document.getElementById("save-name").value = "";
    refreshSaves();
});

loadBtn.addEventListener("click", async () => {
    const name = saveListEl.value;
    if (!name) return;
    if (!confirm(`Load "${name}"? Current checkpoints/metrics will be replaced.`)) return;
    const r = await jpost("/api/load", { name });
    if (!r.ok) { alert("load failed: " + r.error); return; }
    refreshCharts();
});

function setRunningUI(running) {
    startBtn.disabled = running;
    stopBtn.disabled = !running;
}

async function refreshSaves() {
    const saves = await jget("/api/saves");
    saveListEl.innerHTML = "";
    if (!saves.length) {
        saveListEl.innerHTML = "<option>(no saves)</option>";
        return;
    }
    for (const s of saves) {
        const opt = document.createElement("option");
        opt.value = s.name; opt.textContent = s.name;
        saveListEl.appendChild(opt);
    }
}

async function refreshLog() {
    try {
        const r = await jget("/api/log");
        if (r.lines && r.lines.length) {
            logEl.textContent = r.lines.join("");
            logEl.scrollTop = logEl.scrollHeight;
        }
    } catch (e) {}
}

async function refreshCharts() {
    const m = await jget("/api/metrics");
    const hist = m.history || {};

    drawLineChart(document.getElementById("chart-loss"), [
        { label: "total", color: "#4a90ff", points: hist.total_loss || [] },
        { label: "policy", color: "#e6a23c", points: hist.policy_loss || [] },
        { label: "value", color: "#4cb87b", points: hist.value_loss || [] },
    ], { yMin: 0 });

    drawLineChart(document.getElementById("chart-entropy"), [
        { label: "MCTS visit entropy", color: "#a87cff", points: hist.mcts_visit_entropy || [] },
    ], { yMin: 0 });

    drawLineChart(document.getElementById("chart-buffer"), [
        { label: "buffer size", color: "#4a90ff", points: hist.buffer_size || [] },
        { label: "avg age", color: "#e85d63", points: hist.buffer_avg_age || [] },
    ], { yMin: 0 });

    drawLineChart(document.getElementById("chart-throughput"), [
        { label: "games/min", color: "#4cb87b", points: hist.games_per_min || [] },
    ], { yMin: 0 });

    drawLineChart(document.getElementById("chart-eval"), [
        { label: "overall win rate", color: "#a87cff", points: hist.eval_overall_win_rate || [] },
    ], { yMin: 0, yMax: 1 });

    // Render the latest eval breakdown
    renderEvalBreakdown(m.latest && m.latest.eval_results);
}

function renderEvalBreakdown(evalResults) {
    const el = document.getElementById("eval-breakdown");
    if (!el) return;
    if (!evalResults || !evalResults.win_rates) {
        el.innerHTML = "<span class='muted'>no evaluation yet — runs every 10 iterations</span>";
        return;
    }
    const wr = evalResults.win_rates;
    const opponents = Object.keys(wr);
    if (!opponents.length) {
        el.innerHTML = "<span class='muted'>(no opponent results yet)</span>";
        return;
    }
    // Collect all player counts seen
    const pcSet = new Set();
    opponents.forEach(op => Object.keys(wr[op]).forEach(pc => pcSet.add(pc)));
    const pcs = Array.from(pcSet).sort((a, b) => +a - +b);

    let html = `<table class="data-table"><thead><tr><th>Opponent</th>`;
    for (const pc of pcs) html += `<th>${pc}p</th>`;
    html += `<th>avg</th></tr></thead><tbody>`;
    for (const op of opponents) {
        html += `<tr><td>${op}</td>`;
        const rates = [];
        for (const pc of pcs) {
            const r = wr[op][pc];
            if (r === undefined) {
                html += `<td class="muted">—</td>`;
            } else {
                rates.push(r);
                const color = r >= 0.5 ? "var(--success)" : (r >= 0.3 ? "var(--text)" : "var(--danger)");
                html += `<td style="color:${color}">${(r*100).toFixed(1)}%</td>`;
            }
        }
        const avg = rates.length ? rates.reduce((a,b)=>a+b,0)/rates.length : 0;
        html += `<td><strong>${(avg*100).toFixed(1)}%</strong></td></tr>`;
    }
    html += `</tbody></table>`;
    if (evalResults.iteration !== undefined) {
        html += `<div class="muted" style="margin-top: 6px;">measured at iteration ${evalResults.iteration}, overall ${(evalResults.overall_win_rate*100).toFixed(2)}%</div>`;
    }
    el.innerHTML = html;
}

async function refreshStatus() {
    const s = await jget("/api/status");
    const running = s.training && s.training.running;
    setRunningUI(!!running);
    if (running) {
        const el = s.training.elapsed_seconds || 0;
        const m = Math.floor(el / 60), sec = Math.floor(el % 60);
        elapsedSpan.textContent = `running (${m}m${sec}s)`;
    } else {
        elapsedSpan.textContent = "";
    }
}

// Polling
setInterval(refreshCharts, 4000);
setInterval(refreshLog, 4000);
setInterval(refreshStatus, 2000);
refreshCharts();
refreshLog();
refreshStatus();
refreshSaves();
