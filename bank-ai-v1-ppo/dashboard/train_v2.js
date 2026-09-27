// V2 Training tab logic.
// Polls /api/v2/* endpoints to drive its own training subprocess (separate from v1).

(function () {
    "use strict";

    let v2StartTimestamp = null;
    let v2Charts = {};

    async function jget(path) {
        const r = await fetch(path, { cache: "no-store" });
        if (!r.ok) throw new Error(`${path} returned ${r.status}`);
        return r.json();
    }
    async function jpost(path, body) {
        const r = await fetch(path, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(body || {}),
            cache: "no-store",
        });
        return r.json();
    }

    let v2HasLatestCheckpoint = false;

    // ---- Phase presets ----
    const V2_PRESETS = {
        early: {
            workers: 8, sims_per_move: 15, games_per_iter: 8, steps_per_iter: 25,
            batch_size: 256, eval_interval: 25, eval_games: 2,
        },
        middle: {
            workers: 8, sims_per_move: 50, games_per_iter: 20, steps_per_iter: 50,
            batch_size: 256, eval_interval: 10, eval_games: 3,
        },
        late: {
            workers: 8, sims_per_move: 100, games_per_iter: 30, steps_per_iter: 100,
            batch_size: 256, eval_interval: 10, eval_games: 5,
        },
    };

    // Map legacy preset names so users with old saved settings don't get reset to defaults.
    const V2_PRESET_ALIASES = {
        aggressive: "early",
        balanced: "middle",
        quality: "late",
    };

    const SETTING_FIELDS = [
        ["workers", "v2-set-workers"],
        ["sims_per_move", "v2-set-sims"],
        ["games_per_iter", "v2-set-games-per-iter"],
        ["steps_per_iter", "v2-set-steps-per-iter"],
        ["batch_size", "v2-set-batch"],
        ["eval_interval", "v2-set-eval-interval"],
        ["eval_games", "v2-set-eval-games"],
    ];

    window.v2ApplyPreset = function () {
        const preset = document.getElementById("v2-set-preset").value;
        if (preset === "custom") return;  // keep current values
        const vals = V2_PRESETS[preset];
        if (!vals) return;
        for (const [k, id] of SETTING_FIELDS) {
            const el = document.getElementById(id);
            if (el && vals[k] !== undefined) el.value = vals[k];
        }
        saveV2Settings();
    };

    function saveV2Settings() {
        const settings = { preset: document.getElementById("v2-set-preset").value };
        for (const [k, id] of SETTING_FIELDS) {
            const el = document.getElementById(id);
            if (el) settings[k] = el.value;
        }
        const maxIter = document.getElementById("v2-set-max-iter");
        if (maxIter) settings.max_iterations = maxIter.value;
        try { localStorage.setItem("v2_settings", JSON.stringify(settings)); } catch (e) {}
    }

    function loadV2Settings() {
        let s;
        try { s = JSON.parse(localStorage.getItem("v2_settings") || "null"); } catch (e) { s = null; }
        if (!s) return;
        const presetSel = document.getElementById("v2-set-preset");
        if (presetSel && s.preset) {
            // Migrate legacy preset names (aggressive/balanced/quality -> early/middle/late)
            const migrated = V2_PRESET_ALIASES[s.preset] || s.preset;
            // Only set if the option exists; otherwise leave HTML default
            if ([...presetSel.options].some(o => o.value === migrated)) {
                presetSel.value = migrated;
            }
        }
        for (const [k, id] of SETTING_FIELDS) {
            const el = document.getElementById(id);
            if (el && s[k] !== undefined && s[k] !== null) el.value = s[k];
        }
        const maxIter = document.getElementById("v2-set-max-iter");
        if (maxIter && s.max_iterations) maxIter.value = s.max_iterations;
    }

    // Auto-link games/iter and steps/iter (2.5 steps per game).
    // Editing one updates the other proportionally so you only need to think about one number.
    const STEPS_PER_GAME_RATIO = 2.5;
    let _linkInProgress = false;

    function linkGamesToSteps() {
        if (_linkInProgress) return;
        _linkInProgress = true;
        const games = parseFloat(document.getElementById("v2-set-games-per-iter").value);
        if (!isNaN(games) && games > 0) {
            const steps = Math.max(10, Math.round(games * STEPS_PER_GAME_RATIO));
            document.getElementById("v2-set-steps-per-iter").value = steps;
        }
        _linkInProgress = false;
    }

    function linkStepsToGames() {
        if (_linkInProgress) return;
        _linkInProgress = true;
        const steps = parseFloat(document.getElementById("v2-set-steps-per-iter").value);
        if (!isNaN(steps) && steps > 0) {
            const games = Math.max(1, Math.round(steps / STEPS_PER_GAME_RATIO));
            document.getElementById("v2-set-games-per-iter").value = games;
        }
        _linkInProgress = false;
    }

    // Mark preset as "custom" if any setting is hand-edited; auto-link games <-> steps
    function wireSettingsAutoSave() {
        for (const [_, id] of SETTING_FIELDS) {
            const el = document.getElementById(id);
            if (!el) continue;
            el.addEventListener("change", () => {
                if (id === "v2-set-games-per-iter") linkGamesToSteps();
                else if (id === "v2-set-steps-per-iter") linkStepsToGames();
                document.getElementById("v2-set-preset").value = "custom";
                saveV2Settings();
            });
        }
        const maxIter = document.getElementById("v2-set-max-iter");
        if (maxIter) maxIter.addEventListener("change", saveV2Settings);
    }

    function getV2Args(includeResume) {
        const args = {
            max_iterations: document.getElementById("v2-set-max-iter").value,
            games_per_iter: document.getElementById("v2-set-games-per-iter").value,
            steps_per_iter: document.getElementById("v2-set-steps-per-iter").value,
            sims_per_move: document.getElementById("v2-set-sims").value,
            batch_size: document.getElementById("v2-set-batch").value,
            workers: document.getElementById("v2-set-workers").value,
            eval_interval: document.getElementById("v2-set-eval-interval").value,
            eval_games: document.getElementById("v2-set-eval-games").value,
        };
        if (includeResume) {
            args.resume = "models/latest.pt";
        }
        return args;
    }

    function updateStartButtonLabel() {
        const btn = document.getElementById("v2-btn-start");
        if (!btn) return;
        if (v2HasLatestCheckpoint) {
            btn.textContent = "Resume Training";
            btn.title = "Continue from models/latest.pt. To start fresh, click Reset first.";
        } else {
            btn.textContent = "Start Training";
            btn.title = "Start a fresh training run.";
        }
    }

    window.v2StartTraining = async function () {
        const r = await jpost("/api/v2/start", { args: getV2Args(v2HasLatestCheckpoint) });
        if (r.error) {
            alert("Could not start v2 training: " + r.error);
            return;
        }
        v2StartTimestamp = Date.now();
        setRunningUI(true);
    };

    window.v2StopTraining = async function () {
        const r = await jpost("/api/v2/stop");
        if (r.error) alert("Stop failed: " + r.error);
        setRunningUI(false);
    };

    window.v2ResetTraining = async function () {
        if (!confirm("Delete all v2 checkpoints and metrics? This cannot be undone.")) return;
        const r = await jpost("/api/v2/reset");
        if (r.error) alert("Reset failed: " + r.error);
        refreshAll();
    };

    window.v2SavePrompt = async function () {
        const name = prompt("Save name (letters, numbers, dashes/underscores):", "");
        if (!name) return;
        const r = await jpost("/api/v2/save", { name });
        if (r.error) alert("Save failed: " + r.error);
        refreshSaves();
    };

    window.v2LoadSelected = async function () {
        const sel = document.getElementById("v2-saves-select");
        const name = sel.value;
        if (!name) { alert("No save selected."); return; }
        if (!confirm(`Load "${name}"? Current v2 checkpoints/metrics will be replaced.`)) return;
        const r = await jpost("/api/v2/load_save", { name });
        if (r.error) alert("Load failed: " + r.error);
        refreshAll();
    };

    function setRunningUI(running) {
        document.getElementById("v2-btn-start").classList.toggle("hidden", running);
        document.getElementById("v2-btn-stop").classList.toggle("hidden", !running);
        const badge = document.getElementById("v2-status-badge");
        badge.textContent = running ? "RUNNING" : "WAITING";
        badge.className = "status-badge " + (running ? "status-running" : "status-waiting");
    }

    async function refreshStatus() {
        let s;
        try { s = await jget("/api/v2/status"); } catch (e) { return; }
        setRunningUI(!!s.running);
        if (s.status && s.status !== "running" && s.status !== "idle") {
            const badge = document.getElementById("v2-status-badge");
            badge.textContent = s.status.toUpperCase();
            badge.className = "status-badge status-" + s.status;
        }
        if (v2StartTimestamp && s.running) {
            const el = (Date.now() - v2StartTimestamp) / 1000;
            const m = Math.floor(el / 60), sec = Math.floor(el % 60);
            document.getElementById("v2-elapsed-time").textContent = `${m}m${sec.toString().padStart(2, "0")}s`;
        } else {
            document.getElementById("v2-elapsed-time").textContent = "";
        }
    }

    async function refreshLogs() {
        try {
            const r = await jget("/api/v2/logs");
            const el = document.getElementById("v2-log");
            if (r.lines && r.lines.length) {
                el.innerHTML = r.lines.map(line =>
                    `<div class="log-line">${escapeHtml(line)}</div>`
                ).join("");
                el.scrollTop = el.scrollHeight;
            }
        } catch (e) { /* ignore */ }
    }

    function escapeHtml(s) {
        return String(s)
            .replace(/&/g, "&amp;").replace(/</g, "&lt;")
            .replace(/>/g, "&gt;").replace(/"/g, "&quot;");
    }

    async function refreshCheckpoints() {
        try {
            const r = await jget("/api/v2/checkpoints");
            const el = document.getElementById("v2-checkpoints-list");
            const list = r.checkpoints || [];
            const hasLatest = list.some(c => c.filename === "latest.pt");
            if (hasLatest !== v2HasLatestCheckpoint) {
                v2HasLatestCheckpoint = hasLatest;
                updateStartButtonLabel();
            }
            if (!list.length) {
                el.innerHTML = "<p class='text-dim'>(none yet)</p>";
                return;
            }
            el.innerHTML = list.map(c =>
                `<div class="checkpoint-item">
                    <div class="ckpt-info">
                        <div class="ckpt-name">${escapeHtml(c.filename)}</div>
                        <div class="ckpt-meta">${escapeHtml(c.modified)} · ${c.size_mb} MB</div>
                    </div>
                </div>`
            ).join("");
        } catch (e) {}
    }

    async function refreshSaves() {
        try {
            const r = await jget("/api/v2/saves");
            const sel = document.getElementById("v2-saves-select");
            sel.innerHTML = "";
            if (!r.saves || !r.saves.length) {
                sel.innerHTML = "<option value=''>(no saved sessions)</option>";
                return;
            }
            for (const s of r.saves) {
                const opt = document.createElement("option");
                opt.value = s.name; opt.textContent = `${s.name} (${s.modified})`;
                sel.appendChild(opt);
            }
        } catch (e) {}
    }

    // Shared chart options matching v1's behavior — hover anywhere on the x-axis
    // and the tooltip shows the value at that iteration.
    function _v2ChartOpts() {
        const hoverRadius = 20;
        return {
            responsive: true,
            maintainAspectRatio: false,
            animation: false,
            interaction: { mode: "nearest", axis: "x", intersect: false },
            plugins: {
                legend: { labels: { color: "#8b8fa3", font: { size: 11 } } },
                tooltip: {
                    mode: "nearest", intersect: false,
                    callbacks: {
                        title: (items) => items.length ? `Iteration ${items[0].parsed.x}` : "",
                        label: (ctx) => `${ctx.dataset.label}: ${ctx.parsed.y.toFixed(4)}`,
                    },
                },
            },
            scales: {
                x: {
                    type: "linear",
                    title: { display: true, text: "iteration", color: "#8b8fa3" },
                    ticks: { color: "#8b8fa3", maxTicksLimit: 10 },
                    grid: { color: "rgba(42,46,61,0.5)" },
                },
                y: {
                    beginAtZero: true,
                    ticks: { color: "#8b8fa3" },
                    grid: { color: "rgba(42,46,61,0.5)" },
                },
            },
            elements: { point: { hitRadius: hoverRadius, hoverRadius: 6, radius: 0 } },
        };
    }

    function makeChart(canvasId, label, color) {
        const ctx = document.getElementById(canvasId);
        if (!ctx) return null;
        return new Chart(ctx, {
            type: "line",
            data: { datasets: [{ label, data: [], borderColor: color, backgroundColor: color + "33", tension: 0.2 }] },
            options: _v2ChartOpts(),
        });
    }
    function makeMultiChart(canvasId, series) {
        const ctx = document.getElementById(canvasId);
        if (!ctx) return null;
        return new Chart(ctx, {
            type: "line",
            data: {
                datasets: series.map(s => ({
                    label: s.label, data: [], borderColor: s.color,
                    backgroundColor: s.color + "33", tension: 0.2,
                })),
            },
            options: _v2ChartOpts(),
        });
    }

    function ensureCharts() {
        if (v2Charts.loss) return;
        v2Charts.loss = makeMultiChart("v2-chart-loss", [
            { label: "total", color: "#4a90ff" },
            { label: "policy", color: "#e6a23c" },
            { label: "value", color: "#4cb87b" },
        ]);
        v2Charts.entropy = makeChart("v2-chart-entropy", "MCTS visit entropy", "#a87cff");
        v2Charts.eval = makeMultiChart("v2-chart-eval", [
            { label: "2p", color: "#4a90ff" },
            { label: "3p", color: "#4cb87b" },
            { label: "4p", color: "#e6a23c" },
            { label: "5p", color: "#a87cff" },
            { label: "6p", color: "#e85d63" },
            { label: "overall", color: "#8b8fa3" },  // backstop — always present, even from old data
        ]);
        v2Charts.throughput = makeChart("v2-chart-throughput", "games/min", "#4cb87b");
    }

    function setSeries(chart, datasetIdx, points) {
        if (!chart) return;
        chart.data.datasets[datasetIdx].data = (points || []).map(([x, y]) => ({ x, y }));
        chart.update("none");
    }

    async function refreshMetrics() {
        ensureCharts();
        let m;
        try { m = await jget("/api/v2/metrics"); } catch (e) { return; }
        const hist = m.history || {};
        setSeries(v2Charts.loss, 0, hist.total_loss);
        setSeries(v2Charts.loss, 1, hist.policy_loss);
        setSeries(v2Charts.loss, 2, hist.value_loss);
        setSeries(v2Charts.entropy, 0, hist.mcts_visit_entropy);
        setSeries(v2Charts.eval, 0, hist.eval_win_rate_2p);
        setSeries(v2Charts.eval, 1, hist.eval_win_rate_3p);
        setSeries(v2Charts.eval, 2, hist.eval_win_rate_4p);
        setSeries(v2Charts.eval, 3, hist.eval_win_rate_5p);
        setSeries(v2Charts.eval, 4, hist.eval_win_rate_6p);
        setSeries(v2Charts.eval, 5, hist.eval_overall_win_rate);
        setSeries(v2Charts.throughput, 0, hist.games_per_min);

        const latest = m.latest || {};
        if (latest.iteration) {
            document.getElementById("v2-iter-counter").textContent = `Iteration ${latest.iteration}`;
        }
        renderEvalBreakdown(latest.eval_results);
        renderPhaseEstimates(hist.games_per_min);
    }

    // Approximate number of games typically needed to complete each phase.
    // These are rough — real numbers depend heavily on luck, opponent mix, and tuning.
    const PHASE_GAME_BUDGETS = {
        early: 1500,    // network learns basic pot threshold + round number
        middle: 10000,  // refines round-aware play, opponent-rank reactions
        late: 30000,    // polishes nuances toward the ceiling
    };

    // Multiplicative slowdown vs an untrained AI: smarter AIs play longer rounds
    // (more decisions per game = more MCTS calls = lower games/min). Calibrated to
    // ~15% slower at mid-training and ~30% slower at late-training.
    const SMARTNESS_DRAG = {
        early: 1.00,
        middle: 1.15,
        late: 1.30,
    };

    function renderPhaseEstimates(gpmHistory) {
        const el = document.getElementById("v2-phase-estimates");
        if (!el) return;
        // Average the last N games/min readings for stability
        const recent = (gpmHistory || []).slice(-10).map(p => p[1]).filter(v => isFinite(v) && v > 0);
        if (recent.length === 0) {
            el.innerHTML = "<span class='text-dim'>Start training to see estimates based on actual games/min.</span>";
            return;
        }
        const avgGpmCurrent = recent.reduce((a, b) => a + b, 0) / recent.length;

        // Determine current sims/move so we can scale to each phase's expected speed.
        // Speed scales roughly inversely with sims_per_move (workers are constant across phases).
        const currentSims = parseFloat(document.getElementById("v2-set-sims").value) || 50;

        // For each phase, compute expected games/min (scaling for both sims AND
        // the AI getting slower as it learns), iteration count, and time budget.
        const rows = [];
        let totalSeconds = 0;
        let totalIters = 0;
        for (const [phase, budget] of Object.entries(PHASE_GAME_BUDGETS)) {
            const phasePreset = V2_PRESETS[phase];
            const phaseSims = phasePreset.sims_per_move;
            const phaseGamesPerIter = phasePreset.games_per_iter;
            const drag = SMARTNESS_DRAG[phase] || 1.0;
            // Speed scales inversely with sims and with the smartness drag
            const phaseGpm = avgGpmCurrent * (currentSims / phaseSims) / drag;
            const seconds = (budget / Math.max(0.01, phaseGpm)) * 60;
            // Iterations needed for this phase (using that phase's games/iter setting)
            const iters = Math.ceil(budget / phaseGamesPerIter);
            totalSeconds += seconds;
            totalIters += iters;
            rows.push({ phase, budget, gpm: phaseGpm, drag, seconds, iters });
        }

        let html = `<div style="display:grid;grid-template-columns:auto 1fr auto auto;gap:8px 14px;align-items:center;font-size:0.85rem;">`;
        html += `<div style="grid-column:1/-1;color:var(--text-dim);font-size:0.75rem;">At current speed (${avgGpmCurrent.toFixed(1)} games/min on ${currentSims}-sim settings). Includes smartness drag — smarter AIs play longer rounds.</div>`;
        for (const { phase, budget, gpm, drag, seconds, iters } of rows) {
            const label = phase.charAt(0).toUpperCase() + phase.slice(1) + " phase";
            const dragLabel = drag > 1.01 ? ` <span style="opacity:0.7;">×${drag.toFixed(2)} drag</span>` : "";
            html += `<div><b>${label}</b></div>`;
            html += `<div class='text-dim'>${iters.toLocaleString()} iters · ~${budget.toLocaleString()} games · ${gpm.toFixed(1)} g/min${dragLabel}</div>`;
            html += `<div></div>`;
            html += `<div><b>${formatDuration(seconds)}</b></div>`;
        }
        html += `<div style="grid-column:1/-1;border-top:1px solid var(--border);margin-top:4px;padding-top:6px;display:flex;justify-content:space-between;font-size:0.85rem;">`;
        html += `<span><b>Total (sequential)</b> <span class='text-dim'>${totalIters.toLocaleString()} iters</span></span>`;
        html += `<b style='color:var(--accent);'>${formatDuration(totalSeconds)}</b>`;
        html += `</div>`;
        html += `</div>`;
        html += `<div class='text-dim' style='font-size:0.7rem;margin-top:8px;'>Iteration counts assume you use each phase's preset settings. Move on when the loss curve flattens.</div>`;
        el.innerHTML = html;
    }

    function formatDuration(seconds) {
        if (!isFinite(seconds) || seconds <= 0) return "—";
        if (seconds < 60) return `${Math.round(seconds)} sec`;
        if (seconds < 3600) return `${(seconds / 60).toFixed(1)} min`;
        if (seconds < 86400) return `${(seconds / 3600).toFixed(1)} hr`;
        return `${(seconds / 86400).toFixed(1)} days`;
    }

    function renderEvalBreakdown(eval_results) {
        const el = document.getElementById("v2-eval-breakdown");
        if (!eval_results || !eval_results.win_rates) {
            el.innerHTML = "<span class='muted'>Eval runs every 10 iterations.</span>";
            return;
        }
        const wr = eval_results.win_rates;
        const opps = Object.keys(wr);
        if (!opps.length) { el.innerHTML = "<span class='muted'>(no opponent results yet)</span>"; return; }
        const pcSet = new Set();
        opps.forEach(op => Object.keys(wr[op]).forEach(pc => pcSet.add(pc)));
        const pcs = Array.from(pcSet).sort((a, b) => +a - +b);
        let html = `<table style="width:100%; border-collapse:collapse;"><thead><tr><th style="text-align:left">Opponent</th>`;
        for (const pc of pcs) html += `<th>${pc}p</th>`;
        html += `<th>avg</th></tr></thead><tbody>`;
        for (const op of opps) {
            html += `<tr><td>${op}</td>`;
            const rates = [];
            for (const pc of pcs) {
                const r = wr[op][pc];
                if (r === undefined) html += `<td class='muted'>—</td>`;
                else { rates.push(r); html += `<td>${(r*100).toFixed(1)}%</td>`; }
            }
            const avg = rates.length ? rates.reduce((a,b)=>a+b,0)/rates.length : 0;
            html += `<td><strong>${(avg*100).toFixed(1)}%</strong></td></tr>`;
        }
        html += `</tbody></table>`;
        if (eval_results.iteration !== undefined) {
            html += `<div class="muted" style="margin-top:6px;">measured at iteration ${eval_results.iteration}, overall ${((eval_results.overall_win_rate||0)*100).toFixed(2)}%</div>`;
        }
        el.innerHTML = html;
    }

    function refreshAll() {
        refreshStatus();
        refreshLogs();
        refreshCheckpoints();
        refreshSaves();
        refreshMetrics();
    }

    setInterval(refreshStatus, 2000);
    setInterval(refreshLogs, 4000);
    setInterval(refreshMetrics, 4000);
    setInterval(refreshCheckpoints, 8000);
    setInterval(refreshSaves, 10000);
    refreshAll();
    loadV2Settings();
    wireSettingsAutoSave();
})();
