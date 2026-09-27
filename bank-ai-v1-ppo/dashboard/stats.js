// ============================================================
// Game Statistics — simulate games, show probability distributions
// ============================================================

let statsCharts = {};
let statsData = null;
let statsNumGames = 0;

// Per-chart settings
const chartSettings = {
    maxPoints:  { smooth: false, compact: true, percentile: 95, logX: true },
    maxRound:   { smooth: false, compact: true, percentile: 95, logX: true },
    bestRound:  { smooth: false, compact: true, percentile: 95, logX: true },
    rollOf7:    { smooth: false, compact: false },
    immPoints:  { smooth: false, compact: false },
};

function runStats() {
    const numGames = parseInt(document.getElementById("stats-num-games").value);
    if (isNaN(numGames) || numGames < 100) { alert("Minimum 100 games"); return; }

    document.getElementById("stats-run-btn").disabled = true;
    document.getElementById("stats-status").classList.remove("hidden");
    document.getElementById("stats-results").classList.add("hidden");
    document.getElementById("stats-status-text").textContent = `Simulating ${numGames.toLocaleString()} games...`;

    setTimeout(() => {
        statsData = simulateGames(numGames);
        statsNumGames = numGames;
        renderStats();
        document.getElementById("stats-run-btn").disabled = false;
        document.getElementById("stats-status").classList.add("hidden");
        document.getElementById("stats-results").classList.remove("hidden");
    }, 50);
}

function toggleSmooth(chartKey) {
    chartSettings[chartKey].smooth = !chartSettings[chartKey].smooth;
    if (statsData) renderStats();
}

function toggleLogX(chartKey) {
    chartSettings[chartKey].logX = !chartSettings[chartKey].logX;
    if (statsData) renderStats();
}

function setRange(chartKey, mode) {
    if (mode === "compact") {
        chartSettings[chartKey].compact = true;
    } else {
        chartSettings[chartKey].compact = false;
    }
    if (statsData) renderStats();
}

function setPercentile(chartKey, val) {
    chartSettings[chartKey].percentile = Math.max(50, Math.min(100, parseInt(val)));
    if (statsData) renderStats();
}

function simulateGames(numGames) {
    const maxPointsPerGame = [];
    const maxPointsPerRound = [];
    const bestRoundPerGame = [];
    const rollOf7 = [];
    const pointsAfterImmunity = [];
    const singleDieValues = [];
    const diceTotals = [];
    const potByRoll = {};

    for (let g = 0; g < numGames; g++) {
        let gameMaxPoints = 0;
        for (let round = 0; round < 10; round++) {
            let pot = 0, rollNum = 0, maxPotThisRound = 0;
            while (true) {
                rollNum++;
                const d1 = Math.floor(Math.random() * 6) + 1;
                const d2 = Math.floor(Math.random() * 6) + 1;
                const sum = d1 + d2;
                singleDieValues.push(d1);
                singleDieValues.push(d2);
                diceTotals.push(sum);
                if (rollNum <= 3) {
                    pot += (sum === 7) ? 70 : sum;
                    if (rollNum === 3) pointsAfterImmunity.push(pot);
                } else {
                    if (sum === 7) { rollOf7.push(rollNum); break; }
                    else if (d1 === d2) pot *= 2;
                    else pot += sum;
                }
                // Track pot at each roll number
                if (!potByRoll[rollNum]) potByRoll[rollNum] = [];
                potByRoll[rollNum].push(pot);
                if (rollNum >= 3) maxPotThisRound = Math.max(maxPotThisRound, pot);
            }
            maxPointsPerRound.push(maxPotThisRound);
            gameMaxPoints += maxPotThisRound;
        }
        maxPointsPerGame.push(gameMaxPoints);
        bestRoundPerGame.push(Math.max(...maxPointsPerRound.slice(-10)));
    }
    return { maxPointsPerGame, maxPointsPerRound, bestRoundPerGame, rollOf7, pointsAfterImmunity, singleDieValues, diceTotals, potByRoll };
}

function renderStats() {
    const data = statsData;
    const numGames = statsNumGames;
    const avg = arr => arr.reduce((a, b) => a + b, 0) / arr.length;
    const med = arr => { const s = [...arr].sort((a,b) => a-b); const m = Math.floor(s.length/2); return s.length%2 ? s[m] : (s[m-1]+s[m])/2; };

    document.getElementById("stats-summary").innerHTML = `
        <div class="stats-summary-grid">
            <div class="stat-card"><span class="stat-label">Games</span><span class="stat-val">${numGames.toLocaleString()}</span></div>
            <div class="stat-card"><span class="stat-label">Rounds</span><span class="stat-val">${(numGames * 10).toLocaleString()}</span></div>
            <div class="stat-card"><span class="stat-label">Avg Max Pts/Game</span><span class="stat-val">${avg(data.maxPointsPerGame).toFixed(0)}</span></div>
            <div class="stat-card"><span class="stat-label">Avg Max Pts/Round</span><span class="stat-val">${avg(data.maxPointsPerRound).toFixed(0)}</span></div>
            <div class="stat-card"><span class="stat-label">Avg Roll # of 7</span><span class="stat-val">${avg(data.rollOf7).toFixed(1)}</span></div>
            <div class="stat-card"><span class="stat-label">Avg Points After Immunity</span><span class="stat-val">${avg(data.pointsAfterImmunity).toFixed(0)}</span></div>
        </div>
    `;

    Object.values(statsCharts).forEach(c => { if (c && c.destroy) c.destroy(); });
    statsCharts = {};

    // Render per-chart controls + charts
    renderFlexChartWithControls("maxPoints", "chart-max-points", "ctrl-max-points", data.maxPointsPerGame, "Max Points", "#60a5fa");
    renderFlexChartWithControls("maxRound", "chart-max-round", "ctrl-max-round", data.maxPointsPerRound, "Max Points", "#818cf8");
    renderFlexChartWithControls("bestRound", "chart-best-round", "ctrl-best-round", data.bestRoundPerGame, "Points", "#f472b6");
    renderDiscreteChartWithControls("rollOf7", "chart-rolls-before-7", "ctrl-rolls7", data.rollOf7, "Roll #", "#f87171");
    renderFlexChartWithControls("immPoints", "chart-imm-points", "ctrl-imm-points", data.pointsAfterImmunity, "Points", "#34d399");

    // Survival rate chart
    statsCharts.survival = makeSurvivalChart("chart-survival", data.rollOf7);

    // Avg pot by roll number
    statsCharts.avgPot = makeAvgPotChart("chart-avg-pot", data.potByRoll);

    // Expected points by roll #
    statsCharts.expectedPts = makeExpectedPointsChart("chart-expected-pts", data.rollOf7, data.potByRoll);

    // P(round >= X) + Expected Points overlay
    statsCharts.pointSurvival = makePointSurvivalOverlayChart("chart-point-survival", data.maxPointsPerRound);

    // Dice — always discrete, no controls
    statsCharts.singleDie = makeDiscreteChart("chart-single-die", data.singleDieValues, "Die Value", "#fbbf24", 1, 6);
    statsCharts.diceTotal = makeDiscreteChart("chart-dice-total", data.diceTotals, "Total", "#c084fc", 2, 12);
}

function renderFlexChartWithControls(key, canvasId, ctrlId, values, xLabel, color) {
    const s = chartSettings[key];
    const ctrl = document.getElementById(ctrlId);
    if (ctrl) {
        let html = '';
        html += `<button class="btn btn-ctl ${s.compact ? 'active' : ''}" onclick="setRange('${key}','compact')">Compact</button>`;
        html += `<button class="btn btn-ctl ${!s.compact ? 'active' : ''}" onclick="setRange('${key}','all')">All Data</button>`;
        html += `<button class="btn btn-ctl ${s.smooth ? 'active' : ''}" onclick="toggleSmooth('${key}')">Smooth</button>`;
        if (s.logX !== undefined) {
            html += `<button class="btn btn-ctl ${s.logX ? 'active' : ''}" onclick="toggleLogX('${key}')">Log Scale</button>`;
        }
        if (s.compact && s.percentile !== undefined) {
            html += `<span class="ctl-pct-label">Range:</span>`;
            html += `<input type="range" class="ctl-slider" min="50" max="100" value="${s.percentile}" oninput="setPercentile('${key}', this.value); document.getElementById('pct-${key}').textContent=this.value+'%'">`;
            html += `<span class="ctl-pct-val" id="pct-${key}">${s.percentile}%</span>`;
        }
        ctrl.innerHTML = html;
    }
    statsCharts[key] = makeFlexChart(canvasId, values, xLabel, color, s.smooth, s.compact, s.percentile || 95, s.logX || false);
}

function renderDiscreteChartWithControls(key, canvasId, ctrlId, values, xLabel, color) {
    const s = chartSettings[key];
    const ctrl = document.getElementById(ctrlId);
    if (ctrl) {
        let html = '';
        html += `<button class="btn btn-ctl ${s.compact ? 'active' : ''}" onclick="setRange('${key}','compact')">Compact</button>`;
        html += `<button class="btn btn-ctl ${!s.compact ? 'active' : ''}" onclick="setRange('${key}','all')">All Data</button>`;
        html += `<button class="btn btn-ctl ${s.smooth ? 'active' : ''}" onclick="toggleSmooth('${key}')">Smooth</button>`;
        ctrl.innerHTML = html;
    }
    statsCharts[key] = makeDiscreteChartFlex(canvasId, values, xLabel, color, s.smooth, s.compact);
}

function getBaseOpts() {
    return {
        responsive: true, maintainAspectRatio: false, animation: false,
        interaction: { mode: "index", intersect: false },
        plugins: { legend: { display: false }, tooltip: { mode: "index", intersect: false } },
        scales: {
            x: { title: { display: true, color: "#8b8fa3" }, ticks: { color: "#8b8fa3", maxRotation: 0, maxTicksLimit: 20 }, grid: { color: "rgba(42,46,61,0.3)" } },
            y: { title: { display: true, text: "Frequency %", color: "#8b8fa3" }, ticks: { color: "#8b8fa3", callback: v => v.toFixed(1) + "%" }, grid: { color: "rgba(42,46,61,0.3)" } },
        },
    };
}

function makeFlexChart(canvasId, values, xLabel, color, smooth, compact, percentile, logScale) {
    const sorted = [...values].sort((a, b) => a - b);
    let lo, hi;
    if (compact) {
        const tail = (100 - percentile) / 200;
        lo = sorted[Math.floor(sorted.length * tail)];
        hi = sorted[Math.floor(sorted.length * (1 - tail))];
    } else {
        lo = sorted[0];
        hi = sorted[sorted.length - 1];
    }
    lo = Math.max(1, lo); // avoid log(0)

    let labels, pcts;

    if (logScale) {
        // Logarithmic bins: edges grow exponentially
        const logLo = Math.log10(lo);
        const logHi = Math.log10(hi);
        const numBins = 40;
        const binEdges = [];
        for (let i = 0; i <= numBins; i++) {
            binEdges.push(Math.round(Math.pow(10, logLo + (logHi - logLo) * i / numBins)));
        }
        // Deduplicate edges
        const edges = [...new Set(binEdges)].sort((a, b) => a - b);

        const bins = new Array(edges.length - 1).fill(0);
        for (const v of values) {
            if (v < lo || v > hi) continue;
            // Binary search for bin
            let b = 0;
            for (let i = edges.length - 2; i >= 0; i--) {
                if (v >= edges[i]) { b = i; break; }
            }
            if (b < bins.length) bins[b]++;
        }

        labels = [];
        pcts = [];
        for (let i = 0; i < edges.length - 1; i++) {
            labels.push(edges[i]);
            pcts.push((bins[i] / values.length) * 100);
        }
    } else {
        // Linear bins
        const range = hi - lo;
        let binWidth = Math.max(1, Math.ceil(range / 40));
        if (binWidth > 200) binWidth = Math.ceil(binWidth / 100) * 100;
        else if (binWidth > 50) binWidth = Math.ceil(binWidth / 50) * 50;
        else if (binWidth > 10) binWidth = Math.ceil(binWidth / 10) * 10;
        else if (binWidth > 5) binWidth = Math.ceil(binWidth / 5) * 5;

        const binStart = Math.floor(lo / Math.max(1, binWidth)) * binWidth;
        const bins = {};
        for (const v of values) {
            if (v < lo || v > hi) continue;
            const bin = Math.floor((v - binStart) / binWidth) * binWidth + binStart;
            bins[bin] = (bins[bin] || 0) + 1;
        }

        labels = [];
        pcts = [];
        for (let b = binStart; b <= hi + binWidth; b += binWidth) {
            if (b >= lo - binWidth && b <= hi + binWidth) {
                labels.push(b);
                pcts.push(((bins[b] || 0) / values.length) * 100);
            }
        }
    }

    const opts = getBaseOpts();
    opts.scales.x.title = { display: true, text: xLabel + (logScale ? " (log scale)" : ""), color: "#8b8fa3" };

    if (smooth) {
        return new Chart(document.getElementById(canvasId), {
            type: "line",
            data: { labels, datasets: [{ data: pcts, borderColor: color, borderWidth: 2, pointRadius: 0, tension: 0.3, fill: true, backgroundColor: color + "22" }] },
            options: opts,
        });
    }
    return new Chart(document.getElementById(canvasId), {
        type: "bar",
        data: { labels, datasets: [{ data: pcts, backgroundColor: color + "66", borderColor: color, borderWidth: 1 }] },
        options: opts,
    });
}

function makeDiscreteChartFlex(canvasId, values, xLabel, color, smooth, compact) {
    const counts = {};
    for (const v of values) counts[v] = (counts[v] || 0) + 1;
    let keys = Object.keys(counts).map(Number).sort((a, b) => a - b);

    if (compact) {
        let cumul = 0;
        const p025 = values.length * 0.025, p975 = values.length * 0.975;
        let lo = keys[0], hi = keys[keys.length - 1];
        cumul = 0; for (const k of keys) { cumul += counts[k] || 0; if (cumul >= p025) { lo = k; break; } }
        cumul = 0; for (const k of keys) { cumul += counts[k] || 0; if (cumul >= p975) { hi = k; break; } }
        keys = keys.filter(k => k >= lo && k <= hi);
    }

    const pcts = keys.map(k => ((counts[k] || 0) / values.length) * 100);
    const opts = getBaseOpts();
    opts.scales.x.title = { display: true, text: xLabel, color: "#8b8fa3" };

    if (smooth) {
        return new Chart(document.getElementById(canvasId), {
            type: "line",
            data: { labels: keys, datasets: [{ data: pcts, borderColor: color, borderWidth: 2, pointRadius: 0, tension: 0.3, fill: true, backgroundColor: color + "22" }] },
            options: opts,
        });
    }
    return new Chart(document.getElementById(canvasId), {
        type: "bar",
        data: { labels: keys, datasets: [{ data: pcts, backgroundColor: color + "66", borderColor: color, borderWidth: 1 }] },
        options: opts,
    });
}

function makeSurvivalChart(canvasId, rollOf7Values) {
    // For each roll number, what % of rounds have NOT yet been busted?
    // Roll 3 = 100% (immunity just ended, no 7 possible yet)
    // Roll N = % of rounds where the 7 hit on roll N+1 or later
    const total = rollOf7Values.length;
    const counts = {};
    for (const v of rollOf7Values) counts[v] = (counts[v] || 0) + 1;

    const maxRoll = Math.max(...rollOf7Values);
    const labels = [];
    const survival = [];

    let survived = total; // everyone starts alive after roll 3
    for (let r = 3; r <= maxRoll; r++) {
        labels.push(r);
        survival.push((survived / total) * 100);
        // After this roll, those who got a 7 on roll r+1 die
        survived -= (counts[r + 1] || 0);
    }

    // Only show rolls where survival > 0.1%
    let cutoff = labels.length;
    for (let i = labels.length - 1; i >= 0; i--) {
        if (survival[i] >= 0.1) { cutoff = i + 1; break; }
    }

    const opts = getBaseOpts();
    opts.scales.x.title = { display: true, text: "After Roll #", color: "#8b8fa3" };
    opts.scales.y.title = { display: true, text: "Survival %", color: "#8b8fa3" };
    opts.scales.y.ticks = { color: "#8b8fa3", callback: v => v.toFixed(0) + "%" };
    opts.scales.y.min = 0;
    opts.scales.y.max = 100;

    return new Chart(document.getElementById(canvasId), {
        type: "line",
        data: {
            labels: labels.slice(0, cutoff),
            datasets: [{
                data: survival.slice(0, cutoff),
                borderColor: "#f59e0b",
                borderWidth: 2.5,
                pointRadius: 3,
                pointBackgroundColor: "#f59e0b",
                tension: 0.2,
                fill: true,
                backgroundColor: "rgba(245,158,11,0.1)",
            }]
        },
        options: opts,
    });
}

function makePointSurvivalOverlayChart(canvasId, maxPerRound) {
    const sorted = [...maxPerRound].sort((a, b) => a - b);
    const total = sorted.length;

    // Build points at logarithmic intervals
    const max = sorted[Math.floor(total * 0.995)];
    const logMax = Math.log10(Math.max(10, max));
    const numPoints = 100;
    const labels = [];
    const survivalPcts = [];
    const expectedPts = [];

    for (let i = 0; i <= numPoints; i++) {
        const pts = i === 0 ? 0 : Math.round(Math.pow(10, (logMax * i / numPoints)));
        let lo = 0, hi = total;
        while (lo < hi) {
            const mid = (lo + hi) >> 1;
            if (sorted[mid] < pts) lo = mid + 1;
            else hi = mid;
        }
        const survivalRate = (total - lo) / total; // 0 to 1
        labels.push(pts);
        survivalPcts.push(survivalRate * 100);
        // Expected points = points * survival_rate * 10
        expectedPts.push(pts * survivalRate * 10);
    }

    // Find the point where survival crosses 10%
    let p10Index = -1;
    for (let i = 0; i < survivalPcts.length; i++) {
        if (survivalPcts[i] <= 10) { p10Index = i; break; }
    }
    const p10Label = p10Index >= 0 ? labels[p10Index] : null;

    const opts = getBaseOpts();
    opts.scales.x.title = { display: true, text: "Points", color: "#8b8fa3" };
    opts.scales.y = {
        position: "left",
        title: { display: true, text: "P(round >= X) %", color: "#2dd4bf" },
        ticks: { color: "#2dd4bf", callback: v => v.toFixed(0) + "%" },
        grid: { color: "rgba(42,46,61,0.3)" },
        min: 0, max: 100,
    };
    opts.scales.y1 = {
        position: "right",
        title: { display: true, text: "Expected Points (x * P * 10)", color: "#e879f9" },
        ticks: { color: "#e879f9", callback: v => Math.round(v) },
        grid: { drawOnChartArea: false },
    };

    // Vertical line plugin for the 10% marker
    const verticalLinePlugin = {
        id: "p10line",
        afterDraw(chart) {
            if (p10Index < 0) return;
            const xScale = chart.scales.x;
            const yScale = chart.scales.y;
            const x = xScale.getPixelForValue(p10Index);
            const ctx = chart.ctx;
            ctx.save();
            ctx.beginPath();
            ctx.setLineDash([6, 4]);
            ctx.strokeStyle = "#f87171";
            ctx.lineWidth = 1.5;
            ctx.moveTo(x, yScale.top);
            ctx.lineTo(x, yScale.bottom);
            ctx.stroke();
            // Label
            ctx.setLineDash([]);
            ctx.fillStyle = "#f87171";
            ctx.font = "bold 11px sans-serif";
            ctx.textAlign = "left";
            const midY = (yScale.top + yScale.bottom) / 2;
            ctx.fillText(`10% @ ${p10Label} pts`, x + 6, midY);
            ctx.restore();
        }
    };

    return new Chart(document.getElementById(canvasId), {
        type: "line",
        data: {
            labels,
            datasets: [
                {
                    label: "P(round >= X)",
                    data: survivalPcts,
                    borderColor: "#2dd4bf",
                    borderWidth: 2.5,
                    pointRadius: 0,
                    tension: 0.2,
                    fill: true,
                    backgroundColor: "rgba(45,212,191,0.08)",
                    yAxisID: "y",
                },
                {
                    label: "Expected Points",
                    data: expectedPts,
                    borderColor: "#e879f9",
                    borderWidth: 2.5,
                    pointRadius: 0,
                    tension: 0.2,
                    fill: false,
                    yAxisID: "y1",
                },
            ]
        },
        options: opts,
        plugins: [verticalLinePlugin],
    });
}

function makeExpectedPointsChart(canvasId, rollOf7Values, potByRoll) {
    // Compute survival rate at each roll
    const totalRounds = rollOf7Values.length;
    const counts = {};
    for (const v of rollOf7Values) counts[v] = (counts[v] || 0) + 1;

    const rolls = Object.keys(potByRoll).map(Number).sort((a, b) => a - b);
    const minSamples = (potByRoll[1] ? potByRoll[1].length : 1000) * 0.001;
    const filtered = rolls.filter(r => potByRoll[r].length >= minSamples);

    // Build survival: % still alive after each roll
    let survived = totalRounds;
    const survivalByRoll = {};
    for (let r = 3; r <= Math.max(...filtered); r++) {
        survivalByRoll[r] = survived / totalRounds;
        survived -= (counts[r + 1] || 0);
    }

    // Expected points = (1 / survival_rate) / 10 * avg_pot
    const labels = [];
    const values = [];
    for (const r of filtered) {
        if (r < 3) continue;
        const sr = survivalByRoll[r];
        if (!sr || sr <= 0) continue;
        const avg = potByRoll[r].reduce((a, b) => a + b, 0) / potByRoll[r].length;
        const expectedPts = (1 / sr) / 10 * avg;
        labels.push(r);
        values.push(expectedPts);
    }

    const opts = getBaseOpts();
    opts.scales.x.title = { display: true, text: "Roll #", color: "#8b8fa3" };
    opts.scales.y = {
        type: "logarithmic",
        title: { display: true, text: "Expected Points (log)", color: "#8b8fa3" },
        ticks: { color: "#8b8fa3", callback: v => Number.isInteger(Math.log10(v)) || v < 100 ? Math.round(v) : "" },
        grid: { color: "rgba(42,46,61,0.3)" },
        min: 1,
    };

    return new Chart(document.getElementById(canvasId), {
        type: "line",
        data: {
            labels,
            datasets: [{
                data: values,
                borderColor: "#e879f9",
                borderWidth: 2.5,
                pointRadius: 3,
                pointBackgroundColor: "#e879f9",
                tension: 0.2,
                fill: true,
                backgroundColor: "rgba(232,121,249,0.1)",
            }]
        },
        options: opts,
    });
}

function makeAvgPotChart(canvasId, potByRoll) {
    const rolls = Object.keys(potByRoll).map(Number).sort((a, b) => a - b);
    // Only show rolls with meaningful sample size (> 0.1% of roll 1 count)
    const minSamples = (potByRoll[1] ? potByRoll[1].length : 1000) * 0.001;
    const filtered = rolls.filter(r => potByRoll[r].length >= minSamples);

    const avgPots = filtered.map(r => {
        const vals = potByRoll[r];
        return vals.reduce((a, b) => a + b, 0) / vals.length;
    });

    const opts = getBaseOpts();
    opts.scales.x.title = { display: true, text: "Roll #", color: "#8b8fa3" };
    opts.scales.y = {
        type: "logarithmic",
        title: { display: true, text: "Avg Pot (log)", color: "#8b8fa3" },
        ticks: { color: "#8b8fa3", callback: v => Number.isInteger(Math.log10(v)) || v < 100 ? Math.round(v) : "" },
        grid: { color: "rgba(42,46,61,0.3)" },
        min: 1,
    };

    return new Chart(document.getElementById(canvasId), {
        type: "line",
        data: {
            labels: filtered,
            datasets: [{
                data: avgPots,
                borderColor: "#fb923c",
                borderWidth: 2.5,
                pointRadius: 3,
                pointBackgroundColor: "#fb923c",
                tension: 0.2,
                fill: true,
                backgroundColor: "rgba(251,146,60,0.1)",
            }]
        },
        options: opts,
    });
}

function makeDiscreteChart(canvasId, values, xLabel, color, minVal, maxVal) {
    const counts = {};
    for (const v of values) counts[v] = (counts[v] || 0) + 1;
    const keys = [];
    for (let i = minVal; i <= maxVal; i++) keys.push(i);
    const pcts = keys.map(k => ((counts[k] || 0) / values.length) * 100);
    const opts = getBaseOpts();
    opts.scales.x.title = { display: true, text: xLabel, color: "#8b8fa3" };

    return new Chart(document.getElementById(canvasId), {
        type: "bar",
        data: { labels: keys, datasets: [{ data: pcts, backgroundColor: color + "66", borderColor: color, borderWidth: 1 }] },
        options: opts,
    });
}
