// Shared utilities used by all dashboard tabs.

// ----- Tab navigation -----
document.addEventListener("click", (e) => {
    const t = e.target.closest(".tab");
    if (!t) return;
    document.querySelectorAll(".tab").forEach(x => x.classList.toggle("active", x === t));
    const target = t.dataset.tab;
    document.querySelectorAll(".tab-panel").forEach(p => {
        p.classList.toggle("active", p.id === `panel-${target}`);
    });
    if (target === "stats" && window.statsTab) window.statsTab.refresh();
});

// ----- Fetch helpers -----
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

// ----- Status badge -----
async function refreshStatusBadge() {
    try {
        const m = await jget("/api/metrics");
        const badge = document.getElementById("status-badge");
        const status = m.status || "idle";
        badge.textContent = status + (m.status_message ? ` — ${m.status_message}` : "");
        badge.className = `status-badge status-${status}`;
    } catch (e) {
        // ignore
    }
}
setInterval(refreshStatusBadge, 3000);
refreshStatusBadge();

// ----- Tiny canvas line-chart (no external deps) -----
function drawLineChart(canvas, series, opts = {}) {
    const ctx = canvas.getContext("2d");
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth;
    const h = canvas.clientHeight || 200;
    canvas.width = w * dpr;
    canvas.height = h * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    const padL = 50, padR = 12, padT = 14, padB = 28;
    const innerW = w - padL - padR;
    const innerH = h - padT - padB;

    ctx.clearRect(0, 0, w, h);

    // Compute combined min/max
    const all = series.flatMap(s => s.points).filter(p => isFinite(p[1]));
    if (!all.length) {
        ctx.fillStyle = "#98a2b3";
        ctx.font = "12px sans-serif";
        ctx.fillText("no data yet", padL, h / 2);
        return;
    }
    const xs = all.map(p => p[0]);
    const ys = all.map(p => p[1]);
    const xMin = Math.min(...xs), xMax = Math.max(...xs);
    const yMinRaw = Math.min(...ys), yMaxRaw = Math.max(...ys);
    let yMin = opts.yMin !== undefined ? opts.yMin : yMinRaw;
    let yMax = opts.yMax !== undefined ? opts.yMax : yMaxRaw;
    if (yMin === yMax) { yMin -= 1; yMax += 1; }

    const xRange = (xMax - xMin) || 1;
    const yRange = (yMax - yMin) || 1;

    // Grid + axis labels
    ctx.strokeStyle = "#2a3340";
    ctx.fillStyle = "#98a2b3";
    ctx.font = "11px sans-serif";
    ctx.lineWidth = 1;
    const ticks = 4;
    for (let i = 0; i <= ticks; i++) {
        const y = padT + (i / ticks) * innerH;
        ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(padL + innerW, y); ctx.stroke();
        const v = yMax - (i / ticks) * yRange;
        ctx.fillText(v.toFixed(2), 4, y + 3);
    }
    // x-axis ticks
    const xTicks = 4;
    for (let i = 0; i <= xTicks; i++) {
        const x = padL + (i / xTicks) * innerW;
        const v = xMin + (i / xTicks) * xRange;
        ctx.fillText(Math.round(v).toString(), x - 8, h - 10);
    }

    // Series
    series.forEach(s => {
        const pts = s.points;
        if (!pts.length) return;
        ctx.strokeStyle = s.color;
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        pts.forEach((p, i) => {
            const x = padL + ((p[0] - xMin) / xRange) * innerW;
            const y = padT + ((yMax - p[1]) / yRange) * innerH;
            if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        });
        ctx.stroke();
    });

    // Legend
    let lx = padL + 4, ly = padT + 4;
    series.forEach(s => {
        ctx.fillStyle = s.color;
        ctx.fillRect(lx, ly, 10, 10);
        ctx.fillStyle = "#e6edf3";
        ctx.font = "11px sans-serif";
        ctx.fillText(s.label, lx + 14, ly + 9);
        lx += 18 + ctx.measureText(s.label).width;
    });
}

window.dash = { jget, jpost, drawLineChart };
