// Game Stats tab — pure dice probability tables.

// jget is a global from common.js.

async function refresh() {
    let stats;
    try { stats = await jget("/api/dice_stats"); } catch (e) { return; }

    // Sum distribution chart
    const canvas = document.getElementById("chart-dice-sums");
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth, h = canvas.clientHeight || 240;
    canvas.width = w * dpr; canvas.height = h * dpr;
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);

    const padL = 40, padR = 12, padT = 12, padB = 28;
    const innerW = w - padL - padR, innerH = h - padT - padB;
    const sums = stats.sums;
    const maxP = Math.max(...sums.map(s => s.probability));
    const barW = innerW / sums.length - 4;

    ctx.fillStyle = "#98a2b3";
    ctx.font = "11px sans-serif";
    sums.forEach((s, i) => {
        const x = padL + i * (barW + 4);
        const barH = (s.probability / maxP) * innerH;
        const y = padT + (innerH - barH);
        ctx.fillStyle = s.sum === 7 ? "#e85d63" : (s.sum % 2 === 0 ? "#4a90ff" : "#5a677a");
        ctx.fillRect(x, y, barW, barH);
        ctx.fillStyle = "#e6edf3";
        ctx.fillText(`${s.sum}`, x + barW / 2 - 5, h - 12);
        ctx.fillStyle = "#98a2b3";
        ctx.fillText(`${(s.probability * 100).toFixed(1)}%`, x + 1, y - 4);
    });

    // Special events table
    const tableSE = document.getElementById("table-special-events");
    tableSE.innerHTML = `
        <thead><tr><th>Event</th><th>Probability</th></tr></thead>
        <tbody>
            <tr><td>Roll a 7</td><td>${(stats.seven_probability * 100).toFixed(2)}%</td></tr>
            <tr><td>Roll doubles</td><td>${(stats.doubles_probability * 100).toFixed(2)}%</td></tr>
            <tr><td>7 OR doubles (post-immunity)</td><td>${(stats.bust_or_double_post_immunity_probability * 100).toFixed(2)}%</td></tr>
        </tbody>
    `;

    // Post-immunity table
    const e = stats.expected_pot_change_per_roll_post_immunity;
    document.getElementById("table-post-immunity").innerHTML = `
        <thead><tr><th>Outcome</th><th>Probability</th><th>Effect on pot</th></tr></thead>
        <tbody>
            <tr><td>7 (bust)</td><td>${(e.p_bust * 100).toFixed(2)}%</td><td>pot → 0, round ends</td></tr>
            <tr><td>Doubles (not 7)</td><td>${(e.p_double_pot * 100).toFixed(2)}%</td><td>pot × 2</td></tr>
            <tr><td>Other</td><td>${(e.p_increment * 100).toFixed(2)}%</td><td>pot += ~${e.avg_increment.toFixed(2)} (avg)</td></tr>
        </tbody>
    `;
}

window.statsTab = { refresh };
refresh();
