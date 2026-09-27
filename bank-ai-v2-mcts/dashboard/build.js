// Build AI tab — run a hand-coded `decide(state)` function vs a bot opponent.

const { jpost } = window.dash;

document.getElementById("btn-build-run").addEventListener("click", async () => {
    const code = document.getElementById("build-code").value;
    const num_players = parseInt(document.getElementById("build-num-players").value, 10);
    const num_games = parseInt(document.getElementById("build-games").value, 10);
    const threshold = parseInt(document.getElementById("build-opp-threshold").value, 10);
    const out = document.getElementById("build-results");
    out.innerHTML = "running...";
    const r = await jpost("/api/build_ai/run", {
        code,
        num_players,
        num_games,
        opponent: { type: "threshold", threshold },
    });
    if (!r.ok) {
        out.innerHTML = `<span style="color: var(--danger)">error: ${r.error}</span>`;
        return;
    }
    out.innerHTML = `
        <h3>Results vs Threshold(${threshold}) — ${num_players}-player, ${r.games} games</h3>
        <table class="data-table">
            <tr><th>Win rate</th><td>${(r.win_rate * 100).toFixed(2)}% (${r.wins}/${r.games})</td></tr>
            <tr><th>Avg score</th><td>${r.avg_score.toFixed(1)}</td></tr>
            <tr><th>Min score</th><td>${r.min_score}</td></tr>
            <tr><th>Max score</th><td>${r.max_score}</td></tr>
        </table>
    `;
});
