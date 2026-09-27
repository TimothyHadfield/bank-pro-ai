// Cross-version AI vs AI (V1 + V2 + Build AI) — runs server-side via /api/aivai_combined.

(function () {
    "use strict";

    let competitors = [];
    let nextId = 1;

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

    function getBuildAISaves() {
        try { return JSON.parse(localStorage.getItem("buildAI_saves") || "{}"); }
        catch (e) { return {}; }
    }

    window.ava2AddCompetitor = function (type) {
        competitors.push({
            id: nextId++,
            type,
            name: type === "v1" ? `v1-${nextId}` :
                  type === "v2" ? `v2-${nextId}` :
                  `Custom-${nextId}`,
            checkpoint: "",
            sims_per_move: 100,
            buildAIName: "",
        });
        renderCompetitors();
    };

    async function renderCompetitors() {
        const container = document.getElementById("ava2-competitors");
        if (!container) return;
        if (competitors.length === 0) {
            container.innerHTML = "<p class='text-dim' style='margin-top:8px;'>Add competitors with the buttons above.</p>";
            return;
        }

        // Fetch checkpoint lists once
        let v1List = [], v2List = [];
        try { v1List = (await jget("/api/checkpoints")).checkpoints || []; } catch (e) {}
        try { v2List = (await jget("/api/v2/checkpoints")).checkpoints || []; } catch (e) {}
        const buildSaves = getBuildAISaves();

        let html = "";
        for (const c of competitors) {
            html += `<div class="ava2-competitor-card" data-id="${c.id}" style="background: var(--bg-card); border: 1px solid var(--border); border-radius: 6px; padding: 10px; margin-bottom: 8px; display: flex; flex-wrap: wrap; gap: 10px; align-items: center;">`;
            html += `<span style="font-weight:600; color:var(--accent); min-width:60px;">${c.type.toUpperCase()}</span>`;

            html += `<label style="display:flex;flex-direction:column;font-size:0.75rem;color:var(--text-dim);gap:2px;">
                Name<input class="setting-input" data-field="name" value="${escapeAttr(c.name)}" style="width:130px;"></label>`;

            if (c.type === "v1") {
                html += `<label style="display:flex;flex-direction:column;font-size:0.75rem;color:var(--text-dim);gap:2px;">
                    Checkpoint<select class="control-select" data-field="checkpoint" style="min-width:200px;">
                        <option value="">(pick one)</option>
                        ${v1List.map(cp => `<option value="${escapeAttr(cp.filename)}" ${cp.filename===c.checkpoint?"selected":""}>${cp.filename}</option>`).join("")}
                    </select></label>`;
            } else if (c.type === "v2") {
                html += `<label style="display:flex;flex-direction:column;font-size:0.75rem;color:var(--text-dim);gap:2px;">
                    Checkpoint<select class="control-select" data-field="checkpoint" style="min-width:200px;">
                        <option value="">(pick one)</option>
                        ${v2List.map(cp => `<option value="${escapeAttr(cp.filename)}" ${cp.filename===c.checkpoint?"selected":""}>${cp.filename}</option>`).join("")}
                    </select></label>`;
                html += `<label style="display:flex;flex-direction:column;font-size:0.75rem;color:var(--text-dim);gap:2px;">
                    Sims/Move<input class="setting-input" type="number" data-field="sims_per_move" value="${c.sims_per_move}" min="10" max="800" style="width:80px;"></label>`;
            } else if (c.type === "custom") {
                const names = Object.keys(buildSaves);
                html += `<label style="display:flex;flex-direction:column;font-size:0.75rem;color:var(--text-dim);gap:2px;">
                    Build AI<select class="control-select" data-field="buildAIName" style="min-width:180px;">
                        <option value="">(pick one)</option>
                        ${names.map(n => `<option value="${escapeAttr(n)}" ${n===c.buildAIName?"selected":""}>${n}</option>`).join("")}
                    </select></label>`;
                if (!names.length) {
                    html += `<span style="color:var(--yellow);font-size:0.72rem;">Save a strategy in Build AI first.</span>`;
                }
            }

            html += `<button class="btn btn-del" style="font-size:0.72rem;padding:5px 9px;margin-left:auto;" onclick="ava2RemoveCompetitor(${c.id})">remove</button>`;
            html += `</div>`;
        }
        container.innerHTML = html;

        // Wire up change handlers
        container.querySelectorAll("[data-field]").forEach(input => {
            input.addEventListener("change", (e) => {
                const card = e.target.closest(".ava2-competitor-card");
                const id = parseInt(card.dataset.id, 10);
                const c = competitors.find(c => c.id === id);
                if (!c) return;
                const field = e.target.dataset.field;
                let val = e.target.value;
                if (e.target.type === "number") val = parseFloat(val);
                c[field] = val;
            });
        });
    }

    window.ava2RemoveCompetitor = function (id) {
        competitors = competitors.filter(c => c.id !== id);
        renderCompetitors();
    };

    function escapeAttr(s) {
        return String(s).replace(/&/g, "&amp;").replace(/"/g, "&quot;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
    }
    function escapeHtml(s) {
        return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
    }

    window.ava2RunTournament = async function () {
        if (competitors.length < 2) { alert("Add at least 2 competitors."); return; }
        const num_players = parseInt(document.getElementById("ava-num-players").value, 10);
        const games_per_matchup = parseInt(document.getElementById("ava-games-per-matchup").value, 10);
        if (competitors.length < num_players) {
            alert(`Need at least ${num_players} competitors for ${num_players}-player games. You have ${competitors.length}.`);
            return;
        }

        const buildSaves = getBuildAISaves();

        // Validate and build payload
        const payload = { num_players, games_per_matchup, competitors: [] };
        for (const c of competitors) {
            if (c.type === "v1" || c.type === "v2") {
                if (!c.checkpoint) { alert(`${c.name}: pick a checkpoint.`); return; }
                payload.competitors.push({
                    type: c.type, name: c.name, checkpoint: c.checkpoint,
                    sims_per_move: c.sims_per_move,
                });
            } else if (c.type === "custom") {
                const code = buildSaves[c.buildAIName];
                if (!code) { alert(`${c.name}: pick a saved Build AI strategy.`); return; }
                payload.competitors.push({
                    type: "custom", name: c.name, code,
                });
            }
        }

        document.getElementById("ava-status").classList.remove("hidden");
        document.getElementById("ava-results").classList.add("hidden");
        document.getElementById("ava-run-btn").disabled = true;

        try {
            const r = await jpost("/api/aivai_combined", payload);
            document.getElementById("ava-status").classList.add("hidden");
            document.getElementById("ava-run-btn").disabled = false;
            if (r.error) {
                alert("Tournament failed: " + r.error);
                return;
            }
            renderResults(r);
        } catch (e) {
            document.getElementById("ava-status").classList.add("hidden");
            document.getElementById("ava-run-btn").disabled = false;
            alert("Network error: " + e);
        }
    };

    function renderResults(r) {
        const div = document.getElementById("ava-results");
        const body = document.getElementById("ava-results-body");
        div.classList.remove("hidden");

        let html = `<table class="data-table" style="width:100%;border-collapse:collapse;margin-bottom:16px;">
            <thead><tr><th style="text-align:left;padding:6px 10px;border-bottom:1px solid var(--border);">Competitor</th>
            <th style="padding:6px 10px;border-bottom:1px solid var(--border);">Type</th>
            <th style="padding:6px 10px;border-bottom:1px solid var(--border);">Games</th>
            <th style="padding:6px 10px;border-bottom:1px solid var(--border);">Wins</th>
            <th style="padding:6px 10px;border-bottom:1px solid var(--border);">Win rate</th>
            <th style="padding:6px 10px;border-bottom:1px solid var(--border);">Avg score</th></tr></thead><tbody>`;
        for (const row of r.leaderboard) {
            html += `<tr>
                <td style="padding:6px 10px;border-bottom:1px solid var(--border);">${escapeHtml(row.name)}</td>
                <td style="padding:6px 10px;border-bottom:1px solid var(--border);">${row.kind.toUpperCase()}</td>
                <td style="padding:6px 10px;border-bottom:1px solid var(--border);">${row.games}</td>
                <td style="padding:6px 10px;border-bottom:1px solid var(--border);">${row.wins}</td>
                <td style="padding:6px 10px;border-bottom:1px solid var(--border);"><strong>${(row.win_rate*100).toFixed(2)}%</strong></td>
                <td style="padding:6px 10px;border-bottom:1px solid var(--border);">${row.avg_score}</td></tr>`;
        }
        html += `</tbody></table>`;
        html += `<p class="text-dim">${r.num_players}-player games, ${r.games_per_matchup} games per matchup.</p>`;
        body.innerHTML = html;
    }

    // First render
    document.addEventListener("DOMContentLoaded", renderCompetitors);
    setTimeout(renderCompetitors, 200);
})();
