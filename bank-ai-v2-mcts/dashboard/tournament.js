// Tournament tab.

// jget and jpost are globals from common.js.

const competitorsEl = document.getElementById("trn-competitors");
const resultsEl = document.getElementById("trn-results");
let nextId = 1;
let competitors = [];

function render() {
    competitorsEl.innerHTML = "";
    competitors.forEach(c => {
        const card = document.createElement("div");
        card.className = "competitor-card";
        card.dataset.id = c.id;
        let inner = `<div><strong>${c.label}</strong></div>
            <label>name <input type="text" value="${c.name}" data-field="name"></label>`;
        if (c.type === "checkpoint") {
            inner += `<label>checkpoint
                <select data-field="path"></select>
            </label>
            <label>sims/move <input type="number" value="${c.sims_per_move || 100}" min="10" data-field="sims_per_move"></label>`;
        } else if (c.type === "threshold") {
            inner += `<label>threshold <input type="number" value="${c.threshold || 200}" min="1" data-field="threshold"></label>`;
        } else if (c.type === "random") {
            inner += `<label>p_bank <input type="number" value="${c.p_bank || 0.5}" step="0.05" min="0" max="1" data-field="p_bank"></label>`;
        }
        inner += `<button class="removebtn">remove</button>`;
        card.innerHTML = inner;
        competitorsEl.appendChild(card);

        if (c.type === "checkpoint") {
            const sel = card.querySelector('[data-field="path"]');
            populateCheckpointSelect(sel, c.path);
        }
        card.querySelectorAll("[data-field]").forEach(input => {
            input.addEventListener("change", () => {
                const field = input.dataset.field;
                let v = input.value;
                if (input.type === "number") v = parseFloat(v);
                c[field] = v;
            });
        });
        card.querySelector(".removebtn").addEventListener("click", () => {
            competitors = competitors.filter(x => x.id !== c.id);
            render();
        });
    });
}

async function populateCheckpointSelect(sel, currentPath) {
    sel.innerHTML = '<option value="">(none / untrained)</option>';
    try {
        const list = await jget("/api/checkpoints");
        for (const cp of list) {
            const opt = document.createElement("option");
            opt.value = cp.path; opt.textContent = cp.name;
            if (cp.path === currentPath) opt.selected = true;
            sel.appendChild(opt);
        }
        sel.addEventListener("change", () => {
            const id = sel.closest(".competitor-card").dataset.id;
            const c = competitors.find(c => c.id == id);
            if (c) c.path = sel.value;
        });
    } catch (e) {}
}

document.getElementById("btn-trn-add-checkpoint").addEventListener("click", () => {
    competitors.push({ id: nextId++, type: "checkpoint", label: "Checkpoint", name: `AI-${nextId}`, path: "", sims_per_move: 100 });
    render();
});
document.getElementById("btn-trn-add-threshold").addEventListener("click", () => {
    competitors.push({ id: nextId++, type: "threshold", label: "Threshold bot", name: `Thresh-${nextId}`, threshold: 200 });
    render();
});
document.getElementById("btn-trn-add-random").addEventListener("click", () => {
    competitors.push({ id: nextId++, type: "random", label: "Random bot", name: `Rand-${nextId}`, p_bank: 0.5 });
    render();
});

document.getElementById("btn-trn-run").addEventListener("click", async () => {
    if (competitors.length < 2) { alert("add at least 2 competitors"); return; }
    const num_players = parseInt(document.getElementById("trn-num-players").value, 10);
    const games_per_matchup = parseInt(document.getElementById("trn-games-per-matchup").value, 10);
    if (competitors.length < num_players) {
        alert(`need at least ${num_players} competitors for ${num_players}-player games`);
        return;
    }
    resultsEl.innerHTML = "<p class='muted'>running tournament... (this may take a while)</p>";
    const r = await jpost("/api/tournament", {
        num_players,
        games_per_matchup,
        competitors: competitors.map(c => ({
            type: c.type,
            name: c.name,
            path: c.path,
            sims_per_move: c.sims_per_move,
            threshold: c.threshold,
            p_bank: c.p_bank,
        })),
    });
    if (!r.ok) {
        resultsEl.innerHTML = `<p class='muted' style='color: var(--danger)'>error: ${r.error}</p>`;
        return;
    }
    let html = `<h3>Results (${num_players}-player games, ${r.games_per_matchup} games/matchup)</h3>`;
    html += `<table class="data-table"><thead><tr><th>Competitor</th><th>Games</th><th>Wins</th><th>Win rate</th></tr></thead><tbody>`;
    for (const row of r.leaderboard) {
        html += `<tr><td>${row.name}</td><td>${row.games}</td><td>${row.wins}</td><td>${(row.win_rate * 100).toFixed(2)}%</td></tr>`;
    }
    html += `</tbody></table>`;
    resultsEl.innerHTML = html;
});

render();
