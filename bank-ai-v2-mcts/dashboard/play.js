// Play vs AI tab.
//
// No-peeking guarantee: the human submits their decision FIRST, then the server runs
// MCTS for all AI seats independently and applies all actions in one engine step().
// We only reveal what each AI decided AFTER the step has been applied.

const { jget, jpost } = window.dash;

const numPlayersSel = document.getElementById("play-num-players");
const humanSeatSel = document.getElementById("play-human-seat");
const checkpointSel = document.getElementById("play-checkpoint");
const simsSel = document.getElementById("play-sims");
const newBtn = document.getElementById("btn-play-new");
const boardCard = document.getElementById("play-board-card");
const boardEl = document.getElementById("play-board");
const bankBtn = document.getElementById("btn-bank");
const continueBtn = document.getElementById("btn-continue");
const statusMsg = document.getElementById("play-status-msg");
const historyEl = document.getElementById("play-history");

let currentState = null;
let myHumanSeat = 0;
let waitingForServer = false;

function refreshSeatOptions() {
    const n = parseInt(numPlayersSel.value, 10);
    humanSeatSel.innerHTML = "";
    for (let i = 0; i < n; i++) {
        const opt = document.createElement("option");
        opt.value = i; opt.textContent = `${i}${i === 0 ? " (you)" : ""}`;
        humanSeatSel.appendChild(opt);
    }
}
numPlayersSel.addEventListener("change", refreshSeatOptions);
refreshSeatOptions();

async function refreshCheckpoints() {
    try {
        const list = await jget("/api/checkpoints");
        const sel = checkpointSel;
        sel.innerHTML = '<option value="">(latest)</option>';
        for (const c of list) {
            const opt = document.createElement("option");
            opt.value = c.path; opt.textContent = c.name;
            sel.appendChild(opt);
        }
    } catch (e) {}
}
refreshCheckpoints();
setInterval(refreshCheckpoints, 8000);

newBtn.addEventListener("click", async () => {
    myHumanSeat = parseInt(humanSeatSel.value, 10);
    newBtn.disabled = true;
    newBtn.textContent = "Loading model...";
    const r = await jpost("/api/play/new", {
        num_players: parseInt(numPlayersSel.value, 10),
        human_seats: [myHumanSeat],
        checkpoint: checkpointSel.value || "models/latest.pt",
        sims_per_move: parseInt(simsSel.value, 10),
    });
    newBtn.disabled = false;
    newBtn.textContent = "Start game";
    if (!r.ok) {
        alert("could not start game: " + (r.error || "unknown"));
        return;
    }
    currentState = r.state;
    boardCard.style.display = "block";
    historyEl.innerHTML = "";
    renderState();
});

bankBtn.addEventListener("click", () => submitMove("bank"));
continueBtn.addEventListener("click", () => submitMove("continue"));

async function submitMove(action) {
    if (!currentState || waitingForServer) return;
    if (!currentState.players_in_round.includes(myHumanSeat)) return;
    waitingForServer = true;
    bankBtn.disabled = true;
    continueBtn.disabled = true;
    statusMsg.textContent = `you: ${action}. waiting on AI(s)...`;
    const r = await jpost("/api/play/move", {
        actions: { [String(myHumanSeat)]: action },
    });
    waitingForServer = false;
    if (!r.ok) {
        statusMsg.textContent = "error: " + (r.error || "unknown");
        return;
    }
    currentState = r.state;
    renderState();
    if (r.state.revealed_actions) {
        const items = Object.entries(r.state.revealed_actions)
            .map(([pid, a]) => `seat ${pid}: ${a}`)
            .join(" · ");
        const dice = r.state.last_dice ? ` → roll ${r.state.last_dice.join("+")} = ${r.state.last_dice[0]+r.state.last_dice[1]}` : "";
        const li = document.createElement("div");
        li.textContent = `R${currentState.round_number} ${items}${dice}`;
        historyEl.prepend(li);
    }
}

function renderState() {
    if (!currentState) return;
    const s = currentState;
    const dice = s.last_dice ? `${s.last_dice[0]} + ${s.last_dice[1]} = ${s.last_dice[0]+s.last_dice[1]}` : "—";
    const immStatus = s.immunity_active ? `<span class="muted">(immunity, roll ${s.roll_number}/3)</span>` : "";

    let players = "";
    for (let i = 0; i < s.num_players; i++) {
        const isYou = i === myHumanSeat;
        const inRound = s.players_in_round.includes(i);
        const cls = ["player-card"];
        if (isYou) cls.push("you");
        if (!inRound) cls.push("banked");
        const status = !inRound ? "banked" : (s.ai_seats.includes(i) ? "AI" : "human");
        players += `<div class="${cls.join(' ')}">
            <div class="pname">P${i}${isYou ? " (you)" : ""}</div>
            <div class="pscore">${s.banked_scores[i]}</div>
            <div class="pstatus">${status}</div>
        </div>`;
    }

    boardEl.innerHTML = `
        <div class="pot-banner">
            <div class="pot-value">${s.pot}</div>
            <div class="pot-label">pot · round ${s.round_number}/10 · roll ${s.roll_number} ${immStatus}</div>
            <div class="dice-display">last dice: ${dice}</div>
        </div>
        <div class="players-row">${players}</div>
    `;

    if (s.game_over) {
        statusMsg.textContent = "GAME OVER. Winners: " + (s.winners.join(", ") || "(none)");
        bankBtn.disabled = true;
        continueBtn.disabled = true;
        return;
    }

    const youInRound = s.players_in_round.includes(myHumanSeat);
    if (youInRound) {
        bankBtn.disabled = false;
        continueBtn.disabled = false;
        statusMsg.textContent = "your turn — bank or continue?";
    } else {
        bankBtn.disabled = true;
        continueBtn.disabled = true;
        statusMsg.textContent = "you have already banked this round — auto-advancing...";
        // Auto-advance: send empty action; server will run AI seats
        if (!waitingForServer) {
            setTimeout(() => submitMoveEmpty(), 400);
        }
    }
}

async function submitMoveEmpty() {
    if (!currentState || waitingForServer) return;
    if (currentState.players_in_round.includes(myHumanSeat)) return;
    waitingForServer = true;
    const r = await jpost("/api/play/move", { actions: {} });
    waitingForServer = false;
    if (!r.ok) {
        statusMsg.textContent = "error: " + (r.error || "unknown");
        return;
    }
    currentState = r.state;
    renderState();
}
