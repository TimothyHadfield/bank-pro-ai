// Server-side Play vs AI flow for V2 and Build AI modes.
//
// V1 mode keeps using the original play.js (fast browser-side inference).
// V2 and Custom modes route through this module: every action POSTs to the server,
// which runs MCTS or compiled strategies and returns the next state.

(function () {
    "use strict";

    let unifiedActive = false;        // true when we're driving the Play tab
    let currentState = null;
    let currentAIType = null;
    let currentAILabel = null;
    let humanSeats = [];
    let waitingForServer = false;

    // --------- type-switching dropdown ---------

    window.playOnAITypeChange = async function () {
        const t = document.getElementById("play-ai-type").value;
        document.getElementById("play-row-v1").classList.toggle("hidden", t !== "v1");
        document.getElementById("play-row-v2").classList.toggle("hidden", t !== "v2");
        document.getElementById("play-row-v2-sims").classList.toggle("hidden", t !== "v2");
        document.getElementById("play-row-custom").classList.toggle("hidden", t !== "custom");

        // Adjust the start button: in V1 mode it's gated on weights loading, otherwise
        // it's enabled as soon as a model/strategy is selected.
        const btn = document.getElementById("play-start-btn");
        if (t === "v1") {
            // Let v1's playLoadModel control the button state.
            const cur = document.getElementById("play-checkpoint").value;
            if (cur && typeof playLoadModel === "function") playLoadModel(cur);
        } else if (t === "v2") {
            await refreshV2Checkpoints();
            btn.disabled = !document.getElementById("play-v2-checkpoint").value;
            document.getElementById("play-model-status").textContent =
                btn.disabled ? "No V2 checkpoints found. Train V2 first." : "Ready.";
        } else if (t === "custom") {
            refreshBuildAIs();
            btn.disabled = !document.getElementById("play-custom-name").value;
            document.getElementById("play-model-status").textContent =
                btn.disabled ? "No Build AI strategies saved. Create one in Build AI tab." : "Ready.";
        }
    };

    async function refreshV2Checkpoints() {
        const sel = document.getElementById("play-v2-checkpoint");
        try {
            const r = await fetch("/api/v2/checkpoints", { cache: "no-store" }).then(r => r.json());
            sel.innerHTML = "";
            if (!r.checkpoints || !r.checkpoints.length) {
                sel.innerHTML = "<option value=''>(no V2 checkpoints)</option>";
                return;
            }
            for (const c of r.checkpoints) {
                const opt = document.createElement("option");
                opt.value = c.filename; opt.textContent = c.filename;
                sel.appendChild(opt);
            }
            sel.onchange = () => {
                const btn = document.getElementById("play-start-btn");
                btn.disabled = !sel.value;
            };
        } catch (e) {
            sel.innerHTML = "<option value=''>(error loading)</option>";
        }
    }

    function refreshBuildAIs() {
        const sel = document.getElementById("play-custom-name");
        let saves = {};
        try { saves = JSON.parse(localStorage.getItem("buildAI_saves") || "{}"); } catch (e) {}
        sel.innerHTML = "";
        const names = Object.keys(saves);
        if (!names.length) {
            sel.innerHTML = "<option value=''>(no saved Build AI strategies)</option>";
            return;
        }
        for (const n of names) {
            const opt = document.createElement("option");
            opt.value = n; opt.textContent = n;
            sel.appendChild(opt);
        }
        sel.onchange = () => {
            const btn = document.getElementById("play-start-btn");
            btn.disabled = !sel.value;
        };
    }

    // Hook into the existing playStart() so we can intercept v2 and custom modes.
    const _origPlayStart = window.playStart;
    window.playStart = function () {
        const aiType = document.getElementById("play-ai-type").value;
        if (aiType === "v1") {
            unifiedActive = false;
            return _origPlayStart && _origPlayStart();
        }
        unifiedActive = true;
        return startUnifiedGame(aiType);
    };

    // Override Bank/Continue when in unified mode
    const _origPlayBank = window.playBank;
    const _origPlayContinue = window.playContinue;
    window.playBank = function () {
        if (unifiedActive) return submitUnifiedAction("bank");
        return _origPlayBank && _origPlayBank();
    };
    window.playContinue = function () {
        if (unifiedActive) return submitUnifiedAction("continue");
        return _origPlayContinue && _origPlayContinue();
    };
    const _origPlayBackToSetup = window.playBackToSetup;
    window.playBackToSetup = function () {
        unifiedActive = false;
        if (_origPlayBackToSetup) _origPlayBackToSetup();
    };
    const _origPlayNextRound = window.playNextRound;
    window.playNextRound = function () {
        if (unifiedActive) {
            // In unified mode there's no "next round" button — engine auto-advances.
            // If shown, just hide it.
            document.getElementById("play-next-round").classList.add("hidden");
            return;
        }
        return _origPlayNextRound && _origPlayNextRound();
    };

    // --------- starting and running a unified game ---------

    async function startUnifiedGame(aiType) {
        const numHumans = parseInt(document.getElementById("play-num-humans").value, 10);
        const numAI = parseInt(document.getElementById("play-num-ai").value, 10);
        const numPlayers = numHumans + numAI;
        if (numPlayers < 2 || numPlayers > 6) {
            alert(`Total players must be 2-6 (got ${numPlayers}).`);
            return;
        }
        humanSeats = [];
        for (let i = 0; i < numHumans; i++) humanSeats.push(i);

        const payload = {
            ai_type: aiType,
            num_players: numPlayers,
            human_seats: humanSeats,
        };
        if (aiType === "v1") {
            payload.ai_checkpoint = document.getElementById("play-checkpoint").value;
        } else if (aiType === "v2") {
            payload.ai_checkpoint = document.getElementById("play-v2-checkpoint").value;
            payload.sims_per_move = parseInt(document.getElementById("play-v2-sims").value, 10);
        } else if (aiType === "custom") {
            const name = document.getElementById("play-custom-name").value;
            const saves = JSON.parse(localStorage.getItem("buildAI_saves") || "{}");
            payload.ai_code = saves[name];
            payload.ai_label = name;
            if (!payload.ai_code) { alert("Could not find saved strategy."); return; }
        }

        const startBtn = document.getElementById("play-start-btn");
        startBtn.disabled = true;
        startBtn.textContent = "Loading...";

        let r;
        try {
            r = await fetch("/api/play_unified/new", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload),
            }).then(x => x.json());
        } catch (e) {
            alert("Network error: " + e);
            startBtn.disabled = false; startBtn.textContent = "Start Game";
            return;
        }
        startBtn.disabled = false; startBtn.textContent = "Start Game";

        if (!r.ok) {
            alert("Could not start: " + (r.error || "unknown"));
            return;
        }
        currentAIType = r.ai_type;
        currentAILabel = r.ai_label;
        currentState = r.state;

        // Switch UI to game view
        document.getElementById("play-setup").classList.add("hidden");
        document.getElementById("play-area").classList.remove("hidden");
        document.getElementById("play-gameover").classList.add("hidden");
        document.getElementById("play-log").innerHTML = "";

        renderUnifiedState();
        playUnifiedLog(`Game started: ${currentAILabel} (${aiType.toUpperCase()})`, "log-dim");
    }

    async function submitUnifiedAction(action) {
        if (waitingForServer || !currentState) return;
        // Find the human seat that's currently waiting
        const waitingSeat = humanSeats.find(s => currentState.players_in_round.includes(s));
        if (waitingSeat === undefined) return;
        waitingForServer = true;
        document.getElementById("play-buttons").classList.add("hidden");
        playUnifiedLog(`P${waitingSeat + 1}: ${action.toUpperCase()}`, "log-human");

        // Show "thinking" indicator
        const turnLabel = document.getElementById("play-turn-label");
        if (currentAIType === "v2") {
            turnLabel.textContent = `${currentAILabel} is thinking...`;
            turnLabel.classList.remove("hidden");
        }

        const actions = {};
        actions[String(waitingSeat)] = action;

        let r;
        try {
            r = await fetch("/api/play_unified/move", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ actions }),
            }).then(x => x.json());
        } catch (e) {
            playUnifiedLog("Network error: " + e, "log-warn");
            waitingForServer = false;
            return;
        }
        waitingForServer = false;
        turnLabel.classList.add("hidden");

        if (!r.ok) {
            playUnifiedLog("Server error: " + (r.error || "unknown"), "log-warn");
            return;
        }
        currentState = r.state;

        // Log AI decisions
        if (r.state.revealed_actions) {
            const aiLines = [];
            for (const [pidStr, act] of Object.entries(r.state.revealed_actions)) {
                const pid = parseInt(pidStr, 10);
                if (!humanSeats.includes(pid)) {
                    aiLines.push(`P${pid + 1} (AI): ${act}`);
                }
            }
            if (aiLines.length) playUnifiedLog(aiLines.join("  ·  "), "log-ai");
            if (r.state.last_dice) {
                const sum = r.state.last_dice[0] + r.state.last_dice[1];
                let tag = `${r.state.last_dice[0]}+${r.state.last_dice[1]} = ${sum}`;
                if (sum === 7 && !r.state.immunity_active) tag += " (BUST)";
                else if (r.state.last_dice[0] === r.state.last_dice[1] && !r.state.immunity_active) tag += " (×2)";
                playUnifiedLog(`Roll: ${tag}`, "log-dice");
            }
        }

        renderUnifiedState();
    }

    function renderUnifiedState() {
        const s = currentState;
        if (!s) return;
        // Pot, round, roll
        document.getElementById("play-pot-val").textContent = s.pot;
        document.getElementById("play-round").textContent = s.round_number;
        document.getElementById("play-roll").textContent = s.roll_number;
        const immEl = document.getElementById("play-imm");
        if (s.immunity_active) {
            immEl.classList.remove("hidden");
            immEl.classList.add("imm-on");
            immEl.textContent = "IMMUNITY";
        } else {
            immEl.classList.remove("imm-on");
            immEl.classList.add("hidden");
        }

        // Dice
        if (s.last_dice) {
            playUnifiedSetDie("play-d1", s.last_dice[0]);
            playUnifiedSetDie("play-d2", s.last_dice[1]);
            const sum = s.last_dice[0] + s.last_dice[1];
            document.getElementById("play-dice-sum").textContent = "= " + sum;
            const tagEl = document.getElementById("play-tag");
            tagEl.classList.remove("tag-normal", "tag-bust", "tag-double", "tag-imm7");
            if (sum === 7 && !s.immunity_active) {
                tagEl.textContent = "BUST"; tagEl.classList.add("tag-bust");
            } else if (s.last_dice[0] === s.last_dice[1] && !s.immunity_active) {
                tagEl.textContent = "× 2"; tagEl.classList.add("tag-double");
            } else if (sum === 7 && s.immunity_active) {
                tagEl.textContent = "+70"; tagEl.classList.add("tag-imm7");
            } else {
                tagEl.textContent = "+" + sum; tagEl.classList.add("tag-normal");
            }
        }

        // Scores
        const scoresEl = document.getElementById("play-scores");
        let html = "";
        for (let i = 0; i < s.num_players; i++) {
            const isHuman = s.human_seats.includes(i);
            const inRound = s.players_in_round.includes(i);
            const cls = ["play-score-row"];
            if (isHuman) cls.push("play-score-human");
            if (!inRound) cls.push("play-score-banked");
            const label = isHuman ? `P${i + 1} (You)` : `P${i + 1} (AI)`;
            html += `<div class="${cls.join(' ')}">
                <span>${label}</span>
                <b>${s.banked_scores[i]}</b>
                ${inRound ? "" : "<span class='text-dim' style='font-size:0.7rem;'>banked</span>"}
            </div>`;
        }
        scoresEl.innerHTML = html;

        // Buttons / game-over
        const btns = document.getElementById("play-buttons");
        const gameoverEl = document.getElementById("play-gameover");
        if (s.game_over) {
            btns.classList.add("hidden");
            document.getElementById("play-area").classList.add("hidden");
            gameoverEl.classList.remove("hidden");
            const winners = s.winners;
            const youWon = winners.some(w => humanSeats.includes(w));
            const titleEl = document.getElementById("play-go-title");
            titleEl.textContent = youWon ? (winners.length > 1 ? "Tie for the win!" : "You Win!") : "AI Wins!";
            titleEl.classList.toggle("go-win", youWon);
            titleEl.classList.toggle("go-loss", !youWon);
            let rows = "";
            for (let i = 0; i < s.num_players; i++) {
                const isHuman = s.human_seats.includes(i);
                rows += `<div class="play-go-row">
                    <span>${isHuman ? `P${i + 1} (You)` : `P${i + 1} (AI)`}</span>
                    <b>${s.banked_scores[i]}</b>
                    ${winners.includes(i) ? "<span style='color:var(--green);'>★</span>" : ""}
                </div>`;
            }
            document.getElementById("play-go-scores").innerHTML = rows;
            return;
        }

        // Active game: show buttons if a human seat is waiting
        const humanWaiting = humanSeats.some(s2 => s.players_in_round.includes(s2));
        if (humanWaiting && !waitingForServer) {
            btns.classList.remove("hidden");
        } else {
            btns.classList.add("hidden");
        }

        // If only AI seats remain in the round, auto-advance
        if (!humanWaiting && !s.game_over && s.players_in_round.length > 0 && !waitingForServer) {
            setTimeout(() => submitEmptyMove(), 300);
        }
    }

    async function submitEmptyMove() {
        if (waitingForServer) return;
        waitingForServer = true;
        const r = await fetch("/api/play_unified/move", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ actions: {} }),
        }).then(x => x.json());
        waitingForServer = false;
        if (!r.ok) {
            playUnifiedLog("Auto-step error: " + (r.error || "unknown"), "log-warn");
            return;
        }
        currentState = r.state;
        if (r.state.revealed_actions) {
            const aiLines = [];
            for (const [pidStr, act] of Object.entries(r.state.revealed_actions)) {
                aiLines.push(`P${parseInt(pidStr, 10) + 1} (AI): ${act}`);
            }
            if (aiLines.length) playUnifiedLog(aiLines.join("  ·  "), "log-ai");
            if (r.state.last_dice) {
                const sum = r.state.last_dice[0] + r.state.last_dice[1];
                playUnifiedLog(`Roll: ${r.state.last_dice[0]}+${r.state.last_dice[1]} = ${sum}`, "log-dice");
            }
        }
        renderUnifiedState();
    }

    function playUnifiedSetDie(id, v) {
        const el = document.getElementById(id);
        if (!el) return;
        el.className = "play-die dv-" + v;
        // Try to draw pips. If v1's CSS doesn't render bare span pips, this still
        // shows dot count via simple inner HTML.
        const pipCount = v;
        let pips = "";
        for (let i = 0; i < pipCount; i++) pips += "<span class='pip'></span>";
        el.innerHTML = pips || "<span class='pip'></span>";
    }

    function playUnifiedLog(text, cls) {
        const el = document.getElementById("play-log");
        if (!el) return;
        const div = document.createElement("div");
        div.className = "play-log-line " + (cls || "");
        div.textContent = text;
        el.appendChild(div);
        el.scrollTop = el.scrollHeight;
    }
})();
