// ============================================================
// Bank AI — Fully client-side game engine + neural network
// ============================================================

// ==================== NEURAL NETWORK ====================

class BankNetwork {
    constructor(weightsData) {
        this.stateSize = weightsData.state_size;
        this.hiddenSize = weightsData.hidden_size;
        const w = weightsData.weights;

        // Policy network: net.0 = linear1, net.2 = linear2, net.4 = output
        this.fc1_weight = w["net.0.weight"];
        this.fc1_bias = w["net.0.bias"];
        this.fc2_weight = w["net.2.weight"];
        this.fc2_bias = w["net.2.bias"];
        this.out_weight = w["net.4.weight"];
        this.out_bias = w["net.4.bias"];
    }

    _linear(weight, bias, input) {
        const out = new Array(weight.length);
        for (let i = 0; i < weight.length; i++) {
            let sum = bias[i];
            for (let j = 0; j < input.length; j++) {
                sum += weight[i][j] * input[j];
            }
            out[i] = sum;
        }
        return out;
    }

    _relu(arr) { return arr.map(x => Math.max(0, x)); }

    _softmax(arr) {
        const max = Math.max(...arr);
        const exps = arr.map(x => Math.exp(x - max));
        const sum = exps.reduce((a, b) => a + b, 0);
        return exps.map(x => x / sum);
    }

    forward(stateVec) {
        let h = this._linear(this.fc1_weight, this.fc1_bias, stateVec);
        h = this._relu(h);
        h = this._linear(this.fc2_weight, this.fc2_bias, h);
        h = this._relu(h);
        const logits = this._linear(this.out_weight, this.out_bias, h);
        return { probs: this._softmax(logits) };
    }

    decide(stateVec, deterministic = false) {
        const { probs } = this.forward(stateVec);
        if (deterministic) return probs[0] >= probs[1] ? "bank" : "continue";
        // Sample from distribution
        return Math.random() < probs[0] ? "bank" : "continue";
    }
}


// ==================== GAME ENGINE ====================

class BankGame {
    constructor(numPlayers = 4, numRounds = 10) {
        this.numPlayers = numPlayers;
        this.numRounds = numRounds;
        this.reset();
    }

    reset() {
        this.roundNumber = 1;
        this.rollNumber = 0;
        this.pot = 0;
        this.bankedScores = new Array(this.numPlayers).fill(0);
        this.playersInRound = new Set();
        for (let i = 0; i < this.numPlayers; i++) this.playersInRound.add(i);
        this.gameOver = false;
        this.roundOver = false;
        this.waitingForDecisions = false;
        this.lastDice = [0, 0];
        this.immunityHistory = [];

        this._rollImmunity();
    }

    _randInt(min, max) {
        return Math.floor(Math.random() * (max - min + 1)) + min;
    }

    _rollDice() {
        this.rollNumber++;
        const d1 = this._randInt(1, 6);
        const d2 = this._randInt(1, 6);
        this.lastDice = [d1, d2];
        const diceSum = d1 + d2;
        const isDoubles = d1 === d2;
        const isSeven = diceSum === 7;
        const immunityActive = this.rollNumber <= 3;

        if (immunityActive) {
            if (isSeven) {
                this.pot += 70;
            } else {
                this.pot += diceSum;
            }
            this.roundOver = false;
            this.waitingForDecisions = false; // No decisions during immunity
        } else {
            if (isSeven) {
                this.pot = 0;
                this.roundOver = true;
                this.waitingForDecisions = false;
            } else if (isDoubles) {
                this.pot *= 2;
                this.roundOver = false;
                this.waitingForDecisions = true;
            } else {
                this.pot += diceSum;
                this.roundOver = false;
                this.waitingForDecisions = true;
            }
        }
    }

    _rollImmunity() {
        this.immunityHistory = [];
        for (let i = 0; i < 3; i++) {
            this._rollDice();
            this.immunityHistory.push({
                dice: [...this.lastDice],
                pot: this.pot,
                rollNumber: this.rollNumber,
            });
        }
        // Players decide BEFORE any post-immunity roll
        this.waitingForDecisions = true;
    }

    _startNewRound() {
        this.roundNumber++;
        if (this.roundNumber > this.numRounds) {
            this.gameOver = true;
            return;
        }
        this.rollNumber = 0;
        this.pot = 0;
        this.playersInRound = new Set();
        for (let i = 0; i < this.numPlayers; i++) this.playersInRound.add(i);
        this.roundOver = false;
        this._rollImmunity();
    }

    getStateVector(playerId) {
        const myScore = this.bankedScores[playerId];
        const allScores = [...this.bankedScores];
        const oppScores = [];
        const oppIn = [];
        for (let i = 0; i < this.numPlayers; i++) {
            if (i === playerId) continue;
            oppScores.push(this.bankedScores[i]);
            oppIn.push(this.playersInRound.has(i) ? 1.0 : 0.0);
        }
        const highest = Math.max(...allScores);
        const lowest = Math.min(...allScores);
        const highestOpp = oppScores.length > 0 ? Math.max(...oppScores) : 0;
        let rank = 1;
        for (const s of allScores) if (s > myScore) rank++;

        const vec = [
            this.pot / 1000.0,
            this.rollNumber / 10.0,
            this.roundNumber / 10.0,
            myScore / 1000.0,
            this.playersInRound.size / 6.0,
            (this.numRounds - this.roundNumber) / 10.0,
            this.roundNumber === this.numRounds ? 1.0 : 0.0,
            Math.max(0, this.rollNumber - 3) / 10.0,
            rank / this.numPlayers,
            (myScore - highest) / 1000.0,
            (myScore - lowest) / 1000.0,
            (myScore + this.pot) >= highestOpp ? 1.0 : 0.0,
        ];
        for (let i = 0; i < 5; i++) vec.push(i < oppScores.length ? oppScores[i] / 1000.0 : 0.0);
        for (let i = 0; i < 5; i++) vec.push(i < oppIn.length ? oppIn[i] : 0.0);
        return vec;
    }

    step(actions) {
        const info = {
            dice: [...this.lastDice],
            potBeforeActions: this.pot,
            bankedPlayers: [],
            roundEndedBySeven: false,
            bustDice: null,
        };

        const bankers = [];
        for (const pid of this.playersInRound) {
            if (actions[pid] === "bank") {
                this.bankedScores[pid] += this.pot;
                bankers.push(pid);
            }
        }
        info.bankedPlayers = bankers;

        for (const pid of bankers) {
            this.playersInRound.delete(pid);
        }

        // All banked?
        if (this.playersInRound.size === 0) {
            this.roundOver = true;
            this.waitingForDecisions = false;
            this._startNewRound();
            return { roundDone: true, gameDone: this.gameOver, info };
        }

        // Roll next dice (always post-immunity)
        this._rollDice();

        if (this.roundOver) {
            info.roundEndedBySeven = true;
            info.bustDice = [...this.lastDice];
            // DON'T auto-advance — UI will show bust screen and call advanceRound()
            return { roundDone: true, gameDone: false, info };
        }

        return { roundDone: false, gameDone: false, info };
    }

    advanceRound() {
        this._startNewRound();
    }

    getWinners() {
        const maxScore = Math.max(...this.bankedScores);
        const winners = [];
        for (let i = 0; i < this.numPlayers; i++) {
            if (this.bankedScores[i] === maxScore) winners.push(i);
        }
        return winners;
    }
}


// ==================== UI CONTROLLER ====================

let network = null;
let game = null;
let numAi = 3;
let bustPending = false; // True when we need to show bust screen

async function loadWeights() {
    try {
        const resp = await fetch("weights.json");
        if (!resp.ok) throw new Error("Could not load weights.json");
        const data = await resp.json();
        network = new BankNetwork(data);
        document.getElementById("model-info").textContent =
            `Model loaded (batch ${data.batch_num})`;
        document.getElementById("start-btn").disabled = false;
    } catch (e) {
        document.getElementById("model-info").textContent =
            "No weights.json found \u2014 AI will play randomly";
        network = null;
        document.getElementById("start-btn").disabled = false;
    }
}

function startGame() {
    numAi = parseInt(document.getElementById("num-ai-select").value);
    const numPlayers = 1 + numAi;

    game = new BankGame(numPlayers, 10);
    game.reset();
    bustPending = false;

    document.getElementById("setup-screen").classList.add("hidden");
    document.getElementById("game-screen").classList.remove("hidden");
    document.getElementById("gameover-overlay").classList.add("hidden");

    clearLog();
    addLog(`Game started! You vs ${numAi} AI opponent(s). Round 1 begins.`, "system");
    logImmunityRolls();
    updateUI();
}

function showSetup() {
    document.getElementById("setup-screen").classList.remove("hidden");
    document.getElementById("game-screen").classList.add("hidden");
    document.getElementById("gameover-overlay").classList.add("hidden");
    game = null;
    bustPending = false;
}

function logImmunityRolls() {
    if (!game || !game.immunityHistory || game.immunityHistory.length === 0) return;
    // Log in reverse so they appear chronologically (addLog prepends)
    for (let i = game.immunityHistory.length - 1; i >= 0; i--) {
        const roll = game.immunityHistory[i];
        const d = roll.dice;
        const sum = d[0] + d[1];
        let msg, cls;
        if (sum === 7) {
            msg = `Roll ${roll.rollNumber}: ${d[0]}+${d[1]}=7 \u2014 LUCKY 7! +70 \u2014 Pot: ${roll.pot}`;
            cls = "lucky";
        } else if (d[0] === d[1]) {
            msg = `Roll ${roll.rollNumber}: ${d[0]}+${d[1]}=${sum} (doubles, +${sum}) \u2014 Pot: ${roll.pot}`;
            cls = "system";
        } else {
            msg = `Roll ${roll.rollNumber}: ${d[0]}+${d[1]}=${sum} (+${sum}) \u2014 Pot: ${roll.pot}`;
            cls = "system";
        }
        addLog(msg, cls);
    }
    addLog("Immunity phase (auto-rolled):", "immunity");
}

function doAction(action) {
    if (!game || game.gameOver || !game.waitingForDecisions || bustPending) return;
    if (!game.playersInRound.has(0)) return;

    // Hide buttons immediately
    document.getElementById("action-area").classList.add("hidden");

    addLog(`You chose to ${action.toUpperCase()}`, action === "bank" ? "bank" : "continue-log");

    // Collect all actions
    const actions = {};
    const aiDecisions = {};

    for (const pid of game.playersInRound) {
        if (pid === 0) {
            actions[pid] = action;
        } else {
            let aiAction;
            if (network) {
                const sv = game.getStateVector(pid);
                aiAction = network.decide(sv);
            } else {
                aiAction = Math.random() < 0.3 ? "bank" : "continue";
            }
            actions[pid] = aiAction;
            aiDecisions[pid] = aiAction;
        }
    }

    const result = game.step(actions);

    // Log AI decisions
    for (const [pid, dec] of Object.entries(aiDecisions)) {
        const cls = dec === "bank" ? "bank" : "continue-log";
        addLog(`AI ${pid}: ${dec.toUpperCase()}`, cls);
    }

    // Log banking
    for (const pid of result.info.bankedPlayers) {
        const label = pid === 0 ? "You" : `AI ${pid}`;
        addLog(`${label} banked ${result.info.potBeforeActions} points!`, "bank");
    }

    // Check for bust
    if (result.info.roundEndedBySeven) {
        // Show bust screen — don't advance yet
        bustPending = true;
        showBustScreen(result.info.bustDice);
        return;
    }

    if (result.roundDone && !result.gameDone) {
        // All players banked, new round started automatically
        addLog(`Round ended. Starting round ${game.roundNumber}.`, "round-end");
        logImmunityRolls();
    }

    // Auto-advance if human is out but round continues
    autoAdvanceAI();

    updateUI();

    if (game.gameOver) {
        showGameOver();
    }
}

function autoAdvanceAI() {
    while (!game.gameOver && game.waitingForDecisions && !game.playersInRound.has(0) && !bustPending) {
        const autoActions = {};
        const autoDecs = {};
        for (const pid of game.playersInRound) {
            let aiAction;
            if (network) {
                aiAction = network.decide(game.getStateVector(pid));
            } else {
                aiAction = Math.random() < 0.3 ? "bank" : "continue";
            }
            autoActions[pid] = aiAction;
            autoDecs[pid] = aiAction;
        }

        const autoResult = game.step(autoActions);

        for (const [pid, dec] of Object.entries(autoDecs)) {
            addLog(`AI ${pid}: ${dec.toUpperCase()}`, dec === "bank" ? "bank" : "continue-log");
        }
        for (const pid of autoResult.info.bankedPlayers) {
            addLog(`AI ${pid} banked ${autoResult.info.potBeforeActions}`, "bank");
        }

        if (autoResult.info.roundEndedBySeven) {
            // Bust during AI auto-play — show bust screen
            bustPending = true;
            showBustScreen(autoResult.info.bustDice);
            return;
        }

        if (autoResult.roundDone && !autoResult.gameDone) {
            addLog(`Round ended. Starting round ${game.roundNumber}.`, "round-end");
            logImmunityRolls();
        }
    }
}

function showBustScreen(bustDice) {
    addLog("A 7 was rolled! Pot wiped! Round over!", "bust");

    // Show the bust dice
    if (bustDice) {
        setDie("die-1", bustDice[0]);
        setDie("die-2", bustDice[1]);
        document.getElementById("dice-sum-display").textContent = `= 7`;
    }

    // Update tag
    const tag = document.getElementById("roll-tag");
    tag.textContent = "BUST! 7 ROLLED!";
    tag.className = "roll-tag tag-bust";
    tag.classList.remove("hidden");

    // Show pot as 0
    const potEl = document.getElementById("pot-value");
    potEl.textContent = "0";
    potEl.className = "pot-value pot-zero";

    // Update round/roll display
    document.getElementById("round-num").textContent = game.roundNumber;
    document.getElementById("roll-num").textContent = game.rollNumber;

    // Show immunity as expired (bust only happens post-immunity)
    const badge = document.getElementById("immunity-badge");
    badge.classList.remove("active");
    badge.classList.add("expired");
    badge.textContent = "NO IMMUNITY";

    // Update scoreboard
    updateScoreboard();

    // Hide bank/continue, show "Next Round" button
    document.getElementById("action-area").classList.add("hidden");
    document.getElementById("next-round-area").classList.remove("hidden");
}

function nextRound() {
    if (!game || !bustPending) return;
    bustPending = false;

    document.getElementById("next-round-area").classList.add("hidden");

    game.advanceRound();

    if (game.gameOver) {
        updateUI();
        showGameOver();
        return;
    }

    addLog(`Starting round ${game.roundNumber}.`, "round-end");
    logImmunityRolls();

    // If human is out (shouldn't happen at round start, but just in case)
    autoAdvanceAI();

    updateUI();
}

function updateUI() {
    if (!game) return;
    if (bustPending) return; // Don't overwrite bust screen

    const d1 = game.lastDice[0];
    const d2 = game.lastDice[1];
    const diceSum = d1 + d2;
    const isDoubles = d1 === d2;
    const isSeven = diceSum === 7;
    const immunityActive = game.rollNumber <= 3 && !game.gameOver;

    // Round and roll
    document.getElementById("round-num").textContent =
        game.gameOver ? game.numRounds : game.roundNumber;
    document.getElementById("roll-num").textContent = game.rollNumber || 1;

    // Immunity badge
    const badge = document.getElementById("immunity-badge");
    if (immunityActive) {
        badge.classList.add("active");
        badge.classList.remove("expired");
        badge.textContent = "IMMUNITY";
    } else {
        badge.classList.remove("active");
        badge.classList.add("expired");
        badge.textContent = "NO IMMUNITY";
    }

    // Dice
    setDie("die-1", d1);
    setDie("die-2", d2);
    document.getElementById("dice-sum-display").textContent = `= ${diceSum}`;

    // Roll tag
    const tag = document.getElementById("roll-tag");
    tag.classList.remove("hidden");
    if (isSeven && immunityActive) {
        tag.textContent = "LUCKY 7! +70";
        tag.className = "roll-tag tag-lucky";
    } else if (isSeven) {
        tag.textContent = "BUST! 7 ROLLED!";
        tag.className = "roll-tag tag-bust";
    } else if (isDoubles && !immunityActive) {
        tag.textContent = "DOUBLES! POT x2";
        tag.className = "roll-tag tag-doubles";
    } else if (isDoubles && immunityActive) {
        tag.textContent = `DOUBLES (immunity: +${diceSum})`;
        tag.className = "roll-tag tag-normal";
    } else {
        tag.textContent = `+${diceSum}`;
        tag.className = "roll-tag tag-normal";
    }

    // Pot
    const potEl = document.getElementById("pot-value");
    potEl.textContent = game.pot;
    potEl.className = "pot-value";
    if (game.pot === 0) potEl.classList.add("pot-zero");
    else if (game.pot >= 500) potEl.classList.add("pot-high");
    else if (game.pot >= 200) potEl.classList.add("pot-medium");

    // Scoreboard
    updateScoreboard();

    // Action buttons — only show when decisions are needed and human is in
    const showActions = game.waitingForDecisions && game.playersInRound.has(0) && !game.gameOver;
    if (showActions) {
        document.getElementById("action-area").classList.remove("hidden");
    } else {
        document.getElementById("action-area").classList.add("hidden");
    }

    // Make sure next-round button is hidden during normal play
    document.getElementById("next-round-area").classList.add("hidden");
}

function updateScoreboard() {
    const container = document.getElementById("score-entries");
    container.innerHTML = "";

    const humanIn = game.playersInRound.has(0);
    const humanEl = document.createElement("div");
    humanEl.className = "score-entry " + (humanIn ? "in-round" : "banked-out");
    humanEl.innerHTML = `
        <div class="score-name">
            <span class="player-dot human-dot"></span>
            You
            <span class="status-tag">${humanIn ? "IN" : "OUT"}</span>
        </div>
        <div class="score-val">${game.bankedScores[0]}</div>
    `;
    container.appendChild(humanEl);

    for (let i = 1; i <= numAi; i++) {
        const inRound = game.playersInRound.has(i);
        const el = document.createElement("div");
        el.className = "score-entry " + (inRound ? "in-round" : "banked-out");
        el.innerHTML = `
            <div class="score-name">
                <span class="player-dot ai-dot"></span>
                AI ${i}
                <span class="status-tag">${inRound ? "IN" : "OUT"}</span>
            </div>
            <div class="score-val">${game.bankedScores[i]}</div>
        `;
        container.appendChild(el);
    }
}

function setDie(dieId, value) {
    const die = document.getElementById(dieId);
    die.className = "die die-val-" + value;
    const face = die.querySelector(".die-face");
    face.innerHTML = "";
    for (let i = 0; i < value; i++) {
        const pip = document.createElement("span");
        pip.className = "pip";
        face.appendChild(pip);
    }
}

// ==================== LOG ====================

function addLog(text, cls) {
    const container = document.getElementById("log-entries");
    const entry = document.createElement("div");
    entry.className = "log-entry log-" + (cls || "system");
    entry.textContent = text;
    container.prepend(entry);
}

function clearLog() {
    document.getElementById("log-entries").innerHTML = "";
}

// ==================== GAME OVER ====================

function showGameOver() {
    const overlay = document.getElementById("gameover-overlay");
    overlay.classList.remove("hidden");

    const winners = game.getWinners();
    const humanWon = winners.includes(0);

    const title = document.getElementById("gameover-title");
    if (humanWon && winners.length === 1) {
        title.textContent = "You Win!";
        title.className = "gameover-win";
    } else if (humanWon) {
        title.textContent = "Tie!";
        title.className = "gameover-win";
    } else {
        title.textContent = "You Lost";
        title.className = "gameover-lose";
    }

    const scores = [{ name: "You", score: game.bankedScores[0], id: 0 }];
    for (let i = 1; i <= numAi; i++) {
        scores.push({ name: `AI ${i}`, score: game.bankedScores[i], id: i });
    }
    scores.sort((a, b) => b.score - a.score);

    const scoresDiv = document.getElementById("gameover-scores");
    scoresDiv.innerHTML = "";
    for (const entry of scores) {
        const isWinner = winners.includes(entry.id);
        const row = document.createElement("div");
        row.className = "final-score-row" + (isWinner ? " winner" : "");
        row.innerHTML = `
            <span class="final-name">${entry.name}</span>
            <span class="final-score">${entry.score}</span>
            ${isWinner ? '<span class="crown">&#x1F451;</span>' : ""}
        `;
        scoresDiv.appendChild(row);
    }
}

// ==================== KEYBOARD ====================

document.addEventListener("keydown", (e) => {
    if (!game || game.gameOver) return;
    if (bustPending) {
        if (e.key === "Enter" || e.key === " " || e.key === "n" || e.key === "N") {
            nextRound();
        }
        return;
    }
    if (!game.waitingForDecisions || !game.playersInRound.has(0)) return;
    if (e.key === "b" || e.key === "B") doAction("bank");
    if (e.key === "c" || e.key === "C") doAction("continue");
});

// ==================== INIT ====================

window.addEventListener("DOMContentLoaded", loadWeights);
