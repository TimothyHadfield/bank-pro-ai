// ============================================================
// Play vs AI — supports multiple human players
// ============================================================

class PlayNetwork {
    constructor(weightsData) {
        const w = weightsData.weights;
        this.fc1_w = w["net.0.weight"]; this.fc1_b = w["net.0.bias"];
        this.fc2_w = w["net.2.weight"]; this.fc2_b = w["net.2.bias"];
        this.out_w = w["net.4.weight"]; this.out_b = w["net.4.bias"];
    }
    _lin(w, b, x) {
        const o = new Array(w.length);
        for (let i = 0; i < w.length; i++) { let s = b[i]; for (let j = 0; j < x.length; j++) s += w[i][j]*x[j]; o[i] = s; }
        return o;
    }
    _relu(a) { return a.map(x => Math.max(0, x)); }
    _softmax(a) { const m = Math.max(...a); const e = a.map(x => Math.exp(x-m)); const s = e.reduce((a,b)=>a+b,0); return e.map(x=>x/s); }
    decide(sv, deterministic = false) {
        let h = this._relu(this._lin(this.fc1_w, this.fc1_b, sv));
        h = this._relu(this._lin(this.fc2_w, this.fc2_b, h));
        const p = this._softmax(this._lin(this.out_w, this.out_b, h));
        if (deterministic) return p[0] >= p[1] ? "bank" : "continue";
        return Math.random() < p[0] ? "bank" : "continue";
    }
}

class PlayBankGame {
    constructor(np=4, nr=10) { this.np=np; this.nr=nr; this.reset(); }
    reset() {
        this.rn=1; this.roll=0; this.pot=0; this.scores=new Array(this.np).fill(0);
        this.pir=new Set(); for(let i=0;i<this.np;i++) this.pir.add(i);
        this.over=false; this.ro=false; this.wfd=false; this.dice=[0,0]; this.ih=[];
        this._rollImmunity(); this.wfd = true;
    }
    _ri(a,b){return Math.floor(Math.random()*(b-a+1))+a;}
    _rd(){
        this.roll++;const d1=this._ri(1,6),d2=this._ri(1,6);this.dice=[d1,d2];
        const s=d1+d2,db=d1===d2,sv=s===7,im=this.roll<=3;
        if(im){this.pot+=sv?70:s;this.ro=false;this.wfd=false;}
        else{if(sv){this.pot=0;this.ro=true;this.wfd=false;}else if(db){this.pot*=2;this.ro=false;this.wfd=true;}else{this.pot+=s;this.ro=false;this.wfd=true;}}
    }
    _rollImmunity(){
        this.ih=[];for(let i=0;i<3;i++){this._rd();this.ih.push({dice:[...this.dice],pot:this.pot,roll:this.roll});}
    }
    _newRoundRaw(){
        this.rn++;if(this.rn>this.nr){this.over=true;return;}
        this.roll=0;this.pot=0;this.pir=new Set();for(let i=0;i<this.np;i++)this.pir.add(i);this.ro=false;
        this._rollImmunity();this.wfd=true;
    }
    advRound(){this._newRoundRaw();}
    getSV(pid){
        const my=this.scores[pid], all=[...this.scores];
        const op=[],oi=[];
        for(let i=0;i<this.np;i++){if(i===pid)continue;op.push(this.scores[i]);oi.push(this.pir.has(i)?1:0);}
        const hi=Math.max(...all),lo=Math.min(...all),ho=op.length?Math.max(...op):0;
        let rank=1;for(const s of all)if(s>my)rank++;
        const v=[this.pot/1000,this.roll/10,this.rn/10,my/1000,this.pir.size/6,
            (this.nr-this.rn)/10,this.rn===this.nr?1:0,Math.max(0,this.roll-3)/10,
            rank/this.np,(my-hi)/1000,(my-lo)/1000,(my+this.pot)>=ho?1:0];
        for(let i=0;i<5;i++)v.push(i<op.length?op[i]/1000:0);
        for(let i=0;i<5;i++)v.push(i<oi.length?oi[i]:0);
        return v;
    }
    step(actions){
        const info={dice:[...this.dice],potB:this.pot,banked:[],bust:false,bustDice:null};
        const bk=[];for(const p of this.pir)if(actions[p]==="bank"){this.scores[p]+=this.pot;bk.push(p);}
        info.banked=bk;for(const p of bk)this.pir.delete(p);
        if(this.pir.size===0){this.ro=true;this.wfd=false;this._newRoundRaw();return{rd:true,gd:this.over,info,newRoundStarted:true};}
        this._rd();
        if(this.ro){info.bust=true;info.bustDice=[...this.dice];return{rd:true,gd:false,info};}
        return{rd:false,gd:false,info};
    }
    winners(){const m=Math.max(...this.scores);return this.scores.map((_,i)=>i).filter(i=>this.scores[i]===m);}
}

// ==================== PLAY UI STATE ====================
let playNet = null;
let playGame = null;
let playNumAi = 3;
let playNumHumans = 1;
let playBust = false;
let playAnimating = false;
let playAnimateEnabled = true;
const ANIM_DELAY = 2000;

// Multi-human state
let pendingActions = {};    // collected human decisions so far this turn
let humansToDecide = [];    // queue of human player IDs still needing input
let currentHuman = -1;      // which human is currently deciding

function isHuman(pid) { return pid < playNumHumans; }
function humanName(pid) { return playNumHumans === 1 ? "You" : `Player ${pid + 1}`; }
function anyHumanInRound() { for (let i = 0; i < playNumHumans; i++) if (playGame.pir.has(i)) return true; return false; }

async function playLoadModel(checkpoint) {
    if (!checkpoint) { document.getElementById("play-model-status").textContent = "No checkpoint selected"; return; }
    document.getElementById("play-model-status").textContent = "Loading model...";
    document.getElementById("play-start-btn").disabled = true;
    try {
        const resp = await fetch("/api/weights/" + checkpoint);
        const buf = await resp.arrayBuffer();
        const text = new TextDecoder().decode(buf);
        const data = JSON.parse(text);
        if (data.error) throw new Error(data.error);
        playNet = new PlayNetwork(data);
        document.getElementById("play-model-status").textContent = `Loaded: ${checkpoint} (batch ${data.batch_num})`;
        document.getElementById("play-start-btn").disabled = false;
    } catch (e) {
        document.getElementById("play-model-status").textContent = "Failed to load model: " + e.message;
        playNet = null;
        document.getElementById("play-start-btn").disabled = true;
    }
}

function playStart() {
    playNumHumans = parseInt(document.getElementById("play-num-humans").value);
    playNumAi = parseInt(document.getElementById("play-num-ai").value);
    playAnimateEnabled = document.getElementById("play-animate-toggle").checked;
    const np = playNumHumans + playNumAi;
    if (np < 2 || np > 6) { alert("Total players must be 2-6"); return; }

    playGame = new PlayBankGame(np, 10);
    playBust = false;
    playAnimating = false;
    pendingActions = {};
    humansToDecide = [];
    currentHuman = -1;

    document.getElementById("play-setup").classList.add("hidden");
    document.getElementById("play-area").classList.remove("hidden");
    document.getElementById("play-gameover").classList.add("hidden");

    playLogClear();
    playLog(`Game started! ${playNumHumans} human(s) vs ${playNumAi} AI. Round 1.`, "sys");
    playShowRoundStart();
}

async function playShowRoundStart() {
    playAnimating = true;
    document.getElementById("play-buttons").classList.add("hidden");
    document.getElementById("play-turn-label").classList.add("hidden");

    if (playAnimateEnabled) {
        for (let i = 0; i < playGame.ih.length; i++) {
            const r = playGame.ih[i], d = r.dice, s = d[0] + d[1];
            playSetDie("play-d1", d[0]); playSetDie("play-d2", d[1]);
            document.getElementById("play-dice-sum").textContent = `= ${s}`;
            document.getElementById("play-round").textContent = playGame.rn;
            document.getElementById("play-roll").textContent = r.roll;
            const imm = document.getElementById("play-imm");
            imm.textContent = "IMMUNITY"; imm.className = "play-imm imm-on";
            const tag = document.getElementById("play-tag");
            if (s === 7) { tag.textContent = "+70"; tag.className = "play-tag tag-lucky"; }
            else { tag.textContent = `+${s}`; tag.className = "play-tag tag-normal"; }
            const pe = document.getElementById("play-pot-val");
            pe.textContent = r.pot;
            pe.className = "play-pot-val" + (r.pot >= 500 ? " pot-high" : r.pot >= 200 ? " pot-med" : "");
            playLog(`+${s === 7 ? "70" : s} \u2192 Pot: ${r.pot}`, s === 7 ? "lucky" : "sys");
            playUpdateScores();
            await playDelay(ANIM_DELAY);
        }
    } else {
        for (let i = playGame.ih.length - 1; i >= 0; i--) {
            const r = playGame.ih[i], s = r.dice[0] + r.dice[1];
            playLog(`+${s === 7 ? "70" : s} \u2192 Pot: ${r.pot}`, s === 7 ? "lucky" : "sys");
        }
        playLog("Immunity (auto-rolled):", "imm");
    }

    playAnimating = false;
    const lastIm = playGame.ih[playGame.ih.length - 1];
    playSetDie("play-d1", lastIm.dice[0]); playSetDie("play-d2", lastIm.dice[1]);
    document.getElementById("play-dice-sum").textContent = `= ${lastIm.dice[0]+lastIm.dice[1]}`;
    document.getElementById("play-roll").textContent = "3";
    const imm = document.getElementById("play-imm");
    imm.textContent = "IMMUNITY ENDED"; imm.className = "play-imm imm-off";
    const tag = document.getElementById("play-tag");
    tag.textContent = "Bank or risk it?"; tag.className = "play-tag tag-normal";
    const pe = document.getElementById("play-pot-val");
    pe.textContent = playGame.pot;
    pe.className = "play-pot-val" + (playGame.pot >= 500 ? " pot-high" : playGame.pot >= 200 ? " pot-med" : "");
    playUpdateScores();
    promptNextHuman();
}

function playDelay(ms) { return new Promise(r => setTimeout(r, ms)); }

// ==================== MULTI-HUMAN DECISION FLOW ====================

function promptNextHuman() {
    // Find which humans still need to decide this turn
    if (humansToDecide.length === 0) {
        // Start a new decision round: collect all humans still in the round
        pendingActions = {};
        humansToDecide = [];
        for (let i = 0; i < playNumHumans; i++) {
            if (playGame.pir.has(i)) humansToDecide.push(i);
        }
    }

    if (humansToDecide.length === 0) {
        // No humans in round — go straight to AI auto-advance
        playAutoAdvanceAI();
        return;
    }

    // Ask the next human
    currentHuman = humansToDecide[0];
    const label = document.getElementById("play-turn-label");
    if (playNumHumans > 1) {
        label.textContent = `${humanName(currentHuman)}'s turn`;
        label.classList.remove("hidden");
    } else {
        label.classList.add("hidden");
    }
    document.getElementById("play-buttons").classList.remove("hidden");
}

function playBank() { playHumanDecision("bank"); }
function playContinue() { playHumanDecision("continue"); }

function playHumanDecision(action) {
    if (!playGame || playGame.over || !playGame.wfd || playBust || playAnimating) return;
    if (currentHuman < 0) return;

    pendingActions[currentHuman] = action;
    if (action === "bank") playLog(`${humanName(currentHuman)} banked!`, "bank");

    document.getElementById("play-buttons").classList.add("hidden");
    document.getElementById("play-turn-label").classList.add("hidden");

    // Remove this human from the queue
    humansToDecide.shift();

    if (humansToDecide.length > 0) {
        // More humans need to decide
        promptNextHuman();
    } else {
        // All humans decided — now get AI decisions and step
        executeStep();
    }
}

async function executeStep() {
    const acts = { ...pendingActions };

    // AI decisions
    for (const pid of playGame.pir) {
        if (isHuman(pid)) continue; // already in pendingActions
        const a = playNet ? playNet.decide(playGame.getSV(pid)) : (Math.random() < 0.3 ? "bank" : "continue");
        acts[pid] = a;
    }

    const res = playGame.step(acts);

    // Log banks
    for (const p of res.info.banked) {
        if (isHuman(p)) {
            playLog(`${humanName(p)} banked ${res.info.potB}!`, "bank");
        } else {
            playLog(`AI ${p - playNumHumans + 1} banked ${res.info.potB}!`, "bank");
        }
    }

    if (res.info.bust) { playBust = true; playShowBust(res.info.bustDice); return; }

    // Log roll result
    if (!res.rd) {
        const d = playGame.dice, s = d[0] + d[1], isD = d[0] === d[1];
        if (isD) playLog(`Doubles! Pot x2 \u2192 ${playGame.pot}`, "lucky");
        else playLog(`+${s} \u2192 Pot: ${playGame.pot}`, "sys");
    }

    if (res.rd && !res.gd && res.newRoundStarted) {
        playLog(`Round ended. Starting round ${playGame.rn}.`, "rnd");
        await playShowRoundStart();
        if (playGame.over) playShowGameOver();
        return;
    }

    // If no humans left in round, auto-advance AI
    if (!anyHumanInRound()) {
        await playAutoAdvanceAI();
        playUpdateUI();
        if (playGame.over) playShowGameOver();
        return;
    }

    playUpdateUI();
    if (playGame.over) { playShowGameOver(); return; }

    // Reset for next decision round
    humansToDecide = [];
    promptNextHuman();
}

async function playAutoAdvanceAI() {
    while (!playGame.over && playGame.wfd && !anyHumanInRound() && !playBust) {
        const aa = {};
        for (const p of playGame.pir) {
            const a = playNet ? playNet.decide(playGame.getSV(p)) : (Math.random()<0.3?"bank":"continue");
            aa[p] = a;
        }

        if (playAnimateEnabled) {
            playAnimating = true;
            playUpdateUI();
            await playDelay(ANIM_DELAY);
        }

        const r2 = playGame.step(aa);
        for (const p of r2.info.banked) playLog(`AI ${p - playNumHumans + 1} banked ${r2.info.potB}`, "bank");

        if (!r2.rd && !r2.info.bust) {
            const d = playGame.dice, s = d[0] + d[1], isD = d[0] === d[1];
            if (isD) playLog(`Doubles! Pot x2 \u2192 ${playGame.pot}`, "lucky");
            else playLog(`+${s} \u2192 Pot: ${playGame.pot}`, "sys");
        }

        if (r2.info.bust) {
            playAnimating = false;
            playBust = true; playShowBust(r2.info.bustDice); return;
        }

        if (r2.rd && !r2.gd && r2.newRoundStarted) {
            playLog(`Round ended. Starting round ${playGame.rn}.`, "rnd");
            playAnimating = false;
            await playShowRoundStart();
            if (playGame.over) { playShowGameOver(); return; }
            return;
        }

        if (playAnimateEnabled) { playUpdateUI(); await playDelay(ANIM_DELAY); }
    }
    playAnimating = false;
}

function playNextRound() {
    if (!playBust) return;
    playBust = false;
    document.getElementById("play-next-round").classList.add("hidden");
    playGame.advRound();
    if (playGame.over) { playUpdateUI(); playShowGameOver(); return; }
    playLog(`Starting round ${playGame.rn}.`, "rnd");
    playShowRoundStart();
}

function playShowBust(dice) {
    playLog("7 rolled! Pot wiped! Round over!", "bust");
    document.getElementById("play-buttons").classList.add("hidden");
    document.getElementById("play-turn-label").classList.add("hidden");
    document.getElementById("play-next-round").classList.remove("hidden");
    playSetDie("play-d1", dice[0]); playSetDie("play-d2", dice[1]);
    document.getElementById("play-dice-sum").textContent = "= 7";
    document.getElementById("play-pot-val").textContent = "0";
    document.getElementById("play-pot-val").className = "play-pot-val pot-zero";
    document.getElementById("play-tag").textContent = "BUST! 7 ROLLED!";
    document.getElementById("play-tag").className = "play-tag tag-bust";
    playUpdateScores();
}

function playShowGameOver() {
    document.getElementById("play-buttons").classList.add("hidden");
    document.getElementById("play-turn-label").classList.add("hidden");
    document.getElementById("play-next-round").classList.add("hidden");
    document.getElementById("play-gameover").classList.remove("hidden");
    const w = playGame.winners();
    const anyHumanWon = w.some(id => isHuman(id));
    const title = document.getElementById("play-go-title");
    if (anyHumanWon) {
        const winners = w.filter(id => isHuman(id)).map(id => humanName(id));
        title.textContent = winners.length === w.length && w.length === 1 ? winners[0] + " Wins!" : "Tie — " + winners.join(", ");
        title.className = "go-title go-win";
    } else {
        title.textContent = "AI Wins";
        title.className = "go-title go-lose";
    }

    const scores = [];
    for (let i = 0; i < playNumHumans; i++) scores.push({name: humanName(i), score: playGame.scores[i], id: i});
    for (let i = playNumHumans; i < playGame.np; i++) scores.push({name: `AI ${i - playNumHumans + 1}`, score: playGame.scores[i], id: i});
    scores.sort((a,b) => b.score - a.score);

    document.getElementById("play-go-scores").innerHTML = scores.map(e => {
        const isW = w.includes(e.id);
        return `<div class="go-row${isW?" go-winner":""}"><span>${e.name}</span><span class="go-sc">${e.score}</span></div>`;
    }).join("");
}

function playUpdateUI() {
    if (!playGame || playBust) return;
    const g = playGame;
    document.getElementById("play-round").textContent = g.over ? g.nr : g.rn;
    document.getElementById("play-roll").textContent = g.roll || 1;

    const imm = document.getElementById("play-imm");
    if (g.roll <= 3 && !g.over) { imm.textContent = "IMMUNITY"; imm.className = "play-imm imm-on"; }
    else { imm.textContent = "NO IMMUNITY"; imm.className = "play-imm imm-off"; }

    playSetDie("play-d1", g.dice[0]); playSetDie("play-d2", g.dice[1]);
    const ds = g.dice[0]+g.dice[1];
    document.getElementById("play-dice-sum").textContent = `= ${ds}`;

    const tag = document.getElementById("play-tag");
    const isD = g.dice[0]===g.dice[1], is7 = ds===7, im = g.roll<=3;
    if (is7&&im) { tag.textContent="+70"; tag.className="play-tag tag-lucky"; }
    else if (is7) { tag.textContent="BUST! 7 ROLLED!"; tag.className="play-tag tag-bust"; }
    else if (isD&&!im) { tag.textContent="DOUBLES! POT x2"; tag.className="play-tag tag-doubles"; }
    else { tag.textContent=`+${ds}`; tag.className="play-tag tag-normal"; }

    const pe = document.getElementById("play-pot-val");
    pe.textContent = g.pot;
    pe.className = "play-pot-val" + (g.pot===0?" pot-zero":g.pot>=500?" pot-high":g.pot>=200?" pot-med":"");

    playUpdateScores();
    // Don't show buttons here — promptNextHuman handles that
}

function playUpdateScores() {
    const c = document.getElementById("play-scores");
    let h = "";
    for (let i = 0; i < playNumHumans; i++) {
        h += `<div class="ps-row${playGame.pir.has(i)?" ps-in":""}"><span class="ps-dot ps-human"></span><span>${humanName(i)}</span><span class="ps-val">${playGame.scores[i]}</span></div>`;
    }
    for (let i = playNumHumans; i < playGame.np; i++) {
        h += `<div class="ps-row${playGame.pir.has(i)?" ps-in":""}"><span class="ps-dot ps-ai"></span><span>AI ${i - playNumHumans + 1}</span><span class="ps-val">${playGame.scores[i]}</span></div>`;
    }
    c.innerHTML = h;
}

function playSetDie(id, v) {
    const d = document.getElementById(id);
    d.className = "play-die dv-"+v;
    d.innerHTML = "";
    for (let i=0;i<v;i++) { const p=document.createElement("span"); p.className="pip"; d.appendChild(p); }
}

function playLog(text, cls) {
    const c = document.getElementById("play-log");
    const e = document.createElement("div");
    e.className = "pl-entry pl-" + (cls||"sys");
    e.textContent = text;
    c.prepend(e);
}
function playLogClear() { document.getElementById("play-log").innerHTML = ""; }

function playBackToSetup() {
    playAnimating = false;
    document.getElementById("play-setup").classList.remove("hidden");
    document.getElementById("play-area").classList.add("hidden");
    document.getElementById("play-gameover").classList.add("hidden");
}

document.addEventListener("keydown", e => {
    const tab = document.getElementById("tab-play");
    if (!tab || tab.classList.contains("hidden")) return;
    if (playAnimating) return;
    if (playBust && (e.key==="Enter"||e.key===" "||e.key==="n")) { playNextRound(); return; }
    if (!playGame || playGame.over || !playGame.wfd || currentHuman < 0) return;
    if (e.key==="b"||e.key==="B") playBank();
    if (e.key==="c"||e.key==="C") playContinue();
});

async function playLoadCheckpoints() {
    try {
        const resp = await fetch("/api/checkpoints");
        if (!resp.ok) return;
        const text = await resp.text();
        let data;
        try { data = JSON.parse(text); } catch { return; }
        const sel = document.getElementById("play-checkpoint");
        if (!data.checkpoints || data.checkpoints.length === 0) {
            sel.innerHTML = '<option value="">No checkpoints available</option>';
            document.getElementById("play-model-status").textContent = "No checkpoints found. Train the AI first.";
            return;
        }
        sel.innerHTML = data.checkpoints.map(cp => `<option value="${cp.filename}">${cp.filename}</option>`).join("");
        const latest = data.checkpoints[data.checkpoints.length - 1].filename;
        sel.value = latest;
        playLoadModel(latest);
    } catch(e) {
        document.getElementById("play-model-status").textContent = "Could not load checkpoints.";
    }
}
