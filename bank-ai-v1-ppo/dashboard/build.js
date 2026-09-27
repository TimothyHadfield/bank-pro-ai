// ============================================================
// Build AI — write Python-style rules, test against bots
// ============================================================

// Built-in variable names that should NOT be treated as custom vars
const BUILTIN_VARS = new Set([
    "pot", "round", "roll", "players_in", "my_score", "num_players",
    "p1_score", "p2_score", "p3_score", "p4_score", "p5_score",
    "p1_banked", "p2_banked", "p3_banked", "p4_banked", "p5_banked",
    "bank", "continue", "if", "elif", "else", "and", "or", "not",
    "true", "false", "True", "False", "for", "in", "range",
]);

// Detect settable variables in code: only single uppercase letters (A-Z)
function detectCustomVars(code) {
    const vars = new Set();
    const words = code.match(/\b[A-Z]\b/g) || [];
    for (const w of words) {
        if (!BUILTIN_VARS.has(w)) {
            vars.add(w);
        }
    }
    return [...vars].sort();
}

// Parse a variable assignment string like "100-1000, 50" or "80, 100, 200"
function parseVarValues(str) {
    str = str.trim();
    if (!str) return [0];

    // Check for range format: "start-end, step"
    const rangeMatch = str.match(/^(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)$/);
    if (rangeMatch) {
        const start = parseFloat(rangeMatch[1]);
        const end = parseFloat(rangeMatch[2]);
        const step = parseFloat(rangeMatch[3]);
        if (step <= 0) return [start];
        const vals = [];
        for (let v = start; v <= end + 0.0001; v += step) {
            vals.push(Math.round(v * 1000) / 1000);
        }
        return vals.length > 0 ? vals : [start];
    }

    // Comma-separated values: "80, 100, 200"
    const parts = str.split(",").map(s => s.trim()).filter(s => s);
    const vals = parts.map(s => parseFloat(s)).filter(v => !isNaN(v));
    return vals.length > 0 ? vals : [0];
}

// Get stored variable values for all AIs
function getVarAssignments() {
    return JSON.parse(localStorage.getItem("buildAI_vars") || "{}");
}
function setVarAssignments(vars) {
    localStorage.setItem("buildAI_vars", JSON.stringify(vars));
}

// Collect all custom vars across ALL saved AIs
function getAllCustomVars() {
    const saves = getAISaves();
    const allVars = new Set();
    for (const code of Object.values(saves)) {
        for (const v of detectCustomVars(code)) allVars.add(v);
    }
    // Also include vars from the current editor
    const editorCode = document.getElementById("build-code").value;
    for (const v of detectCustomVars(editorCode)) allVars.add(v);
    return [...allVars].sort();
}

// Update the variables panel — always shows all vars from all AIs
function buildUpdateVarsPanel() {
    const customVars = getAllCustomVars();
    const panel = document.getElementById("build-vars-panel");
    const list = document.getElementById("build-vars-list");

    if (customVars.length === 0) {
        panel.classList.add("hidden");
        return;
    }

    panel.classList.remove("hidden");
    const assignments = getVarAssignments();

    list.innerHTML = customVars.map(v => {
        const saved = assignments[v] || "";
        return `
            <div class="build-var-row">
                <span class="build-var-name">${v}</span>
                <span class="build-var-eq">=</span>
                <input class="build-var-input" data-var="${v}" value="${saved}" placeholder="e.g. 200  or  80, 100, 200  or  100-500, 50" onchange="buildSaveVarValue('${v}', this.value)">
            </div>
        `;
    }).join("") + '<div class="build-var-hint">Single value, comma-separated, or range: start-end, step</div>';
}

function buildSaveVarValue(varName, value) {
    const assignments = getVarAssignments();
    assignments[varName] = value;
    setVarAssignments(assignments);
    buildRefreshCompeteSelect();
}

// Generate all variant combinations for an AI
function getAIVariants(name, code) {
    const customVars = detectCustomVars(code);
    if (customVars.length === 0) {
        return [{ name, varValues: {} }];
    }

    const assignments = getVarAssignments();
    const varArrays = customVars.map(v => {
        const raw = assignments[v] || "0";
        return { name: v, values: parseVarValues(raw) };
    });

    // Cartesian product of all variable values
    const variants = [];
    function expand(idx, current) {
        if (idx >= varArrays.length) {
            const label = customVars.map(v => current[v]).join(", ");
            variants.push({ name: `${name} (${label})`, varValues: { ...current } });
            return;
        }
        for (const val of varArrays[idx].values) {
            current[varArrays[idx].name] = val;
            expand(idx + 1, current);
        }
    }
    expand(0, {});
    return variants;
}

// Compile code with custom variable values substituted
function compileBuildCodeWithVars(code, varValues) {
    // Replace custom variable names with their values in the code
    let processed = code;
    for (const [name, value] of Object.entries(varValues)) {
        processed = processed.replace(new RegExp(`\\b${name}\\b`, 'g'), String(value));
    }
    return compileBuildCode(processed);
}

// Parse and compile user code into a decision function
function compileBuildCode(code) {
    // Convert Python-style to JS
    let js = "";
    const lines = code.split("\n");
    const indentStack = [0]; // stack of indent levels that opened blocks

    for (let i = 0; i < lines.length; i++) {
        let line = lines[i];
        const trimmed = line.trimEnd();
        if (trimmed === "" || trimmed.startsWith("#")) continue;

        // Count indent (spaces)
        const indent = line.length - line.trimStart().length;
        const content = trimmed.trim();

        // Close blocks that are deeper than current indent
        // For elif/else, we close down to one level above (the matching if's block)
        const isElse = content.startsWith("elif ") || content === "else:" || content === "else";

        if (isElse) {
            // Close blocks until we're at the same level as this elif/else
            while (indentStack.length > 1 && indentStack[indentStack.length - 1] > indent) {
                js += "}\n";
                indentStack.pop();
            }
            // Pop the matching if/elif block level
            if (indentStack.length > 1 && indentStack[indentStack.length - 1] === indent) {
                // Don't pop — elif/else continues at this level
            }
        } else {
            // Regular statement: close all blocks deeper than current indent
            while (indentStack.length > 1 && indentStack[indentStack.length - 1] > indent) {
                js += "}\n";
                indentStack.pop();
            }
        }

        // Handle actions
        if (content === "bank") {
            js += "return 'bank';\n";
            continue;
        }
        if (content === "continue") {
            js += "return 'continue';\n";
            continue;
        }

        // Handle variable assignment: X = expression
        const assignMatch = content.match(/^([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.+)$/);
        if (assignMatch && !content.endsWith(":") && assignMatch[1] !== "if" && assignMatch[1] !== "elif") {
            const varName = assignMatch[1];
            const expr = convertCondition(assignMatch[2]);
            js += `var ${varName} = ${expr};\n`;
            continue;
        }

        // Handle for loop: for X in range(N): or for X in range(A, B): or for X in range(A, B, step):
        const forMatch = content.match(/^for\s+([A-Za-z_][A-Za-z0-9_]*)\s+in\s+range\((.+)\):$/);
        if (forMatch) {
            const loopVar = forMatch[1];
            const args = forMatch[2].split(",").map(s => convertCondition(s.trim()));
            if (args.length === 1) {
                js += `for (var ${loopVar} = 0; ${loopVar} < ${args[0]}; ${loopVar}++) {\n`;
            } else if (args.length === 2) {
                js += `for (var ${loopVar} = ${args[0]}; ${loopVar} < ${args[1]}; ${loopVar}++) {\n`;
            } else {
                js += `for (var ${loopVar} = ${args[0]}; ${loopVar} < ${args[1]}; ${loopVar} += ${args[2]}) {\n`;
            }
            indentStack.push(indent);
            continue;
        }

        // Handle if/elif/else
        if (content.startsWith("if ") && content.endsWith(":")) {
            const cond = convertCondition(content.slice(3, -1).trim());
            js += `if (${cond}) {\n`;
            indentStack.push(indent);
        } else if (content.startsWith("elif ") && content.endsWith(":")) {
            const cond = convertCondition(content.slice(5, -1).trim());
            js += `} else if (${cond}) {\n`;
        } else if (content === "else:" || content === "else") {
            js += `} else {\n`;
        } else {
            throw new Error(`Line ${i + 1}: Unknown statement: "${content}"`);
        }
    }

    // Close remaining blocks
    while (indentStack.length > 1) {
        js += "}\n";
        indentStack.pop();
    }

    js += "\nreturn 'continue';";

    try {
        return new Function(
            "pot", "round", "roll", "players_in", "my_score", "num_players",
            "p1_score", "p2_score", "p3_score", "p4_score", "p5_score",
            "p1_banked", "p2_banked", "p3_banked", "p4_banked", "p5_banked",
            js
        );
    } catch (e) {
        throw new Error("Compilation error: " + e.message);
    }
}

function convertCondition(cond) {
    // Replace Python operators with JS
    cond = cond.replace(/\band\b/g, "&&");
    cond = cond.replace(/\bor\b/g, "||");
    cond = cond.replace(/\bnot\b/g, "!");
    return cond;
}

// Run the custom AI in a game engine
class BuildBankGame {
    constructor(np = 4) { this.np = np; this.reset(); }
    reset() {
        this.rn = 1; this.roll = 0; this.pot = 0;
        this.scores = new Array(this.np).fill(0);
        this.pir = new Set(); for (let i = 0; i < this.np; i++) this.pir.add(i);
        this.over = false; this.ro = false; this.wfd = false; this.dice = [0, 0];
        this._rollImmunity(); this.wfd = true;
    }
    _ri(a, b) { return Math.floor(Math.random() * (b - a + 1)) + a; }
    _rd() {
        this.roll++;
        const d1 = this._ri(1, 6), d2 = this._ri(1, 6);
        this.dice = [d1, d2]; const s = d1 + d2;
        if (this.roll <= 3) { this.pot += (s === 7) ? 70 : s; this.ro = false; this.wfd = false; }
        else { if (s === 7) { this.pot = 0; this.ro = true; this.wfd = false; } else if (d1 === d2) { this.pot *= 2; this.ro = false; this.wfd = true; } else { this.pot += s; this.ro = false; this.wfd = true; } }
    }
    _rollImmunity() { for (let i = 0; i < 3; i++) this._rd(); this.wfd = true; }
    _newRound() {
        this.rn++; if (this.rn > 10) { this.over = true; return; }
        this.roll = 0; this.pot = 0; this.pir = new Set();
        for (let i = 0; i < this.np; i++) this.pir.add(i);
        this.ro = false; this._rollImmunity(); this.wfd = true;
    }
    step(actions) {
        const bk = [];
        for (const p of this.pir) if (actions[p] === "bank") { this.scores[p] += this.pot; bk.push(p); }
        for (const p of bk) this.pir.delete(p);
        if (this.pir.size === 0) { this.ro = true; this.wfd = false; this._newRound(); return; }
        this._rd();
        if (this.ro) { this._newRound(); }
    }
    getVars(pid) {
        // Sort all scores descending for p1-p5
        const allScores = [...this.scores];
        const sortedScores = allScores.slice().sort((a, b) => b - a);
        // Banking status sorted by rank
        const ranked = allScores.map((s, i) => ({ s, i })).sort((a, b) => b.s - a.s);
        const bankedStatus = ranked.map(r => this.pir.has(r.i) ? 0 : 1);

        return {
            pot: this.pot,
            round: this.rn,
            roll: this.roll,
            players_in: this.pir.size,
            my_score: this.scores[pid],
            num_players: this.np,
            p1_score: sortedScores[0] || 0,
            p2_score: sortedScores[1] || 0,
            p3_score: sortedScores[2] || 0,
            p4_score: sortedScores[3] || 0,
            p5_score: sortedScores[4] || 0,
            p1_banked: bankedStatus[0] || 0,
            p2_banked: bankedStatus[1] || 0,
            p3_banked: bankedStatus[2] || 0,
            p4_banked: bankedStatus[3] || 0,
            p5_banked: bankedStatus[4] || 0,
        };
    }
    winners() { const m = Math.max(...this.scores); return this.scores.map((_, i) => i).filter(i => this.scores[i] === m); }
}

// Bot opponents for testing
const BUILD_BOTS = [
    { name: "Threshold200", decide: (g, p) => g.pot > 200 ? "bank" : "continue" },
    { name: "Threshold300", decide: (g, p) => g.pot > 300 ? "bank" : "continue" },
    { name: "Threshold400", decide: (g, p) => g.pot > 400 ? "bank" : "continue" },
    { name: "Random30", decide: (g, p) => Math.random() < 0.3 ? "bank" : "continue" },
    { name: "Positional", decide: (g, p) => {
        const my = g.scores[p], maxO = Math.max(...g.scores.filter((_, i) => i !== p));
        if (my > maxO) return g.pot > 150 ? "bank" : "continue";
        return g.pot > Math.max(200, (maxO - my) * 0.5) ? "bank" : "continue";
    }},
];

function buildTest() {
    const code = document.getElementById("build-code").value;
    const numGames = parseInt(document.getElementById("build-num-games").value);
    const errDiv = document.getElementById("build-error");
    const resDiv = document.getElementById("build-results");
    errDiv.classList.add("hidden");
    resDiv.classList.add("hidden");

    // Get current variable values (use first value of each for testing)
    const customVars = detectCustomVars(code);
    const assignments = getVarAssignments();
    const testVals = {};
    for (const v of customVars) {
        const vals = parseVarValues(assignments[v] || "0");
        testVals[v] = vals[0]; // Use first value for single test
    }

    let decideFn;
    try {
        decideFn = compileBuildCodeWithVars(code, testVals);
    } catch (e) {
        errDiv.textContent = e.message;
        errDiv.classList.remove("hidden");
        return;
    }

    // Test: run against each bot type
    const results = [];
    for (const bot of BUILD_BOTS) {
        let wins = 0, totalScore = 0;
        for (let g = 0; g < numGames; g++) {
            const game = new BuildBankGame(4);
            // Assign bots: player 0 = custom AI, players 1-3 = this bot
            while (!game.over) {
                if (!game.wfd) break;
                const actions = {};
                for (const pid of game.pir) {
                    if (pid === 0) {
                        const v = game.getVars(0);
                        try {
                            const result = decideFn(v.pot, v.round, v.roll, v.players_in, v.my_score, v.num_players,
                                v.p1_score, v.p2_score, v.p3_score, v.p4_score, v.p5_score,
                                v.p1_banked, v.p2_banked, v.p3_banked, v.p4_banked, v.p5_banked);
                            actions[pid] = result === "bank" ? "bank" : "continue";
                        } catch (e) {
                            errDiv.textContent = "Runtime error: " + e.message;
                            errDiv.classList.remove("hidden");
                            return;
                        }
                    } else {
                        actions[pid] = bot.decide(game, pid);
                    }
                }
                game.step(actions);
            }
            if (game.winners().includes(0)) wins++;
            totalScore += game.scores[0];
        }
        results.push({
            opponent: bot.name,
            winRate: (wins / numGames * 100).toFixed(1),
            avgScore: Math.round(totalScore / numGames),
        });
    }

    // Overall
    const totalWins = results.reduce((s, r) => s + parseFloat(r.winRate), 0);
    const avgWR = (totalWins / results.length).toFixed(1);
    const avgScore = Math.round(results.reduce((s, r) => s + r.avgScore, 0) / results.length);

    resDiv.innerHTML = `
        <h3>Results (${numGames.toLocaleString()} games each)</h3>
        <div class="build-summary">
            <div class="stat-card"><span class="stat-label">Overall Win Rate</span><span class="stat-val">${avgWR}%</span></div>
            <div class="stat-card"><span class="stat-label">Overall Avg Score</span><span class="stat-val">${avgScore}</span></div>
        </div>
        <table class="build-table">
            <tr><th>Opponent</th><th>Win Rate</th><th>Avg Score</th></tr>
            ${results.map(r => `<tr><td>${r.opponent}</td><td>${r.winRate}%</td><td>${r.avgScore}</td></tr>`).join("")}
        </table>
    `;
    resDiv.classList.remove("hidden");
}

// ==================== AI LIST MANAGEMENT ====================
let currentAIName = null;

function getAISaves() {
    return JSON.parse(localStorage.getItem("buildAI_saves") || "{}");
}

function setAISaves(saves) {
    localStorage.setItem("buildAI_saves", JSON.stringify(saves));
}

function buildNewAI() {
    const name = prompt("Name for the new AI:");
    if (!name || !name.trim()) return;
    const saves = getAISaves();
    if (saves[name]) { alert("An AI with that name already exists."); return; }
    saves[name] = "if pot >= 200:\n    bank\nelse:\n    continue";
    setAISaves(saves);
    buildOpenAI(name);
    buildRefreshList();
    buildRefreshCompeteSelect();
}

function buildOpenAI(name) {
    const saves = getAISaves();
    if (!saves[name]) return;
    currentAIName = name;
    document.getElementById("build-code").value = saves[name];
    document.getElementById("build-editing-name").textContent = name;
    document.getElementById("build-editing-label").classList.remove("hidden");
    document.getElementById("build-error").classList.add("hidden");
    document.getElementById("build-results").classList.add("hidden");
    buildUpdateVarsPanel();
    buildRefreshList();
}

function buildSaveCurrent() {
    if (!currentAIName) {
        // No AI selected — create new
        buildNewAI();
        return;
    }
    const saves = getAISaves();
    saves[currentAIName] = document.getElementById("build-code").value;
    setAISaves(saves);

    // Brief visual feedback
    const btn = document.getElementById("build-save-btn");
    btn.textContent = "Saved!";
    btn.style.background = "#34d399";
    setTimeout(() => { btn.textContent = "Save"; btn.style.background = ""; }, 1000);
    buildUpdateVarsPanel();
    buildRefreshCompeteSelect();
}

function buildDeleteAI(name, e) {
    e.stopPropagation();
    const saves = getAISaves();
    delete saves[name];
    setAISaves(saves);
    if (currentAIName === name) {
        currentAIName = null;
        document.getElementById("build-code").value = "";
        document.getElementById("build-editing-label").classList.add("hidden");
        document.getElementById("build-results").classList.add("hidden");
    }
    buildRefreshList();
    buildRefreshCompeteSelect();
}

function buildRenameAI(name, e) {
    e.stopPropagation();
    const newName = prompt("New name:", name);
    if (!newName || !newName.trim() || newName === name) return;
    const saves = getAISaves();
    if (saves[newName]) { alert("An AI with that name already exists."); return; }
    saves[newName] = saves[name];
    delete saves[name];
    setAISaves(saves);
    if (currentAIName === name) currentAIName = newName;
    document.getElementById("build-editing-name").textContent = currentAIName || "";
    buildRefreshList();
    buildRefreshCompeteSelect();
}

function buildRefreshList() {
    const saves = getAISaves();
    const list = document.getElementById("build-ai-list");
    const names = Object.keys(saves);

    if (names.length === 0) {
        list.innerHTML = '<p class="text-dim" style="font-size:0.78rem;">No AIs yet</p>';
        return;
    }

    list.innerHTML = names.map(name => `
        <div class="build-ai-item ${name === currentAIName ? 'active' : ''}" onclick="buildOpenAI('${name.replace(/'/g, "\\'")}')">
            <span>${name}</span>
            <span class="bai-actions">
                <button class="btn btn-tiny" onclick="buildRenameAI('${name.replace(/'/g, "\\'")}', event)" style="opacity:0;transition:opacity 0.15s;">Rename</button>
                <button class="btn btn-tiny bai-del" onclick="buildDeleteAI('${name.replace(/'/g, "\\'")}', event)">Del</button>
            </span>
        </div>
    `).join("");
    // Show buttons on hover
    list.querySelectorAll('.build-ai-item').forEach(item => {
        item.addEventListener('mouseenter', () => item.querySelectorAll('.bai-actions .btn').forEach(b => b.style.opacity = '1'));
        item.addEventListener('mouseleave', () => item.querySelectorAll('.bai-actions .btn').forEach(b => b.style.opacity = '0'));
    });
}

// ==================== COMPETE ====================

// Track which base AIs are enabled and their variable filters
let competeFilters = {}; // baseName -> { enabled: bool, varFilters: { varName: "filterString" } }

function buildRefreshCompeteSelect() {
    const saves = getAISaves();
    const names = Object.keys(saves);
    const filtersDiv = document.getElementById("compete-filters");
    const selectDiv = document.getElementById("compete-select");
    if (!filtersDiv || !selectDiv) return;

    // Init filters for new AIs
    for (const name of names) {
        if (!competeFilters[name]) competeFilters[name] = { enabled: false, varFilters: {} };
    }
    // Remove filters for deleted AIs
    for (const name of Object.keys(competeFilters)) {
        if (!saves[name]) delete competeFilters[name];
    }

    // Build filter UI
    filtersDiv.innerHTML = names.map(name => {
        const f = competeFilters[name];
        const customVars = detectCustomVars(saves[name]);
        const escaped = name.replace(/'/g, "\\'");
        let html = `<div class="cf-ai">`;
        html += `<div class="cf-header" onclick="competeToggleAI('${escaped}')">`;
        html += `<div class="cf-toggle ${f.enabled ? 'on' : ''}">${f.enabled ? '✓' : ''}</div>`;
        html += `<span class="cf-name">${name}</span>`;
        if (customVars.length > 0) {
            const assignments = getVarAssignments();
            const numVariants = getAIVariants(name, saves[name]).length;
            html += `<span class="text-dim" style="font-size:0.72rem;margin-left:auto;">${numVariants} variant${numVariants !== 1 ? 's' : ''}</span>`;
        }
        html += `</div>`;
        if (customVars.length > 0 && f.enabled) {
            html += `<div class="cf-vars">`;
            for (const v of customVars) {
                const filterVal = f.varFilters[v] || "";
                html += `<div class="cf-var-row">`;
                html += `<span class="cf-var-label">${v}:</span>`;
                html += `<input class="cf-var-input" value="${filterVal}" placeholder="all" onchange="competeSetVarFilter('${escaped}','${v}',this.value)">`;
                html += `</div>`;
            }
            html += `</div>`;
        }
        html += `</div>`;
        return html;
    }).join("");

    // Build the variant checkboxes based on filters
    rebuildCompeteCheckboxes();
}

function competeToggleAI(name) {
    if (!competeFilters[name]) competeFilters[name] = { enabled: false, varFilters: {} };
    competeFilters[name].enabled = !competeFilters[name].enabled;
    buildRefreshCompeteSelect();
}

function competeSetVarFilter(aiName, varName, value) {
    if (!competeFilters[aiName]) competeFilters[aiName] = { enabled: true, varFilters: {} };
    competeFilters[aiName].varFilters[varName] = value.trim();
    rebuildCompeteCheckboxes();
}

// Check if a value matches a filter string
// Supports: "200" (exact), "100, 200, 300" (list), "100-600" (range), or combos "100-300, 500, 800-1000"
function matchesFilter(val, filterStr) {
    const parts = filterStr.split(",").map(s => s.trim()).filter(s => s);
    for (const part of parts) {
        // Check for range: "100-600" (no step — just min-max)
        const rangeMatch = part.match(/^(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)$/);
        if (rangeMatch) {
            const lo = parseFloat(rangeMatch[1]);
            const hi = parseFloat(rangeMatch[2]);
            if (val >= lo && val <= hi) return true;
        } else {
            // Exact value
            const num = parseFloat(part);
            if (!isNaN(num) && Math.abs(val - num) < 0.001) return true;
        }
    }
    return false;
}

function rebuildCompeteCheckboxes() {
    const saves = getAISaves();
    const selectDiv = document.getElementById("compete-select");
    const shown = [];

    for (const [name, code] of Object.entries(saves)) {
        const f = competeFilters[name];
        if (!f || !f.enabled) continue;

        const variants = getAIVariants(name, code);

        for (const v of variants) {
            // Check if this variant passes all filters
            let passes = true;
            for (const [varName, filterStr] of Object.entries(f.varFilters)) {
                if (!filterStr) continue; // empty = show all
                const varVal = v.varValues[varName];
                if (varVal === undefined) continue;
                if (!matchesFilter(varVal, filterStr)) {
                    passes = false;
                    break;
                }
            }
            if (passes) shown.push(v.name);
        }
    }

    selectDiv.innerHTML = shown.map(vname => {
        const escaped = vname.replace(/'/g, "\\'").replace(/"/g, "&quot;");
        return `<label class="compete-cb checked"><input type="checkbox" value="${escaped}" checked onchange="this.parentElement.classList.toggle('checked',this.checked)">${vname}</label>`;
    }).join("");

    updateCompeteCount();
    updateNearbyVisibility();
}

function updateCompeteCount() {
    const count = document.querySelectorAll('#compete-select input[type=checkbox]:checked').length;
    const el = document.getElementById("compete-count");
    if (el) el.textContent = `${count} selected`;
}

function updateNearbyVisibility() {
    // Show nearby play option only if all enabled AIs are from the same base with one variable
    const saves = getAISaves();
    const selected = getSelectedCompetitors();
    const allVariantsMap = {};
    for (const baseName of Object.keys(saves)) {
        const variants = getAIVariants(baseName, saves[baseName]);
        for (const v of variants) allVariantsMap[v.name] = { baseName, varValues: v.varValues };
    }
    const bases = new Set();
    const vars = new Set();
    for (const name of selected) {
        const info = allVariantsMap[name];
        if (info) { bases.add(info.baseName); for (const v of Object.keys(info.varValues)) vars.add(v); }
    }
    const show = bases.size === 1 && vars.size === 1 && selected.length >= 3;
    document.getElementById("nearby-row").style.display = show ? "flex" : "none";
    if (!show) {
        document.getElementById("compete-nearby").checked = false;
        document.getElementById("nearby-pct-row").style.display = "none";
    }
}

function competeSelectAll() {
    document.querySelectorAll('#compete-select input[type=checkbox]').forEach(cb => { cb.checked = true; cb.parentElement.classList.add('checked'); });
    updateCompeteCount();
}

function competeSelectNone() {
    document.querySelectorAll('#compete-select input[type=checkbox]').forEach(cb => { cb.checked = false; cb.parentElement.classList.remove('checked'); });
    updateCompeteCount();
}

function getSelectedCompetitors() {
    const checkboxes = document.querySelectorAll('#compete-select input[type=checkbox]:checked');
    return Array.from(checkboxes).map(cb => cb.value);
}

let competeChart = null;
let competeChartData = null;
let competeTop5Only = true;

function competeToggleTop5() {
    competeTop5Only = !competeTop5Only;
    document.getElementById("compete-top5-btn").classList.toggle("active", competeTop5Only);
    if (competeChartData) renderCompeteChart();
}

function buildCompete() {
    const selected = getSelectedCompetitors();
    if (selected.length < 2) { alert("Select at least 2 AIs to compete."); return; }

    const saves = getAISaves();
    const gamesPerAI = parseInt(document.getElementById("compete-num-games").value);
    const minP = Math.max(2, parseInt(document.getElementById("compete-min-players").value));
    const maxP = Math.min(8, Math.max(minP, parseInt(document.getElementById("compete-max-players").value)));

    const statusDiv = document.getElementById("compete-status");
    const statusText = document.getElementById("compete-status-text");
    const resultsDiv = document.getElementById("compete-results");
    const chartWrap = document.getElementById("compete-chart-wrap");
    statusDiv.classList.remove("hidden");
    resultsDiv.classList.add("hidden");
    chartWrap.classList.add("hidden");

    setTimeout(() => {
        // Compile all selected AIs
        const allNames = Object.keys(saves);
        const allVariantsMap = {};
        for (const baseName of allNames) {
            const variants = getAIVariants(baseName, saves[baseName]);
            for (const v of variants) allVariantsMap[v.name] = { baseName, varValues: v.varValues };
        }

        const ais = [];
        for (const name of selected) {
            const info = allVariantsMap[name];
            if (!info) continue;
            try {
                const fn = compileBuildCodeWithVars(saves[info.baseName], info.varValues);
                ais.push({ name, fn, baseName: info.baseName, varValues: info.varValues });
            } catch (e) {
                statusDiv.classList.add("hidden");
                alert(`Error in "${name}": ${e.message}`);
                return;
            }
        }

        const n = ais.length;
        const actualMaxP = Math.min(maxP, n);
        const actualMinP = Math.min(minP, actualMaxP);
        const playerCounts = [];
        for (let k = actualMinP; k <= actualMaxP; k++) playerCounts.push(k);
        if (playerCounts.length === 0) { statusDiv.classList.add("hidden"); alert("Not enough AIs for the player range."); return; }

        // Nearby play: check if all AIs are variants of same base with one variable
        const nearbyEnabled = document.getElementById("compete-nearby").checked;
        const nearbyPct = parseInt(document.getElementById("compete-nearby-pct").value) / 100;
        let nearbyVar = null;
        let nearbyRange = 0;
        let nearbyVals = null;

        const baseSet = new Set(ais.map(a => a.baseName));
        const varSet = new Set();
        for (const a of ais) for (const v of Object.keys(a.varValues)) varSet.add(v);

        if (baseSet.size === 1 && varSet.size === 1 && nearbyEnabled) {
            nearbyVar = [...varSet][0];
            nearbyVals = ais.map(a => a.varValues[nearbyVar] || 0);
            const minVal = Math.min(...nearbyVals);
            const maxVal = Math.max(...nearbyVals);
            nearbyRange = (maxVal - minVal) * nearbyPct;
        }

        // Build allowed opponents for each AI (for nearby play)
        function canPlay(aiIdxA, aiIdxB) {
            if (!nearbyVar) return true;
            const diff = Math.abs(nearbyVals[aiIdxA] - nearbyVals[aiIdxB]);
            return diff <= nearbyRange;
        }
        const gamesPerCount = Math.max(1, Math.floor(gamesPerAI / playerCounts.length));

        // Stats
        const stats = {};
        for (const ai of ais) {
            stats[ai.name] = { totalPoints: 0, games: 0, totalScore: 0 };
            for (const k of playerCounts) {
                stats[ai.name]["pts_" + k] = 0;
                stats[ai.name]["games_" + k] = 0;
            }
        }

        // Generate all combinations of K from N
        function combinations(arr, k) {
            if (k === 1) return arr.map(x => [x]);
            const result = [];
            for (let i = 0; i <= arr.length - k; i++) {
                const rest = combinations(arr.slice(i + 1), k - 1);
                for (const combo of rest) result.push([arr[i], ...combo]);
            }
            return result;
        }

        // Shuffle array in place
        function shuffle(arr) {
            for (let i = arr.length - 1; i > 0; i--) {
                const j = Math.floor(Math.random() * (i + 1));
                [arr[i], arr[j]] = [arr[j], arr[i]];
            }
            return arr;
        }

        // Helper: play one game with given AI indices as seats
        function playGame(seats, k) {
            const shuffledSeats = shuffle([...seats]);
            const game = new BuildBankGame(shuffledSeats.length);
            while (!game.over) {
                if (!game.wfd) break;
                const actions = {};
                for (const pid of game.pir) {
                    const v = game.getVars(pid);
                    try {
                        const result = ais[shuffledSeats[pid]].fn(v.pot, v.round, v.roll, v.players_in, v.my_score, v.num_players,
                            v.p1_score, v.p2_score, v.p3_score, v.p4_score, v.p5_score,
                            v.p1_banked, v.p2_banked, v.p3_banked, v.p4_banked, v.p5_banked);
                        actions[pid] = result === "bank" ? "bank" : "continue";
                    } catch { actions[pid] = "continue"; }
                }
                game.step(actions);
            }

            const winners = game.winners();
            const numWinners = winners.length;
            const ptsPerWinner = shuffledSeats.length / numWinners;

            for (let pid = 0; pid < shuffledSeats.length; pid++) {
                const aiIdx = shuffledSeats[pid];
                const name = ais[aiIdx].name;
                const won = winners.includes(pid);
                const pts = won ? ptsPerWinner : 0;
                stats[name].totalPoints += pts;
                stats[name].games++;
                stats[name].totalScore += game.scores[pid];
                stats[name]["pts_" + k] += pts;
                stats[name]["games_" + k]++;
            }
        }

        // For each player count: cycle through all possible groups
        const allIndices = [...Array(n).keys()];
        for (const k of playerCounts) {
            statusText.textContent = `Generating ${k}-player groups...`;

            let groups = combinations(allIndices, k);

            // Filter groups by nearby play if enabled
            if (nearbyVar) {
                groups = groups.filter(group => {
                    for (let a = 0; a < group.length; a++) {
                        for (let b = a + 1; b < group.length; b++) {
                            if (!canPlay(group[a], group[b])) return false;
                        }
                    }
                    return true;
                });
                if (groups.length === 0) continue; // skip this player count if no valid groups
            }

            // If too many groups, sample a balanced subset
            if (groups.length > 5000) {
                shuffle(groups);
                groups = groups.slice(0, 5000);
            }
            // Each group played once = each AI appears in C(n-1, k-1) games
            // gamesPerAI at this K = gamesPerCount
            // games per cycle per AI = C(n-1, k-1)
            // So we need ceil(gamesPerCount / gamesPerCyclePerAI) full cycles
            const gamesPerCyclePerAI = groups.length * k / n; // each AI appears this many times per cycle
            const numCycles = Math.max(1, Math.ceil(gamesPerCount / gamesPerCyclePerAI));

            statusText.textContent = `Running ${k}-player games (${groups.length} groups × ${numCycles} cycles)...`;

            let gamesPlayed = new Array(n).fill(0);
            const targetPerAI = gamesPerCount;

            for (let cycle = 0; cycle < numCycles; cycle++) {
                shuffle(groups);
                for (const group of groups) {
                    // Check if all AIs in this group still need games
                    const allNeed = group.every(i => gamesPlayed[i] < targetPerAI);
                    if (!allNeed) {
                        // Check if at least some need games — play if majority still need
                        const needCount = group.filter(i => gamesPlayed[i] < targetPerAI).length;
                        if (needCount < Math.ceil(k / 2)) continue; // skip if most are done
                    }

                    playGame(group, k);
                    for (const i of group) gamesPlayed[i]++;

                    // Check if everyone has hit target
                    if (gamesPlayed.every(g => g >= targetPerAI)) break;
                }
                if (gamesPlayed.every(g => g >= targetPerAI)) break;
            }
        }

        // Build leaderboard sorted by points per game
        const leaderboard = Object.entries(stats).map(([name, s]) => ({
            name,
            totalPoints: s.totalPoints,
            games: s.games,
            ppg: s.games > 0 ? (s.totalPoints / s.games).toFixed(2) : "0.00",
            avgScore: s.games > 0 ? Math.round(s.totalScore / s.games) : 0,
            // Per player-count ppg for chart
            perK: {},
        }));

        for (const entry of leaderboard) {
            const s = stats[entry.name];
            for (const k of playerCounts) {
                const g = s["games_" + k];
                entry.perK[k] = g > 0 ? s["pts_" + k] / g : 0;
            }
        }

        // Store per-K ppg on each entry
        for (const entry of leaderboard) {
            entry.gamesPerK = {};
            entry.ppgPerK = {};
            for (const k of playerCounts) {
                entry.gamesPerK[k] = stats[entry.name]["games_" + k] || 0;
                entry.ppgPerK[k] = entry.perK[k] ? entry.perK[k].toFixed(2) : "0.00";
            }
        }

        leaderboard.sort((a, b) => parseFloat(b.ppg) - parseFloat(a.ppg));

        // Store for rendering
        competeChartData = { leaderboard, playerCounts, gamesPerAI, actualMinP, actualMaxP, stats, ais, saves };
        renderCompeteLeaderboard();

        statusDiv.classList.add("hidden");
        resultsDiv.classList.remove("hidden");

        chartWrap.classList.remove("hidden");
        renderCompeteChart();
        renderCompeteVarChart();
    }, 50);
}

// Column mode: "games" or "ppgk"
let competeColMode = "games";
let competeSortCol = "ppg";
let competeSortAsc = false;

function competeSetCol(mode) {
    competeColMode = mode;
    document.getElementById("col-games").classList.toggle("active", mode === "games");
    document.getElementById("col-ppgk").classList.toggle("active", mode === "ppgk");
    renderCompeteLeaderboard();
}

function competeSortBy(col) {
    if (competeSortCol === col) {
        competeSortAsc = !competeSortAsc;
    } else {
        competeSortCol = col;
        competeSortAsc = false;
    }
    renderCompeteLeaderboard();
}

function renderCompeteLeaderboard() {
    if (!competeChartData) return;
    const { leaderboard, playerCounts, gamesPerAI, actualMinP, actualMaxP } = competeChartData;

    // Sort leaderboard
    const sorted = [...leaderboard];
    const dir = competeSortAsc ? 1 : -1;
    sorted.sort((a, b) => {
        let va, vb;
        if (competeSortCol === "name") { return dir * a.name.localeCompare(b.name); }
        if (competeSortCol === "ppg") { va = parseFloat(a.ppg); vb = parseFloat(b.ppg); }
        else if (competeSortCol === "score") { va = a.avgScore; vb = b.avgScore; }
        else if (competeSortCol === "total") { va = a.games; vb = b.games; }
        else if (competeSortCol.startsWith("g_")) { const k = parseInt(competeSortCol.slice(2)); va = a.gamesPerK[k]||0; vb = b.gamesPerK[k]||0; }
        else if (competeSortCol.startsWith("p_")) { const k = parseInt(competeSortCol.slice(2)); va = a.perK[k]||0; vb = b.perK[k]||0; }
        else { va = parseFloat(a.ppg); vb = parseFloat(b.ppg); }
        return dir * (vb - va);
    });

    const arrow = col => competeSortCol === col ? (competeSortAsc ? " ▲" : " ▼") : "";
    const th = (col, label) => `<th class="sortable" onclick="competeSortBy('${col}')">${label}${arrow(col)}</th>`;

    // Extra columns
    let extraHeaders = "", extraAvgCells = "";
    if (competeColMode === "games") {
        extraHeaders = th("total", "Total") + playerCounts.map(k => th("g_"+k, k+"p")).join("");
        const avgTotal = Math.round(sorted.reduce((s,e) => s+e.games, 0) / sorted.length);
        extraAvgCells = `<td>${avgTotal}</td>`;
        for (const k of playerCounts) {
            const avg = Math.round(sorted.reduce((s,e) => s+(e.gamesPerK[k]||0), 0) / sorted.length);
            extraAvgCells += `<td>${avg}</td>`;
        }
    } else {
        extraHeaders = playerCounts.map(k => th("p_"+k, k+"p")).join("");
        for (const k of playerCounts) {
            const avg = (sorted.reduce((s,e) => s+(e.perK[k]||0), 0) / sorted.length).toFixed(2);
            extraAvgCells += `<td>${avg}</td>`;
        }
    }

    const avgPpg = (sorted.reduce((s,e) => s+parseFloat(e.ppg), 0) / sorted.length).toFixed(2);
    const avgScore = Math.round(sorted.reduce((s,e) => s+e.avgScore, 0) / sorted.length);

    const tableWrap = document.getElementById("compete-table-wrap");
    tableWrap.innerHTML = `
        <h3>Leaderboard (${gamesPerAI} games/AI, ${actualMinP}-${actualMaxP}p)</h3>
        <table class="compete-leaderboard compact">
            <tr>${th("rank","#")}${th("name","AI")}${th("ppg","PPG")}${th("score","Score")}${extraHeaders}</tr>
            <tr class="compete-avg-row"><td></td><td>Avg</td><td>${avgPpg}</td><td>${avgScore}</td>${extraAvgCells}</tr>
            ${sorted.map((e, idx) => {
                let extra = "";
                if (competeColMode === "games") {
                    extra = `<td>${e.games}</td>` + playerCounts.map(k => `<td>${e.gamesPerK[k]||0}</td>`).join("");
                } else {
                    extra = playerCounts.map(k => `<td>${(e.perK[k]||0).toFixed(2)}</td>`).join("");
                }
                return `<tr><td>${idx+1}</td><td>${e.name}</td><td>${e.ppg}</td><td>${e.avgScore}</td>${extra}</tr>`;
            }).join("")}
        </table>
    `;
}

const COMPETE_COLORS = ["#60a5fa","#f87171","#34d399","#fbbf24","#c084fc","#fb923c","#e879f9","#2dd4bf","#f472b6","#a3e635"];
let competeVarChart = null;
let competeVarK = "all"; // which player count to show on var chart

function renderCompeteVarChart() {
    const wrap = document.getElementById("compete-var-chart-wrap");
    if (!competeChartData) { wrap.classList.add("hidden"); return; }

    const { leaderboard, playerCounts, saves, ais } = competeChartData;

    // Detect if all selected AIs are variants of the same base AI with one variable
    const allNames = Object.keys(saves);
    const allVariantsMap = {};
    for (const baseName of allNames) {
        const variants = getAIVariants(baseName, saves[baseName]);
        for (const v of variants) allVariantsMap[v.name] = { baseName, varValues: v.varValues };
    }

    // Check: do all leaderboard entries share the same base AI?
    const baseNames = new Set();
    const varNames = new Set();
    for (const entry of leaderboard) {
        const info = allVariantsMap[entry.name];
        if (info) {
            baseNames.add(info.baseName);
            for (const vn of Object.keys(info.varValues)) varNames.add(vn);
        }
    }

    // Only show if all from same base AND exactly one variable
    if (baseNames.size !== 1 || varNames.size !== 1) {
        wrap.classList.add("hidden");
        return;
    }

    const varName = [...varNames][0];
    wrap.classList.remove("hidden");
    document.getElementById("var-chart-title").textContent = `Pts/Game by ${varName}`;

    // Build K selector buttons
    const kSelect = document.getElementById("var-k-select");
    kSelect.innerHTML = `<button class="btn btn-ctl ${competeVarK === 'all' ? 'active' : ''}" onclick="setCompeteVarK('all')">All Games</button>` +
        playerCounts.map(k => `<button class="btn btn-ctl ${competeVarK == k ? 'active' : ''}" onclick="setCompeteVarK(${k})">${k}p</button>`).join("");

    // Build data: x = variable value, y = ppg
    const points = [];
    for (const entry of leaderboard) {
        const info = allVariantsMap[entry.name];
        if (!info) continue;
        const xVal = info.varValues[varName];
        let yVal;
        if (competeVarK === "all") {
            yVal = parseFloat(entry.ppg);
        } else {
            yVal = entry.perK[competeVarK] || 0;
        }
        points.push({ x: xVal, y: yVal });
    }
    points.sort((a, b) => a.x - b.x);

    if (competeVarChart) competeVarChart.destroy();
    const canvas = document.getElementById("chart-compete-var");
    canvas.style.maxHeight = "none";
    competeVarChart = new Chart(canvas, {
        type: "line",
        data: {
            labels: points.map(p => p.x),
            datasets: [{
                label: `Pts/Game (${competeVarK === "all" ? "all" : competeVarK + "p"})`,
                data: points.map(p => p.y),
                borderColor: "#60a5fa",
                borderWidth: 2.5,
                pointRadius: 4,
                pointBackgroundColor: "#60a5fa",
                tension: 0.2,
                fill: true,
                backgroundColor: "rgba(96,165,250,0.08)",
            }]
        },
        options: {
            responsive: true, maintainAspectRatio: false, animation: false,
            interaction: { mode: "index", intersect: false },
            plugins: { legend: { display: false }, tooltip: { mode: "index", intersect: false } },
            scales: {
                x: { title: { display: true, text: varName, color: "#8b8fa3" }, ticks: { color: "#8b8fa3" }, grid: { color: "rgba(42,46,61,0.3)" } },
                y: { title: { display: true, text: "Points Per Game", color: "#8b8fa3" }, ticks: { color: "#8b8fa3" }, grid: { color: "rgba(42,46,61,0.3)" } },
            },
        },
    });
}

function setCompeteVarK(k) {
    competeVarK = k;
    renderCompeteVarChart();
}

function renderCompeteChart() {
    if (!competeChartData) return;
    if (competeChart) competeChart.destroy();

    const { leaderboard, playerCounts } = competeChartData;
    const show = competeTop5Only ? leaderboard.slice(0, 5) : leaderboard;

    const datasets = show.map((entry, idx) => ({
        label: entry.name,
        data: playerCounts.map(k => entry.perK[k] || 0),
        borderColor: COMPETE_COLORS[idx % COMPETE_COLORS.length],
        borderWidth: 2,
        pointRadius: 4,
        pointBackgroundColor: COMPETE_COLORS[idx % COMPETE_COLORS.length],
        tension: 0.2,
    }));

    const canvas = document.getElementById("chart-compete");
    canvas.style.maxHeight = "none";
    competeChart = new Chart(canvas, {
        type: "line",
        data: { labels: playerCounts, datasets },
        options: {
            responsive: true, maintainAspectRatio: false, animation: false,
            interaction: { mode: "index", intersect: false },
            plugins: {
                legend: { labels: { color: "#8b8fa3", font: { size: 11 } } },
                tooltip: {
                    mode: "index",
                    intersect: false,
                    itemSort: (a, b) => b.raw - a.raw,
                },
            },
            scales: {
                x: { title: { display: true, text: "Players Per Game", color: "#8b8fa3" }, ticks: { color: "#8b8fa3" }, grid: { color: "rgba(42,46,61,0.3)" } },
                y: {
                    title: { display: true, text: "Points Per Game", color: "#8b8fa3" },
                    ticks: { color: "#8b8fa3" },
                    grid: { color: "rgba(42,46,61,0.3)" },
                },
            },
        },
    });
}

// Seed default AIs on first use
function buildSeedDefaults() {
    const saves = getAISaves();
    if (!saves["Endgame"]) {
        saves["Endgame"] =
`if my_score >= p1_score and p2_banked == 1:
    bank
elif my_score < p1_score and p1_banked == 1 and my_score + pot > p1_score:
    bank
elif my_score < p1_score and p1_banked == 0:
    continue
else:
    continue`;
        setAISaves(saves);
    }
}

// Tab key inserts spaces
document.addEventListener("DOMContentLoaded", () => {
    buildSeedDefaults();
    buildRefreshList();
    buildRefreshCompeteSelect();
    buildUpdateVarsPanel();
    const editor = document.getElementById("build-code");
    editor.addEventListener("keydown", e => {
        if (e.key === "Tab") {
            e.preventDefault();
            const start = editor.selectionStart;
            const end = editor.selectionEnd;
            editor.value = editor.value.substring(0, start) + "    " + editor.value.substring(end);
            editor.selectionStart = editor.selectionEnd = start + 4;
        }
    });
    // Detect custom vars when code changes
    let varsTimer = null;
    editor.addEventListener("input", () => {
        clearTimeout(varsTimer);
        varsTimer = setTimeout(buildUpdateVarsPanel, 500);
    });
});
