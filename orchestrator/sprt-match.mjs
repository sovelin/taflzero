/**
 * SPRT match runner for AlphaZero Tafl pipeline.
 *
 * Runs paired games (same opening, swap colors) between two engine instances
 * to determine if a candidate network is stronger than the current best.
 *
 * Usage:
 *   node orchestrator/sprt-match.mjs \
 *     --main-net weights/current.onnx \
 *     --candidate-net weights/candidate.onnx \
 *     --engine-bin target/release/zevratafl-rust.exe \
 *     [--nodes 200] [--opening-moves 16] [--workers 24] [--max-pairs 2500]
 *     [--sprt-elo0 0] [--sprt-elo1 5] [--sprt-alpha 0.05] [--sprt-beta 0.05]
 */

import { spawn } from "node:child_process";
import readline from "node:readline";
import path from "node:path";
import { readFileSync, writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const wasmPath = path.join(__dirname, "node_modules/zevratafl-rust/pkg/zevratafl_rust_bg.wasm");
const wasmBytes = readFileSync(wasmPath);

const { default: init, EngineClient, Side, get_total_squares } = await import("zevratafl-rust");
await init(wasmBytes);

// ─── CLI args ────────────────────────────────────────────────────────────────

function parseArgs(argv) {
    const args = {
        mainNet: null,
        candidateNet: null,
        engineBin: null,
        nodes: 200,
        openingMoves: 16,
        workers: 24,
        maxPairs: 2500,
        sprtElo0: 0,
        sprtElo1: 5,
        sprtAlpha: 0.05,
        sprtBeta: 0.05,
        resultFile: null,
        noGate: false,
    };

    for (let i = 0; i < argv.length; i++) {
        const a = argv[i];
        const next = () => argv[++i];

        if (a === "--main-net") args.mainNet = next();
        else if (a === "--candidate-net") args.candidateNet = next();
        else if (a === "--engine-bin") args.engineBin = next();
        else if (a === "--nodes") args.nodes = parseInt(next(), 10);
        else if (a === "--opening-moves") args.openingMoves = parseInt(next(), 10);
        else if (a === "--workers") args.workers = parseInt(next(), 10);
        else if (a === "--max-pairs") args.maxPairs = parseInt(next(), 10);
        else if (a === "--sprt-elo0") args.sprtElo0 = parseFloat(next());
        else if (a === "--sprt-elo1") args.sprtElo1 = parseFloat(next());
        else if (a === "--sprt-alpha") args.sprtAlpha = parseFloat(next());
        else if (a === "--sprt-beta") args.sprtBeta = parseFloat(next());
        else if (a === "--result-file") args.resultFile = next();
        else if (a === "--no-gate") args.noGate = true;
        else if (a === "--help" || a === "-h") { printHelp(); process.exit(0); }
        else throw new Error(`Unknown arg: ${a}`);
    }

    if (!args.mainNet) throw new Error("Missing --main-net");
    if (!args.candidateNet) throw new Error("Missing --candidate-net");
    if (!args.engineBin) throw new Error("Missing --engine-bin");

    return args;
}

function printHelp() {
    console.log([
        "Usage: node orchestrator/sprt-match.mjs --main-net <path> --candidate-net <path> --engine-bin <path> [options]",
        "",
        "  --nodes <N>            MCTS nodes per move (default: 200)",
        "  --opening-moves <N>    Random moves for opening generation (default: 16)",
        "  --workers <N>          Parallel game workers (default: 24)",
        "  --max-pairs <N>        Max game pairs before failing (default: 2500)",
        "  --sprt-elo0 <N>        SPRT H0 elo (default: 0)",
        "  --sprt-elo1 <N>        SPRT H1 elo (default: 5)",
        "  --sprt-alpha <F>       Type I error rate (default: 0.05)",
        "  --sprt-beta <F>        Type II error rate (default: 0.05)",
    ].join("\n"));
}

// ─── SPRT implementation ─────────────────────────────────────────────────────

function eloToProb(elo) {
    return 1.0 / (1.0 + Math.pow(10, -elo / 400.0));
}

function createSprt({ elo0, elo1, alpha, beta }) {
    const p0 = eloToProb(elo0);
    const p1 = eloToProb(elo1);
    const lowerBound = Math.log(beta / (1 - alpha));
    const upperBound = Math.log((1 - beta) / alpha);

    let llr = 0;
    let wins = 0;
    let losses = 0;
    let draws = 0;

    function record(score) {
        // score: 1.0 = win, 0.5 = draw, 0.0 = loss (for candidate)
        if (score === 1.0) wins++;
        else if (score === 0.0) losses++;
        else draws++;

        const total = wins + losses + draws;
        const w = wins / total;
        const l = losses / total;
        const d = draws / total;

        // Trinomial LLR
        llr = 0;
        if (w > 0) llr += w * Math.log(calcExpectedW(p1) / calcExpectedW(p0));
        if (d > 0) llr += d * Math.log(calcExpectedD(p1) / calcExpectedD(p0));
        if (l > 0) llr += l * Math.log(calcExpectedL(p1) / calcExpectedL(p0));
        llr *= total;

        let decision = "continue";
        if (llr >= upperBound) decision = "acceptH1";
        else if (llr <= lowerBound) decision = "acceptH0";

        return { llr, lowerBound, upperBound, decision };
    }

    // Expected probabilities under a given winning probability p
    function calcExpectedW(p) { return p; }
    function calcExpectedL(p) { return 1 - p; }
    function calcExpectedD(p) {
        // For pentanomial/trinomial with draws: use BayesElo-style approximation
        // Simplify: assume draw rate is constant, scale W/L
        // Actually for proper SPRT with W/D/L we use the simpler binomial on score
        return 0; // unused in simplified version
    }

    // Simplified: use binomial SPRT on score (more standard for chess)
    function recordScore(score) {
        if (score === 1.0) wins++;
        else if (score === 0.0) losses++;
        else draws++;

        const total = wins + losses + draws;
        const totalScore = wins + draws * 0.5;
        const observedScore = totalScore / total;

        // Binomial LLR = log(L1/L0) = n * [s*log(p1/p0) + (1-s)*log((1-p1)/(1-p0))]
        if (observedScore <= 0 || observedScore >= 1) {
            llr = observedScore >= 1 ? upperBound + 1 : lowerBound - 1;
        } else {
            llr = total * (
                observedScore * Math.log(p1 / p0) +
                (1 - observedScore) * Math.log((1 - p1) / (1 - p0))
            );
        }

        let decision = "continue";
        if (llr >= upperBound) decision = "acceptH1";
        else if (llr <= lowerBound) decision = "acceptH0";

        return { llr, lowerBound, upperBound, decision, wins, losses, draws, total, score: totalScore, pct: (totalScore / total * 100) };
    }

    // Reset for use with binomial approach
    wins = 0; losses = 0; draws = 0; llr = 0;

    return {
        record: recordScore,
        status() {
            const total = wins + losses + draws;
            const totalScore = wins + draws * 0.5;
            return { llr, lowerBound, upperBound, wins, losses, draws, total, score: totalScore, pct: total > 0 ? totalScore / total * 100 : 0 };
        },
    };
}

// ─── UCI engine process ──────────────────────────────────────────────────────

class UciEngine {
    constructor(command, args, name = "engine") {
        this.name = name;
        this.alive = true;
        this.pending = [];

        this.child = spawn(command, args, {
            stdio: ["pipe", "pipe", "pipe"],
        });
        this.child.stdin.setDefaultEncoding("utf-8");
        this.child.stderr.on("data", () => {});
        this.child.on("exit", (code, signal) => {
            this.alive = false;
            for (const p of this.pending) {
                clearTimeout(p.timer);
                p.reject(new Error(`${this.name} exited (${signal || code})`));
            }
            this.pending = [];
        });

        this.rl = readline.createInterface({ input: this.child.stdout, crlfDelay: Infinity });
        this.rl.on("line", (raw) => {
            const line = raw.trim();
            if (!line) return;
            if (this.pending.length > 0 && this.pending[0].matcher(line)) {
                const p = this.pending.shift();
                clearTimeout(p.timer);
                p.resolve(line);
            }
        });
    }

    send(cmd) {
        if (!this.alive) throw new Error(`${this.name} is dead`);
        this.child.stdin.write(cmd + "\n");
    }

    waitFor(matcher, timeoutMs = 30000) {
        return new Promise((resolve, reject) => {
            const timer = setTimeout(() => {
                this.pending = this.pending.filter((p) => p.resolve !== resolve);
                reject(new Error(`${this.name} timed out`));
            }, timeoutMs);
            this.pending.push({ matcher, resolve, reject, timer });
        });
    }

    async init() {
        this.send("uci");
        await this.waitFor((l) => l.startsWith("uciok") || l.includes("uciok"));
        this.send("isready");
        await this.waitFor((l) => l === "readyok");
    }

    async setPosition(fen, moves = []) {
        const moveStr = moves.length > 0 ? ` moves ${moves.join(" ")}` : "";
        this.send(`position fen ${fen}${moveStr}`);
        this.send("isready");
        await this.waitFor((l) => l === "readyok");
    }

    async goNodes(nodes) {
        this.send(`go nodes ${nodes}`);
        const line = await this.waitFor((l) => l.startsWith("bestmove"), 120000);
        const match = line.match(/bestmove\s+(\S+)/);
        if (!match) return null;
        const move = match[1];
        // a1a1 = null move (from==to), means stalemate
        if (move === "a1a1" || move === "(none)") return null;
        return move;
    }

    dispose() {
        if (!this.alive) return;
        try { this.send("quit"); } catch {}
        this.child.stdin.end();
    }
}

// ─── Opening generation ──────────────────────────────────────────────────────

function createOpening(movesCount) {
    const engine = new EngineClient(4);
    const initialFen = "3aaaaa3/5a5/11/a4d4a/a3ddd3a/aa1ddkdd1aa/a3ddd3a/a4d4a/11/5a5/3aaaaa3 a";
    engine.set_fen(initialFen);

    for (let i = 0; i < movesCount; i++) {
        const totalSq = get_total_squares();
        const moves = [];
        for (let sq = 0; sq < totalSq; sq++) {
            const sqMoves = engine.get_available_moves_from_square(sq);
            for (const mv of sqMoves) {
                moves.push(mv.raw());
            }
        }
        if (moves.length === 0) break;

        const idx = Math.floor(Math.random() * moves.length);
        const fen = engine.get_fen();
        engine.set_position_and_moves(fen, new Uint32Array([moves[idx]]));
    }

    const resultFen = engine.get_fen();
    engine.free();
    return resultFen;
}

// ─── Game logic ──────────────────────────────────────────────────────────────

class GameController {
    constructor() {
        this.engine = new EngineClient(4);
    }

    checkTerminal(fen, moves) {
        this.engine.set_position_and_moves(fen, moves);
        return this.engine.check_terminal_state();
    }

    getSideToMove(fen, moves) {
        this.engine.set_position_and_moves(fen, moves);
        return this.engine.side_to_move();
    }

    moveNumToStr(mv) {
        return this.engine.move_num_to_str(mv);
    }

    moveStrToNum(str) {
        return this.engine.move_str_to_num(str);
    }

    getPieceCount(fen, moves) {
        this.engine.set_position_and_moves(fen, moves);
        const board = this.engine.get_board_state();
        return board.filter((sq) => sq !== 0).length;
    }

    free() {
        try { this.engine.free(); } catch {}
    }
}

/**
 * Play a single game between two engines.
 * Returns: 'main' | 'candidate' | 'draw'
 */
async function playGame(ctrl, mainEngine, candidateEngine, opening, nodes, attackerRole, defenderRole) {
    const gameMoves = [];         // numeric move values
    const gameMoveStrings = [];   // algebraic strings
    let pieceCount = ctrl.getPieceCount(opening, []);
    let noCaptureCount = 0;

    while (true) {
        const terminal = ctrl.checkTerminal(opening, gameMoves);
        if (terminal !== undefined) {
            // terminal is Side.ATTACKERS or Side.DEFENDERS
            return terminal === Side.ATTACKERS ? attackerRole : defenderRole;
        }

        const stm = ctrl.getSideToMove(opening, gameMoves);
        const engineToMove = stm === Side.ATTACKERS
            ? (attackerRole === "candidate" ? candidateEngine : mainEngine)
            : (defenderRole === "candidate" ? candidateEngine : mainEngine);

        const moveAlgTokens = gameMoveStrings;
        await engineToMove.setPosition(opening, moveAlgTokens);

        const bestMoveStr = await engineToMove.goNodes(nodes);
        if (!bestMoveStr || bestMoveStr === "(none)") {
            // No legal moves — opponent wins
            return stm === Side.ATTACKERS ? defenderRole : attackerRole;
        }

        const bestMoveNum = ctrl.moveStrToNum(bestMoveStr);
        gameMoves.push(bestMoveNum);
        gameMoveStrings.push(bestMoveStr);

        let newPieceCount;
        try {
            newPieceCount = ctrl.getPieceCount(opening, gameMoves);
        } catch (err) {
            console.error(`[DEBUG] getPieceCount crashed after move ${gameMoveStrings.length}`);
            console.error(`[DEBUG] FEN: ${opening}`);
            console.error(`[DEBUG] Moves: ${gameMoveStrings.join(" ")}`);
            console.error(`[DEBUG] Last move: ${bestMoveStr} (raw: ${bestMoveNum})`);
            console.error(`[DEBUG] Error: ${err.stack || err.message || err}`);
            throw err;
        }
        if (newPieceCount < pieceCount) {
            noCaptureCount = 0;
            pieceCount = newPieceCount;
        } else {
            noCaptureCount++;
            if (noCaptureCount >= 300) return "draw";
        }

        if (gameMoves.length >= 700) return "draw";
    }
}

/**
 * Play a pair of games from the same opening with swapped colors.
 * Returns pair score for candidate: 1.0, 0.75, 0.5, 0.25, 0.0
 */
async function playPair(ctrl, mainEngine, candidateEngine, opening, nodes) {
    // Game 1: candidate = attacker, main = defender
    const result1 = await playGame(ctrl, mainEngine, candidateEngine, opening, nodes, "candidate", "main");
    // Game 2: candidate = defender, main = attacker
    const result2 = await playGame(ctrl, mainEngine, candidateEngine, opening, nodes, "main", "candidate");

    const score1 = result1 === "candidate" ? 1 : result1 === "draw" ? 0.5 : 0;
    const score2 = result2 === "candidate" ? 1 : result2 === "draw" ? 0.5 : 0;

    // Pair score: average of two games
    return (score1 + score2) / 2;
}

// ─── Worker ──────────────────────────────────────────────────────────────────

async function createWorkerPair(engineBin, mainNet, candidateNet) {
    const mainArgs = ["--net", path.normalize(mainNet)];
    const candidateArgs = ["--net", path.normalize(candidateNet)];

    const mainEngine = new UciEngine(engineBin, mainArgs, "main");
    const candidateEngine = new UciEngine(engineBin, candidateArgs, "candidate");
    const ctrl = new GameController();

    await mainEngine.init();
    await candidateEngine.init();

    return { mainEngine, candidateEngine, ctrl };
}

// ─── Main ────────────────────────────────────────────────────────────────────

async function main() {
    let args;
    try {
        args = parseArgs(process.argv.slice(2));
    } catch (err) {
        console.error(err.message);
        printHelp();
        process.exit(2);
    }

    const sprt = createSprt({
        elo0: args.sprtElo0,
        elo1: args.sprtElo1,
        alpha: args.sprtAlpha,
        beta: args.sprtBeta,
    });

    const initialStatus = sprt.status();
    console.log(`[SPRT] elo0=${args.sprtElo0}, elo1=${args.sprtElo1}, alpha=${args.sprtAlpha}, beta=${args.sprtBeta}`);
    console.log(`[SPRT] bounds: B=${initialStatus.lowerBound.toFixed(3)}, A=${initialStatus.upperBound.toFixed(3)}`);
    console.log(`[Match] main=${args.mainNet}, candidate=${args.candidateNet}`);
    console.log(`[Match] nodes=${args.nodes}, openingMoves=${args.openingMoves}, workers=${args.workers}, maxPairs=${args.maxPairs}`);

    // Spawn worker pairs
    const workerPairs = [];
    for (let i = 0; i < args.workers; i++) {
        workerPairs.push(await createWorkerPair(args.engineBin, args.mainNet, args.candidateNet));
    }

    let pairsCompleted = 0;
    let stopped = false;
    let crossingDecision = null;  // decision when LLR first crossed a boundary

    // Generate openings ahead of time in batches
    const openingQueue = [];
    function refillOpenings(count) {
        for (let i = 0; i < count; i++) {
            openingQueue.push(createOpening(args.openingMoves));
        }
    }

    refillOpenings(args.workers * 2);

    // Run pairs in parallel across workers
    async function workerLoop(worker) {
        while (pairsCompleted < args.maxPairs && !stopped) {
            if (openingQueue.length === 0) {
                refillOpenings(args.workers);
            }
            const opening = openingQueue.shift();

            let pairScore;
            try {
                pairScore = await playPair(worker.ctrl, worker.mainEngine, worker.candidateEngine, opening, args.nodes);
            } catch (err) {
                console.error(`[Worker] pair failed: ${err.message}`);
                break;
            }

            pairsCompleted++;
            const status = sprt.record(pairScore);

            console.log(
                `Pair ${pairsCompleted}: dev=${pairScore.toFixed(2)} | ` +
                `Score: ${status.score.toFixed(1)}/${status.total} (${status.pct.toFixed(1)}%) | ` +
                `SPRT LLR=${status.llr.toFixed(3)} [B=${status.lowerBound.toFixed(3)}, A=${status.upperBound.toFixed(3)}] [${status.decision}]`
            );

            // In no-gate mode, play all pairs without early stopping
            if (!args.noGate) {
                // Only stop when we have enough pairs and cumulative LLR is decisive
                // Require at least 2*workers pairs to avoid race conditions with parallel workers
                if (status.total >= args.workers * 2 && status.decision !== "continue") {
                    if (!crossingDecision) crossingDecision = status.decision;
                    stopped = true;
                    break;
                }
            }
        }
    }

    await Promise.all(workerPairs.map((w) => workerLoop(w)));

    // Cleanup
    for (const w of workerPairs) {
        w.mainEngine.dispose();
        w.candidateEngine.dispose();
        w.ctrl.free();
    }

    // Use crossing decision if LLR crossed a boundary before in-flight pairs diluted it
    const status = sprt.status();
    const finalLlrDecision = status.llr >= status.upperBound ? "acceptH1"
        : status.llr <= status.lowerBound ? "acceptH0"
        : null;
    const actualDecision = finalLlrDecision ?? crossingDecision ?? "maxPairs";
    const passed = actualDecision === "acceptH1";

    // Compute Elo difference from score percentage
    const scorePct = status.total > 0 ? status.score / status.total : 0.5;
    const eloDiff = scorePct > 0 && scorePct < 1
        ? -400 * Math.log10(1 / scorePct - 1)
        : scorePct >= 1 ? 999 : -999;

    // Write result file if requested
    if (args.resultFile) {
        const result = {
            passed,
            decision: actualDecision,
            llr: status.llr,
            score: status.score,
            total: status.total,
            pct: status.pct,
            eloDiff,
            wins: status.wins,
            losses: status.losses,
            draws: status.draws,
        };
        writeFileSync(args.resultFile, JSON.stringify(result));
    }

    if (args.noGate) {
        console.log(`\n[Match] Elo=${eloDiff.toFixed(1)}, Score: ${status.score.toFixed(1)}/${status.total} (${status.pct.toFixed(1)}%)`);
        process.exit(0);
    } else if (passed) {
        console.log(`\n[SPRT] PASSED — candidate is stronger (LLR=${status.llr.toFixed(3)}, Elo=${eloDiff.toFixed(1)})`);
        console.log(`Score: ${status.score.toFixed(1)}/${status.total} (${status.pct.toFixed(1)}%)`);
        process.exit(0);
    } else {
        const reason = actualDecision === "acceptH0" ? "candidate is NOT stronger" : "max pairs reached";
        console.log(`\n[SPRT] FAILED — ${reason} (LLR=${status.llr.toFixed(3)}, Elo=${eloDiff.toFixed(1)})`);
        console.log(`Score: ${status.score.toFixed(1)}/${status.total} (${status.pct.toFixed(1)}%)`);
        process.exit(1);
    }
}

main().catch((err) => {
    console.error(err.stack || String(err));
    process.exit(1);
});
