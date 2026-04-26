import { spawn } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

function parseArgs(argv) {
    const args = {
        iterations: 1,
        startGen: 1,
        gamesPerGen: 10000,
        data: "zero-trainer/selfplay.bin",
        weightsDir: "zero-trainer/weights",
        startNet: "random_init.onnx",
        startCheckpoint: null,
        python: null,
        window: 50000,
        steps: 1000,
        batch: 256,
        lr: 1e-3,
        weightDecay: 1e-4,
        defenderWeight: 0.25,
        engineBin: null,
        debugEngine: false,
        workers: 1,
        // SPRT options
        noSprt: false,
        noGate: false,
        noGatePairs: 100,
        sprtNodes: 200,
        sprtOpeningMoves: 16,
        sprtWorkers: 12,
        sprtMaxPairs: 2500,
        sprtElo0: 0,
        sprtElo1: 5,
        sprtAlpha: 0.05,
        sprtBeta: 0.05,
        anchorNet: null,
        anchorPairs: 100,
        earlyStoppingPatience: 0,
        noRestoreBest: false,
    };

    for (let i = 0; i < argv.length; i += 1) {
        const a = argv[i];
        const next = () => argv[++i];

        if (a === "--iterations") args.iterations = intArg(next(), a, 1);
        else if (a === "--start-gen") args.startGen = intArg(next(), a, 0);
        else if (a === "--games-per-gen") args.gamesPerGen = intArg(next(), a, 1);
        else if (a === "--data") args.data = required(next(), a);
        else if (a === "--weights-dir") args.weightsDir = required(next(), a);
        else if (a === "--start-net") args.startNet = required(next(), a);
        else if (a === "--start-checkpoint") args.startCheckpoint = required(next(), a);
        else if (a === "--python") args.python = required(next(), a);
        else if (a === "--window") args.window = intArg(next(), a, 0);
        else if (a === "--steps") args.steps = intArg(next(), a, 0);
        else if (a === "--batch") args.batch = intArg(next(), a, 1);
        else if (a === "--lr") args.lr = floatArg(next(), a, 0);
        else if (a === "--weight-decay") args.weightDecay = floatArg(next(), a, 0);
        else if (a === "--defender-weight") args.defenderWeight = floatArg(next(), a, 0);
        else if (a === "--workers") args.workers = intArg(next(), a, 1);
        else if (a === "--engine-bin") args.engineBin = required(next(), a);
        else if (a === "--debug-engine") args.debugEngine = true;
        else if (a === "--no-sprt") args.noSprt = true;
        else if (a === "--no-gate") args.noGate = true;
        else if (a === "--no-gate-pairs") args.noGatePairs = intArg(next(), a, 1);
        else if (a === "--sprt-nodes") args.sprtNodes = intArg(next(), a, 1);
        else if (a === "--sprt-opening-moves") args.sprtOpeningMoves = intArg(next(), a, 0);
        else if (a === "--sprt-workers") args.sprtWorkers = intArg(next(), a, 1);
        else if (a === "--sprt-max-pairs") args.sprtMaxPairs = intArg(next(), a, 1);
        else if (a === "--sprt-elo0") args.sprtElo0 = floatArg(next(), a, -Infinity);
        else if (a === "--sprt-elo1") args.sprtElo1 = floatArg(next(), a, -Infinity);
        else if (a === "--sprt-alpha") args.sprtAlpha = floatArg(next(), a, 0);
        else if (a === "--sprt-beta") args.sprtBeta = floatArg(next(), a, 0);
        else if (a === "--anchor-net") args.anchorNet = required(next(), a);
        else if (a === "--anchor-pairs") args.anchorPairs = intArg(next(), a, 1);
        else if (a === "--early-stopping-patience") args.earlyStoppingPatience = intArg(next(), a, 0);
        else if (a === "--no-restore-best") args.noRestoreBest = true;
        else if (a === "--help" || a === "-h") {
            printHelp();
            process.exit(0);
        } else {
            throw new Error(`Unknown arg: ${a}`);
        }
    }

    return args;
}

function required(value, name) {
    if (value == null || value === "") {
        throw new Error(`Missing value for ${name}`);
    }
    return value;
}

function intArg(value, name, min) {
    const n = Number(value);
    if (!Number.isInteger(n) || n < min) {
        throw new Error(`Invalid ${name}: ${value}`);
    }
    return n;
}

function floatArg(value, name, min) {
    const n = Number(value);
    if (!Number.isFinite(n) || n < min) {
        throw new Error(`Invalid ${name}: ${value}`);
    }
    return n;
}

function printHelp() {
    console.log(
        [
            "Usage: node orchestrator/orchestrate-zero.mjs [options]",
            "",
            "Core options:",
            "  --iterations <N>          Number of generations to run (default: 1)",
            "  --start-gen <N>           First generation index for output names (default: 1)",
            "  --games-per-gen <N>       Self-play games generated per generation (default: 10000)",
            "  --start-net <path>        ONNX net used for first datagen (default: random_init.onnx)",
            "  --start-checkpoint <path> QNXX checkpoint to resume training from (optional)",
            "  --data <path>             Shared dataset .bin path, append mode (default: zero-trainer/selfplay.bin)",
            "  --weights-dir <dir>       Where genN.onnx/genN.onxx are saved (default: zero-trainer/weights)",
            "",
            "Train args forwarded to train.py:",
            "  --window <N> --steps <N> --batch <N> --lr <F> --weight-decay <F> --defender-weight <F>",
            "  --no-restore-best             Don't restore best val_loss checkpoint, use final model weights",
            "",
            "SPRT validation (after each training):",
            "  --no-sprt                 Skip SPRT validation (accept every network)",
            "  --no-gate                 Always accept candidate, but run match for Elo measurement",
            "  --no-gate-pairs <N>       Number of pairs to play in no-gate mode (default: 100)",
            "  --sprt-nodes <N>          MCTS nodes per move for SPRT games (default: 200)",
            "  --sprt-opening-moves <N>  Random moves for opening generation (default: 16)",
            "  --sprt-workers <N>        Parallel SPRT game workers (default: 24)",
            "  --sprt-max-pairs <N>      Max game pairs before failing (default: 2500)",
            "  --sprt-elo0 <N>           SPRT H0 elo (default: 0)",
            "  --sprt-elo1 <N>           SPRT H1 elo (default: 5)",
            "  --sprt-alpha <F>          Type I error rate (default: 0.05)",
            "  --sprt-beta <F>           Type II error rate (default: 0.05)",
            "",
            "Anchor testing (absolute Elo measurement):",
            "  --anchor-net <path>       Play each accepted net against this anchor for absolute Elo",
            "  --anchor-pairs <N>        Pairs to play against anchor (default: 100)",
            "",
            "Runtime:",
            "  --workers <N>             Parallel engine processes for datagen (default: 1)",
            "  --python <path>           Python executable (auto-detected if omitted)",
            "  --debug-engine            Use debug cargo build for datagen",
        ].join("\n"),
    );
}

function run(cmd, cmdArgs, cwd = process.cwd()) {
    return new Promise((resolve, reject) => {
        const child = spawn(cmd, cmdArgs, { stdio: "inherit", cwd, shell: false });
        child.on("error", reject);
        child.on("exit", (code) => {
            if (code === 0) resolve();
            else reject(new Error(`${cmd} exited with code ${code ?? "null"}`));
        });
    });
}

/**
 * Run a command and return whether it succeeded (exit code 0).
 */
function runCheck(cmd, cmdArgs, cwd = process.cwd()) {
    return new Promise((resolve) => {
        const child = spawn(cmd, cmdArgs, { stdio: "inherit", cwd, shell: false });
        child.on("error", () => resolve(false));
        child.on("exit", (code) => resolve(code === 0));
    });
}

function pickPython(explicit, projectRoot) {
    if (explicit) return explicit;
    const venvPython = path.join(projectRoot, "zero-trainer", ".venv", "Scripts", "python.exe");
    if (fs.existsSync(venvPython)) return venvPython;
    return "python";
}

function genName(idx) {
    return `gen${String(idx).padStart(4, "0")}`;
}

function resolveEngineBinary(debugEngine) {
    const profileDir = debugEngine ? "debug" : "release";
    const exeName = process.platform === "win32" ? "taflzero.exe" : "taflzero";
    //return path.join("target", profileDir, exeName);
    return exeName;
}

async function ensureEngineBinary(debugEngine, engineBinPath) {
    if (fs.existsSync(engineBinPath)) {
        return;
    }

    const buildArgs = ["build"];
    if (!debugEngine) {
        buildArgs.push("--release");
    }
    await run("cargo", buildArgs);

    if (!fs.existsSync(engineBinPath)) {
        throw new Error(`Engine binary not found after build: ${engineBinPath}`);
    }
}

function tryUnlink(filePath) {
    try { fs.unlinkSync(filePath); } catch {}
}

async function main() {
    let args;
    try {
        args = parseArgs(process.argv.slice(2));
    } catch (err) {
        console.error(String(err.message ?? err));
        printHelp();
        process.exit(2);
    }

    const projectRoot = path.resolve(__dirname, "..");

    // Resolve paths: user-supplied relative paths against cwd, defaults against project root
    const defaults = { data: "zero-trainer/selfplay.bin", weightsDir: "zero-trainer/weights", startNet: "random_init.onnx" };
    const resolvePath = (p, key) => {
        if (path.isAbsolute(p)) return p;
        // If it's still the default value, resolve against project root
        if (defaults[key] && p === defaults[key]) return path.resolve(projectRoot, p);
        // User-supplied relative path — resolve against cwd
        return path.resolve(p);
    };
    args.data = resolvePath(args.data, "data");
    args.weightsDir = resolvePath(args.weightsDir, "weightsDir");
    args.startNet = resolvePath(args.startNet, "startNet");
    if (args.startCheckpoint) args.startCheckpoint = path.resolve(args.startCheckpoint);
    if (args.anchorNet) args.anchorNet = path.resolve(args.anchorNet);

    fs.mkdirSync(path.dirname(args.data), { recursive: true });
    fs.mkdirSync(args.weightsDir, { recursive: true });

    const python = pickPython(args.python, projectRoot);
    let currentNet = path.normalize(args.startNet);
    let currentCheckpoint = args.startCheckpoint ? path.normalize(args.startCheckpoint) : null;
    const engineBin = args.engineBin
        ? path.resolve(args.engineBin)
        : path.normalize(resolveEngineBinary(args.debugEngine));
    if (!args.engineBin) {
        await ensureEngineBinary(args.debugEngine, engineBin);
    }

    const sprtMatchScript = path.join(__dirname, "sprt-match.mjs");

    let genIdx = args.startGen;
    let accepted = 0;
    while (accepted < args.iterations) {
        const name = genName(genIdx);
        const finalOnnx = path.join(args.weightsDir, `${name}.onnx`);
        const finalQnxx = path.join(args.weightsDir, `${name}.onxx`);
        const candidateOnnx = path.join(args.weightsDir, `${name}.candidate.onnx`);
        const candidateQnxx = path.join(args.weightsDir, `${name}.candidate.onxx`);

        console.log(`\n=== Generation ${genIdx} ===`);
        console.log(`Datagen net: ${currentNet}`);
        console.log(`Dataset: ${args.data}`);

        // ── Step 1: Data generation ──────────────────────────────────
        const genDatasetArgs = [
            path.join(__dirname, "gen-dataset.mjs"),
            "--out",
            path.normalize(args.data),
            "--net",
            currentNet,
            "--engine-bin",
            engineBin,
            "--datagen-count",
            String(args.gamesPerGen),
        ];
        if (args.workers > 1) {
            genDatasetArgs.push("--workers", String(args.workers));
        }
        if (args.debugEngine) {
            genDatasetArgs.push("--debug");
        }
        await run("node", genDatasetArgs);

        // ── Step 2: Training ─────────────────────────────────────────
        console.log(`Training -> candidate: ${candidateOnnx}`);
        const trainArgs = [
            path.join(projectRoot, "zero-trainer", "train.py"),
            "--data",
            path.normalize(args.data),
            "--out",
            path.normalize(candidateOnnx),
            "--save-checkpoint",
            path.normalize(candidateQnxx),
            "--window",
            String(args.window),
            "--steps",
            String(args.steps),
            "--batch",
            String(args.batch),
            "--lr",
            String(args.lr),
            "--weight-decay",
            String(args.weightDecay),
            "--defender-weight",
            String(args.defenderWeight),
            "--early-stopping-patience",
            String(args.earlyStoppingPatience),
        ];
        if (args.noRestoreBest) {
            trainArgs.push("--no-restore-best");
        }
        if (currentCheckpoint) {
            trainArgs.push("--checkpoint", currentCheckpoint);
        }
        await run(python, trainArgs);

        // ── Step 3: SPRT validation ──────────────────────────────────
        if (args.noSprt) {
            // No SPRT — accept candidate directly
            fs.renameSync(candidateOnnx, finalOnnx);
            fs.renameSync(candidateQnxx, finalQnxx);
            currentNet = path.normalize(finalOnnx);
            currentCheckpoint = path.normalize(finalQnxx);
            console.log(`[no-sprt] Accepted candidate as ${name}`);
        } else if (args.noGate) {
            // No gating — always accept, but run match for Elo measurement
            console.log(`\n--- Elo measurement: ${name} candidate vs current (${args.noGatePairs} pairs) ---`);
            const resultFile = path.join(args.weightsDir, `${name}.sprt.json`);
            const matchArgs = [
                sprtMatchScript,
                "--main-net", currentNet,
                "--candidate-net", path.normalize(candidateOnnx),
                "--engine-bin", engineBin,
                "--nodes", String(args.sprtNodes),
                "--opening-moves", String(args.sprtOpeningMoves),
                "--workers", String(args.sprtWorkers),
                "--max-pairs", String(args.noGatePairs),
                "--no-gate",
                "--result-file", resultFile,
            ];

            const MAX_RETRIES = 3;
            for (let attempt = 1; attempt <= MAX_RETRIES; attempt++) {
                tryUnlink(resultFile);
                const ok = await runCheck("node", matchArgs);
                if (ok || fs.existsSync(resultFile)) break;
                console.log(`[Match] Engine crashed (attempt ${attempt}/${MAX_RETRIES}), restarting...`);
            }

            // Always accept candidate
            fs.renameSync(candidateOnnx, finalOnnx);
            fs.renameSync(candidateQnxx, finalQnxx);
            currentNet = path.normalize(finalOnnx);
            currentCheckpoint = path.normalize(finalQnxx);

            // Log Elo to CSV
            if (fs.existsSync(resultFile)) {
                try {
                    const result = JSON.parse(fs.readFileSync(resultFile, "utf-8"));
                    const csvPath = path.join(args.weightsDir, "sprt-results.csv");
                    const needsHeader = !fs.existsSync(csvPath);
                    const csvLine = `${name},${result.eloDiff.toFixed(1)},${result.pct.toFixed(1)},${result.score},${result.total},${result.wins},${result.losses},${result.draws},${result.llr.toFixed(3)}\n`;
                    if (needsHeader) {
                        fs.writeFileSync(csvPath, "generation,elo,score_pct,score,pairs,wins,losses,draws,llr\n" + csvLine);
                    } else {
                        fs.appendFileSync(csvPath, csvLine);
                    }
                    console.log(`[no-gate] Accepted ${name} (Elo: ${result.eloDiff.toFixed(1)})`);
                } catch {
                    console.log(`[no-gate] Accepted ${name} (Elo measurement failed)`);
                }
            } else {
                console.log(`[no-gate] Accepted ${name} (no measurement)`);
            }

            // Clean up result file
            tryUnlink(resultFile);
        } else {
            console.log(`\n--- SPRT validation: ${name} candidate vs current ---`);
            const resultFile = path.join(args.weightsDir, `${name}.sprt.json`);
            const sprtArgs = [
                sprtMatchScript,
                "--main-net", currentNet,
                "--candidate-net", path.normalize(candidateOnnx),
                "--engine-bin", engineBin,
                "--nodes", String(args.sprtNodes),
                "--opening-moves", String(args.sprtOpeningMoves),
                "--workers", String(args.sprtWorkers),
                "--max-pairs", String(args.sprtMaxPairs),
                "--sprt-elo0", String(args.sprtElo0),
                "--sprt-elo1", String(args.sprtElo1),
                "--sprt-alpha", String(args.sprtAlpha),
                "--sprt-beta", String(args.sprtBeta),
                "--result-file", resultFile,
            ];

            const MAX_SPRT_RETRIES = 3;
            let passed = false;
            for (let attempt = 1; attempt <= MAX_SPRT_RETRIES; attempt++) {
                tryUnlink(resultFile);
                const ok = await runCheck("node", sprtArgs);
                if (ok || fs.existsSync(resultFile)) {
                    // Clean exit (pass or fail) — result file exists
                    passed = ok;
                    break;
                }
                // No result file = crash, retry
                console.log(`[SPRT] Engine crashed (attempt ${attempt}/${MAX_SPRT_RETRIES}), restarting test...`);
                if (attempt === MAX_SPRT_RETRIES) {
                    console.log(`[SPRT] All ${MAX_SPRT_RETRIES} attempts crashed — treating as FAILED`);
                }
            }

            if (passed) {
                fs.renameSync(candidateOnnx, finalOnnx);
                fs.renameSync(candidateQnxx, finalQnxx);
                currentNet = path.normalize(finalOnnx);
                currentCheckpoint = path.normalize(finalQnxx);
                console.log(`[SPRT] PASSED — promoted candidate to ${name}`);

                // Append to CSV log
                if (fs.existsSync(resultFile)) {
                    try {
                        const result = JSON.parse(fs.readFileSync(resultFile, "utf-8"));
                        const csvPath = path.join(args.weightsDir, "sprt-results.csv");
                        const needsHeader = !fs.existsSync(csvPath);
                        const csvLine = `${name},${result.eloDiff.toFixed(1)},${result.pct.toFixed(1)},${result.score},${result.total},${result.wins},${result.losses},${result.draws},${result.llr.toFixed(3)}\n`;
                        if (needsHeader) {
                            fs.writeFileSync(csvPath, "generation,elo,score_pct,score,pairs,wins,losses,draws,llr\n" + csvLine);
                        } else {
                            fs.appendFileSync(csvPath, csvLine);
                        }
                    } catch {}
                }
            } else {
                tryUnlink(candidateOnnx);
                tryUnlink(candidateQnxx);
                console.log(`[SPRT] FAILED — keeping current net: ${currentNet}`);
            }

            // Clean up result file
            tryUnlink(resultFile);
        }

        // Only advance generation counter on acceptance
        if (args.noSprt || args.noGate || (currentNet === path.normalize(finalOnnx))) {
            // ── Step 4: Anchor test (absolute Elo) ──────────────────────
            if (args.anchorNet) {
                console.log(`\n--- Anchor test: ${name} vs anchor (${args.anchorPairs} pairs) ---`);
                const anchorResultFile = path.join(args.weightsDir, `${name}.anchor.json`);
                const anchorArgs = [
                    sprtMatchScript,
                    "--main-net", path.normalize(args.anchorNet),
                    "--candidate-net", path.normalize(finalOnnx),
                    "--engine-bin", engineBin,
                    "--nodes", String(args.sprtNodes),
                    "--opening-moves", String(args.sprtOpeningMoves),
                    "--workers", String(args.sprtWorkers),
                    "--max-pairs", String(args.anchorPairs),
                    "--no-gate",
                    "--result-file", anchorResultFile,
                ];

                const MAX_RETRIES = 3;
                for (let attempt = 1; attempt <= MAX_RETRIES; attempt++) {
                    tryUnlink(anchorResultFile);
                    const ok = await runCheck("node", anchorArgs);
                    if (ok || fs.existsSync(anchorResultFile)) break;
                    console.log(`[Anchor] Engine crashed (attempt ${attempt}/${MAX_RETRIES}), restarting...`);
                }

                if (fs.existsSync(anchorResultFile)) {
                    try {
                        const result = JSON.parse(fs.readFileSync(anchorResultFile, "utf-8"));
                        const csvPath = path.join(args.weightsDir, "anchor-results.csv");
                        const needsHeader = !fs.existsSync(csvPath);
                        const csvLine = `${name},${result.eloDiff.toFixed(1)},${result.pct.toFixed(1)},${result.score},${result.total},${result.wins},${result.losses},${result.draws}\n`;
                        if (needsHeader) {
                            fs.writeFileSync(csvPath, "generation,elo_vs_anchor,score_pct,score,pairs,wins,losses,draws\n" + csvLine);
                        } else {
                            fs.appendFileSync(csvPath, csvLine);
                        }
                        console.log(`[Anchor] ${name}: Elo ${result.eloDiff.toFixed(1)} vs anchor`);
                    } catch {
                        console.log(`[Anchor] ${name}: measurement failed`);
                    }
                }
                tryUnlink(anchorResultFile);
            }

            genIdx++;
            accepted++;
            console.log(`Accepted generation ${genIdx - 1} (${accepted}/${args.iterations})`);
        } else {
            console.log(`Rejected generation ${genIdx}, will retry`);
        }
    }

    console.log("\nOrchestration finished.");
}

main().catch((err) => {
    console.error(err.stack || String(err));
    process.exit(1);
});
