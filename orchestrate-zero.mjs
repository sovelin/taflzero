import { spawn } from "node:child_process";
import fs from "node:fs";
import path from "node:path";

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
        debugEngine: false,
        workers: 1,
        saveBestCheckpoint: true,
        useBestCheckpoint: true,
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
        else if (a === "--debug-engine") args.debugEngine = true;
        else if (a === "--no-save-best-checkpoint") {
            args.saveBestCheckpoint = false;
            args.useBestCheckpoint = false;
        } else if (a === "--use-last-checkpoint") {
            args.useBestCheckpoint = false;
        }
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
            "Usage: node orchestrate-zero.mjs [options]",
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
            "",
            "Checkpointing:",
            "  --no-save-best-checkpoint Disable saving best .onxx checkpoint per generation (default: save best)",
            "  --use-last-checkpoint     Use last .onxx for next generation instead of best (default: use best)",
            "",
            "Runtime:",
            "  --workers <N>             Parallel engine processes for datagen (default: 1)",
            "  --python <path>           Python executable (default: zero-trainer/.venv/Scripts/python.exe if exists, else python)",
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

function pickPython(explicit) {
    if (explicit) return explicit;
    const venvPython = path.join("zero-trainer", ".venv", "Scripts", "python.exe");
    if (fs.existsSync(venvPython)) return venvPython;
    return "python";
}

function genName(idx) {
    return `gen${String(idx).padStart(4, "0")}`;
}

function resolveEngineBinary(debugEngine) {
    const profileDir = debugEngine ? "debug" : "release";
    const exeName = process.platform === "win32" ? "zevratafl-rust.exe" : "zevratafl-rust";
    return path.join("target", profileDir, exeName);
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

async function main() {
    let args;
    try {
        args = parseArgs(process.argv.slice(2));
    } catch (err) {
        console.error(String(err.message ?? err));
        printHelp();
        process.exit(2);
    }

    fs.mkdirSync(path.dirname(args.data), { recursive: true });
    fs.mkdirSync(args.weightsDir, { recursive: true });

    const python = pickPython(args.python);
    let currentNet = path.normalize(args.startNet);
    let currentCheckpoint = args.startCheckpoint ? path.normalize(args.startCheckpoint) : null;
    const engineBin = path.normalize(resolveEngineBinary(args.debugEngine));
    await ensureEngineBinary(args.debugEngine, engineBin);

    for (let i = 0; i < args.iterations; i += 1) {
        const genIdx = args.startGen + i;
        const name = genName(genIdx);
        const nextOnnx = path.join(args.weightsDir, `${name}.onnx`);
        const nextQnxx = path.join(args.weightsDir, `${name}.onxx`);
        const nextBestQnxx = path.join(args.weightsDir, `${name}.best.onxx`);
        const nextBestOnnx = path.join(args.weightsDir, `${name}.best.onnx`);

        console.log(`\n=== Generation ${genIdx} ===`);
        console.log(`Datagen net: ${currentNet}`);
        console.log(`Dataset: ${args.data}`);

        const genDatasetArgs = [
            "gen-dataset.mjs",
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

        console.log(`Training -> ${nextOnnx} / ${nextQnxx}`);
        const trainArgs = [
            path.join("zero-trainer", "train.py"),
            "--data",
            path.normalize(args.data),
            "--out",
            path.normalize(nextOnnx),
            "--save-checkpoint",
            path.normalize(nextQnxx),
            "--save-best-checkpoint",
            path.normalize(nextBestQnxx),
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
        ];
        if (!args.saveBestCheckpoint) {
            const idx = trainArgs.indexOf("--save-best-checkpoint");
            if (idx >= 0) {
                trainArgs.splice(idx, 2);
            }
        }
        if (currentCheckpoint) {
            trainArgs.push("--checkpoint", currentCheckpoint);
        }
        await run(python, trainArgs);

        if (
            args.useBestCheckpoint &&
            args.saveBestCheckpoint &&
            fs.existsSync(nextBestQnxx) &&
            fs.existsSync(nextBestOnnx)
        ) {
            currentNet = path.normalize(nextBestOnnx);
            currentCheckpoint = path.normalize(nextBestQnxx);
            console.log(`Using best checkpoint for next gen: ${currentCheckpoint}`);
        } else {
            currentNet = path.normalize(nextOnnx);
            currentCheckpoint = path.normalize(nextQnxx);
            console.log(`Using last checkpoint for next gen: ${currentCheckpoint}`);
        }
        console.log(`Completed generation ${genIdx}`);
    }

    console.log("\nOrchestration finished.");
}

main().catch((err) => {
    console.error(err.stack || String(err));
    process.exit(1);
});
