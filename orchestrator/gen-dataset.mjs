import { spawn } from "node:child_process";
import fs from "node:fs";
import path from "node:path";

function parseArgs(argv) {
    const args = {
        out: null,
        net: "./gen1.onxx",
        datagenCount: null,
        engineBin: null,
        release: true,
        workers: 1,
    };

    for (let i = 0; i < argv.length; i += 1) {
        const a = argv[i];
        if (a === "--out") {
            args.out = argv[++i] ?? null;
        } else if (a === "--net") {
            args.net = argv[++i] ?? args.net;
        } else if (a === "--datagen-count") {
            const raw = argv[++i];
            const v = Number(raw);
            if (!Number.isInteger(v) || v <= 0) {
                throw new Error(`Invalid --datagen-count: ${raw}`);
            }
            args.datagenCount = v;
        } else if (a === "--engine-bin") {
            args.engineBin = argv[++i] ?? null;
        } else if (a === "--workers") {
            const raw = argv[++i];
            const v = Number(raw);
            if (!Number.isInteger(v) || v < 1) {
                throw new Error(`Invalid --workers: ${raw}`);
            }
            args.workers = v;
        } else if (a === "--debug") {
            args.release = false;
        } else if (a === "--help" || a === "-h") {
            printHelp();
            process.exit(0);
        } else {
            throw new Error(`Unknown arg: ${a}`);
        }
    }

    if (!args.out) {
        throw new Error("Missing required --out <output.bin>");
    }

    return args;
}

function printHelp() {
    console.log(
        [
            "Usage: node gen-dataset.mjs --out <output.bin> [--net <model.onnx>] [--datagen-count <games>] [--engine-bin <path>] [--workers <N>] [--debug]",
            "",
            "  --workers <N>   Number of parallel engine processes (default: 1)",
            "Runs engine binary directly if --engine-bin is set, otherwise falls back to cargo run [--release]",
        ].join("\n"),
    );
}

function run(cmd, cmdArgs, cwd = process.cwd()) {
    return new Promise((resolve, reject) => {
        const child = spawn(cmd, cmdArgs, { stdio: "inherit", cwd, shell: false });
        child.on("error", reject);
        child.on("exit", (code) => {
            if (code === 0) {
                resolve();
            } else {
                reject(new Error(`${cmd} exited with code ${code ?? "null"}`));
            }
        });
    });
}

function buildEngineArgs(netPath, outPath, count) {
    const engineArgs = [
        "--net",
        path.normalize(netPath),
        "--datagen",
        path.normalize(outPath),
    ];
    if (count != null) {
        engineArgs.push("--datagen-count", String(count));
    }
    return engineArgs;
}

function runEngine(args, engineArgs) {
    if (args.engineBin) {
        return run(path.normalize(args.engineBin), engineArgs);
    }
    const cargoArgs = ["run"];
    if (args.release) {
        cargoArgs.push("--release");
    }
    cargoArgs.push("--", ...engineArgs);
    return run("cargo", cargoArgs);
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

    const totalGames = args.datagenCount;
    const workers = args.workers;

    if (workers === 1 || totalGames == null) {
        const engineArgs = buildEngineArgs(args.net, args.out, totalGames);
        await runEngine(args, engineArgs);
        return;
    }

    // Dynamic work queue - workers pick up batches until all games are played
    const BATCH_SIZE = Math.max(1, Math.floor(totalGames / workers / 10));
    let gamesRemaining = totalGames;
    let batchIndex = 0;
    const tmpFiles = [];

    let gamesCompleted = 0;

    async function runWorker(workerId) {
        while (gamesRemaining > 0) {
            const batch = Math.min(BATCH_SIZE, gamesRemaining);
            gamesRemaining -= batch;
            if (batch <= 0) break;
            const idx = batchIndex++;
            const tmpFile = `${args.out}.worker${workerId}.${idx}.tmp`;
            tmpFiles.push(tmpFile);
            const engineArgs = buildEngineArgs(args.net, tmpFile, batch);
            await runEngine(args, engineArgs);
            gamesCompleted += batch;
            const pct = ((gamesCompleted / totalGames) * 100).toFixed(1);
            console.log(`Progress: ${gamesCompleted}/${totalGames} games (${pct}%)`);
        }
    }

    const promises = [];
    for (let w = 0; w < workers; w++) {
        promises.push(runWorker(w));
    }

    await Promise.all(promises);

    // Concatenate all temp files into the final output (append mode)
    console.log(`Merging ${tmpFiles.length} parts into ${args.out}`);
    const outFd = fs.openSync(args.out, "a");
    try {
        for (const tmp of tmpFiles) {
            const data = fs.readFileSync(tmp);
            fs.writeSync(outFd, data);
            fs.unlinkSync(tmp);
        }
    } finally {
        fs.closeSync(outFd);
    }
}

main().catch((err) => {
    console.error(err.stack || String(err));
    process.exit(1);
});
