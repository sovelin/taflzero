import { spawn } from "node:child_process";

const fenDir = process.argv[2];
const threads = Number(process.argv[3] ?? 4);
const BIN = "target/release/zevratafl-rust.exe";

if (!fenDir) {
    console.error("Usage: node dataset-dataset.mjs <fenDir> [threads]");
    process.exit(1);
}

const sleep = (ms) => new Promise(res => setTimeout(res, ms));

(async () => {
    for (let i = 1; i <= threads; i++) {
        const fen = `${fenDir}/${i}.bin`;

        spawn(
            BIN,
            [fen],
            { stdio: "inherit" }
        );

        await sleep(1000); // 🔥 1000 ms between starts
    }
})();
