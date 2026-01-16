#!/usr/bin/env node
import fs from "fs";
import os from "os";
import path from "path";
import readline from "readline";
import crypto from "crypto";
import { spawn } from "child_process";

const BUCKETS = 1024;
const TMP_DIR = "./buckets";
const OUTPUT = "output.txt";
const SEED = "bucket-shuffle-v1";

fs.mkdirSync(TMP_DIR, { recursive: true });

/* ---------- utils ---------- */

function hashBucket(line) {
  const h = crypto
    .createHash("sha256")
    .update(SEED)
    .update(line)
    .digest();

  return h.readUInt32BE(0) % BUCKETS;
}

function shuffleArray(arr) {
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
}

/* ---------- phase 1: bucket split ---------- */

const writers = Array.from({ length: BUCKETS }, (_, i) =>
  fs.createWriteStream(path.join(TMP_DIR, `b${i}.tmp`))
);

const rl = readline.createInterface({
  input: process.stdin,
  crlfDelay: Infinity,
});

for await (const line of rl) {
  const b = hashBucket(line);
  writers[b].write(line + "\n");
}

await Promise.all(writers.map(w => new Promise(r => w.end(r))));

/* ---------- phase 2: shuffle + merge ---------- */

const bucketIds = Array.from({ length: BUCKETS }, (_, i) => i);
shuffleArray(bucketIds);

const out = fs.createWriteStream(OUTPUT, { flags: "w" });

for (const i of bucketIds) {
  const file = path.join(TMP_DIR, `b${i}.tmp`);

  if (!fs.existsSync(file) || fs.statSync(file).size === 0) {
    fs.rmSync(file, { force: true });
    continue;
  }

  await new Promise((resolve, reject) => {
    const shuf = spawn("shuf", [file]);

    shuf.stdout.pipe(out, { end: false });
    shuf.stderr.on("data", d => process.stderr.write(d));

    shuf.on("error", reject);
    shuf.on("close", code => {
      if (code !== 0) reject(new Error(`shuf exited ${code}`));
      else resolve();
    });
  });

  fs.rmSync(file);
}

out.end();
