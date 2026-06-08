// Ecology probe (parallel): spawns one worker process per seed so a multi-seed
// run finishes in roughly single-seed wall-clock time on a multi-core machine.
//
// Usage: node bench/probe.mjs [ticks] [seedCount] [seedBase]
//   SIM_FILE=<path>  target a specific sim build (defaults to ../sim.js)
import { spawn } from "node:child_process";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { formatReport } from "./analyze.mjs";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const TICKS = Number(process.argv[2] || 4000);
const SEEDS = Number(process.argv[3] || 4);
const SEED_BASE = Number(process.argv[4] || 1000);
const SAMPLE = 25;

function runSeed(seed) {
  return new Promise((resolve, reject) => {
    // spawn (not fork): no IPC channel, so the worker exits cleanly when its
    // script ends instead of lingering and hanging the parent.
    const child = spawn(process.execPath, [path.join(__dirname, "worker.mjs"), String(seed), String(TICKS), String(SAMPLE)], {
      stdio: ["ignore", "pipe", "inherit"],
      env: process.env,
    });
    let out = "";
    child.stdout.on("data", (d) => (out += d));
    child.on("error", reject);
    child.on("close", (code) => {
      if (code !== 0) return reject(new Error(`worker seed ${seed} exited ${code}`));
      try {
        resolve(JSON.parse(out.trim().split("\n").pop()));
      } catch (e) {
        reject(new Error(`bad worker output for seed ${seed}: ${out.slice(0, 200)}`));
      }
    });
  });
}

const seeds = Array.from({ length: SEEDS }, (_, i) => SEED_BASE + i * 137);
const t0 = Date.now();
const runs = await Promise.all(seeds.map(runSeed));
const secs = ((Date.now() - t0) / 1000).toFixed(1);
process.stdout.write(formatReport(runs, { seeds: SEEDS, ticks: TICKS }));
process.stdout.write(`(${seeds.length} seeds in ${secs}s wall-clock, parallel)\n`);
