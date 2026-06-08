// One-seed worker: runs the sim and prints analyzed metrics as a JSON line.
// Invoked by probe.mjs via child_process for cross-core parallelism.
import { loadSim, runSim } from "./harness.mjs";
import { analyzeRun } from "./analyze.mjs";

const seed = Number(process.argv[2]);
const ticks = Number(process.argv[3]);
const sample = Number(process.argv[4] || 25);

const sim = loadSim({ seed });
const series = runSim(sim, { ticks, sample });
const result = analyzeRun(series);
// Attach per-guild population caps so the report can show cap utilization
// (mean / cap): a guild well below its cap is emergently regulated, one pinned
// near it is still ceiling-limited.
const info = sim.getTypeInfo();
result.caps = Object.fromEntries(Object.keys(info).map((t) => [t, info[t].maxCount]));
process.stdout.write(JSON.stringify(result) + "\n");
