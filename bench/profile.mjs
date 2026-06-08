import { loadSim } from "./harness.mjs";
const sim = loadSim({ seed: 1000 });
const N = Number(process.argv[2] || 600);
for (let i = 0; i < N; i += 1) sim.tick();
console.log("ticked", N, "agents", sim.getAgents().filter((a) => !a.dead).length);
