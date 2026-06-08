// Comprehensive browser-path check (headless). Exercises the code paths the
// browser uses but the quiet probe skips: render() (run at sim init),
// updateUi + pushHistory, every disturbance event, and the public EcosystemLab
// API. Fails loudly on any exception.
import { loadSim } from "./harness.mjs";

function main() {
  // loadSim runs the IIFE, which calls initWorld() and render() once -> this
  // already exercises drawWorld/drawFlora/drawAgent/colorForCell/drawHistory.
  const sim = loadSim({ seed: 4242 });
  sim.setQuiet(false); // exercise updateUi + pushHistory each tick

  for (let i = 0; i < 80; i++) sim.tick();

  // Exercise every disturbance event handler, and re-run render() explicitly
  // under each map overlay so all colorForCell branches are covered.
  for (const ev of ["drought", "flood", "fire", "disease"]) {
    sim.trigger(ev);
  }
  for (const overlay of ["biome", "elevation", "water", "nutrients", "temperature", "toxicity"]) {
    sim.getState().overlay = overlay;
    sim.render();
  }
  for (let i = 0; i < 60; i++) sim.tick();

  const s = sim.stats();
  const audit = sim.energyAudit();
  const live = sim.getAgents().filter((a) => !a.dead).length;
  const ok =
    live > 0 &&
    Number.isFinite(s.plantBiomass) &&
    Number.isFinite(s.stability) &&
    Object.values(s.counts).every((c) => Number.isFinite(c) && c >= 0);
  console.log("UI-path OK:", ok);
  console.log(
    "  agents:", live,
    "counts:", JSON.stringify(s.counts),
    "plant:", Math.round(s.plantBiomass),
    "stability:", Math.round(s.stability),
    "disease%:", (s.diseasePct * 100).toFixed(2),
  );
  console.log("  matter pools: soil", Math.round(audit.soilNutrients), "detritus", Math.round(audit.detritus));
  if (!ok) process.exit(1);
}

try {
  main();
} catch (e) {
  console.error("UI-PATH ERROR:", e.message, "\n", e.stack);
  process.exit(1);
}
