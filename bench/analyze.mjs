// Shared statistics + ecology scoring used by the probe and its workers.
export const GUILDS = ["herbivore", "predator", "decomposer", "pollinator", "engineer"];

export function mean(xs) {
  return xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : 0;
}
export function std(xs) {
  if (xs.length < 2) return 0;
  const m = mean(xs);
  return Math.sqrt(mean(xs.map((x) => (x - m) ** 2)));
}
export function cv(xs) {
  const m = mean(xs);
  return m > 1e-9 ? std(xs) / m : 0;
}

// Cross-correlation of b vs a at integer lags. Positive lag => b follows a.
export function bestLag(a, b, maxLag) {
  const n = Math.min(a.length, b.length);
  const am = mean(a);
  const bm = mean(b);
  const ad = a.map((x) => x - am);
  const bd = b.map((x) => x - bm);
  const denom = Math.sqrt(ad.reduce((s, x) => s + x * x, 0) * bd.reduce((s, x) => s + x * x, 0)) || 1;
  let best = { lag: 0, corr: -Infinity };
  for (let lag = -maxLag; lag <= maxLag; lag += 1) {
    let s = 0;
    for (let i = 0; i < n; i += 1) {
      const j = i + lag;
      if (j < 0 || j >= n) continue;
      s += ad[i] * bd[j];
    }
    const corr = s / denom;
    if (corr > best.corr) best = { lag, corr };
  }
  return best;
}

export function analyzeRun(series) {
  const burn = Math.floor(series.length * 0.25);
  const tail = series.slice(burn);

  const minCounts = {};
  const meanCounts = {};
  for (const g of GUILDS) {
    const xs = tail.map((s) => s.counts[g]);
    minCounts[g] = Math.min(...xs);
    meanCounts[g] = mean(xs);
  }

  const maxLag = Math.min(40, Math.floor(tail.length / 3));
  const lag = bestLag(
    tail.map((s) => s.counts.herbivore),
    tail.map((s) => s.counts.predator),
    maxLag,
  );

  const pool = tail.map((s) => s.soilNutrients + s.detritus + s.plantBiomass);
  const driftOf = (xs) => (xs.length ? (xs[xs.length - 1] - xs[0]) / (Math.abs(xs[0]) || 1) : 0);
  const poolDriftPct = driftOf(pool);
  const soilDriftPct = driftOf(tail.map((s) => s.soilNutrients));
  const plantDriftPct = driftOf(tail.map((s) => s.plantBiomass));
  const detritusDriftPct = driftOf(tail.map((s) => s.detritus));

  return {
    minCounts,
    meanCounts,
    extinct: GUILDS.filter((g) => minCounts[g] <= 0),
    herbCV: cv(tail.map((s) => s.counts.herbivore)),
    predCV: cv(tail.map((s) => s.counts.predator)),
    plantCV: cv(tail.map((s) => s.plantBiomass)),
    lagSamples: lag.lag,
    lagCorr: lag.corr,
    pyramid: {
      plant: mean(tail.map((s) => s.plantBiomass)),
      herb: mean(tail.map((s) => s.counts.herbivore)),
      pred: mean(tail.map((s) => s.counts.predator)),
    },
    // Lindeman energy pyramid: standing somatic energy per trophic level.
    herbEnergy: mean(tail.map((s) => (s.energyByType ? s.energyByType.herbivore : 0))),
    predEnergy: mean(tail.map((s) => (s.energyByType ? s.energyByType.predator : 0))),
    poolDriftPct,
    soilDriftPct,
    plantDriftPct,
    detritusDriftPct,
    meanSoil: mean(tail.map((s) => s.soilNutrients)),
    meanDetritus: mean(tail.map((s) => s.detritus)),
    finalStability: mean(tail.map((s) => s.stability)),
    finalToxicity: mean(tail.map((s) => s.toxicity)),
    finalDisease: mean(tail.map((s) => s.diseasePct)),
    finalShannon: mean(tail.map((s) => s.shannon)),
  };
}

function fmt(n, d = 2) {
  return Number.isFinite(n) ? n.toFixed(d) : "NaN";
}

export function formatReport(runs, { seeds, ticks }) {
  const lines = [];
  lines.push(`\n=== Ecology Probe: ${seeds} seeds x ${ticks} ticks ===\n`);

  lines.push("Extinction rate (fraction of runs where guild hit 0):");
  for (const g of GUILDS) {
    const rate = mean(runs.map((r) => (r.extinct.includes(g) ? 1 : 0)));
    lines.push(`  ${g.padEnd(12)} ${fmt(rate * 100, 0)}%`);
  }

  lines.push("\nMean populations (post burn-in) and cap utilization:");
  for (const g of GUILDS) {
    const meanPop = mean(runs.map((r) => r.meanCounts[g]));
    const cap = runs[0] && runs[0].caps ? runs[0].caps[g] : null;
    const util = cap ? `  (${fmt((meanPop / cap) * 100, 0)}% of cap ${cap}${meanPop / cap > 0.85 ? " <-- ceiling-limited" : ""})` : "";
    lines.push(`  ${g.padEnd(12)} ${fmt(meanPop, 1)}${util}`);
  }

  lines.push("\nPredator-prey coupling:");
  lines.push(`  best lag (predator after herbivore): ${fmt(mean(runs.map((r) => r.lagSamples)), 1)} samples`);
  lines.push(`  correlation at best lag:              ${fmt(mean(runs.map((r) => r.lagCorr)), 2)}`);

  lines.push("\nVariability (CV = std/mean; ~0 means frozen, >1 means boom-bust):");
  lines.push(`  herbivore CV ${fmt(mean(runs.map((r) => r.herbCV)), 2)}`);
  lines.push(`  predator CV  ${fmt(mean(runs.map((r) => r.predCV)), 2)}`);
  lines.push(`  plant CV     ${fmt(mean(runs.map((r) => r.plantCV)), 2)}`);

  lines.push("\nTrophic pyramid (mean stock; expect plant >> herbivore > predator):");
  lines.push(`  plant biomass ${fmt(mean(runs.map((r) => r.pyramid.plant)), 0)}`);
  lines.push(`  herbivores    ${fmt(mean(runs.map((r) => r.pyramid.herb)), 1)}`);
  lines.push(`  predators     ${fmt(mean(runs.map((r) => r.pyramid.pred)), 1)}`);
  lines.push(`  herb/pred ratio ${fmt(mean(runs.map((r) => r.pyramid.herb / Math.max(0.01, r.pyramid.pred))), 1)}`);
  const predHerbE = mean(runs.map((r) => r.predEnergy / Math.max(0.01, r.herbEnergy)));
  lines.push(`  predator/herbivore ENERGY ${fmt(predHerbE * 100, 1)}%  (Lindeman target ~10-20%)`);

  lines.push("\nMatter pools (tail drift; near 0 = at steady state / conserved):");
  lines.push(`  soil nutrient drift ${fmt(mean(runs.map((r) => r.soilDriftPct)) * 100, 1)}%  (level ${fmt(mean(runs.map((r) => r.meanSoil)), 0)})`);
  lines.push(`  detritus drift      ${fmt(mean(runs.map((r) => r.detritusDriftPct)) * 100, 1)}%  (level ${fmt(mean(runs.map((r) => r.meanDetritus)), 0)})`);
  lines.push(`  plant biomass drift ${fmt(mean(runs.map((r) => r.plantDriftPct)) * 100, 1)}%`);
  lines.push(`  total pool drift    ${fmt(mean(runs.map((r) => r.poolDriftPct)) * 100, 1)}%`);

  lines.push("\nHealth signals:");
  lines.push(`  stability    ${fmt(mean(runs.map((r) => r.finalStability)), 1)}/100`);
  lines.push(`  toxicity     ${fmt(mean(runs.map((r) => r.finalToxicity)) * 100, 1)}%`);
  lines.push(`  disease      ${fmt(mean(runs.map((r) => r.finalDisease)) * 100, 1)}%`);
  lines.push(`  biodiversity ${fmt(mean(runs.map((r) => r.finalShannon)), 2)}`);
  lines.push("");
  return lines.join("\n");
}
