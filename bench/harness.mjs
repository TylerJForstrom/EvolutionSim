// Headless harness: stub a minimal DOM, load sim.js, and expose EcosystemLab.
// This lets us run the browser simulator with no rendering so we can measure
// its ecological behavior over long horizons.
import fs from "node:fs";
import path from "node:path";
import vm from "node:vm";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
// SIM_FILE lets the probe target either the live sim.js or a frozen baseline
// copy, so improvements can be measured against the original without races.
const SIM_PATH = process.env.SIM_FILE
  ? path.resolve(process.env.SIM_FILE)
  : path.join(__dirname, "..", "sim.js");

// Control values the sim reads from DOM inputs. Defaults mirror index.html.
const DEFAULT_CONTROLS = {
  speedControl: "4",
  rainfallControl: "58",
  temperatureControl: "0",
  disturbanceControl: "26",
};

function makeFakeCanvasContext() {
  const noop = () => {};
  return new Proxy(
    {
      canvas: { width: 0, height: 0 },
      setTransform: noop,
      clearRect: noop,
      fillRect: noop,
      strokeRect: noop,
      beginPath: noop,
      moveTo: noop,
      lineTo: noop,
      arc: noop,
      closePath: noop,
      fill: noop,
      stroke: noop,
      save: noop,
      restore: noop,
      translate: noop,
      rotate: noop,
      fillText: noop,
      measureText: () => ({ width: 0 }),
    },
    {
      get(target, prop) {
        if (prop in target) return target[prop];
        return () => {};
      },
      set() {
        return true;
      },
    },
  );
}

function makeFakeElement(id, controls) {
  const listeners = {};
  const el = {
    id,
    value: controls[id] ?? "0",
    textContent: "",
    innerHTML: "",
    className: "",
    dataset: {},
    style: {},
    classList: { add() {}, remove() {}, toggle() {}, contains: () => false },
    addEventListener: (type, fn) => {
      (listeners[type] ||= []).push(fn);
    },
    removeEventListener() {},
    getBoundingClientRect: () => ({ width: 640, height: 384, top: 0, left: 0 }),
    getContext: () => makeFakeCanvasContext(),
    width: 640,
    height: 384,
    querySelectorAll: () => [],
  };
  return el;
}

export function loadSim({ controls = {}, seed } = {}) {
  const mergedControls = { ...DEFAULT_CONTROLS, ...controls };
  const elements = new Map();
  const getElement = (id) => {
    if (!elements.has(id)) elements.set(id, makeFakeElement(id, mergedControls));
    return elements.get(id);
  };

  const documentStub = {
    getElementById: (id) => getElement(id),
    querySelectorAll: () => [],
    querySelector: () => null,
    addEventListener: () => {},
    documentElement: { dataset: {} },
    createElement: () => makeFakeElement("created", mergedControls),
  };

  const sandbox = {
    Math,
    Date,
    JSON,
    Intl,
    Float32Array,
    Array,
    Object,
    Number,
    isFinite,
    URLSearchParams,
    console,
    document: documentStub,
    requestAnimationFrame: () => 0, // prevents the animation loop from running
    cancelAnimationFrame: () => {},
  };
  const windowStub = {
    requestAnimationFrame: () => 0,
    cancelAnimationFrame: () => {},
    addEventListener: () => {},
    devicePixelRatio: 1,
    location: { search: seed != null ? `?paused=1` : "" },
  };
  sandbox.window = windowStub;
  sandbox.globalThis = sandbox;
  windowStub.document = documentStub;

  const code = fs.readFileSync(SIM_PATH, "utf8");
  const context = vm.createContext(sandbox);
  vm.runInContext(code, context, { filename: "sim.js" });

  const lab = windowStub.EcosystemLab;
  if (!lab || !lab.headless) {
    throw new Error("EcosystemLab.headless was not exposed by sim.js");
  }
  if (lab.headless.setQuiet) lab.headless.setQuiet(true);
  if (seed != null) lab.headless.reseed(seed);
  return lab.headless;
}

// Run `ticks` steps, sampling stats/audit every `sample` ticks.
export function runSim(sim, { ticks = 4800, sample = 50 } = {}) {
  const series = [];
  for (let t = 0; t < ticks; t += 1) {
    sim.tick();
    if (t % sample === 0) {
      const s = sim.stats();
      const a = sim.energyAudit();
      series.push({
        tick: a.tick,
        counts: { ...s.counts },
        energyByType: {
          herbivore: a.byType.herbivore.energy,
          predator: a.byType.predator.energy,
          decomposer: a.byType.decomposer.energy,
          pollinator: a.byType.pollinator.energy,
          engineer: a.byType.engineer.energy,
        },
        plantBiomass: s.plantBiomass,
        animalEnergy: a.animalEnergy,
        detritus: a.detritus,
        soilNutrients: a.soilNutrients,
        nutrients: s.nutrients,
        moisture: s.moisture,
        toxicity: s.toxicity,
        diseasePct: s.diseasePct,
        stability: s.stability,
        shannon: s.shannon,
        avgHunger: s.avgHunger,
        avgThirst: s.avgThirst,
      });
    }
  }
  return series;
}
