(() => {
  "use strict";

  const GRID_W = 160;
  const GRID_H = 96;
  const CELL_COUNT = GRID_W * GRID_H;
  const YEAR_TICKS = 2400;
  const HISTORY_LIMIT = 260;
  const MAX_AGENTS = 760;
  const DIRECTIONS = 16;

  const TYPE_ORDER = ["herbivore", "predator", "decomposer", "pollinator", "engineer"];
  const TYPE_INFO = {
    herbivore: {
      label: "Herbivores",
      color: "#b7862c",
      baseEnergy: 72,
      maxAge: 3200,
      minAge: 120,
      reproEnergy: 108,
      reproCost: 38,
      reproChance: 0.07,
      maxCount: 230,
      speed: 0.19,
      metabolism: 0.045,
      foodEnergy: 52,
      eatRate: 0.05,
    },
    predator: {
      label: "Predators",
      color: "#bd4f39",
      baseEnergy: 86,
      maxAge: 2600,
      minAge: 170,
      reproEnergy: 175,
      reproCost: 62,
      reproChance: 0.009,
      maxCount: 28,
      speed: 0.22,
      metabolism: 0.085,
      foodEnergy: 1,
      eatRate: 0,
    },
    decomposer: {
      label: "Decomposers",
      color: "#76569a",
      baseEnergy: 58,
      maxAge: 1250,
      minAge: 70,
      reproEnergy: 108,
      reproCost: 42,
      reproChance: 0.028,
      maxCount: 92,
      speed: 0.14,
      metabolism: 0.025,
      foodEnergy: 64,
      eatRate: 0.058,
    },
    pollinator: {
      label: "Pollinators",
      color: "#287a9b",
      baseEnergy: 45,
      maxAge: 1450,
      minAge: 55,
      reproEnergy: 78,
      reproCost: 28,
      reproChance: 0.07,
      maxCount: 140,
      speed: 0.27,
      metabolism: 0.026,
      foodEnergy: 80,
      eatRate: 0.034,
    },
    engineer: {
      label: "Engineers",
      color: "#2d8f86",
      baseEnergy: 88,
      maxAge: 3200,
      minAge: 190,
      reproEnergy: 158,
      reproCost: 68,
      reproChance: 0.022,
      maxCount: 38,
      speed: 0.14,
      metabolism: 0.055,
      foodEnergy: 44,
      eatRate: 0.038,
    },
  };

  const els = {
    world: document.getElementById("worldCanvas"),
    history: document.getElementById("historyCanvas"),
    clock: document.getElementById("clockReadout"),
    system: document.getElementById("systemReadout"),
    runToggle: document.getElementById("runToggle"),
    stepOnce: document.getElementById("stepOnce"),
    resetWorld: document.getElementById("resetWorld"),
    speed: document.getElementById("speedControl"),
    rainfall: document.getElementById("rainfallControl"),
    temperature: document.getElementById("temperatureControl"),
    disturbance: document.getElementById("disturbanceControl"),
    metricGrid: document.getElementById("metricGrid"),
    signalList: document.getElementById("signalList"),
    traitRows: document.getElementById("traitRows"),
    eventLog: document.getElementById("eventLog"),
  };

  const ctx = els.world.getContext("2d");
  const hctx = els.history.getContext("2d");
  const moistureBuffer = new Float32Array(CELL_COUNT);

  let cells = [];
  let agents = [];
  let nextAgentId = 1;
  let lastRenderStats = null;

  const state = {
    seed: 49321,
    rng: mulberry32(49321),
    running: true,
    speed: 4,
    overlay: "biome",
    tick: 0,
    droughtTicks: 0,
    floodTicks: 0,
    firePatches: [],
    events: [],
    history: [],
    totals: {
      births: 0,
      deaths: 0,
      predation: 0,
      primaryProduction: 0,
      recycled: 0,
      diseaseCases: 0,
    },
  };

  function mulberry32(seed) {
    return function random() {
      let t = (seed += 0x6d2b79f5);
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function rand() {
    return state.rng();
  }

  function randRange(min, max) {
    return min + (max - min) * rand();
  }

  function clamp(value, min, max) {
    return Math.max(min, Math.min(max, value));
  }

  function lerp(a, b, t) {
    return a + (b - a) * t;
  }

  function smoothstep(t) {
    return t * t * (3 - 2 * t);
  }

  function distance(ax, ay, bx, by) {
    return Math.hypot(ax - bx, ay - by);
  }

  function hashNoise(ix, iy, seed) {
    const value = Math.sin(ix * 127.1 + iy * 311.7 + seed * 74.7) * 43758.5453123;
    return value - Math.floor(value);
  }

  function smoothNoise(x, y, scale, seed) {
    const sx = x / scale;
    const sy = y / scale;
    const x0 = Math.floor(sx);
    const y0 = Math.floor(sy);
    const tx = smoothstep(sx - x0);
    const ty = smoothstep(sy - y0);
    const a = hashNoise(x0, y0, seed);
    const b = hashNoise(x0 + 1, y0, seed);
    const c = hashNoise(x0, y0 + 1, seed);
    const d = hashNoise(x0 + 1, y0 + 1, seed);
    return lerp(lerp(a, b, tx), lerp(c, d, tx), ty);
  }

  function fractalNoise(x, y, seed) {
    return (
      smoothNoise(x, y, 32, seed) * 0.52 +
      smoothNoise(x, y, 14, seed + 13) * 0.31 +
      smoothNoise(x, y, 6, seed + 29) * 0.17
    );
  }

  function cellIndex(x, y) {
    const cx = clamp(Math.floor(x), 0, GRID_W - 1);
    const cy = clamp(Math.floor(y), 0, GRID_H - 1);
    return cy * GRID_W + cx;
  }

  function getCell(x, y) {
    return cells[cellIndex(x, y)];
  }

  function neighborIndex(x, y, dx, dy) {
    const nx = clamp(x + dx, 0, GRID_W - 1);
    const ny = clamp(y + dy, 0, GRID_H - 1);
    return ny * GRID_W + nx;
  }

  function seasonState() {
    const cycle = (state.tick % YEAR_TICKS) / YEAR_TICKS;
    const year = Math.floor(state.tick / YEAR_TICKS) + 1;
    const day = Math.floor(cycle * 360) + 1;
    if (cycle < 0.25) {
      return {
        name: "Spring",
        year,
        day,
        sun: 0.76 + cycle * 0.56,
        rain: 1.15,
        temp: -1 + cycle * 14,
        flower: 1.25,
      };
    }
    if (cycle < 0.5) {
      return {
        name: "Summer",
        year,
        day,
        sun: 0.98,
        rain: 0.78,
        temp: 9,
        flower: 1,
      };
    }
    if (cycle < 0.75) {
      return {
        name: "Autumn",
        year,
        day,
        sun: 0.84 - (cycle - 0.5) * 0.7,
        rain: 0.92,
        temp: 5 - (cycle - 0.5) * 18,
        flower: 0.42,
      };
    }
    return {
      name: "Winter",
      year,
      day,
      sun: 0.44,
      rain: 0.72,
      temp: -8,
      flower: 0.08,
    };
  }

  function bell(value, preferred, width) {
    const normalized = (value - preferred) / width;
    return Math.exp(-normalized * normalized);
  }

  function logEvent(title, detail) {
    state.events.unshift({
      title,
      detail,
      tick: state.tick,
    });
    if (state.events.length > 9) state.events.length = 9;
  }

  function initWorld(seed = Date.now() % 100000) {
    state.seed = seed;
    state.rng = mulberry32(seed);
    state.tick = 0;
    state.droughtTicks = 0;
    state.floodTicks = 0;
    state.firePatches = [];
    state.events = [];
    state.history = [];
    state.totals = {
      births: 0,
      deaths: 0,
      predation: 0,
      primaryProduction: 0,
      recycled: 0,
      diseaseCases: 0,
    };
    cells = new Array(CELL_COUNT);
    nextAgentId = 1;

    for (let y = 0; y < GRID_H; y += 1) {
      const riverCenter =
        GRID_W * (0.5 + 0.22 * Math.sin(y * 0.075 + seed * 0.0007)) +
        (fractalNoise(0, y, seed + 88) - 0.5) * 26;
      for (let x = 0; x < GRID_W; x += 1) {
        const nx = x / (GRID_W - 1) - 0.5;
        const ny = y / (GRID_H - 1) - 0.5;
        const radial = Math.sqrt(nx * nx + ny * ny);
        const terrain = fractalNoise(x, y, seed);
        const broadHighland = fractalNoise(x * 0.45, y * 0.45, seed + 17);
        const ridgeNoise = fractalNoise(x * 1.7, y * 1.7, seed + 7);
        const ridges = Math.abs(ridgeNoise - 0.5) * 2;
        const valleyCut = Math.exp(-((x - riverCenter) ** 2) / 42);
        const elevation = clamp(
          0.5 +
            (terrain - 0.5) * 0.78 +
            (broadHighland - 0.5) * 0.58 +
            ridges * 0.28 -
            radial * 0.22 -
            valleyCut * 0.22,
          0,
          1,
        );
        const river = Math.exp(-((x - riverCenter) ** 2) / 30);
        const roughness = clamp(ridges * 0.62 + Math.abs(terrain - broadHighland) * 0.55 + elevation * 0.18, 0, 1);
        const moistureNoise = fractalNoise(x, y, seed + 41);
        const baseMoisture = clamp(0.36 + (moistureNoise - 0.5) * 0.55 + river * 0.64 + valleyCut * 0.12 - elevation * 0.3, 0, 1.2);
        const nutrientNoise = fractalNoise(x, y, seed + 99);
        const nutrients = clamp(0.32 + (nutrientNoise - 0.5) * 0.4 + baseMoisture * 0.2 + valleyCut * 0.08 - elevation * 0.1, 0.05, 1);
        const baseTemp =
          29 -
          (y / (GRID_H - 1)) * 18 -
          elevation * 8 +
          (fractalNoise(x, y, seed + 151) - 0.5) * 4;
        const water = baseMoisture > 0.76 || river > 0.64 || elevation < 0.12;
        const vegetation = water
          ? clamp(0.05 + nutrients * 0.26 + rand() * 0.07, 0, 0.6)
          : clamp(0.08 + baseMoisture * 0.48 + nutrients * 0.28 - elevation * 0.11 + rand() * 0.08, 0.02, 0.95);
        const shelter = clamp(fractalNoise(x, y, seed + 301) * 0.8 + vegetation * 0.25, 0, 1);

        cells[y * GRID_W + x] = {
          x,
          y,
          elevation,
          roughness,
          river,
          moisture: baseMoisture,
          baseTemp,
          temp: baseTemp,
          nutrients,
          vegetation,
          flower: vegetation * 0.22,
          detritus: 0.06 + rand() * 0.06,
          toxicity: 0,
          pathogen: 0,
          oxygen: water ? 0.72 : 1,
          water,
          shelter,
          pollinated: 0,
          decomposerBoost: 0,
          engineerBoost: 0,
          burning: 0,
        };
      }
    }

    initAgents();
    logEvent("World seeded", `Seed ${seed} with ${agents.length} organisms`);
    pushHistory();
    updateUi();
  }

  function initAgents() {
    agents = [];
    spawnGroup("herbivore", 170);
    spawnGroup("predator", 18);
    spawnGroup("decomposer", 58);
    spawnGroup("pollinator", 72);
    spawnGroup("engineer", 16);
  }

  function spawnGroup(type, count) {
    for (let i = 0; i < count; i += 1) {
      const cell = randomSpawnCell(type);
      agents.push(createAgent(type, cell.x + rand(), cell.y + rand(), null, null));
    }
  }

  function randomSpawnCell(type) {
    let best = cells[Math.floor(rand() * cells.length)];
    let bestScore = -Infinity;
    for (let tries = 0; tries < 80; tries += 1) {
      const c = cells[Math.floor(rand() * cells.length)];
      let score = c.vegetation + c.moisture * 0.35 + c.nutrients * 0.25 - c.toxicity;
      if (type === "predator") score += c.shelter * 0.35;
      if (type === "decomposer") score = c.detritus + c.moisture * 0.5 + c.nutrients * 0.2;
      if (type === "pollinator") score = c.flower * 2 + c.vegetation * 0.25;
      if (type === "engineer") score = c.moisture + c.vegetation * 0.45 - c.elevation * 0.15;
      if (c.water && type !== "decomposer") score -= 0.8;
      if (score > bestScore) {
        best = c;
        bestScore = score;
      }
    }
    return best;
  }

  function createGenes(type) {
    const tempCenter = type === "predator" ? 21 : type === "pollinator" ? 24 : 20;
    return {
      speed: randRange(0.75, 1.25),
      sense: randRange(4.8, 10.5),
      metabolism: randRange(0.82, 1.22),
      tempPref: randRange(tempCenter - 6, tempCenter + 6),
      heatTolerance: randRange(0.45, 1.15),
      waterPref: randRange(0.38, 0.78),
      resistance: randRange(0.18, 0.82),
      aggression: randRange(0.2, 0.9),
      body: randRange(0.75, 1.35),
    };
  }

  function createBrain(type) {
    const defaults = {
      food: type === "predator" ? 1.3 : 1.05,
      water: 0.72,
      danger: type === "predator" ? 0.16 : 1.16,
      mate: 0.32,
      comfort: 0.5,
      crowd: type === "predator" ? -0.1 : -0.32,
      wander: 0.34,
    };
    return Object.fromEntries(
      Object.entries(defaults).map(([key, value]) => [key, value * randRange(0.72, 1.28)]),
    );
  }

  function createAgent(type, x, y, parentA, parentB) {
    const info = TYPE_INFO[type];
    let genes = createGenes(type);
    let brain = createBrain(type);
    let generation = 1;
    if (parentA) {
      const mate = parentB || parentA;
      genes = mixGenes(parentA.genes, mate.genes);
      brain = mixBrain(parentA.brain, mate.brain);
      generation = Math.max(parentA.generation, mate.generation) + 1;
    }
    return {
      id: nextAgentId++,
      type,
      x: clamp(x, 0.1, GRID_W - 0.2),
      y: clamp(y, 0.1, GRID_H - 0.2),
      vx: randRange(-0.05, 0.05),
      vy: randRange(-0.05, 0.05),
      energy: info.baseEnergy * randRange(0.82, 1.12),
      hunger: randRange(12, 34),
      thirst: randRange(10, 32),
      age: 0,
      generation,
      genes,
      brain,
      infected: false,
      cooldown: Math.floor(randRange(0, 80)),
      lastSlope: 0,
      lastTerrainFactor: 1,
      lastUphillEffort: 0,
      dead: false,
    };
  }

  function mixGenes(a, b) {
    const bounds = {
      speed: [0.45, 1.75],
      sense: [2.5, 16],
      metabolism: [0.55, 1.75],
      tempPref: [-2, 38],
      heatTolerance: [0.15, 1.8],
      waterPref: [0.16, 1],
      resistance: [0.05, 1.25],
      aggression: [0.05, 1.25],
      body: [0.45, 1.9],
    };
    const result = {};
    Object.keys(a).forEach((key) => {
      const base = (a[key] + b[key]) * 0.5;
      const mutation = (rand() - 0.5) * 0.16;
      const [min, max] = bounds[key];
      result[key] = clamp(base * (1 + mutation), min, max);
    });
    return result;
  }

  function mixBrain(a, b) {
    const bounds = {
      food: [0.05, 2.6],
      water: [0.02, 2.1],
      danger: [0.02, 2.6],
      mate: [0, 1.4],
      comfort: [0, 1.8],
      crowd: [-1.3, 0.8],
      wander: [0.05, 1.2],
    };
    const result = {};
    Object.keys(a).forEach((key) => {
      const base = (a[key] + b[key]) * 0.5;
      const mutation = (rand() - 0.5) * 0.2;
      const [min, max] = bounds[key];
      result[key] = clamp(base * (1 + mutation), min, max);
    });
    return result;
  }

  function groupAgents() {
    const groups = {
      herbivore: [],
      predator: [],
      decomposer: [],
      pollinator: [],
      engineer: [],
    };
    agents.forEach((agent) => {
      if (!agent.dead) groups[agent.type].push(agent);
    });
    return groups;
  }

  function simulateTick() {
    state.tick += 1;
    maybeTriggerAutomaticEvent();
    updateEnvironment();

    const groups = groupAgents();
    const newborns = [];
    for (const agent of agents) {
      if (!agent.dead) updateAgent(agent, groups, newborns);
    }

    agents = agents.filter((agent) => {
      if (agent.dead || agent.energy <= 0 || agent.age > TYPE_INFO[agent.type].maxAge * agent.genes.body) {
        recycleBody(agent, agent.dead ? "loss" : "old age");
        return false;
      }
      return true;
    });

    if (agents.length + newborns.length > MAX_AGENTS) {
      newborns.length = Math.max(0, MAX_AGENTS - agents.length);
    }
    if (newborns.length) agents.push(...newborns);
    supportMigration();

    if (state.tick % 8 === 0) pushHistory();
    if (state.tick % 6 === 0) updateUi();
  }

  function supportMigration() {
    if (state.tick % 160 !== 0 || agents.length >= MAX_AGENTS - 12) return;
    const stats = collectStats();
    if (stats.counts.herbivore < 24 && stats.plantBiomass > 650) {
      migrate("herbivore", Math.min(18, 42 - stats.counts.herbivore), "Herbivore migration");
    }
    if (stats.counts.pollinator < 16 && stats.plantBiomass > 500) {
      migrate("pollinator", Math.min(16, 34 - stats.counts.pollinator), "Pollinator migration");
    }
    if (stats.counts.predator < 3 && stats.counts.herbivore > 90) {
      migrate("predator", 2, "Predator recolonization");
    }
    if (stats.counts.decomposer < 14 && stats.detritus > 0.18) {
      migrate("decomposer", 10, "Decomposer bloom");
    }
    if (stats.counts.engineer < 4 && stats.moisture < 0.5) {
      migrate("engineer", 2, "Engineer recolonization");
    }
  }

  function migrate(type, count, title) {
    const info = TYPE_INFO[type];
    const current = agents.filter((agent) => !agent.dead && agent.type === type).length;
    const allowed = Math.max(0, Math.min(count, info.maxCount - current, MAX_AGENTS - agents.length));
    if (!allowed) return;
    for (let i = 0; i < allowed; i += 1) {
      const cell = randomSpawnCell(type);
      const edge = rand();
      let x = cell.x + rand();
      let y = cell.y + rand();
      if (edge < 0.25) x = randRange(0.2, 2.2);
      else if (edge < 0.5) x = randRange(GRID_W - 2.2, GRID_W - 0.2);
      else if (edge < 0.75) y = randRange(0.2, 2.2);
      else y = randRange(GRID_H - 2.2, GRID_H - 0.2);
      const migrant = createAgent(type, x, y, null, null);
      migrant.energy = info.baseEnergy * randRange(0.95, 1.25);
      agents.push(migrant);
    }
    logEvent(title, `${allowed} organisms entered through habitat corridors`);
  }

  function updateEnvironment() {
    const season = seasonState();
    const rainfall = Number(els.rainfall.value) / 100;
    const tempOffset = Number(els.temperature.value);
    const droughtFactor = state.droughtTicks > 0 ? 0.22 : 1;
    const floodBonus = state.floodTicks > 0 ? 0.009 : 0;
    const droughtHeat = state.droughtTicks > 0 ? 5.5 : 0;

    if (state.droughtTicks > 0) state.droughtTicks -= 1;
    if (state.floodTicks > 0) state.floodTicks -= 1;

    for (let i = 0; i < cells.length; i += 1) {
      const c = cells[i];
      c.temp = c.baseTemp + season.temp + tempOffset + droughtHeat - c.elevation * 1.5;
      const plantWaterUse = c.vegetation * 0.0019;
      const evaporation = (0.0022 + Math.max(0, c.temp) * 0.00006) * (1.1 - rainfall * 0.25);
      const rain = rainfall * 0.0075 * season.rain * droughtFactor * (1 - c.elevation * 0.28);
      c.moisture = clamp(c.moisture + rain + floodBonus - evaporation - plantWaterUse + c.engineerBoost * 0.002, 0, 1.35);
      c.pathogen *= 0.985;
      c.decomposerBoost *= 0.87;
      c.engineerBoost *= 0.94;
      c.burning *= 0.86;
    }

    for (let y = 0; y < GRID_H; y += 1) {
      for (let x = 0; x < GRID_W; x += 1) {
        const i = y * GRID_W + x;
        const c = cells[i];
        const n1 = cells[neighborIndex(x, y, 1, 0)];
        const n2 = cells[neighborIndex(x, y, -1, 0)];
        const n3 = cells[neighborIndex(x, y, 0, 1)];
        const n4 = cells[neighborIndex(x, y, 0, -1)];
        const avgMoisture = (n1.moisture + n2.moisture + n3.moisture + n4.moisture) * 0.25;
        const avgElevation = (n1.elevation + n2.elevation + n3.elevation + n4.elevation) * 0.25;
        const downhill = clamp((avgElevation - c.elevation) * 0.025, -0.018, 0.018);
        moistureBuffer[i] = clamp(lerp(c.moisture, avgMoisture, 0.025) + downhill, 0, 1.35);
      }
    }

    for (let i = 0; i < cells.length; i += 1) {
      const c = cells[i];
      c.moisture = moistureBuffer[i];
      c.water = c.moisture > 0.82 || c.river > 0.62 || c.elevation < 0.12;
      const tooDry = clamp((0.18 - c.moisture) * 2.5, 0, 1);
      const tooWet = c.water ? 0 : clamp((c.moisture - 0.95) * 1.5, 0, 1);
      const waterFactor = c.water ? clamp(c.moisture, 0.4, 1.1) : clamp(c.moisture / 0.52, 0, 1.25) * (1 - tooWet * 0.5);
      const tempFactor = bell(c.temp, 22, 15);
      const nutrientFactor = clamp(c.nutrients / 0.42, 0, 1.4);
      const capacity = c.water ? 0.46 + c.nutrients * 0.45 : 0.72 + c.shelter * 0.22 + c.nutrients * 0.34;
      const competition = clamp(1 - c.vegetation / capacity, -0.18, 1);
      const pollination = 1 + clamp(c.pollinated, 0, 0.75);
      const light = clamp(season.sun - c.elevation * 0.08, 0.2, 1.05);
      const growth = Math.max(
        0,
        c.vegetation *
          0.018 *
          competition *
          light *
          waterFactor *
          tempFactor *
          nutrientFactor *
          (1 - c.toxicity * 0.85) *
          pollination,
      );
      const seedling =
        c.vegetation < 0.035 && c.nutrients > 0.16 && c.moisture > 0.16
          ? 0.0012 * light * nutrientFactor * (1 + c.pollinated)
          : 0;
      const stressDeath = c.vegetation * (tooDry * 0.005 + tooWet * 0.003 + (1 - tempFactor) * 0.002);
      c.vegetation = clamp(c.vegetation + growth + seedling - stressDeath, 0, capacity);
      c.detritus = clamp(c.detritus + stressDeath * 0.82, 0, 2.8);
      c.nutrients = clamp(c.nutrients - growth * 0.08 + stressDeath * 0.05, 0.02, 1.4);
      c.flower = clamp(c.vegetation * 0.3 * season.flower * tempFactor * (1 - c.toxicity), 0, 0.55);
      c.pollinated *= 0.9;
      state.totals.primaryProduction += growth + seedling;

      const microbe = clamp(c.moisture / 0.58, 0, 1.3) * bell(c.temp, 21, 14) * (1 - c.toxicity * 0.65);
      const decomposed = Math.min(c.detritus, c.detritus * (0.006 + microbe * 0.018 + c.decomposerBoost * 0.03));
      c.detritus -= decomposed;
      c.nutrients = clamp(c.nutrients + decomposed * 0.58, 0.02, 1.4);
      state.totals.recycled += decomposed * 0.58;

      const wastePressure = Math.max(0, c.detritus - 1.15) + c.pathogen * 0.3;
      c.toxicity = clamp(c.toxicity + wastePressure * 0.0008 - (microbe + c.moisture) * 0.0016, 0, 1);
      c.oxygen = c.water
        ? clamp(0.85 - c.detritus * 0.16 - Math.max(0, c.temp - 22) * 0.012 + c.vegetation * 0.12, 0, 1)
        : 1;
    }

    updateFire();
  }

  function updateFire() {
    if (!state.firePatches.length) return;
    const survivors = [];
    for (const fire of state.firePatches) {
      fire.radius += 0.022;
      fire.ttl -= 1;
      const minX = clamp(Math.floor(fire.x - fire.radius - 2), 0, GRID_W - 1);
      const maxX = clamp(Math.ceil(fire.x + fire.radius + 2), 0, GRID_W - 1);
      const minY = clamp(Math.floor(fire.y - fire.radius - 2), 0, GRID_H - 1);
      const maxY = clamp(Math.ceil(fire.y + fire.radius + 2), 0, GRID_H - 1);
      for (let y = minY; y <= maxY; y += 1) {
        for (let x = minX; x <= maxX; x += 1) {
          const d = distance(x + 0.5, y + 0.5, fire.x, fire.y);
          if (d > fire.radius + 2) continue;
          const c = cells[y * GRID_W + x];
          const burn = clamp((fire.radius + 2 - d) / 3.2, 0, 1) * (0.2 + c.vegetation);
          if (c.moisture > 0.74) continue;
          const plantLoss = Math.min(c.vegetation, burn * 0.018);
          c.vegetation -= plantLoss;
          c.detritus = clamp(c.detritus + plantLoss * 0.46, 0, 2.8);
          c.nutrients = clamp(c.nutrients + plantLoss * 0.15, 0, 1.4);
          c.moisture = clamp(c.moisture - burn * 0.005, 0, 1.35);
          c.toxicity = clamp(c.toxicity + burn * 0.002, 0, 1);
          c.burning = Math.max(c.burning, burn);
        }
      }
      if (fire.ttl > 0) survivors.push(fire);
    }
    state.firePatches = survivors;

    for (const agent of agents) {
      if (agent.dead) continue;
      const c = getCell(agent.x, agent.y);
      if (c.burning > 0.3) {
        agent.energy -= c.burning * 1.8;
        if (rand() < c.burning * 0.018) agent.dead = true;
      }
    }
  }

  function maybeTriggerAutomaticEvent() {
    const disturbance = Number(els.disturbance.value) / 100;
    if (disturbance <= 0) return;
    if (rand() > disturbance * 0.00035) return;
    const season = seasonState();
    const roll = rand();
    if (season.name === "Summer" && roll < 0.34) triggerEvent("fire", true);
    else if (roll < 0.28) triggerEvent("disease", true);
    else if (roll < 0.55) triggerEvent("drought", true);
    else triggerEvent("flood", true);
  }

  function updateAgent(agent, groups, newborns) {
    const info = TYPE_INFO[agent.type];
    const cell = getCell(agent.x, agent.y);
    agent.age += 1;
    if (agent.cooldown > 0) agent.cooldown -= 1;

    const tempStress = Math.max(0, 1 - bell(cell.temp, agent.genes.tempPref, 11 + agent.genes.heatTolerance * 8));
    const thirstStress = Math.max(0, agent.genes.waterPref - cell.moisture) * 0.03;
    const diseaseCost = agent.infected ? 0.034 * (1.15 - agent.genes.resistance) : 0;
    const toxinCost = cell.toxicity * 0.062;
    const oxygenCost = cell.water && cell.oxygen < 0.3 ? 0.04 : 0;
    agent.energy -=
      info.metabolism * agent.genes.metabolism +
      tempStress * 0.055 +
      thirstStress +
      diseaseCost +
      toxinCost +
      oxygenCost;

    const move = chooseMovement(agent, groups);
    agent.vx = lerp(agent.vx, move.x, 0.55);
    agent.vy = lerp(agent.vy, move.y, 0.55);
    const baseSpeed = info.speed * agent.genes.speed * (agent.infected ? 0.72 : 1);
    const intendedX = clamp(agent.x + agent.vx * baseSpeed, 0.05, GRID_W - 0.05);
    const intendedY = clamp(agent.y + agent.vy * baseSpeed, 0.05, GRID_H - 0.05);
    const terrain = terrainMovement(cell, getCell(intendedX, intendedY), baseSpeed, agent);
    agent.x = clamp(agent.x + agent.vx * terrain.speed, 0.05, GRID_W - 0.05);
    agent.y = clamp(agent.y + agent.vy * terrain.speed, 0.05, GRID_H - 0.05);
    agent.lastSlope = terrain.slope;
    agent.lastTerrainFactor = terrain.factor;
    agent.lastUphillEffort = terrain.uphillEffort;
    agent.energy -= terrain.effort * 0.01 * agent.genes.body;
    updateNeeds(agent, getCell(agent.x, agent.y), terrain.effort, tempStress);

    interactWithCell(agent, groups);
    updateDisease(agent, groups);
    maybeReproduce(agent, groups, newborns);
  }

  function terrainMovement(fromCell, toCell, baseSpeed, agent) {
    const slope = toCell.elevation - fromCell.elevation;
    const uphill = Math.max(0, slope);
    const downhill = Math.max(0, -slope);
    const roughness = (fromCell.roughness + toCell.roughness) * 0.5;
    const waterDrag = toCell.water && agent.type !== "decomposer" ? 0.22 : 0;
    const uphillPenalty = uphill * (3.4 + agent.genes.body * 0.9);
    const roughPenalty = roughness * 0.22;
    const downhillBoost = clamp(downhill * 0.75, 0, 0.16);
    const factor = clamp(1 - uphillPenalty - roughPenalty - waterDrag + downhillBoost, 0.22, 1.16);
    const effort =
      baseSpeed *
      (1 + uphill * (8.5 + agent.genes.body * 2.2) + roughness * 0.8 + waterDrag * 1.1 + Math.max(0, downhill - 0.08) * 0.8);
    return {
      slope,
      factor,
      speed: baseSpeed * factor,
      effort,
      uphillEffort: uphill * effort,
    };
  }

  function updateNeeds(agent, cell, speed, tempStress) {
    const info = TYPE_INFO[agent.type];
    const hotStress = clamp((cell.temp - agent.genes.tempPref) / 18, 0, 1.6);
    const dryStress = Math.max(0, agent.genes.waterPref - cell.moisture);
    agent.hunger = clamp(
      agent.hunger +
        info.metabolism * agent.genes.metabolism * 0.35 +
        speed * 0.16 +
        agent.genes.body * 0.012 +
        (agent.infected ? 0.045 : 0),
      0,
      140,
    );
    agent.thirst = clamp(
      agent.thirst + 0.022 + speed * 0.2 + hotStress * 0.1 + dryStress * 0.13 + tempStress * 0.04,
      0,
      150,
    );

    if (cell.water || cell.moisture > 0.58) {
      const drink = cell.water ? 3.6 : cell.moisture * 1.35;
      agent.thirst = clamp(agent.thirst - drink, 0, 150);
      if (!cell.water) cell.moisture = clamp(cell.moisture - 0.0008 * agent.genes.body, 0, 1.35);
      if (cell.water && cell.oxygen < 0.22) agent.energy -= 0.035;
    }

    const starvationCost = Math.max(0, agent.hunger - 88) * 0.007;
    const dehydrationCost = Math.max(0, agent.thirst - 82) * 0.013;
    agent.energy -= starvationCost + dehydrationCost;
  }

  function chooseMovement(agent, groups) {
    const food = senseResource(agent, foodKind(agent.type));
    const water = senseResource(agent, "water");
    const comfort = senseResource(agent, "comfort");
    const mate = agent.energy > TYPE_INFO[agent.type].reproEnergy * 0.78 ? nearestSame(agent, groups[agent.type], agent.genes.sense) : null;
    const danger = dangerFor(agent, groups);
    const crowd = crowdVector(agent, groups[agent.type], 3.3);
    const b = agent.brain;
    const hungerDrive = 0.55 + agent.hunger / 58;
    const thirstDrive = 0.55 + agent.thirst / 46;
    let dx =
      food.x * b.food * hungerDrive +
      water.x * b.water * thirstDrive +
      comfort.x * b.comfort +
      crowd.x * b.crowd +
      (mate ? mate.x * b.mate : 0) -
      (danger ? danger.x * b.danger : 0);
    let dy =
      food.y * b.food * hungerDrive +
      water.y * b.water * thirstDrive +
      comfort.y * b.comfort +
      crowd.y * b.crowd +
      (mate ? mate.y * b.mate : 0) -
      (danger ? danger.y * b.danger : 0);

    dx += (rand() - 0.5) * b.wander;
    dy += (rand() - 0.5) * b.wander;

    const len = Math.hypot(dx, dy);
    if (len < 0.0001) {
      return {
        x: randRange(-1, 1),
        y: randRange(-1, 1),
      };
    }
    return {
      x: dx / len,
      y: dy / len,
    };
  }

  function foodKind(type) {
    if (type === "predator") return "prey";
    if (type === "decomposer") return "detritus";
    if (type === "pollinator") return "flower";
    return "plant";
  }

  function senseResource(agent, kind) {
    if (kind === "prey") {
      const prey = nearestAgent(agent, agents.filter((a) => !a.dead && (a.type === "herbivore" || a.type === "pollinator")), agent.genes.sense);
      return prey || { x: 0, y: 0, score: 0 };
    }

    let bestScore = scoreCell(getCell(agent.x, agent.y), kind, agent);
    let best = { x: 0, y: 0, score: bestScore };
    const sense = agent.genes.sense;
    for (let i = 0; i < DIRECTIONS; i += 1) {
      const angle = (Math.PI * 2 * i) / DIRECTIONS;
      const dx = Math.cos(angle);
      const dy = Math.sin(angle);
      for (let r = 0.35; r <= 1; r += 0.325) {
        const c = getCell(agent.x + dx * sense * r, agent.y + dy * sense * r);
        const score = scoreCell(c, kind, agent) - r * 0.03;
        if (score > bestScore) {
          bestScore = score;
          best = {
            x: dx,
            y: dy,
            score,
          };
        }
      }
    }
    return best;
  }

  function scoreCell(cell, kind, agent) {
    if (!cell) return -1;
    if (kind === "plant") {
      return cell.vegetation * 1.45 + cell.moisture * 0.18 + cell.nutrients * 0.18 - cell.toxicity * 1.1 - (cell.water ? 0.45 : 0);
    }
    if (kind === "detritus") {
      return cell.detritus * 1.3 + cell.moisture * 0.28 - cell.toxicity * 0.28;
    }
    if (kind === "flower") {
      return cell.flower * 2.2 + cell.vegetation * 0.08 - cell.toxicity * 0.65;
    }
    if (kind === "water") {
      return cell.moisture * 1.2 + (cell.water ? 0.35 : 0) - Math.max(0, 0.3 - cell.oxygen) * 0.4;
    }
    if (kind === "comfort") {
      return bell(cell.temp, agent.genes.tempPref, 12 + agent.genes.heatTolerance * 8) + cell.shelter * 0.15 - cell.toxicity;
    }
    return 0;
  }

  function nearestAgent(agent, candidates, radius) {
    let best = null;
    let bestD = radius;
    for (const other of candidates) {
      if (other === agent || other.dead) continue;
      const d = distance(agent.x, agent.y, other.x, other.y);
      if (d < bestD) {
        bestD = d;
        const dx = (other.x - agent.x) / Math.max(0.001, d);
        const dy = (other.y - agent.y) / Math.max(0.001, d);
        best = { x: dx, y: dy, dist: d, agent: other, score: 1 - d / radius };
      }
    }
    return best;
  }

  function nearestSame(agent, candidates, radius) {
    let best = null;
    let bestD = radius;
    for (const other of candidates) {
      if (other === agent || other.dead || other.energy < TYPE_INFO[other.type].reproEnergy * 0.65) continue;
      const d = distance(agent.x, agent.y, other.x, other.y);
      if (d < bestD) {
        bestD = d;
        best = {
          x: (other.x - agent.x) / Math.max(0.001, d),
          y: (other.y - agent.y) / Math.max(0.001, d),
          dist: d,
          agent: other,
        };
      }
    }
    return best;
  }

  function dangerFor(agent, groups) {
    if (agent.type === "predator") return fireDanger(agent);
    const predator = nearestAgent(agent, groups.predator, agent.genes.sense * 1.15);
    const fire = fireDanger(agent);
    if (!predator) return fire;
    if (!fire) return predator;
    return predator.score > fire.score ? predator : fire;
  }

  function fireDanger(agent) {
    const c = getCell(agent.x, agent.y);
    if (c.burning <= 0.1 && c.toxicity < 0.35) return null;
    return {
      x: (rand() - 0.5) * 2,
      y: (rand() - 0.5) * 2,
      score: c.burning + c.toxicity,
    };
  }

  function crowdVector(agent, candidates, radius) {
    let dx = 0;
    let dy = 0;
    let count = 0;
    for (const other of candidates) {
      if (other === agent || other.dead) continue;
      const d = distance(agent.x, agent.y, other.x, other.y);
      if (d > 0 && d < radius) {
        dx += (other.x - agent.x) / d;
        dy += (other.y - agent.y) / d;
        count += 1;
      }
    }
    if (!count) return { x: 0, y: 0 };
    return { x: dx / count, y: dy / count };
  }

  function interactWithCell(agent, groups) {
    const cell = getCell(agent.x, agent.y);
    if (agent.type === "herbivore") {
      graze(agent, cell, 1);
    } else if (agent.type === "engineer") {
      graze(agent, cell, 0.85);
      engineerHabitat(agent);
    } else if (agent.type === "decomposer") {
      decompose(agent, cell);
    } else if (agent.type === "pollinator") {
      pollinate(agent, cell);
    } else if (agent.type === "predator") {
      hunt(agent, groups);
    }

    if (rand() < 0.2) {
      cell.detritus = clamp(cell.detritus + 0.002 * agent.genes.body, 0, 2.8);
      cell.nutrients = clamp(cell.nutrients + 0.0009 * agent.genes.body, 0, 1.4);
    }
  }

  function graze(agent, cell, scale) {
    if (cell.water) return;
    const info = TYPE_INFO[agent.type];
    const amount = Math.min(cell.vegetation, info.eatRate * scale * agent.genes.body);
    if (amount <= 0) return;
    cell.vegetation -= amount;
    cell.detritus = clamp(cell.detritus + amount * 0.08, 0, 2.8);
    cell.nutrients = clamp(cell.nutrients + amount * 0.018, 0, 1.4);
    agent.energy = clamp(agent.energy + amount * info.foodEnergy * (1 - cell.toxicity * 0.45), 0, 220);
    agent.hunger = clamp(agent.hunger - amount * 130, 0, 140);
  }

  function hunt(agent, groups) {
    const preyList = groups.herbivore.concat(groups.pollinator);
    const target = nearestAgent(agent, preyList, 1.25 + agent.genes.body * 0.65);
    if (!target) return;
    const prey = target.agent;
    const success = 0.16 + agent.genes.aggression * 0.16 + agent.genes.speed * 0.05 - prey.genes.speed * 0.11;
    if (rand() < clamp(success, 0.03, 0.46)) {
      prey.dead = true;
      agent.energy = clamp(agent.energy + prey.energy * 0.66 + prey.genes.body * 26, 0, 240);
      agent.hunger = clamp(agent.hunger - 62, 0, 140);
      state.totals.predation += 1;
      const c = getCell(prey.x, prey.y);
      c.detritus = clamp(c.detritus + prey.genes.body * 0.18, 0, 2.8);
      if (state.totals.predation % 12 === 0) logEvent("Predation pulse", "Predators converted prey biomass into detritus");
    }
  }

  function decompose(agent, cell) {
    const amount = Math.min(cell.detritus, TYPE_INFO.decomposer.eatRate * agent.genes.body);
    if (amount <= 0) return;
    cell.detritus -= amount;
    cell.nutrients = clamp(cell.nutrients + amount * 0.62, 0, 1.4);
    cell.toxicity = clamp(cell.toxicity - amount * 0.035, 0, 1);
    cell.decomposerBoost = clamp(cell.decomposerBoost + 0.08, 0, 1);
    state.totals.recycled += amount * 0.62;
    agent.energy = clamp(agent.energy + amount * TYPE_INFO.decomposer.foodEnergy, 0, 160);
    agent.hunger = clamp(agent.hunger - amount * 105, 0, 140);
  }

  function pollinate(agent, cell) {
    const nectar = Math.min(cell.flower, TYPE_INFO.pollinator.eatRate * agent.genes.body);
    if (nectar > 0) {
      cell.flower -= nectar;
      agent.energy = clamp(agent.energy + nectar * TYPE_INFO.pollinator.foodEnergy, 0, 130);
      agent.hunger = clamp(agent.hunger - nectar * 110, 0, 140);
      agent.thirst = clamp(agent.thirst - nectar * 28, 0, 150);
    }
    cell.pollinated = clamp(cell.pollinated + 0.12, 0, 1.2);
    const x = Math.floor(agent.x);
    const y = Math.floor(agent.y);
    for (let dy = -1; dy <= 1; dy += 1) {
      for (let dx = -1; dx <= 1; dx += 1) {
        cells[neighborIndex(x, y, dx, dy)].pollinated = clamp(cells[neighborIndex(x, y, dx, dy)].pollinated + 0.025, 0, 1.2);
      }
    }
  }

  function engineerHabitat(agent) {
    const x = Math.floor(agent.x);
    const y = Math.floor(agent.y);
    for (let dy = -2; dy <= 2; dy += 1) {
      for (let dx = -2; dx <= 2; dx += 1) {
        const c = cells[neighborIndex(x, y, dx, dy)];
        const d = Math.hypot(dx, dy);
        if (d > 2.2) continue;
        c.moisture = clamp(c.moisture + 0.0025 * (2.3 - d), 0, 1.35);
        c.shelter = clamp(c.shelter + 0.0009, 0, 1);
        c.engineerBoost = clamp(c.engineerBoost + 0.02, 0, 1);
      }
    }
  }

  function updateDisease(agent, groups) {
    const cell = getCell(agent.x, agent.y);
    if (agent.infected) {
      cell.pathogen = clamp(cell.pathogen + 0.007, 0, 1.5);
      if (rand() < 0.006 + agent.genes.resistance * 0.007) {
        agent.infected = false;
      }
      return;
    }

    let nearby = 0;
    for (const other of groups[agent.type]) {
      if (other === agent || other.dead) continue;
      if (distance(agent.x, agent.y, other.x, other.y) < 2.2) nearby += 1;
    }
    const pressure = cell.pathogen * 0.003 + nearby * 0.00015 + cell.toxicity * 0.0008;
    if (rand() < pressure * (1.12 - agent.genes.resistance)) {
      agent.infected = true;
      state.totals.diseaseCases += 1;
      if (state.totals.diseaseCases % 8 === 0) logEvent("Disease spread", "Crowding and waste raised pathogen pressure");
    }
  }

  function maybeReproduce(agent, groups, newborns) {
    const info = TYPE_INFO[agent.type];
    if (agents.length + newborns.length >= MAX_AGENTS) return;
    if (groups[agent.type].length >= info.maxCount) return;
    if (agent.cooldown > 0 || agent.age < info.minAge || agent.energy < info.reproEnergy) return;
    if (agent.hunger > 58 || agent.thirst > 62) return;
    const mate = nearestSame(agent, groups[agent.type], Math.max(4.5, agent.genes.sense * 0.9));
    const fallbackMateChance = {
      herbivore: 0.28,
      predator: 0.1,
      decomposer: 1,
      pollinator: 0.34,
      engineer: 0.14,
    };
    if (!mate && rand() > fallbackMateChance[agent.type]) return;
    const chance = info.reproChance * (agent.infected ? 0.35 : 1);
    if (rand() > chance) return;

    const parentB = mate ? mate.agent : agent;
    const baby = createAgent(agent.type, agent.x + randRange(-0.6, 0.6), agent.y + randRange(-0.6, 0.6), agent, parentB);
    baby.energy = info.baseEnergy * 0.62;
    agent.energy -= info.reproCost;
    agent.cooldown = Math.floor(randRange(90, 210));
    if (mate) {
      parentB.energy -= info.reproCost * 0.32;
      parentB.cooldown = Math.floor(randRange(70, 180));
    }
    newborns.push(baby);
    state.totals.births += 1;
    if (state.totals.births % 18 === 0) logEvent("New generation", `${TYPE_INFO[agent.type].label} reached generation ${baby.generation}`);
  }

  function recycleBody(agent, reason) {
    if (agent._recycled) return;
    agent._recycled = true;
    state.totals.deaths += 1;
    const c = getCell(agent.x, agent.y);
    c.detritus = clamp(c.detritus + agent.genes.body * 0.42 + Math.max(0, agent.energy) * 0.004, 0, 2.8);
    c.nutrients = clamp(c.nutrients + agent.genes.body * 0.08, 0, 1.4);
    if (agent.infected) c.pathogen = clamp(c.pathogen + 0.18, 0, 1.5);
    if (state.totals.deaths % 24 === 0) logEvent("Mortality", `${state.totals.deaths} deaths recycled through detritus and soil`);
    return reason;
  }

  function triggerEvent(type, automatic = false) {
    if (type === "drought") {
      state.droughtTicks = Math.max(state.droughtTicks, 420);
      logEvent(automatic ? "Dry season shock" : "Drought started", "Rainfall fell and evaporation increased");
    } else if (type === "flood") {
      state.floodTicks = Math.max(state.floodTicks, 210);
      for (let i = 0; i < 280; i += 1) {
        const c = cells[Math.floor(rand() * cells.length)];
        c.moisture = clamp(c.moisture + randRange(0.08, 0.22), 0, 1.35);
        c.detritus = clamp(c.detritus + randRange(0, 0.05), 0, 2.8);
      }
      logEvent(automatic ? "Flood pulse" : "Flood released", "Water spread through low terrain and moved detritus");
    } else if (type === "fire") {
      const starts = automatic ? 1 : 3;
      for (let i = 0; i < starts; i += 1) {
        const c = dryFuelCell();
        state.firePatches.push({ x: c.x + 0.5, y: c.y + 0.5, radius: randRange(1.5, 3.2), ttl: Math.floor(randRange(120, 240)) });
      }
      logEvent(automatic ? "Lightning fire" : "Wildfire started", "Dry producer biomass burned into ash and detritus");
    } else if (type === "disease") {
      const vulnerable = agents.filter((a) => !a.dead && (a.type === "herbivore" || a.type === "pollinator" || a.type === "predator"));
      const cases = Math.max(3, Math.floor(vulnerable.length * (automatic ? 0.055 : 0.12)));
      for (let i = 0; i < cases && vulnerable.length; i += 1) {
        const idx = Math.floor(rand() * vulnerable.length);
        vulnerable[idx].infected = true;
        const c = getCell(vulnerable[idx].x, vulnerable[idx].y);
        c.pathogen = clamp(c.pathogen + 0.28, 0, 1.5);
        vulnerable.splice(idx, 1);
      }
      state.totals.diseaseCases += cases;
      logEvent(automatic ? "Pathogen bloom" : "Disease introduced", `${cases} organisms became infected`);
    }
  }

  function dryFuelCell() {
    let best = cells[0];
    let bestScore = -Infinity;
    for (let i = 0; i < 120; i += 1) {
      const c = cells[Math.floor(rand() * cells.length)];
      const score = c.vegetation * 1.8 - c.moisture * 1.2 + Math.max(0, c.temp - 24) * 0.03;
      if (score > bestScore) {
        best = c;
        bestScore = score;
      }
    }
    return best;
  }

  function collectStats() {
    const counts = Object.fromEntries(TYPE_ORDER.map((type) => [type, 0]));
    let infected = 0;
    let speedSum = 0;
    let senseSum = 0;
    let hungerSum = 0;
    let thirstSum = 0;
    let plantBiomass = 0;
    let treeStands = 0;
    let nutrients = 0;
    let moisture = 0;
    let detritus = 0;
    let toxicity = 0;
    let pathogen = 0;
    let elevation = 0;
    let roughness = 0;
    let oxygen = 0;
    let waterCells = 0;
    let highCells = 0;

    for (const cell of cells) {
      plantBiomass += cell.vegetation;
      if (!cell.water && cell.vegetation > 0.62 && cell.shelter > 0.38) treeStands += 1;
      nutrients += cell.nutrients;
      moisture += cell.moisture;
      detritus += cell.detritus;
      toxicity += cell.toxicity;
      pathogen += cell.pathogen;
      elevation += cell.elevation;
      roughness += cell.roughness;
      if (cell.elevation > 0.68) highCells += 1;
      if (cell.water) {
        oxygen += cell.oxygen;
        waterCells += 1;
      }
    }

    for (const agent of agents) {
      counts[agent.type] += 1;
      if (agent.infected) infected += 1;
      speedSum += agent.genes.speed;
      senseSum += agent.genes.sense;
      hungerSum += agent.hunger;
      thirstSum += agent.thirst;
    }

    let agentElevation = 0;
    let terrainFactor = 0;
    let uphillEffort = 0;
    for (const agent of agents) {
      const c = getCell(agent.x, agent.y);
      agentElevation += c.elevation;
      terrainFactor += agent.lastTerrainFactor || 1;
      uphillEffort += agent.lastUphillEffort || 0;
    }

    const totalAgents = Math.max(1, agents.length);
    const shannon = biodiversity(counts);
    const carrying =
      plantBiomass /
      Math.max(1, counts.herbivore * 0.85 + counts.engineer * 0.65 + counts.pollinator * 0.2);
    const recycling = state.totals.recycled / Math.max(1, state.totals.primaryProduction);
    const avgToxicity = toxicity / CELL_COUNT;
    const diseasePct = infected / totalAgents;
    const avgHunger = hungerSum / totalAgents;
    const avgThirst = thirstSum / totalAgents;
    const stability =
      100 -
      clamp((0.55 - carrying) * 95, 0, 45) -
      clamp(diseasePct * 120, 0, 25) -
      clamp(avgToxicity * 140, 0, 25) -
      clamp((0.25 - moisture / CELL_COUNT) * 60, 0, 20) -
      clamp((avgHunger - 55) * 0.35, 0, 18) -
      clamp((avgThirst - 55) * 0.45, 0, 22);

    return {
      counts,
      totalAgents: agents.length,
      infected,
      diseasePct,
      plantBiomass,
      nutrients: nutrients / CELL_COUNT,
      moisture: moisture / CELL_COUNT,
      detritus: detritus / CELL_COUNT,
      toxicity: avgToxicity,
      pathogen: pathogen / CELL_COUNT,
      elevation: elevation / CELL_COUNT,
      roughness: roughness / CELL_COUNT,
      highlandPct: highCells / CELL_COUNT,
      oxygen: waterCells ? oxygen / waterCells : 1,
      shannon,
      carrying,
      recycling,
      avgSpeed: speedSum / totalAgents,
      avgSense: senseSum / totalAgents,
      avgHunger,
      avgThirst,
      avgAgentElevation: agentElevation / totalAgents,
      avgTerrainFactor: terrainFactor / totalAgents,
      avgUphillEffort: uphillEffort / totalAgents,
      treeStands,
      stability: clamp(stability, 0, 100),
    };
  }

  function biodiversity(counts) {
    const total = Object.values(counts).reduce((sum, count) => sum + count, 0);
    if (!total) return 0;
    let value = 0;
    for (const count of Object.values(counts)) {
      if (!count) continue;
      const p = count / total;
      value -= p * Math.log(p);
    }
    return value / Math.log(TYPE_ORDER.length);
  }

  function pushHistory() {
    const stats = collectStats();
    state.history.push({
      tick: state.tick,
      herbivore: stats.counts.herbivore,
      predator: stats.counts.predator,
      decomposer: stats.counts.decomposer,
      pollinator: stats.counts.pollinator,
      engineer: stats.counts.engineer,
      plant: stats.plantBiomass,
    });
    if (state.history.length > HISTORY_LIMIT) state.history.shift();
    lastRenderStats = stats;
  }

  function formatNumber(value, digits = 0) {
    return new Intl.NumberFormat("en-US", {
      maximumFractionDigits: digits,
      minimumFractionDigits: digits,
    }).format(Number.isFinite(value) ? value : 0);
  }

  function formatPct(value, digits = 1) {
    return `${formatNumber(value * 100, digits)}%`;
  }

  function updateUi() {
    const stats = collectStats();
    lastRenderStats = stats;
    const season = seasonState();
    document.documentElement.dataset.ecosystemStats = JSON.stringify({
      tick: state.tick,
      counts: stats.counts,
      plantBiomass: Math.round(stats.plantBiomass),
      treeStands: stats.treeStands,
      stability: Math.round(stats.stability),
      biodiversity: Number(stats.shannon.toFixed(2)),
      diseasePct: Number(stats.diseasePct.toFixed(3)),
      avgHunger: Number(stats.avgHunger.toFixed(1)),
      avgThirst: Number(stats.avgThirst.toFixed(1)),
      avgElevation: Number(stats.elevation.toFixed(3)),
      highlandPct: Number(stats.highlandPct.toFixed(3)),
      avgTerrainFactor: Number(stats.avgTerrainFactor.toFixed(3)),
      avgUphillEffort: Number(stats.avgUphillEffort.toFixed(4)),
    });
    els.clock.textContent = `Year ${season.year}, ${season.name}, Day ${season.day}`;
    els.system.textContent = systemStatus(stats);
    const metrics = [
      ["producers", "Plants/Trees", formatNumber(stats.plantBiomass, 0)],
      ["herbivores", "Herbivores", stats.counts.herbivore],
      ["predators", "Predators", stats.counts.predator],
      ["decomposers", "Decomposers", stats.counts.decomposer],
      ["pollinators", "Pollinators", stats.counts.pollinator],
      ["engineers", "Engineers", stats.counts.engineer],
    ];
    els.metricGrid.innerHTML = metrics
      .map(
        ([cls, label, value]) => `
          <article class="metric ${cls}">
            <span>${label}</span>
            <strong>${value}</strong>
          </article>
        `,
      )
      .join("");

    const signals = [
      ["Stability", `${formatNumber(stats.stability, 0)}/100`],
      ["Water", formatPct(stats.moisture, 1)],
      ["Soil nutrients", formatPct(stats.nutrients / 1.4, 1)],
      ["Detritus", formatNumber(stats.detritus, 2)],
      ["Dissolved oxygen", formatPct(stats.oxygen, 1)],
      ["Toxicity", formatPct(stats.toxicity, 1)],
      ["Disease", formatPct(stats.diseasePct, 1)],
      ["Avg hunger", `${formatNumber(stats.avgHunger, 1)}/100`],
      ["Avg thirst", `${formatNumber(stats.avgThirst, 1)}/100`],
      ["Avg elevation", formatPct(stats.elevation, 1)],
      ["Highland area", formatPct(stats.highlandPct, 1)],
      ["Travel speed", formatPct(stats.avgTerrainFactor, 1)],
      ["Uphill effort", formatNumber(stats.avgUphillEffort, 3)],
      ["Tree stands", formatNumber(stats.treeStands, 0)],
      ["Biodiversity", formatNumber(stats.shannon, 2)],
      ["Carrying ratio", formatNumber(stats.carrying, 2)],
      ["Recycling", formatPct(clamp(stats.recycling, 0, 1.8), 1)],
    ];
    els.signalList.innerHTML = signals
      .map(([label, value]) => `<div><dt>${label}</dt><dd>${value}</dd></div>`)
      .join("");

    els.traitRows.innerHTML = TYPE_ORDER.map((type) => traitRow(type)).join("");
    els.eventLog.innerHTML = state.events
      .map(
        (event) => `
          <div class="event-item">
            <strong>${event.title}</strong>
            <span>${event.detail}</span>
          </div>
        `,
      )
      .join("");
  }

  function systemStatus(stats) {
    if (stats.counts.herbivore < 12) return "Herbivore bottleneck";
    if (stats.counts.predator > stats.counts.herbivore * 0.55) return "Predator pressure high";
    if (stats.diseasePct > 0.22) return "Disease pressure high";
    if (stats.avgThirst > 70) return "Thirst stress high";
    if (stats.avgHunger > 72) return "Hunger stress high";
    if (stats.avgTerrainFactor < 0.5) return "Steep terrain slowing movement";
    if (stats.carrying < 0.55) return "Food carrying capacity low";
    if (stats.toxicity > 0.18) return "Waste toxicity rising";
    if (state.droughtTicks > 0) return "Drought stress";
    if (state.floodTicks > 0) return "Flood pulse";
    if (state.firePatches.length) return "Active fire disturbance";
    return "Stable";
  }

  function traitRow(type) {
    const list = agents.filter((agent) => agent.type === type && !agent.dead);
    const info = TYPE_INFO[type];
    if (!list.length) {
      return `<tr><td>${info.label}</td><td>0</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr>`;
    }
    const avg = (key) => list.reduce((sum, agent) => sum + agent.genes[key], 0) / list.length;
    const avgNeed = (key) => list.reduce((sum, agent) => sum + agent[key], 0) / list.length;
    const gen = list.reduce((sum, agent) => sum + agent.generation, 0) / list.length;
    return `
      <tr>
        <td>${info.label}</td>
        <td>${list.length}</td>
        <td>${formatNumber(gen, 1)}</td>
        <td>${formatNumber(avg("speed"), 2)}</td>
        <td>${formatNumber(avg("sense"), 1)}</td>
        <td>${formatNumber(avg("resistance"), 2)}</td>
        <td>${formatNumber(avg("tempPref"), 1)} C</td>
        <td>${formatNumber(avgNeed("hunger"), 1)}/100</td>
        <td>${formatNumber(avgNeed("thirst"), 1)}/100</td>
      </tr>
    `;
  }

  function sizeCanvas(canvas, context) {
    const rect = canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    const width = Math.max(1, Math.floor(rect.width));
    const height = Math.max(1, Math.floor(rect.height));
    canvas.width = Math.max(1, Math.floor(width * dpr));
    canvas.height = Math.max(1, Math.floor(height * dpr));
    context.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { width, height };
  }

  function render() {
    drawWorld();
    drawHistory();
  }

  function drawWorld() {
    const { width, height } = sizeCanvas(els.world, ctx);
    const cellW = width / GRID_W;
    const cellH = height / GRID_H;
    ctx.clearRect(0, 0, width, height);
    for (const cell of cells) {
      ctx.fillStyle = colorForCell(cell, state.overlay);
      ctx.fillRect(cell.x * cellW, cell.y * cellH, Math.ceil(cellW) + 1, Math.ceil(cellH) + 1);
    }

    ctx.save();
    ctx.globalAlpha = 0.42;
    for (const cell of cells) {
      if (cell.vegetation <= 0.12 || state.overlay !== "biome") continue;
      ctx.fillStyle = `rgba(99, 154, 67, ${clamp(cell.vegetation * 0.38, 0.04, 0.34)})`;
      ctx.fillRect(cell.x * cellW, cell.y * cellH, Math.ceil(cellW) + 1, Math.ceil(cellH) + 1);
    }
    ctx.restore();

    if (state.overlay === "biome") {
      for (const cell of cells) drawFlora(cell, cellW, cellH);
    }

    for (const agent of agents) {
      if (!agent.dead) drawAgent(agent, cellW, cellH);
    }
  }

  function colorForCell(cell, overlay) {
    if (overlay === "elevation") {
      const e = clamp(cell.elevation, 0, 1);
      if (e < 0.18) {
        const t = e / 0.18;
        return rgb(lerp(36, 69, t), lerp(103, 139, t), lerp(136, 106, t));
      }
      if (e < 0.45) {
        const t = (e - 0.18) / 0.27;
        return rgb(lerp(69, 85, t), lerp(139, 147, t), lerp(106, 78, t));
      }
      if (e < 0.68) {
        const t = (e - 0.45) / 0.23;
        return rgb(lerp(85, 142, t), lerp(147, 118, t), lerp(78, 72, t));
      }
      const t = (e - 0.68) / 0.32;
      const shade = cell.roughness * 18;
      return rgb(lerp(142, 216, t) - shade, lerp(118, 214, t) - shade, lerp(72, 206, t) - shade);
    }
    if (overlay === "water") {
      const wet = clamp(cell.moisture / 1.1, 0, 1);
      const oxygen = clamp(cell.oxygen, 0, 1);
      return rgb(
        lerp(120, 36, wet),
        lerp(96, 133 + oxygen * 35, wet),
        lerp(70, 160 + oxygen * 45, wet),
      );
    }
    if (overlay === "nutrients") {
      const n = clamp(cell.nutrients / 1.2, 0, 1);
      return rgb(lerp(92, 50, n), lerp(68, 126, n), lerp(45, 64, n));
    }
    if (overlay === "temperature") {
      const t = clamp((cell.temp + 8) / 42, 0, 1);
      return rgb(lerp(48, 177, t), lerp(111, 76, t), lerp(154, 49, t));
    }
    if (overlay === "toxicity") {
      const tox = clamp(cell.toxicity, 0, 1);
      const path = clamp(cell.pathogen, 0, 1);
      return rgb(lerp(62, 146, tox), lerp(91, 54, tox + path * 0.2), lerp(74, 94, path));
    }

    if (cell.burning > 0.08) {
      return rgb(lerp(93, 205, cell.burning), lerp(65, 80, cell.burning), lerp(40, 32, cell.burning));
    }
    if (cell.water) {
      const deep = clamp(cell.moisture - 0.65, 0, 0.6) / 0.6;
      return rgb(lerp(50, 22, deep), lerp(111, 102, deep), lerp(132, 158, deep));
    }
    const veg = clamp(cell.vegetation, 0, 1);
    const dry = clamp(0.38 - cell.moisture, 0, 0.38) / 0.38;
    const high = clamp(cell.elevation - 0.68, 0, 0.32) / 0.32;
    const r = lerp(129, 47, veg) + dry * 32 + high * 24;
    const g = lerp(99, 126, veg) - dry * 18 + high * 14;
    const b = lerp(67, 68, veg) - dry * 28 + high * 24;
    return rgb(r, g, b);
  }

  function rgb(r, g, b) {
    return `rgb(${Math.round(clamp(r, 0, 255))}, ${Math.round(clamp(g, 0, 255))}, ${Math.round(clamp(b, 0, 255))})`;
  }

  function drawFlora(cell, cellW, cellH) {
    if (cell.water || cell.vegetation < 0.18) return;
    const density = clamp((cell.vegetation - 0.18) / 0.78, 0, 1);
    const treeNoise = hashNoise(cell.x, cell.y, state.seed + 701);
    const plantNoise = hashNoise(cell.x, cell.y, state.seed + 919);
    const x = (cell.x + 0.2 + hashNoise(cell.x, cell.y, state.seed + 317) * 0.6) * cellW;
    const y = (cell.y + 0.22 + hashNoise(cell.x, cell.y, state.seed + 521) * 0.58) * cellH;
    const base = Math.max(2, Math.min(cellW, cellH));

    if (density > 0.58 && cell.shelter > 0.34 && treeNoise < density * 0.55) {
      const trunkH = base * (1.1 + density * 0.8);
      const canopy = base * (0.9 + density * 0.95);
      ctx.save();
      ctx.globalAlpha = 0.92;
      ctx.strokeStyle = "#5b432b";
      ctx.lineWidth = Math.max(1, base * 0.18);
      ctx.beginPath();
      ctx.moveTo(x, y + trunkH * 0.52);
      ctx.lineTo(x, y - trunkH * 0.18);
      ctx.stroke();
      ctx.fillStyle = cell.moisture < 0.32 ? "#789747" : "#2f7d4f";
      ctx.beginPath();
      ctx.arc(x, y - trunkH * 0.28, canopy, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = "rgba(142, 179, 57, 0.75)";
      ctx.beginPath();
      ctx.arc(x - canopy * 0.35, y - trunkH * 0.38, canopy * 0.42, 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();
      return;
    }

    if (plantNoise < density * 0.75) {
      ctx.save();
      ctx.globalAlpha = 0.85;
      ctx.strokeStyle = cell.moisture < 0.28 ? "#8b8b44" : "#3f8c4f";
      ctx.lineWidth = Math.max(1, base * 0.13);
      ctx.beginPath();
      ctx.moveTo(x, y + base * 0.42);
      ctx.lineTo(x, y - base * 0.5);
      ctx.moveTo(x, y - base * 0.15);
      ctx.lineTo(x - base * 0.46, y - base * 0.42);
      ctx.moveTo(x, y - base * 0.2);
      ctx.lineTo(x + base * 0.42, y - base * 0.5);
      ctx.stroke();
      if (cell.flower > 0.08) {
        ctx.fillStyle = "#d6b44a";
        ctx.beginPath();
        ctx.arc(x, y - base * 0.62, Math.max(1, base * 0.22), 0, Math.PI * 2);
        ctx.fill();
      }
      ctx.restore();
    }
  }

  function drawAgent(agent, cellW, cellH) {
    const x = agent.x * cellW;
    const y = agent.y * cellH;
    const info = TYPE_INFO[agent.type];
    const size = clamp(2.6 + agent.genes.body * 2.2, 3.2, 6.2);
    ctx.save();
    ctx.translate(x, y);
    ctx.fillStyle = info.color;
    ctx.strokeStyle = "#10231d";
    ctx.lineWidth = 1.1;

    if (agent.type === "predator") {
      ctx.beginPath();
      ctx.moveTo(0, -size);
      ctx.lineTo(size * 0.9, size * 0.85);
      ctx.lineTo(-size * 0.9, size * 0.85);
      ctx.closePath();
      ctx.fill();
      ctx.stroke();
    } else if (agent.type === "decomposer") {
      ctx.fillRect(-size * 0.75, -size * 0.75, size * 1.5, size * 1.5);
      ctx.strokeRect(-size * 0.75, -size * 0.75, size * 1.5, size * 1.5);
    } else if (agent.type === "pollinator") {
      ctx.rotate(Math.PI / 4);
      ctx.fillRect(-size * 0.65, -size * 0.65, size * 1.3, size * 1.3);
      ctx.strokeRect(-size * 0.65, -size * 0.65, size * 1.3, size * 1.3);
    } else if (agent.type === "engineer") {
      ctx.beginPath();
      for (let i = 0; i < 6; i += 1) {
        const a = (Math.PI * 2 * i) / 6;
        const px = Math.cos(a) * size;
        const py = Math.sin(a) * size;
        if (i === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
      }
      ctx.closePath();
      ctx.fill();
      ctx.stroke();
    } else {
      ctx.beginPath();
      ctx.arc(0, 0, size, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
    }

    if (agent.infected) {
      ctx.strokeStyle = "#141414";
      ctx.lineWidth = 1.8;
      ctx.beginPath();
      ctx.arc(0, 0, size + 2, 0, Math.PI * 2);
      ctx.stroke();
    }
    drawNeedBars(agent, size);
    ctx.restore();
  }

  function drawNeedBars(agent, size) {
    const hunger = clamp(agent.hunger / 100, 0, 1);
    const thirst = clamp(agent.thirst / 100, 0, 1);
    const w = size * 2.8;
    const h = 2;
    const y = -size - 7;
    ctx.fillStyle = "rgba(12, 24, 20, 0.62)";
    ctx.fillRect(-w / 2, y, w, h);
    ctx.fillRect(-w / 2, y + 3, w, h);
    ctx.fillStyle = hunger > 0.72 ? "#bd4f39" : "#d6a13d";
    ctx.fillRect(-w / 2, y, w * hunger, h);
    ctx.fillStyle = thirst > 0.72 ? "#bd4f39" : "#287a9b";
    ctx.fillRect(-w / 2, y + 3, w * thirst, h);
  }

  function drawHistory() {
    const { width, height } = sizeCanvas(els.history, hctx);
    hctx.clearRect(0, 0, width, height);
    hctx.fillStyle = "#ffffff";
    hctx.fillRect(0, 0, width, height);
    const margin = { top: 16, right: 16, bottom: 26, left: 44 };
    const plotW = Math.max(10, width - margin.left - margin.right);
    const plotH = Math.max(10, height - margin.top - margin.bottom);
    const rows = state.history;
    if (rows.length < 2) return;
    const maxCount = Math.max(
      10,
      ...rows.flatMap((row) => [row.herbivore, row.predator, row.decomposer, row.pollinator, row.engineer]),
    );

    hctx.strokeStyle = "#d8e0dc";
    hctx.lineWidth = 1;
    hctx.font = "12px system-ui, sans-serif";
    hctx.fillStyle = "#60706a";
    hctx.textAlign = "right";
    hctx.textBaseline = "middle";
    for (let i = 0; i <= 4; i += 1) {
      const y = margin.top + (plotH / 4) * i;
      hctx.beginPath();
      hctx.moveTo(margin.left, y);
      hctx.lineTo(margin.left + plotW, y);
      hctx.stroke();
      hctx.fillText(formatNumber(maxCount - (maxCount / 4) * i, 0), margin.left - 8, y);
    }

    const series = [
      ["herbivore", "#b7862c", "Herb"],
      ["predator", "#bd4f39", "Pred"],
      ["decomposer", "#76569a", "Dec"],
      ["pollinator", "#287a9b", "Poll"],
      ["engineer", "#2d8f86", "Eng"],
    ];
    series.forEach(([key, color]) => {
      hctx.strokeStyle = color;
      hctx.lineWidth = 2.2;
      hctx.beginPath();
      rows.forEach((row, idx) => {
        const x = margin.left + (idx / (rows.length - 1)) * plotW;
        const y = margin.top + plotH - (row[key] / maxCount) * plotH;
        if (idx === 0) hctx.moveTo(x, y);
        else hctx.lineTo(x, y);
      });
      hctx.stroke();
    });

    hctx.textAlign = "left";
    hctx.textBaseline = "top";
    series.forEach(([key, color, label], idx) => {
      const x = margin.left + idx * 72;
      hctx.fillStyle = color;
      hctx.fillRect(x, 6, 15, 3);
      hctx.fillStyle = "#16201d";
      hctx.fillText(label, x + 20, 2);
    });
  }

  function bindControls() {
    els.runToggle.addEventListener("click", () => {
      state.running = !state.running;
      els.runToggle.textContent = state.running ? "Pause" : "Run";
    });
    els.stepOnce.addEventListener("click", () => {
      simulateTick();
      render();
    });
    els.resetWorld.addEventListener("click", () => {
      initWorld(Math.floor(randRange(1, 999999)));
      render();
    });
    els.speed.addEventListener("input", () => {
      state.speed = Number(els.speed.value);
    });
    document.querySelectorAll("[data-overlay]").forEach((button) => {
      button.addEventListener("click", () => {
        state.overlay = button.dataset.overlay;
        document.querySelectorAll("[data-overlay]").forEach((node) => {
          node.classList.toggle("active", node === button);
        });
        render();
      });
    });
    document.querySelectorAll("[data-event]").forEach((button) => {
      button.addEventListener("click", () => {
        triggerEvent(button.dataset.event, false);
        updateUi();
        render();
      });
    });
    window.addEventListener("resize", render);
  }

  function applyStartupParams() {
    const params = new URLSearchParams(window.location.search);
    const testTicks = clamp(Number(params.get("testTicks") || 0), 0, 5000);
    if (params.get("paused") === "1") {
      state.running = false;
      els.runToggle.textContent = "Run";
    }
    if (testTicks > 0) {
      const wasRunning = state.running;
      state.running = false;
      for (let i = 0; i < testTicks; i += 1) simulateTick();
      state.running = wasRunning && params.get("paused") !== "1";
      els.runToggle.textContent = state.running ? "Pause" : "Run";
      updateUi();
    }
  }

  function animationLoop() {
    if (state.running) {
      for (let i = 0; i < state.speed; i += 1) simulateTick();
    }
    render();
    window.requestAnimationFrame(animationLoop);
  }

  bindControls();
  initWorld(state.seed);
  window.EcosystemLab = {
    stats: () => collectStats(),
    trigger: (type) => triggerEvent(type, false),
    setRunning: (running) => {
      state.running = Boolean(running);
      els.runToggle.textContent = state.running ? "Pause" : "Run";
    },
  };
  applyStartupParams();
  render();
  window.requestAnimationFrame(animationLoop);
})();
