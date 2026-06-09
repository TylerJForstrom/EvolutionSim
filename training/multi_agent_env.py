"""Vectorized multi-species ecosystem for coevolutionary RL.

A numpy-vectorized world that hosts all five guilds from sim.js (herbivore,
predator, decomposer, pollinator, engineer). Every alive agent of a species
acts through that species' shared policy; the policies are trained
simultaneously, so the environment from any single species' perspective is
non-stationary — that's the point. Each species adapts to the others as they
adapt to it (Red Queen / coevolution).

The world is preallocated and column-oriented so per-tick updates are matrix
ops, not Python loops. With ~1000 agents and a 64x48 grid, one tick runs in
single-digit milliseconds.

Mechanics mirror sim.js qualitatively (logistic primary production, Holling II
predation, Kleiber/Q10 metabolism, SIRS-style disease pressure, condition-
dependent reproduction, mass-conserving nutrient cycle), tuned to a smaller
grid for training speed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HERBIVORE = 0
PREDATOR = 1
DECOMPOSER = 2
POLLINATOR = 3
ENGINEER = 4
N_SPECIES = 5
SPECIES_NAMES = ["herbivore", "predator", "decomposer", "pollinator", "engineer"]
PREY_SPECIES = (HERBIVORE, POLLINATOR)  # what predators eat

GRID_W = 64
GRID_H = 48
N_CELLS = GRID_W * GRID_H

YEAR_STEPS = 1200
KLEIBER_EXP = 0.75
Q10 = 2.3
NUTRIENT_PER_BIOMASS = 0.09

N_ACTIONS = 9  # stay + 8 cardinal/diagonal moves
# Pre-computed (dx, dy) per action
_ACT_DX = np.array([0, 0, 0.707, 1, 0.707, 0, -0.707, -1, -0.707])
_ACT_DY = np.array([0, -1, -0.707, 0, 0.707, 1, 0.707, 0, -0.707])

OBS_DIM = 28

# Per-species ecological parameters (mirroring sim.js TYPE_INFO, rescaled for
# the smaller grid and shorter year). max_count is a non-binding safety rail.
SPECIES_PARAMS: list[dict] = [
    # HERBIVORE
    dict(
        speed=0.55, metabolism=0.045, base_energy=72.0, min_age=60, max_age=1500,
        repro_energy=100.0, repro_cost=35.0, repro_chance=0.045,
        eat_rate=0.06, food_energy=48.0, max_count=420, init_count=80,
    ),
    # PREDATOR. Iterative tuning history:
    # - Round 1 (first multi-agent run): predator reward dominated by sparse
    #   kills; 0% repro in greedy eval.
    # - Round 2: lowered repro_energy 140->110, metabolism 0.16->0.13, added
    #   per-species looser hunger/thirst repro gates. Predator repro reached
    #   1.6% but prey simultaneously learned to flee (pollinator speed 0.95,
    #   wider threat radius) and predator food signal COLLAPSED from 63% to
    #   33% — classic Red Queen, prey side won.
    # - Round 3 (this): predator pursuit upgrades so they can keep up with
    #   smarter prey. Speed 0.65 -> 0.85 (still under pollinator's 0.95 so
    #   the prey advantage is preserved), attack 0.22 -> 0.28 (higher per-
    #   encounter success), capture_radius 1.4 -> 1.55 (slightly more
    #   forgiving). Assimilation deliberately left at 0.17 to keep the
    #   Lindeman energy pyramid in its ~10-20% target band.
    dict(
        speed=0.85, metabolism=0.13, base_energy=86.0, min_age=70, max_age=1300,
        repro_energy=110.0, repro_cost=38.0, repro_chance=0.022,
        eat_rate=0.0, food_energy=0.0, max_count=80, init_count=14,
        attack=0.28, handling_min=22, handling_max=40, capture_radius=1.55,
        assimilation=0.17,
    ),
    # DECOMPOSER
    dict(
        speed=0.4, metabolism=0.044, base_energy=58.0, min_age=35, max_age=620,
        repro_energy=100.0, repro_cost=38.0, repro_chance=0.018,
        eat_rate=0.06, food_energy=50.0, max_count=220, init_count=30,
    ),
    # POLLINATOR — eat_rate raised because the breakdown showed pollinators
    # earning only 1.3% of their reward from food; they were dying of
    # condition penalty (41%) and threat (24%) before they could harvest
    # enough nectar to reproduce. After 200 updates threat % was still 49%
    # so speed bumped a notch more (predator speed is 0.65, pollinator 0.95
    # gives reliable escape margin in unobstructed terrain) and the threat
    # observation/reward terms below widened to give pollinators an earlier
    # learning signal.
    dict(
        speed=0.95, metabolism=0.045, base_energy=45.0, min_age=30, max_age=700,
        repro_energy=70.0, repro_cost=26.0, repro_chance=0.045,
        eat_rate=0.048, food_energy=55.0, max_count=260, init_count=40,
    ),
    # ENGINEER — round 1 nerf (repro_chance 0.025 -> 0.018, bonus 0.05 -> 0.02)
    # only dropped them from ~53 to ~43 of ~70 total agents — still ~60%
    # of the ecosystem, still crowding prey out. Round 2 is aggressive:
    # repro_chance 0.018 -> 0.010, repro_energy 140 -> 180 (much harder to
    # qualify), max_count 60 -> 35 (hard cap), engineer_bonus 0.02 -> 0.005
    # in _compute_rewards. Goal: engineer pop ~10-15 in eval so prey have
    # room to breathe.
    dict(
        speed=0.4, metabolism=0.055, base_energy=88.0, min_age=95, max_age=1500,
        repro_energy=180.0, repro_cost=60.0, repro_chance=0.010,
        eat_rate=0.04, food_energy=42.0, max_count=35, init_count=10,
    ),
]

# Slot budget: enough room for births to outpace deaths transiently without
# losing reproduction events to a full table.
_MAX_AGENTS = sum(p["max_count"] for p in SPECIES_PARAMS) * 2 + 64

# Reward shaping per species. Tuned so reproduction is the dominant signal
# (it IS lifetime fitness), with eating/drinking small dense rewards to make
# early learning tractable, and species-specific costs/threats.
_REPRO_REWARD = 22.0
_DEATH_PENALTY = 10.0  # was 25; lower so per-life reward isn't dominated by deaths during early instability

# Per-species maximum stored energy. Matches sim.js TYPE_INFO clamps.
_MAX_ENERGY = np.array([220.0, 280.0, 160.0, 130.0, 220.0])


# ---------------------------------------------------------------------------
# World
# ---------------------------------------------------------------------------

@dataclass
class StepInfo:
    """Per-tick bookkeeping returned by World.step_world().

    The trainer keys these per agent slot id to attribute rewards.
    """
    rewards: np.ndarray              # shape (_MAX_AGENTS,), reward this tick
    alive_after: np.ndarray          # shape (_MAX_AGENTS,), bool
    just_died: np.ndarray            # shape (_MAX_AGENTS,), bool
    just_born_slots: np.ndarray      # int slot ids of newborns this tick


class World:
    """Vectorized multi-species ecosystem."""

    def __init__(self, seed: int = 1):
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.steps = 0

        # Cell-level state (flat row-major arrays).
        self.elevation = np.zeros(N_CELLS)
        self.roughness = np.zeros(N_CELLS)
        self.moisture = np.zeros(N_CELLS)
        self.water = np.zeros(N_CELLS, dtype=bool)
        self.vegetation = np.zeros(N_CELLS)
        self.veg_cap = np.zeros(N_CELLS)
        self.nutrients = np.zeros(N_CELLS)
        self.detritus = np.zeros(N_CELLS)
        self.flower = np.zeros(N_CELLS)
        self.flower_cap = np.zeros(N_CELLS)
        self.pathogen = np.zeros(N_CELLS)
        self.toxicity = np.zeros(N_CELLS)
        self.temp = np.zeros(N_CELLS)
        self.base_temp = np.zeros(N_CELLS)
        self.shelter = np.zeros(N_CELLS)

        # Agent table (preallocated, slot-stable across the episode).
        self.type = np.full(_MAX_AGENTS, -1, dtype=np.int32)
        self.x = np.zeros(_MAX_AGENTS)
        self.y = np.zeros(_MAX_AGENTS)
        self.body = np.ones(_MAX_AGENTS)
        self.energy = np.zeros(_MAX_AGENTS)
        self.hunger = np.zeros(_MAX_AGENTS)
        self.thirst = np.zeros(_MAX_AGENTS)
        self.age = np.zeros(_MAX_AGENTS, dtype=np.int32)
        self.alive = np.zeros(_MAX_AGENTS, dtype=bool)
        self.cooldown = np.zeros(_MAX_AGENTS, dtype=np.int32)
        self.hunt_lock = np.zeros(_MAX_AGENTS, dtype=np.int32)
        self.last_slope = np.zeros(_MAX_AGENTS)
        self.last_terrain = np.ones(_MAX_AGENTS)
        # Pending action per slot (set by trainer before step_world()).
        self.pending_action = np.zeros(_MAX_AGENTS, dtype=np.int32)

        # Action effects accumulated within a tick, used for reward shaping.
        self._ate = np.zeros(_MAX_AGENTS)
        self._drank = np.zeros(_MAX_AGENTS)
        self._caught = np.zeros(_MAX_AGENTS)
        self._engineered = np.zeros(_MAX_AGENTS)
        self._offspring = np.zeros(_MAX_AGENTS, dtype=np.int32)
        self._predator_threat = np.zeros(_MAX_AGENTS)
        # Predator "closeness to prey" signal — small dense reward proportional
        # to proximity to nearest prey, so the predator policy gets a gradient
        # toward stalking even on ticks where it doesn't successfully kill.
        # Without this the predator's only food signal is the rare kill event,
        # and its policy stays effectively random (the main pathology observed
        # in the first multi-agent training run).
        self._stalk_closeness = np.zeros(_MAX_AGENTS)
        # Per-slot reward breakdown from the most recent step, used by the
        # trainer to log "what fraction of each species' reward came from
        # food vs reproduction vs threat" — essential for diagnosing why a
        # species isn't learning.
        self._reward_components: dict[str, np.ndarray] = {
            "base": np.zeros(_MAX_AGENTS),
            "food": np.zeros(_MAX_AGENTS),
            "drink": np.zeros(_MAX_AGENTS),
            "repro": np.zeros(_MAX_AGENTS),
            "threat": np.zeros(_MAX_AGENTS),
            "condition": np.zeros(_MAX_AGENTS),
            "engineer_bonus": np.zeros(_MAX_AGENTS),
            "death": np.zeros(_MAX_AGENTS),
        }

        self._make_map()
        self._spawn_initial()

    # ----------------------------- map gen ---------------------------------
    def _make_map(self) -> None:
        # Multi-octave fractal noise. We compute it here in numpy via sin
        # hashing rather than calling a per-cell function, so map generation
        # is also vectorized.
        ys, xs = np.indices((GRID_H, GRID_W))
        x_n = xs / (GRID_W - 1) - 0.5
        y_n = ys / (GRID_H - 1) - 0.5
        radial = np.sqrt(x_n * x_n + y_n * y_n)

        def hash_noise(ix, iy, seed):
            v = np.sin(ix * 127.1 + iy * 311.7 + seed * 74.7) * 43758.5453123
            return v - np.floor(v)

        def smoothstep(t):
            return t * t * (3.0 - 2.0 * t)

        def smooth_noise(x, y, scale, seed):
            sx = x / scale
            sy = y / scale
            x0 = np.floor(sx).astype(np.int64)
            y0 = np.floor(sy).astype(np.int64)
            tx = smoothstep(sx - x0)
            ty = smoothstep(sy - y0)
            a = hash_noise(x0, y0, seed)
            b = hash_noise(x0 + 1, y0, seed)
            c = hash_noise(x0, y0 + 1, seed)
            d = hash_noise(x0 + 1, y0 + 1, seed)
            return (a + (b - a) * tx) + ((c + (d - c) * tx) - (a + (b - a) * tx)) * ty

        def fractal(x, y, seed):
            return (
                smooth_noise(x, y, 28, seed) * 0.52
                + smooth_noise(x, y, 12, seed + 13) * 0.31
                + smooth_noise(x, y, 5, seed + 29) * 0.17
            )

        seed = self.seed * 9973 + 17
        terrain = fractal(xs.astype(np.float64), ys.astype(np.float64), seed)
        broad = fractal(xs * 0.5, ys * 0.5, seed + 17)
        ridges = np.abs(fractal(xs * 1.7, ys * 1.7, seed + 7) - 0.5) * 2.0

        river_center = GRID_W * (0.5 + 0.22 * np.sin(ys * 0.09 + seed * 0.001))
        valley = np.exp(-((xs - river_center) ** 2) / 26.0)
        moisture_noise = fractal(xs.astype(np.float64), ys.astype(np.float64), seed + 41)
        nutrient_noise = fractal(xs.astype(np.float64), ys.astype(np.float64), seed + 99)

        elevation = np.clip(
            0.5 + (terrain - 0.5) * 0.78 + (broad - 0.5) * 0.58 + ridges * 0.28
            - radial * 0.22 - valley * 0.22,
            0.0, 1.0,
        )
        roughness = np.clip(ridges * 0.62 + np.abs(terrain - broad) * 0.55 + elevation * 0.18, 0.0, 1.0)
        moisture = np.clip(0.34 + (moisture_noise - 0.5) * 0.55 + valley * 0.18 - elevation * 0.3, 0.0, 1.2)
        water = (moisture > 0.78) | (valley > 0.7) | (elevation < 0.1)
        nutrients = np.clip(0.32 + (nutrient_noise - 0.5) * 0.4 + moisture * 0.2 - elevation * 0.1, 0.05, 1.0)
        base_temp = 27.0 - (ys / (GRID_H - 1)) * 14.0 - elevation * 8.0
        # Vegetation carrying capacity.
        veg_cap = np.where(
            water,
            np.clip(0.05 + nutrients * 0.26, 0.0, 0.6),
            np.clip(0.08 + moisture * 0.5 + nutrients * 0.28 - elevation * 0.11, 0.02, 0.95),
        )
        # Initial vegetation seeded near capacity so the world isn't sterile.
        vegetation = veg_cap * (0.55 + 0.4 * self.rng.random(veg_cap.shape))
        flower_cap = np.clip(vegetation * 0.35, 0.0, 0.7)
        flower = flower_cap * 0.6

        self.elevation = elevation.flatten()
        self.roughness = roughness.flatten()
        self.moisture = moisture.flatten()
        self.water = water.flatten()
        self.vegetation = vegetation.flatten()
        self.veg_cap = veg_cap.flatten()
        self.nutrients = nutrients.flatten()
        # Initial detritus pool is sized so decomposers find food in the
        # early-training phase. Without enough starting detritus they go
        # straight to chronic starvation (~75% condition penalty observed
        # in the first breakdown log) and never get a meaningful learning
        # signal — by the time the system reaches steady state they've
        # already been culled by rescue migration cycles.
        self.detritus = 0.10 + 0.14 * self.rng.random(N_CELLS)
        self.flower = flower.flatten()
        self.flower_cap = flower_cap.flatten()
        self.pathogen = np.zeros(N_CELLS)
        self.toxicity = np.zeros(N_CELLS)
        self.base_temp = base_temp.flatten()
        self.temp = self.base_temp.copy()
        # Shelter (loose proxy: more cover where roughness is high and elevation moderate).
        self.shelter = np.clip(roughness * 0.6 + (1.0 - np.abs(elevation - 0.5)) * 0.4, 0.0, 1.0).flatten()

    # --------------------------- initial spawn -----------------------------
    def _spawn_initial(self) -> None:
        for species in range(N_SPECIES):
            p = SPECIES_PARAMS[species]
            for _ in range(p["init_count"]):
                slot = self._alloc_slot()
                if slot < 0:
                    break
                # Place on a habitable cell — vegetated, non-water, low elevation.
                cell = self._pick_spawn_cell(species)
                cx = cell % GRID_W
                cy = cell // GRID_W
                self.x[slot] = cx + self.rng.random()
                self.y[slot] = cy + self.rng.random()
                self.type[slot] = species
                self.body[slot] = float(self.rng.uniform(0.85, 1.15))
                self.energy[slot] = p["base_energy"] * float(self.rng.uniform(0.85, 1.1))
                self.hunger[slot] = float(self.rng.uniform(12.0, 32.0))
                self.thirst[slot] = float(self.rng.uniform(10.0, 30.0))
                self.age[slot] = int(self.rng.integers(0, p["min_age"]))
                self.alive[slot] = True
                self.cooldown[slot] = int(self.rng.integers(0, 80))

    def _pick_spawn_cell(self, species: int) -> int:
        # Score 80 random candidates and take the best — matches sim.js style.
        candidates = self.rng.integers(0, N_CELLS, size=80)
        veg = self.vegetation[candidates]
        moi = self.moisture[candidates]
        nut = self.nutrients[candidates]
        water_pen = np.where(self.water[candidates], -0.6, 0.0)
        det = self.detritus[candidates]
        if species == DECOMPOSER:
            scores = det + moi * 0.4 + nut * 0.2 + water_pen
        elif species == POLLINATOR:
            scores = self.flower[candidates] * 2.0 + veg * 0.25 + water_pen
        elif species == ENGINEER:
            scores = moi + veg * 0.45 - self.elevation[candidates] * 0.15 + water_pen
        elif species == PREDATOR:
            scores = veg + self.shelter[candidates] * 0.4 + water_pen
        else:  # HERBIVORE
            scores = veg + moi * 0.3 + nut * 0.2 + water_pen
        return int(candidates[int(np.argmax(scores))])

    def _alloc_slot(self) -> int:
        # Find first dead slot. Linear scan is fine — _MAX_AGENTS is small.
        free = np.where(~self.alive)[0]
        return int(free[0]) if free.size else -1

    # ----------------------------- helpers ---------------------------------
    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self.seed = int(seed)
            self.rng = np.random.default_rng(self.seed)
        self.steps = 0
        self.alive[:] = False
        self.type[:] = -1
        self.age[:] = 0
        self.cooldown[:] = 0
        self.hunt_lock[:] = 0
        self.last_slope[:] = 0.0
        self.last_terrain[:] = 1.0
        self._make_map()
        self._spawn_initial()

    def season_sun(self) -> float:
        phase = (self.steps % YEAR_STEPS) / YEAR_STEPS
        return 0.72 + 0.28 * math.sin(phase * 2.0 * math.pi - math.pi / 2.0)

    def season_flower(self) -> float:
        # Spring-summer flower pulse with winter near-silence.
        phase = (self.steps % YEAR_STEPS) / YEAR_STEPS
        return max(0.08, 0.65 + 0.6 * math.sin(phase * 2.0 * math.pi - math.pi / 2.0))

    def _cell_idx_at(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        ix = np.clip(x.astype(np.int32), 0, GRID_W - 1)
        iy = np.clip(y.astype(np.int32), 0, GRID_H - 1)
        return iy * GRID_W + ix

    # ----------------------------- environment ------------------------------
    def _update_environment(self) -> None:
        sun = self.season_sun()
        # Temperature: base + small daily noise via seasonal sin (already in base via map seasonal not modeled tick-to-tick, kept constant base_temp).
        self.temp = self.base_temp.copy()
        # Vegetation logistic growth + small seed term so depleted cells recover.
        comp = np.clip(1.0 - self.vegetation / np.maximum(self.veg_cap, 1e-6), -0.2, 1.0)
        light = np.clip(sun - self.elevation * 0.06, 0.2, 1.05)
        nutrient_factor = np.clip(self.nutrients / 0.42, 0.0, 1.4)
        moisture_factor = np.where(
            self.water,
            np.clip(self.moisture, 0.4, 1.1),
            np.clip(self.moisture / 0.52, 0.0, 1.25),
        )
        bell_temp = np.exp(-((self.temp - 22.0) / 15.0) ** 2)
        growth = np.maximum(
            0.0,
            self.vegetation * 0.018 * comp * light * moisture_factor * bell_temp * nutrient_factor * (1 - self.toxicity * 0.8),
        )
        seedling = np.where(
            (self.vegetation < 0.035) & (self.nutrients > 0.16) & (self.moisture > 0.16),
            0.0012 * light * nutrient_factor,
            0.0,
        )
        stress = self.vegetation * (np.maximum(0.0, 0.18 - self.moisture) * 0.005 * 2.5)
        self.vegetation = np.clip(self.vegetation + growth + seedling - stress, 0.0, self.veg_cap)
        # Nutrient cycle: producers take up, detritus mineralizes back.
        uptake = (growth + seedling) * NUTRIENT_PER_BIOMASS
        self.detritus = np.clip(self.detritus + stress, 0.0, 4.0)
        self.nutrients = np.clip(self.nutrients - uptake, 0.0, 1.6)
        microbe = np.clip(self.moisture / 0.58, 0.0, 1.3) * bell_temp
        decomposed = np.minimum(self.detritus, self.detritus * (0.006 + microbe * 0.018))
        self.detritus -= decomposed
        released = decomposed * NUTRIENT_PER_BIOMASS
        weathering = 0.00008
        leaching = self.nutrients * self.moisture * 0.0009
        self.nutrients = np.clip(self.nutrients + released + weathering - leaching, 0.0, 1.6)
        # Nectar regenerates toward a vegetation/season-set target. Regen
        # rate bumped 0.08 -> 0.15 and the cap raised so depleted flowers
        # recover within ~7 ticks instead of ~12. The breakdown log showed
        # pollinators earning only 1.3% of their reward from food — they
        # were finding flowers but each visit harvested too little nectar
        # before depletion, so they couldn't keep up with metabolism.
        flower_season = self.season_flower()
        self.flower_cap = np.clip(self.vegetation * 0.35 * flower_season * (1 - self.toxicity), 0.0, 0.7)
        self.flower = np.clip(self.flower + (self.flower_cap - self.flower) * 0.15, 0.0, 0.7)
        # Pathogen decay.
        self.pathogen *= 0.992

    # ----------------------------- observation -----------------------------
    def observe(self, species: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (obs, slot_ids) for all alive agents of `species`."""
        slots = np.where(self.alive & (self.type == species))[0]
        if slots.size == 0:
            return np.zeros((0, OBS_DIM), dtype=np.float64), slots
        obs = np.zeros((slots.size, OBS_DIM))
        x = self.x[slots]
        y = self.y[slots]
        cell = self._cell_idx_at(x, y)

        p = SPECIES_PARAMS[species]
        obs[:, 0] = self.energy[slots] / 200.0
        obs[:, 1] = self.hunger[slots] / 140.0
        obs[:, 2] = self.thirst[slots] / 150.0
        obs[:, 3] = self.age[slots] / 1000.0
        obs[:, 4] = self.body[slots]
        repro_ready = (
            (self.energy[slots] >= p["repro_energy"]) &
            (self.cooldown[slots] == 0) &
            (self.age[slots] >= p["min_age"])
        ).astype(np.float64)
        obs[:, 5] = repro_ready
        obs[:, 6] = self.vegetation[cell]
        obs[:, 7] = self.flower[cell]
        obs[:, 8] = self.water[cell].astype(np.float64)
        obs[:, 9] = np.clip(self.moisture[cell], 0.0, 1.5)
        obs[:, 10] = self.nutrients[cell]
        obs[:, 11] = np.clip(self.detritus[cell], 0.0, 4.0) / 4.0
        obs[:, 12] = self.elevation[cell]
        obs[:, 13] = self.roughness[cell]
        obs[:, 14] = self.temp[cell] / 38.0
        obs[:, 15] = np.clip(self.last_slope[slots], -1.0, 1.0)
        obs[:, 16] = self.last_terrain[slots]
        obs[:, 17] = self.season_sun()

        # Resource senses: scan a small ring of offsets and pick the highest-
        # scoring cell. Vectorized over agents.
        food_dx, food_dy, food_score = self._sense_resource(species, x, y, kind="food")
        water_dx, water_dy, water_score = self._sense_resource(species, x, y, kind="water")
        obs[:, 18] = food_dx
        obs[:, 19] = food_dy
        obs[:, 20] = food_score
        obs[:, 21] = water_dx
        obs[:, 22] = water_dy
        obs[:, 23] = water_score

        # Predator/prey awareness slot. Prey species (HERB, POLL) see the
        # direction & proximity of the nearest PREDATOR (threat). Predators
        # see the direction & proximity of the nearest PREY (target) — this
        # was missing before round 3 and is why predators were essentially
        # foraging blind, relying on vegetation as a proxy for prey location.
        # Other species get zeros (irrelevant).
        if species in PREY_SPECIES:
            pdx, pdy, pscore = self._sense_predators(x, y)
        elif species == PREDATOR:
            pdx, pdy, pscore = self._sense_prey(x, y)
        else:
            pdx = np.zeros(slots.size)
            pdy = np.zeros(slots.size)
            pscore = np.zeros(slots.size)
        obs[:, 24] = pdx
        obs[:, 25] = pdy
        obs[:, 26] = pscore

        # Crowd / mate signal: count of own species nearby.
        obs[:, 27] = self._crowd_signal(species, slots, x, y)
        return obs, slots

    def _sense_resource(self, species: int, ax: np.ndarray, ay: np.ndarray, kind: str):
        # Ring of 8 directions x 3 radii. Each agent picks the best of 24
        # sampled cells by species-specific score.
        n = ax.size
        offs = np.array([(np.cos(a), np.sin(a)) for a in np.linspace(0, 2 * np.pi, 8, endpoint=False)])
        radii = np.array([2.0, 4.5, 8.0])
        # Build (n, 8, 3) sample positions, then flatten the last two dims.
        dx_grid = offs[:, 0][:, None] * radii[None, :]  # (8, 3)
        dy_grid = offs[:, 1][:, None] * radii[None, :]
        sx = ax[:, None, None] + dx_grid[None, :, :]
        sy = ay[:, None, None] + dy_grid[None, :, :]
        cells = self._cell_idx_at(sx, sy)  # (n, 8, 3)
        if kind == "water":
            # Use moisture as a continuous proxy (water tiles ~ moisture high).
            scores = np.where(self.water[cells], 1.0, np.clip(self.moisture[cells], 0.0, 1.0))
        else:  # food, species-specific
            if species == HERBIVORE or species == ENGINEER:
                scores = self.vegetation[cells]
            elif species == POLLINATOR:
                scores = self.flower[cells]
            elif species == DECOMPOSER:
                scores = self.detritus[cells]
            elif species == PREDATOR:
                # Predators "smell" cells that recently had prey activity. Use
                # vegetation as a coarse proxy (prey congregate near food).
                # Refined further at hunt time via spatial pairwise query.
                scores = self.vegetation[cells] * 0.6 + np.clip(self.detritus[cells], 0.0, 1.0) * 0.2
            else:
                scores = np.zeros_like(cells, dtype=np.float64)
            # Mild elevation penalty.
            scores = scores - self.elevation[cells] * 0.04
        # Flatten the (8, 3) dims so we can argmax.
        flat = scores.reshape(n, -1)
        best = np.argmax(flat, axis=1)
        bi, ri = best // 3, best % 3
        best_dx = offs[bi, 0]
        best_dy = offs[bi, 1]
        best_score = np.clip(flat[np.arange(n), best], 0.0, 1.0)
        return best_dx, best_dy, best_score

    def _sense_prey(self, ax: np.ndarray, ay: np.ndarray):
        """Predator-only: direction & proximity of nearest prey. Mirror of
        _sense_predators (which prey use to see the nearest predator), but
        from the hunter's perspective."""
        prey_slots = np.where(self.alive & ((self.type == HERBIVORE) | (self.type == POLLINATOR)))[0]
        if prey_slots.size == 0:
            return np.zeros(ax.size), np.zeros(ax.size), np.zeros(ax.size)
        qx = self.x[prey_slots]
        qy = self.y[prey_slots]
        dx = qx[None, :] - ax[:, None]
        dy = qy[None, :] - ay[:, None]
        d2 = dx * dx + dy * dy
        nearest = np.argmin(d2, axis=1)
        d = np.sqrt(d2[np.arange(ax.size), nearest])
        dir_x = np.where(d > 1e-6, dx[np.arange(ax.size), nearest] / np.maximum(d, 1e-6), 0.0)
        dir_y = np.where(d > 1e-6, dy[np.arange(ax.size), nearest] / np.maximum(d, 1e-6), 0.0)
        # Closer prey = higher value. Same 18-cell saturation as the threat
        # signal so the network can repurpose the same circuitry per species.
        score = np.clip(1.0 - d / 18.0, 0.0, 1.0)
        return dir_x, dir_y, score

    def _sense_predators(self, ax: np.ndarray, ay: np.ndarray):
        pred_slots = np.where(self.alive & (self.type == PREDATOR))[0]
        if pred_slots.size == 0:
            return np.zeros(ax.size), np.zeros(ax.size), np.zeros(ax.size)
        px = self.x[pred_slots]
        py = self.y[pred_slots]
        # Pairwise distance (prey rows, predator cols).
        dx = px[None, :] - ax[:, None]
        dy = py[None, :] - ay[:, None]
        d2 = dx * dx + dy * dy
        nearest = np.argmin(d2, axis=1)
        d = np.sqrt(d2[np.arange(ax.size), nearest])
        dir_x = np.where(d > 1e-6, dx[np.arange(ax.size), nearest] / np.maximum(d, 1e-6), 0.0)
        dir_y = np.where(d > 1e-6, dy[np.arange(ax.size), nearest] / np.maximum(d, 1e-6), 0.0)
        # Threat awareness in OBSERVATION saturates at 18 cells (was 12) so
        # prey species detect predators earlier and have time to flee before
        # the threat REWARD penalty (which triggers at 10 cells, below) kicks
        # in hard. Earlier signal = earlier learning of flee behavior.
        score = np.clip(1.0 - d / 18.0, 0.0, 1.0)
        return dir_x, dir_y, score

    def _crowd_signal(self, species: int, slots: np.ndarray, ax: np.ndarray, ay: np.ndarray) -> np.ndarray:
        same = np.where(self.alive & (self.type == species))[0]
        if same.size <= 1:
            return np.zeros(slots.size)
        sx = self.x[same]
        sy = self.y[same]
        # Pairwise distances ego x same — small numbers so brute force is fine.
        dx = sx[None, :] - ax[:, None]
        dy = sy[None, :] - ay[:, None]
        d2 = dx * dx + dy * dy
        # Count within radius 4, minus self (zero distance).
        in_range = ((d2 < 16.0) & (d2 > 1e-6)).sum(axis=1)
        return np.clip(in_range / 8.0, 0.0, 1.0)

    # ----------------------------- actions ---------------------------------
    def set_actions(self, species: int, slots: np.ndarray, actions: np.ndarray) -> None:
        self.pending_action[slots] = actions

    # ------------------------------ step -----------------------------------
    def step_world(self) -> StepInfo:
        """Advance one tick. Returns per-slot rewards + death/birth events."""
        self._ate[:] = 0.0
        self._drank[:] = 0.0
        self._caught[:] = 0.0
        self._engineered[:] = 0.0
        self._offspring[:] = 0
        self._predator_threat[:] = 0.0
        self._stalk_closeness[:] = 0.0
        for arr in self._reward_components.values():
            arr[:] = 0.0

        self._update_environment()

        live = np.where(self.alive)[0]
        if live.size == 0:
            self.steps += 1
            rewards = np.zeros(_MAX_AGENTS)
            return StepInfo(rewards, self.alive.copy(), np.zeros(_MAX_AGENTS, dtype=bool), np.array([], dtype=np.int64))

        # ---- Movement
        action = self.pending_action[live]
        dx = _ACT_DX[action]
        dy = _ACT_DY[action]
        # Species speed table.
        speeds = np.array([SPECIES_PARAMS[i]["speed"] for i in range(N_SPECIES)])
        base_speed = speeds[self.type[live]]
        old_cell = self._cell_idx_at(self.x[live], self.y[live])
        target_x = np.clip(self.x[live] + dx * base_speed, 0.0, GRID_W - 1.001)
        target_y = np.clip(self.y[live] + dy * base_speed, 0.0, GRID_H - 1.001)
        new_cell = self._cell_idx_at(target_x, target_y)
        slope = self.elevation[new_cell] - self.elevation[old_cell]
        uphill = np.maximum(0.0, slope)
        downhill = np.maximum(0.0, -slope)
        roughness = (self.roughness[old_cell] + self.roughness[new_cell]) * 0.5
        water_drag = np.where(self.water[new_cell] & (self.type[live] != DECOMPOSER), 0.22, 0.0)
        terrain_factor = np.clip(1.0 - uphill * 3.6 - roughness * 0.25 - water_drag + downhill * 0.5, 0.2, 1.18)
        effort = base_speed * (1.0 + uphill * 9.0 + roughness * 0.8 + water_drag)
        self.x[live] = np.clip(self.x[live] + dx * base_speed * terrain_factor, 0.0, GRID_W - 1.001)
        self.y[live] = np.clip(self.y[live] + dy * base_speed * terrain_factor, 0.0, GRID_H - 1.001)
        self.last_slope[live] = slope
        self.last_terrain[live] = terrain_factor

        # ---- Metabolism (vectorized Kleiber + Q10)
        cell_now = self._cell_idx_at(self.x[live], self.y[live])
        cell_temp = self.temp[cell_now]
        mass_metab = self.body[live] ** KLEIBER_EXP
        q10 = np.clip(Q10 ** ((cell_temp - 20.0) / 10.0), 0.55, 2.6)
        # Per-species metabolism rate.
        metab_rates = np.array([SPECIES_PARAMS[i]["metabolism"] for i in range(N_SPECIES)])
        basal = metab_rates[self.type[live]] * mass_metab * q10
        self.energy[live] -= basal + effort * 0.035 + uphill * 0.45

        # ---- Needs (hunger, thirst)
        self.hunger[live] = np.clip(self.hunger[live] + 0.22 + effort * 0.22, 0.0, 140.0)
        self.thirst[live] = np.clip(
            self.thirst[live] + 0.18 + effort * 0.2 + np.maximum(0.0, cell_temp - 22.0) * 0.012,
            0.0, 150.0,
        )

        # ---- Aging & cooldowns
        self.age[live] += 1
        self.cooldown[live] = np.maximum(0, self.cooldown[live] - 1)
        self.hunt_lock[live] = np.maximum(0, self.hunt_lock[live] - 1)

        # ---- Per-species interactions (eat/drink/hunt/decompose/pollinate/engineer)
        self._do_drink(live, cell_now)
        self._do_graze(live, cell_now)
        self._do_decompose(live, cell_now)
        self._do_pollinate(live, cell_now)
        self._do_engineer(live, cell_now)
        # Hunting is pairwise and updates self._caught + kills prey.
        self._do_hunt()

        # ---- Disease pressure (simple density-dependent risk)
        self._apply_disease(live, cell_now)

        # ---- Starvation/dehydration energy cost
        starv = np.maximum(0.0, self.hunger[live] - 90.0) * 0.03
        dehyd = np.maximum(0.0, self.thirst[live] - 84.0) * 0.04
        self.energy[live] -= starv + dehyd

        # ---- Per-species energy cap (applied once, after all interactions).
        # Each species has its own ceiling per sim.js; capping per-call inside
        # each interaction function would clobber other species' energy.
        caps = _MAX_ENERGY[self.type[live]]
        self.energy[live] = np.minimum(self.energy[live], caps)

        # ---- Reproduction
        born_slots = self._do_reproduction(live)

        # ---- Death (energy <=0 OR age > maxAge)
        max_ages = np.array([SPECIES_PARAMS[i]["max_age"] for i in range(N_SPECIES)])
        too_old = self.age[live] > max_ages[self.type[live]]
        starved = self.energy[live] <= 0.0
        die_mask = too_old | starved
        died_slots = live[die_mask]
        # Carcasses become detritus (mass conservation).
        if died_slots.size:
            dcell = self._cell_idx_at(self.x[died_slots], self.y[died_slots])
            mass = self.body[died_slots]
            np.add.at(self.detritus, dcell, mass * 0.3)
        self.alive[died_slots] = False

        # ---- Reward per slot
        rewards = self._compute_rewards(live)

        # ---- Rescue migration (sim.js parallel): a closed patch otherwise
        # suffers permanent extinctions a real connected landscape would avoid
        # via dispersal. This is RESCUE only (small refill on near-collapse),
        # not population pegging — so emergent regulation is preserved while
        # training-time extinction cascades are prevented.
        rescues = self._rescue_migration()
        if rescues.size:
            born_slots = np.concatenate([born_slots, rescues]) if born_slots.size else rescues

        self.steps += 1
        just_died = np.zeros(_MAX_AGENTS, dtype=bool)
        just_died[died_slots] = True
        return StepInfo(
            rewards=rewards,
            alive_after=self.alive.copy(),
            just_died=just_died,
            just_born_slots=born_slots,
        )

    def _rescue_migration(self) -> np.ndarray:
        """Spawn a few new individuals into species that have nearly collapsed.
        Fires periodically (not every tick) and only below a low threshold
        per species, so it never pegs populations at a target."""
        if self.steps % 110 != 0:
            return np.array([], dtype=np.int64)
        new_slots: list[int] = []
        # (species, threshold, refill_count, gate condition)
        plant_biomass = float(self.vegetation.sum())
        detritus_pool = float(self.detritus.sum())
        herb_count = int((self.alive & (self.type == HERBIVORE)).sum())
        rules = [
            (HERBIVORE,   4, 4, plant_biomass > 30.0),
            (POLLINATOR,  4, 4, plant_biomass > 20.0),
            (PREDATOR,    2, 2, herb_count > 40),
            (DECOMPOSER,  4, 4, detritus_pool > 5.0),
            (ENGINEER,    3, 3, plant_biomass > 20.0),
        ]
        for sp, thresh, count, gate in rules:
            current = int((self.alive & (self.type == sp)).sum())
            if current >= thresh or not gate:
                continue
            p = SPECIES_PARAMS[sp]
            for _ in range(count):
                slot = self._alloc_slot()
                if slot < 0:
                    break
                cell = self._pick_spawn_cell(sp)
                cx = cell % GRID_W
                cy = cell // GRID_W
                # Spawn near a map edge (dispersal from neighbouring patch).
                edge = float(self.rng.random())
                if edge < 0.25:
                    cx = float(self.rng.uniform(0.2, 2.2))
                elif edge < 0.5:
                    cx = float(self.rng.uniform(GRID_W - 2.2, GRID_W - 0.2))
                elif edge < 0.75:
                    cy = float(self.rng.uniform(0.2, 2.2))
                else:
                    cy = float(self.rng.uniform(GRID_H - 2.2, GRID_H - 0.2))
                self.x[slot] = float(np.clip(cx, 0.0, GRID_W - 1.001))
                self.y[slot] = float(np.clip(cy, 0.0, GRID_H - 1.001))
                self.type[slot] = sp
                self.body[slot] = float(self.rng.uniform(0.9, 1.1))
                self.energy[slot] = p["base_energy"] * float(self.rng.uniform(0.95, 1.1))
                self.hunger[slot] = 18.0
                self.thirst[slot] = 16.0
                self.age[slot] = 0
                self.alive[slot] = True
                self.cooldown[slot] = 0
                self.hunt_lock[slot] = 0
                new_slots.append(slot)
        return np.array(new_slots, dtype=np.int64) if new_slots else np.array([], dtype=np.int64)

    # ------------------------ interactions ---------------------------------
    def _do_drink(self, live: np.ndarray, cell: np.ndarray) -> None:
        # Anyone on a water tile or wet cell drinks.
        wet = self.water[cell] | (self.moisture[cell] > 0.58)
        drink_mask = wet & (self.thirst[live] > 10.0)
        if not drink_mask.any():
            return
        amount = np.where(self.water[cell], 0.22, self.moisture[cell] * 0.18)
        amount = np.where(drink_mask, amount, 0.0)
        self.thirst[live] = np.clip(self.thirst[live] - amount * 95.0, 0.0, 150.0)
        # Wet cells deplete a tiny bit; water tiles are non-depleting.
        wet_only = drink_mask & ~self.water[cell]
        if wet_only.any():
            np.subtract.at(self.moisture, cell[wet_only], 0.0006)
        self._drank[live] = np.maximum(self._drank[live], amount)

    def _do_graze(self, live: np.ndarray, cell: np.ndarray) -> None:
        # Herbivores and engineers graze vegetation. Engineers eat less.
        is_herb = self.type[live] == HERBIVORE
        is_engr = self.type[live] == ENGINEER
        graze_mask = (is_herb | is_engr) & (self.vegetation[cell] > 0.04) & ~self.water[cell]
        if not graze_mask.any():
            return
        # Rate per species.
        rate = np.where(is_herb, SPECIES_PARAMS[HERBIVORE]["eat_rate"], SPECIES_PARAMS[ENGINEER]["eat_rate"])
        max_take = rate * self.body[live]
        take = np.minimum(self.vegetation[cell], max_take)
        take = np.where(graze_mask, take, 0.0)
        np.subtract.at(self.vegetation, cell, take)
        # Egesta returns to detritus (mass conservation).
        # Egesta fraction bumped 0.22 -> 0.30 (closer to real ruminant
        # digestion efficiency) so grazing keeps the detritus pool higher
        # and decomposers stay fed.
        np.add.at(self.detritus, cell, take * 0.30)
        food_energy = np.where(is_herb, SPECIES_PARAMS[HERBIVORE]["food_energy"], SPECIES_PARAMS[ENGINEER]["food_energy"])
        gain = take * food_energy * (1.0 - self.toxicity[cell] * 0.4)
        # Add unconditionally (gain is 0 for non-grazers). Per-species cap is
        # applied once at the end of step_world so each species respects its
        # own energy ceiling without cross-species clobbering.
        self.energy[live] += gain
        hunger_delta = np.where(graze_mask, take * 130.0, 0.0)
        self.hunger[live] = np.clip(self.hunger[live] - hunger_delta, 0.0, 140.0)
        self._ate[live] = take

    def _do_decompose(self, live: np.ndarray, cell: np.ndarray) -> None:
        is_dec = self.type[live] == DECOMPOSER
        # Lower threshold (0.04 -> 0.015) so decomposers find more cells with
        # eatable detritus. The starting pool plus egesta-from-grazing keep
        # this realistic — they're not vacuuming up trace amounts, just
        # everything above ~3 g/m^2.
        mask = is_dec & (self.detritus[cell] > 0.015)
        if not mask.any():
            return
        rate = SPECIES_PARAMS[DECOMPOSER]["eat_rate"]
        take = np.minimum(self.detritus[cell], rate * self.body[live])
        take = np.where(mask, take, 0.0)
        np.subtract.at(self.detritus, cell, take)
        # Mineralize: nutrients back to soil at stoichiometric ratio.
        np.add.at(self.nutrients, cell, take * NUTRIENT_PER_BIOMASS)
        gain = take * SPECIES_PARAMS[DECOMPOSER]["food_energy"]
        self.energy[live] += gain
        hunger_delta = np.where(mask, take * 105.0, 0.0)
        self.hunger[live] = np.clip(self.hunger[live] - hunger_delta, 0.0, 140.0)
        self._ate[live] = np.maximum(self._ate[live], take)

    def _do_pollinate(self, live: np.ndarray, cell: np.ndarray) -> None:
        is_poll = self.type[live] == POLLINATOR
        mask = is_poll & (self.flower[cell] > 0.02)
        if not mask.any():
            return
        rate = SPECIES_PARAMS[POLLINATOR]["eat_rate"]
        take = np.minimum(self.flower[cell], rate * self.body[live])
        take = np.where(mask, take, 0.0)
        np.subtract.at(self.flower, cell, take)
        gain = take * SPECIES_PARAMS[POLLINATOR]["food_energy"]
        self.energy[live] += gain
        hunger_delta = np.where(mask, take * 110.0, 0.0)
        thirst_delta = np.where(mask, take * 28.0, 0.0)
        self.hunger[live] = np.clip(self.hunger[live] - hunger_delta, 0.0, 140.0)
        self.thirst[live] = np.clip(self.thirst[live] - thirst_delta, 0.0, 150.0)
        self._ate[live] = np.maximum(self._ate[live], take)

    def _do_engineer(self, live: np.ndarray, cell: np.ndarray) -> None:
        is_eng = self.type[live] == ENGINEER
        if not is_eng.any():
            return
        # Engineers boost moisture in their local cell (terraforming).
        eng_cells = cell[is_eng]
        bonus = 0.004 * np.ones(eng_cells.size)
        np.add.at(self.moisture, eng_cells, bonus)
        np.minimum(self.moisture, 1.35, out=self.moisture)
        self._engineered[live] = np.where(is_eng, 1.0, 0.0)

    def _do_hunt(self) -> None:
        pred_slots = np.where(self.alive & (self.type == PREDATOR) & (self.hunt_lock == 0))[0]
        if pred_slots.size == 0:
            return
        prey_mask = self.alive & ((self.type == HERBIVORE) | (self.type == POLLINATOR))
        prey_slots = np.where(prey_mask)[0]
        if prey_slots.size == 0:
            return
        p = SPECIES_PARAMS[PREDATOR]
        cap_r = p["capture_radius"]
        # Pairwise pred x prey distance.
        px = self.x[pred_slots]
        py = self.y[pred_slots]
        qx = self.x[prey_slots]
        qy = self.y[prey_slots]
        dx = qx[None, :] - px[:, None]
        dy = qy[None, :] - py[:, None]
        d2 = dx * dx + dy * dy
        within = d2 < cap_r * cap_r
        # Compute pairwise pred-prey distance once and reuse for: (1) prey
        # threat signal, (2) predator stalk-closeness signal, (3) capture
        # logic below.
        dx_all = self.x[prey_slots][None, :] - self.x[pred_slots][:, None]
        dy_all = self.y[prey_slots][None, :] - self.y[pred_slots][:, None]
        d_all = np.sqrt(dx_all * dx_all + dy_all * dy_all)  # (n_pred, n_prey)
        # Prey threat radius widened 6 -> 10 cells so the penalty turns on
        # smoothly while prey can still escape. With the previous 6-cell
        # cliff, by the time the threat penalty was meaningful the predator
        # was already nearly in capture range and the prey policy had no
        # chance to learn anticipatory avoidance.
        min_d_prey = np.min(d_all, axis=0)
        threat = np.clip(1.0 - min_d_prey / 10.0, 0.0, 1.0)
        self._predator_threat[prey_slots] = threat
        # Predator stalk closeness = closeness of nearest prey within ~10
        # cells. Goes up smoothly as a predator approaches prey, giving the
        # policy a dense per-tick gradient instead of relying only on rare
        # kill events.
        min_d_pred = np.min(d_all, axis=1)
        stalk = np.clip(1.0 - min_d_pred / 10.0, 0.0, 1.0)
        self._stalk_closeness[pred_slots] = stalk
        # For each predator, count nearby prey (Holling II saturating term).
        prey_count = within.sum(axis=1)
        attack = p["attack"]
        capture_prob = 1.0 - np.exp(-attack * prey_count)
        rolls = self.rng.random(pred_slots.size)
        successful = (capture_prob > rolls) & (prey_count > 0)
        if not successful.any():
            return
        # Pick nearest prey for each successful predator.
        # Mask out non-within distances to +inf, then argmin.
        d2_masked = np.where(within, d2, np.inf)
        target = np.argmin(d2_masked, axis=1)
        winners = pred_slots[successful]
        target_prey = prey_slots[target[successful]]
        # Apply kills; if two predators target the same prey only the first
        # wins. Resolve via unique.
        unique_prey, first_idx = np.unique(target_prey, return_index=True)
        kill_preds = winners[first_idx]
        kill_prey = unique_prey
        # Energy transfer (Lindeman: only assimilated fraction). Predator cap
        # applied at end-of-step alongside the other species' caps.
        prey_body_energy = np.maximum(0.0, self.energy[kill_prey]) + self.body[kill_prey] * 8.0
        self.energy[kill_preds] += prey_body_energy * p["assimilation"]
        self.hunger[kill_preds] = np.clip(self.hunger[kill_preds] - 70.0, 0.0, 140.0)
        # Handling time.
        self.hunt_lock[kill_preds] = self.rng.integers(p["handling_min"], p["handling_max"], size=kill_preds.size)
        self._caught[kill_preds] = 1.0
        # Mark prey dead (carcass → detritus done later in death sweep is
        # missed — do it here too so killed prey return mass).
        pcell = self._cell_idx_at(self.x[kill_prey], self.y[kill_prey])
        np.add.at(self.detritus, pcell, self.body[kill_prey] * 0.3 + np.maximum(0.0, self.energy[kill_prey]) * 0.003)
        self.alive[kill_prey] = False
        # Set their energy <=0 so the death sweep recognizes them too (so any
        # downstream reward for prey dying is consistent).
        self.energy[kill_prey] = 0.0

    def _apply_disease(self, live: np.ndarray, cell: np.ndarray) -> None:
        # Simplified: a small per-tick energy drain proportional to local
        # pathogen density. Pathogen is produced trivially by crowding (acts
        # as a soft density regulator without a full SIRS state machine).
        crowd = self._cell_crowd(live, cell)
        new_pathogen = crowd * 0.003
        np.add.at(self.pathogen, cell, new_pathogen)
        np.minimum(self.pathogen, 1.0, out=self.pathogen)
        # Local pathogen costs energy.
        self.energy[live] -= self.pathogen[cell] * 0.05

    def _cell_crowd(self, live: np.ndarray, cell: np.ndarray) -> np.ndarray:
        counts = np.bincount(cell, minlength=N_CELLS)
        return counts[cell] - 1  # subtract self

    def _do_reproduction(self, live: np.ndarray) -> np.ndarray:
        """Spawn new agents for those that meet condition gates. Returns
        the slot ids of the newborns (so the trainer can register them)."""
        new_slots: list[int] = []
        # Per-species masks and gates. Predator uses looser hunger/thirst
        # condition gates because (a) obligate carnivores don't biologically
        # require well-hydrated state to breed and (b) keeping the prey
        # default at 55/60 forced the predator's greedy policy to choose
        # between hunting (which keeps thirst above 60) and breeding — so it
        # never bred in eval. Per-species thresholds let predators reproduce
        # right after a successful kill.
        repro_hunger_gate = [55.0, 75.0, 55.0, 55.0, 55.0]
        repro_thirst_gate = [60.0, 80.0, 60.0, 60.0, 60.0]
        species = self.type[live]
        for sp in range(N_SPECIES):
            p = SPECIES_PARAMS[sp]
            sp_mask = species == sp
            if not sp_mask.any():
                continue
            agent_slots = live[sp_mask]
            ready = (
                (self.energy[agent_slots] >= p["repro_energy"]) &
                (self.cooldown[agent_slots] == 0) &
                (self.age[agent_slots] >= p["min_age"]) &
                (self.hunger[agent_slots] < repro_hunger_gate[sp]) &
                (self.thirst[agent_slots] < repro_thirst_gate[sp])
            )
            ready_slots = agent_slots[ready]
            if ready_slots.size == 0:
                continue
            # Random gate (sim-style probability per tick).
            rolls = self.rng.random(ready_slots.size)
            # Population-aware: throttle if at safety cap.
            current_count = int((self.alive & (self.type == sp)).sum())
            cap = p["max_count"]
            cap_pressure = max(0.0, 1.0 - current_count / cap)
            chance = p["repro_chance"] * cap_pressure
            spawn_mask = rolls < chance
            parents = ready_slots[spawn_mask]
            for parent in parents:
                slot = self._alloc_slot()
                if slot < 0:
                    break
                self.x[slot] = float(np.clip(self.x[parent] + self.rng.uniform(-0.6, 0.6), 0.0, GRID_W - 1.001))
                self.y[slot] = float(np.clip(self.y[parent] + self.rng.uniform(-0.6, 0.6), 0.0, GRID_H - 1.001))
                self.type[slot] = sp
                self.body[slot] = float(np.clip(self.body[parent] * (1.0 + self.rng.uniform(-0.08, 0.08)), 0.6, 1.6))
                self.energy[slot] = p["base_energy"] * 0.62
                self.hunger[slot] = 20.0
                self.thirst[slot] = 18.0
                self.age[slot] = 0
                self.alive[slot] = True
                self.cooldown[slot] = 0
                self.hunt_lock[slot] = 0
                self.energy[parent] -= p["repro_cost"]
                self.cooldown[parent] = int(self.rng.integers(80, 200))
                self._offspring[parent] += 1
                new_slots.append(slot)
        return np.array(new_slots, dtype=np.int64) if new_slots else np.array([], dtype=np.int64)

    # ------------------------------- reward --------------------------------
    def _compute_rewards(self, live: np.ndarray) -> np.ndarray:
        """Per-species reward shaping. Reproduction dominates (it IS fitness);
        small dense signals (eat/drink/decompose/etc.) make early learning
        tractable; species-specific threats apply. Each component is also
        written into self._reward_components so the trainer can log per-
        species breakdowns."""
        r = np.zeros(_MAX_AGENTS)
        species = self.type[live]
        ate = self._ate[live]
        drank = self._drank[live]
        caught = self._caught[live]
        engineered = self._engineered[live]
        offspring = self._offspring[live]
        threat = self._predator_threat[live]
        stalk = self._stalk_closeness[live]
        hunger = self.hunger[live]
        thirst = self.thirst[live]

        # Component breakdown (each one written per-slot for the trainer to
        # aggregate). Order: base + food + drink + repro + engineer_bonus
        # - condition - threat - death.
        base = np.full(live.size, 0.01)
        # Species-specific food signal. Predators get a small dense
        # closeness-to-prey signal on every tick PLUS the big sparse kill
        # reward.
        food = np.where(
            (species == HERBIVORE) | (species == ENGINEER), ate * 4.5,
            np.where(species == DECOMPOSER, ate * 5.0,
            np.where(species == POLLINATOR, ate * 5.5,
            np.where(species == PREDATOR, caught * 14.0 + stalk * 0.35, 0.0))),
        )
        drink_r = drank * 2.5
        repro_r = offspring * _REPRO_REWARD
        # Engineer terraforming bonus: 0.05 -> 0.02 (round 1) wasn't enough,
        # engineer pop dropped only 53 -> 43. Round 2: 0.02 -> 0.005,
        # combined with stricter SPECIES_PARAMS gates (repro_energy 140 ->
        # 180, max_count 60 -> 35), so they no longer crowd the ecosystem.
        engineer_bonus = np.where(species == ENGINEER, engineered * 0.005, 0.0)
        # Threat coefficient is sizable so prey species get a STRONG signal to
        # flee predators; without this prey policies never learn avoidance
        # before predator policies learn to hunt, and the system collapses.
        threat_pen = threat * 1.4
        condition_pen = (
            np.maximum(0.0, hunger - 60.0) * 0.004
            + np.maximum(0.0, thirst - 60.0) * 0.005
        )

        per_live = base + food + drink_r + repro_r + engineer_bonus - threat_pen - condition_pen
        r[live] = per_live

        # Store breakdown.
        self._reward_components["base"][live] = base
        self._reward_components["food"][live] = food
        self._reward_components["drink"][live] = drink_r
        self._reward_components["repro"][live] = repro_r
        self._reward_components["engineer_bonus"][live] = engineer_bonus
        self._reward_components["threat"][live] = -threat_pen
        self._reward_components["condition"][live] = -condition_pen

        # Death penalty (energy<=0 OR exceeded maxAge).
        max_ages = np.array([SPECIES_PARAMS[i]["max_age"] for i in range(N_SPECIES)])
        died = (self.energy[live] <= 0.0) | (self.age[live] > max_ages[self.type[live]])
        died_slots = live[died]
        r[died_slots] -= _DEATH_PENALTY
        self._reward_components["death"][died_slots] = -_DEATH_PENALTY
        return r

    def reward_components(self) -> dict[str, np.ndarray]:
        """Per-slot reward breakdown from the most recent step. Read-only view."""
        return self._reward_components

    # ------------------------------ stats ----------------------------------
    def population_counts(self) -> dict[str, int]:
        out = {}
        for sp in range(N_SPECIES):
            out[SPECIES_NAMES[sp]] = int((self.alive & (self.type == sp)).sum())
        return out


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Run a random-action world for a few hundred ticks and print population.
    world = World(seed=1)
    print("initial:", world.population_counts())
    for t in range(800):
        for sp in range(N_SPECIES):
            obs, slots = world.observe(sp)
            if slots.size:
                actions = world.rng.integers(0, N_ACTIONS, size=slots.size)
                world.set_actions(sp, slots, actions)
        world.step_world()
        if t % 100 == 0:
            print(f"t={t:4d} pop={world.population_counts()}")
    print("final:", world.population_counts())
