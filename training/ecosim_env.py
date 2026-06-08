"""Headless ecosystem training environment.

Dependency-free so it runs immediately. It mirrors the browser simulator's key
ecological mechanics so a policy trained here faces the same pressures a real
herbivore does:

- renewable forage: vegetation regrows *logistically* toward a local carrying
  capacity, so over-grazing a patch has consequences (it does not just deplete),
- mobile, pursuing predators (not stationary hazards),
- Kleiber/Q10 metabolism: energy burn rises with body mass^0.75 and temperature,
- elevation, slope cost, roughness, and water drag on movement,
- hunger, thirst, starvation, and dehydration,
- fitness = lifetime reproductive success: a well-fed adult can reproduce, which
  is the real selection target, not mere survival time.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field


GRID_W = 96
GRID_H = 64
ACTIONS = ("stay", "n", "ne", "e", "se", "s", "sw", "w", "nw")

# Ecological constants (mirroring sim.js).
KLEIBER_EXP = 0.75
Q10 = 2.3
YEAR_STEPS = 600  # one seasonal cycle


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def smoothstep(t: float) -> float:
    return t * t * (3.0 - 2.0 * t)


def hash_noise(ix: int, iy: int, seed: int) -> float:
    value = math.sin(ix * 127.1 + iy * 311.7 + seed * 74.7) * 43758.5453123
    return value - math.floor(value)


def smooth_noise(x: float, y: float, scale: float, seed: int) -> float:
    sx = x / scale
    sy = y / scale
    x0 = math.floor(sx)
    y0 = math.floor(sy)
    tx = smoothstep(sx - x0)
    ty = smoothstep(sy - y0)
    a = hash_noise(x0, y0, seed)
    b = hash_noise(x0 + 1, y0, seed)
    c = hash_noise(x0, y0 + 1, seed)
    d = hash_noise(x0 + 1, y0 + 1, seed)
    return (a + (b - a) * tx) + ((c + (d - c) * tx) - (a + (b - a) * tx)) * ty


def fractal_noise(x: float, y: float, seed: int) -> float:
    return (
        smooth_noise(x, y, 30, seed) * 0.52
        + smooth_noise(x, y, 13, seed + 13) * 0.31
        + smooth_noise(x, y, 6, seed + 29) * 0.17
    )


@dataclass
class Cell:
    elevation: float
    roughness: float
    water: float
    food: float
    food_cap: float
    temp: float


@dataclass
class Predator:
    x: float
    y: float


@dataclass
class Animal:
    x: float
    y: float
    body: float = 1.0
    hunger: float = 25.0
    thirst: float = 25.0
    energy: float = 100.0
    age: int = 0
    offspring: int = 0
    breed_cooldown: int = 0


class EcosystemEnv:
    """Single-herbivore training environment with renewable forage, mobile
    predators, metabolic temperature dependence, and reproductive fitness."""

    # Tunables.
    BASE_SPEED = 0.85
    METABOLISM = 0.05
    PRED_SPEED = 0.55
    PRED_DETECT = 9.0
    PRED_CATCH = 1.3
    FOOD_REGROW = 0.035
    REPRO_ENERGY = 90.0
    REPRO_COST = 32.0
    REPRO_MIN_AGE = 40

    def __init__(self, seed: int = 1, max_steps: int = 600) -> None:
        self.seed = seed
        self.rng = random.Random(seed)
        self.max_steps = max_steps
        self.map_seed = seed * 9973 + 17
        self.cells: list[Cell] = []
        self.predators: list[Predator] = []
        self.animal = Animal(0.0, 0.0)
        self.steps = 0
        self.last_slope = 0.0
        self.last_terrain_factor = 1.0
        self.reset(seed)

    def reset(self, seed: int | None = None) -> list[float]:
        if seed is not None:
            self.seed = seed
            self.rng = random.Random(seed)
            self.map_seed = seed * 9973 + 17
        self.steps = 0
        self.last_slope = 0.0
        self.last_terrain_factor = 1.0
        self.cells = self._make_map()
        self.predators = self._spawn_predators(10)
        self.animal = self._spawn_herbivore()
        return self.observe()

    # --- environment dynamics ------------------------------------------------
    def season_sun(self) -> float:
        phase = (self.steps % YEAR_STEPS) / YEAR_STEPS
        # Smooth seasonal light/forage signal in [0.45, 1.0].
        return 0.72 + 0.28 * math.sin(phase * 2.0 * math.pi - math.pi / 2.0)

    def _regrow_food(self) -> None:
        sun = self.season_sun()
        rate = self.FOOD_REGROW * sun
        for c in self.cells:
            if c.food_cap <= 0.0:
                continue
            # Logistic regrowth toward the cell's carrying capacity.
            c.food = clamp(c.food + rate * c.food * (1.0 - c.food / c.food_cap) + 0.0008 * c.food_cap, 0.0, c.food_cap)

    def _move_predators(self) -> None:
        ax, ay = self.animal.x, self.animal.y
        for p in self.predators:
            dx = ax - p.x
            dy = ay - p.y
            dist = math.hypot(dx, dy)
            if dist < self.PRED_DETECT and dist > 1e-6:
                # Pursue the herbivore.
                p.x = clamp(p.x + (dx / dist) * self.PRED_SPEED, 0.0, GRID_W - 1.001)
                p.y = clamp(p.y + (dy / dist) * self.PRED_SPEED, 0.0, GRID_H - 1.001)
            else:
                # Random search walk.
                p.x = clamp(p.x + self.rng.uniform(-1.0, 1.0) * self.PRED_SPEED * 0.6, 0.0, GRID_W - 1.001)
                p.y = clamp(p.y + self.rng.uniform(-1.0, 1.0) * self.PRED_SPEED * 0.6, 0.0, GRID_H - 1.001)

    def step(self, action_index: int) -> tuple[list[float], float, bool, dict[str, float]]:
        action_index = max(0, min(len(ACTIONS) - 1, int(action_index)))
        dx, dy = self._action_delta(action_index)
        old_cell = self.cell_at(self.animal.x, self.animal.y)

        target_x = clamp(self.animal.x + dx * self.BASE_SPEED, 0.0, GRID_W - 1.001)
        target_y = clamp(self.animal.y + dy * self.BASE_SPEED, 0.0, GRID_H - 1.001)
        next_cell = self.cell_at(target_x, target_y)

        slope = next_cell.elevation - old_cell.elevation
        uphill = max(0.0, slope)
        downhill = max(0.0, -slope)
        roughness = (old_cell.roughness + next_cell.roughness) * 0.5
        water_drag = 0.22 if next_cell.water > 0.68 else 0.0
        terrain_factor = clamp(1.0 - uphill * 4.0 - roughness * 0.25 - water_drag + downhill * 0.7, 0.18, 1.18)
        effort = self.BASE_SPEED * (1.0 + uphill * 10.0 + roughness * 0.9 + water_drag)

        self.animal.x = clamp(self.animal.x + dx * self.BASE_SPEED * terrain_factor, 0.0, GRID_W - 1.001)
        self.animal.y = clamp(self.animal.y + dy * self.BASE_SPEED * terrain_factor, 0.0, GRID_H - 1.001)
        cell = self.cell_at(self.animal.x, self.animal.y)
        self.last_slope = slope
        self.last_terrain_factor = terrain_factor

        self.animal.age += 1
        self.steps += 1
        if self.animal.breed_cooldown > 0:
            self.animal.breed_cooldown -= 1

        # Renewable forage and mobile predators update each step.
        self._regrow_food()
        self._move_predators()

        # Kleiber + Q10 metabolism: basal burn scales with mass^0.75 and rises
        # with temperature above the comfort band.
        mass_metab = self.animal.body ** KLEIBER_EXP
        q10 = clamp(Q10 ** ((cell.temp - 20.0) / 10.0), 0.55, 2.6)
        basal = self.METABOLISM * mass_metab * q10

        self.animal.hunger = clamp(self.animal.hunger + 0.22 + effort * 0.24, 0.0, 140.0)
        self.animal.thirst = clamp(
            self.animal.thirst + 0.18 + effort * 0.22 + max(0.0, cell.temp - 22.0) * 0.015,
            0.0,
            150.0,
        )
        self.animal.energy -= basal + effort * 0.04 + uphill * 0.5

        ate = 0.0
        if cell.food > 0.08:
            ate = min(cell.food, 0.12 * self.animal.body)
            cell.food -= ate
            self.animal.hunger = clamp(self.animal.hunger - ate * 145.0, 0.0, 140.0)
            self.animal.energy = clamp(self.animal.energy + ate * 24.0, 0.0, 160.0)

        drank = 0.0
        if cell.water > 0.55:
            drank = min(cell.water, 0.18)
            self.animal.thirst = clamp(self.animal.thirst - drank * 95.0, 0.0, 150.0)

        predator_dist = self.nearest_predator_distance()
        predator_penalty = 0.0
        caught = False
        if predator_dist < self.PRED_CATCH:
            predator_penalty = 45.0
            self.animal.energy -= 80.0
            caught = self.animal.energy <= 0.0
        elif predator_dist < 5.0:
            predator_penalty = (5.0 - predator_dist) * 0.7

        starvation = max(0.0, self.animal.hunger - 92.0) * 0.035
        dehydration = max(0.0, self.animal.thirst - 86.0) * 0.045
        self.animal.energy -= starvation + dehydration

        # Reproduction = realized fitness. A well-fed, hydrated adult breeds.
        repro_reward = 0.0
        if (
            self.animal.energy >= self.REPRO_ENERGY
            and self.animal.age >= self.REPRO_MIN_AGE
            and self.animal.breed_cooldown == 0
            and self.animal.hunger < 55.0
            and self.animal.thirst < 60.0
        ):
            self.animal.energy -= self.REPRO_COST
            self.animal.offspring += 1
            self.animal.breed_cooldown = 35
            repro_reward = 12.0

        reward = (
            0.04
            + ate * 5.0
            + drank * 3.0
            + repro_reward
            - self.animal.hunger * 0.003
            - self.animal.thirst * 0.004
            - uphill * 0.22
            - roughness * 0.02
            - predator_penalty
        )
        if self.animal.energy <= 0.0:
            reward -= 35.0

        done = self.animal.energy <= 0.0 or self.steps >= self.max_steps
        info = {
            "energy": self.animal.energy,
            "hunger": self.animal.hunger,
            "thirst": self.animal.thirst,
            "slope": slope,
            "terrain_factor": terrain_factor,
            "predator_dist": predator_dist,
            "ate": ate,
            "drank": drank,
            "offspring": float(self.animal.offspring),
            "caught": 1.0 if caught else 0.0,
        }
        return self.observe(), reward, done, info

    def observe(self) -> list[float]:
        cell = self.cell_at(self.animal.x, self.animal.y)
        food_dx, food_dy, food_score = self._sense("food")
        water_dx, water_dy, water_score = self._sense("water")
        pred_dx, pred_dy, pred_score = self._sense_predator()
        repro_ready = 1.0 if (self.animal.energy >= self.REPRO_ENERGY and self.animal.breed_cooldown == 0) else 0.0
        return [
            self.animal.energy / 160.0,
            self.animal.hunger / 140.0,
            self.animal.thirst / 150.0,
            cell.food,
            cell.water,
            cell.elevation,
            cell.roughness,
            cell.temp / 38.0,
            self.last_slope,
            self.last_terrain_factor,
            food_dx,
            food_dy,
            food_score,
            water_dx,
            water_dy,
            water_score,
            pred_dx,
            pred_dy,
            pred_score,
            repro_ready,
            self.season_sun(),
        ]

    @property
    def observation_size(self) -> int:
        return len(self.observe())

    @property
    def action_size(self) -> int:
        return len(ACTIONS)

    def cell_at(self, x: float, y: float) -> Cell:
        ix = max(0, min(GRID_W - 1, int(x)))
        iy = max(0, min(GRID_H - 1, int(y)))
        return self.cells[iy * GRID_W + ix]

    def nearest_predator_distance(self) -> float:
        return min(math.hypot(p.x - self.animal.x, p.y - self.animal.y) for p in self.predators)

    def _make_map(self) -> list[Cell]:
        cells: list[Cell] = []
        for y in range(GRID_H):
            river_center = GRID_W * (0.5 + 0.22 * math.sin(y * 0.09 + self.map_seed * 0.001))
            for x in range(GRID_W):
                nx = x / (GRID_W - 1) - 0.5
                ny = y / (GRID_H - 1) - 0.5
                radial = math.sqrt(nx * nx + ny * ny)
                terrain = fractal_noise(x, y, self.map_seed)
                broad = fractal_noise(x * 0.5, y * 0.5, self.map_seed + 17)
                ridge_noise = fractal_noise(x * 1.7, y * 1.7, self.map_seed + 7)
                ridges = abs(ridge_noise - 0.5) * 2.0
                valley = math.exp(-((x - river_center) ** 2) / 26.0)
                elevation = clamp(0.5 + (terrain - 0.5) * 0.78 + (broad - 0.5) * 0.58 + ridges * 0.28 - radial * 0.22 - valley * 0.22, 0.0, 1.0)
                roughness = clamp(ridges * 0.62 + abs(terrain - broad) * 0.55 + elevation * 0.18, 0.0, 1.0)
                moisture = clamp(0.34 + valley * 0.65 + (fractal_noise(x, y, self.map_seed + 41) - 0.5) * 0.5 - elevation * 0.3, 0.0, 1.0)
                water = 1.0 if moisture > 0.73 or valley > 0.7 or elevation < 0.1 else moisture
                temp = 29.0 - (y / (GRID_H - 1)) * 16.0 - elevation * 8.0
                food_cap = clamp(0.08 + moisture * 0.6 - elevation * 0.18 + fractal_noise(x, y, self.map_seed + 99) * 0.25, 0.0, 1.0)
                if water > 0.82:
                    food_cap *= 0.5
                cells.append(Cell(elevation=elevation, roughness=roughness, water=water, food=food_cap, food_cap=food_cap, temp=temp))
        return cells

    def _spawn_herbivore(self) -> Animal:
        for _ in range(300):
            x = self.rng.uniform(0, GRID_W - 1)
            y = self.rng.uniform(0, GRID_H - 1)
            c = self.cell_at(x, y)
            if c.food > 0.45 and c.water < 0.85:
                return Animal(x=x, y=y)
        return Animal(x=GRID_W * 0.5, y=GRID_H * 0.5)

    def _spawn_predators(self, count: int) -> list[Predator]:
        predators: list[Predator] = []
        for _ in range(count):
            predators.append(Predator(self.rng.uniform(0, GRID_W - 1), self.rng.uniform(0, GRID_H - 1)))
        return predators

    def _action_delta(self, action: int) -> tuple[float, float]:
        if action == 0:
            return (0.0, 0.0)
        angle = (action - 1) * (math.pi / 4.0) - math.pi / 2.0
        return (math.cos(angle), math.sin(angle))

    def _sense(self, kind: str) -> tuple[float, float, float]:
        best_score = -1.0
        best_dx = 0.0
        best_dy = 0.0
        for i in range(16):
            angle = i * (math.pi * 2.0 / 16)
            dx = math.cos(angle)
            dy = math.sin(angle)
            for radius in (2, 4, 7, 10):
                c = self.cell_at(self.animal.x + dx * radius, self.animal.y + dy * radius)
                score = c.food if kind == "food" else c.water
                score -= c.elevation * 0.04
                if score > best_score:
                    best_score = score
                    best_dx = dx
                    best_dy = dy
        return (best_dx, best_dy, clamp(best_score, 0.0, 1.0))

    def _sense_predator(self) -> tuple[float, float, float]:
        best = (999.0, 0.0, 0.0)
        for p in self.predators:
            dist = math.hypot(p.x - self.animal.x, p.y - self.animal.y)
            if dist < best[0]:
                best = (dist, p.x, p.y)
        dist, px, py = best
        if dist <= 0.001:
            return (0.0, 0.0, 1.0)
        return ((px - self.animal.x) / dist, (py - self.animal.y) / dist, clamp(1.0 - dist / 14.0, 0.0, 1.0))
