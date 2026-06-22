"""Bake the trained-policy behavior + the training story into compact JSON for
the browser demo (web/). No heavy env port needed in JS -- the browser just
plays back what the real Python simulator did under the trained policies.

Outputs (web/data/):
  playback_trained.json  - per-tick agent positions under the shipped 194.8 policies
  playback_random.json   - same env, random actions (the "before learning" contrast)
  training_curve.json     - real eval-score + population curves: the original run's
                            collapse vs a fixed run staying stable (the ML story)

Run:  python web/record_demo.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "training"))

from multi_agent_env import World, N_SPECIES, SPECIES_NAMES, GRID_W, GRID_H  # noqa: E402
from multi_agent_policy import ActorCritic  # noqa: E402

DATA = Path(__file__).resolve().parent / "data"
DATA.mkdir(parents=True, exist_ok=True)

TICKS = 600          # the eval-episode horizon
DEMO_SEED = 7        # scanned for the most balanced trained run (all 5 species persist)
POS_SCALE = 4        # quarter-cell position resolution (compact ints, still smooth)
VEG_EVERY = 12       # record a downsampled vegetation field every N ticks
VEG_W, VEG_H = 32, 24


def _load_policies():
    pols = []
    for sp in range(N_SPECIES):
        d = json.loads((_ROOT / "shipped_policies" / f"{SPECIES_NAMES[sp]}_policy.json").read_text())
        pols.append(ActorCritic.from_dict(d, seed=sp))
    return pols


def _downsample_veg(world: World) -> list[int]:
    """Vegetation field downsampled to VEG_W x VEG_H, quantized 0-9 for a tiny
    background tint in the browser."""
    veg = world.vegetation.reshape(GRID_H, GRID_W)
    cap = np.maximum(world.veg_cap.reshape(GRID_H, GRID_W), 1e-6)
    frac = np.clip(veg / cap, 0.0, 1.0)
    ys = (np.linspace(0, GRID_H - 1, VEG_H)).astype(int)
    xs = (np.linspace(0, GRID_W - 1, VEG_W)).astype(int)
    small = frac[np.ix_(ys, xs)]
    return [int(round(v * 9)) for v in small.flatten()]


def record(greedy: bool, seed: int = 3) -> dict:
    pols = _load_policies()
    world = World(seed=seed)
    rng = np.random.default_rng(seed + 99)

    ticks = []
    veg_frames = []
    pops_series = []
    for t in range(TICKS):
        for sp in range(N_SPECIES):
            obs, slots = world.observe(sp)
            if slots.size == 0:
                continue
            if greedy:
                acts, _, _, _ = pols[sp].act(obs, greedy=True)
            else:
                acts = rng.integers(0, 9, size=slots.size)
            world.set_actions(sp, slots, acts)
        world.step_world()

        alive = np.where(world.alive)[0]
        flat = []
        for s in alive:
            flat.append(int(world.type[s]))
            flat.append(int(round(float(world.x[s]) * POS_SCALE)))
            flat.append(int(round(float(world.y[s]) * POS_SCALE)))
        ticks.append(flat)  # flat [type,x,y, type,x,y, ...] keeps the file small

        pc = world.population_counts()
        pops_series.append([pc[SPECIES_NAMES[sp]] for sp in range(N_SPECIES)])
        if t % VEG_EVERY == 0:
            veg_frames.append(_downsample_veg(world))

    return {
        "grid": {"w": GRID_W, "h": GRID_H},
        "posScale": POS_SCALE,
        "species": SPECIES_NAMES,
        "ticks": ticks,
        "pops": pops_series,
        "veg": {"w": VEG_W, "h": VEG_H, "every": VEG_EVERY, "frames": veg_frames},
    }


def _curve_from_history(path: Path, step: int = 10) -> dict:
    h = json.loads(path.read_text())
    us, scores, best = [], [], []
    pops = {SPECIES_NAMES[sp]: [] for sp in range(N_SPECIES)}
    for rec in h[::step]:
        us.append(rec["update"])
        s = rec["eval_score"]
        scores.append(None if (isinstance(s, float) and s != s) else round(float(s), 1))
        b = rec["best_eval_score"]
        best.append(None if (isinstance(b, float) and b != b) else round(float(b), 1))
        for sp in range(N_SPECIES):
            v = rec["eval_avg_pop"][SPECIES_NAMES[sp]]
            pops[SPECIES_NAMES[sp]].append(round(float(v), 1))
    return {"updates": us, "eval": scores, "best": best, "pops": pops}


def main() -> None:
    print("recording trained playback...")
    (DATA / "playback_trained.json").write_text(json.dumps(record(greedy=True, seed=DEMO_SEED)))
    print("recording random (untrained) playback...")
    (DATA / "playback_random.json").write_text(json.dumps(record(greedy=False, seed=DEMO_SEED)))

    print("extracting training curves...")
    original = _ROOT / "shipped_policies" / "multi_agent_history.json"          # the collapse
    fixed = _ROOT / "shipped_policies" / "round4" / "seed_9710" / "multi_agent_history.json"  # stable to 5000
    curves = {
        "original": _curve_from_history(original),
        "fixed": _curve_from_history(fixed) if fixed.exists() else None,
    }
    (DATA / "training_curve.json").write_text(json.dumps(curves))

    for f in ("playback_trained.json", "playback_random.json", "training_curve.json"):
        kb = (DATA / f).stat().st_size / 1024
        print(f"  {f}: {kb:.0f} KB")
    print("done -> web/data/")


if __name__ == "__main__":
    main()
