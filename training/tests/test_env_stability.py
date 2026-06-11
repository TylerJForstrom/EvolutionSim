"""Global carrying-capacity stress test for the ecosystem env (no RL, ~seconds).

The training collapse happens late (u~2000), so a short RL smoke can't prove
stability. The failure is structural: once predators thin out, the herbivore
guild's reproduction is limited only by the per-cell local cap and the toothless
global max_count (420), so a population spread across the 64x48 map runs away
(eval herb hit ~67), overgrazes, and boom-bust-crashes the food web.

The full explosion is emergent from five co-trained policies and can't be
reproduced by hand-scripting one guild. But the *mechanism* we changed — the
global density-dependent reproduction throttle — is testable directly and
deterministically: drive herbivores at maximal breeding pressure (well-fed,
hydrated, of age, spread out so the per-cell local cap stays slack) and confirm
that the new global carrying capacity (carry_k) bounds the population where the
old max_count throttle let it run to the safety rail.

Run directly:
    python training/tests/test_env_stability.py
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np

_TRAIN_DIR = Path(__file__).resolve().parents[1]
if str(_TRAIN_DIR) not in sys.path:
    sys.path.insert(0, str(_TRAIN_DIR))

import multi_agent_env as E  # noqa: E402
from multi_agent_env import HERBIVORE, PREDATOR, N_SPECIES, World  # noqa: E402


@contextmanager
def _herb_carry_k(value):
    p = E.SPECIES_PARAMS[HERBIVORE]
    saved = p.get("carry_k", p["max_count"])
    p["carry_k"] = value
    try:
        yield
    finally:
        p["carry_k"] = saved


def run_max_pressure(herb_carry: int, ticks: int = 3000, seed: int = 3, init_herb: int = 5) -> dict:
    """Grow herbivores from a low start under maximal breeding pressure with no
    predators. Each tick we top up herbivore energy/hydration/age so the
    condition gates always pass; movement is random so they disperse and the
    per-cell local cap stays slack — leaving the GLOBAL carry_k throttle (plus
    the post-birth cooldown) as the only thing limiting growth."""
    with _herb_carry_k(herb_carry):
        w = World(seed=seed)
        w.alive[w.type == PREDATOR] = False
        herb_slots = np.where(w.alive & (w.type == HERBIVORE))[0]
        if herb_slots.size > init_herb:
            w.alive[herb_slots[init_herb:]] = False

        peak = 0
        series = []
        for t in range(ticks):
            hm = w.alive & (w.type == HERBIVORE)
            w.energy[hm] = 160.0
            w.hunger[hm] = 0.0
            w.thirst[hm] = 0.0
            w.age[hm] = np.maximum(w.age[hm], 70)
            for sp in range(N_SPECIES):
                _obs, slots = w.observe(sp)
                if slots.size == 0:
                    continue
                w.set_actions(sp, slots, w.rng.integers(0, 9, size=slots.size))
            w.step_world()
            w.alive[w.type == PREDATOR] = False  # keep the predator rescue from refilling
            hc = int((w.alive & (w.type == HERBIVORE)).sum())
            peak = max(peak, hc)
            if t % 300 == 0:
                series.append((t, hc))
        final = int((w.alive & (w.type == HERBIVORE)).sum())
        return {"herb_carry": herb_carry, "peak": peak, "final": final, "series": series}


def test_global_cap_bounds_population():
    """Under maximal breeding pressure, the capped herbivore population must
    plateau near carry_k (not run away) and not crash to extinction."""
    carry = E.SPECIES_PARAMS[HERBIVORE]["carry_k"]
    r = run_max_pressure(carry)
    assert r["peak"] <= carry + 4, f"population overshot carry_k={carry}: peak={r['peak']}"
    assert r["final"] >= carry // 2, f"population didn't sustain near carry_k: final={r['final']}"


def test_cap_is_load_bearing():
    """Control: the SAME scenario with the cap disabled (carry_k = max_count, the
    historical behavior) must run far past carry_k toward the safety rail. This
    proves the carry_k throttle — not some other change — bounds the population."""
    carry = E.SPECIES_PARAMS[HERBIVORE]["carry_k"]
    maxc = E.SPECIES_PARAMS[HERBIVORE]["max_count"]
    uncapped = run_max_pressure(maxc)
    assert uncapped["peak"] > carry * 3, (
        f"uncapped run failed to demonstrate the runaway: peak={uncapped['peak']} "
        f"(expected >> carry_k={carry})"
    )


def test_cap_is_noop_in_healthy_regime():
    """The trained healthy herbivore level is ~1-3 (predators suppress them). The
    cap must barely touch reproduction there, so it can't perturb the good
    equilibrium. cap_pressure = 1 - n/carry_k must stay > 0.9 for n <= 3."""
    carry = E.SPECIES_PARAMS[HERBIVORE]["carry_k"]
    for n in (1, 2, 3):
        cap_pressure = max(0.0, 1.0 - n / carry)
        assert cap_pressure > 0.9, f"cap throttles the healthy regime at n={n}: {cap_pressure:.3f}"


def _main() -> int:
    carry = E.SPECIES_PARAMS[HERBIVORE]["carry_k"]
    maxc = E.SPECIES_PARAMS[HERBIVORE]["max_count"]
    print(f"herbivore carry_k={carry}  max_count={maxc}\n")
    for label, k in [(f"UNCAPPED (historical, carry_k={maxc})", maxc), (f"CAPPED (carry_k={carry})", carry)]:
        r = run_max_pressure(k)
        print(f"== {label} ==")
        print(f"   peak={r['peak']}  final={r['final']}  herb(t)={r['series']}")
        print()
    failures = 0
    for t in (test_global_cap_bounds_population, test_cap_is_load_bearing, test_cap_is_noop_in_healthy_regime):
        try:
            t()
            print(f"PASS  {t.__name__}")
        except AssertionError as e:
            failures += 1
            print(f"FAIL  {t.__name__}: {e}")
    print(f"\n{3 - failures}/3 passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_main())
