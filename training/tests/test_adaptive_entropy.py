"""Tests for the adaptive entropy controller (_adapt_entropy_beta).

The round-2 3-seed run showed the sparse-reward foragers (predator, pollinator)
mode-collapse: entropy held ~1.0 then cliffed to ~0 while the population stayed
healthy, and a fixed entropy floor couldn't stop it. The adaptive controller
raises the entropy coefficient when entropy falls below a target and relaxes it
back toward the floor when entropy is comfortably above -- so it can't over-
explore early but clamps the late collapse.

Run directly:
    python training/tests/test_adaptive_entropy.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_TRAIN_DIR = Path(__file__).resolve().parents[1]
if str(_TRAIN_DIR) not in sys.path:
    sys.path.insert(0, str(_TRAIN_DIR))

from train_multi_agent import (  # noqa: E402
    _adapt_entropy_beta, ADAPTIVE_ENTROPY_CEILING, SPECIES_TUNING,
)
from multi_agent_env import PREDATOR, POLLINATOR  # noqa: E402

FLOOR = 0.02
TARGET = 0.7


def test_low_entropy_raises_beta():
    # Entropy well below target -> coefficient must increase.
    new = _adapt_entropy_beta(0.02, entropy=0.1, target=TARGET, floor=FLOOR)
    assert new > 0.02, new


def test_high_entropy_relaxes_toward_floor():
    # Entropy well above target -> coefficient must decrease.
    new = _adapt_entropy_beta(0.3, entropy=1.5, target=TARGET, floor=FLOOR)
    assert new < 0.3, new


def test_clamped_to_floor():
    # Even with entropy hugely above target, never drop below the floor.
    new = _adapt_entropy_beta(FLOOR, entropy=2.0, target=TARGET, floor=FLOOR)
    assert new >= FLOOR - 1e-12, new


def test_clamped_to_ceiling():
    # Even with entropy at zero forever, never exceed the ceiling (can't force
    # a uniform-random policy).
    beta = FLOOR
    for _ in range(200):
        beta = _adapt_entropy_beta(beta, entropy=0.0, target=TARGET, floor=FLOOR)
    assert beta <= ADAPTIVE_ENTROPY_CEILING + 1e-12, beta
    assert beta == ADAPTIVE_ENTROPY_CEILING, beta  # should have saturated


def test_converges_to_target_in_closed_loop():
    # Toy closed loop standing in for the real collapse regime: a policy whose
    # entropy RISES with the coefficient, H = H0 + c*log(beta) (more entropy
    # pressure -> more entropy), and whose natural entropy at the floor (0.10)
    # sits BELOW the target -- so the controller must ramp beta UP to hold the
    # line. Params place the fixed point at H=target with beta inside the range.
    import math
    H0, c = 1.27, 0.30
    beta = FLOOR
    H = H0 + c * math.log(beta)
    assert H < TARGET, "toy setup invalid: natural entropy already above target"
    for _ in range(300):
        H = H0 + c * math.log(beta)       # environment responds to the coefficient
        beta = _adapt_entropy_beta(beta, entropy=H, target=TARGET, floor=FLOOR)
    assert abs(H - TARGET) < 0.1, f"did not converge: H={H:.3f} beta={beta:.4f}"
    assert FLOOR < beta < ADAPTIVE_ENTROPY_CEILING, f"beta should settle in-range: {beta:.4f}"


def test_targets_are_set_for_sparse_foragers():
    # Guard the wiring: the two collapsing species must have a target.
    assert SPECIES_TUNING[PREDATOR].get("entropy_target") is not None
    assert SPECIES_TUNING[POLLINATOR].get("entropy_target") is not None


def _main() -> int:
    tests = [
        test_low_entropy_raises_beta,
        test_high_entropy_relaxes_toward_floor,
        test_clamped_to_floor,
        test_clamped_to_ceiling,
        test_converges_to_target_in_closed_loop,
        test_targets_are_set_for_sparse_foragers,
    ]
    failures = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except AssertionError as e:
            failures += 1
            print(f"FAIL  {t.__name__}: {e}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_main())
