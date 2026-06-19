"""Tests for the optional predator LR decay schedule (_decayed_lr).

Peak-chasing config: a high lr early reaches a high peak fast (the runs that hit
184.7 used lr~0.005), then decaying toward a low lr keeps the policy from
over-sharpening into the late collapse a constant high lr suffers. Decay is
opt-in (only when lr_final is set); otherwise the lr is the validated constant.

Run directly:
    python training/tests/test_lr_decay.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_TRAIN_DIR = Path(__file__).resolve().parents[1]
if str(_TRAIN_DIR) not in sys.path:
    sys.path.insert(0, str(_TRAIN_DIR))

from train_multi_agent import _decayed_lr, SPECIES_TUNING  # noqa: E402
from multi_agent_env import PREDATOR  # noqa: E402


def test_start_at_frac_zero():
    assert _decayed_lr(0.005, 0.0015, 0.0) == 0.005


def test_end_at_frac_one():
    assert abs(_decayed_lr(0.005, 0.0015, 1.0) - 0.0015) < 1e-12


def test_midpoint():
    assert abs(_decayed_lr(0.005, 0.0015, 0.5) - 0.00325) < 1e-12


def test_monotonic_decreasing():
    vals = [_decayed_lr(0.005, 0.0015, f / 20.0) for f in range(21)]
    assert all(vals[i] >= vals[i + 1] for i in range(len(vals) - 1)), vals


def test_default_predator_has_no_decay():
    # The validated default is a constant lr (no lr_final). Decay must be opt-in
    # so merging to main doesn't silently change training behavior.
    assert "lr_final" not in SPECIES_TUNING[PREDATOR], (
        "predator should default to a constant lr; decay is opt-in via --pred-lr-final"
    )


def _main() -> int:
    tests = [
        test_start_at_frac_zero,
        test_end_at_frac_one,
        test_midpoint,
        test_monotonic_decreasing,
        test_default_predator_has_no_decay,
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
