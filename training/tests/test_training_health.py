"""Ground-truth tests for the training-health diagnostic.

Validates the verdict against the shipped 5000-update history, whose trajectory
we understand exactly: healthy and improving through u~501 (peak eval=194.8),
predator-entropy mode-collapse by u~1650, NaN by u~3875, dead by u=5000.

Run directly:
    python training/tests/test_training_health.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_TRAIN_DIR = Path(__file__).resolve().parents[1]
_REPO = _TRAIN_DIR.parent
if str(_TRAIN_DIR) not in sys.path:
    sys.path.insert(0, str(_TRAIN_DIR))

from training_health import assess, load_history  # noqa: E402

_SHIPPED = _REPO / "shipped_policies" / "multi_agent_history.json"


def _history():
    return load_history(_SHIPPED)


def test_full_collapsed_run_flagged_crit():
    r = assess(_history())
    assert r["level"] == "crit", r["verdict"]
    names = {c["name"] for c in r["checks"] if c["level"] == "crit"}
    assert "nan" in names, "should flag NaN corruption in the dead run"


def test_healthy_peak_slice_is_ok_and_improving():
    r = assess(_history()[:501])
    assert r["level"] == "ok", r["verdict"]
    assert r["direction"] == "improving", r["verdict"]


def test_mid_collapse_caught_before_nan():
    # By u~1700 the predator policy has mode-collapsed (entropy 0.000) but the
    # NaN doesn't appear until u~3875 -- the detector must catch the collapse
    # from the entropy signal, well before the NaN.
    r = assess(_history()[:1700])
    assert r["level"] == "crit", r["verdict"]
    crit = {c["name"] for c in r["checks"] if c["level"] == "crit"}
    assert "predator_entropy" in crit, "should catch predator mode-collapse pre-NaN"
    assert "nan" not in crit, "there is no NaN yet at u=1700"


def test_very_early_slice_is_early():
    r = assess(_history()[:80])
    assert r["level"] == "early", r["verdict"]


def test_empty_history_is_handled():
    r = assess([])
    assert r["level"] == "crit"


def _main() -> int:
    tests = [
        test_full_collapsed_run_flagged_crit,
        test_healthy_peak_slice_is_ok_and_improving,
        test_mid_collapse_caught_before_nan,
        test_very_early_slice_is_early,
        test_empty_history_is_handled,
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
