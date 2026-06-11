"""PPO numerical-stability regression tests.

Covers the NaN that killed the back half of the 5000-update run: a
near-deterministic policy whose snapshot recorded an almost-zero probability
for a sampled action produces log_ratio ~ +100; exp(100) overflows float32 in
the torch backend -> inf -> NaN -> permanent weight corruption (first seen at
decomposer u=3875). The numpy backend survived only because its float64 + the
1e-12 log floor capped the ratio at ~1e12. The fix clamps the log-ratio before
exp in BOTH backends (PPO_LOGRATIO_CLAMP).

Run directly (no pytest needed):
    python training/tests/test_ppo_stability.py
or under pytest:
    pytest training/tests/test_ppo_stability.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_TRAIN_DIR = Path(__file__).resolve().parents[1]
if str(_TRAIN_DIR) not in sys.path:
    sys.path.insert(0, str(_TRAIN_DIR))

from multi_agent_policy import ActorCritic, TrainConfig, PPO_LOGRATIO_CLAMP  # noqa: E402

try:
    import torch  # noqa: F401
    from multi_agent_policy_torch import ActorCriticTorch
    _HAS_TORCH = True
except ImportError:  # numpy-only environment
    _HAS_TORCH = False


OBS, ACT, N = 28, 9, 64


def _favor_action0(policy, torch_backend: bool) -> None:
    """Force a near-deterministic policy that strongly favors action 0,
    regardless of input, by setting the policy-head bias to a big value and
    zeroing its weight."""
    bias = np.zeros(ACT)
    bias[0] = 100.0
    if torch_backend:
        import torch as _t
        with _t.no_grad():
            policy.policy_head.weight.zero_()
            policy.policy_head.bias.copy_(_t.tensor(bias, dtype=_t.float32))
    else:
        policy.W2 = np.zeros_like(policy.W2)
        policy.b2 = bias.copy()


def _weights_finite(policy, torch_backend: bool) -> bool:
    if torch_backend:
        import torch as _t
        return all(_t.isfinite(p).all().item() for p in policy.parameters())
    return all(
        np.isfinite(a).all()
        for a in (policy.W1, policy.b1, policy.W2, policy.b2, policy.Wv, np.array([policy.bv]))
    )


def _run_pathological(make_policy, torch_backend: bool) -> None:
    """The exact mid-training mismatch that produced the historical NaN: the
    snapshot (old_log_probs) assigned action 0 an almost-zero probability
    (log_prob = -100), but the current policy now strongly favors action 0.
    Without the clamp, ratio = exp(0 - (-100)) overflows float32."""
    rng = np.random.default_rng(0)
    obs = rng.standard_normal((N, OBS))
    acts = np.zeros(N, dtype=np.int64)  # chosen action 0
    old_log_probs = np.full(N, -100.0)
    adv = np.ones(N) * 2.0

    p = make_policy()
    _favor_action0(p, torch_backend)
    _, values = p.log_probs_and_values(obs, acts)
    returns_target = values + adv

    cfg = TrainConfig(hidden=8)
    m = p.update_ppo(obs, acts, old_log_probs, adv, returns_target, cfg, 0.01, target_kl=None)

    assert np.isfinite(m["value_loss"]), f"value_loss not finite: {m['value_loss']}"
    assert np.isfinite(m["entropy"]), f"entropy not finite: {m['entropy']}"
    assert _weights_finite(p, torch_backend), "weights went non-finite after pathological update"


def test_numpy_pathological_stays_finite():
    _run_pathological(lambda: ActorCritic(OBS, ACT, 8, seed=2), torch_backend=False)


def test_torch_pathological_stays_finite():
    if not _HAS_TORCH:
        print("  [skip] torch not installed")
        return
    _run_pathological(lambda: ActorCriticTorch(OBS, ACT, 8, seed=2, device="cpu"), torch_backend=True)


def test_clamp_is_noop_in_normal_regime():
    """On a healthy batch (ratios ~ 1) the clamp must not change the update, so
    the numpy and torch backends must still agree closely on the reported
    metrics. This guards against the clamp accidentally biting normal updates."""
    if not _HAS_TORCH:
        print("  [skip] torch not installed")
        return
    rng = np.random.default_rng(7)
    obs = rng.standard_normal((N, OBS))
    acts = rng.integers(0, ACT, size=N)
    cfg = TrainConfig(hidden=16)

    pn = ActorCritic(OBS, ACT, 16, seed=5)
    pt = ActorCriticTorch.from_dict(pn.to_dict(), seed=5, device="cpu")  # identical weights

    old_n, val_n = pn.log_probs_and_values(obs, acts)
    old_t, val_t = pt.log_probs_and_values(obs, acts)
    # Same starting weights => same old log-probs/values up to float tolerance.
    assert np.allclose(old_n, old_t, atol=1e-4), "backends disagree on old log-probs"

    adv = rng.standard_normal(N)
    ret = val_n + adv
    mn = pn.update_ppo(obs, acts, old_n, adv, ret, cfg, 0.01, target_kl=None)
    mt = pt.update_ppo(obs, acts, old_t, adv, ret, cfg, 0.01, target_kl=None)
    # First-epoch ratio is exactly 1 (no clamping); metrics should match well.
    assert abs(mn["entropy"] - mt["entropy"]) < 1e-2, (mn["entropy"], mt["entropy"])
    assert abs(mn["value_loss"] - mt["value_loss"]) < 1e-2, (mn["value_loss"], mt["value_loss"])


def test_clamp_constant_is_float32_safe():
    # exp(clamp) must not overflow float32 (max ~3.4e38).
    assert np.exp(PPO_LOGRATIO_CLAMP) < 3.0e38


def _main() -> int:
    tests = [
        test_clamp_constant_is_float32_safe,
        test_numpy_pathological_stays_finite,
        test_torch_pathological_stays_finite,
        test_clamp_is_noop_in_normal_regime,
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
