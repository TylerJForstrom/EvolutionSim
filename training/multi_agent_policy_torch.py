"""PyTorch ActorCritic — drop-in alternative to the numpy ActorCritic.

Mathematically identical to multi_agent_policy.ActorCritic (one hidden layer,
tanh, softmax policy head, linear value head, PPO-clipped objective). The
point of having both is:

- numpy version: dependency-free (just numpy), small, runs anywhere, used
  inside the multiprocessing workers in episode_worker.py
- torch version: autograd-driven so we don't maintain the manual gradient
  math by hand AND it can run on a GPU. Same training loss, same diagnostics.

Compatibility design: the torch policy exposes `numpy_snapshot()` which
produces a dict in the same layout as
`multi_agent_policy.ActorCritic.to_dict()` (re-row-majoring the linear
weights so numpy's `obs @ W1` and torch's `nn.Linear` agree). That lets the
existing multiprocessing rollout workers — which keep using the numpy
ActorCritic for inference — be fed snapshots from a torch policy without
any changes to episode_worker.py.

On a CUDA device this is the unlock the project was missing: numpy never
hits the GPU at all.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:  # numpy-only environments stay usable
    torch = None  # type: ignore
    nn = None     # type: ignore
    F = None      # type: ignore
    TORCH_AVAILABLE = False


def torch_available() -> bool:
    return TORCH_AVAILABLE


def cuda_available() -> bool:
    return TORCH_AVAILABLE and torch.cuda.is_available()


# We reuse TrainConfig from the numpy module so the trainer doesn't care
# which backend it picks.
from multi_agent_policy import TrainConfig  # noqa: E402


class ActorCriticTorch(nn.Module if TORCH_AVAILABLE else object):
    """One-hidden-layer actor-critic implemented in PyTorch.

    Public surface deliberately mirrors numpy ActorCritic so the trainer is
    backend-agnostic:
        forward()            – not directly used by trainer
        act(obs_np)          – numpy in, numpy out (action, hidden, probs, value)
        log_probs_and_values(obs_np, actions_np) -> (log_probs_np, values_np)
        update_ppo(...)      – PPO-clipped objective, K epochs, autograd-driven
        numpy_snapshot()     – ndarray dict matching multi_agent_policy.to_dict()
                               so rollout workers can run inference via numpy.
        to_dict() / from_dict() – JSON-friendly inference checkpoint (same
                                  layout as numpy version)
        save_torch_state() / load_torch_state() – full resume state (.pt)
    """

    def __init__(self, obs_size: int, action_size: int, hidden: int,
                 seed: int = 0, device: str = "cpu") -> None:
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is not installed; use the numpy ActorCritic instead.")
        super().__init__()
        torch.manual_seed(seed)
        self.obs_size = int(obs_size)
        self.action_size = int(action_size)
        self.hidden_size = int(hidden)
        self.device = torch.device(device)

        self.l1 = nn.Linear(obs_size, hidden, bias=True)
        self.policy_head = nn.Linear(hidden, action_size, bias=True)
        self.value_head = nn.Linear(hidden, 1, bias=True)

        # Orthogonal init per the stable-baselines3 / spinningup PPO recipe.
        # Numpy backend uses an identical scheme so behaviour is comparable
        # across backends. The per-layer gains: hidden=1.0 (tanh-friendly),
        # policy head=0.01 (start near-uniform so PPO doesn't commit before
        # seeing data), value head=1.0.
        with torch.no_grad():
            nn.init.orthogonal_(self.l1.weight, gain=1.0)
            self.l1.bias.zero_()
            nn.init.orthogonal_(self.policy_head.weight, gain=0.01)
            self.policy_head.bias.zero_()
            nn.init.orthogonal_(self.value_head.weight, gain=1.0)
            self.value_head.bias.zero_()

        self.to(self.device)

        # Running return statistics for value-target normalization. Same
        # semantics as numpy version.
        self.ret_mean = 0.0
        self.ret_var = 1.0
        self.ret_count = 1e-4

        # Lazily created Adam optimizer (built on first update so we can pick
        # up the cfg.lr from the call site).
        self._optim: "torch.optim.Adam | None" = None

        # CPU-side numpy generator for action sampling parity with the numpy
        # backend's act(). Sampling on GPU is fine functionally but a numpy
        # generator keeps action-sampling reproducibility across backends.
        self._action_rng = np.random.default_rng(seed)

    # ------------------------------ forward --------------------------------
    def forward(self, obs: "torch.Tensor"):
        h = torch.tanh(self.l1(obs))
        logits = self.policy_head(h)
        values = self.value_head(h).squeeze(-1)
        return h, logits, values

    # ----------------------- numpy <-> torch boundary ----------------------
    def _to_t(self, arr: np.ndarray, dtype=None):
        t = torch.from_numpy(np.asarray(arr))
        if dtype is not None:
            t = t.to(dtype)
        return t.to(self.device, non_blocking=True)

    def act(self, obs_np: np.ndarray, greedy: bool = False):
        """Numpy-in / numpy-out so the env doesn't care about backend."""
        N = obs_np.shape[0]
        if N == 0:
            return (np.zeros(0, dtype=np.int32),
                    np.zeros((0, self.hidden_size)),
                    np.zeros((0, self.action_size)),
                    np.zeros(0))
        with torch.no_grad():
            obs = self._to_t(obs_np, torch.float32)
            h, logits, values = self.forward(obs)
            probs = F.softmax(logits, dim=-1)
            probs_np = probs.cpu().numpy()
        if greedy:
            actions = probs_np.argmax(axis=1).astype(np.int32)
        else:
            cum = probs_np.cumsum(axis=1)
            rolls = self._action_rng.random((N, 1))
            actions = (cum < rolls).sum(axis=1).astype(np.int32)
            actions = np.clip(actions, 0, self.action_size - 1)
        return (actions,
                h.cpu().numpy(),
                probs_np,
                values.cpu().numpy())

    def log_probs_and_values(self, obs_np: np.ndarray, actions_np: np.ndarray):
        if obs_np.shape[0] == 0:
            return np.zeros(0), np.zeros(0)
        with torch.no_grad():
            obs = self._to_t(obs_np, torch.float32)
            actions = self._to_t(actions_np, torch.long)
            _h, logits, values = self.forward(obs)
            log_probs = F.log_softmax(logits, dim=-1)
            chosen = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
        return chosen.cpu().numpy(), values.cpu().numpy()

    # ---------------------- return stats (numpy parity) --------------------
    def update_ret_stats(self, returns_np: np.ndarray) -> None:
        for g in returns_np:
            self.ret_count += 1.0
            delta = float(g) - self.ret_mean
            self.ret_mean += delta / self.ret_count
            self.ret_var += delta * (float(g) - self.ret_mean)

    def ret_std(self) -> float:
        var = max(self.ret_var / max(1.0, self.ret_count), 1e-6)
        return math.sqrt(var) + 1e-6

    # ------------------------------ PPO update -----------------------------
    def update_ppo(self, obs_np: np.ndarray, actions_np: np.ndarray,
                   old_log_probs_np: np.ndarray, advantages_np: np.ndarray,
                   returns_target_np: np.ndarray, cfg: TrainConfig,
                   entropy_beta: float,
                   *,
                   lr_override: float | None = None,
                   ppo_clip_override: float | None = None,
                   target_kl: float | None = None) -> dict:
        """Per-species overrides match the numpy backend exactly:
        - lr_override / ppo_clip_override let the trainer dial individual
          species without mutating the shared TrainConfig
        - target_kl enables early-stopping the PPO epoch loop when the
          approximate KL exceeds 1.5x the target (standard PPO refinement)"""
        if obs_np.shape[0] == 0:
            return {"entropy": 0.0, "value_loss": 0.0, "n": 0, "kl": 0.0, "clip_frac": 0.0, "ppo_epochs_run": 0}

        lr = cfg.lr if lr_override is None else lr_override
        clip_eps = cfg.ppo_clip if ppo_clip_override is None else ppo_clip_override

        self.update_ret_stats(returns_target_np)
        rmean = self.ret_mean
        rstd = self.ret_std()
        rn_np = (returns_target_np - rmean) / rstd

        adv_np = advantages_np
        if adv_np.size > 1:
            adv_np = (adv_np - adv_np.mean()) / (adv_np.std() + 1e-6)

        obs = self._to_t(obs_np, torch.float32)
        actions = self._to_t(actions_np, torch.long)
        old_log_probs = self._to_t(old_log_probs_np, torch.float32)
        advantages = self._to_t(adv_np, torch.float32)
        returns_n = self._to_t(rn_np, torch.float32)

        if self._optim is None:
            self._optim = torch.optim.Adam(self.parameters(), lr=lr,
                                           betas=(0.9, 0.999), eps=1e-8)
        # Keep lr current per-species (overrides take precedence over cfg.lr).
        for g in self._optim.param_groups:
            g["lr"] = lr

        last = {"entropy": 0.0, "value_loss": 0.0, "kl": 0.0, "clip_frac": 0.0, "ppo_epochs_run": 0}

        for epoch_idx in range(cfg.ppo_epochs):
            _h, logits, values = self.forward(obs)
            log_probs_all = F.log_softmax(logits, dim=-1)
            new_log_probs = log_probs_all.gather(1, actions.unsqueeze(1)).squeeze(1)

            ratio = (new_log_probs - old_log_probs).exp()

            # Early-stop the PPO epoch loop if approximate KL has drifted too
            # far from the snapshot policy. Standard PPO refinement.
            with torch.no_grad():
                approx_kl_t = (old_log_probs - new_log_probs).mean()
            approx_kl = float(approx_kl_t.item())
            if target_kl is not None and approx_kl > 1.5 * target_kl:
                last["ppo_epochs_run"] = epoch_idx
                break

            unclipped = ratio * advantages
            clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
            policy_loss = -torch.min(unclipped, clipped).mean()

            value_loss = 0.5 * (values - returns_n).pow(2).mean()

            probs = log_probs_all.exp()
            entropy = -(probs * log_probs_all).sum(dim=-1).mean()

            total_loss = policy_loss + cfg.value_coef * value_loss - entropy_beta * entropy

            self._optim.zero_grad()
            total_loss.backward()
            if cfg.grad_clip > 0:
                nn.utils.clip_grad_norm_(self.parameters(), cfg.grad_clip)
            self._optim.step()

            with torch.no_grad():
                clip_frac = ((ratio - 1.0).abs() > clip_eps).float().mean().item()
                last["entropy"] = entropy.item()
                last["value_loss"] = value_loss.item()
                last["kl"] = approx_kl
                last["clip_frac"] = clip_frac
                last["ppo_epochs_run"] = epoch_idx + 1

        last["n"] = int(obs.shape[0])
        return last

    # --------------- snapshot for multiprocessing workers ------------------
    def numpy_snapshot(self) -> dict:
        """Compact pickleable dict matching the numpy backend's layout.

        nn.Linear stores weight as (out, in); the numpy ActorCritic uses
        (in, out) because it does `obs @ W1`. We transpose on the way out
        so workers can load and run inference unchanged.
        """
        with torch.no_grad():
            return {
                "obs_size": self.obs_size,
                "action_size": self.action_size,
                "hidden_size": self.hidden_size,
                "W1": self.l1.weight.detach().cpu().numpy().T.copy(),
                "b1": self.l1.bias.detach().cpu().numpy().copy(),
                "W2": self.policy_head.weight.detach().cpu().numpy().T.copy(),
                "b2": self.policy_head.bias.detach().cpu().numpy().copy(),
                "Wv": self.value_head.weight.detach().cpu().numpy().flatten().copy(),
                "bv": float(self.value_head.bias.detach().cpu().item()),
            }

    # ----------------------- JSON inference checkpoint ---------------------
    def to_dict(self) -> dict:
        """Inference-only checkpoint in the SAME layout as numpy ActorCritic
        so existing tools (load via `multi_agent_policy.ActorCritic.from_dict`)
        keep working regardless of which backend trained the policy."""
        snap = self.numpy_snapshot()
        return {
            "obs_size": snap["obs_size"],
            "action_size": snap["action_size"],
            "hidden_size": snap["hidden_size"],
            "W1": snap["W1"].tolist(),
            "b1": snap["b1"].tolist(),
            "W2": snap["W2"].tolist(),
            "b2": snap["b2"].tolist(),
            "Wv": snap["Wv"].tolist(),
            "bv": snap["bv"],
        }

    @classmethod
    def from_dict(cls, d: dict, seed: int = 0, device: str = "cpu") -> "ActorCriticTorch":
        p = cls(d["obs_size"], d["action_size"], d["hidden_size"], seed=seed, device=device)
        with torch.no_grad():
            # Reverse the snapshot transpose: incoming W1 is (in, out), Linear
            # wants (out, in).
            p.l1.weight.copy_(torch.tensor(d["W1"], dtype=torch.float32).t())
            p.l1.bias.copy_(torch.tensor(d["b1"], dtype=torch.float32))
            p.policy_head.weight.copy_(torch.tensor(d["W2"], dtype=torch.float32).t())
            p.policy_head.bias.copy_(torch.tensor(d["b2"], dtype=torch.float32))
            p.value_head.weight.copy_(torch.tensor(d["Wv"], dtype=torch.float32).unsqueeze(0))
            p.value_head.bias.fill_(float(d["bv"]))
        p.to(p.device)
        return p

    # --------------- full state save/load for resume -----------------------
    def save_torch_state(self, path: Path) -> None:
        """Full state including Adam moments. Used for `--resume` when the
        torch backend is active. Single .pt file per species."""
        payload = {
            "model_state": self.state_dict(),
            "optim_state": self._optim.state_dict() if self._optim is not None else None,
            "ret_mean": self.ret_mean,
            "ret_var": self.ret_var,
            "ret_count": self.ret_count,
            "obs_size": self.obs_size,
            "action_size": self.action_size,
            "hidden_size": self.hidden_size,
        }
        torch.save(payload, str(path))

    def load_torch_state(self, path: Path) -> None:
        payload = torch.load(str(path), map_location=self.device)
        self.load_state_dict(payload["model_state"])
        if payload.get("optim_state") is not None:
            if self._optim is None:
                self._optim = torch.optim.Adam(self.parameters())
            self._optim.load_state_dict(payload["optim_state"])
        self.ret_mean = float(payload.get("ret_mean", 0.0))
        self.ret_var = float(payload.get("ret_var", 1.0))
        self.ret_count = float(payload.get("ret_count", 1e-4))
