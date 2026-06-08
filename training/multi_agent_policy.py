"""Per-species actor-critic policies, numpy-vectorized.

One actor-critic network per species. All alive agents of a species evaluate
their policy in a single batched forward pass per tick. Backprop is also
batched over a buffer of all transitions collected from that species across an
episode (so a 600-agent herbivore guild contributes a much bigger gradient
signal than a 15-agent engineer guild — which is fine; rare species update
less often, just like real evolution).

The network is a one-hidden-layer MLP with tanh activations, a softmax policy
head, and a linear value head sharing the hidden features. Adam optimizer.
Mathematically identical to training/train_policy.py — this file just
generalizes it to per-species batched updates and uses numpy.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass

import numpy as np


@dataclass
class TrainConfig:
    lr: float = 0.005
    gamma: float = 0.99
    entropy_beta_init: float = 0.015
    entropy_beta_final: float = 0.001
    value_coef: float = 0.5
    grad_clip: float = 5.0
    hidden: int = 64


class ActorCritic:
    """One-hidden-layer actor-critic with a shared trunk.

    Shapes:
        W1: (obs, hidden), b1: (hidden,)
        W2: (hidden, actions), b2: (actions,)   # policy head
        Wv: (hidden,), bv: scalar               # value head
    """

    def __init__(self, obs_size: int, action_size: int, hidden: int, seed: int) -> None:
        self.obs_size = obs_size
        self.action_size = action_size
        self.hidden_size = hidden
        self.rng = np.random.default_rng(seed)
        scale1 = 1.0 / math.sqrt(obs_size)
        scale2 = 1.0 / math.sqrt(hidden)
        self.W1 = self.rng.uniform(-scale1, scale1, size=(obs_size, hidden))
        self.b1 = np.zeros(hidden)
        self.W2 = self.rng.uniform(-scale2, scale2, size=(hidden, action_size))
        self.b2 = np.zeros(action_size)
        self.Wv = self.rng.uniform(-scale2, scale2, size=hidden)
        self.bv = 0.0
        # Running return statistics (for normalizing value targets).
        self.ret_mean = 0.0
        self.ret_var = 1.0
        self.ret_count = 1e-4
        # Adam state.
        self.adam_t = 0
        self._init_adam()

    def _init_adam(self) -> None:
        self.mW1 = np.zeros_like(self.W1); self.vW1 = np.zeros_like(self.W1)
        self.mb1 = np.zeros_like(self.b1); self.vb1 = np.zeros_like(self.b1)
        self.mW2 = np.zeros_like(self.W2); self.vW2 = np.zeros_like(self.W2)
        self.mb2 = np.zeros_like(self.b2); self.vb2 = np.zeros_like(self.b2)
        self.mWv = np.zeros_like(self.Wv); self.vWv = np.zeros_like(self.Wv)
        self.mbv = 0.0; self.vbv = 0.0

    # ----------------------------- forward ----------------------------------
    def forward(self, obs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Batched forward pass.

        obs: (N, obs_size)  →  (hidden (N, H), probs (N, A), value (N,))
        """
        if obs.shape[0] == 0:
            H = self.hidden_size; A = self.action_size
            return np.zeros((0, H)), np.zeros((0, A)), np.zeros(0)
        z1 = obs @ self.W1 + self.b1
        hidden = np.tanh(z1)
        logits = hidden @ self.W2 + self.b2
        # Numerically stable softmax.
        logits = logits - logits.max(axis=1, keepdims=True)
        exps = np.exp(logits)
        probs = exps / exps.sum(axis=1, keepdims=True)
        value = hidden @ self.Wv + self.bv
        return hidden, probs, value

    def act(self, obs: np.ndarray, greedy: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        hidden, probs, value = self.forward(obs)
        if hidden.shape[0] == 0:
            return np.zeros(0, dtype=np.int32), hidden, probs, value
        if greedy:
            actions = probs.argmax(axis=1).astype(np.int32)
        else:
            # Sample per row from each row's categorical distribution.
            cum = probs.cumsum(axis=1)
            rolls = self.rng.random(probs.shape[0])[:, None]
            actions = (cum < rolls).sum(axis=1).astype(np.int32)
            actions = np.clip(actions, 0, self.action_size - 1)
        return actions, hidden, probs, value

    # ---------------------- return statistics -------------------------------
    def update_ret_stats(self, returns: np.ndarray) -> None:
        # Welford running mean/variance.
        for g in returns:
            self.ret_count += 1.0
            delta = g - self.ret_mean
            self.ret_mean += delta / self.ret_count
            self.ret_var += delta * (g - self.ret_mean)

    def ret_std(self) -> float:
        var = max(self.ret_var / max(1.0, self.ret_count), 1e-6)
        return math.sqrt(var) + 1e-6

    # ----------------------------- update ----------------------------------
    def update(
        self,
        obs: np.ndarray,        # (N, obs)
        actions: np.ndarray,    # (N,)
        returns: np.ndarray,    # (N,) raw returns
        cfg: TrainConfig,
        entropy_beta: float,
    ) -> dict:
        """One Adam step on the actor-critic objective for this species.

        Returns diagnostic metrics: entropy, value loss, mean advantage.
        """
        if obs.shape[0] == 0:
            return {"entropy": 0.0, "value_loss": 0.0, "n": 0}

        # Normalize returns for value target stability.
        self.update_ret_stats(returns)
        rmean = self.ret_mean
        rstd = self.ret_std()
        rn = (returns - rmean) / rstd

        hidden, probs, value = self.forward(obs)
        N, A = probs.shape

        # Advantages: normalized return - value head. Then center+scale per
        # batch for low-variance policy gradient.
        adv = rn - value
        if adv.size > 1:
            adv = (adv - adv.mean()) / (adv.std() + 1e-6)

        # Policy gradient on logits: A * (1{a=action} - p_a)
        onehot = np.zeros_like(probs)
        onehot[np.arange(N), actions] = 1.0
        g_logits = adv[:, None] * (onehot - probs)

        # Entropy gradient on logits: -p_a * (log p_a + H_row), broadcast row-wise.
        log_p = np.log(probs + 1e-12)
        H_row = -(probs * log_p).sum(axis=1, keepdims=True)  # (N, 1)
        ent_grad = -probs * (log_p + H_row)
        g_logits = g_logits + entropy_beta * ent_grad

        # ---- Output layer (policy head) gradients
        # logits = hidden @ W2 + b2
        gW2 = hidden.T @ g_logits / N
        gb2 = g_logits.mean(axis=0)
        # Backprop into hidden through W2: (N, A) @ (A, H) -> (N, H)
        dh = g_logits @ self.W2.T

        # ---- Value head: loss = 0.5 * (value - rn)^2; ascent dir = -(value - rn)
        v_err = value - rn
        gWv = -cfg.value_coef * (v_err[:, None] * hidden).mean(axis=0)
        gbv = -cfg.value_coef * v_err.mean()
        # Value gradient also shapes hidden features.
        dh += -cfg.value_coef * v_err[:, None] * self.Wv[None, :]

        # ---- Through tanh and first layer
        dz1 = dh * (1.0 - hidden * hidden)
        gW1 = obs.T @ dz1 / N
        gb1 = dz1.mean(axis=0)

        # ---- Global grad-norm clip
        grads = [gW1, gb1, gW2, gb2, gWv, np.array([gbv])]
        norm = math.sqrt(sum(float((g * g).sum()) for g in grads))
        clip = min(1.0, cfg.grad_clip / (norm + 1e-12)) if cfg.grad_clip > 0 else 1.0
        gW1 *= clip; gb1 *= clip; gW2 *= clip; gb2 *= clip; gWv *= clip; gbv *= clip

        # ---- Adam step (ascent on the combined objective)
        self.adam_t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        bc1 = 1.0 - b1 ** self.adam_t
        bc2 = 1.0 - b2 ** self.adam_t

        def _adam(p, m, v, g):
            m[...] = b1 * m + (1.0 - b1) * g
            v[...] = b2 * v + (1.0 - b2) * g * g
            p[...] += cfg.lr * (m / bc1) / (np.sqrt(v / bc2) + eps)

        _adam(self.W1, self.mW1, self.vW1, gW1)
        _adam(self.b1, self.mb1, self.vb1, gb1)
        _adam(self.W2, self.mW2, self.vW2, gW2)
        _adam(self.b2, self.mb2, self.vb2, gb2)
        _adam(self.Wv, self.mWv, self.vWv, gWv)
        # Scalar bv.
        self.mbv = b1 * self.mbv + (1.0 - b1) * gbv
        self.vbv = b2 * self.vbv + (1.0 - b2) * gbv * gbv
        self.bv += cfg.lr * (self.mbv / bc1) / (math.sqrt(self.vbv / bc2) + eps)

        # Diagnostics.
        entropy_mean = float(H_row.mean())
        value_loss = float(0.5 * (v_err * v_err).mean())
        return {"entropy": entropy_mean, "value_loss": value_loss, "n": int(N)}

    # -------------------------- serialization -------------------------------
    def to_dict(self) -> dict:
        return {
            "obs_size": int(self.obs_size),
            "action_size": int(self.action_size),
            "hidden_size": int(self.hidden_size),
            "W1": self.W1.tolist(),
            "b1": self.b1.tolist(),
            "W2": self.W2.tolist(),
            "b2": self.b2.tolist(),
            "Wv": self.Wv.tolist(),
            "bv": float(self.bv),
        }

    @classmethod
    def from_dict(cls, d: dict, seed: int = 0) -> "ActorCritic":
        p = cls(d["obs_size"], d["action_size"], d["hidden_size"], seed)
        p.W1 = np.array(d["W1"]); p.b1 = np.array(d["b1"])
        p.W2 = np.array(d["W2"]); p.b2 = np.array(d["b2"])
        p.Wv = np.array(d["Wv"]); p.bv = float(d["bv"])
        return p

    # ----------------------- full save/load (resume) ------------------------
    # `to_dict` writes inference-only weights; for resuming training we also
    # need the Adam moment estimates, the running return statistics, and the
    # Adam step counter. We keep those in a parallel npz file rather than
    # bloating the JSON.
    def adam_state(self) -> dict[str, np.ndarray | float | int]:
        return {
            "mW1": self.mW1, "vW1": self.vW1,
            "mb1": self.mb1, "vb1": self.vb1,
            "mW2": self.mW2, "vW2": self.vW2,
            "mb2": self.mb2, "vb2": self.vb2,
            "mWv": self.mWv, "vWv": self.vWv,
            "mbv": np.array([self.mbv]), "vbv": np.array([self.vbv]),
            "adam_t": np.array([self.adam_t], dtype=np.int64),
            "ret_mean": np.array([self.ret_mean]),
            "ret_var": np.array([self.ret_var]),
            "ret_count": np.array([self.ret_count]),
        }

    def load_adam_state(self, state: dict) -> None:
        self.mW1 = state["mW1"]; self.vW1 = state["vW1"]
        self.mb1 = state["mb1"]; self.vb1 = state["vb1"]
        self.mW2 = state["mW2"]; self.vW2 = state["vW2"]
        self.mb2 = state["mb2"]; self.vb2 = state["vb2"]
        self.mWv = state["mWv"]; self.vWv = state["vWv"]
        self.mbv = float(state["mbv"][0]); self.vbv = float(state["vbv"][0])
        self.adam_t = int(state["adam_t"][0])
        self.ret_mean = float(state["ret_mean"][0])
        self.ret_var = float(state["ret_var"][0])
        self.ret_count = float(state["ret_count"][0])
