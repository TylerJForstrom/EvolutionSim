"""Train a small herbivore neural policy with batched, baseline-corrected
policy gradients.

Usage:
    python training/train_policy.py --updates 200 --batch 8

The model is intentionally dependency-free (pure Python). Compared with a plain
single-episode REINFORCE loop it adds the pieces that actually matter for stable
learning on a noisy survival task:

- batched updates (average the gradient over several episodes per step),
- a learned value baseline (actor-critic style advantage = return - V(s)),
- advantage normalization,
- an entropy bonus (annealed) so the policy keeps exploring early,
- global gradient-norm clipping,
- periodic greedy evaluation and best-checkpoint saving.

It saves JSON weights that can be inspected or loaded elsewhere.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

from ecosim_env import ACTIONS, EcosystemEnv


@dataclass
class UpdateStats:
    update: int
    train_reward: float
    eval_reward: float
    steps: float
    entropy: float
    value_loss: float
    best_eval: float


class ActorCritic:
    """One-hidden-layer policy with a linear value head sharing the features."""

    def __init__(self, obs_size: int, action_size: int, hidden_size: int = 48, seed: int = 1) -> None:
        self.obs_size = obs_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.rng = random.Random(seed)
        scale1 = 1.0 / math.sqrt(obs_size)
        scale2 = 1.0 / math.sqrt(hidden_size)
        self.w1 = [[self.rng.uniform(-scale1, scale1) for _ in range(obs_size)] for _ in range(hidden_size)]
        self.b1 = [0.0 for _ in range(hidden_size)]
        self.w2 = [[self.rng.uniform(-scale2, scale2) for _ in range(hidden_size)] for _ in range(action_size)]
        self.b2 = [0.0 for _ in range(action_size)]
        # Value head.
        self.wv = [self.rng.uniform(-scale2, scale2) for _ in range(hidden_size)]
        self.bv = 0.0
        # Running statistics of returns, used to normalize value targets and
        # advantages to ~unit scale. Without this the raw returns (hundreds) make
        # the value gradient swamp the policy gradient under a shared grad clip.
        self.ret_mean = 0.0
        self.ret_var = 1.0
        self.ret_count = 1e-4
        # Adam optimizer state (first/second moment estimates per parameter).
        self.adam_t = 0
        self._init_adam()

    def _init_adam(self) -> None:
        self.m_w1 = [[0.0] * self.obs_size for _ in range(self.hidden_size)]
        self.v_w1 = [[0.0] * self.obs_size for _ in range(self.hidden_size)]
        self.m_b1 = [0.0] * self.hidden_size
        self.v_b1 = [0.0] * self.hidden_size
        self.m_w2 = [[0.0] * self.hidden_size for _ in range(self.action_size)]
        self.v_w2 = [[0.0] * self.hidden_size for _ in range(self.action_size)]
        self.m_b2 = [0.0] * self.action_size
        self.v_b2 = [0.0] * self.action_size
        self.m_wv = [0.0] * self.hidden_size
        self.v_wv = [0.0] * self.hidden_size
        self.m_bv = 0.0
        self.v_bv = 0.0

    def update_ret_stats(self, returns: list[float]) -> None:
        # Welford-style running merge of a batch of returns.
        for g in returns:
            self.ret_count += 1.0
            delta = g - self.ret_mean
            self.ret_mean += delta / self.ret_count
            self.ret_var += delta * (g - self.ret_mean)

    def ret_std(self) -> float:
        return math.sqrt(max(self.ret_var / max(1.0, self.ret_count), 1e-6)) + 1e-6

    def forward(self, obs: list[float]) -> tuple[list[float], list[float], float]:
        hidden = []
        for row, bias in zip(self.w1, self.b1):
            z = bias + sum(weight * value for weight, value in zip(row, obs))
            hidden.append(math.tanh(z))
        logits = []
        for row, bias in zip(self.w2, self.b2):
            logits.append(bias + sum(weight * value for weight, value in zip(row, hidden)))
        probs = self._softmax(logits)
        value = self.bv + sum(w * h for w, h in zip(self.wv, hidden))
        return hidden, probs, value

    def act(self, obs: list[float], rng: random.Random, greedy: bool = False):
        hidden, probs, value = self.forward(obs)
        if greedy:
            action = max(range(len(probs)), key=lambda i: probs[i])
            return action, hidden, probs, value
        roll = rng.random()
        acc = 0.0
        for idx, prob in enumerate(probs):
            acc += prob
            if roll <= acc:
                return idx, hidden, probs, value
        return len(probs) - 1, hidden, probs, value

    def zero_grad(self) -> dict:
        return {
            "w1": [[0.0] * self.obs_size for _ in range(self.hidden_size)],
            "b1": [0.0] * self.hidden_size,
            "w2": [[0.0] * self.hidden_size for _ in range(self.action_size)],
            "b2": [0.0] * self.action_size,
            "wv": [0.0] * self.hidden_size,
            "bv": 0.0,
        }

    def accumulate(self, grad: dict, transition, entropy_beta: float, value_coef: float) -> None:
        obs, hidden, probs, action, advantage, ret, value = transition

        # Entropy of the categorical policy and its gradient wrt logits.
        entropy = -sum(p * math.log(p + 1e-12) for p in probs)
        # Policy gradient (ascent) wrt each logit, plus entropy bonus.
        g_logits = [0.0] * self.action_size
        for a in range(self.action_size):
            onehot = 1.0 if a == action else 0.0
            pg = advantage * (onehot - probs[a])
            ent_grad = -probs[a] * (math.log(probs[a] + 1e-12) + entropy)
            g_logits[a] = pg + entropy_beta * ent_grad

        # Value head gradient (descend 0.5*(v-ret)^2 -> ascent direction is -(v-ret)).
        v_err = value - ret
        for h in range(self.hidden_size):
            grad["wv"][h] += -value_coef * v_err * hidden[h]
        grad["bv"] += -value_coef * v_err

        # Backprop policy grad into the output layer and shared hidden features.
        dhidden = [0.0] * self.hidden_size
        for a in range(self.action_size):
            gl = g_logits[a]
            row = self.w2[a]
            for h in range(self.hidden_size):
                grad["w2"][a][h] += gl * hidden[h]
                dhidden[h] += gl * row[h]
            grad["b2"][a] += gl
        # Value error also shapes the shared features (small coefficient).
        for h in range(self.hidden_size):
            dhidden[h] += -value_coef * v_err * self.wv[h]

        for h in range(self.hidden_size):
            dh = dhidden[h] * (1.0 - hidden[h] * hidden[h])
            row = grad["w1"][h]
            for i in range(self.obs_size):
                row[i] += dh * obs[i]
            grad["b1"][h] += dh

    def apply(self, grad: dict, lr: float, scale: float, clip: float) -> None:
        # Adam ascent on the combined objective (the value-loss sign is already
        # baked into the accumulated gradient). Per-parameter step adaptation
        # makes learning robust to the raw REINFORCE gradient scale, which a
        # fixed-step SGD update could not handle here.
        # Scale (average over the batch) and clip the global gradient norm first.
        norm_sq = 0.0
        for h in range(self.hidden_size):
            for i in range(self.obs_size):
                grad["w1"][h][i] *= scale
                norm_sq += grad["w1"][h][i] ** 2
            grad["b1"][h] *= scale
            grad["wv"][h] *= scale
            norm_sq += grad["b1"][h] ** 2 + grad["wv"][h] ** 2
        for a in range(self.action_size):
            for h in range(self.hidden_size):
                grad["w2"][a][h] *= scale
                norm_sq += grad["w2"][a][h] ** 2
            grad["b2"][a] *= scale
            norm_sq += grad["b2"][a] ** 2
        grad["bv"] *= scale
        norm_sq += grad["bv"] ** 2
        norm = math.sqrt(norm_sq) + 1e-12
        gclip = min(1.0, clip / norm) if clip > 0 else 1.0

        self.adam_t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        bc1 = 1.0 - b1 ** self.adam_t
        bc2 = 1.0 - b2 ** self.adam_t

        def adam(param_row, m_row, v_row, g_row, idx):
            g = g_row[idx] * gclip
            m_row[idx] = b1 * m_row[idx] + (1.0 - b1) * g
            v_row[idx] = b2 * v_row[idx] + (1.0 - b2) * g * g
            mhat = m_row[idx] / bc1
            vhat = v_row[idx] / bc2
            param_row[idx] += lr * mhat / (math.sqrt(vhat) + eps)

        for h in range(self.hidden_size):
            for i in range(self.obs_size):
                adam(self.w1[h], self.m_w1[h], self.v_w1[h], grad["w1"][h], i)
            adam(self.b1, self.m_b1, self.v_b1, grad["b1"], h)
            adam(self.wv, self.m_wv, self.v_wv, grad["wv"], h)
        for a in range(self.action_size):
            for h in range(self.hidden_size):
                adam(self.w2[a], self.m_w2[a], self.v_w2[a], grad["w2"][a], h)
            adam(self.b2, self.m_b2, self.v_b2, grad["b2"], a)
        # Scalar value-head bias.
        g = grad["bv"] * gclip
        self.m_bv = b1 * self.m_bv + (1.0 - b1) * g
        self.v_bv = b2 * self.v_bv + (1.0 - b2) * g * g
        self.bv += lr * (self.m_bv / bc1) / (math.sqrt(self.v_bv / bc2) + eps)

    def to_dict(self) -> dict:
        return {
            "type": "actor_critic_one_hidden_layer",
            "obs_size": self.obs_size,
            "action_size": self.action_size,
            "hidden_size": self.hidden_size,
            "actions": list(ACTIONS),
            "w1": self.w1,
            "b1": self.b1,
            "w2": self.w2,
            "b2": self.b2,
            "wv": self.wv,
            "bv": self.bv,
        }

    @staticmethod
    def _softmax(logits: list[float]) -> list[float]:
        top = max(logits)
        exps = [math.exp(value - top) for value in logits]
        total = sum(exps)
        return [value / total for value in exps]


def reward_to_go(rewards: list[float], gamma: float) -> list[float]:
    returns = [0.0 for _ in rewards]
    running = 0.0
    for idx in range(len(rewards) - 1, -1, -1):
        running = rewards[idx] + gamma * running
        returns[idx] = running
    return returns


def run_episode(env: EcosystemEnv, policy: ActorCritic, rng: random.Random, gamma: float, greedy: bool = False):
    obs = env.reset(rng.randint(1, 999_999))
    transitions = []
    rewards = []
    total_reward = 0.0
    done = False
    info = {"energy": 0.0, "hunger": 0.0, "thirst": 0.0}
    while not done:
        action, hidden, probs, value = policy.act(obs, rng, greedy=greedy)
        next_obs, reward, done, info = env.step(action)
        transitions.append((obs, hidden, probs, action, value))
        rewards.append(reward)
        total_reward += reward
        obs = next_obs

    returns = reward_to_go(rewards, gamma)
    return total_reward, env.steps, transitions, returns


def normalize_advantages(rows: list) -> list:
    advs = [r[4] for r in rows]
    if len(advs) < 2:
        return rows
    mean = sum(advs) / len(advs)
    var = sum((a - mean) ** 2 for a in advs) / max(1, len(advs) - 1)
    scale = math.sqrt(var) + 1e-6
    return [(o, h, p, a, (adv - mean) / scale, ret, v) for (o, h, p, a, adv, ret, v) in rows]


def train(args: argparse.Namespace) -> None:
    rng = random.Random(args.seed)
    env = EcosystemEnv(seed=args.seed, max_steps=args.max_steps)
    policy = ActorCritic(env.observation_size, env.action_size, hidden_size=args.hidden, seed=args.seed)
    history: list[dict] = []
    best_eval = -1e9
    best_model = policy.to_dict()

    for update in range(1, args.updates + 1):
        # Anneal the entropy bonus from beta -> ~0 over training.
        frac = update / args.updates
        entropy_beta = args.entropy_beta * (1.0 - frac) + args.entropy_final * frac

        episodes: list = []
        batch_reward = 0.0
        batch_steps = 0
        all_returns: list = []
        for _ in range(args.batch):
            total_reward, steps, transitions, returns = run_episode(env, policy, rng, args.gamma)
            episodes.append((transitions, returns))
            all_returns.extend(returns)
            batch_reward += total_reward
            batch_steps += steps

        # Normalize returns to ~unit scale: value head fits normalized returns and
        # advantage = (normalized return) - value, so policy and value gradients
        # are comparably scaled.
        policy.update_ret_stats(all_returns)
        rmean, rstd = policy.ret_mean, policy.ret_std()
        batch_rows: list = []
        for transitions, returns in episodes:
            for (o, h, p, a, v), g in zip(transitions, returns):
                gn = (g - rmean) / rstd
                batch_rows.append((o, h, p, a, gn - v, gn, v))
        # Center+scale advantages across the whole batch (lower variance).
        batch_rows = normalize_advantages(batch_rows)

        grad = policy.zero_grad()
        ent_sum = 0.0
        vloss_sum = 0.0
        for row in batch_rows:
            policy.accumulate(grad, row, entropy_beta, args.value_coef)
            ent_sum += -sum(p * math.log(p + 1e-12) for p in row[2])
            vloss_sum += 0.5 * (row[6] - row[5]) ** 2
        # Average the gradient over the batch; Adam adapts the per-parameter step
        # so the update is robust to the raw REINFORCE gradient magnitude.
        policy.apply(grad, args.lr, 1.0 / max(1, len(batch_rows)), args.clip)

        # Greedy evaluation for an unbiased performance signal.
        eval_reward = 0.0
        eval_rng = random.Random(args.seed * 31 + update)
        for _ in range(args.eval_episodes):
            r, _s, _t, _ret = run_episode(env, policy, eval_rng, args.gamma, greedy=True)
            eval_reward += r
        eval_reward /= max(1, args.eval_episodes)

        if eval_reward > best_eval:
            best_eval = eval_reward
            best_model = policy.to_dict()

        stats = UpdateStats(
            update=update,
            train_reward=batch_reward / args.batch,
            eval_reward=eval_reward,
            steps=batch_steps / args.batch,
            entropy=ent_sum / max(1, len(batch_rows)),
            value_loss=vloss_sum / max(1, len(batch_rows)),
            best_eval=best_eval,
        )
        history.append(asdict(stats))
        if update == 1 or update % args.log_every == 0:
            print(
                f"update={update:4d} train={stats.train_reward:8.2f} eval={stats.eval_reward:8.2f} "
                f"steps={stats.steps:6.1f} H={stats.entropy:5.3f} vloss={stats.value_loss:7.2f} "
                f"best={best_eval:8.2f}"
            )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "herbivore_policy.json"
    history_path = out_dir / "training_history.json"
    model = best_model
    model["training"] = {
        "updates": args.updates,
        "batch": args.batch,
        "max_steps": args.max_steps,
        "gamma": args.gamma,
        "lr": args.lr,
        "entropy_beta": args.entropy_beta,
        "value_coef": args.value_coef,
        "seed": args.seed,
        "best_eval": best_eval,
    }
    model_path.write_text(json.dumps(model, indent=2), encoding="utf-8")
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"saved best model (eval={best_eval:.2f}): {model_path}")
    print(f"saved history: {history_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a headless herbivore actor-critic policy.")
    parser.add_argument("--updates", type=int, default=120)
    parser.add_argument("--batch", type=int, default=8, help="episodes per gradient update")
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--hidden", type=int, default=48)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--clip", type=float, default=5.0, help="global grad-norm clip (0 disables)")
    parser.add_argument("--entropy-beta", type=float, default=0.012, help="initial entropy bonus")
    parser.add_argument("--entropy-final", type=float, default=0.001, help="final entropy bonus")
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--eval-episodes", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--out-dir", default="models")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
