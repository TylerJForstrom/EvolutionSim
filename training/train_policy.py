"""Train a small herbivore neural policy.

Usage:
    python training/train_policy.py --episodes 200

The model is intentionally dependency-free. It uses a one-hidden-layer neural
policy and REINFORCE policy-gradient updates, then saves JSON weights that can
be inspected or later loaded by the browser sim.
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
class EpisodeStats:
    episode: int
    reward: float
    steps: int
    energy: float
    hunger: float
    thirst: float
    best_reward: float


class NeuralPolicy:
    def __init__(self, obs_size: int, action_size: int, hidden_size: int = 32, seed: int = 1) -> None:
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

    def forward(self, obs: list[float]) -> tuple[list[float], list[float]]:
        hidden = []
        for row, bias in zip(self.w1, self.b1):
            z = bias + sum(weight * value for weight, value in zip(row, obs))
            hidden.append(math.tanh(z))
        logits = []
        for row, bias in zip(self.w2, self.b2):
            logits.append(bias + sum(weight * value for weight, value in zip(row, hidden)))
        return hidden, self._softmax(logits)

    def act(self, obs: list[float], rng: random.Random) -> tuple[int, list[float], list[float]]:
        hidden, probs = self.forward(obs)
        roll = rng.random()
        acc = 0.0
        for idx, prob in enumerate(probs):
            acc += prob
            if roll <= acc:
                return idx, hidden, probs
        return len(probs) - 1, hidden, probs

    def update(self, trajectories: list[tuple[list[float], list[float], list[float], int, float]], lr: float) -> None:
        # One-step policy gradient over collected transitions.
        for obs, hidden, probs, action, advantage in trajectories:
            dlogits = probs[:]
            dlogits[action] -= 1.0
            for a in range(self.action_size):
                grad = -advantage * dlogits[a]
                for h in range(self.hidden_size):
                    self.w2[a][h] += lr * grad * hidden[h]
                self.b2[a] += lr * grad

            dhidden = [0.0 for _ in range(self.hidden_size)]
            for h in range(self.hidden_size):
                downstream = sum((-advantage * dlogits[a]) * self.w2[a][h] for a in range(self.action_size))
                dhidden[h] = downstream * (1.0 - hidden[h] * hidden[h])

            for h in range(self.hidden_size):
                for i in range(self.obs_size):
                    self.w1[h][i] += lr * dhidden[h] * obs[i]
                self.b1[h] += lr * dhidden[h]

    def to_dict(self) -> dict:
        return {
            "type": "one_hidden_layer_policy",
            "obs_size": self.obs_size,
            "action_size": self.action_size,
            "hidden_size": self.hidden_size,
            "actions": list(ACTIONS),
            "w1": self.w1,
            "b1": self.b1,
            "w2": self.w2,
            "b2": self.b2,
        }

    @staticmethod
    def _softmax(logits: list[float]) -> list[float]:
        top = max(logits)
        exps = [math.exp(value - top) for value in logits]
        total = sum(exps)
        return [value / total for value in exps]


def discounted_returns(rewards: list[float], gamma: float) -> list[float]:
    returns = [0.0 for _ in rewards]
    running = 0.0
    for idx in range(len(rewards) - 1, -1, -1):
        running = rewards[idx] + gamma * running
        returns[idx] = running
    if not returns:
        return returns
    mean = sum(returns) / len(returns)
    variance = sum((value - mean) ** 2 for value in returns) / max(1, len(returns) - 1)
    scale = math.sqrt(variance) + 1e-6
    return [(value - mean) / scale for value in returns]


def run_episode(env: EcosystemEnv, policy: NeuralPolicy, rng: random.Random, gamma: float) -> tuple[EpisodeStats, list]:
    obs = env.reset(rng.randint(1, 999_999))
    transitions = []
    rewards = []
    total_reward = 0.0
    done = False
    info = {"energy": 0.0, "hunger": 0.0, "thirst": 0.0}
    while not done:
        action, hidden, probs = policy.act(obs, rng)
        next_obs, reward, done, info = env.step(action)
        transitions.append((obs, hidden, probs, action))
        rewards.append(reward)
        total_reward += reward
        obs = next_obs

    advantages = discounted_returns(rewards, gamma)
    training_rows = [
        (obs_row, hidden, probs, action, advantage)
        for (obs_row, hidden, probs, action), advantage in zip(transitions, advantages)
    ]
    stats = EpisodeStats(
        episode=0,
        reward=total_reward,
        steps=env.steps,
        energy=info["energy"],
        hunger=info["hunger"],
        thirst=info["thirst"],
        best_reward=total_reward,
    )
    return stats, training_rows


def train(args: argparse.Namespace) -> None:
    rng = random.Random(args.seed)
    env = EcosystemEnv(seed=args.seed, max_steps=args.max_steps)
    policy = NeuralPolicy(env.observation_size, env.action_size, hidden_size=args.hidden, seed=args.seed)
    best_reward = -1e9
    history: list[dict] = []

    for episode in range(1, args.episodes + 1):
        stats, rows = run_episode(env, policy, rng, args.gamma)
        policy.update(rows, args.lr)
        best_reward = max(best_reward, stats.reward)
        stats.episode = episode
        stats.best_reward = best_reward
        history.append(asdict(stats))
        if episode == 1 or episode % args.log_every == 0:
            print(
                f"episode={episode:4d} reward={stats.reward:8.2f} "
                f"steps={stats.steps:4d} hunger={stats.hunger:6.1f} "
                f"thirst={stats.thirst:6.1f} best={best_reward:8.2f}"
            )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "herbivore_policy.json"
    history_path = out_dir / "training_history.json"
    model = policy.to_dict()
    model["training"] = {
        "episodes": args.episodes,
        "max_steps": args.max_steps,
        "gamma": args.gamma,
        "lr": args.lr,
        "seed": args.seed,
        "best_reward": best_reward,
    }
    model_path.write_text(json.dumps(model, indent=2), encoding="utf-8")
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"saved model: {model_path}")
    print(f"saved history: {history_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a headless herbivore policy.")
    parser.add_argument("--episodes", type=int, default=80)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--gamma", type=float, default=0.985)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--out-dir", default="models")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
