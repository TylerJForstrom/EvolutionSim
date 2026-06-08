"""Train all five species simultaneously via Independent Learners.

Per-species shared policy: every alive agent of a species selects actions
through that species' policy each tick. After each fixed-length episode of the
world simulation, every agent's lifetime trajectory (from spawn to death or
episode end) is bundled into that species' training batch and one Adam step is
applied per species. So all five policies update from the same shared
simulation, each learning to thrive in a world that is also adapting.

This is coevolutionary RL: the environment from any one species' view is
non-stationary by construction. That's intentional — it's the Red Queen.

Usage:
    python training/train_multi_agent.py --updates 60 --episode-ticks 600 --batch 2
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

# Allow `python training/train_multi_agent.py` and `python -m training.train_multi_agent`.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from multi_agent_env import (  # noqa: E402
    N_ACTIONS, N_SPECIES, OBS_DIM, SPECIES_NAMES, World, _MAX_AGENTS,
)
from multi_agent_policy import ActorCritic, TrainConfig  # noqa: E402


# ---------------------------------------------------------------------------
# Trajectory buffer
# ---------------------------------------------------------------------------

class TrajBuffer:
    """Per-slot lifetime buffer. Slot ids are stable across an episode."""

    def __init__(self):
        self.obs: list[list | None] = [None] * _MAX_AGENTS
        self.acts: list[list | None] = [None] * _MAX_AGENTS
        self.rews: list[list | None] = [None] * _MAX_AGENTS
        self.species: list[int | None] = [None] * _MAX_AGENTS

    def reset(self) -> None:
        for s in range(_MAX_AGENTS):
            self.obs[s] = None
            self.acts[s] = None
            self.rews[s] = None
            self.species[s] = None

    def start(self, slot: int, species: int) -> None:
        if self.obs[slot] is not None:
            return  # already tracking
        self.obs[slot] = []
        self.acts[slot] = []
        self.rews[slot] = []
        self.species[slot] = species

    def append(self, slot: int, obs_row: np.ndarray, action: int, reward: float) -> None:
        if self.obs[slot] is None:
            return  # we don't know its species yet
        self.obs[slot].append(obs_row)
        self.acts[slot].append(int(action))
        self.rews[slot].append(float(reward))

    def to_species_batches(self, gamma: float):
        """Return dict[species_idx] -> (obs, actions, returns) as numpy arrays."""
        per_obs = [[] for _ in range(N_SPECIES)]
        per_acts = [[] for _ in range(N_SPECIES)]
        per_returns = [[] for _ in range(N_SPECIES)]
        per_episode_rewards = [0.0] * N_SPECIES
        per_lifetimes: list[list[int]] = [[] for _ in range(N_SPECIES)]
        for slot in range(_MAX_AGENTS):
            sp = self.species[slot]
            if sp is None:
                continue
            rews = self.rews[slot]
            if not rews:
                continue
            # Discounted return-to-go for this lifetime.
            G = 0.0
            returns = [0.0] * len(rews)
            for i in range(len(rews) - 1, -1, -1):
                G = rews[i] + gamma * G
                returns[i] = G
            per_obs[sp].extend(self.obs[slot])
            per_acts[sp].extend(self.acts[slot])
            per_returns[sp].extend(returns)
            per_episode_rewards[sp] += sum(rews)
            per_lifetimes[sp].append(len(rews))
        # Stack to arrays.
        out = {}
        for sp in range(N_SPECIES):
            if per_obs[sp]:
                out[sp] = (
                    np.array(per_obs[sp]),
                    np.array(per_acts[sp], dtype=np.int64),
                    np.array(per_returns[sp]),
                )
            else:
                out[sp] = (
                    np.zeros((0, OBS_DIM)),
                    np.zeros(0, dtype=np.int64),
                    np.zeros(0),
                )
        return out, per_episode_rewards, per_lifetimes


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_episode(
    world: World,
    policies: list[ActorCritic],
    episode_ticks: int,
    greedy: bool = False,
):
    """Roll out one episode. Returns the populated TrajBuffer plus per-species
    final population counts."""
    buf = TrajBuffer()
    # Initialize trajectories for all agents alive at episode start.
    for slot in np.where(world.alive)[0]:
        buf.start(int(slot), int(world.type[slot]))

    for _ in range(episode_ticks):
        # Per species: observe, act, set pending actions on world.
        per_species_slots: list[np.ndarray] = [np.array([], dtype=np.int64)] * N_SPECIES
        per_species_obs: list[np.ndarray] = [np.zeros((0, OBS_DIM))] * N_SPECIES
        per_species_acts: list[np.ndarray] = [np.zeros(0, dtype=np.int32)] * N_SPECIES
        for sp in range(N_SPECIES):
            obs, slots = world.observe(sp)
            if slots.size == 0:
                continue
            actions, _h, _p, _v = policies[sp].act(obs, greedy=greedy)
            world.set_actions(sp, slots, actions)
            per_species_slots[sp] = slots
            per_species_obs[sp] = obs
            per_species_acts[sp] = actions

        # Advance world; rewards/deaths/births returned per slot.
        info = world.step_world()

        # Record transitions for agents that acted this tick.
        for sp in range(N_SPECIES):
            slots = per_species_slots[sp]
            if slots.size == 0:
                continue
            obs = per_species_obs[sp]
            acts = per_species_acts[sp]
            rewards = info.rewards[slots]
            for i in range(slots.size):
                buf.append(int(slots[i]), obs[i], int(acts[i]), float(rewards[i]))

        # Start trajectories for newborns (they'll observe & act next tick).
        for slot in info.just_born_slots:
            buf.start(int(slot), int(world.type[slot]))

        # If everything is dead, stop early.
        if not world.alive.any():
            break

    return buf, world.population_counts()


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    rng_seed = args.seed
    train_cfg = TrainConfig(
        lr=args.lr,
        gamma=args.gamma,
        entropy_beta_init=args.entropy_beta,
        entropy_beta_final=args.entropy_final,
        value_coef=args.value_coef,
        grad_clip=args.clip,
        hidden=args.hidden,
    )
    policies = [
        ActorCritic(OBS_DIM, N_ACTIONS, hidden=train_cfg.hidden, seed=rng_seed * 17 + sp)
        for sp in range(N_SPECIES)
    ]

    world = World(seed=rng_seed)
    history: list[dict] = []
    best_eval_score = -1e18
    best_models = {sp: policies[sp].to_dict() for sp in range(N_SPECIES)}

    t_start = time.time()
    for update in range(1, args.updates + 1):
        frac = update / args.updates
        entropy_beta = train_cfg.entropy_beta_init * (1.0 - frac) + train_cfg.entropy_beta_final * frac

        # ---- Collect rollouts: `batch` episodes per update.
        species_obs: list[list[np.ndarray]] = [[] for _ in range(N_SPECIES)]
        species_acts: list[list[np.ndarray]] = [[] for _ in range(N_SPECIES)]
        species_rets: list[list[np.ndarray]] = [[] for _ in range(N_SPECIES)]
        ep_rewards_total = [0.0] * N_SPECIES
        ep_lifetimes: list[list[int]] = [[] for _ in range(N_SPECIES)]
        final_pop_running = {sp: 0 for sp in range(N_SPECIES)}
        for ep in range(args.batch):
            world.reset(seed=rng_seed * 31 + update * 7 + ep)
            buf, pops = run_episode(world, policies, args.episode_ticks, greedy=False)
            batches, ep_rewards, lifetimes = buf.to_species_batches(train_cfg.gamma)
            for sp in range(N_SPECIES):
                o, a, r = batches[sp]
                if o.shape[0] > 0:
                    species_obs[sp].append(o)
                    species_acts[sp].append(a)
                    species_rets[sp].append(r)
                ep_rewards_total[sp] += ep_rewards[sp]
                ep_lifetimes[sp].extend(lifetimes[sp])
                final_pop_running[sp] += pops[SPECIES_NAMES[sp]]

        # ---- Per-species update.
        update_metrics = {}
        for sp in range(N_SPECIES):
            if not species_obs[sp]:
                update_metrics[sp] = {"entropy": 0.0, "value_loss": 0.0, "n": 0}
                continue
            obs = np.concatenate(species_obs[sp], axis=0)
            acts = np.concatenate(species_acts[sp], axis=0)
            rets = np.concatenate(species_rets[sp], axis=0)
            m = policies[sp].update(obs, acts, rets, train_cfg, entropy_beta)
            update_metrics[sp] = m

        # ---- Greedy eval on fixed seed (so best_score is comparable across updates).
        world.reset(seed=rng_seed * 31)
        eval_buf, eval_pops = run_episode(world, policies, args.episode_ticks, greedy=True)
        eval_batches, eval_rewards, eval_lifetimes = eval_buf.to_species_batches(train_cfg.gamma)
        # "score" = sum of mean episode reward per species (so all species
        # matter equally), normalized so it doesn't blow up when a species
        # has few agents.
        eval_score = 0.0
        for sp in range(N_SPECIES):
            n_lives = max(1, len(eval_lifetimes[sp]))
            eval_score += eval_rewards[sp] / n_lives

        if eval_score > best_eval_score:
            best_eval_score = eval_score
            best_models = {sp: policies[sp].to_dict() for sp in range(N_SPECIES)}

        # ---- Log.
        elapsed = time.time() - t_start
        if update == 1 or update % args.log_every == 0:
            train_lines = []
            eval_lines = []
            for sp in range(N_SPECIES):
                n_lives = max(1, len(ep_lifetimes[sp]))
                avg_life = sum(ep_lifetimes[sp]) / n_lives if ep_lifetimes[sp] else 0.0
                avg_rew = ep_rewards_total[sp] / max(1, args.batch) / max(1, n_lives / max(1, args.batch))
                m = update_metrics[sp]
                train_lines.append(
                    f"{SPECIES_NAMES[sp][:4]:>4}: r={avg_rew:6.2f} life={avg_life:5.1f} H={m['entropy']:.2f} vL={m['value_loss']:.2f} n={m['n']}"
                )
                eval_pop = eval_pops.get(SPECIES_NAMES[sp], 0)
                eval_lines.append(f"{SPECIES_NAMES[sp][:4]:>4}={eval_pop}")
            print(
                f"u={update:3d} t={elapsed:6.0f}s eval_score={eval_score:7.2f} best={best_eval_score:7.2f}"
                f"  | eval_final_pop: " + " ".join(eval_lines)
            )
            for line in train_lines:
                print("    " + line)

        history.append({
            "update": update,
            "elapsed_s": elapsed,
            "eval_score": eval_score,
            "best_eval_score": best_eval_score,
            "eval_population": eval_pops,
            "train_metrics": {SPECIES_NAMES[sp]: update_metrics[sp] for sp in range(N_SPECIES)},
        })

    # ---- Save.
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for sp in range(N_SPECIES):
        path = out_dir / f"{SPECIES_NAMES[sp]}_policy.json"
        payload = best_models[sp]
        payload["species"] = SPECIES_NAMES[sp]
        payload["best_eval_score"] = best_eval_score
        payload["actions"] = ["stay", "n", "ne", "e", "se", "s", "sw", "w", "nw"]
        payload["obs_dim"] = OBS_DIM
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (out_dir / "multi_agent_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"\nsaved {N_SPECIES} per-species best policies to {out_dir}/")
    print(f"best multi-species eval_score = {best_eval_score:.2f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train all five species simultaneously.")
    parser.add_argument("--updates", type=int, default=60)
    parser.add_argument("--batch", type=int, default=2, help="episodes per gradient update")
    parser.add_argument("--episode-ticks", type=int, default=600)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--clip", type=float, default=5.0)
    parser.add_argument("--entropy-beta", type=float, default=0.015)
    parser.add_argument("--entropy-final", type=float, default=0.001)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--log-every", type=int, default=2)
    parser.add_argument("--out-dir", default="models_multi")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
