"""Worker-side episode runner for the multi-agent trainer.

This module exists as a separate top-level file so it can be imported and
pickled into multiprocessing.Pool workers cleanly. Windows uses the `spawn`
start method, which re-imports the worker callable in a fresh Python process
— anything pickled into a worker pool must be importable by name from the
worker, so closures and nested functions inside the trainer file won't work.

The worker reconstructs per-species policies from numpy snapshots (sent via
pickle), runs one full episode in a fresh World, and returns per-trajectory
data plus aggregate metrics. No Adam state, no gradient — pure inference.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

# Make sure the training/ directory is importable from the worker.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from multi_agent_env import N_ACTIONS, N_SPECIES, OBS_DIM, World  # noqa: E402
from multi_agent_policy import ActorCritic  # noqa: E402

# Reward category list must match the trainer; we duplicate it here so workers
# don't need to import from train_multi_agent (avoids circular imports).
REWARD_CATEGORIES = ("base", "food", "drink", "repro", "engineer_bonus", "threat", "condition", "death")


# ---------------------------------------------------------------------------
# Pickleable policy snapshots
# ---------------------------------------------------------------------------

def policy_snapshot(policy: ActorCritic) -> dict:
    """Compact, pickleable view of just the weights needed for inference."""
    return {
        "obs_size": int(policy.obs_size),
        "action_size": int(policy.action_size),
        "hidden_size": int(policy.hidden_size),
        "W1": policy.W1,
        "b1": policy.b1,
        "W2": policy.W2,
        "b2": policy.b2,
        "Wv": policy.Wv,
        "bv": float(policy.bv),
    }


def policy_from_snapshot(snap: dict, seed: int = 0) -> ActorCritic:
    p = ActorCritic(snap["obs_size"], snap["action_size"], snap["hidden_size"], seed)
    # Reference assignment is fine — the worker isn't going to mutate these.
    p.W1 = snap["W1"]
    p.b1 = snap["b1"]
    p.W2 = snap["W2"]
    p.b2 = snap["b2"]
    p.Wv = snap["Wv"]
    p.bv = float(snap["bv"])
    return p


# ---------------------------------------------------------------------------
# Trajectory buffer (inline copy of the trainer's, so the worker doesn't
# pull in train_multi_agent and create an import cycle)
# ---------------------------------------------------------------------------

class _WorkerTrajBuffer:
    """Slimmed-down per-slot buffer used inside an episode worker."""

    def __init__(self, max_agents: int):
        self._max = max_agents
        self.obs: list[list | None] = [None] * max_agents
        self.acts: list[list | None] = [None] * max_agents
        self.rews: list[list | None] = [None] * max_agents
        self.species: list[int | None] = [None] * max_agents

    def start(self, slot: int, species: int) -> None:
        if self.obs[slot] is not None:
            return
        self.obs[slot] = []
        self.acts[slot] = []
        self.rews[slot] = []
        self.species[slot] = species

    def append(self, slot: int, obs_row: np.ndarray, action: int, reward: float) -> None:
        if self.obs[slot] is None:
            return
        self.obs[slot].append(obs_row)
        self.acts[slot].append(int(action))
        self.rews[slot].append(float(reward))

    def to_species_trajectories(self):
        per_traj: list[list[tuple[np.ndarray, np.ndarray, np.ndarray]]] = [[] for _ in range(N_SPECIES)]
        per_episode_rewards = [0.0] * N_SPECIES
        per_lifetimes: list[list[int]] = [[] for _ in range(N_SPECIES)]
        for slot in range(self._max):
            sp = self.species[slot]
            if sp is None:
                continue
            rews = self.rews[slot]
            if not rews:
                continue
            obs_arr = np.array(self.obs[slot])
            act_arr = np.array(self.acts[slot], dtype=np.int64)
            rew_arr = np.array(rews)
            per_traj[sp].append((obs_arr, act_arr, rew_arr))
            per_episode_rewards[sp] += float(rew_arr.sum())
            per_lifetimes[sp].append(len(rews))
        return per_traj, per_episode_rewards, per_lifetimes


# ---------------------------------------------------------------------------
# Episode runner (inline, see comment above)
# ---------------------------------------------------------------------------

def _run_episode_inline(world: World, policies: list[ActorCritic], episode_ticks: int, greedy: bool, collect_breakdown: bool):
    buf = _WorkerTrajBuffer(world.alive.size)
    for slot in np.where(world.alive)[0]:
        buf.start(int(slot), int(world.type[slot]))

    breakdown = {sp: {cat: 0.0 for cat in REWARD_CATEGORIES} for sp in range(N_SPECIES)} if collect_breakdown else None
    breakdown_steps = {sp: 0 for sp in range(N_SPECIES)} if collect_breakdown else None

    for _ in range(episode_ticks):
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

        info = world.step_world()

        for sp in range(N_SPECIES):
            slots = per_species_slots[sp]
            if slots.size == 0:
                continue
            obs = per_species_obs[sp]
            acts = per_species_acts[sp]
            rewards = info.rewards[slots]
            for i in range(slots.size):
                buf.append(int(slots[i]), obs[i], int(acts[i]), float(rewards[i]))

        if collect_breakdown:
            components = world.reward_components()
            for sp in range(N_SPECIES):
                slots = per_species_slots[sp]
                if slots.size == 0:
                    continue
                breakdown_steps[sp] += int(slots.size)
                for cat in REWARD_CATEGORIES:
                    breakdown[sp][cat] += float(components[cat][slots].sum())

        for slot in info.just_born_slots:
            buf.start(int(slot), int(world.type[slot]))

        if not world.alive.any():
            break

    return buf, world.population_counts(), breakdown, breakdown_steps


# ---------------------------------------------------------------------------
# Public worker entry point — top-level function so it's pickleable
# ---------------------------------------------------------------------------

def run_episode_remote(args):
    """Pickleable callable for multiprocessing.Pool.

    args is a tuple to keep pickle payload compact:
        (snapshots, seed, episode_ticks, greedy, collect_breakdown)
    """
    snapshots, seed, episode_ticks, greedy, collect_breakdown = args
    policies = [policy_from_snapshot(s) for s in snapshots]
    world = World(seed=seed)
    buf, pops, breakdown, bd_steps = _run_episode_inline(
        world, policies, episode_ticks, greedy=greedy, collect_breakdown=collect_breakdown,
    )
    trajs, ep_rewards, lifetimes = buf.to_species_trajectories()
    return trajs, pops, ep_rewards, lifetimes, breakdown, bd_steps
