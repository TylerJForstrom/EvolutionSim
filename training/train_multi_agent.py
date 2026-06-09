"""Train all five species simultaneously via Independent Learners.

Per-species shared policy: every alive agent of a species selects actions
through that species' policy each tick. After each fixed-length episode of the
world simulation, every agent's lifetime trajectory (from spawn to death or
episode end) is bundled into that species' training batch and one Adam step is
applied per species. So all five policies update from the same shared
simulation, each learning to thrive in a world that is also adapting.

This is coevolutionary RL: the environment from any one species' view is
non-stationary by construction. That's intentional — it's the Red Queen.

Long-run features for real training:
- multi-seed greedy eval (averages out per-map noise so `best_eval` actually
  tracks policy improvement instead of luck)
- periodic best-and-last checkpointing every `--save-every` updates (so an
  interrupted run isn't lost; last/ also saves Adam state for resume)
- `--resume <dir>` reloads policies + Adam state + history and continues
- per-species reward breakdown logged every `--log-breakdown-every` updates
  (shows what fraction of each species' reward came from food vs reproduction
  vs threat vs death — essential for diagnosing stuck learners)

Usage:
    python training/train_multi_agent.py --updates 200 --batch 3 --episode-ticks 600 --log-every 10
    python training/train_multi_agent.py --resume models_multi/last  # continue
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
    N_ACTIONS, N_SPECIES, OBS_DIM, PREDATOR, SPECIES_NAMES, World, _MAX_AGENTS,
)
from multi_agent_policy import ActorCritic, TrainConfig  # noqa: E402


REWARD_CATEGORIES = ("base", "food", "drink", "repro", "engineer_bonus", "threat", "condition", "death")


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

    def to_species_trajectories(self):
        """Return per-species lists of (obs, actions, rewards) numpy arrays,
        one entry per agent lifetime. The trainer uses these to compute GAE
        advantages against policy values, which needs the per-trajectory
        structure (returns are not enough)."""
        per_traj: list[list[tuple[np.ndarray, np.ndarray, np.ndarray]]] = [[] for _ in range(N_SPECIES)]
        per_episode_rewards = [0.0] * N_SPECIES
        per_lifetimes: list[list[int]] = [[] for _ in range(N_SPECIES)]
        for slot in range(_MAX_AGENTS):
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

    # Kept for older eval code paths that just want raw returns.
    def to_species_batches(self, gamma: float):
        per_traj, per_rew, per_lives = self.to_species_trajectories()
        out = {}
        for sp in range(N_SPECIES):
            obs_chunks: list[np.ndarray] = []
            act_chunks: list[np.ndarray] = []
            ret_chunks: list[np.ndarray] = []
            for obs_arr, act_arr, rew_arr in per_traj[sp]:
                # Monte Carlo return-to-go (no bootstrap).
                G = 0.0
                returns = np.zeros_like(rew_arr)
                for i in range(rew_arr.size - 1, -1, -1):
                    G = float(rew_arr[i]) + gamma * G
                    returns[i] = G
                obs_chunks.append(obs_arr)
                act_chunks.append(act_arr)
                ret_chunks.append(returns)
            if obs_chunks:
                out[sp] = (
                    np.concatenate(obs_chunks, axis=0),
                    np.concatenate(act_chunks, axis=0),
                    np.concatenate(ret_chunks, axis=0),
                )
            else:
                out[sp] = (
                    np.zeros((0, OBS_DIM)),
                    np.zeros(0, dtype=np.int64),
                    np.zeros(0),
                )
        return out, per_rew, per_lives


# ---------------------------------------------------------------------------
# GAE computation
# ---------------------------------------------------------------------------

def compute_gae(rewards: np.ndarray, values: np.ndarray, gamma: float, lam: float) -> tuple[np.ndarray, np.ndarray]:
    """GAE-λ advantages for a single trajectory.

    Returns (advantages, returns_target) where:
        δ_t          = r_t + γ V(s_{t+1}) - V(s_t)   (with V(s_T) = 0)
        A_t          = δ_t + γλ A_{t+1}
        R_target_t   = A_t + V(s_t)                  (target for value head)

    We treat trajectory end (death OR episode truncation) as terminal (next
    value = 0). This adds a small bias for truncated trajectories of healthy
    agents but keeps the implementation simple.
    """
    T = rewards.size
    if T == 0:
        return np.zeros(0), np.zeros(0)
    advantages = np.zeros(T)
    gae = 0.0
    for t in range(T - 1, -1, -1):
        next_value = values[t + 1] if t + 1 < T else 0.0
        delta = float(rewards[t]) + gamma * next_value - float(values[t])
        gae = delta + gamma * lam * gae
        advantages[t] = gae
    returns_target = advantages + values
    return advantages, returns_target


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_episode(
    world: World,
    policies: list[ActorCritic],
    episode_ticks: int,
    greedy: bool = False,
    collect_breakdown: bool = False,
):
    """Roll out one episode. Returns the populated TrajBuffer, per-species
    final pop counts, and (optionally) reward-category totals per species."""
    buf = TrajBuffer()
    for slot in np.where(world.alive)[0]:
        buf.start(int(slot), int(world.type[slot]))

    # Per-species per-category running totals across the episode.
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
            types = world.type
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
# Checkpoint I/O
# ---------------------------------------------------------------------------

def _write_policy_files(out_dir: Path, policies: list[ActorCritic], best_eval: float) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for sp in range(N_SPECIES):
        d = policies[sp].to_dict()
        d["species"] = SPECIES_NAMES[sp]
        d["best_eval_score"] = best_eval
        d["actions"] = ["stay", "n", "ne", "e", "se", "s", "sw", "w", "nw"]
        d["obs_dim"] = OBS_DIM
        (out_dir / f"{SPECIES_NAMES[sp]}_policy.json").write_text(json.dumps(d, indent=2), encoding="utf-8")


def save_checkpoint(
    root: Path,
    kind: str,
    policies: list[ActorCritic],
    history: list[dict],
    best_eval: float,
    update: int,
    *,
    save_adam: bool,
) -> None:
    """kind is 'best' or 'last'. Writes per-species policy JSON; for 'last',
    also writes Adam state per species and a training_state.json so a future
    --resume can pick up exactly where we left off."""
    sub = root / kind
    sub.mkdir(parents=True, exist_ok=True)
    _write_policy_files(sub, policies, best_eval)
    state = {
        "update": int(update),
        "best_eval_score": float(best_eval),
        "kind": kind,
    }
    (sub / "training_state.json").write_text(json.dumps(state, indent=2), encoding="utf-8")
    # History is the same for best/last — write at root.
    (root / "multi_agent_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    if save_adam:
        for sp in range(N_SPECIES):
            adam = policies[sp].adam_state()
            np.savez(sub / f"{SPECIES_NAMES[sp]}_adam.npz", **adam)


def load_checkpoint(
    root: Path,
    train_cfg: TrainConfig,
) -> tuple[list[ActorCritic], list[dict], int, float]:
    """Load policies (with Adam state), history, last update, and best_eval.

    `root` should be a directory written by save_checkpoint with kind='last'.
    """
    if not (root / "training_state.json").exists():
        raise FileNotFoundError(f"no training_state.json at {root}")
    state = json.loads((root / "training_state.json").read_text(encoding="utf-8"))
    policies: list[ActorCritic] = []
    for sp in range(N_SPECIES):
        pdict = json.loads((root / f"{SPECIES_NAMES[sp]}_policy.json").read_text(encoding="utf-8"))
        p = ActorCritic.from_dict(pdict, seed=sp)
        adam_path = root / f"{SPECIES_NAMES[sp]}_adam.npz"
        if adam_path.exists():
            with np.load(adam_path) as data:
                p.load_adam_state({k: data[k] for k in data.files})
        policies.append(p)
    history_path = root.parent / "multi_agent_history.json"
    history = json.loads(history_path.read_text(encoding="utf-8")) if history_path.exists() else []
    return policies, history, int(state["update"]), float(state["best_eval_score"])


# ---------------------------------------------------------------------------
# Multi-seed eval
# ---------------------------------------------------------------------------

def evaluate(
    world: World,
    policies: list[ActorCritic],
    episode_ticks: int,
    eval_seed_base: int,
    n_episodes: int,
):
    """Run n_episodes greedy episodes, each at a different fixed seed, and
    return averaged metrics: eval_score, per-species mean per-life reward,
    per-species final pop, per-species reward breakdown."""
    species_rewards = [0.0] * N_SPECIES
    species_lifetimes = [0] * N_SPECIES
    pop_totals = {SPECIES_NAMES[sp]: 0 for sp in range(N_SPECIES)}
    breakdown_totals = {sp: {cat: 0.0 for cat in REWARD_CATEGORIES} for sp in range(N_SPECIES)}
    breakdown_steps = {sp: 0 for sp in range(N_SPECIES)}
    for ep in range(n_episodes):
        world.reset(seed=eval_seed_base + ep * 13)
        buf, pops, bd, bd_steps = run_episode(
            world, policies, episode_ticks, greedy=True, collect_breakdown=True,
        )
        _, ep_rewards, lifetimes = buf.to_species_batches(0.99)
        for sp in range(N_SPECIES):
            species_rewards[sp] += ep_rewards[sp]
            species_lifetimes[sp] += len(lifetimes[sp])
            pop_totals[SPECIES_NAMES[sp]] += pops[SPECIES_NAMES[sp]]
            for cat in REWARD_CATEGORIES:
                breakdown_totals[sp][cat] += bd[sp][cat]
            breakdown_steps[sp] += bd_steps[sp]
    eval_score = 0.0
    for sp in range(N_SPECIES):
        lives = max(1, species_lifetimes[sp])
        eval_score += species_rewards[sp] / lives
    avg_pop = {k: v / n_episodes for k, v in pop_totals.items()}
    return {
        "eval_score": eval_score,
        "species_rewards_per_life": [
            species_rewards[sp] / max(1, species_lifetimes[sp]) for sp in range(N_SPECIES)
        ],
        "species_lifetimes": species_lifetimes,
        "avg_pop": avg_pop,
        "breakdown_totals": breakdown_totals,
        "breakdown_steps": breakdown_steps,
    }


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

    out_dir = Path(args.out_dir)

    # --- Resume or fresh start ---
    start_update = 1
    history: list[dict] = []
    best_eval_score = -1e18
    best_models: dict[int, dict] | None = None
    if args.resume:
        resume_path = Path(args.resume)
        policies, history, start_update_loaded, best_eval_score = load_checkpoint(resume_path, train_cfg)
        start_update = start_update_loaded + 1
        print(f"resumed from {resume_path} at update {start_update_loaded}, best_eval={best_eval_score:.2f}")
    else:
        policies = [
            ActorCritic(OBS_DIM, N_ACTIONS, hidden=train_cfg.hidden, seed=rng_seed * 17 + sp)
            for sp in range(N_SPECIES)
        ]

    world = World(seed=rng_seed)
    t_start = time.time()
    eval_seed_base = rng_seed * 31

    for update in range(start_update, args.updates + 1):
        frac = update / args.updates
        entropy_beta = train_cfg.entropy_beta_init * (1.0 - frac) + train_cfg.entropy_beta_final * frac
        # Predator gets a slower entropy decay because its training task is
        # the hardest and its population is smallest. Without this it stays
        # near-uniform and never escapes random.
        pred_frac = frac * 0.4
        pred_entropy_beta = (
            train_cfg.entropy_beta_init * (1.0 - pred_frac)
            + train_cfg.entropy_beta_final * pred_frac
        )

        # --- Collect rollouts: keep per-trajectory structure for GAE ---
        species_trajs: list[list[tuple[np.ndarray, np.ndarray, np.ndarray]]] = [[] for _ in range(N_SPECIES)]
        ep_rewards_total = [0.0] * N_SPECIES
        ep_lifetimes: list[list[int]] = [[] for _ in range(N_SPECIES)]
        for ep in range(args.batch):
            world.reset(seed=rng_seed * 31 + update * 7 + ep)
            buf, pops, _bd, _bds = run_episode(world, policies, args.episode_ticks, greedy=False)
            per_traj, ep_rewards, lifetimes = buf.to_species_trajectories()
            for sp in range(N_SPECIES):
                species_trajs[sp].extend(per_traj[sp])
                ep_rewards_total[sp] += ep_rewards[sp]
                ep_lifetimes[sp].extend(lifetimes[sp])

        # --- Per-species PPO+GAE update ---
        # For each species:
        # 1. flatten all trajectories into one batch (obs, acts, rewards)
        # 2. forward through current policy to get old log-probs and values
        # 3. compute GAE per trajectory using those values
        # 4. concatenate advantages and returns_target across trajectories
        # 5. run K PPO epochs over the flat batch
        update_metrics = {}
        for sp in range(N_SPECIES):
            trajs = species_trajs[sp]
            if not trajs:
                update_metrics[sp] = {"entropy": 0.0, "value_loss": 0.0, "n": 0, "kl": 0.0, "clip_frac": 0.0}
                continue
            obs = np.concatenate([t[0] for t in trajs], axis=0)
            acts = np.concatenate([t[1] for t in trajs], axis=0)

            old_log_probs, values = policies[sp].log_probs_and_values(obs, acts)

            # GAE per trajectory, then concatenate.
            advantages_chunks: list[np.ndarray] = []
            returns_chunks: list[np.ndarray] = []
            cursor = 0
            for _obs_arr, _act_arr, rew_arr in trajs:
                T = rew_arr.size
                traj_values = values[cursor:cursor + T]
                adv, ret_t = compute_gae(rew_arr, traj_values, train_cfg.gamma, train_cfg.gae_lambda)
                advantages_chunks.append(adv)
                returns_chunks.append(ret_t)
                cursor += T
            advantages = np.concatenate(advantages_chunks)
            returns_target = np.concatenate(returns_chunks)

            beta = pred_entropy_beta if sp == PREDATOR else entropy_beta
            m = policies[sp].update_ppo(
                obs, acts, old_log_probs, advantages, returns_target, train_cfg, beta,
            )
            update_metrics[sp] = m

        # --- Multi-seed greedy eval ---
        eval_result = evaluate(
            world, policies, args.episode_ticks, eval_seed_base, args.eval_episodes,
        )
        eval_score = eval_result["eval_score"]

        new_best = eval_score > best_eval_score
        if new_best:
            best_eval_score = eval_score

        # --- Log ---
        elapsed = time.time() - t_start
        log_this = update == start_update or update % args.log_every == 0 or update == args.updates
        log_breakdown = args.log_breakdown_every > 0 and (update == start_update or update % args.log_breakdown_every == 0 or update == args.updates)
        if log_this:
            eval_pop_str = " ".join(f"{SPECIES_NAMES[sp][:4]}={eval_result['avg_pop'][SPECIES_NAMES[sp]]:.1f}" for sp in range(N_SPECIES))
            print(
                f"u={update:3d} t={elapsed:6.0f}s eval_score={eval_score:7.2f} best={best_eval_score:7.2f}"
                f"  | eval_avg_pop: {eval_pop_str}"
                + ("  [NEW BEST]" if new_best else "")
            )
            for sp in range(N_SPECIES):
                n_lives = max(1, len(ep_lifetimes[sp]))
                avg_life = sum(ep_lifetimes[sp]) / n_lives if ep_lifetimes[sp] else 0.0
                m = update_metrics[sp]
                train_r = ep_rewards_total[sp] / max(1, len(ep_lifetimes[sp])) if ep_lifetimes[sp] else 0.0
                eval_r = eval_result["species_rewards_per_life"][sp]
                kl = m.get("kl", 0.0)
                clip_frac = m.get("clip_frac", 0.0)
                print(
                    f"    {SPECIES_NAMES[sp][:4]:>4}: train_r={train_r:6.2f} eval_r={eval_r:6.2f} life={avg_life:5.1f} "
                    f"H={m['entropy']:.2f} vL={m['value_loss']:.2f} KL={kl:+.3f} clip={clip_frac*100:4.1f}% n={m['n']}"
                )
        if log_breakdown:
            print("    -- reward breakdown (% of total positive + negative reward per species, from eval) --")
            for sp in range(N_SPECIES):
                steps = max(1, eval_result["breakdown_steps"][sp])
                totals = eval_result["breakdown_totals"][sp]
                total_abs = sum(abs(v) for v in totals.values()) or 1.0
                pct = {cat: 100.0 * totals[cat] / total_abs for cat in REWARD_CATEGORIES}
                pct_str = " ".join(f"{cat[:5]}={pct[cat]:+5.1f}%" for cat in REWARD_CATEGORIES)
                print(f"    {SPECIES_NAMES[sp][:4]:>4}: steps={steps:6d}  {pct_str}")

        history.append({
            "update": update,
            "elapsed_s": elapsed,
            "eval_score": eval_score,
            "best_eval_score": best_eval_score,
            "eval_avg_pop": eval_result["avg_pop"],
            "train_metrics": {SPECIES_NAMES[sp]: update_metrics[sp] for sp in range(N_SPECIES)},
        })

        # --- Periodic checkpointing ---
        if new_best:
            save_checkpoint(out_dir, "best", policies, history, best_eval_score, update, save_adam=False)
        if args.save_every > 0 and (update % args.save_every == 0 or update == args.updates):
            save_checkpoint(out_dir, "last", policies, history, best_eval_score, update, save_adam=True)

    # --- Final save ---
    save_checkpoint(out_dir, "last", policies, history, best_eval_score, args.updates, save_adam=True)
    print(f"\nbest multi-species eval_score = {best_eval_score:.2f}")
    print(f"best checkpoints (inference): {out_dir}/best/")
    print(f"last checkpoint (resume-able): {out_dir}/last/")
    print(f"history: {out_dir}/multi_agent_history.json")
    print(f"\nresume with:  python training/train_multi_agent.py --resume {out_dir}/last --updates <bigger_N>")


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
    parser.add_argument("--log-breakdown-every", type=int, default=10,
                        help="log per-species reward breakdown every N updates (0 to disable)")
    parser.add_argument("--eval-episodes", type=int, default=3,
                        help="number of greedy eval episodes per update")
    parser.add_argument("--save-every", type=int, default=5,
                        help="checkpoint 'last' every N updates (best is saved on improvement)")
    parser.add_argument("--out-dir", default="models_multi")
    parser.add_argument("--resume", default=None,
                        help="path to a `last/` checkpoint dir to resume from")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
