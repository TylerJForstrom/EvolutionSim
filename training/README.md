# Training Harness

A headless, dependency-free training harness for the ecosystem. It trains one
herbivore policy under the same ecological pressures the browser simulator
applies, then saves JSON weights.

## The environment (`ecosim_env.py`)

A single herbivore lives on a noise-generated landscape (elevation, roughness,
water, temperature, forage) and must manage energy, hunger, and thirst. The
environment mirrors the simulator's ecology:

- **Renewable forage** — vegetation regrows *logistically* toward each cell's
  carrying capacity, so over-grazing a patch has real consequences.
- **Mobile, pursuing predators** — predators search, then chase when the
  herbivore comes within detection range, and can kill on contact.
- **Kleiber + Q10 metabolism** — basal energy burn scales with body mass^0.75
  and rises with temperature (Q10 ≈ 2.3), exactly as in `sim.js`.
- **Terrain costs** — uphill movement, roughness, and water drag slow movement
  and burn energy.
- **Reproductive fitness** — a well-fed, hydrated adult reproduces. The agent is
  rewarded for *lifetime reproductive success*, which is the real selection
  target, not survival time alone.

Observation (21 dims): energy / hunger / thirst, local cell state (food, water,
elevation, roughness, temperature), last slope and terrain factor, nearest
food / water / predator direction+strength, a reproduction-ready flag, and the
seasonal light signal.

## The trainer (`train_policy.py`)

A one-hidden-layer **actor-critic** trained with batched policy gradients:

- batched updates (gradient averaged over several episodes),
- a **learned value baseline** (advantage = return − V(s)),
- advantage normalization,
- an **entropy bonus** annealed over training to keep early exploration,
- global gradient-norm clipping,
- periodic **greedy evaluation** and best-checkpoint saving.

## Run

From the project root:

```bash
python training/train_policy.py --updates 120 --batch 8
```

Fast smoke test:

```bash
python training/train_policy.py --updates 12 --batch 6 --max-steps 300 --log-every 2
```

Outputs (best greedy-eval checkpoint):

- `models/herbivore_policy.json`
- `models/training_history.json`

## Multi-agent coevolution (all five species, simultaneously)

`multi_agent_env.py` + `multi_agent_policy.py` + `train_multi_agent.py` is a
second, larger training stack that runs **all five species at once**, each
with its own shared policy network, training simultaneously in the same world
simulation. The environment from any one species' perspective is non-stationary
by construction — that's the point (Red Queen coevolution).

### What's different

- **Numpy-vectorized** (single dependency): with ~1000 agents per tick, pure
  Python would be untenable. The world is column-oriented and every per-tick
  step (movement, metabolism, eating, hunting, reproduction, death) is a
  matrix op.
- **Per-species shared policy**: every alive herbivore samples its action
  from the herbivore network; same for the other four species. One Adam step
  per species per update.
- **Slot-stable preallocated agent table** so agents have stable IDs across
  birth/death, enabling per-lifetime trajectory buffers without dictionary
  churn.
- **Rescue migration** (mirroring sim.js): when a guild nearly collapses, a
  small refill drops in from "neighbouring habitat." Prevents permanent
  extinction during the unstable early-training phase, without pegging
  populations at a target.
- **Per-species rewards**: reproduction is the dominant signal across all
  species (it IS lifetime fitness); plus small dense food/water bonuses for
  learnability, and species-specific costs (prey gets a threat penalty when
  a predator is close, etc.).

### Run

Smoke test (~30 s):

```bash
python training/train_multi_agent.py --updates 4 --batch 1 --episode-ticks 200 --log-every 1
```

Real training run (~10–15 min):

```bash
python training/train_multi_agent.py --updates 80 --batch 2 --episode-ticks 600 --log-every 4
```

Outputs (best multi-species eval checkpoint):

- `models_multi/herbivore_policy.json`
- `models_multi/predator_policy.json`
- `models_multi/decomposer_policy.json`
- `models_multi/pollinator_policy.json`
- `models_multi/engineer_policy.json`
- `models_multi/multi_agent_history.json`

### What to expect

Multi-agent RL is **not** stable the way single-agent RL is. The Red Queen
dynamic produces a tug-of-war: predators learn to hunt faster than prey learn
to flee, or vice versa, and the trailing species can collapse before learning
catches up. The rescue migration prevents permanent extinction, but a healthy
training run still shows large per-update swings. Look for monotonic eval
score improvement only over many updates; expect noise tick-to-tick. Entropy
(`H`) dropping below ~2.0 indicates the policies have actually broken away
from random.

### Limitations and honest caveats

- **Independent Learners** ignores other species' policies in the gradient; it
  gets non-stationarity in the environment for free but doesn't address it
  explicitly. More sophisticated multi-agent algorithms (MADDPG, COMA, MAPPO)
  would help. This is fine for a first pass.
- **The trained policies live in this env, not in `sim.js`.** Loading them
  back into the browser sim would require porting the obs/action conventions
  across, which is non-trivial.
- **Reward shaping is a design choice, not a derivation.** The per-species
  reward weights affect what behavior the policies converge to. Tune them in
  `multi_agent_env.py` (`_REPRO_REWARD`, `_DEATH_PENALTY`, the `food_signal`
  block in `_compute_rewards`) if you want different priorities.

### Single-species fallback

The original single-herbivore stack (`ecosim_env.py` + `train_policy.py`) is
still here and still works. It's much faster to train and gives you a clean
baseline to compare the multi-agent policies against.
