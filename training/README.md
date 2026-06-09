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

- **PPO with GAE-λ** (clipped surrogate, multi-epoch). Each update reuses the
  batch for K Adam steps under a clipped objective that prevents the policy
  from walking off into a distribution it can't recover from. GAE-λ replaces
  Monte-Carlo returns with smoother TD-residual advantages. Together this is
  ~5× more sample-efficient than the vanilla REINFORCE-with-baseline the
  trainer started with (best eval_score 108 in 10 updates versus 12 updates of
  vanilla hitting ~50).
- **Numpy-vectorized** (single dependency): with ~1000 agents per tick, pure
  Python would be untenable. The world is column-oriented and every per-tick
  step (movement, metabolism, eating, hunting, reproduction, death) is a
  matrix op.
- **Parallel rollouts** via `--num-workers N` (multiprocessing.Pool). Both the
  per-update batch episodes and the multi-seed eval episodes are independent,
  so fanning them out to a worker pool gives ~2–2.5× wall-clock speedup at
  steady state on a 3-batch / 3-eval-episode config.
- **Per-species shared policy**: every alive herbivore samples its action
  from the herbivore network; same for the other four species. One PPO update
  per species per outer step.
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
- **Per-species reward breakdown logging** (`--log-breakdown-every N`) so you
  can see exactly what fraction of each species' reward came from food vs
  reproduction vs threat vs death. Essential for diagnosing stuck learners.

### Run

Smoke test (~30 s):

```bash
python training/train_multi_agent.py --updates 4 --batch 1 --episode-ticks 200 --log-every 1
```

Real training run (~10–15 min, fully sequential):

```bash
python training/train_multi_agent.py --updates 80 --batch 2 --episode-ticks 600 --log-every 4
```

Real training run with parallel rollouts (~3–6 min for the same work):

```bash
python training/train_multi_agent.py --updates 80 --batch 3 --episode-ticks 600 --log-every 4 --num-workers 3
```

Set `--num-workers` to at least `--batch` to fully parallelise rollouts;
setting it ≥ `--eval-episodes` also parallelises greedy eval. Workers stay
alive across the whole run, so the multiprocessing startup cost (~5 s on
Windows) amortises after the first update.

Outputs after a real run:

- `models_multi/best/{species}_policy.json` — best inference-only checkpoint
  for each species, captured whenever the multi-species eval score sets a new
  high
- `models_multi/last/{species}_policy.json` + `{species}_adam.npz` — most
  recent state of each policy plus the Adam moment estimates, used by
  `--resume` to continue training without restarting from random weights
- `models_multi/last/training_state.json` — current update count and best
  eval score (read by `--resume`)
- `models_multi/multi_agent_history.json` — full eval-score / population /
  per-species-metric trajectory across the whole run

### Resume / extend training

```bash
python training/train_multi_agent.py --resume models_multi/last --updates <bigger>
```

`<bigger>` is the *new total* — e.g. if a run finished at 2280 updates and
you set `--updates 4000`, the resumed run does another 1720. Adam moments,
return-normalization stats, and history are all restored.

### Trained policies in this repo

`models_multi/best/` contains the per-species policies from an overnight
training run (2280 updates, best at update 1056, multi-species
`eval_score = 243.49` — about 2.4× the strongest score achieved in any
single-round 200-update tune). The policy files are pure JSON, dependency-
free to load:

```python
import json
weights = json.load(open("models_multi/best/herbivore_policy.json"))
# weights has keys: W1, b1, W2, b2, Wv, bv  (numpy-style row-major)
# plus species name, obs_dim, action list, best_eval_score
```

The training history (`models_multi/multi_agent_history.json`) is a list of
per-update dicts you can plot with any tool. Useful columns:
`eval_score`, `best_eval_score`, `eval_avg_pop`, and per-species
`train_metrics` (entropy, value loss, transitions). The `models_multi/`
directory is gitignored so the policy files stay local — re-run training
with `--out-dir <your_dir>` to point elsewhere.

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
