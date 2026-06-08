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

## Next upgrade

Replace the pure-Python trainer with PyTorch PPO/SAC and extend to multi-agent
training for predators and pollinators, sharing the same ecological env.
