# Adaptive Ecosystem Lab

A self-contained 2D browser simulator for an evolving ecosystem. Open
`index.html` directly, or serve the folder with any static server.

## Ecological fidelity

The model is tuned to reproduce the *qualitative* behaviour of real ecosystems
rather than a single calibrated biome. The mechanics that matter:

- **Logistic primary production** — producers grow toward a resource-set
  carrying capacity (light, water, temperature, nutrients).
- **Emergent carrying capacity** — consumer numbers are limited by food,
  predation, and disease, not by hard population ceilings (the per-type caps are
  only non-binding safety rails).
- **Lindeman energy flow** — predators gain only the assimilated fraction of
  prey energy with no free bonus, so the standing-crop pyramid
  (producer ≫ herbivore > predator) emerges from respiration losses; measured
  predator/herbivore standing energy sits at ~10–20%.
- **Depletable resources** — nectar is a standing stock pollinators draw down
  and that regenerates gradually, so pollinator numbers are nectar-limited (every
  consumer guild settles below its population cap rather than against it).
- **Holling type II predation** — predator intake saturates with prey density
  via attack rate and handling time.
- **Kleiber's law** — basal metabolism scales with body mass^0.75.
- **Q10 temperature dependence** — ectotherm metabolic rate rises ~2.3× per
  +10 °C.
- **Seasonal breeding** — animals recruit in a spring/summer pulse.
- **Mass-conserving nutrient cycle** — nutrients flow plant → detritus → soil →
  plant with stoichiometric transfers plus small weathering/leaching fluxes.
- **SIRS disease** — Susceptible → Infected → Recovered(immune) → Susceptible,
  with density-dependent contact transmission and disease mortality.
- **Rescue effect** — rare recolonization prevents permanent local extinction in
  the closed patch, as a connected landscape would.

These properties are measured headlessly by the probe in `bench/` (trophic
pyramid, predator–prey lag, population variability, extinctions, and nutrient
conservation). See `bench/ECOLOGY_NOTES.md`.

## Included systems

- Abiotic terrain: a larger varied elevation map with ridges, highlands, river
  valleys, water, rainfall, temperature, sunlight, seasons, dissolved oxygen,
  soil nutrients, roughness, and toxicity.
- Producers: plant/algae biomass grows from light, water, temperature, and
  nutrients. Dense producer cells render visible plants and tree canopies on the
  map instead of only tinting the ground.
- Consumers: herbivores, predators, pollinators, decomposers, and keystone
  engineer agents.
- Hunger and thirst: every animal has hunger and thirst meters. Movement,
  metabolism, heat, dryness, and disease increase them; eating and drinking
  reduce them; starvation and dehydration drain energy and block reproduction.
- Elevation-aware movement: the Elevation overlay shows lowlands, hills, and
  high ridges. Animals move slower uphill, spend more energy climbing, move a
  little easier downhill, and are also slowed by rough terrain and water.
- Decomposition: dead bodies, waste, and plant litter become detritus, then
  decomposers and microbes return nutrients to soil.
- Evolution: agents carry inherited genes and a small neural-style policy
  weighting food, water, danger, mates, comfort, crowding, and wandering.
  Offspring mutate those values over generations.
- Regulation: energy costs, carrying capacity, predation, disease, aging,
  reproduction costs, resource scarcity, and crowding.
- Disturbance: drought, flood, wildfire, and disease can occur from controls or
  automatically from the disturbance slider.

The simulator intentionally skips full deep-learning backpropagation because
that would need a training stack and long-running experiments. It uses
neuroevolution instead, which is a practical first step for adaptive behavior in
an ecosystem sandbox.

## View locally

```powershell
.\.venv\Scripts\python.exe -m http.server 8098 --bind 127.0.0.1 --directory ecosystem
```

Then open `http://127.0.0.1:8098/index.html`.

## Train a policy

The `training/` folder contains a headless reinforcement-learning harness. The
environment mirrors the simulator's ecology (renewable forage, mobile pursuing
predators, Kleiber/Q10 metabolism, terrain costs) and rewards *lifetime
reproductive success*, not just survival. The trainer is a batched actor-critic
(learned value baseline, advantage + return normalization, entropy bonus, Adam,
gradient clipping, greedy evaluation, best-checkpoint saving) — all
dependency-free Python.

```bash
python training/train_policy.py --updates 120 --batch 8
```

The trainer saves the best greedy-eval checkpoint:

- `models/herbivore_policy.json`
- `models/training_history.json`

See `training/README.md` for details.
