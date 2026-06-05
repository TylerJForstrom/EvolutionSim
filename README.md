# Adaptive Ecosystem Lab

A self-contained 2D browser simulator for an evolving ecosystem. Open
`index.html` directly, or serve the folder with any static server.

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
