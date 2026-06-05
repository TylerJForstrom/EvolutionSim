# Training Harness

This folder contains a first headless training harness for the ecosystem.

## What it trains

`train_policy.py` trains one herbivore policy to:

- find food,
- find water,
- avoid predators,
- avoid starvation/dehydration,
- account for elevation, rough terrain, and uphill movement cost.

The model is a small neural network trained with a simple policy-gradient loop.
It is dependency-free so it runs with normal Python.

## Run

From the project root:

```powershell
python training\train_policy.py --episodes 80
```

For a fast smoke test:

```powershell
python training\train_policy.py --episodes 5 --max-steps 120 --log-every 1
```

Outputs are written to:

- `models/herbivore_policy.json`
- `models/training_history.json`

## Next upgrade

The next serious step is to replace the dependency-free trainer with PyTorch
PPO or SAC, then add multi-agent training for predators and pollinators.
