# Running training on RunPod

The multi-agent trainer has a PyTorch backend (`--use-torch --device cuda`)
that runs on a GPU. With ~$0.50–2/hr GPU rental this gets your wall-clock
down from ~16 hours (local CPU/numpy) to ~10–30 minutes.

This document is a copy-paste recipe.

## 1. Pick a pod

Recommended: **PyTorch 2.x base image, RTX 4090 or A100 40GB**.

- RTX 4090: cheap (~$0.5/hr), plenty fast for this workload
- A100 40GB: faster (~$1.5/hr), worth it if you're chaining many runs
- Anything bigger is overkill — the policies are tiny (one hidden layer)

Make sure the template has:
- CUDA 11.8+ or 12.x
- Python 3.10+
- `torch` already installed (this is the default on PyTorch pod templates)
- `pip` available

## 2. Connect via SSH and clone

```bash
# Once the pod is up, ssh in as instructed by the RunPod UI
cd /workspace
git clone https://github.com/TylerJForstrom/EvolutionSim.git
cd EvolutionSim
```

Verify torch sees the GPU:

```bash
python -c "import torch; print('cuda:', torch.cuda.is_available(), 'device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
```

Expected output:
```
cuda: True device: NVIDIA GeForce RTX 4090
```

## 3. Install the rest

PyTorch is already there. We just need numpy (and that's it).

```bash
pip install -r requirements.txt
```

## 4. Run training

```bash
python training/train_multi_agent.py \
    --use-torch --device cuda \
    --updates 5000 \
    --batch 4 \
    --episode-ticks 600 \
    --log-every 50 \
    --log-breakdown-every 200 \
    --save-every 20 \
    --num-workers 4 \
    --out-dir models_multi
```

Notes:
- `--num-workers 4` keeps rollout episodes in parallel on the CPU side while
  the policy update runs on the GPU. Set it equal to `--batch`.
- `--episode-ticks 600` matches the env's year length.
- `--save-every 20` writes a resumable checkpoint every ~5 min so a pod
  restart doesn't lose more than that.

### Estimated wall-clock on a 4090

| Updates | Wall-clock (rough) |
|---|---|
| 500 (sanity check) | ~3 min |
| 5,000 | ~30 min |
| 50,000 (deep run) | ~5 hours |

These scale roughly linearly with `--updates` until the populations stabilize;
after that the per-update cost trends down because fewer agents die mid-
episode and the trajectory batches get cleaner.

## 5. Get the trained models back to your laptop

The simplest path is git: commit the JSON inference checkpoints, ignore the
.pt resume artifacts (they're big).

On the pod:

```bash
# Bring the best policies out from under the gitignored models_multi/
mkdir -p shipped_policies
cp models_multi/best/*_policy.json shipped_policies/
cp models_multi/best/training_state.json shipped_policies/
cp models_multi/multi_agent_history.json shipped_policies/
git add shipped_policies/
git -c user.name="$USER" -c user.email="runpod@local" commit -m "Trained on RunPod"
git push origin main
```

Then on your laptop:

```bash
git pull
# Inspect / use them
python -c "import json; print(json.load(open('shipped_policies/predator_policy.json'))['best_eval_score'])"
```

If you'd rather skip git entirely, RunPod's web file browser can download
the JSON files directly, or you can `scp` them.

## 6. Resume an interrupted run

Pods can disappear (spot pricing, idle timeout, whatever). Just resume:

```bash
python training/train_multi_agent.py \
    --resume models_multi/last \
    --updates 10000 \
    --use-torch --device cuda \
    --num-workers 4 \
    --out-dir models_multi
```

`--updates 10000` is the *new total*. Resume picks up where the previous
run left off; the optimizer state and return statistics are preserved.

## 7. Tear down the pod

Don't forget. RunPod bills by the hour even when training is done.
