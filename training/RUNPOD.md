# Running training on RunPod

The multi-agent trainer has a PyTorch backend (`--use-torch --device cuda`)
that runs on a GPU. With ~$0.50–2/hr GPU rental this gets your wall-clock
down from ~16 hours (local CPU/numpy) to ~10–30 minutes.

This document is a complete walkthrough — you should be able to copy-paste
from here without filling in blanks.

---

## 1. Account, billing, credits

You need:
- A RunPod account (https://runpod.io)
- A payment method on file
- Credits loaded. **$10–20 is plenty** to start; a 30-min RTX 4090 run is
  about $0.25.

That's it. Skip ahead.

---

## 2. Pick the right pod

You're choosing **three** things: a template (the disk image), a GPU, and a
region.

### 2a. Template

In the RunPod console, **Deploy → Community Cloud → Templates**, search for:

> **`RunPod PyTorch 2.4`** (or any "PyTorch 2.x" template)

These templates come with:
- CUDA 12.x
- Python 3.10+
- `torch` (GPU build) pre-installed
- `git`, `curl`, `wget`
- SSH server running

**Avoid:** the "bare Ubuntu" template — you'd waste 5 minutes installing
torch.

### 2b. GPU

Two reasonable picks for this project:

| GPU | VRAM | Approx $/hr | Best for |
|---|---|---|---|
| **RTX 4090** | 24 GB | ~$0.50 | Cheap, fast enough — the obvious default |
| **A100 40GB** | 40 GB | ~$1.50 | Only worth it for very long chained runs |

The policies in this project are *tiny* (one hidden layer, ~50k parameters).
You don't need an A100. **Pick the cheapest 4090 available.** If they're all
gone in your region, try L40 or A40 next.

### 2c. Region

Pick the **closest region to where you live** — SSH latency matters when
you're typing into the pod.

- US East / US West → northern US, Canada, Mexico users
- EU Central / EU West → Europe users
- AP Southeast → Asia

Cost varies slightly between regions; closest is usually cheapest anyway.

### 2d. Disk + network volumes

This is the one decision people overthink. Two checkboxes:

- **Container Disk: 20 GB** is plenty. The repo is small, torch is already
  in the image, total disk use during training is < 5 GB.
- **Network Volume: skip it for now.** Network volumes are useful if you
  plan to spin up many pods and share state between them. For a single
  training run, just use the container disk.

If you want to be extra safe (pod gets killed mid-run for some reason), you
*can* attach a small 10 GB network volume mounted at `/workspace/checkpoints`
and write `--out-dir /workspace/checkpoints/models_multi`. But honestly,
just push checkpoints to git every N minutes (see step 6 below).

### 2e. Deploy

Click **Deploy On-Demand** (not Spot — spot pods can be killed without
warning mid-training; the savings aren't worth the headache for a 30-min
run).

The pod will show **"Provisioning"** for 30–90 seconds, then **"Running."**

---

## 3. Connect via SSH

RunPod gives you SSH access two ways. The clean way:

### 3a. Add your SSH key to RunPod (one-time setup)

On your local machine:

```bash
# Print your public key (create one with ssh-keygen if you don't have it)
cat ~/.ssh/id_ed25519.pub   # or id_rsa.pub
```

Paste the contents into **RunPod → Settings → SSH Public Keys → Add Key**.

### 3b. Connect to the pod

In the RunPod console, click your running pod → **Connect** → **SSH over
exposed TCP**. You get a command like:

```
ssh root@<some-host>.proxy.runpod.net -p 12345 -i ~/.ssh/id_ed25519
```

Paste it into your local terminal. First time, type `yes` to accept the
host key. You should land in `/root` or `/workspace`.

### 3c. (Optional) Open the web terminal

If SSH is being annoying (Windows ssh client issues, etc.), RunPod has a
**Web Terminal** button in the pod page that just gives you a browser shell.
Works fine for short sessions.

---

## 4. Clone the repo and train

In the pod terminal:

```bash
cd /workspace
git clone https://github.com/TylerJForstrom/EvolutionSim.git
cd EvolutionSim
pip install -r requirements.txt

# Verify the GPU is visible
python -c "import torch; print('cuda:', torch.cuda.is_available(), '| device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
```

Expected output:
```
cuda: True | device: NVIDIA GeForce RTX 4090
```

Now train:

```bash
python training/train_multi_agent.py --profile gpu --updates 5000 \
    --episode-ticks 600 --log-every 50 --log-breakdown-every 200 --save-every 20 \
    --out-dir models_multi
```

`--profile gpu` is a preset that sets `--use-torch --device cuda
--hidden 128 --batch 4 --num-workers 4` for you. You can still override any
of those individually:

```bash
# Same as above but with batch=6 and a wider net
python training/train_multi_agent.py --profile gpu --batch 6 --hidden 192 \
    --updates 5000 --episode-ticks 600 --out-dir models_multi
```

### Estimated wall-clock on a 4090 (with --profile gpu)

| Updates | Wall-clock | Cost @ $0.5/hr |
|---|---|---|
| 500 (sanity check) | ~3 min | < $0.05 |
| 5,000 | ~30 min | ~$0.25 |
| 20,000 (deep run) | ~2 hours | ~$1.00 |
| 50,000 (overkill) | ~5 hours | ~$2.50 |

These are rough; the actual speed depends on how many agents are alive at
once. Steady-state populations stabilise the per-update cost.

---

## 5. Watch progress

The trainer prints a log line every `--log-every` updates. To follow it
live in a separate terminal:

```bash
# Re-attach to the pod via a second SSH session, then:
tail -f /workspace/EvolutionSim/training.log
```

(If you used `python ... > training.log 2>&1 &` to background it. The
default just prints to stdout — keep the SSH session open OR use `nohup`
or `tmux`.)

Recommended: run training inside `tmux` so an SSH disconnect doesn't kill it:

```bash
tmux new -s train
# Now run the training command. Detach with Ctrl-B then D.
# Re-attach later with: tmux attach -t train
```

---

## 6. Get the trained models back

The `models_multi/` directory is gitignored, so it's just sitting in the
pod's container disk. Two ways to get the best policies to your laptop:

### Easiest: git commit the inference JSON

On the pod (it has git; if you'll commit, RunPod's HTTP push works fine):

```bash
cd /workspace/EvolutionSim
mkdir -p shipped_policies
cp models_multi/best/*_policy.json shipped_policies/
cp models_multi/best/training_state.json shipped_policies/
cp models_multi/multi_agent_history.json shipped_policies/

git config user.name "RunPod"
git config user.email "runpod@local"
git add shipped_policies/
git commit -m "Trained on RunPod (5000 updates, 4090)"
git push origin main
```

You'll need a Personal Access Token from GitHub (Settings → Developer
Settings → Personal Access Tokens) since you can't paste an SSH private
key on a shared pod. Use the token as the password when git prompts.

Then on your laptop:

```bash
git pull
ls shipped_policies/
```

### Alternative: scp from the pod

```bash
# On your laptop:
scp -P 12345 -i ~/.ssh/id_ed25519 \
    'root@<host>.proxy.runpod.net:/workspace/EvolutionSim/models_multi/best/*' \
    ./models_multi/best/
```

### Alternative: RunPod web file browser

In the pod page, click **Open File Browser**. Navigate to
`/workspace/EvolutionSim/models_multi/best/` and download files
individually. Slow for many files but no setup required.

---

## 7. Resume after a pod restart

Spot pods can disappear; on-demand pods can hit idle timeout. To pick up
where you left off:

```bash
python training/train_multi_agent.py --profile gpu --resume models_multi/last \
    --updates 10000 --episode-ticks 600 --out-dir models_multi
```

`--updates 10000` is the **new total**, not "10000 more updates". The
optimizer state, return statistics, and history are all preserved. The
trainer auto-detects the torch backend from the checkpoint, so you don't
strictly need `--profile gpu` here — but adding it doesn't hurt.

---

## 8. Tear down the pod

**This is the easiest step to forget and the most expensive mistake.**

When the training command exits:

1. Verify the models you wanted are in `shipped_policies/` and pushed to git
   (or downloaded via scp / file browser).
2. In the RunPod console, click the pod → **Stop** → **Terminate**.

"Stop" pauses billing but the disk persists (cheap). "Terminate" wipes the
pod and disk entirely (free). Once your trained models are off the pod, you
want Terminate.

If you forget for hours, you'll wake up to a bill. Set a phone alarm.

---

## Troubleshooting

**`cuda: False` despite picking a GPU pod.** You probably deployed a non-
PyTorch template. Either redeploy with the PyTorch template, or
`pip install torch --index-url https://download.pytorch.org/whl/cu121` to
get CUDA torch into the current pod.

**Training is slow (worse than 10 s/update).** Run
`nvidia-smi -l 1` in a second terminal. If GPU utilization is <30%,
the bottleneck is on the CPU side (env rollouts). Bump `--num-workers`
to match `--batch`, and `--episode-ticks` higher.

**`ModuleNotFoundError: torch`** in the pod. Either the template didn't
install it (rare) or you're inside a virtualenv that doesn't see it.
`pip install -r requirements.txt` will sort it out.

**Pod won't start ("queued").** That region/GPU combo is full. Either pick
a different region or a different GPU.

**SSH disconnects mid-training.** Use `tmux` (see step 5). Training keeps
running; just reattach next time.
