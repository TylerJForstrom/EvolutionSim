"""Training-health diagnostic for the multi-agent ecosystem trainer.

Reads a run's history (multi_agent_history.json) and prints a one-line verdict
on whether training is healthy and trending the right way, or sliding into one
of the known failure modes -- so you don't have to eyeball the raw numbers.

The failure signatures it checks for are exactly the ones from the 5000-update
collapse this was written to catch:
  - NaN corruption in any train metric (the run is dead and won't recover)
  - predator policy mode-collapse (entropy -> ~0, the 0.000 cliff at u=1650)
  - herbivore population explosion (eval pop running past the carry_k plateau
    toward the destructive ~67)
  - predator functional extinction (eval pop -> ~0 and staying there)
  - eval-score collapse (recent score far below the best achieved)

It also reports DIRECTION: improving / stable / regressing, from the recent
eval-score trend and how recently `best_eval_score` last improved.

CLI:
    python training/training_health.py models_multi
    python training/training_health.py runs/models_seed_1594 runs/models_seed_206
    python training/training_health.py shipped_policies/multi_agent_history.json

Programmatic:
    from training_health import load_history, assess
    report = assess(load_history("models_multi"))
    print(report["verdict"])          # the headline
    report["level"]                   # "ok" | "warn" | "crit" | "early"
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

# --- Thresholds (tunable). Tied to the collapse we diagnosed. ----------------
EARLY_UPDATES = 150        # below this, the policy is still ~random; only NaN/liveness is meaningful
RECENT_FRAC = 0.15         # fraction of the run treated as the "recent" window
MIN_RECENT = 5             # ...but always look at >= this many records

PRED_ENTROPY_CRIT = 0.05   # predator entropy below this = mode-collapse (cliff went to 0.000)
PRED_ENTROPY_WARN = 0.15
HERB_POP_WARN = 45.0       # herbivore eval pop above this = explosion (carry_k plateau ~34, destructive ~67)
HERB_POP_CRIT = 60.0
PRED_POP_WARN = 1.5        # predator eval pop at/below this (past EARLY) = near-extinction
PRED_POP_CRIT = 0.5
EVAL_REGRESS_FRAC = 0.5    # recent eval below this * best (best>0) = meaningful regression

SPECIES = ["herbivore", "predator", "decomposer", "pollinator", "engineer"]


# --- IO ----------------------------------------------------------------------
def load_history(path: str | Path) -> list[dict]:
    """Accept either a multi_agent_history.json file or a run dir containing one."""
    p = Path(path)
    if p.is_dir():
        p = p / "multi_agent_history.json"
    if not p.exists():
        raise FileNotFoundError(f"no history at {p}")
    # The trainer writes NaN tokens (json default allow_nan=True); json.loads reads them back.
    return json.loads(p.read_text(encoding="utf-8"))


# --- helpers -----------------------------------------------------------------
def _is_nan(x) -> bool:
    return isinstance(x, float) and math.isnan(x)


def _mean(xs: list[float]) -> float:
    xs = [x for x in xs if x is not None and not _is_nan(x)]
    return sum(xs) / len(xs) if xs else float("nan")


def _pop(rec: dict, sp: str) -> float | None:
    try:
        return float(rec["eval_avg_pop"][sp])
    except (KeyError, TypeError, ValueError):
        return None


def _entropy(rec: dict, sp: str) -> float | None:
    try:
        return float(rec["train_metrics"][sp]["entropy"])
    except (KeyError, TypeError, ValueError):
        return None


def _last_best_improve_update(history: list[dict]) -> int:
    """Update at which best_eval_score last increased."""
    best = -float("inf")
    at = history[0].get("update", 1)
    for r in history:
        b = r.get("best_eval_score")
        if b is not None and not _is_nan(b) and b > best + 1e-9:
            best = b
            at = r.get("update", at)
    return at


# --- core assessment ---------------------------------------------------------
def assess(history: list[dict], *, recent_frac: float = RECENT_FRAC) -> dict:
    """Return a structured health report for one run's history.

    Keys: update, eval_score, best, level ('ok'|'warn'|'crit'|'early'),
    direction, verdict (headline str), checks (list of {name, level, detail})."""
    if not history:
        return {"update": 0, "eval_score": float("nan"), "best": float("nan"),
                "level": "crit", "direction": "n/a",
                "verdict": "no history records found", "checks": []}

    latest = history[-1]
    update = latest.get("update", len(history))
    k = max(MIN_RECENT, int(len(history) * recent_frac))
    recent = history[-k:]
    earlier = history[-2 * k:-k] if len(history) >= 2 * k else history[:max(1, len(history) - k)]

    eval_recent = _mean([r.get("eval_score") for r in recent])
    eval_earlier = _mean([r.get("eval_score") for r in earlier])
    best = latest.get("best_eval_score", float("nan"))
    since_best = update - _last_best_improve_update(history)

    checks: list[dict] = []

    # 1. NaN corruption (any species, any metric in the recent window).
    nan_species = []
    for r in recent:
        for sp, m in (r.get("train_metrics") or {}).items():
            if any(_is_nan(v) for v in (m or {}).values()):
                if sp not in nan_species:
                    nan_species.append(sp)
    if nan_species:
        checks.append({"name": "nan", "level": "crit",
                       "detail": f"NaN in train metrics for: {', '.join(nan_species)} (run is corrupted, will not recover)"})
    else:
        checks.append({"name": "nan", "level": "ok", "detail": "no NaN in train metrics"})

    early = update < EARLY_UPDATES

    # 2. Predator entropy (mode-collapse).
    pred_H = [_entropy(r, "predator") for r in recent]
    pred_H_min = min([h for h in pred_H if h is not None and not _is_nan(h)], default=float("nan"))
    if not math.isnan(pred_H_min):
        if pred_H_min < PRED_ENTROPY_CRIT:
            checks.append({"name": "predator_entropy", "level": "crit",
                           "detail": f"predator entropy collapsed to {pred_H_min:.3f} (policy mode-collapse; can't hunt even with prey present)"})
        elif pred_H_min < PRED_ENTROPY_WARN:
            checks.append({"name": "predator_entropy", "level": "warn",
                           "detail": f"predator entropy low ({pred_H_min:.3f}); approaching mode-collapse"})
        else:
            checks.append({"name": "predator_entropy", "level": "ok",
                           "detail": f"predator entropy healthy (min {pred_H_min:.2f})"})

    # 3. Herbivore explosion.
    herb_recent = _mean([_pop(r, "herbivore") for r in recent])
    if not math.isnan(herb_recent):
        if herb_recent > HERB_POP_CRIT:
            checks.append({"name": "herbivore_pop", "level": "crit",
                           "detail": f"herbivore pop {herb_recent:.0f} -- runaway explosion (carry_k plateau is ~34)"})
        elif herb_recent > HERB_POP_WARN:
            checks.append({"name": "herbivore_pop", "level": "warn",
                           "detail": f"herbivore pop {herb_recent:.0f} climbing above the ~34 plateau"})
        else:
            checks.append({"name": "herbivore_pop", "level": "ok",
                           "detail": f"herbivore pop bounded ({herb_recent:.1f})"})

    # 4. Predator near-extinction (only meaningful past the early/random phase).
    pred_recent = _mean([_pop(r, "predator") for r in recent])
    if not early and not math.isnan(pred_recent):
        if pred_recent < PRED_POP_CRIT:
            checks.append({"name": "predator_pop", "level": "crit",
                           "detail": f"predators functionally extinct ({pred_recent:.2f}); no top-down control"})
        elif pred_recent < PRED_POP_WARN:
            checks.append({"name": "predator_pop", "level": "warn",
                           "detail": f"predators very low ({pred_recent:.2f}); at risk of losing control of herbivores"})
        else:
            checks.append({"name": "predator_pop", "level": "ok",
                           "detail": f"predators present ({pred_recent:.1f})"})

    # 5. Eval-score collapse vs the best achieved.
    if not early and not math.isnan(eval_recent):
        if best is not None and not _is_nan(best) and best > 20 and eval_recent < 0:
            checks.append({"name": "eval_score", "level": "crit",
                           "detail": f"recent eval {eval_recent:.1f} is negative while best was {best:.1f} (food web collapsed)"})
        elif best is not None and not _is_nan(best) and best > 0 and eval_recent < EVAL_REGRESS_FRAC * best:
            checks.append({"name": "eval_score", "level": "warn",
                           "detail": f"recent eval {eval_recent:.1f} fell well below best {best:.1f}"})
        else:
            checks.append({"name": "eval_score", "level": "ok",
                           "detail": f"recent eval {eval_recent:.1f} (best {best:.1f})"})

    # --- Direction --------------------------------------------------------
    if since_best <= k:
        direction = "improving"
    elif math.isnan(eval_recent) or math.isnan(eval_earlier):
        direction = "unknown"
    elif eval_recent >= eval_earlier - max(2.0, 0.05 * abs(eval_earlier)):
        direction = "stable"
    else:
        direction = "regressing"

    # --- Overall level & verdict -----------------------------------------
    worst = "ok"
    for c in checks:
        if c["level"] == "crit":
            worst = "crit"
            break
        if c["level"] == "warn":
            worst = "warn"

    if early and worst != "crit":
        level = "early"
        verdict = (f"EARLY (u={update} < {EARLY_UPDATES}): too soon to judge. No NaN, training is warming up. "
                   f"eval_score is expected to be low/negative here. Re-check around u=1500-2500.")
    elif worst == "crit":
        level = "crit"
        verdict = f"COLLAPSED / collapsing at u={update}. " + "; ".join(c["detail"] for c in checks if c["level"] == "crit")
    elif worst == "warn":
        level = "warn"
        verdict = (f"AT RISK at u={update} ({direction}). "
                   + "; ".join(c["detail"] for c in checks if c["level"] == "warn"))
    else:
        level = "ok"
        verdict = f"HEALTHY at u={update}, {direction} (eval {eval_recent:.1f}, best {best:.1f})."

    return {
        "update": update,
        "eval_score": eval_recent,
        "best": best,
        "level": level,
        "direction": direction,
        "since_best": since_best,
        "verdict": verdict,
        "checks": checks,
    }


# --- pretty printer ----------------------------------------------------------
_ICON = {"ok": "[ OK ]", "warn": "[WARN]", "crit": "[CRIT]", "early": "[ .. ]"}
_HEAD = {"ok": "HEALTHY", "warn": "AT RISK", "crit": "COLLAPSED", "early": "EARLY"}


def print_report(report: dict, name: str | None = None) -> None:
    head = _HEAD.get(report["level"], "?")
    label = f" {name}" if name else ""
    print(f"\n=== training health{label} : {head} ===")
    print(report["verdict"])
    print(f"  update={report['update']}  recent_eval={report['eval_score']:.1f}  "
          f"best={report['best']:.1f}  direction={report['direction']}  since_best={report.get('since_best','?')}")
    for c in report["checks"]:
        print(f"  {_ICON.get(c['level'],'?')} {c['name']:16s} {c['detail']}")


def _main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    worst_rank = {"ok": 0, "early": 0, "warn": 1, "crit": 2}
    overall = 0
    for path in argv:
        try:
            hist = load_history(path)
            report = assess(hist)
        except FileNotFoundError as e:
            print(f"\n=== {path} : ERROR ===\n  {e}")
            overall = max(overall, 2)
            continue
        print_report(report, name=str(path))
        overall = max(overall, worst_rank.get(report["level"], 0))
    # Exit non-zero if anything is collapsing (handy for scripts/cron alerts).
    return 1 if overall >= 2 else 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
