from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


BASELINE_METHODS = [
    {
        "key": "regular",
        "label": "Regular SFT",
        "short": "SFT",
        "family": "baseline",
        "description": "Plain next-token supervised fine-tuning.",
        "paper": 0.5729,
        "paper_std": 0.0532,
    },
    {
        "key": "jepa_l1_p1",
        "label": "Original LLM-JEPA",
        "short": "JEPA",
        "family": "baseline",
        "description": "Paper-style LLM-JEPA with lambda=1, k=1, no cap-anchor additions.",
        "paper": 0.7146,
        "paper_std": 0.0134,
    },
]

SPHERE_VARIANTS = [
    {
        "key": "own_d0",
        "label": "Own D0",
        "short": "Own D0",
        "family": "frozen-own",
        "cap_mode": "own",
        "anchor_type": "frozen",
        "sigma_max": 0.0,
        "lambda_cap": 1.0,
        "description": "Frozen own-view cap anchor, clean sigma=0 control.",
    },
    {
        "key": "own_d1",
        "label": "Own D1",
        "short": "Own D1",
        "family": "frozen-own",
        "cap_mode": "own",
        "anchor_type": "frozen",
        "sigma_max": 0.5,
        "lambda_cap": 1.0,
        "description": "Frozen own-view cap anchor with spherical noise.",
    },
    {
        "key": "cross_d0",
        "label": "Cross D0",
        "short": "Cross D0",
        "family": "frozen-cross",
        "cap_mode": "cross",
        "anchor_type": "frozen",
        "sigma_max": 0.0,
        "lambda_cap": 1.0,
        "description": "Frozen text-to-code cross-view cap anchor, clean sigma=0 control.",
    },
    {
        "key": "cross_d1",
        "label": "Cross D1",
        "short": "Cross D1",
        "family": "frozen-cross",
        "cap_mode": "cross",
        "anchor_type": "frozen",
        "sigma_max": 0.5,
        "lambda_cap": 1.0,
        "description": "Frozen text-to-code cross-view cap anchor with spherical noise.",
    },
    {
        "key": "both_d0",
        "label": "Both D0",
        "short": "Both D0",
        "family": "frozen-both",
        "cap_mode": "both",
        "anchor_type": "frozen",
        "sigma_max": 0.0,
        "lambda_cap": 1.0,
        "description": "Frozen own-view plus symmetric cross-view cap anchors, clean sigma=0 control.",
    },
    {
        "key": "both_d1",
        "label": "Both D1",
        "short": "Both D1",
        "family": "frozen-both",
        "cap_mode": "both",
        "anchor_type": "frozen",
        "sigma_max": 0.5,
        "lambda_cap": 1.0,
        "description": "Frozen own-view plus symmetric cross-view cap anchors with spherical noise.",
    },
    {
        "key": "c_own_d1",
        "label": "Co-trained Own D1",
        "short": "C Own",
        "family": "moving-anchor",
        "cap_mode": "own",
        "anchor_type": "cotrained",
        "sigma_max": 0.5,
        "lambda_cap": 1.0,
        "description": "Own-view cap target from the same trainable model.",
    },
    {
        "key": "ema_own_d1",
        "label": "EMA Own D1",
        "short": "EMA Own",
        "family": "moving-anchor",
        "cap_mode": "own",
        "anchor_type": "ema",
        "sigma_max": 0.5,
        "lambda_cap": 1.0,
        "description": "Own-view cap target from an EMA copy of the model.",
    },
    {
        "key": "c_own_d0_l005",
        "label": "Co-trained Own D0 lambda 0.05",
        "short": "C Own D0 .05",
        "family": "dose",
        "cap_mode": "own",
        "anchor_type": "cotrained",
        "sigma_max": 0.0,
        "lambda_cap": 0.05,
        "description": "Lower-dose co-trained own-view clean cap target, matched sigma=0 control for the low-dose noisy run.",
    },
    {
        "key": "c_own_d1_l005",
        "label": "Co-trained Own D1 lambda 0.05",
        "short": "C Own .05",
        "family": "dose",
        "cap_mode": "own",
        "anchor_type": "cotrained",
        "sigma_max": 0.5,
        "lambda_cap": 0.05,
        "description": "Lower-dose co-trained own-view cap target.",
    },
    {
        "key": "c_own_d1_l010",
        "label": "Co-trained Own D1 lambda 0.10",
        "short": "C Own .10",
        "family": "dose",
        "cap_mode": "own",
        "anchor_type": "cotrained",
        "sigma_max": 0.5,
        "lambda_cap": 0.10,
        "description": "Lower-dose co-trained own-view cap target.",
    },
    {
        "key": "c_own_d1_l025",
        "label": "Co-trained Own D1 lambda 0.25",
        "short": "C Own .25",
        "family": "dose",
        "cap_mode": "own",
        "anchor_type": "cotrained",
        "sigma_max": 0.5,
        "lambda_cap": 0.25,
        "description": "Lower-dose co-trained own-view cap target.",
    },
    {
        "key": "c_own_d1_l050",
        "label": "Co-trained Own D1 lambda 0.50",
        "short": "C Own .50",
        "family": "dose",
        "cap_mode": "own",
        "anchor_type": "cotrained",
        "sigma_max": 0.5,
        "lambda_cap": 0.50,
        "description": "Lower-dose co-trained own-view cap target.",
    },
    {
        "key": "c_own_d1_delay1_ramp3",
        "label": "Co-trained Own D1 delayed/ramped",
        "short": "C Own ramp",
        "family": "schedule",
        "cap_mode": "own",
        "anchor_type": "cotrained",
        "sigma_max": 0.5,
        "lambda_cap": 1.0,
        "description": "Co-trained own-view cap target with one epoch delay and three epoch warmup.",
    },
    {
        "key": "own_d0_l025",
        "label": "Own D0 lambda 0.25",
        "short": "Own D0 .25",
        "family": "dose",
        "cap_mode": "own",
        "anchor_type": "frozen",
        "sigma_max": 0.0,
        "lambda_cap": 0.25,
        "description": "Lower-dose clean frozen own-view cap target.",
    },
    {
        "key": "own_d1_l025",
        "label": "Own D1 lambda 0.25",
        "short": "Own D1 .25",
        "family": "dose",
        "cap_mode": "own",
        "anchor_type": "frozen",
        "sigma_max": 0.5,
        "lambda_cap": 0.25,
        "description": "Lower-dose noisy frozen own-view cap target.",
    },
    {
        "key": "cross_d0_l025",
        "label": "Cross D0 lambda 0.25",
        "short": "Cross D0 .25",
        "family": "dose",
        "cap_mode": "cross",
        "anchor_type": "frozen",
        "sigma_max": 0.0,
        "lambda_cap": 0.25,
        "description": "Lower-dose clean frozen cross-view cap target.",
    },
    {
        "key": "cross_d1_l025",
        "label": "Cross D1 lambda 0.25",
        "short": "Cross D1 .25",
        "family": "dose",
        "cap_mode": "cross",
        "anchor_type": "frozen",
        "sigma_max": 0.5,
        "lambda_cap": 0.25,
        "description": "Lower-dose noisy frozen cross-view cap target.",
    },
]

METHOD_ORDER = BASELINE_METHODS + SPHERE_VARIANTS
METHODS_BY_KEY = {row["key"]: row for row in METHOD_ORDER}

RESEARCH_QUESTIONS = [
    {
        "id": "rq1",
        "question": "Does the spherical cap loss improve the already-replicated original LLM-JEPA baseline?",
        "primary_contrast": "Best completed cap variant minus original LLM-JEPA on matched seeds.",
        "decision_rule": "Treat as promising only if the paired mean delta is positive and at least four of five seeds are non-negative.",
    },
    {
        "id": "rq2",
        "question": "Does spherical noise help independently of anchor/view choice?",
        "primary_contrast": "Noisy minus clean within the same frozen cap family: own_d1-own_d0, cross_d1-cross_d0, both_d1-both_d0.",
        "decision_rule": "Noise is supported only by a positive paired mean in at least two families or one family with seed-level consistency.",
    },
    {
        "id": "rq3",
        "question": "Is the useful target an own-view reconstruction, a cross-view prediction, or both?",
        "primary_contrast": "cross_d1-own_d1 and both_d1-best(own_d1,cross_d1) on matched seeds.",
        "decision_rule": "Prefer the simplest view target that improves exact match without hurting seed consistency.",
    },
    {
        "id": "rq4",
        "question": "Do moving anchors avoid over-constraining generation, or do fixed anchors provide the better signal?",
        "primary_contrast": "c_own_d1-own_d1 and ema_own_d1-own_d1.",
        "decision_rule": "Moving anchors are useful only if they recover exact match while preserving a positive or neutral cap effect.",
    },
    {
        "id": "rq5",
        "question": "If cap variants underperform, is the failure due to cap-loss weight/schedule conflict rather than the spherical idea?",
        "primary_contrast": "Follow-up lambda/schedule sweep for the least-bad cap family.",
        "decision_rule": "Run lambda_cap in {0.05, 0.1, 0.25, 0.5} and a delayed-cap schedule before rejecting the mechanism.",
    },
]

PALETTE = {
    "regular": "#2b6f8e",
    "jepa_l1_p1": "#bd6b28",
    "own_d0": "#4b8b61",
    "own_d1": "#246f48",
    "cross_d0": "#7d6ab6",
    "cross_d1": "#59449a",
    "both_d0": "#c58b39",
    "both_d1": "#9b6425",
    "c_own_d1": "#b84f5f",
    "ema_own_d1": "#d08b90",
    "c_own_d0_l005": "#355f6d",
    "c_own_d1_l005": "#8f3f4c",
    "c_own_d1_l010": "#a84657",
    "c_own_d1_l025": "#c25a68",
    "c_own_d1_l050": "#d47a84",
    "c_own_d1_delay1_ramp3": "#6b8a99",
    "own_d0_l025": "#71a77c",
    "own_d1_l025": "#3d8f60",
    "cross_d0_l025": "#9c8aca",
    "cross_d1_l025": "#715bb0",
}


def setup_theme() -> None:
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams.update(
        {
            "figure.facecolor": "#f6f7f2",
            "axes.facecolor": "#ffffff",
            "savefig.facecolor": "#f6f7f2",
            "axes.edgecolor": "#dfe5dc",
            "grid.color": "#d9ded8",
            "font.family": "DejaVu Sans",
        }
    )


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def values(rows: list[dict[str, Any]], metric: str) -> list[float]:
    return [float(row[metric]) for row in rows if finite(row.get(metric))]


def mean_or_nan(items: list[float]) -> float:
    return mean(items) if items else math.nan


def std_or_nan(items: list[float]) -> float:
    if not items:
        return math.nan
    return stdev(items) if len(items) > 1 else 0.0


def pct(value: Any, digits: int = 2) -> str:
    if not finite(value):
        return "n/a"
    return f"{float(value) * 100:.{digits}f}%"


def pp(value: Any, digits: int = 2) -> str:
    if not finite(value):
        return "n/a"
    return f"{float(value) * 100:.{digits}f} pp"


def num(value: Any) -> str:
    if not finite(value):
        return "n/a"
    value = float(value)
    if abs(value) >= 1000:
        return f"{value:.0f}"
    if abs(value) >= 10:
        return f"{value:.2f}"
    return f"{value:.3f}"


def clip(text: Any, limit: int = 180) -> str:
    value = " ".join(str(text or "").split())
    if len(value) <= limit:
        return value
    return value[: limit - 1] + "..."


def trainer_summary(model_dir: Path) -> dict[str, Any]:
    state = read_json(model_dir / "trainer_state.json")
    if state is None:
        return {"train_status": "missing"}
    history = []
    final = {}
    for row in state.get("log_history", []):
        if "loss" in row:
            history.append(
                {
                    "step": int(row.get("step", 0)),
                    "epoch": float(row.get("epoch", 0.0)),
                    "loss": float(row["loss"]),
                    "learning_rate": float(row.get("learning_rate", 0.0)),
                    "grad_norm": float(row.get("grad_norm", 0.0)),
                }
            )
        if "train_loss" in row:
            final = row
    out = {
        "train_status": "trained",
        "global_step": int(state.get("global_step", 0)),
        "num_train_epochs": float(state.get("num_train_epochs", 0.0)),
        "history": history,
    }
    if final:
        out.update(
            {
                "train_loss": float(final.get("train_loss", math.nan)),
                "train_runtime": float(final.get("train_runtime", math.nan)),
                "train_samples_per_second": float(final.get("train_samples_per_second", math.nan)),
            }
        )
    if history:
        out["final_log_loss"] = float(history[-1]["loss"])
        out["min_log_loss"] = min(float(point["loss"]) for point in history)
    return out


def eval_summary(eval_file: Path) -> dict[str, Any]:
    rows = read_jsonl(eval_file)
    if not rows:
        return {"eval_status": "missing", "eval_count": 0}
    matches = sum(1 for row in rows if row.get("exact_match"))
    first_line = 0
    contains = 0
    mismatches = []
    for row in rows:
        generated = str(row.get("generated_response", "")).strip()
        target = str(row.get("ground_truth", "")).strip()
        generated_first = next((line.strip() for line in generated.splitlines() if line.strip()), "")
        first_line += int(generated_first == target)
        contains += int(bool(target) and target in generated)
        if not row.get("exact_match") and len(mismatches) < 5:
            mismatches.append(
                {
                    "index": int(row.get("index", -1)),
                    "generated_response": clip(generated),
                    "ground_truth": clip(target),
                }
            )
    return {
        "eval_status": "evaluated",
        "eval_count": len(rows),
        "eval_matches": matches,
        "eval_exact_match": matches / len(rows),
        "eval_first_line_exact": first_line / len(rows),
        "eval_contains_target": contains / len(rows),
        "sample_mismatches": mismatches,
    }


def collect_records(
    baseline_root: Path,
    sphere_root: Path,
    seeds: list[int],
) -> list[dict[str, Any]]:
    records = []
    for seed in seeds:
        for method in BASELINE_METHODS:
            key = method["key"]
            model_dir = baseline_root / f"seed{seed}_{key}"
            eval_file = baseline_root / "eval" / f"seed{seed}_{key}.jsonl"
            row = {
                **method,
                "seed": seed,
                "method": key,
                "model_dir": str(model_dir),
                "eval_file": str(eval_file),
            }
            row.update(trainer_summary(model_dir))
            row.update(eval_summary(eval_file))
            records.append(row)
        for variant in SPHERE_VARIANTS:
            key = variant["key"]
            model_dir = sphere_root / f"seed{seed}_{key}"
            eval_file = sphere_root / "eval" / f"seed{seed}_{key}.jsonl"
            row = {
                **variant,
                "seed": seed,
                "method": key,
                "model_dir": str(model_dir),
                "eval_file": str(eval_file),
            }
            row.update(trainer_summary(model_dir))
            row.update(eval_summary(eval_file))
            records.append(row)
    return records


def aggregate(records: list[dict[str, Any]], seeds: list[int]) -> dict[str, Any]:
    by_method = {}
    for method in METHOD_ORDER:
        key = method["key"]
        rows = [row for row in records if row["method"] == key]
        metric_values = values(rows, "eval_exact_match")
        by_method[key] = {
            "label": method["label"],
            "short": method["short"],
            "family": method["family"],
            "n": len(metric_values),
            "mean_exact_match": mean_or_nan(metric_values),
            "std_exact_match": std_or_nan(metric_values),
            "values": metric_values,
            "paper_exact_match": method.get("paper"),
            "paper_std": method.get("paper_std"),
        }

    by_seed_method = {(row["seed"], row["method"]): row for row in records}
    seed_rows = []
    for seed in seeds:
        row = {"seed": seed}
        for method in METHOD_ORDER:
            value = by_seed_method.get((seed, method["key"]), {}).get("eval_exact_match")
            row[method["key"]] = float(value) if finite(value) else math.nan
        seed_rows.append(row)

    paired_vs_jepa = {}
    jepa_key = "jepa_l1_p1"
    for method in METHOD_ORDER:
        key = method["key"]
        if key == jepa_key:
            continue
        deltas = []
        seed_deltas = []
        for seed in seeds:
            left = by_seed_method.get((seed, key), {}).get("eval_exact_match")
            right = by_seed_method.get((seed, jepa_key), {}).get("eval_exact_match")
            if finite(left) and finite(right):
                delta = float(left) - float(right)
                deltas.append(delta)
                seed_deltas.append({"seed": seed, "delta": delta, "left": float(left), "right": float(right)})
        paired_vs_jepa[key] = {
            "label": method["label"],
            "n": len(deltas),
            "mean_delta": mean_or_nan(deltas),
            "std_delta": std_or_nan(deltas),
            "nonnegative_seeds": sum(1 for value in deltas if value >= 0),
            "max_possible_nonnegative_seeds": sum(1 for value in deltas if value >= 0) + len(seeds) - len(deltas),
            "can_pass_four_of_five": sum(1 for value in deltas if value >= 0) + len(seeds) - len(deltas) >= 4,
            "seed_deltas": seed_deltas,
        }

    contrasts = {}
    for left, right, name in [
        ("own_d1", "own_d0", "noise_own_d1_minus_d0"),
        ("cross_d1", "cross_d0", "noise_cross_d1_minus_d0"),
        ("both_d1", "both_d0", "noise_both_d1_minus_d0"),
        ("c_own_d1_l005", "c_own_d0_l005", "noise_cotrained_lowdose_d1_minus_d0"),
        ("cross_d1", "own_d1", "cross_d1_minus_own_d1"),
        ("both_d1", "own_d1", "both_d1_minus_own_d1"),
        ("c_own_d1", "own_d1", "cotrained_own_minus_frozen_own"),
        ("ema_own_d1", "own_d1", "ema_own_minus_frozen_own"),
    ]:
        deltas = []
        seed_deltas = []
        for seed in seeds:
            l_val = by_seed_method.get((seed, left), {}).get("eval_exact_match")
            r_val = by_seed_method.get((seed, right), {}).get("eval_exact_match")
            if finite(l_val) and finite(r_val):
                delta = float(l_val) - float(r_val)
                deltas.append(delta)
                seed_deltas.append({"seed": seed, "delta": delta, "left": float(l_val), "right": float(r_val)})
        contrasts[name] = {
            "left": left,
            "right": right,
            "n": len(deltas),
            "mean_delta": mean_or_nan(deltas),
            "std_delta": std_or_nan(deltas),
            "nonnegative_seeds": sum(1 for value in deltas if value >= 0),
            "seed_deltas": seed_deltas,
        }

    completed_sphere = [
        (key, data)
        for key, data in by_method.items()
        if key not in {"regular", "jepa_l1_p1"} and finite(data.get("mean_exact_match"))
    ]
    best_sphere = max(completed_sphere, key=lambda item: float(item[1]["mean_exact_match"])) if completed_sphere else None
    best_sphere_key = best_sphere[0] if best_sphere else None
    best_delta = paired_vs_jepa.get(best_sphere_key, {}).get("mean_delta") if best_sphere_key else math.nan
    return {
        "methods": by_method,
        "seed_rows": seed_rows,
        "paired_vs_jepa": paired_vs_jepa,
        "contrasts": contrasts,
        "best_sphere_method": best_sphere_key,
        "best_sphere_delta_vs_jepa": best_delta,
    }


def build_interpretation(summary: dict[str, Any]) -> list[str]:
    aggregate_data = summary["aggregate"]
    methods = aggregate_data["methods"]
    observations = []
    regular = methods["regular"]
    jepa = methods["jepa_l1_p1"]
    if finite(regular.get("mean_exact_match")) and finite(jepa.get("mean_exact_match")):
        delta = float(jepa["mean_exact_match"]) - float(regular["mean_exact_match"])
        observations.append(
            f"Original LLM-JEPA is the replication gate: it is {pp(delta)} over regular SFT on completed seeds."
        )
    best_key = aggregate_data.get("best_sphere_method")
    if best_key:
        best = methods[best_key]
        delta = aggregate_data.get("best_sphere_delta_vs_jepa")
        observations.append(
            f"Best completed cap variant so far is {best['label']} at {pct(best['mean_exact_match'])}, with matched-seed delta {pp(delta)} versus original LLM-JEPA."
        )
        if finite(delta) and float(delta) < 0:
            observations.append(
                "If this remains negative after all seeds finish, the next clean test is a cap-loss weight and schedule sweep, not a broader architecture change."
            )
    else:
        observations.append("No cap variant has a completed evaluation yet.")
    failed_by_rule = []
    for method in SPHERE_VARIANTS:
        key = method["key"]
        data = aggregate_data["paired_vs_jepa"].get(key, {})
        if data.get("n", 0) >= 3 and not data.get("can_pass_four_of_five", True):
            failed_by_rule.append(method["short"])
    if failed_by_rule:
        observations.append(
            "Under the 4-of-5 seed-consistency rule, these cap variants can no longer pass on exact match: "
            + ", ".join(failed_by_rule)
            + "."
        )
    for name, label in [
        ("noise_own_d1_minus_d0", "Own-view noise"),
        ("noise_cross_d1_minus_d0", "Cross-view noise"),
        ("noise_both_d1_minus_d0", "Own+cross noise"),
        ("noise_cotrained_lowdose_d1_minus_d0", "Low-dose co-trained noise"),
    ]:
        data = aggregate_data["contrasts"].get(name, {})
        if finite(data.get("mean_delta")):
            observations.append(f"{label} paired delta is {pp(data['mean_delta'])} over {data['n']} completed seeds.")
    return observations


def next_experiments(summary: dict[str, Any]) -> list[dict[str, str]]:
    best_key = summary["aggregate"].get("best_sphere_method")
    best_label = METHODS_BY_KEY.get(best_key, {}).get("label", "the least-bad cap variant")
    best_delta = summary["aggregate"].get("best_sphere_delta_vs_jepa")
    priority = "high" if finite(best_delta) and float(best_delta) < 0 else "medium"
    return [
        {
            "name": "Lambda-cap dose response",
            "priority": priority,
            "question": "Is the cap loss useful only below lambda_cap=1.0?",
            "design": f"Run {best_label} at lambda_cap in {{0.05, 0.1, 0.25, 0.5}} with the same five seeds and the same original LLM-JEPA protocol.",
            "success_rule": "A setting is credible only if its matched-seed delta versus original LLM-JEPA is non-negative and not driven by one seed.",
        },
        {
            "name": "Delayed or annealed cap schedule",
            "priority": priority,
            "question": "Does the cap objective conflict with early language-model adaptation?",
            "design": "Hold lambda_cap=0 for epoch 1, then ramp to the selected lambda over epochs 2-4; compare against constant lambda on the same seeds.",
            "success_rule": "Schedule helps only if it improves exact match without erasing the noise/control contrast.",
        },
        {
            "name": "Geometry versus generation diagnostic",
            "priority": "high",
            "question": "Are cap variants improving hidden-state geometry while hurting regex generation?",
            "design": "For SFT, original LLM-JEPA, and the best cap variant, extract text/code hidden states for a fixed 512-example test subset and plot PCA, cosine histograms, retrieval heatmaps, RankMe, and paired-vs-unpaired cosine.",
            "success_rule": "A geometry-only win is not sufficient; it just tells us whether the objective is doing the intended representation work.",
        },
        {
            "name": "Cross-view symmetry ablation",
            "priority": "medium",
            "question": "Is code-to-text reconstruction helpful or just extra pressure?",
            "design": "If cross_d1 beats own_d1, run a cross symmetric variant and compare text-to-code only versus text-to-code plus code-to-text.",
            "success_rule": "Keep symmetry only if exact match and seed consistency improve over the one-way cross variant.",
        },
    ]


def build_summary(args: argparse.Namespace) -> dict[str, Any]:
    records = collect_records(args.baseline_root, args.sphere_root, args.seeds)
    out = {
        "config": {
            "experiment": args.stem,
            "base_model": args.base_model,
            "dataset": "NL-RX-SYNTH",
            "train_file": str(args.train_file),
            "test_file": str(args.test_file),
            "seeds": args.seeds,
            "epochs": 4,
            "learning_rate": 2e-5,
            "global_batch_size": 128,
            "jepa_lambda": 1.0,
            "predictors": 1,
            "last_token": -2,
            "primary_metric": "strict exact-match string equality on synth_test.jsonl",
        },
        "research_questions": RESEARCH_QUESTIONS,
        "records": records,
        "aggregate": aggregate(records, args.seeds),
    }
    out["interpretation"] = build_interpretation(out)
    out["next_experiments"] = next_experiments(out)
    return out


def save_fig(fig: plt.Figure, path: Path, report_dir: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return str(path.relative_to(report_dir))


def plot_method_means(summary: dict[str, Any], assets_dir: Path, report_dir: Path, stem: str) -> str:
    methods = summary["aggregate"]["methods"]
    keys = [method["key"] for method in METHOD_ORDER if methods[method["key"]]["n"] > 0]
    labels = [methods[key]["short"] for key in keys]
    means = [methods[key]["mean_exact_match"] * 100 for key in keys]
    stds = [methods[key]["std_exact_match"] * 100 for key in keys]
    fig, ax = plt.subplots(figsize=(12.2, 5.4))
    colors = [PALETTE.get(key, "#777777") for key in keys]
    ax.bar(labels, means, color=colors, edgecolor="#24312b", linewidth=0.6)
    ax.errorbar(labels, means, yerr=stds, fmt="none", ecolor="#1f2522", capsize=3, linewidth=1.2)
    paper_x = []
    paper_y = []
    for idx, key in enumerate(keys):
        paper = methods[key].get("paper_exact_match")
        if finite(paper):
            paper_x.append(idx)
            paper_y.append(float(paper) * 100)
    if paper_x:
        ax.scatter(paper_x, paper_y, color="#171d1a", marker="D", s=54, label="paper mean")
        ax.legend(frameon=False, loc="upper right")
    ax.set_ylabel("Exact match (%)")
    ax.set_title("Mean Exact Match By Method")
    ax.set_ylim(0, max(80, max(means + [0]) + 8))
    ax.tick_params(axis="x", labelrotation=35)
    ax.grid(axis="y", color="#d9ded8", linewidth=0.8)
    return save_fig(fig, assets_dir / f"{stem}_method_means.png", report_dir)


def plot_delta_vs_jepa(summary: dict[str, Any], assets_dir: Path, report_dir: Path, stem: str) -> str | None:
    paired = summary["aggregate"]["paired_vs_jepa"]
    keys = [method["key"] for method in METHOD_ORDER if method["key"] != "jepa_l1_p1" and paired.get(method["key"], {}).get("n", 0) > 0]
    if not keys:
        return None
    labels = [METHODS_BY_KEY[key]["short"] for key in keys]
    means = [paired[key]["mean_delta"] * 100 for key in keys]
    fig, ax = plt.subplots(figsize=(12.0, 5.6))
    ax.axhline(0, color="#1f2522", linewidth=1.2)
    ax.bar(labels, means, color=[PALETTE.get(key, "#777777") for key in keys], alpha=0.78, edgecolor="#24312b", linewidth=0.5)
    for idx, key in enumerate(keys):
        for point in paired[key]["seed_deltas"]:
            jitter = ((int(point["seed"]) % 7) - 3) * 0.018
            ax.scatter(idx + jitter, point["delta"] * 100, color="#101714", s=22, alpha=0.75)
    ax.set_ylabel("Exact-match delta versus original LLM-JEPA (pp)")
    ax.set_title("Matched-Seed Deltas Against Original LLM-JEPA")
    ax.tick_params(axis="x", labelrotation=35)
    ax.grid(axis="y", color="#d9ded8", linewidth=0.8)
    return save_fig(fig, assets_dir / f"{stem}_delta_vs_jepa.png", report_dir)


def plot_clean_noisy(summary: dict[str, Any], assets_dir: Path, report_dir: Path, stem: str) -> str | None:
    seed_rows = summary["aggregate"]["seed_rows"]
    pairs = [
        ("own_d0", "own_d1", "Own"),
        ("cross_d0", "cross_d1", "Cross"),
        ("both_d0", "both_d1", "Both"),
        ("c_own_d0_l005", "c_own_d1_l005", "C Own .05"),
    ]
    if not any(finite(row.get(left)) and finite(row.get(right)) for row in seed_rows for left, right, _ in pairs):
        return None
    fig, axes = plt.subplots(1, len(pairs), figsize=(4.4 * len(pairs), 4.8), sharey=True)
    for ax, (left, right, title) in zip(axes, pairs):
        for row in seed_rows:
            if finite(row.get(left)) and finite(row.get(right)):
                y0 = float(row[left]) * 100
                y1 = float(row[right]) * 100
                ax.plot([0, 1], [y0, y1], color="#88948d", linewidth=1.4)
                ax.scatter([0], [y0], color=PALETTE[left], s=42)
                ax.scatter([1], [y1], color=PALETTE[right], s=42)
        ax.set_xticks([0, 1], ["clean", "noisy"])
        ax.set_title(title)
        ax.grid(axis="y", color="#d9ded8", linewidth=0.8)
    axes[0].set_ylabel("Exact match (%)")
    fig.suptitle("Clean Versus Spherical-Noise Controls")
    return save_fig(fig, assets_dir / f"{stem}_clean_noisy.png", report_dir)


def plot_cotrained_dose_response(summary: dict[str, Any], assets_dir: Path, report_dir: Path, stem: str) -> str | None:
    paired = summary["aggregate"]["paired_vs_jepa"]
    series = [
        ("0.05", "c_own_d1_l005"),
        ("0.10", "c_own_d1_l010"),
        ("0.25", "c_own_d1_l025"),
        ("0.50", "c_own_d1_l050"),
        ("1.00", "c_own_d1"),
        ("delay/ramp", "c_own_d1_delay1_ramp3"),
    ]
    active = [(label, key) for label, key in series if paired.get(key, {}).get("n", 0) > 0]
    if not active:
        return None
    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    ax.axhline(0, color="#1f2522", linewidth=1.2)
    xs = list(range(len(active)))
    means = [paired[key]["mean_delta"] * 100 for _, key in active]
    ax.plot(xs, means, color="#b84f5f", linewidth=2.4, marker="o", markersize=7)
    for idx, (_, key) in enumerate(active):
        for point in paired[key]["seed_deltas"]:
            jitter = ((int(point["seed"]) % 7) - 3) * 0.025
            ax.scatter(idx + jitter, point["delta"] * 100, color="#101714", s=24, alpha=0.75)
    ax.set_xticks(xs, [label for label, _ in active])
    ax.set_ylabel("Exact-match delta versus original LLM-JEPA (pp)")
    ax.set_xlabel("Co-trained own cap setting")
    ax.set_title("Co-trained Own-View Cap Dose Response")
    ax.grid(axis="y", color="#d9ded8", linewidth=0.8)
    return save_fig(fig, assets_dir / f"{stem}_cotrained_dose_response.png", report_dir)


def plot_seed_heatmap(summary: dict[str, Any], assets_dir: Path, report_dir: Path, stem: str) -> str | None:
    seed_rows = summary["aggregate"]["seed_rows"]
    keys = [method["key"] for method in METHOD_ORDER if any(finite(row.get(method["key"])) for row in seed_rows)]
    if not keys:
        return None
    matrix = []
    y_labels = []
    for row in seed_rows:
        if any(finite(row.get(key)) for key in keys):
            matrix.append([float(row[key]) * 100 if finite(row.get(key)) else math.nan for key in keys])
            y_labels.append(str(row["seed"]))
    fig, ax = plt.subplots(figsize=(12.6, max(3.4, 0.56 * len(matrix) + 2.0)))
    sns.heatmap(
        matrix,
        ax=ax,
        annot=True,
        fmt=".1f",
        cmap=sns.color_palette("crest", as_cmap=True),
        vmin=0,
        vmax=max(80, max(v for row in matrix for v in row if math.isfinite(v))),
        xticklabels=[METHODS_BY_KEY[key]["short"] for key in keys],
        yticklabels=y_labels,
        cbar_kws={"label": "Exact match (%)"},
    )
    ax.set_xlabel("Method")
    ax.set_ylabel("Seed")
    ax.set_title("Per-Seed Exact Match Heatmap")
    ax.tick_params(axis="x", labelrotation=35)
    return save_fig(fig, assets_dir / f"{stem}_seed_heatmap.png", report_dir)


def plot_training_loss(summary: dict[str, Any], assets_dir: Path, report_dir: Path, stem: str) -> str | None:
    records = [row for row in summary["records"] if row.get("history")]
    if not records:
        return None
    fig, ax = plt.subplots(figsize=(12.0, 5.4))
    for method in METHOD_ORDER:
        rows = [row for row in records if row["method"] == method["key"]]
        by_step: dict[int, list[float]] = {}
        for row in rows:
            for point in row.get("history", []):
                by_step.setdefault(int(point["step"]), []).append(float(point["loss"]))
        if not by_step:
            continue
        steps = sorted(by_step)
        losses = [mean(by_step[step]) for step in steps]
        ax.plot(steps, losses, label=method["short"], color=PALETTE.get(method["key"], "#777777"), linewidth=2.0)
    ax.set_xlabel("Step")
    ax.set_ylabel("Logged training loss")
    ax.set_title("Mean Logged Training Loss By Variant")
    ax.legend(frameon=False, ncols=3, fontsize=9)
    ax.grid(color="#d9ded8", linewidth=0.8)
    return save_fig(fig, assets_dir / f"{stem}_training_loss.png", report_dir)


def make_plots(summary: dict[str, Any], report_dir: Path, stem: str) -> dict[str, str]:
    assets_dir = report_dir / "assets" / "sphere_ablation"
    plots: dict[str, str | None] = {
        "method_means": plot_method_means(summary, assets_dir, report_dir, stem),
        "delta_vs_jepa": plot_delta_vs_jepa(summary, assets_dir, report_dir, stem),
        "clean_noisy": plot_clean_noisy(summary, assets_dir, report_dir, stem),
        "cotrained_dose_response": plot_cotrained_dose_response(summary, assets_dir, report_dir, stem),
        "seed_heatmap": plot_seed_heatmap(summary, assets_dir, report_dir, stem),
        "training_loss": plot_training_loss(summary, assets_dir, report_dir, stem),
    }
    return {key: value for key, value in plots.items() if value}


def render_cards(summary: dict[str, Any]) -> str:
    methods = summary["aggregate"]["methods"]
    best_key = summary["aggregate"].get("best_sphere_method")
    best = methods.get(best_key) if best_key else None
    completed = sum(1 for row in summary["records"] if row.get("eval_status") == "evaluated")
    total = len(summary["records"])
    cards = [
        ("Completed evals", f"{completed}/{total}", "Each row is one seed and method on synth_test.jsonl."),
        ("Regular SFT", pct(methods["regular"].get("mean_exact_match")), "Mean over completed baseline seeds."),
        ("Original LLM-JEPA", pct(methods["jepa_l1_p1"].get("mean_exact_match")), "Replication baseline; cap variants must beat this, not just SFT."),
        ("Best cap variant", f"{best['label']} {pct(best['mean_exact_match'])}" if best else "n/a", "Highest completed mean exact match among cap variants."),
        ("Best cap - JEPA", pp(summary["aggregate"].get("best_sphere_delta_vs_jepa")), "Matched-seed mean delta against original LLM-JEPA."),
    ]
    return "".join(
        "<article class='stat'>"
        f"<div class='label'>{html.escape(label)}</div>"
        f"<div class='value'>{html.escape(value)}</div>"
        f"<div class='note'>{html.escape(note)}</div>"
        "</article>"
        for label, value, note in cards
    )


def render_research_questions(summary: dict[str, Any]) -> str:
    rows = []
    for rq in summary["research_questions"]:
        rows.append(
            "<tr>"
            f"<td><code>{html.escape(rq['id'])}</code></td>"
            f"<td>{html.escape(rq['question'])}</td>"
            f"<td>{html.escape(rq['primary_contrast'])}</td>"
            f"<td>{html.escape(rq['decision_rule'])}</td>"
            "</tr>"
        )
    return (
        "<article class='card wide'><h2>Research Questions</h2>"
        "<div class='table-wrap'><table><thead><tr><th>ID</th><th>Question</th><th>Primary contrast</th><th>Decision rule</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div></article>"
    )


def render_interpretation(summary: dict[str, Any]) -> str:
    items = "".join(f"<li>{html.escape(item)}</li>" for item in summary["interpretation"])
    return f"<article class='card wide'><h2>Current Interpretation</h2><ul>{items}</ul></article>"


def render_next_experiments(summary: dict[str, Any]) -> str:
    rows = []
    for item in summary["next_experiments"]:
        rows.append(
            "<tr>"
            f"<td>{html.escape(item['priority'])}</td>"
            f"<td>{html.escape(item['name'])}</td>"
            f"<td>{html.escape(item['question'])}</td>"
            f"<td>{html.escape(item['design'])}</td>"
            f"<td>{html.escape(item['success_rule'])}</td>"
            "</tr>"
        )
    return (
        "<article class='card wide'><h2>Next Airtight Experiments</h2>"
        "<div class='table-wrap'><table><thead><tr><th>Priority</th><th>Name</th><th>Question</th><th>Design</th><th>Success rule</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div></article>"
    )


def render_contrasts(summary: dict[str, Any]) -> str:
    labels = {
        "noise_own_d1_minus_d0": "Own-view noise: Own D1 - Own D0",
        "noise_cross_d1_minus_d0": "Cross-view noise: Cross D1 - Cross D0",
        "noise_both_d1_minus_d0": "Both-view noise: Both D1 - Both D0",
        "noise_cotrained_lowdose_d1_minus_d0": "Low-dose co-trained noise: C Own D1 .05 - C Own D0 .05",
        "cross_d1_minus_own_d1": "Target view: Cross D1 - Own D1",
        "both_d1_minus_own_d1": "Extra cap terms: Both D1 - Own D1",
        "cotrained_own_minus_frozen_own": "Moving target: Co-trained Own - Frozen Own",
        "ema_own_minus_frozen_own": "Moving target: EMA Own - Frozen Own",
    }
    rows = []
    for key, data in summary["aggregate"]["contrasts"].items():
        cls = "pos" if finite(data.get("mean_delta")) and float(data["mean_delta"]) >= 0 else "neg"
        rows.append(
            "<tr>"
            f"<td>{html.escape(labels.get(key, key))}</td>"
            f"<td class='num {cls}'>{html.escape(pp(data.get('mean_delta')))}</td>"
            f"<td class='num'>{html.escape(str(data.get('nonnegative_seeds', 0)))}/{html.escape(str(data.get('n', 0)))}</td>"
            "</tr>"
        )
    return (
        "<article class='card'><h2>Matched Contrasts</h2>"
        "<div class='table-wrap small'><table><thead><tr><th>Contrast</th><th>Mean delta</th><th>Non-negative seeds</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div></article>"
    )


def render_method_table(summary: dict[str, Any]) -> str:
    methods = summary["aggregate"]["methods"]
    rows = []
    for method in METHOD_ORDER:
        key = method["key"]
        data = methods[key]
        paired = summary["aggregate"]["paired_vs_jepa"].get(key, {})
        delta = paired.get("mean_delta")
        pass_state = "baseline" if key == "jepa_l1_p1" else ("yes" if paired.get("can_pass_four_of_five") else "no")
        nonneg = "n/a" if key == "jepa_l1_p1" else f"{paired.get('nonnegative_seeds', 0)}/{paired.get('n', 0)}"
        rows.append(
            "<tr>"
            f"<td><code>{html.escape(key)}</code></td>"
            f"<td>{html.escape(method['label'])}</td>"
            f"<td>{html.escape(method['description'])}</td>"
            f"<td class='num'>{html.escape(str(data['n']))}</td>"
            f"<td class='num'>{html.escape(pct(data.get('mean_exact_match')))}</td>"
            f"<td class='num'>{html.escape(pp(data.get('std_exact_match')))}</td>"
            f"<td class='num'>{html.escape(pp(delta))}</td>"
            f"<td class='num'>{html.escape(nonneg)}</td>"
            f"<td>{html.escape(pass_state)}</td>"
            "</tr>"
        )
    return (
        "<article class='card wide'><h2>Method Summary</h2>"
        "<div class='table-wrap'><table><thead><tr><th>Key</th><th>Method</th><th>Meaning</th><th>N</th><th>Mean exact</th><th>Std</th><th>Delta vs JEPA</th><th>Non-neg seeds</th><th>Can pass 4/5</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div></article>"
    )


def render_run_table(summary: dict[str, Any]) -> str:
    rows = []
    for row in summary["records"]:
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(row['seed']))}</td>"
            f"<td><code>{html.escape(row['method'])}</code></td>"
            f"<td class='num'>{html.escape(pct(row.get('eval_exact_match')))}</td>"
            f"<td class='num'>{html.escape(pct(row.get('eval_contains_target')))}</td>"
            f"<td class='num'>{html.escape(str(row.get('eval_matches', 0)))}/{html.escape(str(row.get('eval_count', 0)))}</td>"
            f"<td class='num'>{html.escape(num(row.get('train_loss')))}</td>"
            f"<td class='num'>{html.escape(num(row.get('train_runtime')))} s</td>"
            f"<td>{html.escape(row.get('train_status', 'missing'))}/{html.escape(row.get('eval_status', 'missing'))}</td>"
            "</tr>"
        )
    return (
        "<article class='card wide'><h2>Run Status</h2>"
        "<div class='table-wrap'><table><thead><tr><th>Seed</th><th>Method</th><th>Exact</th><th>Contains target</th><th>Matches</th><th>Train loss</th><th>Runtime</th><th>Status</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div></article>"
    )


def render_plots(plots: dict[str, str]) -> str:
    names = {
        "method_means": "Mean Exact Match",
        "delta_vs_jepa": "Matched Deltas Versus JEPA",
        "clean_noisy": "Clean Versus Noisy Controls",
        "cotrained_dose_response": "Co-trained Cap Dose Response",
        "seed_heatmap": "Seed Heatmap",
        "training_loss": "Training Loss",
    }
    cards = []
    for key, path in plots.items():
        cards.append(
            f"<article class='figure'><div class='figure-head'><h3>{html.escape(names.get(key, key))}</h3><a href='../{html.escape(path)}'>Open PNG</a></div>"
            f"<a href='../{html.escape(path)}'><img src='../{html.escape(path)}' alt='{html.escape(names.get(key, key))}'></a></article>"
        )
    return "<section class='fig-grid'>" + "".join(cards) + "</section>"


def build_html(summary: dict[str, Any], plots: dict[str, str]) -> str:
    data = json.dumps(summary, indent=2, ensure_ascii=True)
    title = "Sphere Cap Ablation on Original LLM-JEPA"
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>
    :root {{
      --bg: #f6f7f2;
      --panel: #ffffff;
      --soft: #fbfcf8;
      --ink: #1e2723;
      --muted: #65716c;
      --line: #dfe5dc;
      --green: #1f7a63;
      --red: #b94e4e;
      --shadow: 0 8px 24px rgba(31, 39, 35, 0.08);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; background: var(--bg); color: var(--ink); }}
    header {{ border-bottom: 1px solid var(--line); background: rgba(246, 247, 242, 0.96); }}
    header, main {{ max-width: 1480px; margin: 0 auto; padding: 18px; }}
    h1 {{ margin: 0 0 8px; font-size: 25px; line-height: 1.2; }}
    h2 {{ margin: 0 0 12px; font-size: 18px; }}
    h3 {{ margin: 0; font-size: 15px; }}
    p, li {{ color: var(--muted); line-height: 1.45; }}
    .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 12px; margin-bottom: 14px; }}
    .stat, .card, .figure {{ background: var(--panel); border: 1px solid var(--line); border-radius: 8px; box-shadow: var(--shadow); padding: 14px; }}
    .stat {{ background: var(--soft); min-height: 104px; }}
    .label {{ color: var(--muted); font-size: 12px; text-transform: uppercase; }}
    .value {{ margin-top: 7px; font-size: 26px; font-weight: 780; line-height: 1.05; }}
    .note {{ color: var(--muted); font-size: 12px; margin-top: 8px; }}
    .layout {{ display: grid; grid-template-columns: 1fr; gap: 14px; }}
    .fig-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(410px, 1fr)); gap: 12px; }}
    .figure-head {{ display: flex; align-items: center; justify-content: space-between; gap: 12px; margin-bottom: 10px; }}
    img {{ width: 100%; display: block; border-radius: 6px; }}
    a {{ color: #246a8f; text-decoration: none; font-weight: 650; }}
    .wide {{ grid-column: 1 / -1; }}
    .table-wrap {{ overflow: auto; border: 1px solid var(--line); border-radius: 8px; background: var(--panel); }}
    table {{ width: 100%; min-width: 860px; border-collapse: collapse; font-size: 13px; }}
    .small table {{ min-width: 520px; }}
    th, td {{ padding: 8px 9px; border-bottom: 1px solid var(--line); text-align: left; vertical-align: top; }}
    th {{ background: #eef3ec; color: #405047; font-size: 11px; text-transform: uppercase; }}
    .num {{ text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }}
    code {{ font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace; }}
    .pos {{ color: var(--green); font-weight: 730; }}
    .neg {{ color: var(--red); font-weight: 730; }}
    pre {{ margin: 0; white-space: pre-wrap; word-break: break-word; font-size: 12px; line-height: 1.45; }}
  </style>
</head>
<body>
  <header>
    <h1>{html.escape(title)}</h1>
    <p>Matched-seed cap-anchor ablations layered on top of the reproduced Llama-3.2-1B original LLM-JEPA protocol. The primary metric is strict exact-match regex generation; geometry plots are diagnostic and do not replace downstream exact match.</p>
  </header>
  <main>
    <section class="stats">{render_cards(summary)}</section>
    {render_plots(plots)}
    <section class="layout">
      {render_research_questions(summary)}
      {render_interpretation(summary)}
      {render_contrasts(summary)}
      {render_method_table(summary)}
      {render_next_experiments(summary)}
      {render_run_table(summary)}
      <article class="card wide"><h2>Embedded Summary JSON</h2><pre>{html.escape(data)}</pre></article>
    </section>
  </main>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a multi-seed spherical cap ablation report.")
    parser.add_argument("--baseline-root", type=Path, default=Path("results/replicate_llm_jepa_synth"))
    parser.add_argument("--sphere-root", type=Path, default=Path("results/replicate_llm_jepa_sphere_ablation"))
    parser.add_argument("--report-dir", type=Path, default=Path("reports/gpu_2026-05-10"))
    parser.add_argument("--train-file", type=Path, default=Path("datasets/synth_train.jsonl"))
    parser.add_argument("--test-file", type=Path, default=Path("datasets/synth_test.jsonl"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[82, 23, 37, 84, 4])
    parser.add_argument("--stem", default="replicate_llm_jepa_sphere_ablation")
    parser.add_argument("--base-model", default="meta-llama/Llama-3.2-1B-Instruct")
    args = parser.parse_args()

    setup_theme()
    summary = build_summary(args)
    plots = make_plots(summary, args.report_dir, args.stem)
    summary["plots"] = plots

    json_path = args.report_dir / "json" / f"{args.stem}_summary.json"
    html_path = args.report_dir / "html" / f"{args.stem}.html"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n")
    html_path.write_text(build_html(summary, plots))
    print(f"wrote {json_path}")
    print(f"wrote {html_path}")


if __name__ == "__main__":
    main()
