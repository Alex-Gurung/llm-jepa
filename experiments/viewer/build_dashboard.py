from __future__ import annotations

import argparse
import html
import json
import math
import textwrap
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


EXPERIMENT_META = {
    "exp0_full": {
        "title": "Exp0 Toy Geometry",
        "short_title": "E0 Toy",
        "group": "Toy",
        "summary": "Synthetic paired-view run with continuous, RFF, and frozen-teacher sample-specific anchors.",
        "takeaways": [
            "Co-trained self-anchor collapses hard: low RankMe and cosine near 1.",
            "Noisy continuous and noisy frozen-teacher anchors improve over their matched clean-anchor controls, which is the cleanest evidence that noise can help.",
            "The class-anchor control uses labels. It can look good globally while missing sample-level spread, so within-class RankMe matters.",
        ],
    },
    "exp1_cifar_gpu": {
        "title": "Exp1 CIFAR-10 Bridge",
        "short_title": "E1 CIFAR",
        "group": "Vision bridge",
        "summary": "GPU sanity run on a 10k/2k CIFAR-10 subset with a frozen autoencoder teacher anchor.",
        "takeaways": [
            "The noisy frozen anchor improves spread over the clean frozen anchor, but linear-probe accuracy drops in this pass.",
            "The class-anchor control uses labels, so its high probe accuracy is a supervised control rather than evidence for the anchor mechanism.",
            "Treat this as a bridge/sanity run, not the main downstream result.",
        ],
    },
    "exp2_pythia160m_synth_input_white_anchor_white": {
        "title": "Exp2 Frozen LLM: Source + Anchor Whitening",
        "short_title": "E2 White+White",
        "group": "Frozen LLM",
        "summary": "The useful Pythia-160M diagnostic: source states and anchors are both whitened.",
        "takeaways": [
            "Noisy frozen own-anchor beats clean frozen own-anchor on RankMe, mean cosine, and predicted-code retrieval.",
            "Noisy frozen own-anchor is far stronger than co-trained self-anchor, so external anchoring matters.",
            "InfoNCE and VICReg still dominate direct embedding retrieval, so retrieval-space choice matters.",
        ],
    },
    "exp2_pythia160m_synth_sphere": {
        "title": "Exp2 Frozen LLM: Spherical Anchors",
        "short_title": "E2 Sphere",
        "group": "Frozen LLM",
        "summary": "Pythia-160M synth projection-head run using spherified anchors on raw source states.",
        "takeaways": [
            "Raw source hidden states remain highly anisotropic, so anchor spherification alone is not enough.",
            "This run keeps the full ablation set, including cross-view cap reconstruction.",
            "Use it mainly to compare own-view, cross-view, InfoNCE, VICReg, and SIGReg geometry controls.",
        ],
    },
    "exp2_pythia160m_synth_raw": {
        "title": "Exp2 Frozen LLM: Raw Anchors",
        "short_title": "E2 Raw",
        "group": "Frozen LLM",
        "summary": "D_own anchor preprocessing ablation with raw Pythia hidden states.",
        "takeaways": [
            "Raw anchors expose the source anisotropy problem directly.",
            "This is a useful negative control, not the strongest preprocessing choice.",
        ],
    },
    "exp2_pythia160m_synth_norm": {
        "title": "Exp2 Frozen LLM: Norm Anchors",
        "short_title": "E2 Norm",
        "group": "Frozen LLM",
        "summary": "D_own anchor preprocessing ablation with normalized Pythia hidden states.",
        "takeaways": [
            "Norm-only preprocessing changes scale without fixing most anisotropy.",
            "Compare it against raw, sphere, and white before drawing mechanism conclusions.",
        ],
    },
    "exp2_pythia160m_synth_white": {
        "title": "Exp2 Frozen LLM: White Anchors",
        "short_title": "E2 White",
        "group": "Frozen LLM",
        "summary": "D_own anchor preprocessing ablation with whitened anchors and raw source states.",
        "takeaways": [
            "Whitened anchors are much more isotropic than raw anchors.",
            "Source-side anisotropy still limits the projection-head run unless inputs are whitened too.",
        ],
    },
    "exp3_full_synth": {
        "title": "Exp3 Full LLM-JEPA: SmolLM2 Synth",
        "short_title": "E3 Synth",
        "group": "Full LLM fine-tune",
        "summary": "Full 8k-example SmolLM2-135M synthetic NL-to-regex fine-tune sweep.",
        "takeaways": [
            "The downstream metric is strict exact-match string equality on synth_test.jsonl.",
            "Noisy frozen anchor versus clean frozen anchor isolates the effect of sphere noise.",
            "Training loss and exact match can move in different directions, so both are shown.",
        ],
    },
    "exp3_llama1b_synth": {
        "title": "Exp3 Full LLM-JEPA: Llama 1B Synth",
        "short_title": "E3 Llama",
        "group": "Full LLM fine-tune",
        "summary": "Full 8k-example Llama-3.2-1B synthetic NL-to-regex fine-tune sweep.",
        "takeaways": [
            "This is the larger-model pass for the same full fine-tuning ablations as SmolLM2.",
            "Plain fine-tune, co-trained self-anchor, clean frozen anchor, noisy frozen anchor, momentum self-anchor, cross-view anchor, and combined anchors test the matched controls from the handoff.",
            "Read noisy frozen minus clean frozen as the sphere-noise contribution, and noisy frozen minus self-anchor as the external-anchor contribution.",
        ],
    },
}


RUN_ORDER = [
    "exp0_full",
    "exp1_cifar_gpu",
    "exp2_pythia160m_synth_input_white_anchor_white",
    "exp2_pythia160m_synth_sphere",
    "exp2_pythia160m_synth_raw",
    "exp2_pythia160m_synth_norm",
    "exp2_pythia160m_synth_white",
    "exp3_full_synth",
    "exp3_llama1b_synth",
]


VARIANT_GLOSSARY = [
    ("A", "MSE JEPA baseline", "Predict paired-view embedding with MSE. Mostly a baseline, not the sphere mechanism."),
    ("B", "Cosine JEPA baseline", "Predict paired-view embedding with cosine loss, matching the LLM-JEPA default metric."),
    ("C", "Co-trained self-anchor", "The cap reconstructs a target produced by the same moving encoder. This is collapse-prone and is the anchor-failure control."),
    ("C_detach", "Detached self-anchor", "Like the co-trained self-anchor, but the current target is detached from gradient flow."),
    ("C_ema", "Momentum self-anchor", "The target comes from an exponential moving-average copy of the encoder."),
    ("D0 / D_own_0", "Clean frozen own-anchor", "Same external anchor as the noisy frozen anchor, but sigma=0. This isolates the value of anchoring alone."),
    ("D1 / D_own_1", "Noisy frozen own-anchor", "The main sphere-encoder-style test: noisy spherical cap must recover a sample-specific external anchor."),
    ("D0a/D1a", "Clean/noisy continuous anchor", "Toy run only: target is the original continuous 2-D sample."),
    ("D0b/D1b", "Clean/noisy RFF anchor", "Toy run only: target is a random Fourier feature anchor."),
    ("D0c/D1c", "Clean/noisy frozen-teacher anchor", "Toy run only: target is a frozen teacher/autoencoder feature."),
    ("D2 / D3", "Class-level anchor", "Supervised class-label target. Useful as a cardinality warning: class spread is not sample spread."),
    ("D_cross_*", "Frozen cross-view anchor", "Noisy cap from one view reconstructs the paired view's anchor."),
    ("D_cross_sym_*", "Two-way frozen cross-view anchor", "Cross-view cap reconstruction in both text-to-code and code-to-text directions."),
    ("F", "Regularizer baseline", "VICReg in Exp0/Exp1; InfoNCE in Exp2. Read in the context of each run."),
    ("G", "Regularizer baseline", "SIGReg in Exp0/Exp1; VICReg in Exp2."),
    ("H", "SIGReg baseline", "Exp2 only: cosine alignment plus SIGReg-style isotropic regularization."),
    ("Regular", "Plain fine-tune", "Exp3 only: supervised LLM fine-tuning without the cap-anchor objective."),
    ("Cross / Both", "Cross-view / combined frozen anchors", "Exp3 only: cross-view cap, or own-view plus cross-view cap."),
]


METHOD_GUIDE = [
    (
        "View",
        "One representation of the same example. In toy runs the views are two synthetic augmentations; in CIFAR they are two image augmentations; in frozen LLM runs they are the natural-language side and regex/code side.",
    ),
    (
        "JEPA alignment",
        "A predictor takes one view's embedding and tries to match the paired view's embedding. It learns representations without directly generating the original input.",
    ),
    (
        "Co-trained self-anchor",
        "The target is produced by a network being trained at the same time as the predictor. Because the target can move, this can make the loss easy by letting both sides collapse together.",
    ),
    (
        "Frozen anchor",
        "The target is fixed before training. The model cannot move the target to make the loss easier, so this is the control that tests whether a stable external target prevents collapse.",
    ),
    (
        "Sphere noise",
        "The learned embedding is normalized onto a sphere, perturbed, then asked to recover the same sample's anchor. The intended effect is to make the local neighborhood around each sample meaningful instead of brittle.",
    ),
    (
        "InfoNCE",
        "A contrastive baseline. It pulls true pairs together and pushes all other batch items apart. It often wins direct retrieval because direct retrieval is exactly what it trains for.",
    ),
    (
        "VICReg",
        "A non-contrastive baseline with three terms: match paired embeddings, keep each dimension from collapsing, and reduce correlation between dimensions.",
    ),
    (
        "SIGReg",
        "An isotropy regularizer. In Exp2 it is added to JEPA alignment and pushes embeddings toward a centered, roughly identity-covariance distribution; it does not use frozen anchors or sphere-noise recovery.",
    ),
]


WHITENING_GUIDE = [
    (
        "What whitening does",
        "It subtracts the training-set mean, rotates into principal-component coordinates, and rescales directions so the covariance is close to identity.",
    ),
    (
        "Why it matters here",
        "Frozen LLM hidden states often have a few huge common directions. Without whitening, many examples look artificially similar, which can create high cosine, hubness, and misleading retrieval.",
    ),
    (
        "Source whitening",
        "Whitening the input vectors fed into the trainable projection heads. This fixes the geometry the small heads start from.",
    ),
    (
        "Anchor whitening",
        "Whitening the fixed target vectors that cap heads reconstruct. This makes the target space itself less dominated by a few directions.",
    ),
    (
        "What it is not",
        "It is not a supervised label and not a task answer. It is preprocessing that makes the frozen representation space numerically better behaved.",
    ),
]


EXPECTED_PATTERN_GUIDE = [
    ("Noise contribution", "Noisy frozen anchor should beat clean frozen anchor in the same run. This isolates sphere noise because the anchor and model setup are otherwise matched."),
    ("External-anchor contribution", "Noisy frozen anchor should beat co-trained or momentum self-anchor. This asks whether a fixed target is better than a target that moves with training."),
    ("Geometry health", "Good geometry usually means high RankMe and low mean cosine together. High RankMe alone is not enough if task metrics get worse."),
    ("Downstream usefulness", "For full fine-tuning, exact-match generation must improve over plain fine-tune. Geometry-only wins do not automatically prove a better generator."),
    ("Baseline sanity", "InfoNCE/VICReg/SIGReg should prevent trivial collapse. If they win direct retrieval, that means retrieval favors their objective; it does not automatically disprove the anchor mechanism."),
]


TASK_GLOSSARY = [
    ("Toy geometry", "Synthetic paired views", "Train small networks on controlled synthetic clusters. This is the fastest way to see collapse, spread, and the noise-vs-anchor controls."),
    ("CIFAR bridge", "Image augmentation representation learning", "Train image encoders on CIFAR-10 augmentations, then evaluate both geometry and linear-probe accuracy."),
    ("Frozen LLM heads", "Natural-language/regex pairs with a frozen Pythia backbone", "Cache frozen LLM hidden states, train only small projection/cap heads, and evaluate text-code geometry/retrieval without changing the LLM."),
    ("Full LLM fine-tune", "Generate a regex from a natural-language prompt", "Fine-tune the language model itself and score strict exact-match regex generation on held-out synthetic prompts."),
]


RUN_TASK_NOTES = {
    "exp0_full": "Synthetic paired-view geometry test. This is about collapse, spread, and whether sphere noise helps when the target anchor is controlled.",
    "exp1_cifar_gpu": "CIFAR-10 bridge task. This checks whether geometry improvements survive contact with a simple image representation benchmark.",
    "exp2_pythia160m_synth_input_white_anchor_white": "Frozen Pythia text/code head task. This is the cleanest geometry diagnostic for NL-to-regex pairs because both source and anchor states are whitened.",
    "exp2_pythia160m_synth_sphere": "Frozen Pythia preprocessing control with spherical anchors on raw source states.",
    "exp2_pythia160m_synth_raw": "Frozen Pythia preprocessing control with raw anchors and raw source states.",
    "exp2_pythia160m_synth_norm": "Frozen Pythia preprocessing control with normalized anchors and raw source states.",
    "exp2_pythia160m_synth_white": "Frozen Pythia preprocessing control with whitened anchors and raw source states.",
    "exp3_full_synth": "Full SmolLM2 fine-tune. This is the actual regex-generation task, scored by strict exact-match on held-out synthetic prompts.",
    "exp3_llama1b_synth": "Full Llama 1B fine-tune. This repeats the actual regex-generation task with a stronger base model.",
}


BASE_VARIANT_NAMES = {
    "A": "MSE JEPA baseline",
    "B": "Cosine JEPA baseline",
    "C": "Co-trained self-anchor",
    "C_detach": "Detached self-anchor",
    "C_ema": "Momentum self-anchor",
    "D0": "Clean frozen anchor",
    "D1": "Noisy frozen anchor",
    "D0a": "Clean continuous anchor",
    "D1a": "Noisy continuous anchor",
    "D0b": "Clean RFF anchor",
    "D1b": "Noisy RFF anchor",
    "D0c": "Clean frozen-teacher anchor",
    "D1c": "Noisy frozen-teacher anchor",
    "D2": "Class anchor",
    "D3": "Class anchor",
    "D_own_0": "Clean frozen own-anchor",
    "D_own_1": "Noisy frozen own-anchor",
    "D_cross_0": "Clean frozen cross-anchor",
    "D_cross_1": "Noisy frozen cross-anchor",
    "D_cross_sym_0": "Clean two-way cross-anchor",
    "D_cross_sym_1": "Noisy two-way cross-anchor",
    "H": "SIGReg baseline",
    "Regular": "Plain fine-tune",
    "Cross": "Frozen cross-view anchor",
    "Both": "Own + cross frozen anchors",
}


BASE_VARIANT_SHORT_NAMES = {
    "A": "MSE JEPA",
    "B": "Cosine JEPA",
    "C": "Co-trained self",
    "C_detach": "Detached self",
    "C_ema": "Momentum self",
    "D0": "Clean frozen",
    "D1": "Noisy frozen",
    "D0a": "Clean continuous",
    "D1a": "Noisy continuous",
    "D0b": "Clean RFF",
    "D1b": "Noisy RFF",
    "D0c": "Clean teacher",
    "D1c": "Noisy teacher",
    "D2": "Class anchor",
    "D3": "Class anchor",
    "D_own_0": "Clean own-anchor",
    "D_own_1": "Noisy own-anchor",
    "D_cross_0": "Clean cross-anchor",
    "D_cross_1": "Noisy cross-anchor",
    "D_cross_sym_0": "Clean two-way cross",
    "D_cross_sym_1": "Noisy two-way cross",
    "H": "SIGReg",
    "Regular": "Plain fine-tune",
    "Cross": "Cross-view anchor",
    "Both": "Own + cross anchors",
}


METRIC_GLOSSARY = [
    ("RankMe", "Effective dimensionality. If all samples collapse to one direction, RankMe is near 1. Higher means information is spread across more independent directions, but it is only a geometry metric."),
    ("Mean off-diagonal cosine", "Average similarity between different samples. Lower is better. Near 1 means most samples point in almost the same direction."),
    ("Uniformity", "How evenly normalized points cover the sphere. More negative is usually better; use it with RankMe and cosine, not alone."),
    ("Top eig mass", "Share of variance in the largest principal direction. Lower is better. High values mean one direction dominates the embedding space."),
    ("PCA maps", "A 2-D projection of high-dimensional embeddings. Use it to spot obvious collapse, thin lines, or separated clusters; use metrics for claims."),
    ("Hypersphere maps", "A normalized PCA disk. It is a visual proxy for angular spread: many directions is healthier than one clump at a pole."),
    ("Retrieval heatmaps", "Rows are queries and columns are targets. A bright diagonal is good; bright columns or rows away from the diagonal mean hubness."),
    ("Exact match", "Full fine-tune task metric. The generated regex string must exactly equal the held-out target string."),
]


FIGURE_NOTES = {
    "overview": "Start here. The upper-left area is best for geometry: high RankMe and low mean cosine. The key comparisons are noisy frozen anchor vs clean frozen anchor, and noisy frozen anchor vs self-anchor.",
    "preprocessing": "This explains why whitening matters. Source+anchor whitening is the cleanest Exp2 comparison because both the inputs and fixed targets have the worst common-direction bias removed.",
    "metrics": "Read RankMe and cosine together. A good geometry result has higher RankMe and lower cosine than its matched control; direct retrieval can favor InfoNCE because InfoNCE trains for direct retrieval.",
    "pca": "Each panel is a 2-D projection of a high-dimensional embedding. A tiny dot, line, or stacked classes suggest collapse; a broad cloud suggests spread. PCA is visual evidence, not the final metric.",
    "sphere": "This reprojects points into a unit disk to show angular spread. Healthy embeddings occupy many directions; collapsed embeddings clump in one region.",
    "hist": "These histograms show all pairwise cosines, not just the mean. A spike near 1 means many examples are almost identical; a distribution shifted left is healthier.",
    "spectrum": "This shows how variance is distributed across principal components. A huge first component means anisotropy; a flatter spectrum means more directions are being used.",
    "retrieval": "Recall@1 asks whether each query retrieves its paired target. Direct embedding retrieval and cap-to-anchor retrieval are different spaces, so compare the matching retrieval metric for the method.",
    "heatmap": "Rows are queries and columns are targets. A bright diagonal is good. Bright vertical bands mean some targets are hubs that many queries retrieve.",
    "anchors": "Anchor diagnostics tell whether the fixed target space is usable before training. If anchors are anisotropic, the model can inherit that problem.",
    "history": "Training loss is an optimization trace, not the scientific result. Variants can have different loss terms, so compare final metrics rather than raw loss scale.",
}


def variant_name(code: Any, run: dict[str, Any] | None = None) -> str:
    value = str(code)
    slug = str((run or {}).get("slug", ""))
    if slug.startswith("exp2_pythia160m"):
        run_specific = {"F": "InfoNCE baseline", "G": "VICReg baseline"}
    elif slug in {"exp0_full", "exp1_cifar_gpu"}:
        run_specific = {"F": "VICReg baseline", "G": "SIGReg baseline"}
    else:
        run_specific = {}
    return run_specific.get(value, BASE_VARIANT_NAMES.get(value, value))


def variant_label(code: Any, run: dict[str, Any] | None = None, *, multiline: bool = False) -> str:
    value = str(code)
    name = variant_name(value, run)
    label = value if name == value else f"{name} ({value})"
    if multiline:
        label = label.replace(" + ", " +\n").replace(", ", ",\n")
    return label


def variant_plot_label(code: Any, run: dict[str, Any] | None = None, *, multiline: bool = False) -> str:
    value = str(code)
    slug = str((run or {}).get("slug", ""))
    if slug.startswith("exp2_pythia160m"):
        run_specific = {"F": "InfoNCE", "G": "VICReg"}
    elif slug in {"exp0_full", "exp1_cifar_gpu"}:
        run_specific = {"F": "VICReg", "G": "SIGReg"}
    else:
        run_specific = {}
    label = run_specific.get(value, BASE_VARIANT_SHORT_NAMES.get(value, value))
    if multiline:
        return "\n".join(textwrap.wrap(label, width=16, break_long_words=False))
    return label


def setup_theme() -> None:
    sns.set_theme(
        context="talk",
        style="whitegrid",
        palette=["#276a8c", "#c1564f", "#287c62", "#c28a32", "#6957a8", "#1b8594", "#8c5d2f"],
    )
    plt.rcParams.update(
        {
            "figure.facecolor": "#f8faf6",
            "axes.facecolor": "#ffffff",
            "savefig.facecolor": "#f8faf6",
            "axes.edgecolor": "#cbd6ca",
            "grid.color": "#e6ece3",
            "axes.titleweight": "bold",
            "axes.labelcolor": "#27332e",
            "xtick.color": "#4b5952",
            "ytick.color": "#4b5952",
            "font.family": "DejaVu Sans",
        }
    )


def get_path(row: dict[str, Any], path: str) -> float:
    cur: Any = row
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return math.nan
        cur = cur[part]
    try:
        value = float(cur)
    except (TypeError, ValueError):
        return math.nan
    return value if math.isfinite(value) else math.nan


def finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def fmt(value: Any, kind: str = "number") -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not math.isfinite(number):
        return "n/a"
    if kind == "percent":
        return f"{number * 100:.1f}%"
    if kind == "pp":
        return f"{number * 100:.1f} pp"
    if abs(number) >= 1000:
        return f"{number:.0f}"
    if abs(number) >= 100:
        return f"{number:.1f}"
    if abs(number) >= 10:
        return f"{number:.2f}"
    if abs(number) >= 1:
        return f"{number:.3f}"
    if abs(number) >= 0.01:
        return f"{number:.4f}"
    return f"{number:.2e}"


def slug_from_summary(path: Path) -> str:
    return path.stem.removesuffix("_summary")


def default_meta(slug: str) -> dict[str, Any]:
    return {
        "title": slug.replace("_", " ").title(),
        "short_title": slug.replace("_", " ").title(),
        "group": "Report",
        "summary": "Generated summary report.",
        "takeaways": [],
    }


def relative_to(path: Path, base: Path) -> str:
    try:
        return path.relative_to(base).as_posix()
    except ValueError:
        return path.as_posix()


def load_run(path: Path, report_dir: Path) -> dict[str, Any]:
    slug = slug_from_summary(path)
    meta = {**default_meta(slug), **EXPERIMENT_META.get(slug, {})}
    payload = json.loads(path.read_text())
    html_path = report_dir / "html" / f"{slug}.html"
    return {
        "slug": slug,
        "title": meta["title"],
        "short_title": meta["short_title"],
        "group": meta["group"],
        "summary": meta["summary"],
        "takeaways": meta["takeaways"],
        "json_path": relative_to(path, report_dir),
        "html_path": relative_to(html_path, report_dir) if html_path.exists() else "",
        "config": payload.get("config", {}),
        "cache_config": payload.get("cache_config", {}),
        "headline_contrasts": payload.get("headline_contrasts", {}),
        "anchor_geometry": payload.get("anchor_geometry", {}),
        "records": payload.get("records", []),
    }


def ordered(paths: list[Path]) -> list[Path]:
    order = {slug: idx for idx, slug in enumerate(RUN_ORDER)}
    return sorted(paths, key=lambda path: (order.get(slug_from_summary(path), 10_000), slug_from_summary(path)))


def by_variant(run: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(record.get("variant")): record for record in run.get("records", [])}


def rank_value(record: dict[str, Any], space: str = "auto") -> float:
    if space == "text":
        return get_path(record, "text_rankme")
    if space == "code":
        return get_path(record, "code_rankme")
    value = get_path(record, "rankme")
    return value if finite(value) else get_path(record, "text_rankme")


def mean_cosine(record: dict[str, Any], space: str = "auto") -> float:
    if space == "text":
        return get_path(record, "text_cosine.mean")
    if space == "code":
        return get_path(record, "code_cosine.mean")
    value = get_path(record, "cosine.mean")
    return value if finite(value) else get_path(record, "text_cosine.mean")


def metric_rows(run: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for record in run.get("records", []):
        variant = str(record.get("variant"))
        if finite(get_path(record, "text_rankme")):
            rows.extend(
                [
                    {"variant": variant, "metric": "Text RankMe", "value": get_path(record, "text_rankme")},
                    {"variant": variant, "metric": "Code RankMe", "value": get_path(record, "code_rankme")},
                    {"variant": variant, "metric": "Text mean cosine", "value": get_path(record, "text_cosine.mean")},
                    {"variant": variant, "metric": "Code mean cosine", "value": get_path(record, "code_cosine.mean")},
                ]
            )
            for label, path in [
                ("Direct R@1", "retrieval_embedding.recall@1"),
                ("Pred-code R@1", "retrieval_predicted_code_embedding.recall@1"),
                ("Text cap R@1", "retrieval_text_cap_to_text_anchor.recall@1"),
                ("Code cap R@1", "retrieval_code_cap_to_code_anchor.recall@1"),
            ]:
                value = get_path(record, path)
                if finite(value):
                    rows.append({"variant": variant, "metric": label, "value": value})
        elif finite(get_path(record, "eval_exact_match")):
            for label, path in [
                ("Exact match", "eval_exact_match"),
                ("Train loss", "train_loss"),
                ("Final logged loss", "final_log_loss"),
                ("Samples/s", "train_samples_per_second"),
            ]:
                value = get_path(record, path)
                if finite(value):
                    rows.append({"variant": variant, "metric": label, "value": value})
        else:
            rows.extend(
                [
                    {"variant": variant, "metric": "RankMe", "value": get_path(record, "rankme")},
                    {"variant": variant, "metric": "Within-class RankMe", "value": get_path(record, "within_class_rankme._mean_within_class")},
                    {"variant": variant, "metric": "Mean cosine", "value": get_path(record, "cosine.mean")},
                    {"variant": variant, "metric": "Uniformity", "value": get_path(record, "uniformity")},
                ]
            )
            value = get_path(record, "linear_probe_top1")
            if finite(value):
                rows.append({"variant": variant, "metric": "Linear probe top-1", "value": value})
    return [row for row in rows if finite(row["value"])]


def selected_variants(run: dict[str, Any], limit: int = 10) -> list[dict[str, Any]]:
    preferred = [
        "C",
        "C_detach",
        "C_ema",
        "D0",
        "D1",
        "D0a",
        "D1a",
        "D0b",
        "D1b",
        "D0c",
        "D1c",
        "D3",
        "D_own_0",
        "D_own_1",
        "D_cross_0",
        "D_cross_1",
        "D_cross_sym_0",
        "D_cross_sym_1",
        "F",
        "G",
        "H",
        "Regular",
        "Cross",
        "Both",
    ]
    records = by_variant(run)
    picked = [records[name] for name in preferred if name in records]
    if len(picked) < min(limit, len(records)):
        for record in run.get("records", []):
            if record not in picked:
                picked.append(record)
            if len(picked) >= limit:
                break
    return picked[:limit]


def save_fig(fig: plt.Figure, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path.name


def annotate_wrapped(ax: plt.Axes, text: str, xy: tuple[float, float] = (0.02, 0.98)) -> None:
    ax.text(
        xy[0],
        xy[1],
        textwrap.fill(text, 52),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        color="#33413b",
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "#f6f8f2", "edgecolor": "#d8e0d4", "alpha": 0.94},
    )


def plot_overview(runs: list[dict[str, Any]], assets_dir: Path) -> str:
    rows = []
    for run in runs:
        for record in run.get("records", []):
            rank = rank_value(record)
            cosine = mean_cosine(record)
            if finite(rank) and finite(cosine):
                rows.append(
                    {
                        "run": run["short_title"],
                        "slug": run["slug"],
                        "group": run["group"],
                        "variant": str(record.get("variant")),
                        "variant_label": variant_plot_label(record.get("variant"), run),
                        "rank": rank,
                        "mean_cosine": cosine,
                    }
                )
    df = pd.DataFrame(rows)
    fig, axes = plt.subplots(2, 2, figsize=(17, 11))
    if not df.empty:
        sns.scatterplot(
            data=df,
            x="mean_cosine",
            y="rank",
            hue="group",
            style="run",
            s=110,
            edgecolor="#17211d",
            linewidth=0.45,
            ax=axes[0, 0],
        )
        axes[0, 0].set_title("Geometry Map: Prefer Upper Left, Avoid Right-Side Collapse", fontsize=17)
        axes[0, 0].set_xlabel("Mean off-diagonal cosine (lower is better)")
        axes[0, 0].set_ylabel("RankMe (higher is better)")
        axes[0, 0].legend(loc="best", fontsize=8)
        label_mask = (
            (df["slug"].eq("exp2_pythia160m_synth_input_white_anchor_white") & df["variant"].isin(["C", "D_own_1", "F", "G"]))
            | (df["slug"].eq("exp0_full") & df["variant"].isin(["C", "D1c"]))
            | (df["slug"].eq("exp1_cifar_gpu") & df["variant"].isin(["D0", "D1"]))
        )
        for _, row in df[label_mask].iterrows():
            axes[0, 0].annotate(row["variant_label"], (row["mean_cosine"], row["rank"]), fontsize=7, xytext=(4, 4), textcoords="offset points")
    else:
        axes[0, 0].axis("off")

    focus = next((run for run in runs if run["slug"] == "exp2_pythia160m_synth_input_white_anchor_white"), None)
    if focus:
        records = by_variant(focus)
        focus_rows = []
        for name in ["D_own_0", "D_own_1", "C", "C_ema", "F", "G", "H"]:
            record = records.get(name)
            if not record:
                continue
            focus_rows.append(
                {
                    "variant": name,
                    "variant_label": variant_plot_label(name, focus, multiline=True),
                    "code_rankme": get_path(record, "code_rankme"),
                    "pred_r1": get_path(record, "retrieval_predicted_code_embedding.recall@1"),
                }
            )
        if focus_rows:
            local = pd.DataFrame(focus_rows)
            sns.barplot(data=local, y="variant_label", x="code_rankme", ax=axes[0, 1], color="#276a8c")
            axes[0, 1].set_title("Best Exp2 Run: Code RankMe + Pred-Code R@1", fontsize=17)
            axes[0, 1].set_xlabel("Code RankMe")
            axes[0, 1].set_ylabel("")
            ax_r = axes[0, 1].twiny()
            valid = local.dropna(subset=["pred_r1"])
            ax_r.plot(valid["pred_r1"], valid["variant_label"], color="#b9524d", marker="o", linewidth=2.6)
            ax_r.set_xlim(0, max(1.0, float(valid["pred_r1"].max()) * 1.2 if len(valid) else 1.0))
            ax_r.set_xlabel("Pred-code R@1")
            for row in valid.itertuples(index=False):
                ax_r.annotate(fmt(row.pred_r1, "percent"), (row.pred_r1, row.variant_label), textcoords="offset points", xytext=(7, 0), va="center", fontsize=9, color="#7a332f")

        anchor_rows = []
        for name, diag in (focus.get("anchor_geometry") or {}).items():
            if name in {"raw_text", "raw_code", "source_white_text", "source_white_code", "white_text", "white_code"}:
                anchor_rows.append({"anchor": name, "metric": "Top eig mass", "value": get_path(diag, "top_eig_mass_ratio")})
                anchor_rows.append({"anchor": name, "metric": "Mean cosine", "value": get_path(diag, "cosine_stats.mean")})
        if anchor_rows:
            sns.barplot(data=pd.DataFrame(anchor_rows), x="anchor", y="value", hue="metric", ax=axes[1, 0])
            axes[1, 0].set_title("Whitening Makes the Anchor Usable", fontsize=17)
            axes[1, 0].tick_params(axis="x", rotation=45)
            axes[1, 0].set_xlabel("")
            axes[1, 0].set_ylabel("Lower is better")
            axes[1, 0].legend(fontsize=9)
    else:
        axes[0, 1].axis("off")
        axes[1, 0].axis("off")

    exp1 = next((run for run in runs if run["slug"] == "exp1_cifar_gpu"), None)
    if exp1:
        rows = []
        for record in exp1.get("records", []):
            probe = get_path(record, "linear_probe_top1")
            rank = get_path(record, "rankme")
            if finite(probe) and finite(rank):
                rows.append({"variant": variant_plot_label(record.get("variant"), exp1), "probe": probe, "rank": rank})
        if rows:
            local = pd.DataFrame(rows)
            sns.scatterplot(data=local, x="rank", y="probe", hue="variant", s=130, ax=axes[1, 1])
            axes[1, 1].set_title("CIFAR: Spread Is Not the Same as Probe Accuracy", fontsize=17)
            axes[1, 1].set_xlabel("RankMe")
            axes[1, 1].set_ylabel("Linear probe top-1")
            axes[1, 1].legend(ncol=2, fontsize=8)
    else:
        axes[1, 1].axis("off")

    fig.suptitle("Sphere-JEPA Results: What We Are Trying to See", fontsize=22, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94), h_pad=2.4, w_pad=3.0)
    return save_fig(fig, assets_dir / "overview_geometry.png")


def plot_preprocessing(runs: list[dict[str, Any]], assets_dir: Path) -> str | None:
    rows = []
    for run in runs:
        if not run["slug"].startswith("exp2_pythia160m"):
            continue
        records = by_variant(run)
        d1 = records.get("D_own_1")
        d0 = records.get("D_own_0")
        if not d1:
            continue
        label = f"{run.get('config', {}).get('input_preprocess', 'raw')} / {run.get('config', {}).get('anchor_preprocess', 'raw')}"
        rows.extend(
            [
                {"mode": label, "metric": "Noisy anchor text RankMe", "value": get_path(d1, "text_rankme")},
                {"mode": label, "metric": "Noisy anchor code RankMe", "value": get_path(d1, "code_rankme")},
                {"mode": label, "metric": "Noisy anchor pred-code R@1", "value": get_path(d1, "retrieval_predicted_code_embedding.recall@1")},
                {"mode": label, "metric": "Noisy anchor mean text cosine", "value": get_path(d1, "text_cosine.mean")},
            ]
        )
        if d0:
            rows.extend(
                [
                    {"mode": label, "metric": "Noisy-clean code RankMe", "value": get_path(d1, "code_rankme") - get_path(d0, "code_rankme")},
                    {"mode": label, "metric": "Noisy-clean pred-code R@1", "value": get_path(d1, "retrieval_predicted_code_embedding.recall@1") - get_path(d0, "retrieval_predicted_code_embedding.recall@1")},
                ]
            )
    rows = [row for row in rows if finite(row["value"])]
    if not rows:
        return None
    df = pd.DataFrame(rows)
    metrics = list(dict.fromkeys(df["metric"]))
    fig, axes = plt.subplots(math.ceil(len(metrics) / 2), 2, figsize=(15, 3.6 * math.ceil(len(metrics) / 2)))
    axes = np.ravel(axes)
    for ax, metric in zip(axes, metrics):
        local = df[df["metric"] == metric]
        sns.barplot(data=local, x="mode", y="value", ax=ax, color="#276a8c")
        ax.set_title(metric)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=38)
        if "R@1" in metric:
            ax.set_ylabel("fraction")
        else:
            ax.set_ylabel("")
    for ax in axes[len(metrics):]:
        ax.axis("off")
    fig.suptitle("Exp2 Preprocessing Sweep: Why Source + Anchor Whitening Matters", fontsize=22, fontweight="bold")
    fig.tight_layout()
    return save_fig(fig, assets_dir / "exp2_preprocessing.png")


def plot_metrics(run: dict[str, Any], assets_dir: Path) -> str | None:
    rows = metric_rows(run)
    if not rows:
        return None
    df = pd.DataFrame(rows)
    metrics = list(dict.fromkeys(df["metric"]))
    max_panels = min(6, len(metrics))
    fig, axes = plt.subplots(math.ceil(max_panels / 2), 2, figsize=(15, 3.9 * math.ceil(max_panels / 2)))
    axes = np.ravel(axes)
    for ax, metric in zip(axes, metrics[:max_panels]):
        local = df[df["metric"] == metric].copy()
        order = [variant_plot_label(r.get("variant"), run) for r in run.get("records", [])]
        local["variant_label"] = local["variant"].map(lambda value: variant_plot_label(value, run))
        local["variant_label"] = pd.Categorical(local["variant_label"], categories=order, ordered=True)
        sns.barplot(data=local, y="variant_label", x="value", hue="variant_label", dodge=False, legend=False, ax=ax)
        ax.set_title(metric)
        ax.set_xlabel("")
        ax.set_ylabel("")
    for ax in axes[max_panels:]:
        ax.axis("off")
    fig.suptitle(f"{run['short_title']} Metrics", fontsize=22, fontweight="bold")
    fig.tight_layout()
    return save_fig(fig, assets_dir / f"{run['slug']}_metrics.png")


def scatter_points(record: dict[str, Any]) -> list[tuple[str, list[dict[str, Any]]]]:
    out = []
    for key, label in [
        ("scatter", "Embedding"),
        ("pca_scatter", "Embedding"),
        ("text_pca_scatter", "Text"),
        ("code_pca_scatter", "Code"),
    ]:
        points = record.get(key)
        if isinstance(points, list) and points:
            out.append((label, points))
    seen = set()
    deduped = []
    for label, points in out:
        marker = (label, id(points))
        if marker not in seen:
            seen.add(marker)
            deduped.append((label, points))
    return deduped


def points_df(points: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for point in points:
        try:
            rows.append({"x": float(point["x"]), "y": float(point["y"]), "label": str(point.get("label", ""))})
        except (KeyError, TypeError, ValueError):
            continue
    return pd.DataFrame(rows)


def plot_pca(run: dict[str, Any], assets_dir: Path) -> str | None:
    panels = []
    for record in selected_variants(run, limit=8):
        for space, points in scatter_points(record):
            panels.append((variant_plot_label(record.get("variant"), run), space, points))
    if not panels:
        return None
    cols = min(4, len(panels))
    rows = math.ceil(len(panels) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.8 * rows), squeeze=False)
    for ax, (variant, space, points) in zip(axes.ravel(), panels):
        df = points_df(points)
        if df.empty:
            ax.axis("off")
            continue
        sns.scatterplot(data=df, x="x", y="y", hue="label", palette="tab10", s=14, alpha=0.74, linewidth=0, legend=False, ax=ax)
        ax.set_title(f"{variant}\n{space}", fontsize=11)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
    for ax in axes.ravel()[len(panels):]:
        ax.axis("off")
    fig.suptitle(f"{run['short_title']} PCA Maps", fontsize=22, fontweight="bold")
    fig.tight_layout()
    return save_fig(fig, assets_dir / f"{run['slug']}_pca.png")


def plot_sphere(run: dict[str, Any], assets_dir: Path) -> str | None:
    panels = []
    for record in selected_variants(run, limit=8):
        points = scatter_points(record)
        if points:
            panels.append((variant_plot_label(record.get("variant"), run), points[0][0], points[0][1], record))
    if not panels:
        return None
    cols = min(4, len(panels))
    rows = math.ceil(len(panels) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4.0 * cols, 4.0 * rows), squeeze=False)
    for ax, (variant, space, points, record) in zip(axes.ravel(), panels):
        df = points_df(points)
        if df.empty:
            ax.axis("off")
            continue
        x = df["x"].to_numpy()
        y = df["y"].to_numpy()
        x = (x - x.mean()) / max(x.std(), 1e-9)
        y = (y - y.mean()) / max(y.std(), 1e-9)
        radius = np.maximum(np.sqrt(x * x + y * y), 1.0)
        x = x / radius * 0.96
        y = y / radius * 0.96
        for r in [0.25, 0.5, 0.75, 1.0]:
            circle = plt.Circle((0, 0), r, fill=False, color="#d9e3d6", linewidth=1.0)
            ax.add_artist(circle)
        sns.scatterplot(x=x, y=y, hue=df["label"], palette="tab10", s=14, alpha=0.76, linewidth=0, legend=False, ax=ax)
        ax.set_aspect("equal")
        ax.set_xlim(-1.08, 1.08)
        ax.set_ylim(-1.08, 1.08)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"{variant}\n{space}: RankMe {fmt(rank_value(record))}, cos {fmt(mean_cosine(record))}", fontsize=10)
    for ax in axes.ravel()[len(panels):]:
        ax.axis("off")
    fig.suptitle(f"{run['short_title']} Hypersphere Proxy", fontsize=22, fontweight="bold")
    fig.tight_layout()
    return save_fig(fig, assets_dir / f"{run['slug']}_sphere.png")


def hist_specs(record: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    specs = []
    for key, label in [
        ("cosine_histogram", "Embedding"),
        ("text_cosine_histogram", "Text"),
        ("code_cosine_histogram", "Code"),
    ]:
        value = record.get(key)
        if isinstance(value, dict) and value.get("counts"):
            specs.append((label, value))
    return specs


def spectrum_specs(record: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    specs = []
    for key, label in [
        ("eigen_spectrum", "Embedding"),
        ("text_eigen_spectrum", "Text"),
        ("code_eigen_spectrum", "Code"),
    ]:
        value = record.get(key)
        if isinstance(value, dict) and value.get("mass"):
            specs.append((label, value))
    return specs


def plot_cosine_hists(run: dict[str, Any], assets_dir: Path) -> str | None:
    panels = []
    for record in selected_variants(run, limit=8):
        for label, hist in hist_specs(record):
            panels.append((variant_plot_label(record.get("variant"), run), label, hist))
    if not panels:
        return None
    cols = min(4, len(panels))
    rows = math.ceil(len(panels) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.3 * rows), squeeze=False)
    for ax, (variant, label, hist) in zip(axes.ravel(), panels):
        counts = np.array(hist.get("counts", []), dtype=float)
        bins = np.linspace(-1, 1, len(counts) + 1)
        centers = 0.5 * (bins[:-1] + bins[1:])
        ax.bar(centers, counts, width=2 / max(len(counts), 1), color="#276a8c", alpha=0.86)
        ax.axvline(0, color="#333", linewidth=1)
        ax.set_title(f"{variant}\n{label}", fontsize=10)
        ax.set_xlabel("Pairwise cosine")
        ax.set_ylabel("count")
    for ax in axes.ravel()[len(panels):]:
        ax.axis("off")
    fig.suptitle(f"{run['short_title']} Cosine Histograms", fontsize=22, fontweight="bold")
    fig.tight_layout()
    return save_fig(fig, assets_dir / f"{run['slug']}_cosine_hists.png")


def plot_spectra(run: dict[str, Any], assets_dir: Path) -> str | None:
    panels = []
    for record in selected_variants(run, limit=8):
        for label, spectrum in spectrum_specs(record):
            panels.append((variant_plot_label(record.get("variant"), run), label, spectrum))
    if not panels:
        return None
    cols = min(4, len(panels))
    rows = math.ceil(len(panels) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.3 * rows), squeeze=False)
    for ax, (variant, label, spectrum) in zip(axes.ravel(), panels):
        mass = np.array(spectrum.get("mass", []), dtype=float)
        sns.lineplot(x=np.arange(1, len(mass) + 1), y=mass, marker="o", linewidth=2.0, ax=ax, color="#287c62")
        ax.set_title(f"{variant}\n{label}", fontsize=10)
        ax.set_xlabel("component")
        ax.set_ylabel("eigenvalue mass")
    for ax in axes.ravel()[len(panels):]:
        ax.axis("off")
    fig.suptitle(f"{run['short_title']} Eigenspectra", fontsize=22, fontweight="bold")
    fig.tight_layout()
    return save_fig(fig, assets_dir / f"{run['slug']}_spectra.png")


def plot_anchor_geometry(run: dict[str, Any], assets_dir: Path) -> str | None:
    rows = []
    for name, diag in (run.get("anchor_geometry") or {}).items():
        rows.extend(
            [
                {"anchor": name, "metric": "RankMe", "value": get_path(diag, "rankme")},
                {"anchor": name, "metric": "Top eig mass", "value": get_path(diag, "top_eig_mass_ratio")},
                {"anchor": name, "metric": "Mean cosine", "value": get_path(diag, "cosine_stats.mean")},
                {"anchor": name, "metric": "Uniformity", "value": get_path(diag, "uniformity_after_normalize")},
            ]
        )
    rows = [row for row in rows if finite(row["value"])]
    if not rows:
        return None
    df = pd.DataFrame(rows)
    metrics = list(dict.fromkeys(df["metric"]))
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    for ax, metric in zip(axes.ravel(), metrics[:4]):
        local = df[df["metric"] == metric]
        sns.barplot(data=local, y="anchor", x="value", hue="anchor", dodge=False, legend=False, ax=ax)
        ax.set_title(metric)
        ax.set_xlabel("")
        ax.set_ylabel("")
    fig.suptitle(f"{run['short_title']} Anchor Geometry", fontsize=22, fontweight="bold")
    fig.tight_layout()
    return save_fig(fig, assets_dir / f"{run['slug']}_anchors.png")


def plot_retrieval(run: dict[str, Any], assets_dir: Path) -> str | None:
    rows = []
    for record in run.get("records", []):
        for key, value in record.items():
            if key.startswith("retrieval") and isinstance(value, dict):
                r1 = get_path(value, "recall@1")
                if finite(r1):
                    rows.append({"variant": variant_plot_label(record.get("variant"), run), "space": key.replace("retrieval_", "").replace("_", " "), "recall@1": r1})
    if not rows:
        return None
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(14, max(5, 0.42 * len(df))))
    sns.barplot(data=df, y="variant", x="recall@1", hue="space", ax=ax)
    ax.set_title(f"{run['short_title']} Retrieval Recall@1")
    ax.set_xlabel("recall@1")
    ax.set_ylabel("")
    ax.legend(fontsize=9)
    fig.tight_layout()
    return save_fig(fig, assets_dir / f"{run['slug']}_retrieval.png")


def heatmap_specs(record: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    specs = []
    for key, label in [
        ("embedding_similarity_heatmap", "Embedding similarity"),
        ("text_cap_to_text_anchor_heatmap", "Text cap -> text anchor"),
        ("code_cap_to_code_anchor_heatmap", "Code cap -> code anchor"),
        ("text_cap_to_code_anchor_heatmap", "Text cap -> code anchor"),
    ]:
        value = record.get(key)
        if isinstance(value, dict) and value.get("values"):
            specs.append((label, value))
    return specs


def plot_heatmaps(run: dict[str, Any], assets_dir: Path) -> str | None:
    panels = []
    for record in selected_variants(run, limit=6):
        for label, heatmap in heatmap_specs(record):
            panels.append((variant_plot_label(record.get("variant"), run), label, heatmap))
    if not panels:
        return None
    panels = panels[:12]
    cols = min(4, len(panels))
    rows = math.ceil(len(panels) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4.1 * cols, 3.8 * rows), squeeze=False)
    for ax, (variant, label, heatmap) in zip(axes.ravel(), panels):
        values = np.array(heatmap.get("values", []), dtype=float)
        sns.heatmap(values, cmap="vlag", center=0, vmin=-1, vmax=1, cbar=False, xticklabels=False, yticklabels=False, ax=ax)
        ax.set_title(f"{variant}\n{label}", fontsize=11)
    for ax in axes.ravel()[len(panels):]:
        ax.axis("off")
    fig.suptitle(f"{run['short_title']} Similarity Heatmaps", fontsize=22, fontweight="bold")
    fig.tight_layout()
    return save_fig(fig, assets_dir / f"{run['slug']}_heatmaps.png")


def plot_history(run: dict[str, Any], assets_dir: Path) -> str | None:
    rows = []
    for record in run.get("records", []):
        for point in record.get("history") or []:
            if finite(point.get("step")) and finite(point.get("loss")):
                rows.append({"variant": variant_plot_label(record.get("variant"), run), "step": float(point["step"]), "loss": float(point["loss"])})
    if not rows:
        return None
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.lineplot(data=df, x="step", y="loss", hue="variant", marker="o", ax=ax)
    ax.set_title(f"{run['short_title']} Training Loss Traces")
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.legend(ncol=3, fontsize=9)
    fig.tight_layout()
    return save_fig(fig, assets_dir / f"{run['slug']}_history.png")


def build_metric_table(run: dict[str, Any]) -> str:
    columns = [
        ("RankMe", "rankme"),
        ("Text RankMe", "text_rankme"),
        ("Code RankMe", "code_rankme"),
        ("Within RankMe", "within_class_rankme._mean_within_class"),
        ("Mean Cos", "cosine.mean"),
        ("Text Cos", "text_cosine.mean"),
        ("Code Cos", "code_cosine.mean"),
        ("Uniformity", "uniformity"),
        ("Probe", "linear_probe_top1"),
        ("Direct R@1", "retrieval_embedding.recall@1"),
        ("Pred R@1", "retrieval_predicted_code_embedding.recall@1"),
        ("Exact Match", "eval_exact_match"),
    ]
    active = [(label, path) for label, path in columns if any(finite(get_path(record, path)) for record in run.get("records", []))]
    if not active:
        return ""
    rows = []
    for record in run.get("records", []):
        code = str(record.get("variant"))
        cells = [
            f"<td>{html.escape(variant_name(code, run))}</td>",
            f"<td><code>{html.escape(code)}</code></td>",
        ]
        for _, path in active:
            kind = "percent" if path in {"linear_probe_top1", "eval_exact_match"} or path.endswith("recall@1") else "number"
            cells.append(f"<td class='num'>{html.escape(fmt(get_path(record, path), kind))}</td>")
        rows.append(f"<tr>{''.join(cells)}</tr>")
    return f"""
    <div class="table-wrap">
      <table>
        <thead><tr><th>Variant</th><th>Code</th>{''.join(f'<th>{html.escape(label)}</th>' for label, _ in active)}</tr></thead>
        <tbody>{''.join(rows)}</tbody>
      </table>
    </div>
    """


def contrast_rows(run: dict[str, Any]) -> str:
    records = by_variant(run)
    pairs = [
        ("D1a", "D0a", "Sphere noise on continuous anchor"),
        ("D1b", "D0b", "Sphere noise on RFF anchor"),
        ("D1c", "D0c", "Sphere noise on frozen-teacher anchor"),
        ("D1a", "C", "Frozen continuous anchor vs co-trained self-anchor"),
        ("D1c", "C", "Frozen teacher anchor vs co-trained self-anchor"),
        ("D1", "D0", "Sphere noise on frozen anchor"),
        ("D1", "C", "Frozen anchor vs co-trained self-anchor"),
        ("D1", "D2", "Sample-specific vs class-level"),
        ("D_own_1", "D_own_0", "Sphere noise on frozen own-anchor"),
        ("D_own_1", "C", "Frozen own-anchor vs co-trained self-anchor"),
        ("D_own_1", "C_ema", "Frozen own-anchor vs momentum self-anchor"),
        ("D_cross_1", "D_cross_0", "Sphere noise on cross-view anchor"),
        ("F", "D_own_1", "Regularizer baseline vs noisy frozen own-anchor"),
        ("G", "D_own_1", "Regularizer baseline vs noisy frozen own-anchor"),
        ("H", "D_own_1", "SIGReg baseline vs noisy frozen own-anchor"),
        ("D1", "Regular", "Noisy frozen anchor vs plain fine-tune"),
        ("C", "Regular", "Co-trained self-anchor vs plain fine-tune"),
        ("C_ema", "C", "Momentum self-anchor vs co-trained self-anchor"),
        ("Both", "D1", "Combined anchors vs noisy own-anchor"),
        ("Cross", "D1", "Cross-view anchor vs noisy own-anchor"),
    ]
    rows = []
    for left, right, label in pairs:
        if left not in records or right not in records:
            continue
        l_rec = records[left]
        r_rec = records[right]
        rows.append(
            "<tr>"
            f"<td>{html.escape(label)}</td>"
            f"<td>{html.escape(variant_label(left, run))} - {html.escape(variant_label(right, run))}</td>"
            f"<td class='num'>{html.escape(fmt(rank_value(l_rec) - rank_value(r_rec)))}</td>"
            f"<td class='num'>{html.escape(fmt(mean_cosine(l_rec) - mean_cosine(r_rec)))}</td>"
            f"<td class='num'>{html.escape(fmt(get_path(l_rec, 'retrieval_predicted_code_embedding.recall@1') - get_path(r_rec, 'retrieval_predicted_code_embedding.recall@1'), 'pp'))}</td>"
            f"<td class='num'>{html.escape(fmt(get_path(l_rec, 'eval_exact_match') - get_path(r_rec, 'eval_exact_match'), 'pp'))}</td>"
            "</tr>"
        )
    if not rows:
        return ""
    return f"""
    <div class="table-wrap">
      <table>
        <thead><tr><th>Question</th><th>Pair</th><th>Delta RankMe</th><th>Delta Cos</th><th>Delta Pred R@1</th><th>Delta EM</th></tr></thead>
        <tbody>{''.join(rows)}</tbody>
      </table>
    </div>
    """


def run_reading_notes(run: dict[str, Any]) -> list[str]:
    slug = run["slug"]
    if slug == "exp0_full":
        return [
            "Start with co-trained self-anchor: it is the collapse demonstration. The useful pattern is noisy-anchor variants moving up in RankMe and away from cosine=1 compared with co-trained and matched clean-anchor controls.",
            "Clean/noisy continuous, RFF, and frozen-teacher anchors are matched pairs. The only intended difference is noisy cap reconstruction.",
        ]
    if slug == "exp1_cifar_gpu":
        return [
            "Separate geometry from task usefulness. The noisy frozen anchor can improve spread while still losing probe accuracy.",
            "The class-anchor control is not a fair unsupervised win because it uses labels; use it to see what class-level anchoring can and cannot explain.",
        ]
    if slug == "exp2_pythia160m_synth_input_white_anchor_white":
        return [
            "This is the main frozen-LLM diagnostic. The headline read is noisy frozen own-anchor beating clean frozen own-anchor, and strongly beating co-trained self-anchor, on geometry and predicted-code retrieval.",
            "Direct embedding retrieval is a different question. InfoNCE/VICReg can win there without disproving the cap-anchor geometry mechanism.",
        ]
    if slug.startswith("exp2_pythia160m"):
        return [
            "Use this as a preprocessing control. Raw Pythia states are highly anisotropic, so failures here mostly tell us that the source geometry is hostile.",
            "Compare noisy frozen own-anchor to clean frozen own-anchor within the same preprocessing mode, then compare the whole mode against source+anchor whitening.",
        ]
    if slug == "exp3_full_synth":
        return [
            "This is downstream task behavior, not a pure geometry diagnostic. Exact match answers whether the final generator got better.",
            "Noisy frozen anchor minus clean frozen anchor is the matched noise comparison. Noisy frozen anchor minus plain fine-tune asks whether the cap objective helps beyond standard supervised fine-tuning.",
        ]
    if slug == "exp3_llama1b_synth":
        return [
            "Read this as the stronger-base full ablation. It checks whether the SmolLM2 downstream pattern changes when the base model is Llama 1B.",
            "Use noisy frozen anchor minus clean frozen anchor for the matched noise comparison, and noisy frozen anchor minus co-trained/momentum self-anchor for the external-anchor comparison.",
        ]
    return ["Use matched pairs first, then compare against co-trained and regularizer baselines."]


def variant_method_description(code: str, run: dict[str, Any]) -> str:
    slug = run["slug"]
    if code == "A":
        return "Plain JEPA alignment: one view predicts the paired view with MSE. No frozen anchor and no sphere-noise recovery."
    if code == "B":
        return "Cosine JEPA alignment on normalized embeddings. No frozen anchor and no sphere-noise recovery."
    if code == "C":
        return "Co-trained self-anchor: the target is produced by the model being trained now. This is the collapse-prone moving-target control."
    if code == "C_detach":
        return "Detached self-anchor: the current target is stopped from receiving gradient, but it still comes from the current moving model."
    if code == "C_ema":
        return "Momentum self-anchor: the target comes from a slow-moving EMA copy. More stable than co-trained, but still not a fixed external anchor."
    if code in {"D0", "D_own_0"}:
        return "Clean frozen anchor: use the same fixed sample-specific anchor as the noisy variant, but with no sphere noise."
    if code in {"D1", "D_own_1"}:
        return "Noisy frozen anchor: perturb the spherical embedding and reconstruct that same sample's fixed anchor. This is the main sphere-anchor method."
    if code in {"D0a", "D1a"}:
        return "Toy continuous anchor: reconstruct the original synthetic point; the D1 version adds sphere noise and the D0 version does not."
    if code in {"D0b", "D1b"}:
        return "Toy random-feature anchor: reconstruct random Fourier features; the D1 version adds sphere noise and the D0 version does not."
    if code in {"D0c", "D1c"}:
        return "Toy frozen-teacher anchor: reconstruct features from a fixed teacher; the D1 version adds sphere noise and the D0 version does not."
    if code in {"D2", "D3"}:
        return "Class-anchor control: predict a low-cardinality class label. Useful warning case because class spread can hide within-class collapse."
    if code in {"D_cross_0", "D_cross_1"}:
        return "Frozen cross-view anchor: text-side cap reconstructs the paired code-side anchor. The _1 version adds sphere noise."
    if code in {"D_cross_sym_0", "D_cross_sym_1"}:
        return "Two-way frozen cross-view anchor: text predicts code anchor and code predicts text anchor. The _1 version adds sphere noise."
    if code == "F":
        if slug.startswith("exp2_pythia160m"):
            return "InfoNCE contrastive baseline: pulls true text/code pairs together and pushes other batch pairs apart."
        return "VICReg baseline: matches paired views while explicitly discouraging variance collapse and correlated dimensions."
    if code == "G":
        if slug.startswith("exp2_pythia160m"):
            return "VICReg baseline: matches paired text/code views while discouraging variance collapse and correlated dimensions."
        return "SIGReg baseline: alignment plus an isotropic Gaussian-style regularizer on the learned embeddings."
    if code == "H":
        return "SIGReg baseline: JEPA alignment plus a centered, identity-covariance regularizer. No frozen anchor and no sphere-noise recovery."
    if code == "Regular":
        return "Plain supervised fine-tune. The model learns to generate regex strings without any cap-anchor objective."
    if code == "Cross":
        return "Full fine-tune with a frozen cross-view cap objective in addition to language-model training."
    if code == "Both":
        return "Full fine-tune with both own-view and cross-view frozen cap objectives."
    return "See the variant glossary for this method."


def config_value(value: Any) -> str:
    if value is None:
        return "not set"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def run_setup_rows(run: dict[str, Any]) -> list[tuple[str, str]]:
    slug = run["slug"]
    cfg = run.get("config", {})
    cache = run.get("cache_config", {})
    variants = ", ".join(str(record.get("variant")) for record in run.get("records", []))
    if slug == "exp0_full":
        return [
            ("Data", f"{config_value(cfg.get('clusters'))} synthetic clusters, {config_value(cfg.get('samples_per_cluster'))} samples per cluster."),
            ("Views", "Two noisy paired views of the same synthetic point."),
            ("Training", f"{config_value(cfg.get('steps'))} steps, embedding dim {config_value(cfg.get('emb_dim'))}, sigma_max {config_value(cfg.get('sigma_max'))}."),
            ("Methods", variants),
            ("Task output", "No downstream generator. This run is only about representation geometry and collapse controls."),
        ]
    if slug == "exp1_cifar_gpu":
        return [
            ("Data", f"CIFAR-10 subset: {config_value(cfg.get('train_limit'))} train images and {config_value(cfg.get('test_limit'))} test images."),
            ("Views", "Two image augmentations of the same image."),
            ("Backbone", f"{config_value(cfg.get('arch'))} encoder with frozen autoencoder teacher anchors from {config_value(cfg.get('teacher_ckpt'))}."),
            ("Training", f"{config_value(cfg.get('epochs'))} representation epochs, {config_value(cfg.get('probe_epochs'))} probe epochs, batch {config_value(cfg.get('batch_size'))}."),
            ("Methods", variants),
        ]
    if slug.startswith("exp2_pythia160m"):
        input_pre = cfg.get("input_preprocess", "raw")
        anchor_pre = cfg.get("anchor_preprocess", "raw")
        return [
            ("Data", f"Natural-language/regex synthetic pairs from {config_value(cache.get('input') or cfg.get('cache'))}."),
            ("Frozen model", f"{config_value(cache.get('model_name'))}, {config_value(cache.get('pooling'))}-token pooled hidden states. The LLM itself is not trained."),
            ("Trainable parts", f"Small text/code projection heads and cap heads for {config_value(cfg.get('epochs'))} epochs; embedding dim {config_value(cfg.get('emb_dim'))}."),
            ("Preprocessing", f"Source/input states: {config_value(input_pre)}. Frozen anchors: {config_value(anchor_pre)}."),
            ("Split", f"{config_value(cfg.get('train_fraction'))} train fraction, seed {config_value(cfg.get('seed'))}."),
            ("Methods", variants),
        ]
    if slug.startswith("exp3"):
        return [
            ("Task", config_value(cfg.get("task"))),
            ("Base model", config_value(cfg.get("base_model"))),
            ("Data", f"{config_value(cfg.get('train_examples'))} train prompts and {config_value(cfg.get('test_examples'))} held-out prompts."),
            ("Training", f"{config_value(cfg.get('epochs'))} epochs, batch {config_value(cfg.get('batch_size'))}, learning rate {config_value(cfg.get('learning_rate'))}."),
            ("Evaluation", config_value(cfg.get("eval_metric"))),
            ("Methods", variants),
        ]
    return [("Methods", variants)]


def run_expected_patterns(run: dict[str, Any]) -> list[str]:
    slug = run["slug"]
    if slug == "exp0_full":
        return [
            "Co-trained self-anchor should have very low RankMe or very high cosine if the collapse failure mode is present.",
            "Noisy anchors should beat their matched clean anchors if sphere noise is useful.",
            "Class-anchor can look strong globally; within-class RankMe tells whether it actually preserves sample-level variation.",
        ]
    if slug == "exp1_cifar_gpu":
        return [
            "Noisy frozen anchor should beat clean frozen anchor on geometry if the mechanism transfers to images.",
            "Linear-probe accuracy is the task sanity check. Geometry improving while probe drops means the representation changed, but not in a task-useful way here.",
            "Class-anchor is supervised and should not be counted as an unsupervised mechanism win.",
        ]
    if slug == "exp2_pythia160m_synth_input_white_anchor_white":
        return [
            "This source+anchor-whitened run is the cleanest Exp2 mechanism comparison. It focuses on D_own/C/C_ema/F/G/H and does not include the plain A/B JEPA-only controls.",
            "Noisy frozen own-anchor should beat clean frozen own-anchor if sphere noise helps.",
            "Noisy frozen own-anchor should beat co-trained and momentum self-anchors if fixed external anchors matter.",
            "InfoNCE may win direct embedding retrieval because it trains directly for paired retrieval. The cap-anchor question is better read through predicted-code and cap-to-anchor retrieval.",
        ]
    if slug.startswith("exp2_pythia160m"):
        return [
            "Use this as a preprocessing ablation, not the main mechanism claim.",
            "If raw or norm-only runs collapse, that mainly says frozen Pythia states are anisotropic and hard for small heads.",
            "Compare clean/noisy frozen own-anchor within this preprocessing mode, then compare the whole mode against source+anchor whitening.",
        ]
    if slug.startswith("exp3"):
        return [
            "Plain fine-tune is the baseline to beat for downstream generation.",
            "Noisy frozen anchor minus clean frozen anchor isolates the sphere-noise contribution.",
            "Exact match matters more than training loss because cap objectives change the loss scale across variants.",
        ]
    return ["Use matched controls first; absolute metric values are secondary."]


def render_pairs_table(rows: list[tuple[str, str]], first: str, second: str, *, small: bool = True) -> str:
    body = "".join(f"<tr><td>{html.escape(left)}</td><td>{html.escape(right)}</td></tr>" for left, right in rows)
    cls = "table-wrap small" if small else "table-wrap"
    return f"""
    <div class="{cls}">
      <table>
        <thead><tr><th>{html.escape(first)}</th><th>{html.escape(second)}</th></tr></thead>
        <tbody>{body}</tbody>
      </table>
    </div>
    """


def render_method_table(run: dict[str, Any]) -> str:
    rows = [
        (variant_label(record.get("variant"), run), variant_method_description(str(record.get("variant")), run))
        for record in run.get("records", [])
    ]
    return render_pairs_table(rows, "Method", "What it does in this section")


def render_setup_table(run: dict[str, Any]) -> str:
    return render_pairs_table(run_setup_rows(run), "Run detail", "Value")


def render_expected_list(run: dict[str, Any]) -> str:
    items = "".join(f"<li>{html.escape(item)}</li>" for item in run_expected_patterns(run))
    return f"<ul>{items}</ul>"


def fig_card(src: str | None, title: str, note_key: str) -> str:
    if not src:
        return ""
    asset_path = f"assets/dashboard/{html.escape(src)}"
    return f"""
    <article class="figure-card">
      <div class="figure-head">
        <h4>{html.escape(title)}</h4>
        <a href="{asset_path}" target="_blank" rel="noopener">Open full-size PNG</a>
      </div>
      <a class="figure-link" href="{asset_path}" target="_blank" rel="noopener" aria-label="Open {html.escape(title)} full-size">
        <img src="{asset_path}" alt="{html.escape(title)}" loading="lazy" decoding="async">
      </a>
      <p><strong>How to read:</strong> {html.escape(FIGURE_NOTES[note_key])}</p>
    </article>
    """


def render_glossary() -> str:
    variant_rows = "".join(
        f"<tr><td><code>{html.escape(code)}</code></td><td>{html.escape(name)}</td><td>{html.escape(desc)}</td></tr>"
        for code, name, desc in VARIANT_GLOSSARY
    )
    metric_rows_html = "".join(
        f"<tr><td>{html.escape(metric)}</td><td>{html.escape(desc)}</td></tr>" for metric, desc in METRIC_GLOSSARY
    )
    task_rows_html = "".join(
        f"<tr><td>{html.escape(name)}</td><td>{html.escape(task)}</td><td>{html.escape(question)}</td></tr>"
        for name, task, question in TASK_GLOSSARY
    )
    return f"""
    <section class="panel" id="glossary">
      <h2>How to Read This</h2>
      <div class="callout">
        <p><strong>Are we comparing against JEPA?</strong> Yes, but not only JEPA versus no-JEPA. The JEPA-style family is the predictor/target setup: MSE JEPA, cosine JEPA, co-trained self-anchor, momentum self-anchor, and the frozen-anchor sphere variants. Plain fine-tune is the non-JEPA downstream baseline; InfoNCE, VICReg, and SIGReg are non-JEPA anti-collapse baselines.</p>
        <p><strong>What we are going for:</strong> a useful sphere-cap mechanism should make noisy frozen-anchor variants beat their matched clean-anchor controls, and beat self-anchor targets that can move with the encoder and collapse. High RankMe plus low mean cosine is the main geometry signature.</p>
      </div>
      <div class="guide-grid">
        <article class="text-card">
          <h3>Method Ideas In Plain Language</h3>
          {render_pairs_table(METHOD_GUIDE, 'Term', 'Meaning')}
        </article>
        <article class="text-card">
          <h3>Whitening</h3>
          <p>Whitening is especially important in the frozen LLM sections. It makes the source or anchor cloud less dominated by a few common directions before small heads are trained.</p>
          {render_pairs_table(WHITENING_GUIDE, 'Term', 'Meaning')}
        </article>
        <article class="text-card">
          <h3>Expected Patterns</h3>
          {render_pairs_table(EXPECTED_PATTERN_GUIDE, 'Question', 'What would support it?')}
        </article>
      </div>
      <div class="two-col">
        <div>
          <h3>Variant Glossary</h3>
          <div class="table-wrap small">
            <table>
              <thead><tr><th>Code</th><th>Meaning</th><th>How to interpret it</th></tr></thead>
              <tbody>{variant_rows}</tbody>
            </table>
          </div>
        </div>
        <div>
          <h3>Metric Glossary</h3>
          <div class="table-wrap small">
            <table>
              <thead><tr><th>Metric/Figure</th><th>Reading rule</th></tr></thead>
              <tbody>{metric_rows_html}</tbody>
            </table>
          </div>
        </div>
      </div>
      <h3>Task Map</h3>
      <div class="table-wrap small">
        <table>
          <thead><tr><th>Experiment</th><th>Task</th><th>Question</th></tr></thead>
          <tbody>{task_rows_html}</tbody>
        </table>
      </div>
    </section>
    """


def render_run(run: dict[str, Any], figures: dict[str, str | None]) -> str:
    notes = "".join(f"<li>{html.escape(note)}</li>" for note in run_reading_notes(run))
    takeaways = "".join(f"<li>{html.escape(item)}</li>" for item in run.get("takeaways", []))
    links = []
    if run.get("json_path"):
        links.append(f"<a href='{html.escape(run['json_path'])}'>summary JSON</a>")
    if run.get("html_path"):
        links.append(f"<a href='{html.escape(run['html_path'])}'>old per-run HTML</a>")
    fig_html = "".join(
        [
            fig_card(figures.get("metrics"), "Metric Overview", "metrics"),
            fig_card(figures.get("anchors"), "Anchor Geometry", "anchors"),
            fig_card(figures.get("pca"), "PCA Maps", "pca"),
            fig_card(figures.get("sphere"), "Hypersphere Proxy", "sphere"),
            fig_card(figures.get("hist"), "Cosine Histograms", "hist"),
            fig_card(figures.get("spectrum"), "Eigenspectra", "spectrum"),
            fig_card(figures.get("retrieval"), "Retrieval Bars", "retrieval"),
            fig_card(figures.get("heatmap"), "Similarity Heatmaps", "heatmap"),
            fig_card(figures.get("history"), "Training Traces", "history"),
        ]
    )
    return f"""
    <section class="panel run" id="{html.escape(run['slug'])}">
      <div class="run-head">
        <div>
          <p class="eyebrow">{html.escape(run['group'])}</p>
          <h2>{html.escape(run['title'])}</h2>
          <p>{html.escape(run['summary'])}</p>
          <p class="task-note"><strong>Task:</strong> {html.escape(RUN_TASK_NOTES.get(run['slug'], 'Use matched controls to separate geometry, noise, anchoring, and downstream behavior.'))}</p>
        </div>
        <div class="links">{' / '.join(links)}</div>
      </div>
      <div class="run-grid">
        <article class="text-card">
          <h3>What to Look For</h3>
          <ul>{notes}</ul>
          <h3>Current Read</h3>
          <ul>{takeaways}</ul>
        </article>
        <article class="text-card">
          <h3>How This Was Run</h3>
          {render_setup_table(run)}
        </article>
        <article class="text-card">
          <h3>Methods In This Section</h3>
          {render_method_table(run)}
        </article>
        <article class="text-card">
          <h3>Expected Pattern</h3>
          {render_expected_list(run)}
        </article>
        <article class="text-card">
          <h3>Headline Contrasts</h3>
          {contrast_rows(run) or '<p class="muted">No matched contrast pairs found for this run.</p>'}
        </article>
      </div>
      <div class="fig-grid">{fig_html}</div>
      <details>
        <summary>Metrics table</summary>
        {build_metric_table(run)}
      </details>
    </section>
    """


def build_figures(runs: list[dict[str, Any]], assets_dir: Path) -> tuple[dict[str, str | None], dict[str, dict[str, str | None]]]:
    assets_dir.mkdir(parents=True, exist_ok=True)
    for old in assets_dir.glob("*.png"):
        old.unlink()
    overview = {
        "overview": plot_overview(runs, assets_dir),
        "preprocessing": plot_preprocessing(runs, assets_dir),
    }
    per_run: dict[str, dict[str, str | None]] = {}
    for run in runs:
        per_run[run["slug"]] = {
            "metrics": plot_metrics(run, assets_dir),
            "anchors": plot_anchor_geometry(run, assets_dir),
            "pca": plot_pca(run, assets_dir),
            "sphere": plot_sphere(run, assets_dir),
            "hist": plot_cosine_hists(run, assets_dir),
            "spectrum": plot_spectra(run, assets_dir),
            "retrieval": plot_retrieval(run, assets_dir),
            "heatmap": plot_heatmaps(run, assets_dir),
            "history": plot_history(run, assets_dir),
        }
    return overview, per_run


def build_html(runs: list[dict[str, Any]], overview: dict[str, str | None], per_run: dict[str, dict[str, str | None]]) -> str:
    nav = "".join(f"<a href='#{html.escape(run['slug'])}'>{html.escape(run['short_title'])}</a>" for run in runs)
    run_sections = "".join(render_run(run, per_run[run["slug"]]) for run in runs)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Sphere-JEPA Results Report</title>
  <style>
    :root {{
      --bg: #f5f7f1;
      --panel: #ffffff;
      --soft: #f9fbf6;
      --ink: #1e2924;
      --muted: #66736d;
      --line: #dce5da;
      --accent: #276a8c;
      --green: #287c62;
      --red: #b9524d;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; background: var(--bg); color: var(--ink); }}
    a {{ color: var(--accent); text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    header {{ position: sticky; top: 0; z-index: 10; background: rgba(245,247,241,0.94); border-bottom: 1px solid var(--line); backdrop-filter: blur(12px); }}
    .top {{ max-width: 1440px; margin: 0 auto; padding: 13px 18px; display: grid; grid-template-columns: minmax(280px, 1fr) auto; gap: 16px; align-items: center; }}
    h1 {{ margin: 0; font-size: 23px; line-height: 1.15; }}
    .subtitle {{ margin: 4px 0 0; color: var(--muted); font-size: 13px; }}
    nav {{ display: flex; gap: 8px; flex-wrap: wrap; justify-content: flex-end; }}
    nav a, .links a {{ border: 1px solid var(--line); border-radius: 8px; background: var(--panel); padding: 6px 9px; font-size: 12px; color: var(--ink); }}
    main {{ max-width: 1760px; margin: 0 auto; padding: 18px; }}
    .panel {{ background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 18px; margin-bottom: 18px; box-shadow: 0 10px 26px rgba(30,41,36,0.06); }}
    .hero {{ display: grid; grid-template-columns: 1fr; gap: 18px; align-items: start; }}
    .callout {{ background: #eef5ee; border-left: 5px solid var(--green); padding: 12px 14px; border-radius: 7px; }}
    .two-col {{ display: grid; grid-template-columns: 1.1fr 0.9fr; gap: 18px; align-items: start; }}
    .guide-grid {{ display: grid; grid-template-columns: repeat(3, minmax(280px, 1fr)); gap: 14px; margin: 14px 0 18px; align-items: start; }}
    h2 {{ margin: 0 0 10px; font-size: 22px; }}
    h3 {{ margin: 0 0 9px; font-size: 16px; }}
    h4 {{ margin: 0 0 9px; font-size: 15px; }}
    p, li {{ line-height: 1.48; }}
    .muted, .figure-card p {{ color: var(--muted); }}
    .task-note {{ margin-top: 8px; color: #405047; }}
    .eyebrow {{ margin: 0 0 4px; color: var(--green); font-size: 12px; font-weight: 750; text-transform: uppercase; letter-spacing: 0; }}
    .figure-card, .text-card {{ background: var(--soft); border: 1px solid var(--line); border-radius: 8px; padding: 13px; min-width: 0; }}
    .figure-head {{ display: flex; justify-content: space-between; align-items: baseline; gap: 12px; margin-bottom: 9px; }}
    .figure-head h4 {{ margin: 0; }}
    .figure-head a {{ flex: 0 0 auto; font-size: 12px; font-weight: 700; border: 1px solid var(--line); border-radius: 7px; padding: 5px 8px; background: #fff; color: var(--accent); }}
    .figure-link {{ display: block; cursor: zoom-in; }}
    .figure-link:hover img {{ border-color: var(--accent); box-shadow: 0 0 0 3px rgba(39, 106, 140, 0.14); }}
    .figure-card img {{ display: block; width: 100%; height: auto; border: 1px solid var(--line); border-radius: 7px; background: white; transition: border-color 0.12s ease, box-shadow 0.12s ease; }}
    .fig-grid {{ display: grid; grid-template-columns: minmax(0, 1fr); gap: 18px; margin-top: 14px; }}
    .run-grid {{ display: grid; grid-template-columns: minmax(320px, 0.9fr) minmax(420px, 1.1fr); gap: 14px; }}
    .run-head {{ display: grid; grid-template-columns: minmax(0, 1fr) auto; gap: 14px; align-items: start; border-bottom: 1px solid var(--line); padding-bottom: 13px; margin-bottom: 14px; }}
    .links {{ display: flex; gap: 8px; flex-wrap: wrap; justify-content: flex-end; }}
    .table-wrap {{ width: 100%; overflow: auto; border: 1px solid var(--line); border-radius: 7px; background: white; }}
    .table-wrap.small {{ max-height: 560px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; min-width: 700px; }}
    th, td {{ padding: 8px 9px; border-bottom: 1px solid var(--line); vertical-align: top; text-align: left; }}
    th {{ background: #edf3ea; color: #415047; font-size: 11px; text-transform: uppercase; letter-spacing: 0; position: sticky; top: 0; }}
    td.num {{ text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }}
    code {{ font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace; }}
    summary {{ cursor: pointer; font-weight: 700; padding: 10px 0; }}
    @media (max-width: 900px) {{
      .top, .hero, .two-col, .guide-grid, .run-grid, .run-head {{ grid-template-columns: 1fr; }}
      nav, .links {{ justify-content: flex-start; }}
      .fig-grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <header>
    <div class="top">
      <div>
        <h1>Sphere-JEPA Results Report</h1>
        <p class="subtitle">Matplotlib/seaborn figures with a glossary and reading notes for every experiment. Click any figure to open the full-size PNG.</p>
      </div>
      <nav><a href="#glossary">How to read</a><a href="#overview">Overview</a>{nav}</nav>
    </div>
  </header>
  <main>
    {render_glossary()}
    <section class="panel" id="overview">
      <h2>Overview Figures</h2>
      <div class="hero">
        {fig_card(overview.get('overview'), 'Overall Geometry Map', 'overview')}
        {fig_card(overview.get('preprocessing'), 'Exp2 Preprocessing Sweep', 'preprocessing')}
      </div>
    </section>
    {run_sections}
  </main>
</body>
</html>
"""


def build_dashboard(summary_paths: list[Path], report_dir: Path, out: Path) -> None:
    setup_theme()
    runs = [load_run(path, report_dir) for path in ordered(summary_paths)]
    assets_dir = out.parent / "assets" / "dashboard"
    overview, per_run = build_figures(runs, assets_dir)
    rendered = build_html(runs, overview, per_run)
    rendered = "\n".join(line.rstrip() for line in rendered.splitlines()) + "\n"
    out.write_text(rendered)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a matplotlib/seaborn HTML report for sphere-JEPA summaries.")
    parser.add_argument("--report-dir", type=Path, default=Path("reports/gpu_2026-05-09"))
    parser.add_argument("--summary", action="append", type=Path, default=None, help="Summary JSON file. May be passed multiple times.")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    report_dir = args.report_dir
    summaries = args.summary or sorted((report_dir / "json").glob("*_summary.json"))
    if not summaries:
        raise SystemExit(f"no summary JSON files found under {report_dir / 'json'}")
    out = args.out or report_dir / "index.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    build_dashboard(summaries, report_dir, out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
