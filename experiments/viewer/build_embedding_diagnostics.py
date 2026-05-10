from __future__ import annotations

import argparse
import html
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F

from evaluate import (
    format_conversation,
    get_assistant_messages,
    get_user_messages,
    load_model_and_tokenizer,
)
from sphere_jepa.metrics import (
    cosine_histogram,
    cosine_pair_stats,
    eigen_spectrum,
    paired_vs_unpaired_cosine,
    rankme,
    retrieval_at_k,
    similarity_heatmap,
)


PALETTE = {
    "text": "#2b6f8e",
    "code": "#bd6b28",
    "paired": "#246f48",
    "unpaired": "#7d6ab6",
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


def read_jsonl(path: Path, max_examples: int) -> list[dict[str, Any]]:
    rows = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
            if len(rows) >= max_examples:
                break
    return rows


def parse_model_specs(specs: list[str]) -> list[tuple[str, str]]:
    parsed = []
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"model spec must be label=path, got {spec!r}")
        label, path = spec.split("=", 1)
        label = label.strip()
        path = path.strip()
        if not label or not path:
            raise ValueError(f"model spec must be label=path, got {spec!r}")
        parsed.append((label, path))
    return parsed


def prompt_pairs(rows: list[dict[str, Any]], original_model_name: str, tokenizer: Any) -> tuple[list[str], list[str]]:
    text_prompts = []
    code_prompts = []
    for row in rows:
        messages = row["messages"]
        text_messages = get_user_messages(original_model_name, messages)
        code_messages = get_assistant_messages(original_model_name, messages)
        text_prompts.append(format_conversation(text_messages, tokenizer, similarity=True))
        code_prompts.append(format_conversation(code_messages, tokenizer, include_assistant=True, similarity=True))
    return text_prompts, code_prompts


def infer_device(model: torch.nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def pool_hidden(hidden: torch.Tensor, attention_mask: torch.Tensor, pooling: str) -> torch.Tensor:
    if pooling == "last":
        return hidden[:, -1, :]
    if pooling == "mean":
        mask = attention_mask.to(hidden.device).unsqueeze(-1).float()
        return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
    if pooling == "cls":
        return hidden[:, 0, :]
    raise ValueError(f"unknown pooling: {pooling}")


def embed_prompts(
    model: torch.nn.Module,
    tokenizer: Any,
    prompts: list[str],
    *,
    batch_size: int,
    max_length: int,
    layer: int,
    pooling: str,
) -> torch.Tensor:
    device = infer_device(model)
    chunks = []
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start : start + batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True, use_cache=False)
        hidden = outputs.hidden_states[layer].float()
        pooled = pool_hidden(hidden, inputs["attention_mask"], pooling)
        chunks.append(pooled.detach().cpu())
    return torch.cat(chunks, dim=0)


def pca_coords(z: torch.Tensor, max_points: int = 700) -> tuple[torch.Tensor, torch.Tensor]:
    n = min(len(z), max_points)
    idx = torch.linspace(0, len(z) - 1, steps=n).round().long()
    sampled = z[idx].float()
    centered = sampled - sampled.mean(dim=0, keepdim=True)
    if centered.shape[1] == 1:
        coords = torch.cat([centered, torch.zeros_like(centered)], dim=1)
    else:
        q = min(2, centered.shape[0], centered.shape[1])
        _, _, components = torch.pca_lowrank(centered, q=q, center=False, niter=3)
        coords = centered @ components[:, :q]
        if coords.shape[1] == 1:
            coords = torch.cat([coords, torch.zeros_like(coords)], dim=1)
    return coords, idx


def model_metrics(text_z: torch.Tensor, code_z: torch.Tensor) -> dict[str, Any]:
    paired = paired_vs_unpaired_cosine(text_z, code_z)
    retrieval = retrieval_at_k(text_z, code_z, ks=(1, 5, 10))
    text_hist = cosine_histogram(text_z, bins=50)
    code_hist = cosine_histogram(code_z, bins=50)
    return {
        "text_rankme": rankme(text_z),
        "code_rankme": rankme(code_z),
        "text_cosine_stats": cosine_pair_stats(text_z),
        "code_cosine_stats": cosine_pair_stats(code_z),
        "text_eigen_spectrum": eigen_spectrum(text_z),
        "code_eigen_spectrum": eigen_spectrum(code_z),
        "text_cosine_histogram": text_hist,
        "code_cosine_histogram": code_hist,
        "text_to_code": paired,
        "text_to_code_retrieval": retrieval,
        "similarity_heatmap": similarity_heatmap(text_z, code_z, max_items=48),
    }


def save_fig(fig: plt.Figure, path: Path, report_dir: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return str(path.relative_to(report_dir))


def plot_pca(records: list[dict[str, Any]], assets_dir: Path, report_dir: Path, stem: str) -> str:
    cols = min(3, max(1, len(records)))
    rows = math.ceil(len(records) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5.4 * cols, 4.6 * rows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, record in zip(axes.flat, records):
        text_z = record["_text_embeddings"]
        code_z = record["_code_embeddings"]
        combined = torch.cat([text_z, code_z], dim=0)
        coords, idx = pca_coords(combined)
        n = len(idx) // 2
        labels = torch.arange(len(combined))[idx] >= len(text_z)
        ax.axis("on")
        ax.scatter(coords[~labels, 0], coords[~labels, 1], s=14, alpha=0.58, color=PALETTE["text"], label="text")
        ax.scatter(coords[labels, 0], coords[labels, 1], s=14, alpha=0.58, color=PALETTE["code"], label="code")
        ax.set_title(record["label"])
        ax.set_xticks([])
        ax.set_yticks([])
        if n:
            ax.legend(frameon=False, loc="best", fontsize=8)
    return save_fig(fig, assets_dir / f"{stem}_pca.png", report_dir)


def plot_metric_bars(records: list[dict[str, Any]], assets_dir: Path, report_dir: Path, stem: str) -> str:
    labels = [record["label"] for record in records]
    x = range(len(labels))
    width = 0.22
    fig, ax = plt.subplots(figsize=(max(7.0, 2.2 * len(labels)), 5.2))
    text_rank = [record["metrics"]["text_rankme"] for record in records]
    code_rank = [record["metrics"]["code_rankme"] for record in records]
    recall = [record["metrics"]["text_to_code_retrieval"]["recall@1"] * 100 for record in records]
    ax.bar([i - width for i in x], text_rank, width=width, color=PALETTE["text"], label="text RankMe")
    ax.bar(list(x), code_rank, width=width, color=PALETTE["code"], label="code RankMe")
    ax2 = ax.twinx()
    ax2.bar([i + width for i in x], recall, width=width, color=PALETTE["paired"], alpha=0.78, label="text->code R@1")
    ax.set_xticks(list(x), labels, rotation=25, ha="right")
    ax.set_ylabel("RankMe")
    ax2.set_ylabel("Retrieval@1 (%)")
    ax.set_title("Embedding Spread And Pair Retrieval")
    handles, handle_labels = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(handles + handles2, handle_labels + labels2, frameon=False, loc="upper left")
    return save_fig(fig, assets_dir / f"{stem}_metrics.png", report_dir)


def plot_cosine_hist(records: list[dict[str, Any]], assets_dir: Path, report_dir: Path, stem: str) -> str:
    cols = min(3, max(1, len(records)))
    rows = math.ceil(len(records) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5.4 * cols, 4.2 * rows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, record in zip(axes.flat, records):
        ax.axis("on")
        for space, color, label in [("text", PALETTE["text"], "text"), ("code", PALETTE["code"], "code")]:
            hist = record["metrics"][f"{space}_cosine_histogram"]
            edges = hist["bin_edges"]
            counts = hist["counts"]
            if not edges or not counts:
                continue
            centers = [(edges[i] + edges[i + 1]) / 2 for i in range(len(counts))]
            ax.plot(centers, counts, color=color, linewidth=2.0, label=label)
        ax.set_title(record["label"])
        ax.set_xlabel("Off-diagonal cosine")
        ax.set_ylabel("Count")
        ax.legend(frameon=False, fontsize=8)
    return save_fig(fig, assets_dir / f"{stem}_cosine_hists.png", report_dir)


def plot_heatmaps(records: list[dict[str, Any]], assets_dir: Path, report_dir: Path, stem: str) -> str:
    cols = min(3, max(1, len(records)))
    rows = math.ceil(len(records) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5.2 * cols, 4.6 * rows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, record in zip(axes.flat, records):
        ax.axis("on")
        values = record["metrics"]["similarity_heatmap"]["values"]
        sns.heatmap(values, ax=ax, cmap="crest", vmin=-1, vmax=1, cbar=False, xticklabels=False, yticklabels=False)
        ax.set_title(record["label"])
        ax.set_xlabel("code target")
        ax.set_ylabel("text query")
    return save_fig(fig, assets_dir / f"{stem}_heatmaps.png", report_dir)


def make_plots(records: list[dict[str, Any]], report_dir: Path, stem: str) -> dict[str, str]:
    assets_dir = report_dir / "assets" / "embedding_diagnostics"
    return {
        "pca": plot_pca(records, assets_dir, report_dir, stem),
        "metrics": plot_metric_bars(records, assets_dir, report_dir, stem),
        "cosine_hists": plot_cosine_hist(records, assets_dir, report_dir, stem),
        "heatmaps": plot_heatmaps(records, assets_dir, report_dir, stem),
    }


def pct(value: Any) -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not math.isfinite(value):
        return "n/a"
    return f"{value * 100:.2f}%"


def num(value: Any) -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not math.isfinite(value):
        return "n/a"
    return f"{value:.3f}"


def render_table(records: list[dict[str, Any]]) -> str:
    rows = []
    for record in records:
        metrics = record["metrics"]
        rows.append(
            "<tr>"
            f"<td>{html.escape(record['label'])}</td>"
            f"<td class='num'>{html.escape(num(metrics['text_rankme']))}</td>"
            f"<td class='num'>{html.escape(num(metrics['code_rankme']))}</td>"
            f"<td class='num'>{html.escape(num(metrics['text_cosine_stats']['mean']))}</td>"
            f"<td class='num'>{html.escape(num(metrics['code_cosine_stats']['mean']))}</td>"
            f"<td class='num'>{html.escape(num(metrics['text_to_code']['paired_mean']))}</td>"
            f"<td class='num'>{html.escape(num(metrics['text_to_code']['unpaired_mean']))}</td>"
            f"<td class='num'>{html.escape(pct(metrics['text_to_code_retrieval']['recall@1']))}</td>"
            f"<td class='num'>{html.escape(num(metrics['text_to_code_retrieval']['mean_rank']))}</td>"
            "</tr>"
        )
    return (
        "<div class='table-wrap'><table><thead><tr>"
        "<th>Model</th><th>Text RankMe</th><th>Code RankMe</th><th>Text mean cos</th><th>Code mean cos</th><th>Paired text-code cos</th><th>Unpaired text-code cos</th><th>Text->code R@1</th><th>Mean rank</th>"
        "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div>"
    )


def build_html(summary: dict[str, Any], plots: dict[str, str]) -> str:
    data = json.dumps(summary, indent=2, ensure_ascii=True)
    plot_cards = "".join(
        f"<article class='figure'><div class='figure-head'><h2>{html.escape(name.replace('_', ' ').title())}</h2><a href='../{html.escape(path)}'>Open PNG</a></div><a href='../{html.escape(path)}'><img src='../{html.escape(path)}' alt='{html.escape(name)}'></a></article>"
        for name, path in plots.items()
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Embedding Geometry Diagnostics</title>
  <style>
    :root {{
      --bg: #f6f7f2;
      --panel: #ffffff;
      --ink: #1e2723;
      --muted: #65716c;
      --line: #dfe5dc;
      --shadow: 0 8px 24px rgba(31, 39, 35, 0.08);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; background: var(--bg); color: var(--ink); }}
    header, main {{ max-width: 1420px; margin: 0 auto; padding: 18px; }}
    h1 {{ margin: 0 0 8px; font-size: 25px; }}
    h2 {{ margin: 0; font-size: 16px; }}
    p {{ color: var(--muted); line-height: 1.45; max-width: 980px; }}
    .fig-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(430px, 1fr)); gap: 12px; }}
    .figure, .card {{ background: var(--panel); border: 1px solid var(--line); border-radius: 8px; box-shadow: var(--shadow); padding: 14px; }}
    .figure-head {{ display: flex; justify-content: space-between; align-items: center; gap: 12px; margin-bottom: 10px; }}
    img {{ width: 100%; display: block; border-radius: 6px; }}
    a {{ color: #246a8f; text-decoration: none; font-weight: 650; }}
    .table-wrap {{ overflow: auto; border: 1px solid var(--line); border-radius: 8px; background: var(--panel); }}
    table {{ width: 100%; min-width: 920px; border-collapse: collapse; font-size: 13px; }}
    th, td {{ padding: 8px 9px; border-bottom: 1px solid var(--line); text-align: left; }}
    th {{ background: #eef3ec; color: #405047; font-size: 11px; text-transform: uppercase; }}
    .num {{ text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }}
    pre {{ margin: 0; white-space: pre-wrap; word-break: break-word; font-size: 12px; line-height: 1.45; }}
  </style>
</head>
<body>
  <header>
    <h1>Embedding Geometry Diagnostics</h1>
    <p>Fixed-subset hidden-state diagnostics for trained NL-RX-SYNTH checkpoints. These plots answer whether the cap objective changes representation spread, clustering, and text-code pairing; they do not replace exact-match generation as the primary result.</p>
  </header>
  <main>
    <section class="fig-grid">{plot_cards}</section>
    <article class="card"><h2>Metrics</h2>{render_table(summary['records'])}</article>
    <article class="card"><h2>Embedded JSON</h2><pre>{html.escape(data)}</pre></article>
  </main>
</body>
</html>
"""


def serializable_record(record: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in record.items() if not key.startswith("_")}


def run(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, str]]:
    rows = read_jsonl(args.input_file, args.max_examples)
    records = []
    for label, model_path in parse_model_specs(args.models):
        print(f"diagnostics: loading {label} from {model_path}", flush=True)
        model, tokenizer = load_model_and_tokenizer(
            model_path,
            args.original_model_name,
            device_map=args.device_map,
        )
        tokenizer.padding_side = "left"
        text_prompts, code_prompts = prompt_pairs(rows, args.original_model_name, tokenizer)
        text_z = embed_prompts(
            model,
            tokenizer,
            text_prompts,
            batch_size=args.batch_size,
            max_length=args.max_length,
            layer=args.layer,
            pooling=args.pooling,
        )
        code_z = embed_prompts(
            model,
            tokenizer,
            code_prompts,
            batch_size=args.batch_size,
            max_length=args.max_length,
            layer=args.layer,
            pooling=args.pooling,
        )
        records.append(
            {
                "label": label,
                "model_path": model_path,
                "num_examples": len(rows),
                "layer": args.layer,
                "pooling": args.pooling,
                "metrics": model_metrics(text_z, code_z),
                "_text_embeddings": text_z,
                "_code_embeddings": code_z,
            }
        )
        print(f"diagnostics: finished {label}", flush=True)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    plots = make_plots(records, args.report_dir, args.stem)
    summary = {
        "config": {
            "input_file": str(args.input_file),
            "original_model_name": args.original_model_name,
            "max_examples": args.max_examples,
            "batch_size": args.batch_size,
            "max_length": args.max_length,
            "layer": args.layer,
            "pooling": args.pooling,
            "models": args.models,
        },
        "records": [serializable_record(record) for record in records],
        "plots": plots,
    }
    return summary, plots


def main() -> None:
    parser = argparse.ArgumentParser(description="Build embedding geometry diagnostics for selected LLM-JEPA checkpoints.")
    parser.add_argument("--models", nargs="+", required=True, help="One or more label=path checkpoint specs.")
    parser.add_argument("--original-model-name", default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--input-file", type=Path, default=Path("datasets/synth_test.jsonl"))
    parser.add_argument("--report-dir", type=Path, default=Path("reports/gpu_2026-05-10"))
    parser.add_argument("--stem", default="embedding_diagnostics")
    parser.add_argument("--max-examples", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--layer", type=int, default=-1)
    parser.add_argument("--pooling", choices=["last", "mean", "cls"], default="last")
    parser.add_argument("--device-map", default="cuda:0")
    args = parser.parse_args()

    setup_theme()
    summary, plots = run(args)
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
