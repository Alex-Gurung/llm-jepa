from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import matplotlib.pyplot as plt
import seaborn as sns


METHODS = [
    {
        "key": "regular",
        "label": "Regular SFT",
        "short": "SFT",
        "description": "Original next-token fine-tuning baseline.",
        "paper": 0.5729,
        "paper_std": 0.0532,
    },
    {
        "key": "jepa_l1_p1",
        "label": "Original LLM-JEPA",
        "short": "JEPA",
        "description": "Original LLM-JEPA with lambda=1, k=1, no cap-anchor modifications.",
        "paper": 0.7146,
        "paper_std": 0.0134,
    },
]

COLORS = {"regular": "#2b6f8e", "jepa_l1_p1": "#bd6b28"}


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


def pct(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not math.isfinite(number):
        return "n/a"
    return f"{number * 100:.2f}%"


def num(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not math.isfinite(number):
        return "n/a"
    if abs(number) >= 1000:
        return f"{number:.0f}"
    if abs(number) >= 10:
        return f"{number:.2f}"
    return f"{number:.3f}"


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
                }
            )
        if "train_loss" in row:
            final = row
    out = {
        "train_status": "trained",
        "global_step": int(state.get("global_step", 0)),
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
    return out


def eval_summary(eval_file: Path) -> dict[str, Any]:
    rows = read_jsonl(eval_file)
    if not rows:
        return {"eval_status": "missing", "eval_count": 0}
    matches = sum(1 for row in rows if row.get("exact_match"))
    first_line = 0
    contains = 0
    for row in rows:
        generated = str(row.get("generated_response", "")).strip()
        target = str(row.get("ground_truth", "")).strip()
        generated_first = next((line.strip() for line in generated.splitlines() if line.strip()), "")
        first_line += int(generated_first == target)
        contains += int(bool(target) and target in generated)
    return {
        "eval_status": "evaluated",
        "eval_count": len(rows),
        "eval_matches": matches,
        "eval_exact_match": matches / len(rows),
        "eval_first_line_exact": first_line / len(rows),
        "eval_contains_target": contains / len(rows),
    }


def collect(results_root: Path, seeds: list[int]) -> list[dict[str, Any]]:
    records = []
    for seed in seeds:
        for method in METHODS:
            key = method["key"]
            model_dir = results_root / f"seed{seed}_{key}"
            eval_file = results_root / "eval" / f"seed{seed}_{key}.jsonl"
            row = {
                "seed": seed,
                "method": key,
                "variant": "Regular" if key == "regular" else "LLM_JEPA",
                "label": method["label"],
                "short": method["short"],
                "description": method["description"],
                "model_dir": str(model_dir),
                "eval_file": str(eval_file),
            }
            row.update(trainer_summary(model_dir))
            row.update(eval_summary(eval_file))
            records.append(row)
    return records


def aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_method = {}
    for method in METHODS:
        rows = [
            row
            for row in records
            if row["method"] == method["key"] and isinstance(row.get("eval_exact_match"), float)
        ]
        values = [float(row["eval_exact_match"]) for row in rows]
        by_method[method["key"]] = {
            "label": method["label"],
            "paper_exact_match": method["paper"],
            "paper_std": method["paper_std"],
            "n": len(values),
            "mean_exact_match": mean(values) if values else math.nan,
            "std_exact_match": stdev(values) if len(values) > 1 else 0.0 if values else math.nan,
            "values": values,
        }
    paired = []
    by_seed_method = {(row["seed"], row["method"]): row for row in records}
    for seed in sorted({row["seed"] for row in records}):
        regular = by_seed_method.get((seed, "regular"))
        jepa = by_seed_method.get((seed, "jepa_l1_p1"))
        if (
            regular
            and jepa
            and isinstance(regular.get("eval_exact_match"), float)
            and isinstance(jepa.get("eval_exact_match"), float)
        ):
            paired.append(
                {
                    "seed": seed,
                    "regular": float(regular["eval_exact_match"]),
                    "jepa_l1_p1": float(jepa["eval_exact_match"]),
                    "delta": float(jepa["eval_exact_match"]) - float(regular["eval_exact_match"]),
                }
            )
    deltas = [row["delta"] for row in paired]
    return {
        "methods": by_method,
        "paired": paired,
        "mean_delta": mean(deltas) if deltas else math.nan,
        "std_delta": stdev(deltas) if len(deltas) > 1 else 0.0 if deltas else math.nan,
    }


def plot_summary(summary: dict[str, Any], assets_dir: Path, stem: str) -> dict[str, str]:
    assets_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    methods = summary["methods"]
    labels = [method["short"] for method in METHODS]
    means = [methods[method["key"]]["mean_exact_match"] for method in METHODS]
    stds = [methods[method["key"]]["std_exact_match"] for method in METHODS]

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    sns.barplot(x=labels, y=[value * 100 for value in means], palette=[COLORS[method["key"]] for method in METHODS], hue=labels, dodge=False, legend=False, ax=ax)
    ax.errorbar(labels, [value * 100 for value in means], yerr=[value * 100 for value in stds], fmt="none", ecolor="#1f2522", capsize=4, linewidth=1.4)
    ax.scatter(labels, [methods[method["key"]]["paper_exact_match"] * 100 for method in METHODS], color="#1f2522", marker="D", label="paper mean")
    ax.set_ylabel("Exact match (%)")
    ax.set_title("NL-RX-SYNTH Replication: Mean Over Completed Seeds")
    ax.set_ylim(0, max(80, max([value * 100 for value in means if math.isfinite(value)] + [0]) + 8))
    ax.grid(axis="y", color="#d9ded8", linewidth=0.8)
    ax.legend(frameon=False)
    fig.tight_layout()
    path = assets_dir / f"{stem}_means.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths["means"] = str(path.relative_to(assets_dir.parent.parent))

    paired = summary["paired"]
    if paired:
        fig, ax = plt.subplots(figsize=(8.0, 4.8))
        for row in paired:
            ax.plot([0, 1], [row["regular"] * 100, row["jepa_l1_p1"] * 100], color="#89938d", linewidth=1.6)
            ax.scatter([0], [row["regular"] * 100], color=COLORS["regular"], s=48)
            ax.scatter([1], [row["jepa_l1_p1"] * 100], color=COLORS["jepa_l1_p1"], s=48)
            ax.text(1.03, row["jepa_l1_p1"] * 100, f"seed {row['seed']}", va="center", fontsize=8)
        ax.set_xticks([0, 1], ["Regular SFT", "Original LLM-JEPA"])
        ax.set_ylabel("Exact match (%)")
        ax.set_title("Paired Seed Deltas")
        ax.grid(axis="y", color="#d9ded8", linewidth=0.8)
        fig.tight_layout()
        path = assets_dir / f"{stem}_paired.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths["paired"] = str(path.relative_to(assets_dir.parent.parent))
    return paths


def render_table(records: list[dict[str, Any]]) -> str:
    rows = []
    for row in records:
        rows.append(
            "<tr>"
            f"<td>{row['seed']}</td>"
            f"<td>{html.escape(row['label'])}</td>"
            f"<td class='num'>{pct(row.get('eval_exact_match'))}</td>"
            f"<td class='num'>{pct(row.get('eval_contains_target'))}</td>"
            f"<td class='num'>{row.get('eval_matches', 0)}/{row.get('eval_count', 0)}</td>"
            f"<td class='num'>{num(row.get('train_loss'))}</td>"
            f"<td class='num'>{num(row.get('train_runtime'))} s</td>"
            f"<td>{html.escape(row.get('train_status', 'missing'))}/{html.escape(row.get('eval_status', 'missing'))}</td>"
            "</tr>"
        )
    return (
        "<div class='table-wrap'><table><thead><tr>"
        "<th>Seed</th><th>Method</th><th>Exact</th><th>Contains Target</th><th>Matches</th><th>Train Loss</th><th>Runtime</th><th>Status</th>"
        "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div>"
    )


def build_html(summary: dict[str, Any], records: list[dict[str, Any]], plots: dict[str, str], title: str) -> str:
    regular = summary["methods"]["regular"]
    jepa = summary["methods"]["jepa_l1_p1"]
    data = json.dumps({"summary": summary, "records": records}, indent=2, ensure_ascii=True)
    cards = [
        ("Completed pairs", str(len(summary["paired"])), "Seeds with both methods evaluated."),
        ("Regular mean", pct(regular["mean_exact_match"]), f"Paper: {pct(regular['paper_exact_match'])}."),
        ("LLM-JEPA mean", pct(jepa["mean_exact_match"]), f"Paper: {pct(jepa['paper_exact_match'])}."),
        ("JEPA - Regular", pct(summary["mean_delta"]), "Mean paired exact-match delta."),
    ]
    card_html = "".join(
        f"<article class='stat'><div class='label'>{html.escape(label)}</div><div class='value'>{html.escape(value)}</div><div class='note'>{html.escape(note)}</div></article>"
        for label, value, note in cards
    )
    plot_html = "".join(
        f"<article class='card'><a href='../{html.escape(path)}'><img src='../{html.escape(path)}' alt='{html.escape(name)} plot'></a></article>"
        for name, path in plots.items()
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>
    :root {{
      --bg: #f6f7f2;
      --panel: #fff;
      --ink: #1e2723;
      --muted: #65716c;
      --line: #dfe5dc;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; background: var(--bg); color: var(--ink); }}
    header, main {{ max-width: 1280px; margin: 0 auto; padding: 18px; }}
    h1 {{ margin: 0 0 8px; font-size: 24px; }}
    h2 {{ font-size: 18px; margin: 24px 0 10px; }}
    p {{ color: var(--muted); line-height: 1.45; max-width: 900px; }}
    .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 12px; }}
    .stat, .card {{ background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 14px; }}
    .label {{ color: var(--muted); font-size: 12px; text-transform: uppercase; }}
    .value {{ margin-top: 7px; font-size: 28px; font-weight: 760; }}
    .note {{ color: var(--muted); font-size: 12px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(380px, 1fr)); gap: 12px; }}
    img {{ width: 100%; display: block; }}
    .table-wrap {{ overflow: auto; border: 1px solid var(--line); border-radius: 8px; background: var(--panel); }}
    table {{ width: 100%; border-collapse: collapse; min-width: 780px; font-size: 13px; }}
    th, td {{ border-bottom: 1px solid var(--line); padding: 8px 9px; text-align: left; }}
    th {{ background: #eef3ec; color: #405047; font-size: 11px; text-transform: uppercase; }}
    .num {{ text-align: right; font-variant-numeric: tabular-nums; }}
    pre {{ white-space: pre-wrap; word-break: break-word; font-size: 12px; line-height: 1.45; }}
  </style>
</head>
<body>
  <header>
    <h1>{html.escape(title)}</h1>
    <p>Faithful NL-RX-SYNTH replication pass for Llama-3.2-1B-Instruct: original regular SFT versus original LLM-JEPA, lambda=1, k=1, lr=2e-5, four epochs, large effective batch. This intentionally excludes the new cap-anchor variants.</p>
  </header>
  <main>
    <section class="stats">{card_html}</section>
    <h2>Plots</h2>
    <section class="grid">{plot_html}</section>
    <h2>Runs</h2>
    {render_table(records)}
    <h2>Embedded JSON</h2>
    <article class="card"><pre>{html.escape(data)}</pre></article>
  </main>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Build original LLM-JEPA replication report.")
    parser.add_argument("--results-root", type=Path, default=Path("results/replicate_llm_jepa_synth"))
    parser.add_argument("--report-dir", type=Path, default=Path("reports/gpu_2026-05-09"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[82, 23, 37, 84, 4])
    parser.add_argument("--stem", default="replicate_llm_jepa_synth")
    args = parser.parse_args()
    setup_theme()

    records = collect(args.results_root, args.seeds)
    summary = aggregate(records)
    assets_dir = args.report_dir / "assets" / "replication"
    plots = plot_summary(summary, assets_dir, args.stem)

    out = {
        "config": {
            "experiment": args.stem,
            "base_model": "meta-llama/Llama-3.2-1B-Instruct",
            "dataset": "NL-RX-SYNTH",
            "learning_rate": 2e-5,
            "epochs": 4,
            "jepa_lambda": 1.0,
            "predictors": 1,
            "seeds": args.seeds,
        },
        "summary": summary,
        "records": records,
    }
    json_path = args.report_dir / "json" / f"{args.stem}_summary.json"
    html_path = args.report_dir / "html" / f"{args.stem}.html"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(out, indent=2, ensure_ascii=True) + "\n")
    html_path.write_text(build_html(summary, records, plots, "Original LLM-JEPA NL-RX-SYNTH Replication"))
    print(f"wrote {json_path}")
    print(f"wrote {html_path}")


if __name__ == "__main__":
    main()
