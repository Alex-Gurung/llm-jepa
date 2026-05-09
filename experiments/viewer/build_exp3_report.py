from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path
from typing import Any


SMOLLM_FULL_VARIANTS = [
    {
        "key": "regular",
        "variant": "Regular",
        "model_dir": "regular_model",
        "description": "Regular supervised fine-tune without JEPA cap loss.",
    },
    {
        "key": "c",
        "variant": "C",
        "model_dir": "c_model",
        "description": "Co-trained own-view cap target with sigma_max=0.5.",
    },
    {
        "key": "c_ema",
        "variant": "C_ema",
        "model_dir": "c_ema_model",
        "description": "EMA anchor target with sigma_max=0.5.",
    },
    {
        "key": "d0",
        "variant": "D0",
        "model_dir": "d0_model",
        "description": "Frozen own-view anchor control with sigma_max=0.0.",
    },
    {
        "key": "d1",
        "variant": "D1",
        "model_dir": "d1_model",
        "description": "Frozen own-view anchor with sigma_max=0.5.",
    },
    {
        "key": "cross",
        "variant": "Cross",
        "model_dir": "cross_model",
        "description": "Frozen cross-view cap reconstruction with sigma_max=0.5.",
    },
    {
        "key": "both",
        "variant": "Both",
        "model_dir": "both_model",
        "description": "Frozen own-view plus cross-view cap reconstruction with sigma_max=0.5.",
    },
]

LLAMA_PRIMARY_VARIANTS = [
    {
        "key": "regular",
        "variant": "Regular",
        "model_dir": "regular_model",
        "description": "Regular supervised fine-tune without JEPA cap loss.",
    },
    {
        "key": "c",
        "variant": "C",
        "model_dir": "c_model",
        "description": "Co-trained own-view cap target with sigma_max=0.5.",
    },
    {
        "key": "d0",
        "variant": "D0",
        "model_dir": "d0_model",
        "description": "Frozen own-view anchor control with sigma_max=0.0.",
    },
    {
        "key": "d1",
        "variant": "D1",
        "model_dir": "d1_model",
        "description": "Frozen own-view anchor with sigma_max=0.5.",
    },
]

VARIANT_SETS = {
    "smollm_full": SMOLLM_FULL_VARIANTS,
    "llama_full": SMOLLM_FULL_VARIANTS,
    "llama_primary": LLAMA_PRIMARY_VARIANTS,
}

PALETTE = [
    "#246a8f",
    "#b94e4e",
    "#1f7a63",
    "#bd812d",
    "#6654a2",
    "#188092",
    "#8d5a2b",
]


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


def count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open() as handle:
        return sum(1 for _ in handle)


def clip(text: Any, limit: int = 220) -> str:
    value = str(text or "")
    value = " ".join(value.split())
    if len(value) <= limit:
        return value
    return value[: limit - 1] + "..."


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


def finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def trainer_summary(model_dir: Path) -> dict[str, Any]:
    state = read_json(model_dir / "trainer_state.json")
    if state is None:
        return {"status": "missing"}

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
        "status": "trained",
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
                "train_steps_per_second": float(final.get("train_steps_per_second", math.nan)),
            }
        )
    if history:
        out["final_log_loss"] = float(history[-1]["loss"])
        out["min_log_loss"] = min(float(point["loss"]) for point in history)
    return out


def eval_summary(path: Path) -> dict[str, Any]:
    rows = read_jsonl(path)
    if not rows:
        return {"eval_status": "missing", "eval_count": 0}

    matches = sum(1 for row in rows if row.get("exact_match"))
    starts = 0
    first_line_matches = 0
    contains_matches = 0
    generated_lengths = [len(str(row.get("generated_response", ""))) for row in rows]
    mismatches = [row for row in rows if not row.get("exact_match")][:8]
    examples = []
    for row in rows:
        generated = str(row.get("generated_response", "")).strip()
        ground_truth = str(row.get("ground_truth", "")).strip()
        first_line = next((line.strip() for line in generated.splitlines() if line.strip()), "")
        if row.get("startswith_match") or generated.startswith(ground_truth):
            starts += 1
        if first_line == ground_truth:
            first_line_matches += 1
        if ground_truth and ground_truth in generated:
            contains_matches += 1
    for row in mismatches:
        examples.append(
            {
                "index": int(row.get("index", -1)),
                "prompt": clip(row.get("prompt"), 320),
                "generated_response": clip(row.get("generated_response"), 240),
                "ground_truth": clip(row.get("ground_truth"), 240),
            }
        )

    out = {
        "eval_status": "evaluated",
        "eval_count": len(rows),
        "eval_matches": matches,
        "eval_exact_match": matches / len(rows),
        "eval_startswith_matches": starts,
        "eval_startswith": starts / len(rows),
        "eval_first_line_matches": first_line_matches,
        "eval_first_line_exact": first_line_matches / len(rows),
        "eval_contains_matches": contains_matches,
        "eval_contains_ground_truth": contains_matches / len(rows),
        "eval_mean_generated_chars": sum(generated_lengths) / max(1, len(generated_lengths)),
        "sample_mismatches": examples,
    }
    return out


def build_records(results_root: Path, variants: list[dict[str, str]]) -> list[dict[str, Any]]:
    records = []
    for meta in variants:
        record = {
            "key": meta["key"],
            "variant": meta["variant"],
            "description": meta["description"],
            "model_dir": str(results_root / meta["model_dir"]),
            "eval_file": str(results_root / "eval" / f"{meta['key']}.jsonl"),
        }
        record.update(trainer_summary(results_root / meta["model_dir"]))
        record.update(eval_summary(results_root / "eval" / f"{meta['key']}.jsonl"))
        records.append(record)
    return records


def delta(records: list[dict[str, Any]], left: str, right: str, metric: str) -> float:
    by_variant = {row["variant"]: row for row in records}
    if left not in by_variant or right not in by_variant:
        return math.nan
    l_val = by_variant[left].get(metric)
    r_val = by_variant[right].get(metric)
    if not finite(l_val) or not finite(r_val):
        return math.nan
    return float(l_val) - float(r_val)


def build_summary(
    results_root: Path,
    train_file: Path,
    test_file: Path,
    variants: list[dict[str, str]],
    *,
    experiment: str,
    base_model: str,
    batch_size: int,
    learning_rate: float,
) -> dict[str, Any]:
    records = build_records(results_root, variants)
    return {
        "config": {
            "experiment": experiment,
            "task": "Natural language to regular expression generation",
            "base_model": base_model,
            "train_file": str(train_file),
            "test_file": str(test_file),
            "train_examples": count_lines(train_file),
            "test_examples": count_lines(test_file),
            "epochs": 4,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "eval_metric": "exact-match string equality on synth_test.jsonl",
            "secondary_eval_metrics": [
                "first non-empty generated line exactly equals target",
                "generated text starts with target",
                "generated text contains target",
            ],
            "variants": [row["variant"] for row in records],
        },
        "headline_contrasts": {
            "D1_minus_D0_eval_exact_match": delta(records, "D1", "D0", "eval_exact_match"),
            "D1_minus_Regular_eval_exact_match": delta(records, "D1", "Regular", "eval_exact_match"),
            "C_minus_Regular_eval_exact_match": delta(records, "C", "Regular", "eval_exact_match"),
            "C_ema_minus_C_eval_exact_match": delta(records, "C_ema", "C", "eval_exact_match"),
            "Both_minus_D1_eval_exact_match": delta(records, "Both", "D1", "eval_exact_match"),
            "Cross_minus_D1_eval_exact_match": delta(records, "Cross", "D1", "eval_exact_match"),
        },
        "records": records,
    }


def bar_chart(records: list[dict[str, Any]], metric: str, title: str, kind: str = "number") -> str:
    values = [float(row[metric]) for row in records if finite(row.get(metric))]
    if not values:
        return ""
    min_value = min(0.0, min(values))
    max_value = max(values)
    span = max(max_value - min_value, 1e-9)
    rows = []
    for index, row in enumerate(records):
        if not finite(row.get(metric)):
            continue
        value = float(row[metric])
        width = max(2.0, (value - min_value) / span * 100)
        rows.append(
            "<div class='bar-row'>"
            f"<div class='bar-name'>{html.escape(row['variant'])}</div>"
            f"<div class='bar-track'><span style='width:{width:.2f}%;background:{PALETTE[index % len(PALETTE)]}'></span></div>"
            f"<div class='bar-value'>{html.escape(fmt(value, kind))}</div>"
            "</div>"
        )
    return (
        "<article class='card'>"
        f"<h3>{html.escape(title)}</h3>"
        f"<div class='bar-list'>{''.join(rows)}</div>"
        "</article>"
    )


def loss_chart(records: list[dict[str, Any]]) -> str:
    histories = [row for row in records if row.get("history")]
    if not histories:
        return ""
    width, height = 920, 320
    left, top, right, bottom = 54, 24, 18, 44
    plot_w = width - left - right
    plot_h = height - top - bottom
    all_points = [point for row in histories for point in row["history"]]
    min_step = min(float(point["step"]) for point in all_points)
    max_step = max(float(point["step"]) for point in all_points)
    min_loss = min(float(point["loss"]) for point in all_points)
    max_loss = max(float(point["loss"]) for point in all_points)
    step_span = max(max_step - min_step, 1e-9)
    loss_span = max(max_loss - min_loss, 1e-9)
    polylines = []
    labels = []
    for index, row in enumerate(histories):
        color = PALETTE[index % len(PALETTE)]
        points = []
        for point in row["history"]:
            x = left + (float(point["step"]) - min_step) / step_span * plot_w
            y = top + plot_h - (float(point["loss"]) - min_loss) / loss_span * plot_h
            points.append(f"{x:.1f},{y:.1f}")
        polylines.append(
            f"<polyline points='{' '.join(points)}' fill='none' stroke='{color}' stroke-width='2.2' />"
        )
        labels.append(
            f"<span><i style='background:{color}'></i>{html.escape(row['variant'])}</span>"
        )
    return f"""
    <article class="card wide">
      <h3>Training Loss Traces</h3>
      <svg viewBox="0 0 {width} {height}" role="img" aria-label="training loss traces">
        <rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="#fbfcf8" stroke="#dfe5dc" />
        <line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#738078" />
        <line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#738078" />
        <text x="{left}" y="16" class="axis">loss {html.escape(fmt(min_loss))}-{html.escape(fmt(max_loss))}</text>
        <text x="{left + plot_w - 68}" y="{height - 12}" class="axis">step {html.escape(fmt(max_step))}</text>
        {''.join(polylines)}
      </svg>
      <div class="legend">{''.join(labels)}</div>
    </article>
    """


def metric_table(records: list[dict[str, Any]]) -> str:
    rows = []
    for row in records:
        rows.append(
            "<tr>"
            f"<td><code>{html.escape(row['variant'])}</code></td>"
            f"<td>{html.escape(row.get('description', ''))}</td>"
            f"<td class='num'>{html.escape(fmt(row.get('eval_exact_match'), 'percent'))}</td>"
            f"<td class='num'>{html.escape(fmt(row.get('eval_first_line_exact'), 'percent'))}</td>"
            f"<td class='num'>{html.escape(fmt(row.get('eval_contains_ground_truth'), 'percent'))}</td>"
            f"<td class='num'>{int(row.get('eval_matches', 0))}/{int(row.get('eval_count', 0))}</td>"
            f"<td class='num'>{html.escape(fmt(row.get('train_loss')))}</td>"
            f"<td class='num'>{html.escape(fmt(row.get('final_log_loss')))}</td>"
            f"<td class='num'>{html.escape(fmt(row.get('train_runtime')))} s</td>"
            f"<td class='num'>{html.escape(fmt(row.get('train_samples_per_second')))}</td>"
            f"<td>{html.escape(row.get('status', 'missing'))}/{html.escape(row.get('eval_status', 'missing'))}</td>"
            "</tr>"
        )
    return f"""
    <article class="card wide">
      <h3>Variant Metrics</h3>
      <div class="table-wrap">
        <table>
          <thead>
            <tr><th>Variant</th><th>Meaning</th><th>Exact Match</th><th>First-Line Exact</th><th>Contains Target</th><th>Matches</th><th>Train Loss</th><th>Final Log Loss</th><th>Runtime</th><th>Samples/s</th><th>Status</th></tr>
          </thead>
          <tbody>{''.join(rows)}</tbody>
        </table>
      </div>
    </article>
    """


def contrast_table(summary: dict[str, Any]) -> str:
    rows = []
    labels = {
        "D1_minus_D0_eval_exact_match": "Noise benefit: D1 - D0",
        "D1_minus_Regular_eval_exact_match": "JEPA over regular: D1 - Regular",
        "C_minus_Regular_eval_exact_match": "Co-trained cap over regular: C - Regular",
        "C_ema_minus_C_eval_exact_match": "EMA target over co-trained: C_ema - C",
        "Both_minus_D1_eval_exact_match": "Both caps over own-view D1: Both - D1",
        "Cross_minus_D1_eval_exact_match": "Cross-view cap over own-view D1: Cross - D1",
    }
    for key, value in summary.get("headline_contrasts", {}).items():
        cls = "pos" if finite(value) and float(value) >= 0 else "neg"
        rows.append(
            "<tr>"
            f"<td>{html.escape(labels.get(key, key))}</td>"
            f"<td class='num {cls}'>{html.escape(fmt(value, 'pp'))}</td>"
            "</tr>"
        )
    return f"""
    <article class="card">
      <h3>Exact-Match Contrasts</h3>
      <div class="table-wrap small">
        <table>
          <thead><tr><th>Question</th><th>Delta</th></tr></thead>
          <tbody>{''.join(rows)}</tbody>
        </table>
      </div>
    </article>
    """


def sample_tables(records: list[dict[str, Any]]) -> str:
    sections = []
    for row in records:
        samples = row.get("sample_mismatches") or []
        if not samples:
            continue
        rows = []
        for sample in samples:
            rows.append(
                "<tr>"
                f"<td class='num'>{int(sample.get('index', -1))}</td>"
                f"<td>{html.escape(sample.get('generated_response', ''))}</td>"
                f"<td>{html.escape(sample.get('ground_truth', ''))}</td>"
                "</tr>"
            )
        sections.append(
            f"""
            <details>
              <summary>{html.escape(row['variant'])} mismatch samples</summary>
              <div class="table-wrap">
                <table>
                  <thead><tr><th>Index</th><th>Generated</th><th>Ground truth</th></tr></thead>
                  <tbody>{''.join(rows)}</tbody>
                </table>
              </div>
            </details>
            """
        )
    if not sections:
        return ""
    return f"""
    <article class="card wide">
      <h3>Inspectable Mismatch Samples</h3>
      <p class="muted">These are short snippets from the evaluation JSONL, included so the HTML page shows what failures look like without committing full model outputs.</p>
      {''.join(sections)}
    </article>
    """


def stat_cards(summary: dict[str, Any]) -> str:
    records = summary["records"]
    evaluated = [row for row in records if finite(row.get("eval_exact_match"))]
    best = max(evaluated, key=lambda row: float(row["eval_exact_match"])) if evaluated else None
    d1_d0 = summary["headline_contrasts"].get("D1_minus_D0_eval_exact_match")
    complete = sum(1 for row in records if row.get("status") == "trained")
    eval_complete = sum(1 for row in records if row.get("eval_status") == "evaluated")
    cards = [
        ("Training variants", f"{complete}/{len(records)}", "Checkpoints with trainer_state.json."),
        ("Evaluated variants", f"{eval_complete}/{len(records)}", "Exact-match evaluation on synth_test.jsonl."),
        ("Best exact match", f"{best['variant']} {fmt(best['eval_exact_match'], 'percent')}" if best else "n/a", "Highest downstream string exact-match score."),
        ("D1 - D0", fmt(d1_d0, "pp"), "Frozen noisy own-view anchor minus frozen sigma=0 control."),
        ("Train examples", str(summary["config"].get("train_examples", 0)), "Full synthetic NL-to-regex training split."),
        ("Test examples", str(summary["config"].get("test_examples", 0)), "Held-out synthetic NL-to-regex evaluation split."),
    ]
    return "".join(
        "<article class='stat'>"
        f"<div class='label'>{html.escape(label)}</div>"
        f"<div class='value'>{html.escape(value)}</div>"
        f"<div class='note'>{html.escape(note)}</div>"
        "</article>"
        for label, value, note in cards
    )


def build_html(summary: dict[str, Any]) -> str:
    records = summary["records"]
    data = json.dumps(summary, indent=2, ensure_ascii=True)
    experiment = summary["config"].get("experiment", "exp3")
    base_model = summary["config"].get("base_model", "model")
    title = f"{experiment} LLM-JEPA Report"
    charts = [
        bar_chart(records, "eval_exact_match", "Downstream Exact Match", "percent"),
        bar_chart(records, "eval_first_line_exact", "First-Line Exact Match", "percent"),
        bar_chart(records, "eval_contains_ground_truth", "Contains Ground Truth", "percent"),
        bar_chart(records, "train_loss", "Reported Train Loss"),
        bar_chart(records, "final_log_loss", "Final Logged LM Loss"),
        bar_chart(records, "train_runtime", "Train Runtime"),
        contrast_table(summary),
        loss_chart(records),
        metric_table(records),
        sample_tables(records),
    ]
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
    header {{
      position: sticky;
      top: 0;
      z-index: 4;
      border-bottom: 1px solid var(--line);
      background: rgba(246, 247, 242, 0.94);
      backdrop-filter: blur(12px);
    }}
    .head-inner, main {{ max-width: 1480px; margin: 0 auto; padding: 16px 18px; }}
    h1 {{ margin: 0; font-size: 22px; line-height: 1.2; }}
    h2 {{ margin: 22px 0 10px; font-size: 18px; }}
    h3 {{ margin: 0 0 10px; font-size: 15px; }}
    p {{ color: var(--muted); line-height: 1.45; }}
    .stats, .grid {{ display: grid; gap: 12px; }}
    .stats {{ grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); }}
    .grid {{ grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); }}
    .stat, .card {{
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      box-shadow: var(--shadow);
      padding: 14px;
      min-width: 0;
    }}
    .stat {{ background: var(--soft); min-height: 96px; }}
    .label {{ color: var(--muted); font-size: 12px; text-transform: uppercase; }}
    .value {{ margin-top: 7px; font-size: 25px; font-weight: 780; line-height: 1; }}
    .note, .muted {{ color: var(--muted); font-size: 12px; }}
    .wide {{ grid-column: 1 / -1; }}
    .bar-list {{ display: grid; gap: 8px; }}
    .bar-row {{ display: grid; grid-template-columns: 86px 1fr 72px; gap: 8px; align-items: center; font-size: 12px; }}
    .bar-name {{ overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
    .bar-track {{ height: 12px; border: 1px solid #d9e1d6; border-radius: 999px; background: #edf1eb; overflow: hidden; }}
    .bar-track span {{ display: block; height: 100%; border-radius: 999px; }}
    .bar-value, .num {{ text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }}
    .table-wrap {{ overflow: auto; border: 1px solid var(--line); border-radius: 8px; }}
    table {{ width: 100%; min-width: 820px; border-collapse: collapse; background: var(--panel); font-size: 13px; }}
    .small table {{ min-width: 460px; }}
    th, td {{ padding: 8px 9px; border-bottom: 1px solid var(--line); text-align: left; vertical-align: top; }}
    th {{ background: #f2f5ef; color: #405047; font-size: 11px; text-transform: uppercase; }}
    code {{ font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace; }}
    details {{ margin-top: 10px; border: 1px solid var(--line); border-radius: 8px; background: var(--soft); }}
    summary {{ cursor: pointer; padding: 10px 12px; font-weight: 710; }}
    details .table-wrap {{ margin: 0 12px 12px; }}
    .pos {{ color: var(--green); font-weight: 700; }}
    .neg {{ color: var(--red); font-weight: 700; }}
    svg {{ display: block; width: 100%; height: auto; }}
    .axis {{ fill: var(--muted); font: 12px system-ui, sans-serif; }}
    .legend {{ display: flex; flex-wrap: wrap; gap: 8px; margin-top: 8px; color: var(--muted); font-size: 12px; }}
    .legend span {{ display: inline-flex; gap: 5px; align-items: center; }}
    .legend i {{ width: 9px; height: 9px; border-radius: 99px; display: inline-block; }}
    pre {{ margin: 0; white-space: pre-wrap; word-break: break-word; font-size: 12px; line-height: 1.45; }}
  </style>
</head>
<body>
  <header>
    <div class="head-inner">
      <h1>{html.escape(title)}</h1>
      <p>{html.escape(base_model)} synthetic NL-to-regex sweep. Strict exact match is the primary downstream metric; secondary format diagnostics are included because generations may contain explanations or malformed regex strings.</p>
    </div>
  </header>
  <main>
    <section class="stats">{stat_cards(summary)}</section>
    <h2>Plots</h2>
    <section class="grid">{''.join(chart for chart in charts if chart)}</section>
    <h2>Embedded Summary JSON</h2>
    <article class="card wide"><pre>{html.escape(data)}</pre></article>
  </main>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the Exp3 full LLM-JEPA report from trainer states and eval JSONL files.")
    parser.add_argument("--results-root", type=Path, default=Path("results/exp3_full_synth"))
    parser.add_argument("--train-file", type=Path, default=Path("datasets/synth_train.jsonl"))
    parser.add_argument("--test-file", type=Path, default=Path("datasets/synth_test.jsonl"))
    parser.add_argument("--report-dir", type=Path, default=Path("reports/gpu_2026-05-09"))
    parser.add_argument("--variant-set", choices=sorted(VARIANT_SETS), default="smollm_full")
    parser.add_argument("--experiment", default="exp3_full_synth")
    parser.add_argument("--base-model", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--output-stem", default=None)
    args = parser.parse_args()

    summary = build_summary(
        args.results_root,
        args.train_file,
        args.test_file,
        VARIANT_SETS[args.variant_set],
        experiment=args.experiment,
        base_model=args.base_model,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )
    stem = args.output_stem or args.experiment
    json_path = args.report_dir / "json" / f"{stem}_summary.json"
    html_path = args.report_dir / "html" / f"{stem}.html"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n")
    html_path.write_text(build_html(summary))
    print(f"wrote {json_path}")
    print(f"wrote {html_path}")


if __name__ == "__main__":
    main()
