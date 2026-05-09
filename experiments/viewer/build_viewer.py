from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path
from typing import Any


PALETTE = [
    "#2f6f9f",
    "#d1495b",
    "#2d9d78",
    "#edae49",
    "#6c4f9f",
    "#00798c",
    "#c44536",
    "#5c8001",
]


def get_path(row: dict[str, Any], path: str) -> float:
    cur: Any = row
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return float("nan")
        cur = cur[part]
    value = float(cur)
    return value


def record_rankme(row: dict[str, Any]) -> float:
    return float(row.get("rankme", row.get("text_rankme", float("nan"))))


def record_mean_cosine(row: dict[str, Any]) -> float:
    if "cosine" in row:
        return float(row["cosine"]["mean"])
    if "text_cosine" in row:
        return float(row["text_cosine"]["mean"])
    return float("nan")


def fmt(value: float) -> str:
    if math.isnan(value):
        return "nan"
    if abs(value) >= 100:
        return f"{value:.1f}"
    if abs(value) >= 10:
        return f"{value:.2f}"
    return f"{value:.3f}"


def bar_chart(records: list[dict[str, Any]], metric: str, title: str, *, lower_is_better: bool = False) -> str:
    values = [get_path(record, metric) for record in records]
    finite = [v for v in values if math.isfinite(v)]
    if not finite:
        return ""
    min_v = min(0.0, min(finite))
    max_v = max(finite)
    span = max(max_v - min_v, 1e-9)
    width = 880
    height = 260
    left = 54
    bottom = 46
    top = 34
    chart_h = height - top - bottom
    step = (width - left - 20) / max(1, len(records))
    bar_w = min(44, step * 0.68)
    bars = []
    for idx, (record, value) in enumerate(zip(records, values)):
        x = left + idx * step + (step - bar_w) / 2
        h = 0.0 if not math.isfinite(value) else (value - min_v) / span * chart_h
        y = top + chart_h - h
        color = "#555" if lower_is_better else PALETTE[idx % len(PALETTE)]
        bars.append(
            f"<rect x='{x:.1f}' y='{y:.1f}' width='{bar_w:.1f}' height='{h:.1f}' fill='{color}' rx='3'>"
            f"<title>{html.escape(record['variant'])}: {fmt(value)}</title></rect>"
        )
        bars.append(
            f"<text x='{x + bar_w / 2:.1f}' y='{height - 18}' text-anchor='end' transform='rotate(-35 {x + bar_w / 2:.1f} {height - 18})'>{html.escape(record['variant'])}</text>"
        )
    axis = (
        f"<line x1='{left}' y1='{top + chart_h}' x2='{width - 12}' y2='{top + chart_h}' stroke='#888'/>"
        f"<text x='{left}' y='20' class='axis'>{fmt(max_v)}</text>"
        f"<text x='{left}' y='{top + chart_h - 4}' class='axis'>{fmt(min_v)}</text>"
    )
    return f"""
    <section>
      <h2>{html.escape(title)}</h2>
      <svg viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}">
        {axis}
        {''.join(bars)}
      </svg>
    </section>
    """


def scatter_plot(record: dict[str, Any]) -> str:
    points = record.get("scatter") or []
    if not points:
        return ""
    xs = [float(p["x"]) for p in points]
    ys = [float(p["y"]) for p in points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    span_x = max(max_x - min_x, 1e-9)
    span_y = max(max_y - min_y, 1e-9)
    circles = []
    for p in points:
        x = 20 + (float(p["x"]) - min_x) / span_x * 180
        y = 200 - (float(p["y"]) - min_y) / span_y * 180
        label = int(p.get("label", 0))
        circles.append(
            f"<circle cx='{x:.1f}' cy='{y:.1f}' r='2.0' fill='{PALETTE[label % len(PALETTE)]}' opacity='0.72'/>"
        )
    return f"""
    <article class="scatter-card">
      <h3>{html.escape(record['variant'])}</h3>
      <svg viewBox="0 0 220 220" role="img" aria-label="latent scatter for {html.escape(record['variant'])}">
        <rect x="12" y="12" width="196" height="196" fill="#fbfbf9" stroke="#ddd"/>
        {''.join(circles)}
      </svg>
    </article>
    """


def contrast_rows(records: list[dict[str, Any]]) -> str:
    by_name = {r["variant"]: r for r in records}
    pairs = [
        ("D1a", "D0a", "noise: continuous anchor"),
        ("D1b", "D0b", "noise: RFF anchor"),
        ("D1c", "D0c", "noise: frozen teacher anchor"),
        ("D1a", "C", "anchor: D1a vs co-trained"),
        ("D1b", "C", "anchor: D1b vs co-trained"),
        ("D1c", "C", "anchor: D1c vs co-trained"),
        ("D1a", "D3", "sample-specific vs class-level"),
    ]
    rows = []
    for left, right, label in pairs:
        if left not in by_name or right not in by_name:
            continue
        l_row, r_row = by_name[left], by_name[right]
        delta_rank = record_rankme(l_row) - record_rankme(r_row)
        delta_cos = record_mean_cosine(l_row) - record_mean_cosine(r_row)
        rows.append(
            "<tr>"
            f"<td>{html.escape(label)}</td>"
            f"<td>{html.escape(left)} - {html.escape(right)}</td>"
            f"<td>{fmt(delta_rank)}</td>"
            f"<td>{fmt(delta_cos)}</td>"
            "</tr>"
        )
    if not rows:
        return ""
    return """
    <section>
      <h2>Headline Contrasts</h2>
      <table>
        <thead><tr><th>Question</th><th>Pair</th><th>Delta RankMe</th><th>Delta Mean Cosine</th></tr></thead>
        <tbody>
    """ + "".join(rows) + """
        </tbody>
      </table>
    </section>
    """


def anchor_table(payload: dict[str, Any]) -> str:
    diagnostics = payload.get("anchor_geometry") or {}
    if not diagnostics:
        return ""
    rows = []
    for name, diag in diagnostics.items():
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(name))}</td>"
            f"<td>{fmt(float(diag['rankme']))}</td>"
            f"<td>{fmt(float(diag['top_eig_mass_ratio']))}</td>"
            f"<td>{fmt(float(diag['mean_norm']))}</td>"
            f"<td>{fmt(float(diag['uniformity_after_normalize']))}</td>"
            f"<td>{fmt(float(diag['cosine_stats']['mean']))}</td>"
            "</tr>"
        )
    return """
    <section>
      <h2>Anchor Geometry</h2>
      <table>
        <thead><tr><th>Anchor</th><th>RankMe</th><th>Top Eig Mass</th><th>Mean Norm</th><th>Uniformity</th><th>Mean Cos</th></tr></thead>
        <tbody>
    """ + "".join(rows) + """
        </tbody>
      </table>
    </section>
    """


def metrics_table(records: list[dict[str, Any]]) -> str:
    rows = []
    for r in records:
        within = r.get("within_class_rankme", {}).get("_mean_within_class", float("nan"))
        rank_value = record_rankme(r)
        uniformity = float(r.get("uniformity", r.get("text_uniformity", float("nan"))))
        cosine = r.get("cosine", r.get("text_cosine", {}))
        rows.append(
            "<tr>"
            f"<td>{html.escape(r['variant'])}</td>"
            f"<td>{fmt(rank_value)}</td>"
            f"<td>{fmt(float(within))}</td>"
            f"<td>{fmt(uniformity)}</td>"
            f"<td>{fmt(float(cosine.get('mean', float('nan'))))}</td>"
            f"<td>{fmt(float(cosine.get('p95', float('nan'))))}</td>"
            f"<td>{fmt(float(r.get('alignment_view_ab', float('nan'))))}</td>"
            "</tr>"
        )
    return """
    <section>
      <h2>Metrics</h2>
      <table>
        <thead><tr><th>Variant</th><th>RankMe</th><th>Within-Class RankMe</th><th>Uniformity</th><th>Mean Cos</th><th>P95 Cos</th><th>View Alignment</th></tr></thead>
        <tbody>
    """ + "".join(rows) + """
        </tbody>
      </table>
    </section>
    """


def build_html(payload: dict[str, Any], source: Path) -> str:
    records = payload.get("records", [])
    scatter = "".join(scatter_plot(record) for record in records[:12])
    config = payload.get("config", {})
    title = source.stem
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)} sphere-JEPA results</title>
  <style>
    :root {{ color-scheme: light; font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    body {{ margin: 0; background: #f4f1ea; color: #202426; }}
    header {{ background: #244a5a; color: white; padding: 28px 40px 22px; }}
    header h1 {{ margin: 0 0 8px; font-size: 28px; font-weight: 720; }}
    header p {{ margin: 0; color: #d7e6ea; }}
    main {{ max-width: 1120px; margin: 0 auto; padding: 28px 24px 48px; }}
    section {{ background: #fffefa; border: 1px solid #ddd7ca; border-radius: 8px; padding: 18px; margin-bottom: 18px; box-shadow: 0 1px 3px rgba(0,0,0,0.04); }}
    h2 {{ font-size: 18px; margin: 0 0 14px; }}
    h3 {{ font-size: 14px; margin: 0 0 8px; }}
    svg {{ width: 100%; height: auto; }}
    text {{ font-size: 11px; fill: #303538; }}
    text.axis {{ fill: #697073; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
    th, td {{ border-bottom: 1px solid #e6e0d2; padding: 8px 9px; text-align: left; }}
    th {{ color: #415058; font-size: 12px; text-transform: uppercase; letter-spacing: 0; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 14px; }}
    .scatter-card {{ background: #fffefa; border: 1px solid #ddd7ca; border-radius: 8px; padding: 12px; }}
    code {{ background: #ebe5d8; padding: 2px 5px; border-radius: 4px; }}
  </style>
</head>
<body>
  <header>
    <h1>Sphere-JEPA Results</h1>
    <p>Source: <code>{html.escape(str(source))}</code></p>
  </header>
  <main>
    <section>
      <h2>Run Config</h2>
      <pre>{html.escape(json.dumps(config, indent=2))}</pre>
    </section>
    {anchor_table(payload)}
    {contrast_rows(records)}
    {bar_chart(records, "rankme", "RankMe")}
    {bar_chart(records, "text_rankme", "Text RankMe")}
    {bar_chart(records, "code_rankme", "Code RankMe")}
    {bar_chart(records, "cosine.mean", "Mean Off-Diagonal Cosine", lower_is_better=True)}
    {bar_chart(records, "text_cosine.mean", "Text Mean Off-Diagonal Cosine", lower_is_better=True)}
    {bar_chart(records, "code_cosine.mean", "Code Mean Off-Diagonal Cosine", lower_is_better=True)}
    {bar_chart(records, "uniformity", "Uniformity", lower_is_better=True)}
    {bar_chart(records, "text_uniformity", "Text Uniformity", lower_is_better=True)}
    {bar_chart(records, "code_uniformity", "Code Uniformity", lower_is_better=True)}
    {metrics_table(records)}
    <section>
      <h2>Latent Scatter</h2>
      <div class="grid">{scatter}</div>
    </section>
  </main>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a self-contained HTML viewer for sphere-JEPA result JSON.")
    parser.add_argument("summary", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    payload = json.loads(args.summary.read_text())
    out = args.out or args.summary.with_suffix(".html")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build_html(payload, args.summary))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
