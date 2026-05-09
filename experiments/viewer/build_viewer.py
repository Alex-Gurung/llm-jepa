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
    return scatter_plot_for_key(record, "scatter", record["variant"])


def scatter_plot_for_key(record: dict[str, Any], key: str, title: str) -> str:
    points = record.get(key) or []
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
      <h3>{html.escape(title)}</h3>
      <svg viewBox="0 0 220 220" role="img" aria-label="latent scatter for {html.escape(title)}">
        <rect x="12" y="12" width="196" height="196" fill="#fbfbf9" stroke="#ddd"/>
        {''.join(circles)}
      </svg>
    </article>
    """


def scatter_section(records: list[dict[str, Any]], key: str, title: str) -> str:
    cards = []
    for record in records[:12]:
        label = record["variant"] if key == "scatter" else f"{record['variant']} {title.split()[0]}"
        cards.append(scatter_plot_for_key(record, key, label))
    cards_html = "".join(cards)
    if not cards_html:
        return ""
    return f"""
    <section>
      <h2>{html.escape(title)}</h2>
      <div class="grid">{cards_html}</div>
    </section>
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
        ("D1", "D0", "noise: frozen teacher anchor"),
        ("D1", "C", "anchor: D1 vs co-trained"),
        ("D1", "C_ema", "anchor: D1 vs EMA"),
        ("D1", "D2", "sample-specific vs class-level"),
        ("D_own_1", "D_own_0", "noise: own-view frozen anchor"),
        ("D_own_1", "C", "anchor: own-view vs co-trained"),
        ("D_own_1", "C_ema", "anchor: own-view vs EMA"),
        ("D_cross_1", "D_cross_0", "noise: cross-view frozen anchor"),
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


def retrieval_table(records: list[dict[str, Any]]) -> str:
    rows = []
    for record in records:
        for key, value in record.items():
            if not key.startswith("retrieval") or not isinstance(value, dict):
                continue
            rows.append(
                "<tr>"
                f"<td>{html.escape(record['variant'])}</td>"
                f"<td>{html.escape(key)}</td>"
                f"<td>{fmt(float(value.get('recall@1', float('nan'))))}</td>"
                f"<td>{fmt(float(value.get('recall@10', float('nan'))))}</td>"
                f"<td>{fmt(float(value.get('mean_rank', float('nan'))))}</td>"
                "</tr>"
            )
    if not rows:
        return ""
    return """
    <section>
      <h2>Retrieval</h2>
      <table>
        <thead><tr><th>Variant</th><th>Space</th><th>R@1</th><th>R@10</th><th>Mean Rank</th></tr></thead>
        <tbody>
    """ + "".join(rows) + """
        </tbody>
      </table>
    </section>
    """


def histogram_card(title: str, hist: dict[str, Any]) -> str:
    counts = hist.get("counts") or []
    if not counts:
        return ""
    max_count = max(max(counts), 1)
    width = 260
    height = 145
    left = 28
    bottom = 24
    chart_w = width - left - 10
    chart_h = height - 34 - bottom
    bar_w = chart_w / len(counts)
    bars = []
    for i, count in enumerate(counts):
        h = chart_h * count / max_count
        x = left + i * bar_w
        y = 28 + chart_h - h
        bars.append(f"<rect x='{x:.2f}' y='{y:.2f}' width='{max(1.0, bar_w - 1):.2f}' height='{h:.2f}' fill='#2f6f9f'/>")
    return f"""
    <article class="viz-card">
      <h3>{html.escape(title)}</h3>
      <svg viewBox="0 0 {width} {height}">
        <text x="{left}" y="16" class="axis">cosine histogram</text>
        <line x1="{left}" y1="{28 + chart_h}" x2="{width - 8}" y2="{28 + chart_h}" stroke="#999"/>
        <text x="{left}" y="{height - 5}" class="axis">-1</text>
        <text x="{width - 28}" y="{height - 5}" class="axis">1</text>
        {''.join(bars)}
      </svg>
    </article>
    """


def spectrum_card(title: str, spectrum: dict[str, Any]) -> str:
    mass = spectrum.get("mass") or []
    if not mass:
        return ""
    width = 260
    height = 145
    left = 30
    top = 22
    chart_w = width - left - 12
    chart_h = height - top - 26
    max_v = max(max(mass), 1e-9)
    points = []
    for i, value in enumerate(mass):
        x = left + (chart_w * i / max(1, len(mass) - 1))
        y = top + chart_h - chart_h * value / max_v
        points.append(f"{x:.2f},{y:.2f}")
    circles = []
    for point in points:
        x, y = point.split(",")
        circles.append(f"<circle cx='{x}' cy='{y}' r='1.8' fill='#d1495b'/>")
    return f"""
    <article class="viz-card">
      <h3>{html.escape(title)}</h3>
      <svg viewBox="0 0 {width} {height}">
        <text x="{left}" y="16" class="axis">eigenvalue mass</text>
        <line x1="{left}" y1="{top + chart_h}" x2="{width - 8}" y2="{top + chart_h}" stroke="#999"/>
        <polyline points="{' '.join(points)}" fill="none" stroke="#d1495b" stroke-width="2"/>
        {''.join(circles)}
      </svg>
    </article>
    """


def diagnostics_section(records: list[dict[str, Any]], key: str, title: str, renderer) -> str:
    cards = []
    for record in records[:12]:
        value = record.get(key)
        if value:
            cards.append(renderer(record["variant"], value))
    cards_html = "".join(cards)
    if not cards_html:
        return ""
    return f"""
    <section>
      <h2>{html.escape(title)}</h2>
      <div class="viz-grid">{cards_html}</div>
    </section>
    """


def prefixed_diagnostics_section(records: list[dict[str, Any]], prefix: str, title: str, renderer) -> str:
    cards = []
    for record in records[:12]:
        value = record.get(f"{prefix}_{renderer.__name__.replace('_card', '')}")
        if value:
            cards.append(renderer(f"{record['variant']} {prefix}", value))
    cards_html = "".join(cards)
    if not cards_html:
        return ""
    return f"""
    <section>
      <h2>{html.escape(title)}</h2>
      <div class="viz-grid">{cards_html}</div>
    </section>
    """


def heatmap_card(title: str, heatmap: dict[str, Any]) -> str:
    values = heatmap.get("values") or []
    if not values:
        return ""
    n = len(values)
    cell = max(2.0, 180.0 / n)
    size = cell * n
    rects = []
    for i, row in enumerate(values):
        for j, value in enumerate(row):
            v = max(-1.0, min(1.0, float(value)))
            if v >= 0:
                intensity = int(255 - 115 * v)
                color = f"rgb({intensity},{intensity},{255})"
            else:
                intensity = int(255 + 115 * v)
                color = f"rgb(255,{intensity},{intensity})"
            stroke = "#111" if i == j else color
            rects.append(
                f"<rect x='{j * cell:.2f}' y='{i * cell:.2f}' width='{cell:.2f}' height='{cell:.2f}' fill='{color}' stroke='{stroke}' stroke-width='{0.45 if i == j else 0}'/>"
            )
    return f"""
    <article class="viz-card">
      <h3>{html.escape(title)}</h3>
      <svg viewBox="0 0 {size:.1f} {size:.1f}" class="heatmap">
        {''.join(rects)}
      </svg>
    </article>
    """


def heatmap_section(records: list[dict[str, Any]]) -> str:
    keys = [
        "embedding_similarity_heatmap",
        "text_cap_to_text_anchor_heatmap",
        "code_cap_to_code_anchor_heatmap",
        "text_cap_to_code_anchor_heatmap",
    ]
    cards = []
    for record in records[:8]:
        for key in keys:
            if key in record:
                cards.append(heatmap_card(f"{record['variant']} {key}", record[key]))
    cards_html = "".join(cards)
    if not cards_html:
        return ""
    return f"""
    <section>
      <h2>Retrieval Similarity Heatmaps</h2>
      <div class="viz-grid">{cards_html}</div>
    </section>
    """


def anchor_visuals(payload: dict[str, Any]) -> str:
    diagnostics = payload.get("anchor_geometry") or {}
    if not diagnostics:
        return ""
    hist_cards = []
    spectrum_cards = []
    for name, diag in diagnostics.items():
        if diag.get("cosine_histogram"):
            hist_cards.append(histogram_card(str(name), diag["cosine_histogram"]))
        if diag.get("eigen_spectrum"):
            spectrum_cards.append(spectrum_card(str(name), diag["eigen_spectrum"]))
    sections = []
    if hist_cards:
        sections.append(f"<section><h2>Anchor Cosine Histograms</h2><div class='viz-grid'>{''.join(hist_cards)}</div></section>")
    if spectrum_cards:
        sections.append(f"<section><h2>Anchor Eigenspectra</h2><div class='viz-grid'>{''.join(spectrum_cards)}</div></section>")
    return "".join(sections)


def build_html(payload: dict[str, Any], source: Path) -> str:
    records = payload.get("records", [])
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
    .viz-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 14px; }}
    .scatter-card {{ background: #fffefa; border: 1px solid #ddd7ca; border-radius: 8px; padding: 12px; }}
    .viz-card {{ background: #fffefa; border: 1px solid #ddd7ca; border-radius: 8px; padding: 12px; }}
    .heatmap {{ max-height: 220px; image-rendering: pixelated; }}
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
    {anchor_visuals(payload)}
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
    {retrieval_table(records)}
    {scatter_section(records, "scatter", "PCA Scatter")}
    {scatter_section(records, "text_pca_scatter", "Text PCA Scatter")}
    {scatter_section(records, "code_pca_scatter", "Code PCA Scatter")}
    {diagnostics_section(records, "cosine_histogram", "Cosine Histograms", histogram_card)}
    {diagnostics_section(records, "text_cosine_histogram", "Text Cosine Histograms", histogram_card)}
    {diagnostics_section(records, "code_cosine_histogram", "Code Cosine Histograms", histogram_card)}
    {diagnostics_section(records, "eigen_spectrum", "Eigenspectra", spectrum_card)}
    {diagnostics_section(records, "text_eigen_spectrum", "Text Eigenspectra", spectrum_card)}
    {diagnostics_section(records, "code_eigen_spectrum", "Code Eigenspectra", spectrum_card)}
    {heatmap_section(records)}
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
