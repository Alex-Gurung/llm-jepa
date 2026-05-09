from __future__ import annotations

import argparse
import html
import json
import math
import re
from pathlib import Path
from typing import Any


EXPERIMENT_META = {
    "exp0_full": {
        "title": "Exp0 Toy Geometry",
        "group": "Toy",
        "summary": "Synthetic paired-view run with continuous, RFF, and frozen-teacher sample-specific anchors.",
        "takeaways": [
            "Co-trained targets collapse in C: RankMe is low and off-diagonal cosine is near 1.",
            "Frozen or external anchors restore spread; D1a and D1c improve over matched sigma=0 controls.",
            "Within-class RankMe is useful here because global spread can hide class-level shortcuts.",
        ],
    },
    "exp1_cifar_gpu": {
        "title": "Exp1 CIFAR-10 Bridge",
        "group": "Vision bridge",
        "summary": "GPU sanity run on a 10k/2k CIFAR-10 subset with a frozen autoencoder teacher anchor.",
        "takeaways": [
            "D1 improves representation spread over D0, but the linear probe is lower in this pass.",
            "D2 uses class labels and should be treated as a supervised cardinality control.",
            "This validates the bridge machinery more than it establishes a final task-accuracy claim.",
        ],
    },
    "exp2_pythia160m_synth_sphere": {
        "title": "Exp2 Frozen LLM: Spherical Anchors",
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
        "group": "Frozen LLM",
        "summary": "D_own anchor preprocessing ablation with raw Pythia hidden states.",
        "takeaways": [
            "Raw anchors expose the source anisotropy problem directly.",
            "The D1-vs-D0 contrast is present, but this is not the strongest preprocessing choice.",
        ],
    },
    "exp2_pythia160m_synth_norm": {
        "title": "Exp2 Frozen LLM: Norm Anchors",
        "group": "Frozen LLM",
        "summary": "D_own anchor preprocessing ablation with normalized Pythia hidden states.",
        "takeaways": [
            "Norm-only preprocessing changes scale without fixing most anisotropy.",
            "Compare it against raw, sphere, and white before drawing mechanism conclusions.",
        ],
    },
    "exp2_pythia160m_synth_white": {
        "title": "Exp2 Frozen LLM: White Anchors",
        "group": "Frozen LLM",
        "summary": "D_own anchor preprocessing ablation with whitened anchors and raw source states.",
        "takeaways": [
            "Whitened anchors are much more isotropic than raw anchors.",
            "Source-side anisotropy still limits the projection-head run unless inputs are whitened too.",
        ],
    },
    "exp2_pythia160m_synth_input_white_anchor_white": {
        "title": "Exp2 Frozen LLM: Source + Anchor Whitening",
        "group": "Frozen LLM",
        "summary": "The useful Pythia-160M diagnostic: source states and anchors are both whitened.",
        "takeaways": [
            "D_own_1 beats D_own_0 on RankMe, mean cosine, and predicted-code retrieval.",
            "D_own_1 is far stronger than co-trained C, so external anchoring matters.",
            "InfoNCE and VICReg still dominate direct embedding retrieval, so retrieval-space choice matters.",
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
]


HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Sphere-JEPA Results Dashboard</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f6f7f2;
      --panel: #ffffff;
      --panel-soft: #fbfcf8;
      --ink: #1e2723;
      --muted: #65716c;
      --line: #dfe5dc;
      --line-strong: #c8d2c8;
      --blue: #246a8f;
      --green: #1f7a63;
      --red: #b94e4e;
      --amber: #bd812d;
      --purple: #6654a2;
      --teal: #188092;
      --shadow: 0 8px 24px rgba(31, 39, 35, 0.08);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--ink);
    }
    a { color: var(--blue); text-decoration: none; }
    a:hover { text-decoration: underline; }
    .topbar {
      position: sticky;
      top: 0;
      z-index: 20;
      background: rgba(246, 247, 242, 0.94);
      backdrop-filter: blur(12px);
      border-bottom: 1px solid var(--line);
    }
    .topbar-inner {
      max-width: 1520px;
      margin: 0 auto;
      padding: 12px 18px;
      display: grid;
      grid-template-columns: minmax(240px, 1fr) auto;
      gap: 18px;
      align-items: center;
    }
    .brand h1 {
      margin: 0;
      font-size: 19px;
      line-height: 1.2;
      font-weight: 760;
    }
    .brand p {
      margin: 3px 0 0;
      color: var(--muted);
      font-size: 13px;
    }
    .navlinks {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      justify-content: flex-end;
    }
    .navlinks a,
    .pill {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      min-height: 30px;
      padding: 5px 9px;
      border: 1px solid var(--line-strong);
      border-radius: 8px;
      background: var(--panel);
      color: var(--ink);
      font-size: 12px;
      line-height: 1.2;
      white-space: nowrap;
    }
    .shell {
      max-width: 1520px;
      margin: 0 auto;
      padding: 18px;
    }
    .section {
      margin: 0 0 18px;
    }
    .section-title {
      display: flex;
      justify-content: space-between;
      align-items: baseline;
      gap: 16px;
      margin: 0 0 10px;
    }
    .section-title h2 {
      margin: 0;
      font-size: 19px;
      font-weight: 760;
    }
    .section-title p {
      margin: 0;
      color: var(--muted);
      font-size: 13px;
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
    }
    .panel-pad { padding: 16px; }
    .grid {
      display: grid;
      gap: 12px;
    }
    .stats {
      grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
    }
    .stat {
      padding: 14px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel-soft);
      min-height: 98px;
    }
    .stat .label {
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0;
    }
    .stat .value {
      margin-top: 7px;
      font-size: 26px;
      line-height: 1;
      font-weight: 780;
    }
    .stat .note {
      margin-top: 8px;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.35;
    }
    .cards {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(290px, 1fr));
      gap: 12px;
    }
    .run-card {
      padding: 14px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
    }
    .run-card h3,
    .chart-card h3,
    .table-card h3 {
      margin: 0 0 8px;
      font-size: 15px;
      font-weight: 730;
    }
    .run-card p,
    .chart-card p {
      margin: 0;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.45;
    }
    .takeaways {
      margin: 10px 0 0;
      padding: 0;
      list-style: none;
      display: grid;
      gap: 6px;
    }
    .takeaways li {
      position: relative;
      padding-left: 14px;
      color: #3d4742;
      font-size: 13px;
      line-height: 1.35;
    }
    .takeaways li::before {
      content: "";
      position: absolute;
      left: 0;
      top: 0.62em;
      width: 6px;
      height: 6px;
      border-radius: 999px;
      background: var(--green);
    }
    .chips {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      margin-top: 10px;
    }
    .chip {
      display: inline-flex;
      align-items: center;
      min-height: 24px;
      padding: 3px 7px;
      border-radius: 7px;
      background: #eef3ed;
      border: 1px solid var(--line);
      color: #33413b;
      font-size: 12px;
      line-height: 1.2;
    }
    .run {
      margin-top: 18px;
      overflow: clip;
    }
    .run-head {
      display: grid;
      grid-template-columns: minmax(260px, 1fr) auto;
      gap: 16px;
      padding: 16px;
      border-bottom: 1px solid var(--line);
      background: linear-gradient(90deg, #ffffff, #f9fbf7);
    }
    .run-title h2 {
      margin: 0;
      font-size: 20px;
      font-weight: 780;
    }
    .run-title p {
      margin: 6px 0 0;
      color: var(--muted);
      line-height: 1.45;
      font-size: 13px;
      max-width: 860px;
    }
    .run-actions {
      display: flex;
      align-items: flex-start;
      justify-content: flex-end;
      gap: 8px;
      flex-wrap: wrap;
    }
    .run-body {
      padding: 16px;
      display: grid;
      gap: 16px;
    }
    details {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel-soft);
    }
    summary {
      cursor: pointer;
      padding: 11px 13px;
      font-weight: 710;
      color: #26302b;
    }
    details .details-body {
      padding: 0 13px 13px;
    }
    .chart-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
      gap: 12px;
    }
    .wide-chart-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(340px, 1fr));
      gap: 12px;
    }
    .chart-card,
    .table-card {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      padding: 12px;
      min-width: 0;
    }
    .bar-list {
      display: grid;
      gap: 8px;
    }
    .bar-row {
      display: grid;
      grid-template-columns: minmax(72px, 110px) 1fr minmax(56px, auto);
      gap: 8px;
      align-items: center;
      font-size: 12px;
    }
    .bar-name {
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      color: #2d3833;
    }
    .bar-track {
      position: relative;
      height: 12px;
      border-radius: 999px;
      background: #edf1eb;
      overflow: hidden;
      border: 1px solid #d9e1d6;
    }
    .bar-fill {
      position: absolute;
      inset: 0 auto 0 0;
      min-width: 2px;
      border-radius: 999px;
    }
    .bar-value {
      color: var(--muted);
      text-align: right;
      font-variant-numeric: tabular-nums;
    }
    .canvas-card canvas {
      display: block;
      width: 100%;
      height: auto;
      border-radius: 7px;
      border: 1px solid var(--line);
      background: #fbfcf8;
    }
    .canvas-caption {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
      margin: 8px 0 0;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.3;
    }
    .legend {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      margin-top: 8px;
      color: var(--muted);
      font-size: 12px;
    }
    .legend span {
      display: inline-flex;
      align-items: center;
      gap: 5px;
    }
    .swatch {
      width: 9px;
      height: 9px;
      border-radius: 999px;
      display: inline-block;
    }
    .table-wrap {
      overflow: auto;
      border: 1px solid var(--line);
      border-radius: 8px;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
      min-width: 760px;
      background: var(--panel);
    }
    th,
    td {
      padding: 8px 9px;
      border-bottom: 1px solid var(--line);
      text-align: left;
      vertical-align: top;
    }
    th {
      position: sticky;
      top: 0;
      z-index: 1;
      background: #f2f5ef;
      color: #405047;
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0;
      white-space: nowrap;
    }
    tr:last-child td { border-bottom: 0; }
    td.num {
      text-align: right;
      font-variant-numeric: tabular-nums;
      white-space: nowrap;
    }
    .delta-up { color: var(--green); font-weight: 690; }
    .delta-down { color: var(--red); font-weight: 690; }
    .muted { color: var(--muted); }
    .subtle {
      color: var(--muted);
      font-size: 12px;
      line-height: 1.4;
    }
    pre {
      white-space: pre-wrap;
      word-break: break-word;
      margin: 0;
      color: #34413c;
      font-size: 12px;
      line-height: 1.45;
    }
    code {
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
      font-size: 0.94em;
    }
    .empty {
      color: var(--muted);
      padding: 10px;
      border: 1px dashed var(--line-strong);
      border-radius: 8px;
      background: var(--panel-soft);
      font-size: 13px;
    }
    @media (max-width: 760px) {
      .topbar-inner,
      .run-head {
        grid-template-columns: 1fr;
      }
      .navlinks,
      .run-actions {
        justify-content: flex-start;
      }
      .shell {
        padding: 12px;
      }
      .bar-row {
        grid-template-columns: minmax(58px, 86px) 1fr minmax(48px, auto);
      }
      table {
        min-width: 680px;
      }
    }
  </style>
</head>
<body>
  <header class="topbar">
    <div class="topbar-inner">
      <div class="brand">
        <h1>Sphere-JEPA Results Dashboard</h1>
        <p>Static aggregate viewer for the GPU reports generated on 2026-05-09.</p>
      </div>
      <nav class="navlinks" id="top-nav"></nav>
    </div>
  </header>
  <main class="shell">
    <section class="section" id="overview"></section>
    <section class="section" id="preprocessing"></section>
    <section class="section" id="runs"></section>
  </main>
  <script id="report-data" type="application/json">__REPORT_DATA__</script>
  <script>
    const REPORT = JSON.parse(document.getElementById("report-data").textContent);
    const PALETTE = ["#246a8f", "#b94e4e", "#1f7a63", "#bd812d", "#6654a2", "#188092", "#8d5a2b", "#5c7a24", "#c15b7a", "#4d6fb3"];
    let drawQueue = [];
    let canvasId = 0;

    function esc(value) {
      return String(value ?? "").replace(/[&<>"']/g, (ch) => ({
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&#39;",
      }[ch]));
    }

    function finite(value) {
      return Number.isFinite(value);
    }

    function asNumber(value) {
      const n = Number(value);
      return Number.isFinite(n) ? n : NaN;
    }

    function getPath(obj, path) {
      let cur = obj;
      for (const part of path.split(".")) {
        if (cur == null || typeof cur !== "object" || !(part in cur)) return NaN;
        cur = cur[part];
      }
      return asNumber(cur);
    }

    function fmt(value, kind = "number") {
      const n = asNumber(value);
      if (!Number.isFinite(n)) return "n/a";
      if (kind === "percent") return `${(n * 100).toFixed(Math.abs(n) < 0.1 ? 1 : 0)}%`;
      if (kind === "pp") return `${(n * 100).toFixed(1)} pp`;
      if (Math.abs(n) >= 1000) return n.toFixed(0);
      if (Math.abs(n) >= 100) return n.toFixed(1);
      if (Math.abs(n) >= 10) return n.toFixed(2);
      if (Math.abs(n) >= 1) return n.toFixed(3);
      if (Math.abs(n) >= 0.01) return n.toFixed(4);
      return n.toExponential(2);
    }

    function rankValue(record, space = "auto") {
      if (space === "text") return getPath(record, "text_rankme");
      if (space === "code") return getPath(record, "code_rankme");
      const own = getPath(record, "rankme");
      if (finite(own)) return own;
      return getPath(record, "text_rankme");
    }

    function meanCosine(record, space = "auto") {
      if (space === "text") return getPath(record, "text_cosine.mean");
      if (space === "code") return getPath(record, "code_cosine.mean");
      const own = getPath(record, "cosine.mean");
      if (finite(own)) return own;
      return getPath(record, "text_cosine.mean");
    }

    function withinRank(record) {
      return getPath(record, "within_class_rankme._mean_within_class");
    }

    function recordsByVariant(run) {
      const out = new Map();
      for (const record of run.records || []) out.set(record.variant, record);
      return out;
    }

    function variant(record) {
      return String(record.variant ?? "");
    }

    function configChips(run) {
      const cfg = run.config || {};
      const cache = run.cache_config || {};
      const parts = [];
      if (Array.isArray(cfg.variants)) parts.push(`${cfg.variants.length} variants`);
      if (cfg.emb_dim) parts.push(`emb dim ${cfg.emb_dim}`);
      if (cfg.epochs) parts.push(`${cfg.epochs} epochs`);
      if (cfg.steps) parts.push(`${cfg.steps} steps`);
      if (cfg.anchor_preprocess) parts.push(`anchor ${cfg.anchor_preprocess}`);
      if (cfg.input_preprocess) parts.push(`input ${cfg.input_preprocess}`);
      if (cfg.sigma_max != null) parts.push(`sigma max ${cfg.sigma_max}`);
      if (cache.model_name) parts.push(cache.model_name);
      if (cache.pooling) parts.push(`${cache.pooling} pooling`);
      if (cfg.train_limit) parts.push(`train ${cfg.train_limit}`);
      if (cfg.test_limit) parts.push(`test ${cfg.test_limit}`);
      return parts;
    }

    function htmlLink(path, label) {
      return path ? `<a class="pill" href="${esc(path)}">${esc(label)}</a>` : "";
    }

    function renderTopNav() {
      const nav = document.getElementById("top-nav");
      nav.innerHTML = [
        `<a href="#overview">Overview</a>`,
        `<a href="#preprocessing">Preprocessing</a>`,
        ...REPORT.runs.map((run) => `<a href="#${esc(run.slug)}">${esc(run.short_title)}</a>`),
      ].join("");
    }

    function allRecords() {
      return REPORT.runs.flatMap((run) => run.records || []);
    }

    function bestBy(records, path, larger = true) {
      let best = null;
      for (const record of records) {
        const value = getPath(record, path);
        if (!finite(value)) continue;
        if (!best || (larger ? value > best.value : value < best.value)) {
          best = { record, value };
        }
      }
      return best;
    }

    function overviewStats() {
      const runs = REPORT.runs;
      const records = allRecords();
      const anchorCount = runs.reduce((sum, run) => sum + Object.keys(run.anchor_geometry || {}).length, 0);
      const heatmapCount = records.reduce((sum, record) => {
        return sum + ["embedding_similarity_heatmap", "text_cap_to_text_anchor_heatmap", "code_cap_to_code_anchor_heatmap", "text_cap_to_code_anchor_heatmap"].filter((key) => record[key]).length;
      }, 0);
      const focusRun = runs.find((run) => run.slug === "exp2_pythia160m_synth_input_white_anchor_white");
      const focusBy = focusRun ? recordsByVariant(focusRun) : new Map();
      const d1 = focusBy.get("D_own_1");
      const d0 = focusBy.get("D_own_0");
      const c = focusBy.get("C");
      const predGain = d1 && d0 ? getPath(d1, "retrieval_predicted_code_embedding.recall@1") - getPath(d0, "retrieval_predicted_code_embedding.recall@1") : NaN;
      const codeRankGain = d1 && d0 ? getPath(d1, "code_rankme") - getPath(d0, "code_rankme") : NaN;
      const anchorGain = d1 && c ? getPath(d1, "code_rankme") - getPath(c, "code_rankme") : NaN;
      return [
        { label: "Runs", value: String(runs.length), note: "Exp0 toy, Exp1 CIFAR bridge, and five Exp2 frozen-LLM sweeps." },
        { label: "Variants", value: String(records.length), note: "Every record in the GPU summary JSON is represented in tables and charts." },
        { label: "Anchor diagnostics", value: String(anchorCount), note: "RankMe, top eig mass, norm, uniformity, cosine histograms, and spectra." },
        { label: "Heatmaps", value: String(heatmapCount), note: "Retrieval and cap-to-anchor similarity matrices rendered on canvas." },
        { label: "D1 pred-code gain", value: finite(predGain) ? fmt(predGain, "pp") : "n/a", note: "Source+anchor whitening, D_own_1 minus D_own_0 recall@1." },
        { label: "D1 code RankMe gain", value: finite(codeRankGain) ? fmt(codeRankGain) : "n/a", note: "Source+anchor whitening, D_own_1 minus D_own_0." },
        { label: "Anchor over C", value: finite(anchorGain) ? fmt(anchorGain) : "n/a", note: "Source+anchor whitening, D_own_1 code RankMe minus co-trained C." },
      ];
    }

    function renderOverview() {
      const el = document.getElementById("overview");
      const statHtml = overviewStats().map((item) => `
        <article class="stat">
          <div class="label">${esc(item.label)}</div>
          <div class="value">${esc(item.value)}</div>
          <div class="note">${esc(item.note)}</div>
        </article>
      `).join("");
      const cards = REPORT.runs.map((run) => `
        <article class="run-card">
          <h3>${esc(run.title)}</h3>
          <p>${esc(run.summary)}</p>
          <div class="chips">${configChips(run).slice(0, 7).map((chip) => `<span class="chip">${esc(chip)}</span>`).join("")}</div>
          <ul class="takeaways">${(run.takeaways || []).map((item) => `<li>${esc(item)}</li>`).join("")}</ul>
        </article>
      `).join("");
      el.innerHTML = `
        <div class="section-title">
          <h2>Overview</h2>
          <p>All summary JSON files are embedded in this page; no server is required.</p>
        </div>
        <div class="grid stats">${statHtml}</div>
        <div class="section-title" style="margin-top: 18px;">
          <h2>Run Notes</h2>
          <p>Interpretation cues from the handoff and report README.</p>
        </div>
        <div class="cards">${cards}</div>
      `;
    }

    function renderPreprocessing() {
      const exp2Runs = REPORT.runs.filter((run) => run.slug.startsWith("exp2_pythia160m"));
      const rows = [];
      for (const run of exp2Runs) {
        const by = recordsByVariant(run);
        const d1 = by.get("D_own_1");
        const d0 = by.get("D_own_0");
        if (!d1) continue;
        const cfg = run.config || {};
        rows.push({
          run,
          input: cfg.input_preprocess || "raw",
          anchor: cfg.anchor_preprocess || "raw",
          textRank: getPath(d1, "text_rankme"),
          codeRank: getPath(d1, "code_rankme"),
          textCos: getPath(d1, "text_cosine.mean"),
          codeCos: getPath(d1, "code_cosine.mean"),
          predR1: getPath(d1, "retrieval_predicted_code_embedding.recall@1"),
          dTextRank: d0 ? getPath(d1, "text_rankme") - getPath(d0, "text_rankme") : NaN,
          dCodeRank: d0 ? getPath(d1, "code_rankme") - getPath(d0, "code_rankme") : NaN,
          dPredR1: d0 ? getPath(d1, "retrieval_predicted_code_embedding.recall@1") - getPath(d0, "retrieval_predicted_code_embedding.recall@1") : NaN,
        });
      }
      const tableRows = rows.map((row) => `
        <tr>
          <td><a href="#${esc(row.run.slug)}">${esc(row.run.title)}</a></td>
          <td>${esc(row.input)}</td>
          <td>${esc(row.anchor)}</td>
          <td class="num">${fmt(row.textRank)}</td>
          <td class="num">${fmt(row.codeRank)}</td>
          <td class="num">${fmt(row.textCos)}</td>
          <td class="num">${fmt(row.codeCos)}</td>
          <td class="num">${fmt(row.predR1, "percent")}</td>
          <td class="num ${row.dTextRank >= 0 ? "delta-up" : "delta-down"}">${fmt(row.dTextRank)}</td>
          <td class="num ${row.dCodeRank >= 0 ? "delta-up" : "delta-down"}">${fmt(row.dCodeRank)}</td>
          <td class="num ${row.dPredR1 >= 0 ? "delta-up" : "delta-down"}">${fmt(row.dPredR1, "pp")}</td>
        </tr>
      `).join("");
      const chart = barChart(rows.map((row) => ({
        variant: `${row.input}/${row.anchor}`,
        text_rankme: row.textRank,
        code_rankme: row.codeRank,
        pred_r1: row.predR1,
      })), [
        { label: "D_own_1 text RankMe", path: "text_rankme", color: "#246a8f" },
        { label: "D_own_1 code RankMe", path: "code_rankme", color: "#1f7a63" },
        { label: "Predicted-code R@1", path: "pred_r1", color: "#bd812d", kind: "percent" },
      ]);
      document.getElementById("preprocessing").innerHTML = `
        <div class="section-title">
          <h2>Exp2 Preprocessing Sweep</h2>
          <p>D_own_1 rows compare raw, normalized, spherified, whitened, and source-whitened setups.</p>
        </div>
        <div class="panel panel-pad">
          <div class="wide-chart-grid">
            ${chart}
            <div class="table-card">
              <h3>D_own_1 Across Preprocessing Modes</h3>
              <div class="table-wrap">
                <table>
                  <thead>
                    <tr>
                      <th>Run</th><th>Input</th><th>Anchor</th><th>Text RankMe</th><th>Code RankMe</th>
                      <th>Text Cos</th><th>Code Cos</th><th>Pred R@1</th><th>Delta Text Rank</th><th>Delta Code Rank</th><th>Delta Pred R@1</th>
                    </tr>
                  </thead>
                  <tbody>${tableRows}</tbody>
                </table>
              </div>
              <p class="subtle" style="margin: 9px 0 0;">Deltas are D_own_1 minus D_own_0 inside each preprocessing run.</p>
            </div>
          </div>
        </div>
      `;
    }

    function metricSpecs(run) {
      const records = run.records || [];
      const has = (path) => records.some((record) => finite(getPath(record, path)));
      if (has("text_rankme")) {
        const specs = [
          { label: "Text RankMe", path: "text_rankme", color: "#246a8f" },
          { label: "Code RankMe", path: "code_rankme", color: "#1f7a63" },
          { label: "Text mean cosine", path: "text_cosine.mean", color: "#b94e4e" },
          { label: "Code mean cosine", path: "code_cosine.mean", color: "#bd812d" },
        ];
        if (has("retrieval_predicted_code_embedding.recall@1")) specs.push({ label: "Pred-code R@1", path: "retrieval_predicted_code_embedding.recall@1", color: "#6654a2", kind: "percent" });
        if (has("retrieval_embedding.recall@1")) specs.push({ label: "Direct emb R@1", path: "retrieval_embedding.recall@1", color: "#188092", kind: "percent" });
        return specs;
      }
      const specs = [
        { label: "RankMe", path: "rankme", color: "#246a8f" },
        { label: "Within-class RankMe", path: "within_class_rankme._mean_within_class", color: "#1f7a63" },
        { label: "Mean cosine", path: "cosine.mean", color: "#b94e4e" },
        { label: "Uniformity", path: "uniformity", color: "#bd812d" },
      ];
      if (has("linear_probe_top1")) specs.unshift({ label: "Linear probe top-1", path: "linear_probe_top1", color: "#6654a2", kind: "percent" });
      if (has("alignment_view_ab")) specs.push({ label: "View alignment", path: "alignment_view_ab", color: "#188092" });
      return specs;
    }

    function barChart(records, specs) {
      return specs.map((spec) => {
        const values = records.map((record) => getPath(record, spec.path)).filter(finite);
        if (!values.length) return "";
        const minValue = Math.min(...values, 0);
        const maxValue = Math.max(...values);
        const span = Math.max(maxValue - minValue, 1e-9);
        const rows = records.map((record) => {
          const value = getPath(record, spec.path);
          if (!finite(value)) return "";
          const width = Math.max(2, ((value - minValue) / span) * 100);
          return `
            <div class="bar-row">
              <div class="bar-name" title="${esc(variant(record))}">${esc(variant(record))}</div>
              <div class="bar-track"><div class="bar-fill" style="width: ${width.toFixed(2)}%; background: ${spec.color};"></div></div>
              <div class="bar-value">${fmt(value, spec.kind)}</div>
            </div>
          `;
        }).join("");
        return `
          <div class="chart-card">
            <h3>${esc(spec.label)}</h3>
            <div class="bar-list">${rows}</div>
          </div>
        `;
      }).join("");
    }

    function signedClass(value, largerBetter = true) {
      const n = asNumber(value);
      if (!finite(n) || Math.abs(n) < 1e-12) return "";
      const good = largerBetter ? n > 0 : n < 0;
      return good ? "delta-up" : "delta-down";
    }

    function contrastPairs(run) {
      const names = new Set((run.records || []).map((record) => variant(record)));
      const candidates = [
        ["D1a", "D0a", "Noise: continuous anchor"],
        ["D1b", "D0b", "Noise: RFF anchor"],
        ["D1c", "D0c", "Noise: frozen teacher anchor"],
        ["D1a", "C", "Anchor: D1a vs co-trained"],
        ["D1b", "C", "Anchor: D1b vs co-trained"],
        ["D1c", "C", "Anchor: D1c vs co-trained"],
        ["D1a", "D3", "Sample-specific vs class-level"],
        ["D1", "D0", "Noise: frozen anchor"],
        ["D1", "C", "Anchor: D1 vs co-trained"],
        ["D1", "C_ema", "Anchor: D1 vs EMA"],
        ["D1", "D2", "Sample-specific vs class-level"],
        ["D_own_1", "D_own_0", "Noise: own-view anchor"],
        ["D_own_1", "C", "Anchor: own-view vs co-trained"],
        ["D_own_1", "C_ema", "Anchor: own-view vs EMA"],
        ["D_cross_1", "D_cross_0", "Noise: cross-view anchor"],
        ["D_cross_sym_1", "D_cross_sym_0", "Noise: symmetric cross-view anchor"],
        ["F", "D_own_1", "InfoNCE vs D_own_1"],
        ["G", "D_own_1", "VICReg vs D_own_1"],
        ["H", "D_own_1", "SIGReg vs D_own_1"],
      ];
      return candidates.filter(([left, right]) => names.has(left) && names.has(right));
    }

    function contrastTable(run) {
      const by = recordsByVariant(run);
      const rows = contrastPairs(run).map(([leftName, rightName, label]) => {
        const left = by.get(leftName);
        const right = by.get(rightName);
        const dRank = rankValue(left) - rankValue(right);
        const dCodeRank = getPath(left, "code_rankme") - getPath(right, "code_rankme");
        const dCos = meanCosine(left) - meanCosine(right);
        const dProbe = getPath(left, "linear_probe_top1") - getPath(right, "linear_probe_top1");
        const dPred = getPath(left, "retrieval_predicted_code_embedding.recall@1") - getPath(right, "retrieval_predicted_code_embedding.recall@1");
        return `
          <tr>
            <td>${esc(label)}</td>
            <td><code>${esc(leftName)}</code> - <code>${esc(rightName)}</code></td>
            <td class="num ${signedClass(dRank)}">${fmt(dRank)}</td>
            <td class="num ${signedClass(dCodeRank)}">${fmt(dCodeRank)}</td>
            <td class="num ${signedClass(dCos, false)}">${fmt(dCos)}</td>
            <td class="num ${signedClass(dProbe)}">${fmt(dProbe, "pp")}</td>
            <td class="num ${signedClass(dPred)}">${fmt(dPred, "pp")}</td>
          </tr>
        `;
      }).join("");
      if (!rows) return "";
      return `
        <div class="table-card">
          <h3>Headline Contrasts</h3>
          <div class="table-wrap">
            <table>
              <thead><tr><th>Question</th><th>Pair</th><th>Delta RankMe</th><th>Delta Code RankMe</th><th>Delta Mean Cos</th><th>Delta Probe</th><th>Delta Pred R@1</th></tr></thead>
              <tbody>${rows}</tbody>
            </table>
          </div>
        </div>
      `;
    }

    function metricTable(run) {
      const records = run.records || [];
      const columns = [
        { label: "RankMe", path: "rankme" },
        { label: "Text RankMe", path: "text_rankme" },
        { label: "Code RankMe", path: "code_rankme" },
        { label: "Within RankMe", path: "within_class_rankme._mean_within_class" },
        { label: "Uniformity", path: "uniformity" },
        { label: "Text Uniformity", path: "text_uniformity" },
        { label: "Code Uniformity", path: "code_uniformity" },
        { label: "Mean Cos", path: "cosine.mean" },
        { label: "Text Cos", path: "text_cosine.mean" },
        { label: "Code Cos", path: "code_cosine.mean" },
        { label: "P95 Cos", path: "cosine.p95" },
        { label: "Probe Top-1", path: "linear_probe_top1", kind: "percent" },
        { label: "View Align", path: "alignment_view_ab" },
        { label: "Direct R@1", path: "retrieval_embedding.recall@1", kind: "percent" },
        { label: "Pred-Code R@1", path: "retrieval_predicted_code_embedding.recall@1", kind: "percent" },
        { label: "Text Cap R@1", path: "retrieval_text_cap_to_text_anchor.recall@1", kind: "percent" },
        { label: "Code Cap R@1", path: "retrieval_code_cap_to_code_anchor.recall@1", kind: "percent" },
      ].filter((col) => records.some((record) => finite(getPath(record, col.path))));
      const rows = records.map((record) => `
        <tr>
          <td><code>${esc(variant(record))}</code></td>
          ${columns.map((col) => `<td class="num">${fmt(getPath(record, col.path), col.kind)}</td>`).join("")}
        </tr>
      `).join("");
      return `
        <div class="table-card">
          <h3>Metrics Table</h3>
          <div class="table-wrap">
            <table>
              <thead><tr><th>Variant</th>${columns.map((col) => `<th>${esc(col.label)}</th>`).join("")}</tr></thead>
              <tbody>${rows}</tbody>
            </table>
          </div>
        </div>
      `;
    }

    function retrievalTable(run) {
      const rows = [];
      for (const record of run.records || []) {
        for (const [key, value] of Object.entries(record)) {
          if (!key.startsWith("retrieval") || !value || typeof value !== "object") continue;
          rows.push(`
            <tr>
              <td><code>${esc(variant(record))}</code></td>
              <td>${esc(key.replace(/^retrieval_/, "").replaceAll("_", " "))}</td>
              <td class="num">${fmt(value["recall@1"], "percent")}</td>
              <td class="num">${fmt(value["recall@10"], "percent")}</td>
              <td class="num">${fmt(value.mean_rank)}</td>
            </tr>
          `);
        }
      }
      if (!rows.length) return "";
      return `
        <div class="table-card">
          <h3>Retrieval Spaces</h3>
          <div class="table-wrap">
            <table>
              <thead><tr><th>Variant</th><th>Space</th><th>R@1</th><th>R@10</th><th>Mean Rank</th></tr></thead>
              <tbody>${rows.join("")}</tbody>
            </table>
          </div>
        </div>
      `;
    }

    function anchorTable(run) {
      const anchors = run.anchor_geometry || {};
      const names = Object.keys(anchors);
      if (!names.length) return "";
      const rows = names.map((name) => {
        const diag = anchors[name] || {};
        return `
          <tr>
            <td><code>${esc(name)}</code></td>
            <td class="num">${fmt(diag.rankme)}</td>
            <td class="num">${fmt(diag.top_eig_mass_ratio)}</td>
            <td class="num">${fmt(diag.mean_norm)}</td>
            <td class="num">${fmt(diag.uniformity_after_normalize)}</td>
            <td class="num">${fmt(getPath(diag, "cosine_stats.mean"))}</td>
            <td class="num">${fmt(getPath(diag, "cosine_stats.p95"))}</td>
          </tr>
        `;
      }).join("");
      return `
        <div class="table-card">
          <h3>Anchor Geometry</h3>
          <div class="table-wrap">
            <table>
              <thead><tr><th>Anchor</th><th>RankMe</th><th>Top Eig Mass</th><th>Mean Norm</th><th>Uniformity</th><th>Mean Cos</th><th>P95 Cos</th></tr></thead>
              <tbody>${rows}</tbody>
            </table>
          </div>
        </div>
      `;
    }

    function canvasTask(task, width = 520, height = 320) {
      const id = `chart-${canvasId++}`;
      drawQueue.push({ ...task, id });
      return `<canvas id="${id}" width="${width}" height="${height}" aria-label="${esc(task.title || task.type)}"></canvas>`;
    }

    function scatterKeys(record) {
      return [
        ["scatter", "Embedding"],
        ["pca_scatter", "PCA"],
        ["text_pca_scatter", "Text"],
        ["code_pca_scatter", "Code"],
      ].filter(([key]) => Array.isArray(record[key]) && record[key].length);
    }

    function sphereCards(run) {
      const cards = [];
      for (const record of run.records || []) {
        for (const [key, label] of scatterKeys(record)) {
          const isCode = key.startsWith("code");
          const isText = key.startsWith("text");
          const space = isCode ? "code" : isText ? "text" : "auto";
          const rank = rankValue(record, space);
          const meanCos = meanCosine(record, space);
          const dim = Number(run.config?.emb_dim || run.config?.teacher_dim || 0);
          const occupancy = finite(rank) && dim > 0 ? Math.min(1, rank / dim) : NaN;
          cards.push(`
            <article class="chart-card canvas-card">
              <h3>${esc(variant(record))} ${esc(label)} Hypersphere</h3>
              ${canvasTask({ type: "sphere", points: record[key], title: `${variant(record)} ${label}` }, 360, 300)}
              <div class="canvas-caption">
                <span>RankMe ${fmt(rank)}${finite(occupancy) ? ` / ${fmt(occupancy, "percent")} occupancy` : ""}</span>
                <span>mean cos ${fmt(meanCos)}</span>
              </div>
            </article>
          `);
        }
      }
      return cards.join("");
    }

    function pcaCards(run) {
      const cards = [];
      for (const record of run.records || []) {
        for (const [key, label] of scatterKeys(record)) {
          cards.push(`
            <article class="chart-card canvas-card">
              <h3>${esc(variant(record))} ${esc(label)} Map</h3>
              ${canvasTask({ type: "scatter", points: record[key], title: `${variant(record)} ${label}` }, 520, 340)}
              <div class="canvas-caption"><span>2-D projection of stored embedding samples</span><span>${record[key].length} points</span></div>
            </article>
          `);
        }
      }
      return cards.join("");
    }

    function diagnosticsCards(run) {
      const cards = [];
      for (const [name, diag] of Object.entries(run.anchor_geometry || {})) {
        if (diag.cosine_histogram) {
          cards.push(`
            <article class="chart-card canvas-card">
              <h3>${esc(name)} Anchor Cosines</h3>
              ${canvasTask({ type: "histogram", hist: diag.cosine_histogram, title: `${name} cosines` }, 420, 260)}
              <div class="canvas-caption"><span>Pairwise cosine distribution</span><span>mean ${fmt(getPath(diag, "cosine_stats.mean"))}</span></div>
            </article>
          `);
        }
        if (diag.eigen_spectrum) {
          cards.push(`
            <article class="chart-card canvas-card">
              <h3>${esc(name)} Anchor Spectrum</h3>
              ${canvasTask({ type: "spectrum", spectrum: diag.eigen_spectrum, title: `${name} spectrum` }, 420, 260)}
              <div class="canvas-caption"><span>Eigenvalue mass</span><span>top mass ${fmt(diag.top_eig_mass_ratio)}</span></div>
            </article>
          `);
        }
      }
      for (const record of run.records || []) {
        for (const [key, label] of [
          ["cosine_histogram", "Embedding Cosines"],
          ["text_cosine_histogram", "Text Cosines"],
          ["code_cosine_histogram", "Code Cosines"],
          ["eigen_spectrum", "Embedding Spectrum"],
          ["text_eigen_spectrum", "Text Spectrum"],
          ["code_eigen_spectrum", "Code Spectrum"],
        ]) {
          if (!record[key]) continue;
          const type = key.includes("histogram") ? "histogram" : "spectrum";
          cards.push(`
            <article class="chart-card canvas-card">
              <h3>${esc(variant(record))} ${esc(label)}</h3>
              ${canvasTask({ type, hist: record[key], spectrum: record[key], title: `${variant(record)} ${label}` }, 420, 260)}
              <div class="canvas-caption"><span>${type === "histogram" ? "Pairwise cosine distribution" : "Eigenvalue mass"}</span><span>${esc(run.short_title)}</span></div>
            </article>
          `);
        }
      }
      return cards.join("");
    }

    function heatmapCards(run) {
      const keys = [
        ["embedding_similarity_heatmap", "Embedding Similarity"],
        ["text_cap_to_text_anchor_heatmap", "Text Cap to Text Anchor"],
        ["code_cap_to_code_anchor_heatmap", "Code Cap to Code Anchor"],
        ["text_cap_to_code_anchor_heatmap", "Text Cap to Code Anchor"],
      ];
      const cards = [];
      for (const record of run.records || []) {
        for (const [key, label] of keys) {
          if (!record[key]) continue;
          cards.push(`
            <article class="chart-card canvas-card">
              <h3>${esc(variant(record))} ${esc(label)}</h3>
              ${canvasTask({ type: "heatmap", heatmap: record[key], title: `${variant(record)} ${label}` }, 340, 340)}
              <div class="canvas-caption"><span>blue positive, red negative</span><span>${record[key].values?.length || 0} x ${record[key].values?.length || 0}</span></div>
            </article>
          `);
        }
      }
      return cards.join("");
    }

    function historyCards(run) {
      const cards = [];
      for (const record of run.records || []) {
        if (!Array.isArray(record.history) || !record.history.length) continue;
        cards.push(`
          <article class="chart-card canvas-card">
            <h3>${esc(variant(record))} Loss Trace</h3>
            ${canvasTask({ type: "history", history: record.history, title: `${variant(record)} loss` }, 420, 240)}
            <div class="canvas-caption"><span>${record.history.length} logged points</span><span>final ${fmt(record.history[record.history.length - 1]?.loss)}</span></div>
          </article>
        `);
      }
      return cards.join("");
    }

    function renderRun(run) {
      const chartHtml = barChart(run.records || [], metricSpecs(run));
      const sphereHtml = sphereCards(run);
      const pcaHtml = pcaCards(run);
      const diagHtml = diagnosticsCards(run);
      const heatHtml = heatmapCards(run);
      const histHtml = historyCards(run);
      return `
        <article class="panel run" id="${esc(run.slug)}">
          <div class="run-head">
            <div class="run-title">
              <h2>${esc(run.title)}</h2>
              <p>${esc(run.summary)}</p>
              <div class="chips">${configChips(run).map((chip) => `<span class="chip">${esc(chip)}</span>`).join("")}</div>
            </div>
            <div class="run-actions">
              ${htmlLink(run.json_path, "JSON")}
              ${htmlLink(run.html_path, "Original page")}
            </div>
          </div>
          <div class="run-body">
            <div class="chart-grid">${chartHtml}</div>
            <div class="wide-chart-grid">
              ${contrastTable(run)}
              ${metricTable(run)}
              ${retrievalTable(run)}
              ${anchorTable(run)}
            </div>
            <details open>
              <summary>Hypersphere projections</summary>
              <div class="details-body">
                <p class="subtle">Each card projects stored 2-D embedding samples into a unit-disk view. It is a visual diagnostic for spread on the high-dimensional hypersphere, not a replacement for RankMe or cosine statistics.</p>
                <div class="chart-grid" style="margin-top: 10px;">${sphereHtml || `<div class="empty">No scatter samples stored for this run.</div>`}</div>
              </div>
            </details>
            <details>
              <summary>PCA embedding maps</summary>
              <div class="details-body"><div class="wide-chart-grid">${pcaHtml || `<div class="empty">No PCA samples stored for this run.</div>`}</div></div>
            </details>
            <details>
              <summary>Cosine histograms and eigenspectra</summary>
              <div class="details-body"><div class="chart-grid">${diagHtml || `<div class="empty">No histogram or eigenspectrum diagnostics stored for this run.</div>`}</div></div>
            </details>
            <details>
              <summary>Retrieval heatmaps</summary>
              <div class="details-body"><div class="chart-grid">${heatHtml || `<div class="empty">No heatmaps stored for this run.</div>`}</div></div>
            </details>
            ${histHtml ? `<details><summary>Training traces</summary><div class="details-body"><div class="chart-grid">${histHtml}</div></div></details>` : ""}
            <details>
              <summary>Run config JSON</summary>
              <div class="details-body"><pre>${esc(JSON.stringify({ config: run.config, cache_config: run.cache_config, headline_contrasts: run.headline_contrasts }, null, 2))}</pre></div>
            </details>
          </div>
        </article>
      `;
    }

    function renderRuns() {
      document.getElementById("runs").innerHTML = `
        <div class="section-title">
          <h2>All Runs</h2>
          <p>Open the detail panels for dense embedding, hypersphere, histogram, spectrum, and heatmap views.</p>
        </div>
        ${REPORT.runs.map(renderRun).join("")}
      `;
    }

    function drawAll() {
      for (const task of drawQueue) {
        const canvas = document.getElementById(task.id);
        if (!canvas) continue;
        if (task.type === "scatter") drawScatter(canvas, task.points, false);
        if (task.type === "sphere") drawScatter(canvas, task.points, true);
        if (task.type === "histogram") drawHistogram(canvas, task.hist);
        if (task.type === "spectrum") drawSpectrum(canvas, task.spectrum);
        if (task.type === "heatmap") drawHeatmap(canvas, task.heatmap);
        if (task.type === "history") drawHistory(canvas, task.history);
      }
      drawLegend();
    }

    function setupCanvas(canvas) {
      const ctx = canvas.getContext("2d");
      const w = canvas.width;
      const h = canvas.height;
      ctx.clearRect(0, 0, w, h);
      ctx.fillStyle = "#fbfcf8";
      ctx.fillRect(0, 0, w, h);
      return { ctx, w, h };
    }

    function drawFrame(ctx, x, y, w, h) {
      ctx.strokeStyle = "#dfe5dc";
      ctx.lineWidth = 1;
      ctx.strokeRect(x, y, w, h);
    }

    function drawScatter(canvas, points, sphereMode) {
      const { ctx, w, h } = setupCanvas(canvas);
      const pad = sphereMode ? 24 : 30;
      const xs = points.map((p) => Number(p.x)).filter(Number.isFinite);
      const ys = points.map((p) => Number(p.y)).filter(Number.isFinite);
      if (!xs.length || !ys.length) return;
      const minX = Math.min(...xs);
      const maxX = Math.max(...xs);
      const minY = Math.min(...ys);
      const maxY = Math.max(...ys);
      const spanX = Math.max(maxX - minX, 1e-9);
      const spanY = Math.max(maxY - minY, 1e-9);

      if (sphereMode) {
        const cx = w / 2;
        const cy = h / 2;
        const r = Math.min(w, h) / 2 - pad;
        ctx.strokeStyle = "#c8d2c8";
        ctx.lineWidth = 1;
        for (const frac of [0.25, 0.5, 0.75, 1]) {
          ctx.beginPath();
          ctx.arc(cx, cy, r * frac, 0, Math.PI * 2);
          ctx.stroke();
        }
        ctx.strokeStyle = "#a9b8ad";
        ctx.beginPath();
        ctx.moveTo(cx - r, cy);
        ctx.lineTo(cx + r, cy);
        ctx.moveTo(cx, cy - r);
        ctx.lineTo(cx, cy + r);
        ctx.stroke();
        ctx.save();
        ctx.beginPath();
        ctx.arc(cx, cy, r, 0, Math.PI * 2);
        ctx.clip();
        for (const p of points) {
          const x0 = ((Number(p.x) - minX) / spanX) * 2 - 1;
          const y0 = ((Number(p.y) - minY) / spanY) * 2 - 1;
          const norm = Math.max(1, Math.hypot(x0, y0));
          const x = cx + (x0 / norm) * r * 0.96;
          const y = cy - (y0 / norm) * r * 0.96;
          const label = Number.isFinite(Number(p.label)) ? Number(p.label) : 0;
          ctx.fillStyle = PALETTE[Math.abs(label) % PALETTE.length];
          ctx.globalAlpha = 0.72;
          ctx.beginPath();
          ctx.arc(x, y, 2.0, 0, Math.PI * 2);
          ctx.fill();
        }
        ctx.restore();
        ctx.globalAlpha = 1;
        ctx.strokeStyle = "#5d6b63";
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.arc(cx, cy, r, 0, Math.PI * 2);
        ctx.stroke();
        return;
      }

      const plotX = pad;
      const plotY = 18;
      const plotW = w - pad - 14;
      const plotH = h - 42;
      drawFrame(ctx, plotX, plotY, plotW, plotH);
      ctx.strokeStyle = "#d9e1d6";
      ctx.beginPath();
      ctx.moveTo(plotX, plotY + plotH / 2);
      ctx.lineTo(plotX + plotW, plotY + plotH / 2);
      ctx.moveTo(plotX + plotW / 2, plotY);
      ctx.lineTo(plotX + plotW / 2, plotY + plotH);
      ctx.stroke();
      for (const p of points) {
        const x = plotX + ((Number(p.x) - minX) / spanX) * plotW;
        const y = plotY + plotH - ((Number(p.y) - minY) / spanY) * plotH;
        const label = Number.isFinite(Number(p.label)) ? Number(p.label) : 0;
        ctx.fillStyle = PALETTE[Math.abs(label) % PALETTE.length];
        ctx.globalAlpha = 0.7;
        ctx.beginPath();
        ctx.arc(x, y, 2.0, 0, Math.PI * 2);
        ctx.fill();
      }
      ctx.globalAlpha = 1;
    }

    function drawHistogram(canvas, hist) {
      const { ctx, w, h } = setupCanvas(canvas);
      const counts = (hist?.counts || []).map(Number);
      if (!counts.length) return;
      const maxCount = Math.max(...counts, 1);
      const left = 34;
      const top = 20;
      const chartW = w - left - 14;
      const chartH = h - top - 34;
      drawFrame(ctx, left, top, chartW, chartH);
      const bw = chartW / counts.length;
      for (let i = 0; i < counts.length; i += 1) {
        const barH = chartH * counts[i] / maxCount;
        ctx.fillStyle = i < counts.length / 2 ? "#b94e4e" : "#246a8f";
        ctx.globalAlpha = 0.82;
        ctx.fillRect(left + i * bw, top + chartH - barH, Math.max(1, bw - 1), barH);
      }
      ctx.globalAlpha = 1;
      ctx.fillStyle = "#65716c";
      ctx.font = "12px system-ui, sans-serif";
      ctx.fillText("-1", left, h - 10);
      ctx.fillText("1", w - 24, h - 10);
    }

    function drawSpectrum(canvas, spectrum) {
      const { ctx, w, h } = setupCanvas(canvas);
      const values = (spectrum?.mass || []).map(Number).filter(Number.isFinite);
      if (!values.length) return;
      const left = 34;
      const top = 20;
      const chartW = w - left - 14;
      const chartH = h - top - 34;
      const maxValue = Math.max(...values, 1e-9);
      drawFrame(ctx, left, top, chartW, chartH);
      ctx.strokeStyle = "#1f7a63";
      ctx.lineWidth = 2;
      ctx.beginPath();
      values.forEach((value, i) => {
        const x = left + chartW * i / Math.max(1, values.length - 1);
        const y = top + chartH - chartH * value / maxValue;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.stroke();
      ctx.fillStyle = "#1f7a63";
      values.forEach((value, i) => {
        const x = left + chartW * i / Math.max(1, values.length - 1);
        const y = top + chartH - chartH * value / maxValue;
        ctx.beginPath();
        ctx.arc(x, y, 2, 0, Math.PI * 2);
        ctx.fill();
      });
      ctx.fillStyle = "#65716c";
      ctx.font = "12px system-ui, sans-serif";
      ctx.fillText("eigenvalue mass", left, 14);
    }

    function drawHeatmap(canvas, heatmap) {
      const { ctx, w, h } = setupCanvas(canvas);
      const values = heatmap?.values || [];
      const n = values.length;
      if (!n) return;
      const pad = 16;
      const size = Math.min(w, h) - pad * 2;
      const cell = size / n;
      const x0 = (w - size) / 2;
      const y0 = (h - size) / 2;
      for (let i = 0; i < n; i += 1) {
        for (let j = 0; j < n; j += 1) {
          const v = Math.max(-1, Math.min(1, Number(values[i][j]) || 0));
          if (v >= 0) {
            const a = Math.round(255 - 125 * v);
            ctx.fillStyle = `rgb(${a},${a},255)`;
          } else {
            const a = Math.round(255 + 125 * v);
            ctx.fillStyle = `rgb(255,${a},${a})`;
          }
          ctx.fillRect(x0 + j * cell, y0 + i * cell, Math.ceil(cell), Math.ceil(cell));
        }
      }
      ctx.strokeStyle = "#5d6b63";
      ctx.strokeRect(x0, y0, size, size);
    }

    function drawHistory(canvas, history) {
      const { ctx, w, h } = setupCanvas(canvas);
      const values = (history || []).map((point) => ({ step: Number(point.step), loss: Number(point.loss) })).filter((point) => Number.isFinite(point.step) && Number.isFinite(point.loss));
      if (!values.length) return;
      const left = 38;
      const top = 18;
      const chartW = w - left - 14;
      const chartH = h - top - 32;
      const minStep = Math.min(...values.map((point) => point.step));
      const maxStep = Math.max(...values.map((point) => point.step));
      const minLoss = Math.min(...values.map((point) => point.loss));
      const maxLoss = Math.max(...values.map((point) => point.loss));
      const stepSpan = Math.max(maxStep - minStep, 1e-9);
      const lossSpan = Math.max(maxLoss - minLoss, 1e-9);
      drawFrame(ctx, left, top, chartW, chartH);
      ctx.strokeStyle = "#bd812d";
      ctx.lineWidth = 2;
      ctx.beginPath();
      values.forEach((point, i) => {
        const x = left + (point.step - minStep) / stepSpan * chartW;
        const y = top + chartH - (point.loss - minLoss) / lossSpan * chartH;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.stroke();
      ctx.fillStyle = "#65716c";
      ctx.font = "12px system-ui, sans-serif";
      ctx.fillText(`loss ${fmt(minLoss)}-${fmt(maxLoss)}`, left, 13);
    }

    function drawLegend() {
      for (const el of document.querySelectorAll(".legend[data-labels]")) {
        const labels = JSON.parse(el.dataset.labels);
        el.innerHTML = labels.map((label, idx) => `<span><i class="swatch" style="background: ${PALETTE[idx % PALETTE.length]}"></i>${esc(label)}</span>`).join("");
      }
    }

    function init() {
      renderTopNav();
      renderOverview();
      renderPreprocessing();
      renderRuns();
      requestAnimationFrame(drawAll);
    }

    init();
  </script>
</body>
</html>
"""


def slug_from_summary(path: Path) -> str:
    name = path.stem
    return name.removesuffix("_summary")


def short_title(title: str) -> str:
    title = title.replace("Exp2 Frozen LLM: ", "Exp2 ")
    title = title.replace("Exp0 ", "E0 ")
    title = title.replace("Exp1 ", "E1 ")
    return title


def default_meta(slug: str) -> dict[str, Any]:
    clean = slug.replace("_", " ")
    return {
        "title": clean.title(),
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
    run = {
        "slug": slug,
        "title": meta["title"],
        "short_title": short_title(meta["title"]),
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
    return run


def ordered(paths: list[Path]) -> list[Path]:
    order = {slug: idx for idx, slug in enumerate(RUN_ORDER)}
    return sorted(paths, key=lambda path: (order.get(slug_from_summary(path), 10_000), slug_from_summary(path)))


def build_dashboard(summary_paths: list[Path], report_dir: Path) -> str:
    runs = [load_run(path, report_dir) for path in ordered(summary_paths)]
    payload = {
        "generated_from": [relative_to(path, report_dir) for path in ordered(summary_paths)],
        "runs": runs,
    }
    data = json.dumps(payload, separators=(",", ":"), ensure_ascii=True)
    data = data.replace("</", "<\\/")
    return HTML_TEMPLATE.replace("__REPORT_DATA__", data)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a self-contained all-results dashboard for sphere-JEPA report summaries.")
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
    out.write_text(build_dashboard(summaries, report_dir))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
