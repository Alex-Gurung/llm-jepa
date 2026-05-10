# GPU Report 2026-05-10

Main viewer:

- [Sphere cap ablation HTML](html/replicate_llm_jepa_sphere_ablation.html)
- [Sphere cap ablation summary JSON](json/replicate_llm_jepa_sphere_ablation_summary.json)
- [Embedding diagnostics seed 37](html/embedding_diagnostics_seed37.html)
- [Embedding diagnostics seed 23](html/embedding_diagnostics_seed23.html)

Plot assets:

- `assets/sphere_ablation/`
- `assets/embedding_diagnostics/`

The ablation viewer is intentionally live-updatable. Re-run:

```bash
uv run --extra experiments --no-sync python experiments/viewer/build_sphere_ablation_report.py --report-dir reports/gpu_2026-05-10
```

after new eval JSONL files land under `results/replicate_llm_jepa_sphere_ablation/eval/`.
