---
benchmark: libero
---

## Protocol

- **Standard protocol**: 4-suite average (`spatial`, `object`, `goal`, `10`). Always include `suite_scores`. A 5th suite (`90`) exists but many papers skip it.
- `overall_score` = arithmetic mean of the **4 standard suites only** (`spatial`, `object`, `goal`, `10`). Do NOT include `90` in the overall mean even when reported — store it in `suite_scores.libero_90` only. Entries reporting only a subset of the 4 standard suites must set `overall_score = null`.
- LIBERO-Plus, LIBERO-Pro and LIBERO-Mem are **separate benchmarks**.

## Risky Patterns

- Is `overall_score` the mean of the 4 standard suites (`spatial`, `object`, `goal`, `10`) and does it exclude `libero_90`?
- Does the entry use the standard 50-demo training budget, or a reduced data setup that must be noted?
