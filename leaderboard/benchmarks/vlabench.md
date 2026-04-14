---
benchmark: vlabench
---

**Standard**: VLABench 6-track evaluation system ([OpenMOSS/VLABench](https://github.com/OpenMOSS/VLABench)) with IS (Intention Score) and PS (Progress Score) metrics; `overall_score` = arithmetic mean of Tracks 1–4 PS (`in_dist_PS`, `cross_category_PS`, `commonsense_PS`, `semantic_instruction_PS`).

## Scoring
- `overall_score`: mean of the four Track 1–4 PS values; `null` if any is missing OR if the entry uses the legacy pre-track-system protocol.
- `suite_scores`: canonical keys for Tracks 1–4 PS & IS — `in_dist_PS`, `in_dist_IS`, `cross_category_PS`, `cross_category_IS`, `commonsense_PS`, `commonsense_IS`, `semantic_instruction_PS`, `semantic_instruction_IS`. Optional Track 5 (`cross_task`, skill transfer) and Track 6 (`unseen_texture_PS` / `unseen_texture_IS`) are supplementary. Pre-track-system entries (2412.18194) use legacy keys (`seen_base`, `unseen_commonsense`, etc.) with `overall_score: null`.
- `task_scores`: not used — metrics are reported at the track level only.

## Checks
- Is `overall_score` the mean of Tracks 1–4 PS (NOT IS)? IS-based aggregates must be nulled.
- Are canonical suite keys used (`in_dist_PS`, `cross_category_PS`, `commonsense_PS`, `semantic_instruction_PS`)?
- Is the original VLABench paper (2412.18194) entry stored with legacy keys and `overall_score: null` since it predates the 6-track system?
- Is the task set one of the official tracks? Cherry-picked tasks outside the official tracks must NOT be filed under `vlabench`.

## Methodology axes (record in `notes`, do not null)
- Different papers evaluating the same model produce different scores due to fine-tuning setup and eval seeds. Use separate `model` keys per source paper (e.g. `pi0_acot_vlabench`, `pi0_xvla_vlabench`). This is the benchmark's convention for handling third-party measurements explicitly.
