---
benchmark: vlabench
---

## Protocol

- Official 6-track evaluation system ([OpenMOSS/VLABench](https://github.com/OpenMOSS/VLABench)):
  - Track 1: `in_distribution` — task learning ability
  - Track 2: `cross_category` — object generalization
  - Track 3: `common_sense` — common sense understanding
  - Track 4: `semantic_instruction` — complex instruction understanding
  - Track 5: `cross_task` — skill transfer (kept open, not included in standard)
  - Track 6: `unseen_texture` — visual robustness (optional)
- **Two metrics**: IS (Intention Score, approached correct object) and PS (Progress Score, task completion). IS ≥ PS always.
- **Leaderboard standard**: `overall_score` = **Track 1-4 PS average**. Track 5-6 and IS values go in `suite_scores` as supplementary data.
- **Canonical suite_scores keys**: Use the track-based naming: `in_dist_IS`, `in_dist_PS`, `cross_category_IS`, `cross_category_PS`, `commonsense_IS`, `commonsense_PS`, `semantic_instruction_IS`, `semantic_instruction_PS`, `unseen_texture_IS`, `unseen_texture_PS`. Pre-track-system entries (2412.18194) use legacy keys (`seen_base`, `unseen_commonsense`, etc.) with `overall_score: null`.
- Original VLABench paper (2412.18194) uses a pre-track-system IS-based protocol (seen/unseen × base/commonsense). These entries have `overall_score: null`.
- Non-standard task subsets (e.g. cherry-picked tasks outside the official tracks) must NOT be filed under `vlabench`.
- Different papers evaluating the same model produce different scores due to fine-tuning setup and eval seeds. Use separate `model` keys per source paper (e.g. `pi0_acot_vlabench`, `pi0_xvla_vlabench`).

## Risky Patterns

- Is `overall_score` the Track 1–4 PS average (not IS)? Are the canonical suite keys used (`in_dist_PS`, `cross_category_PS`, `commonsense_PS`, `semantic_instruction_PS`)?
