---
benchmark: libero_plus
display_name: LIBERO-Plus
paper_url: https://arxiv.org/abs/2510.13626
metric:
  name: success_rate
  unit: '%'
  range:
  - 0
  - 100
  higher_is_better: true
suites:
- camera
- robot
- language
- light
- background
- noise
- layout
detail_notes: "LIBERO-Plus (<a href='https://arxiv.org/abs/2510.13626'>2510.13626</a>). Robustness benchmark applying 7 perturbation dimensions (Camera, Robot, Language, Light, Background, Noise, Layout) to LIBERO tasks. Models are trained on standard LIBERO and evaluated zero-shot under perturbations. <strong>overall_score = arithmetic mean of the 7 perturbation dimensions</strong>. Entries reporting fewer than 7 dimensions have overall_score = null."
aggregation:
  container: suite_scores
  keys:
  - camera
  - robot
  - language
  - light
  - background
  - noise
  - layout
---

**Standard**: LIBERO-Plus robustness benchmark ([2510.13626](https://arxiv.org/abs/2510.13626)) — models trained on standard LIBERO, evaluated zero-shot across 7 perturbation dimensions (camera, robot, language, light, background, noise, layout); `overall_score` = arithmetic mean of all 7 dimensions.

## Scoring
- `overall_score`: arithmetic mean of `suite_scores.camera`, `robot`, `language`, `light`, `background`, `noise`, `layout`; `null` if any of the 7 is missing.
- `suite_scores`: canonical keys `camera`, `robot`, `language`, `light`, `background`, `noise`, `layout`.
- `task_scores`: not used.

## Checks
- Are all 7 perturbation dimensions present? Missing any → `null`.
- Is `weight_type` set correctly? `shared` for zero-shot models (LIBERO-trained), `finetuned` for models trained on LIBERO-Plus data.
- Is this LIBERO-Plus and not LIBERO-Pro (which uses different perturbations)?

## Methodology axes (record in `notes`, do not null)
- Reduced training data: some papers (e.g. JEPA-VLA) use 1/10 LIBERO. Record the training-data fraction when disclosed.
