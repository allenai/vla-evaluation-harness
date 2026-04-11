---
benchmark: libero_plus
---

## Protocol

- Robustness benchmark ([2510.13626](https://arxiv.org/abs/2510.13626)) with **7 perturbation dimensions**: Camera, Robot, Language, Light, Background, Noise, Layout.
- Models are trained on standard LIBERO and evaluated **zero-shot** under perturbations.
- `overall_score` = arithmetic mean of **all 7** perturbation dimensions. Always include `suite_scores`. Entries with fewer than 7 dimensions must set `overall_score = null`.
- `weight_type`: `"shared"` for zero-shot models (LIBERO-trained); `"finetuned"` for models trained on LIBERO-Plus data.
- Some papers (e.g. JEPA-VLA) use reduced training data (1/10 LIBERO) — record in `notes`.

## Risky Patterns

- Are all 7 perturbation dimensions (`camera`, `robot`, `language`, `light`, `background`, `noise`, `layout`) present? If any missing → `overall_score` must be `null`.
