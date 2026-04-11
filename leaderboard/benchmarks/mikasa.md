---
benchmark: mikasa
---

## Protocol

- **Standard protocol**: 5-task VLA evaluation ([2502.10550](https://arxiv.org/abs/2502.10550)): ShellGameTouch, InterceptMedium, RememberColor3, RememberColor5, RememberColor9. Endorsed as the standard by MemoryVLA (ICLR 2026). 100 evaluation episodes per task.
- `overall_score` = arithmetic mean of 5 task success rates. Always include `task_scores`.
- Entries using non-standard task sets (e.g., ELMUR 4-task: RC3/5/9 + TakeItBack) must set `overall_score = null`. Store the paper's reported aggregate in `suite_scores.reported_avg`.
- Some scores are third-party reproductions (e.g. MemoryVLA paper). Check `notes`.

## Risky Patterns

- Is this the 5-task VLA protocol (ShellGameTouch, InterceptMedium, RememberColor3/5/9) or the ELMUR 4-task variant? The 4-task variant must have `overall_score = null`.
