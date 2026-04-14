---
benchmark: _global
---

## Risky Patterns

These apply to every benchmark entry:

- Is the reported score a re-run by this paper, or a cited baseline copied from another work? (→ affects `reported_paper` and whether to create a new model key)
- Is the model a reproduction/retrain by this paper, a download-and-run of an existing checkpoint, or an entirely new model? (→ affects the model key suffix and `weight_type`)
- Is the paper using the same version of the benchmark as the registry (v1 vs v2, patch revisions, task subset changes)?
- Is `weight_type` correctly inferred? (`shared` = same checkpoint across benchmarks, `finetuned` = trained specifically on this benchmark's data)
- Does the paper's reported protocol match the benchmark's standard protocol? If not, the entry's `overall_score` must be `null`.
