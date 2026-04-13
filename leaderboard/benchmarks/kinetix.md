---
benchmark: kinetix
---

## Protocol

- **Not the Kinetix simulator** — it's the 12-task eval protocol from the RTC paper ([2506.07339](https://arxiv.org/abs/2506.07339)). State-based, no vision/language.
- Scores depend on `(inference_delay d, execution_horizon e)` settings. Always record both in `notes`.
- Entries at different `d` values are **not directly comparable** (e.g., d=0 scores ~11pp higher than d=4 for the same method). Prefer grouping by `d` when comparing.

## Risky Patterns

- Are `inference_delay d` and `execution_horizon e` recorded in `notes`? Different `d` values are not directly comparable.
