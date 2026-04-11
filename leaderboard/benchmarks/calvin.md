---
benchmark: calvin
---

## Protocol

- **Standard protocol**: ABC→D split (train on A/B/C, eval on D), 1000 eval chains. ABCD→D inflates scores — do not add.
- Metric: avg completed subtasks in chain of 5 (0–5), not success rate.
- Record deviations from 1000 chains in `notes`.

## Risky Patterns

- Is the training split `ABC→D` (standard) or `ABCD→D` (inflated — REJECT)? `D→D` is also REJECT.
- Is the evaluation over 1000 chains, and is the metric `avg_len` (0–5) rather than success rate?
