# {Model Name} Reproduction Report

> **This file is a template.** Copy it to `{model-name}.md` and fill in the sections.
> Delete directive comments (`<!-- ... -->`) when done.

{One-line description of the model and what was evaluated.}

---

## Model Info

| Field | Value |
|-------|-------|
| **Model** | {model name and version} |
| **Architecture** | {architecture, key inference params} |
| **Loading** | {how the model is loaded, e.g. `from_pretrained(...)` call} |

<!-- If there were code modifications shared across all benchmarks, document them here.
     Otherwise delete this subsection. -->

### Common Code Modifications

**{Title}**: {problem → fix → impact}

---

## {Benchmark Name}

<!-- Repeat this section for each benchmark evaluated with this model. -->

| Field | Value |
|-------|-------|
| **Status** | {`complete` / `draft`} |
| **Date** | {YYYY-MM-DD} |
| **Harness commit** | {git commit SHA or tag} |
| **Benchmark** | {benchmark name, task count, episode count} |
| **Hardware** | {GPU, node setup} |
| **Action space** | {dimensionality, chunk size, max steps} |

### How to Reproduce

```bash
# 1. Start model server
vla-eval serve --config {configs/model_servers/....yaml}

# 2. Run evaluation
vla-eval run --config {configs/....yaml}
```

<!-- If benchmark-specific code modifications were needed, add: -->

### Benchmark-Specific Fixes

**{Title}**: {problem → fix → impact}

### Results

| Task / Suite | Score | Reference | Diff | Verdict |
|-------------|:-----:|:---------:|:----:|:-------:|
| {task 1} | **{score}** | {ref} | {diff} | {verdict} |
| **Average** | **{score}** | **{ref}** | **{diff}** | **{verdict}** |

<!-- Include wall-clock time. Put per-task breakdowns in <details> blocks. -->

### Discussion

- {Reproducibility verdict and rationale}

---

## Changelog

| Date | Benchmark | Change |
|------|-----------|--------|
| {YYYY-MM-DD} | {benchmark} | Initial evaluation |
