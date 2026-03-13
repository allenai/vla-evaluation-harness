# Design Philosophy

## Freshness — Keep benchmarks meaningful

Benchmarks are living things. As the community advances, existing benchmarks saturate and lose the ability to discriminate between models. This project does not fall behind.

- LIBERO is already saturated — many models converge to high success rates, and multiple studies have exposed its limitations. We actively include improved successors like LIBERO-Pro.
- When a new benchmark gains traction in the community, we add an adapter promptly. Conversely, benchmarks that lose discriminative value are deprecated.
- The default benchmark suite is periodically updated to reflect "what constitutes a meaningful evaluation right now."

## Convenience — Zero-friction evaluation

Researchers' time should go to model development, not environment setup.

- Evaluate a model across diverse robot benchmarks without manually reconciling complex environment configurations or interface mismatches between benchmarks.
- Docker-based environment isolation eliminates dependency conflicts at the root. A single CLI command starts evaluation.
- The model server interface is kept minimal — a few lines of modification to existing model code is all it takes to integrate.

## Layered Abstraction — Generality and convenience coexist

Two goals:

1. **Generality**: Define a framework that accommodates future research paradigms and techniques without major changes.
2. **Convenience**: High generality means few constraints, which in turn means less convenience for researchers who "just want to get it running quickly."

These cannot coexist in a single layer. The solution is multiple layers — from a low-level layer with minimal constraints (high generality) to a high-level layer with more constraints but easier usage (high convenience).

This principle applies across the entire project — server interface, client interface, benchmark adapters, and execution tools all follow the same philosophy. At every layer, the lower layer imposes only minimal constraints so that "this framework can't support my use case" never happens.

## Quality — Verify both code and results

To trust the harness's results, the harness itself must be trustworthy.

- Code quality: strict type checking, linting, and formatting.
- Testing: tests for critical paths — protocol serialization, adapter correctness, metric computation.
- CI/CD: automated validation pipeline on every PR. Integration tests required when adding benchmark adapters.
- Result verification: maintain reference scores for known model+benchmark combinations and regression-test that framework changes don't affect results.

## Reproducibility — Same conditions, same results

If evaluation results vary by environment, the benchmark is meaningless.

- Pin exact simulation environment versions via Docker images.
- Explicitly record seed management, episode composition, and evaluation protocol so that anyone, anywhere, can reproduce identical results.
- Automatically record the configuration used for each evaluation run (benchmark version, seeds, episode list, etc.) alongside the results.

## Openness — Grow with the community

The framework's value is proportional to the breadth of benchmarks it supports. Expanding that breadth is not the maintainers' job alone.

- Aim for a structure where external contributors can easily add new benchmarks and model adapters.
- Clearly document the interfaces required for adding benchmark adapters, and provide contribution guides and templates to lower the barrier to entry.
- Provide example model servers and adapters as reference implementations — a clear starting point that says "follow this."
