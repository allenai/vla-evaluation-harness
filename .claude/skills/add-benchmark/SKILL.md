---
name: add-benchmark
description: "Add a new simulation benchmark to the VLA evaluation harness. Use this skill whenever the user wants to integrate, create, or add a new benchmark or simulation environment — e.g. 'add ManiSkill3', 'integrate OmniGibson', 'hook up a new sim'. Also use when they ask how benchmarks are structured or want to understand the benchmark interface."
---

# Add Benchmark

Integrate a new simulation benchmark into vla-eval. Benchmarks run inside Docker containers and communicate with model servers over WebSocket + msgpack.

## 1. Gather requirements

Ask the user for (if not already provided):
- **Benchmark name** (e.g. `maniskill3`)
- **Simulation framework** (e.g. MuJoCo, SAPIEN, PyBullet, Isaac Sim)
- **Key pip dependencies** needed inside Docker
- **Observation format** — cameras, resolution, whether to include proprioceptive state
- **Action space** — dimension, format (e.g. 7-DoF delta EEF + gripper)
- **Success condition** — how to detect task completion
- **Max steps per episode**

## 2. Create the benchmark module

Create `src/vla_eval/benchmarks/<name>/`:
```
src/vla_eval/benchmarks/<name>/
├── __init__.py      # empty
├── benchmark.py     # main implementation
└── utils.py         # optional helpers
```

Subclass `StepBenchmark` from `vla_eval.benchmarks.base` and implement the required methods:

```python
from typing import Any

import numpy as np

from vla_eval.benchmarks.base import StepBenchmark, StepResult
from vla_eval.specs import DimSpec
from vla_eval.types import Action, EpisodeResult, Observation, Task


class MyBenchmark(StepBenchmark):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__()
        # Accept benchmark-specific params from config YAML `params:` section.
        # Lazily import heavy deps (MuJoCo, SAPIEN) — NOT at module level,
        # because the registry resolves the class without loading the sim.
        ...

    # --- Required methods (6) ---

    def get_tasks(self) -> list[Task]:
        # Return list of task dicts. Each MUST have a "name" key.
        # May include "suite" for task filtering.
        ...

    def reset(self, task: Task) -> Any:
        # Reset env for task. Store env on self. Return initial raw observation.
        # task dict has "episode_idx" (int) injected by the orchestrator.
        ...

    def step(self, action: Action) -> StepResult:
        # action dict has "actions" key (np.ndarray from model server).
        # Return StepResult(obs, reward, done, info).
        ...

    def make_obs(self, raw_obs: Any, task: Task) -> Observation:
        # Convert raw env observation to dict for model server.
        # Convention:
        #   {"images": {"cam_name": np.ndarray HWC uint8},
        #    "task_description": str}
        # Optionally add "state": np.ndarray for proprioception.
        ...

    def get_step_result(self, step_result: StepResult) -> EpisodeResult:
        # Extract episode result from the final StepResult.
        # Must return at least {"success": bool}.
        ...

    # --- Optional overrides ---

    def check_done(self, step_result: StepResult) -> bool:
        # Default: step_result.done. Override for custom termination logic.
        return step_result.done

    def get_action_spec(self) -> dict[str, DimSpec]:
        # Declare the action format this benchmark's env consumes.
        # The orchestrator compares this against the model server's spec
        # and warns on mismatches — catching convention bugs early.
        ...

    def get_observation_spec(self) -> dict[str, DimSpec]:
        # Declare the observation format this benchmark produces.
        ...

    def get_metric_keys(self) -> dict[str, str]:
        # Declare which metrics from get_step_result() to aggregate.
        # Default: {"success": "mean"} (= success rate).
        # Aggregation options: "mean", "sum", "max", "min".
        return {"success": "mean"}

    def get_metadata(self) -> dict[str, Any]:
        # Return {"max_steps": N} for benchmark default.
        return {}

    def cleanup(self) -> None:
        # Release resources (envs, renderers). Called at end of evaluation.
        ...
```

### Async bridge (automatic)

`StepBenchmark` auto-bridges your sync methods to the async `Benchmark` parent interface. The orchestrator/runners call the async methods (`start_episode`, `apply_action`, `get_observation`, `is_done`, `get_result`) — you never implement those directly.

### Key patterns from existing implementations

- **Lazy imports**: Put heavy sim imports (`torch`, `robosuite`, `sapien`) inside methods, not at module level.
- **Env reuse**: LIBERO reuses envs across episodes of the same task. SimplerEnv creates fresh envs per episode. Choose based on your sim's reset semantics.
- **Action processing**: Model servers output raw continuous actions. Convert to sim-specific format in `step()` (e.g. discretize gripper, convert euler→axis-angle).
- **Image preprocessing**: Handle non-standard images (flipped, wrong resolution) in `make_obs()`.
- **EGL headless rendering**: Add `os.environ.setdefault("PYOPENGL_PLATFORM", "egl")` at module top if the sim uses OpenGL.

## 3. Create config YAML

Create `configs/<name>_eval.yaml`:

```yaml
server:
  url: "ws://localhost:8000"

docker:
  image: ghcr.io/allenai/vla-evaluation-harness/<name>:latest
  env: []     # e.g. ["NVIDIA_DRIVER_CAPABILITIES=all"] for Vulkan
  volumes: [] # e.g. ["/path/to/data:/data:ro"]

output_dir: "./results"

benchmarks:
  - benchmark: "vla_eval.benchmarks.<name>.benchmark:MyBenchmark"
    mode: sync
    episodes_per_task: 50
    params:
      # All keys here passed as **kwargs to MyBenchmark.__init__()
      suite: default
      seed: 7
```

- `benchmark` field: `module.path:ClassName` import string
- `params`: arbitrary dict passed to constructor — no schema enforcement
- `max_steps`: omit to use `get_metadata()["max_steps"]`, or set explicitly to override

## 4. Create Dockerfile

Create `docker/Dockerfile.<name>`:

```dockerfile
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

# Install benchmark-specific dependencies
RUN pip install <benchmark-packages>

# Copy benchmark code
COPY src/vla_eval/benchmarks/<name>/ src/vla_eval/benchmarks/<name>/
```

All benchmark Dockerfiles inherit from the base image (`docker/Dockerfile.base`) which already installs the harness. Your Dockerfile only needs to add benchmark-specific dependencies and code.

## 5. Register in build/push scripts

Add to the `BENCHMARKS` array in `docker/build.sh` and the `IMAGES` array in `docker/push.sh`:

```bash
BENCHMARKS=(... <name> ...)
```

Underscores in names are auto-converted to hyphens for Docker image names (e.g. `mikasa_robo` → `mikasa-robo`).

## 6. Verify

```bash
make check                                    # lint + format + type check
make test                                     # existing tests still pass
vla-eval test --validate                      # validate all config import strings
vla-eval test -c configs/<name>_eval.yaml     # smoke-test (1 episode, EchoModelServer, no GPU needed — requires Docker + image)
```

**Don't add `tests/test_<name>_benchmark.py` with mocked sim modules.**
`tests/` is for harness mechanics, not per-sim integration.  Fake
`omnigibson` / `sapien` / `mujoco` modules drift from upstream each
release and miss the real bugs (import paths, action encoding,
physics determinism).  Verify via the smoke test above.

## Reference implementations

| Benchmark | File | Key patterns |
|---|---|---|
| LIBERO | `benchmarks/libero/benchmark.py` | MuJoCo tabletop, env reuse, suite-specific max_steps, image flip |
| SimplerEnv | `benchmarks/simpler/benchmark.py` | SAPIEN+Vulkan, new env per episode, euler→axis-angle conversion |
| CALVIN | `benchmarks/calvin/benchmark.py` | PyBullet, chained subtasks, delta actions, hardcoded normalization |
