# Skill: add-benchmark

Add a new simulation benchmark to the VLA evaluation harness.

## Trigger

User asks to add/create/integrate a new benchmark (e.g. "add ManiSkill3 benchmark", "integrate OmniGibson").

## Steps

### 1. Gather Requirements

Ask the user (if not already provided):
- **Benchmark name** (e.g. `maniskill3`)
- **Simulation framework** (e.g. MuJoCo, SAPIEN, PyBullet, Isaac Sim)
- **Key dependencies** (pip packages needed inside Docker)
- **Observation format** (which cameras, image resolution, whether to include proprioceptive state)
- **Action space** (dimension, format — e.g. 7-DoF delta EEF + gripper)
- **Success condition** (how to detect task completion)
- **Max steps per episode** (if fixed or per-task)

### 2. Create Benchmark Module

Create `src/vla_eval/benchmarks/<name>/`:

```
src/vla_eval/benchmarks/<name>/
├── __init__.py      # empty
├── benchmark.py     # main implementation
└── utils.py         # optional helpers
```

**`benchmark.py`** must subclass `Benchmark` from `vla_eval.benchmarks.base` and implement **6 required methods**:

```python
from vla_eval.benchmarks.base import Benchmark, StepResult

class MyBenchmark(Benchmark):
    def __init__(self, **kwargs):
        # Accept benchmark-specific params from config YAML `params:` section.
        # Lazily import heavy deps (MuJoCo, SAPIEN, etc.) — NOT at module level.
        ...

    def get_tasks(self) -> list[dict[str, Any]]:
        # Return list of task dicts. Each MUST have a "name" key.
        # May include "suite" for task filtering.
        ...

    def reset(self, task: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
        # Reset env for task. Returns (env_handle, initial_obs_dict).
        # env_handle is opaque — passed back to step().
        # obs_dict should be the output of make_obs().
        # task dict has "episode_idx" (int) injected by orchestrator.
        ...

    def step(self, env: Any, action: dict[str, Any]) -> StepResult:
        # action dict has "actions" key (np.ndarray from model server).
        # Return StepResult(obs, reward, done, info).
        ...

    def make_obs(self, raw_obs: Any, task: dict[str, Any]) -> dict[str, Any]:
        # Convert raw env observation to dict for model server.
        # Convention: {"images": {"cam_name": np.ndarray HWC uint8},
        #              "task_description": str}
        # Optionally add "states": np.ndarray for proprioception.
        ...

    def is_done(self, step_result: StepResult) -> bool:
        # Return True to end the episode.
        ...

    def get_result(self, step_result: StepResult) -> dict[str, Any]:
        # Must return at least {"success": bool}.
        ...

    def get_metadata(self) -> dict[str, Any]:
        # Optional. Return {"max_steps": N} for benchmark default.
        ...
```

### Key Patterns (from existing implementations)

- **Lazy imports**: Put heavy sim imports (torch, robosuite, sapien) inside methods, not at module top. This allows the registry to resolve the class without loading the sim.
- **Env reuse**: LIBERO reuses env across episodes of the same task. SimplerEnv creates a fresh env per episode. Choose based on the sim's reset semantics.
- **Action processing**: Model servers output raw continuous actions. The benchmark must convert to sim-specific format (e.g. discretize gripper, convert euler→axis-angle).
- **Image preprocessing**: If the sim outputs non-standard images (flipped, wrong resolution), handle in `make_obs()`.
- **EGL headless rendering**: Set `os.environ.setdefault("PYOPENGL_PLATFORM", "egl")` at module top if the sim uses OpenGL.

### 3. Create Config YAML

Create `configs/<name>_eval.yaml`:

```yaml
server:
  url: "ws://localhost:8000"

docker:
  image: <name>
  env: []     # e.g. ["NVIDIA_DRIVER_CAPABILITIES=all"] for Vulkan
  volumes: [] # e.g. ["/path/to/data:/data:ro"]

output_dir: "./results"

benchmarks:
  - benchmark: "vla_eval.benchmarks.<name>.benchmark:MyBenchmark"
    mode: sync
    episodes_per_task: 50
    params:
      # All keys here are passed as **kwargs to MyBenchmark.__init__()
      suite: default
      seed: 7
```

- `benchmark` field: full import string in `module.path:ClassName` format
- `params`: arbitrary dict passed to constructor — no schema enforcement
- `max_steps`: omit to use `get_metadata()["max_steps"]`, or set explicitly to override

### 4. Create Dockerfile

Create `docker/Dockerfile.<name>`:

```dockerfile
FROM <base_image>

# Install harness
WORKDIR /workspace
COPY pyproject.toml README.md ./
COPY src/ src/
ARG HARNESS_VERSION=0.0.0
ENV SETUPTOOLS_SCM_PRETEND_VERSION=${HARNESS_VERSION}
RUN pip install .

COPY configs/ configs/

ENTRYPOINT ["vla-eval"]
CMD ["run", "--config", "/workspace/configs/<name>_eval.yaml"]
```

### 5. Register in Build/Push Scripts

Add the new benchmark to the arrays in `docker/build.sh` and `docker/push.sh`:

- `BENCHMARKS=(... <name> ...)` in `docker/build.sh`
- `IMAGES=(... <name> ...)` in `docker/push.sh`

If the name contains underscores (e.g. `mikasa_robo`), the scripts automatically convert them to hyphens for the Docker image name (`mikasa-robo`).

### 6. Verify

1. Run `make check` — lint + format + type check
2. Run `make test` — ensure existing tests still pass
3. Run `vla-eval test --validate` — validate all config import strings (including the new one)
4. Run `vla-eval test -c configs/<name>_eval.yaml` — smoke-test the benchmark (requires Docker + the benchmark image; runs 1 episode with an EchoModelServer, no real model or GPU needed)

### Reference Implementations

- **LIBERO** (`benchmarks/libero/benchmark.py`): MuJoCo tabletop, env reuse, suite-specific max_steps, image flip preprocessing
- **SimplerEnv** (`benchmarks/simpler/benchmark.py`): SAPIEN+Vulkan, new env per episode, Euler→axis-angle action conversion
- **CALVIN** (`benchmarks/calvin/benchmark.py`): PyBullet, chained subtasks, delta actions, hardcoded normalization stats

