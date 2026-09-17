"""Typed configuration dataclasses for YAML config files.

Converts raw ``dict[str, Any]`` from YAML into typed objects with defaults
and validation.  Each dataclass has a ``from_dict`` classmethod for
construction and an ``to_dict`` method for serialization (e.g. into result
JSON).
"""

from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, MutableMapping

logger = logging.getLogger(__name__)


def _parse_paced(data: dict[str, Any]) -> bool:
    """Parse pacing config.

    Accepts ``paced: bool`` or ``pace: 1.0`` (real-time).  For max speed,
    use ``paced: false``.  Numeric ``pace`` values other than 1.0 are
    rejected — the exact multiplier is misleading because actual speed is
    bounded by env.step time.
    """
    has_paced = "paced" in data
    has_pace = "pace" in data
    if has_pace:
        pace = float(data["pace"])
        if pace != 1.0:
            raise ValueError(
                f"pace: {pace} is not supported — actual speed is bounded by env.step time, "
                f"not the pace multiplier. Use 'paced: false' for max speed, or 'pace: 1.0' for real-time."
            )
        if has_paced and not data["paced"]:
            raise ValueError("Conflicting config: pace: 1.0 (real-time) with paced: false (max speed).")
    if has_paced:
        return bool(data["paced"])
    return True  # default: real-time


@dataclass
class ServerConfig:
    """Model server connection settings.

    Attributes:
        url: WebSocket URL of the model server.
        timeout: Seconds to wait for each server response in ``Connection.act()``.
    """

    url: str = "ws://localhost:8000"
    timeout: float = 30.0

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> ServerConfig:
        if not data:
            return cls()
        return cls(
            url=data.get("url", cls.url),
            timeout=float(data.get("timeout", cls.timeout)),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BuildConfig:
    """``docker.build``: how to build ``docker.image``; ``context`` is made absolute by the config loader."""

    context: str
    dockerfile: str = "Dockerfile"

    @property
    def dockerfile_path(self) -> str:
        """``dockerfile`` resolved against ``context`` (compose semantics)."""
        return str(Path(self.context, self.dockerfile))

    @property
    def default_image(self) -> str:
        """Tag used when ``docker.image`` is omitted: ``<parent>-<dir>:vla-eval``, like compose."""
        ctx = Path(self.context).resolve()
        return re.sub(r"[^a-z0-9._-]", "-", f"{ctx.parent.name}-{ctx.name}".lower()).strip("-") + ":vla-eval"

    @classmethod
    def from_value(cls, value: Any) -> BuildConfig | None:
        if value is None:
            return None
        if isinstance(value, str):
            return cls(context=value)
        if isinstance(value, dict) and value.get("context"):
            return cls(context=str(value["context"]), dockerfile=str(value.get("dockerfile") or "Dockerfile"))
        raise ValueError("docker.build must be a context path or {context, dockerfile}")


@dataclass
class DockerConfig:
    """Docker execution settings.

    Attributes:
        image: Docker image name.  ``None`` means run without Docker.
        volumes: Extra ``-v`` bind-mount strings.
        env: Extra ``-e`` environment variable strings.
        cpus: CPU range for benchmark containers (e.g. ``"0-31"``).
            ``None`` uses all host CPUs.  Cores are partitioned across shards.
        gpus: GPU devices for benchmark containers (e.g. ``"0,1"``).
            ``None`` or ``"all"`` exposes all GPUs.  Devices are round-robin
            distributed across shards.
        user: Optional ``docker run --user`` value. ``None`` (default) →
            no flag (image-default user). ``"host"`` → ``$(id -u):$(id -g)``.
            ``"<uid>:<gid>"`` → explicit pin.
        runtime: ``"docker"`` (default) or ``"charliecloud"``; ``--runtime`` / ``$VLA_EVAL_RUNTIME`` override.
        build: Compose-style ``{context, dockerfile}`` (or a context string). ``image`` is the tag
            (default ``<parent>-<dir>:vla-eval``); built when missing locally, or always with ``--build``.
    """

    image: str | None = None
    volumes: list[str] = field(default_factory=list)
    env: list[str] = field(default_factory=list)
    cpus: str | None = None
    gpus: str | None = None
    user: str | None = None
    runtime: str | None = None
    build: BuildConfig | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> DockerConfig:
        if not data:
            return cls()
        build = BuildConfig.from_value(data.get("build"))
        return cls(
            image=data.get("image") or (build.default_image if build else None),
            volumes=data.get("volumes", []),
            env=data.get("env", []),
            cpus=data.get("cpus"),
            gpus=data.get("gpus"),
            user=data.get("user") or None,
            runtime=data.get("runtime") or None,
            build=build,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EvalConfig:
    """Single evaluation entry in the ``benchmarks`` list.

    Attributes:
        benchmark: Import string in ``module.path:ClassName`` format.
        mode: Execution mode — ``"sync"`` (env waits for inference) or ``"live"``
            (env advances on a clock, decoupled from inference; real-time at pace=1).
        name: Display name.  Defaults to class name from ``benchmark``.
        subname: Disambiguator appended to name (e.g. suite name).  Use when
            multiple benchmarks in the same config share a class.
        episodes_per_task: Number of episodes to run per task.
        max_steps: Step limit per episode.  ``None`` → use benchmark metadata.
        max_tasks: Cap on number of tasks.  ``None`` → run all tasks.
        tasks: Filter to specific task names/suites.  ``None`` → all tasks.
        params: Benchmark-specific kwargs passed to the constructor.
        recording: Optional dict overriding per-episode recording defaults.
            Keys: ``output_dir``, ``filename_stem``, ``record_video``,
            ``record_step``, ``video_fps``. The benchmark is *not* involved
            in this — the orchestrator builds the recorder from this dict
            and the task dict directly; filename templates also receive
            ``benchmark_safe_name``, ``task_idx``, and ``episode_id``.
            Absent → record step rows and episode results with video off.
            ``vla-eval run --no-save`` disables recording entirely.
    """

    benchmark: str = ""
    mode: str = "sync"
    name: str | None = None
    subname: str | None = None
    episodes_per_task: int = 1
    max_steps: int | None = None
    max_tasks: int | None = None
    tasks: list[str] | None = None
    params: dict[str, Any] = field(default_factory=dict)
    recording: dict[str, Any] | None = None
    # Live-mode params (used when mode starts with "live"); pace 1.0 = real-time
    hz: float = 10.0
    # Throughput testing mode: relaxes benchmark constraints (e.g. initial state reuse)
    throughput_mode: bool = False
    # Whether to pace the step loop to real-time (True=real-time, False=max speed)
    paced: bool = True
    # Wait for first action before starting step loop (sanity check: should match sync)
    wait_first_action: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvalConfig:
        benchmark_path = data.get("benchmark")
        if not benchmark_path:
            raise ValueError("Config error: 'benchmark' field is required and cannot be empty.")
        recording = data.get("recording")
        if recording is not None and not isinstance(recording, dict):
            raise ValueError("Config error: 'recording' must be a mapping or null.")

        return cls(
            benchmark=benchmark_path,
            mode=data.get("mode", "sync"),
            name=data.get("name"),
            subname=data.get("subname"),
            episodes_per_task=data.get("episodes_per_task", 1),
            max_steps=data.get("max_steps"),
            max_tasks=data.get("max_tasks"),
            tasks=data.get("tasks"),
            params=data.get("params", {}),
            recording=recording,
            hz=data.get("hz", 10.0),
            throughput_mode=data.get("throughput_mode", False),
            paced=_parse_paced(data),
            wait_first_action=data.get("wait_first_action", False),
        )

    def resolved_name(self) -> str:
        """Display name: ``name``, or ``ClassName_subname``, or ``ClassName``.

        When multiple benchmarks in the same config share a class (e.g. LIBERO
        suites), set ``subname`` to disambiguate file names and merge targets.
        """
        base = self.name or self.benchmark.rsplit(":", 1)[-1]
        if self.subname:
            return f"{base}_{self.subname}"
        return base

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def merge_benchmark_overrides(config: dict[str, Any], overrides: Mapping[str, Any]) -> None:
    """Apply *overrides* to every ``benchmarks[]`` entry of a raw (pre-:class:`EvalConfig`) config.

    ``params`` merges key by key; every other key replaces the entry's value.
    Shared by ``vla_eval.run(benchmark_overrides=...)`` and ``vla-eval run``'s
    ``--param`` / ``--benchmark-field``, so the two agree by construction.

    Entry keys are deliberately not validated against :class:`EvalConfig`: the
    schema is open, and checked-in configs carry keys other consumers read
    (e.g. ``action_dim``, used by ``vla-eval test``'s echo server).
    """
    for idx, entry in enumerate(config.get("benchmarks") or []):
        # MutableMapping, not dict: a config handed to vla_eval.run() as an
        # OmegaConf DictConfig still has DictConfig entries after dict(config).
        if not isinstance(entry, MutableMapping):
            raise ValueError(f"benchmarks[{idx}] must be a mapping")
        for key, value in overrides.items():
            if key == "params" and isinstance(value, Mapping):
                params = entry.setdefault("params", {})
                if not isinstance(params, MutableMapping):
                    raise ValueError(f"benchmarks[{idx}].params must be a mapping")
                params.update(value)
            else:
                entry[key] = value
