"""Typed configuration dataclasses for YAML config files.

Converts raw ``dict[str, Any]`` from YAML into typed objects with defaults
and validation.  Each dataclass has a ``from_dict`` classmethod for
construction and an ``to_dict`` method for serialization (e.g. into result
JSON).
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any

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
    """

    image: str | None = None
    volumes: list[str] = field(default_factory=list)
    env: list[str] = field(default_factory=list)
    cpus: str | None = None
    gpus: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> DockerConfig:
        if not data:
            return cls()
        return cls(
            image=data.get("image"),
            volumes=data.get("volumes", []),
            env=data.get("env", []),
            cpus=data.get("cpus"),
            gpus=data.get("gpus"),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EvalConfig:
    """Single evaluation entry in the ``benchmarks`` list.

    Attributes:
        benchmark: Import string in ``module.path:ClassName`` format.
        mode: Execution mode (currently only ``"sync"``).
        name: Display name.  Defaults to class name from ``benchmark``.
        subname: Disambiguator appended to name (e.g. suite name).  Use when
            multiple benchmarks in the same config share a class.
        episodes_per_task: Number of episodes to run per task.
        max_steps: Step limit per episode.  ``None`` → use benchmark metadata.
        max_tasks: Cap on number of tasks.  ``None`` → run all tasks.
        tasks: Filter to specific task names/suites.  ``None`` → all tasks.
        params: Benchmark-specific kwargs passed to the constructor.
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
    # Real-time evaluation params (used when mode starts with "realtime")
    hz: float = 10.0
    hold_policy: str = "repeat_last"
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
            hz=data.get("hz", 10.0),
            hold_policy=data.get("hold_policy", "repeat_last"),
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
