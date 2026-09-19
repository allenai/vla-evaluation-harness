"""Embed vla-eval in a Python process.

Three entry points, composable or used alone:

* :func:`serve_background` hosts a :class:`~vla_eval.model_servers.base.ModelServer`
  on a daemon thread and returns its URL.
* :func:`run` executes an eval config (the body of ``vla-eval run``) against a
  server URL and returns the per-benchmark results.
* :func:`evaluate` does both: serve *this* model in-process, run the benchmark
  (in Docker by default, exactly as the CLI would), return the results.

The CLI keeps the process-level conveniences that do not belong inside a
training script: the stall watchdog (which ``os._exit``s the process) is off
unless ``watchdog_timeout_s`` is given, and no tracking run is created unless
the config asks for one.
"""

from __future__ import annotations

import copy
import logging
import math
import os
import uuid
from concurrent.futures import Future
from pathlib import Path
from typing import Any, Mapping

import anyio
from anyio.abc import TaskStatus
from anyio.from_thread import start_blocking_portal

from vla_eval import watchdog
from vla_eval.config import DockerConfig
from vla_eval.model_servers.base import ModelServer
from vla_eval.model_servers.serve import serve_async
from vla_eval.orchestrator import Orchestrator
from vla_eval.render import check_run_render_support, resolve_run_render_mode

logger = logging.getLogger(__name__)

__all__ = ["ServerHandle", "evaluate", "run", "serve_background"]


class ServerHandle:
    """A model server running on a background thread. Use as a context manager or call :meth:`close`."""

    def __init__(self, host: str, port: int, portal_cm: Any, server_future: Future[Any]) -> None:
        self.host = host
        self.port = port
        self._portal_cm = portal_cm
        self._server_future = server_future

    @property
    def url(self) -> str:
        return f"ws://{self.host}:{self.port}"

    def close(self) -> None:
        """Cancel the server and stop its thread. Idempotent."""
        if self._portal_cm is None:
            return
        self._server_future.cancel()
        self._portal_cm.__exit__(None, None, None)
        self._portal_cm = None

    def __enter__(self) -> ServerHandle:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def serve_background(
    model_server: ModelServer, *, host: str = "127.0.0.1", port: int = 0, ready_timeout: float = 30.0
) -> ServerHandle:
    """Serve *model_server* on a daemon thread; returns once the socket is listening.

    ``port=0`` picks a free port. The server shares the caller's process, so a policy
    that lives on the training GPU is served without copying weights anywhere.
    """
    portal_cm = start_blocking_portal()
    try:
        server_future, bound_port = portal_cm.__enter__().start_task(
            _serve_or_timeout, model_server, host, port, ready_timeout
        )
    except BaseException as exc:
        portal_cm.__exit__(type(exc), exc, exc.__traceback__)  # an exception makes the portal cancel its tasks
        raise
    return ServerHandle(host, bound_port, portal_cm, server_future)


async def _serve_or_timeout(
    model_server: ModelServer, host: str, port: int, ready_timeout: float, *, task_status: TaskStatus[int]
) -> None:
    """``serve_async`` with a deadline on startup only: lifted once the port is reported."""
    with anyio.CancelScope(deadline=anyio.current_time() + ready_timeout) as scope:

        class _Started(TaskStatus[int]):
            def started(self, value: int | None = None) -> None:
                scope.deadline = math.inf
                task_status.started(value if value is not None else port)

        await serve_async(model_server, host, port, task_status=_Started())
    if scope.cancelled_caught:
        raise TimeoutError(f"model server did not start listening within {ready_timeout}s")


def _merge_benchmark_overrides(config: dict[str, Any], overrides: Mapping[str, Any]) -> None:
    """Apply *overrides* to every ``benchmarks[]`` entry; ``params`` merges instead of replacing."""
    for entry in config.get("benchmarks") or []:
        for key, value in overrides.items():
            if key == "params" and isinstance(value, Mapping):
                entry.setdefault("params", {}).update(value)
            else:
                entry[key] = value


def run(
    config: str | Path | Mapping[str, Any],
    *,
    server_url: str | None = None,
    output_dir: str | Path | None = None,
    eval_id: str | None = None,
    no_save: bool = False,
    docker: bool | None = None,
    runtime: str | None = None,
    pull: bool = False,
    build: bool = False,
    benchmark_overrides: Mapping[str, Any] | None = None,
    watchdog_timeout_s: float | None = None,
) -> list[dict[str, Any]]:
    """Run an eval config and return one :class:`~vla_eval.results.collector.BenchmarkResult` per entry.

    Args:
        config: Path to an eval YAML (``extends`` and ``${oc.env:...}`` resolved) or a config dict.
        server_url: Overrides ``server.url``; usually :attr:`ServerHandle.url`.
        output_dir: Overrides ``output_dir``.
        eval_id: Recording id; generated when omitted.
        no_save: Skip the SQLite recording (in-memory results only). Local runs only.
        docker: ``None`` follows the config (``docker.image`` set → container), ``False`` forces
            an in-process run, ``True`` requires ``docker.image``.
        runtime: Container runtime for the image: ``"docker"`` or ``"charliecloud"`` (no daemon, no
            root; see docs/runtimes.md). Default follows ``docker.runtime`` / ``$VLA_EVAL_RUNTIME``.
        pull: Allow pulling a missing image without a prompt (images are often tens of GB).
        build: Rebuild ``docker.image`` from ``docker.build`` even if it exists locally.
        benchmark_overrides: Keys applied to every benchmark entry, e.g.
            ``{"episodes_per_task": 10, "max_tasks": 1, "params": {"seed": 3}}``.
        watchdog_timeout_s: Stall watchdog for this run only: in-process it is disarmed on return,
            in a container it is forwarded as ``VLA_EVAL_WATCHDOG_TIMEOUT_S``. It ``os._exit``s
            the *whole process* on a stall, so leave it off inside a training loop.
    """
    if isinstance(config, (str, Path)):
        from vla_eval.cli.config_loader import load_config

        cfg = load_config(str(config))
    else:
        cfg = copy.deepcopy(dict(config))

    if server_url is not None:
        cfg.setdefault("server", {})["url"] = server_url
    if output_dir is not None:
        cfg["output_dir"] = str(output_dir)
    if benchmark_overrides:
        _merge_benchmark_overrides(cfg, benchmark_overrides)
    cfg["output_dir"] = str(Path(cfg.get("output_dir") or "./results").resolve())

    render_mode = resolve_run_render_mode(cfg, None, None)
    check_run_render_support(cfg, render_mode)

    docker_cfg = DockerConfig.from_dict(cfg.get("docker"))
    if docker is None:
        from vla_eval.cli._docker import inside_docker

        use_docker = bool(docker_cfg.image) and not inside_docker()
    else:
        use_docker = docker
    if use_docker and not docker_cfg.image:
        raise ValueError("docker=True requires docker.image in the config")

    if use_docker:
        if no_save:
            raise ValueError("no_save is not available for Docker runs: results come back through the recording")
        from vla_eval.cli._docker import run_in_container
        from vla_eval.results.export import export_eval

        eval_id = eval_id or str(uuid.uuid4())
        env_key, previous = "VLA_EVAL_WATCHDOG_TIMEOUT_S", os.environ.get("VLA_EVAL_WATCHDOG_TIMEOUT_S")
        if watchdog_timeout_s is not None:
            os.environ[env_key] = str(watchdog_timeout_s)  # the container's own watchdog reads this
        try:
            rc = run_in_container(
                cfg, runtime=runtime, auto_yes=pull, eval_id=eval_id, no_save=False, force_build=build
            )
        except SystemExit as exc:  # the docker helpers exit on missing daemon/image
            raise RuntimeError(f"benchmark container could not be started (exit {exc.code})") from exc
        finally:
            if watchdog_timeout_s is not None:
                if previous is None:
                    os.environ.pop(env_key, None)
                else:
                    os.environ[env_key] = previous
        if rc != 0:
            raise RuntimeError(f"benchmark container exited with status {rc}")
        return export_eval(Path(cfg["output_dir"]), eval_id)

    if watchdog_timeout_s is not None:
        watchdog.start(watchdog_timeout_s)
    try:
        orchestrator = Orchestrator(cfg, eval_id=eval_id, no_save=no_save)
        results = anyio.run(orchestrator.run)
    finally:
        if watchdog_timeout_s is not None:
            watchdog.stop()  # it would otherwise os._exit the caller once the run goes quiet
    if no_save:
        return results
    # Same shape and source as the Docker path: what ``vla-eval export`` materialised from the recording.
    from vla_eval.results.export import export_eval

    return export_eval(Path(cfg["output_dir"]), orchestrator.eval_id)


def evaluate(
    model_server: ModelServer, config: str | Path | Mapping[str, Any], **run_kwargs: Any
) -> list[dict[str, Any]]:
    """Serve *model_server* in-process and :func:`run` *config* against it.

    Blocking; call it from a training loop between optimizer steps (with the policy in eval
    mode). Keyword arguments are forwarded to :func:`run`.
    """
    with serve_background(model_server) as handle:
        return run(config, server_url=handle.url, **run_kwargs)
