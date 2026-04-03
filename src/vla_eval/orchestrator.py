"""Orchestrator: coordinates benchmark evaluation runs."""

from __future__ import annotations

import inspect
import json
import logging
import math
import re
import time
from pathlib import Path
from typing import Any, cast

from filelock import FileLock, Timeout

import websockets

from vla_eval.config import EvalConfig, ServerConfig
from vla_eval.connection import Connection
from vla_eval.registry import resolve_import_string
from vla_eval.specs import DimSpec, check_specs
from vla_eval.results.collector import EpisodeResult, ResultCollector
from vla_eval.runners.async_runner import AsyncEpisodeRunner
from vla_eval.runners.clock import Clock
from vla_eval.runners.sync_runner import SyncEpisodeRunner

logger = logging.getLogger(__name__)


class Orchestrator:
    """Coordinates evaluation: creates benchmarks, runners, connections, and runs episodes.

    Execution flow:
        1. For each benchmark in config, resolve the import string to a class.
        2. Instantiate the benchmark with ``params`` from config.
        3. Determine ``max_steps``: if config omits ``max_steps``, the
           benchmark's ``get_metadata()["max_steps"]`` is used instead.
           An explicit config value always takes precedence.
        4. Build a flat list of (task, episode) work items.
        5. If sharding is enabled, select this shard's subset via round-robin
           (``item_index % num_shards == shard_id``).
        6. Run each work item, recording results.  Failures are isolated per
           episode — one crash does not abort the entire benchmark.

    Error recovery:
        - ``ConnectionError`` (server unreachable after retries): aborts the
          benchmark and saves partial results.
        - ``ConnectionClosed`` / ``TimeoutError``: marks the episode as failed,
          attempts reconnection, and continues with the next episode.
        - Other exceptions: marks the episode as failed and continues.

    Result files:
        - Non-sharded: ``{name}_{partial|sync}_{unix_timestamp}.json``
        - Sharded: ``{name}_shard{id}of{total}.json`` (deterministic, no timestamp).
    """

    def __init__(
        self,
        config: dict[str, Any],
        shard_id: int | None = None,
        num_shards: int | None = None,
    ) -> None:
        self.config = config
        self._server_cfg = ServerConfig.from_dict(config.get("server"))
        self.shard_id = shard_id
        self.num_shards = num_shards

    async def run(self) -> list[dict[str, Any]]:
        """Run all benchmarks defined in config."""
        benchmark_configs = self.config.get("benchmarks", [])
        all_results = []

        for bench_cfg in benchmark_configs:
            result = await self._run_benchmark(bench_cfg)
            all_results.append(result)

        return all_results

    async def _run_benchmark(self, bench_cfg: dict[str, Any]) -> dict[str, Any]:
        """Run a single benchmark evaluation."""
        cfg = EvalConfig.from_dict(bench_cfg)
        name = cfg.resolved_name()

        logger.info("Starting benchmark: %s (mode=%s)", name, cfg.mode)

        # Fail fast: claim the output path via file lock (shard mode).
        self._output_file_lock: FileLock | None = None
        if self.num_shards is not None and self.shard_id is not None:
            output_dir = Path(self.config.get("output_dir", "./results"))
            output_dir.mkdir(parents=True, exist_ok=True)
            safe_name = re.sub(r"[^\w\-.]", "_", name)
            output_path = output_dir / f"{safe_name}_shard{self.shard_id}of{self.num_shards}.json"
            if output_path.exists():
                raise FileExistsError(
                    f"Result file already exists: {output_path}\nRemove it or use a different output_dir."
                )
            lock = FileLock(str(output_path) + ".lock", timeout=0)
            try:
                lock.acquire()
                self._output_file_lock = lock
            except Timeout:
                raise FileExistsError(f"Another eval is already writing to {output_path}")

        # Connect to model server FIRST to get observation requirements
        conn = Connection(self._server_cfg.url, timeout=self._server_cfg.timeout)
        await conn.connect(benchmark=cfg.benchmark)

        # Resolve benchmark class and inspect its __init__ signature once
        benchmark_cls = resolve_import_string(cfg.benchmark)
        sig = inspect.signature(benchmark_cls.__init__)

        # Merge server's observation_params into benchmark config.
        # Only fills in keys not already set (--param / YAML values take precedence).
        obs_params = conn.server_info.get("observation_params", {})
        merged_params = dict(cfg.params)
        if obs_params:
            for key, value in obs_params.items():
                if key not in merged_params and key in sig.parameters:
                    merged_params[key] = value
                    logger.info("Auto-configured from model server: %s=%s", key, value)

        try:
            benchmark = benchmark_cls(**merged_params)
        except Exception:
            await conn.close()
            raise

        # Cross-validate action/observation specs between server and benchmark.
        try:
            bench_action_spec: dict[str, DimSpec] = {}
            bench_obs_spec: dict[str, DimSpec] = {}
            server_action_spec: dict[str, DimSpec] = {}
            server_obs_spec: dict[str, DimSpec] = {}
            try:
                bench_action_spec = benchmark.get_action_spec()
                bench_obs_spec = benchmark.get_observation_spec()
            except NotImplementedError:
                logger.debug("Benchmark %s does not implement specs yet", name)
            # Deserialize server specs from HELLO handshake
            for key, raw in conn.server_info.get("action_spec", {}).items():
                server_action_spec[key] = DimSpec.from_dict(raw)
            for key, raw in conn.server_info.get("observation_spec", {}).items():
                server_obs_spec[key] = DimSpec.from_dict(raw)
            if (server_action_spec or server_obs_spec) and (bench_action_spec or bench_obs_spec):
                warnings = check_specs(server_action_spec, bench_action_spec, server_obs_spec, bench_obs_spec)
                for w in warnings:
                    logger.warning("Spec mismatch: %s", w)
                if not warnings:
                    logger.info("Spec validation passed (server↔benchmark compatible)")
        except Exception as exc:
            logger.warning("Spec validation failed: %s", exc)

        # Warn if benchmark supports seeding but config doesn't specify one
        if "seed" in sig.parameters and "seed" not in merged_params:
            default = sig.parameters["seed"].default
            logger.warning(
                "%s accepts 'seed' but config doesn't specify one (using default=%r). "
                "Set seed explicitly in config params for reproducible results.",
                name,
                default,
            )

        metadata = benchmark.get_metadata()

        # max_steps: config value wins; otherwise benchmark metadata; otherwise 300.
        max_steps = cfg.max_steps if cfg.max_steps is not None else metadata.get("max_steps", 300)

        # Create runner
        if cfg.mode.startswith("realtime"):
            runner = AsyncEpisodeRunner(
                hz=cfg.hz,
                hold_policy=cfg.hold_policy,
                action_dim=metadata.get("action_dim", 7),
                clock=Clock(pace=1.0 if cfg.paced else math.inf),
                wait_first_action=cfg.wait_first_action,
            )
        else:
            runner = SyncEpisodeRunner()

        # Get tasks
        tasks = benchmark.get_tasks()
        if cfg.tasks:
            tasks = [t for t in tasks if t.get("suite") in cfg.tasks or t.get("name") in cfg.tasks]
        if cfg.max_tasks:
            tasks = tasks[: cfg.max_tasks]

        # Build flat work-item list and apply sharding
        work_items = [(task, ep) for task in tasks for ep in range(cfg.episodes_per_task)]
        if self.num_shards is not None and self.shard_id is not None:
            # Sharding: round-robin by work-item index, not by task.
            # E.g. 2 tasks × 3 episodes = 6 items → shard 0 gets items 0,2,4.
            work_items = [w for i, w in enumerate(work_items) if i % self.num_shards == self.shard_id]
            logger.info(
                "Shard %d/%d: %d episodes assigned",
                self.shard_id,
                self.num_shards,
                len(work_items),
            )

        collector = ResultCollector(benchmark_name=name, mode=cfg.mode, metric_keys=benchmark.get_metric_keys())

        total_items = len(work_items)

        try:
            for item_idx, (task, ep) in enumerate(work_items):
                task_name = task.get("name", str(task))
                try:
                    episode_idx = ep
                    max_ep = metadata.get("max_episodes_per_task")
                    if cfg.throughput_mode and max_ep is not None:
                        episode_idx = ep % max_ep
                    task = {**task, "episode_idx": episode_idx}
                    raw = await runner.run_episode(benchmark, task, conn, max_steps=max_steps)
                    raw["episode_id"] = ep
                    ep_result = cast(EpisodeResult, raw)
                    collector.record(task_name, ep_result)
                    status = "SUCCESS" if ep_result.get("metrics", {}).get("success") else "FAIL"
                    logger.info(
                        "  [%d/%d] %s ep%d: %s (steps=%d)",
                        item_idx + 1,
                        total_items,
                        task_name,
                        ep,
                        status,
                        ep_result.get("steps", 0),
                    )
                except ConnectionError:
                    # Server unreachable after all retries — save partial and abort
                    logger.error(
                        "  [%d/%d] %s ep%d: server unreachable, aborting benchmark",
                        item_idx + 1,
                        total_items,
                        task_name,
                        ep,
                    )
                    collector.record(
                        task_name,
                        {
                            "episode_id": ep,
                            "metrics": {"success": False},
                            "failure_reason": "server_unreachable",
                        },
                    )
                    return self._save_results(collector, cfg, partial=True, server_info=conn.server_info)
                except websockets.exceptions.ConnectionClosed as exc:
                    close_code = exc.rcvd.code if exc.rcvd else None
                    close_reason = exc.rcvd.reason if exc.rcvd else None
                    logger.warning(
                        "  [%d/%d] %s ep%d: ConnectionClosed code=%s reason=%s",
                        item_idx + 1,
                        total_items,
                        task_name,
                        ep,
                        close_code,
                        close_reason,
                    )
                    collector.record(
                        task_name,
                        {
                            "episode_id": ep,
                            "metrics": {"success": False},
                            "failure_reason": f"connection_closed_{close_code}",
                        },
                    )
                    try:
                        await conn.reconnect()
                    except ConnectionError:
                        logger.error("Reconnect failed, aborting benchmark")
                        return self._save_results(collector, cfg, partial=True, server_info=conn.server_info)
                except TimeoutError:
                    logger.warning(
                        "  [%d/%d] %s ep%d: TimeoutError (act timeout=%ss)",
                        item_idx + 1,
                        total_items,
                        task_name,
                        ep,
                        self._server_cfg.timeout,
                    )
                    collector.record(
                        task_name,
                        {
                            "episode_id": ep,
                            "metrics": {"success": False},
                            "failure_reason": "timeout",
                        },
                    )
                    try:
                        await conn.reconnect()
                    except ConnectionError:
                        logger.error("Reconnect failed, aborting benchmark")
                        return self._save_results(collector, cfg, partial=True, server_info=conn.server_info)
                except Exception:
                    logger.exception(
                        "  [%d/%d] %s ep%d: ERROR",
                        item_idx + 1,
                        total_items,
                        task_name,
                        ep,
                    )
                    collector.record(
                        task_name,
                        {
                            "episode_id": ep,
                            "metrics": {"success": False},
                            "failure_reason": "exception",
                        },
                    )
        finally:
            benchmark.cleanup()
            await conn.close()

        return self._save_results(collector, cfg, partial=False, server_info=conn.server_info)

    def _save_results(
        self,
        collector: ResultCollector,
        cfg: EvalConfig,
        *,
        partial: bool,
        server_info: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Save results to disk. Marks output as partial when the run was interrupted."""
        collector.print_summary()

        output_dir = Path(self.config.get("output_dir", "./results"))
        output_dir.mkdir(parents=True, exist_ok=True)
        output: dict[str, Any] = {**collector.get_benchmark_result(config=cfg.to_dict())}

        if server_info is not None:
            output["server_info"] = server_info

        if partial:
            output["partial"] = True

        # Sanitize benchmark name to prevent path traversal
        name = cfg.resolved_name()
        safe_name = re.sub(r"[^\w\-.]", "_", name)

        # Add shard metadata
        if self.num_shards is not None and self.shard_id is not None:
            output["shard"] = {"id": self.shard_id, "total": self.num_shards}
            output_path = output_dir / f"{safe_name}_shard{self.shard_id}of{self.num_shards}.json"
        else:
            tag = "partial" if partial else cfg.mode
            output_path = output_dir / f"{safe_name}_{tag}_{int(time.time())}.json"

        output_path.write_text(json.dumps(output, indent=2, default=str))
        logger.info("Results saved to %s", output_path)

        # Release file lock after successful save
        if self._output_file_lock is not None:
            self._output_file_lock.release()
            self._output_file_lock = None

        return output
