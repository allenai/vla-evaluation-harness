"""Orchestrator: coordinates benchmark evaluation runs."""

from __future__ import annotations

import inspect
import json
import logging
import math
import re
import traceback
import uuid
from pathlib import Path
from typing import Any, cast

import websockets

from vla_eval import __version__, watchdog
from vla_eval.config import EvalConfig, ServerConfig
from vla_eval.connection import Connection
from vla_eval.recording import (
    DEFAULT_FILENAME_STEM,
    EpisodeRecorder,
    EpisodeStatus,
    NullEpisodeRecorder,
    RecordingStore,
    db_path_for_eval,
    recording_filename_context,
    serializable_task_kwargs,
)
from vla_eval.registry import resolve_import_string
from vla_eval.specs import DimSpec, check_specs
from vla_eval.results.collector import EpisodeResult, ResultCollector
from vla_eval.runners.live_runner import LiveEpisodeRunner
from vla_eval.runners.clock import Clock
from vla_eval.runners.sync_runner import SyncEpisodeRunner
from vla_eval.tracking import Tracker, call_each, get_reporting_trackers

logger = logging.getLogger(__name__)

_SAFE_NAME_RE = re.compile(r"[^\w\-.]")
_DEFAULT_RECORDING_CONFIG: dict[str, Any] = {"record_step": True, "record_video": False}


def _effective_recording_config(raw: dict[str, Any] | None, *, no_save: bool) -> dict[str, Any] | None:
    """Return the recorder policy for one benchmark entry.

    Runs persist episode results and step rows by default. ``recording:`` is an
    optional override block, and ``--no-save`` is the only global off switch.
    """
    if no_save:
        return None
    if raw is None:
        return dict(_DEFAULT_RECORDING_CONFIG)
    if not isinstance(raw, dict):
        raise TypeError("recording must be a mapping or null")
    return {**_DEFAULT_RECORDING_CONFIG, **raw}


class Orchestrator:
    """Coordinates evaluation: creates benchmarks, runners, connections, and runs episodes.

    Execution flow:
        1. For each benchmark in config, resolve the import string to a class.
        2. Instantiate the benchmark with ``params`` from config.
        3. Determine ``max_steps``: if config omits it, the benchmark's
           ``get_metadata()["max_steps"]`` is used.
        4. Build a flat list of (task, episode) work items.
        5. If sharding is enabled, select this shard's subset via round-robin
           (``item_index % num_shards == shard_id``).
        6. Run each work item via the runner. Recording goes through SQLite.
           Failures are isolated per episode.

    Error recovery:
        - ``ConnectionError`` (server unreachable after retries): abort the
          benchmark, return partial.
        - ``ConnectionClosed`` / ``TimeoutError``: mark episode failed,
          reconnect, continue.
        - Other exceptions: mark episode failed, continue.

    """

    def __init__(
        self,
        config: dict[str, Any],
        shard_id: int | None = None,
        num_shards: int | None = None,
        eval_id: str | None = None,
        no_save: bool = False,
    ) -> None:
        self.config = config
        self._server_cfg = ServerConfig.from_dict(config.get("server"))
        self.shard_id = shard_id
        self.num_shards = num_shards
        self.no_save = no_save
        self._eval_id = eval_id or str(uuid.uuid4())
        self._sid = str(uuid.uuid4())  # one per shard process
        self._progress_path: Path | None = None
        self._progress_last: tuple[int, int, int] | None = None
        self._store: RecordingStore | None = None

        # Trackers are instantiated on every shard so config errors surface
        # fast, but live emission only fires when there's a single writer —
        # ``vla-eval merge`` owns the aggregate push for sharded runs.
        self._trackers: list[Tracker] = get_reporting_trackers((config.get("tracking") or {}).get("report_to"))
        self._live_tracking = num_shards is None
        if self._trackers and not self._live_tracking:
            logger.warning(
                "tracking.report_to set with sharding active; per-episode and eval-end "
                "emission deferred to `vla-eval merge`."
            )

    @property
    def eval_id(self) -> str:
        return self._eval_id

    @property
    def _output_dir(self) -> Path:
        d = Path(self.config.get("output_dir", "./results")).resolve()
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _shard_stem(self, safe_name: str) -> str:
        if self.num_shards is not None and self.shard_id is not None:
            return f"{safe_name}_shard{self.shard_id}of{self.num_shards}"
        return safe_name

    async def run(self) -> list[dict[str, Any]]:
        """Run all benchmarks defined in config."""
        if self._live_tracking:
            call_each(self._trackers, "on_eval_begin", self._eval_id, self.config)

        all_results = []
        try:
            for bench_cfg in self.config.get("benchmarks", []):
                result = await self._run_benchmark(bench_cfg)
                all_results.append(result)
                if self._live_tracking:
                    call_each(self._trackers, "on_benchmark_end", result.get("benchmark", ""), result)
        finally:
            if self._store is not None:
                self._store.close()
                self._store = None
            if self._live_tracking:
                call_each(self._trackers, "on_eval_end", all_results)
                call_each(self._trackers, "close")

        return all_results

    def _ensure_store(self) -> RecordingStore:
        """Open the recording store on first benchmark that records."""
        if self._store is None:
            self._store = RecordingStore(db_path_for_eval(self._output_dir, self._eval_id))
        return self._store

    def _update_progress(self, completed: int, total: int, errors: int) -> None:
        """Atomic per-shard progress file for live monitoring; skips no-op writes."""
        if self._progress_path is None:
            return
        snap = (completed, total, errors)
        if snap == self._progress_last:
            return
        self._progress_last = snap
        tmp = self._progress_path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"completed": completed, "total": total, "errors": errors}))
        tmp.replace(self._progress_path)

    async def _run_benchmark(self, bench_cfg: dict[str, Any]) -> dict[str, Any]:
        cfg = EvalConfig.from_dict(bench_cfg)
        name = cfg.resolved_name()
        safe_name = _SAFE_NAME_RE.sub("_", name)

        logger.info("Starting benchmark: %s (mode=%s)", name, cfg.mode)
        if self._live_tracking:
            call_each(self._trackers, "on_benchmark_begin", name, bench_cfg)
        return await self._run_benchmark_inner(cfg, name, safe_name)

    async def _run_benchmark_inner(self, cfg: EvalConfig, name: str, safe_name: str) -> dict[str, Any]:
        self._progress_path = self._output_dir / f"{self._shard_stem(safe_name)}.progress"
        self._progress_last = None

        conn = Connection(self._server_cfg.url, timeout=self._server_cfg.timeout)
        await conn.connect(benchmark=cfg.benchmark)

        benchmark_cls = resolve_import_string(cfg.benchmark)
        sig = inspect.signature(benchmark_cls.__init__)

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

        # Spec cross-validation
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

        if "seed" in sig.parameters and "seed" not in merged_params:
            default = sig.parameters["seed"].default
            logger.warning(
                "%s accepts 'seed' but config doesn't specify one (using default=%r). "
                "Set seed explicitly in config params for reproducible results.",
                name,
                default,
            )

        metadata = benchmark.get_metadata()
        max_steps = cfg.max_steps if cfg.max_steps is not None else metadata.get("max_steps", 300)

        if cfg.mode.startswith("live"):
            # Fail fast before any episode: a real-time benchmark must declare its
            # stale-tick hold, else every episode would raise mid-run and the
            # per-episode error isolation would flood logs while wasting resources.
            try:
                benchmark.get_hold_action(None)
            except NotImplementedError as exc:
                await conn.close()
                raise RuntimeError(
                    f"Benchmark {name} is configured for real-time mode but does not implement "
                    "get_hold_action(); declare the embodiment's safe do-nothing action."
                ) from exc
            runner = LiveEpisodeRunner(
                hz=cfg.hz,
                clock=Clock(pace=1.0 if cfg.paced else math.inf),
                wait_first_action=cfg.wait_first_action,
            )
        else:
            runner = SyncEpisodeRunner()

        tasks = benchmark.get_tasks()
        if cfg.tasks:
            tasks = [t for t in tasks if t.get("suite") in cfg.tasks or t.get("name") in cfg.tasks]
        if cfg.max_tasks:
            tasks = tasks[: cfg.max_tasks]

        work_items = [
            (task_idx, task, ep) for task_idx, task in enumerate(tasks) for ep in range(cfg.episodes_per_task)
        ]
        if self.num_shards is not None and self.shard_id is not None:
            work_items = [w for i, w in enumerate(work_items) if i % self.num_shards == self.shard_id]
            logger.info("Shard %d/%d: %d episodes assigned", self.shard_id, self.num_shards, len(work_items))

        collector = ResultCollector(benchmark_name=name, mode=cfg.mode, metric_keys=benchmark.get_metric_keys())
        total_items = len(work_items)
        self._update_progress(0, total_items, 0)

        bench_eval_id = f"{self._eval_id}-{safe_name}"
        bench_metadata = {
            "benchmark": name,
            "mode": cfg.mode,
            "config": cfg.to_dict(),
            "metric_keys": benchmark.get_metric_keys(),
            "harness_version": __version__,
            "server_info": conn.server_info,
        }
        rec_cfg = _effective_recording_config(cfg.recording, no_save=self.no_save)
        if rec_cfg is not None:
            self._ensure_store().upsert_eval_metadata(bench_eval_id, safe_name, bench_metadata)
        if rec_cfg is not None and work_items:
            task_idx, first_task, ep = work_items[0]
            self._validate_filename_stem(rec_cfg, first_task, safe_name, task_idx, ep)

        def record_failure(reason: str, detail: str) -> dict[str, Any]:
            fail: dict[str, Any] = {
                "episode_id": ep,
                "metrics": {"success": False},
                "failure_reason": reason,
                "failure_detail": detail,
            }
            collector.record(task_name, cast(EpisodeResult, fail))
            self._update_progress(item_idx + 1, total_items, collector.error_count)
            return fail

        def close_recorder(ep_dict: dict[str, Any], status: EpisodeStatus) -> None:
            recorder.close(
                status=status,
                metrics=ep_dict.get("metrics") or {},
                task_name=task_name,
                episode_id=int(ep_dict.get("episode_id", ep)),
                steps=int(ep_dict.get("steps", 0)),
                elapsed_sec=float(ep_dict.get("elapsed_sec", 0.0)),
                failure_reason=ep_dict.get("failure_reason"),
                failure_detail=ep_dict.get("failure_detail"),
            )
            # Fire from this site so error terminations (status != "success") reach
            # trackers too — collector.record() above misses the reconnect paths.
            if self._live_tracking:
                call_each(self._trackers, "on_episode_end", name, task_name, ep_dict, status)

        try:
            for item_idx, (task_idx, task, ep) in enumerate(work_items):
                task_name = task.get("name", str(task))
                watchdog.pet(f"{safe_name} {task_name} ep{ep}")
                recorder: EpisodeRecorder = NullEpisodeRecorder()
                try:
                    episode_idx = ep
                    max_ep = metadata.get("max_episodes_per_task")
                    if cfg.throughput_mode and max_ep is not None:
                        episode_idx = ep % max_ep
                    task = {**task, "episode_idx": episode_idx}
                    recorder = self._build_recorder(rec_cfg, task, bench_eval_id, safe_name, task_idx, ep, benchmark)
                    raw = await runner.run_episode(benchmark, task, conn, max_steps=max_steps, recorder=recorder)
                    raw["episode_id"] = ep
                    ep_result = cast(EpisodeResult, raw)
                    collector.record(task_name, ep_result)
                    ep_dict = dict(ep_result)
                    success = bool((ep_dict.get("metrics") or {}).get("success"))
                    logger.info(
                        "  [%d/%d] %s ep%d: %s (steps=%d)",
                        item_idx + 1,
                        total_items,
                        task_name,
                        ep,
                        "SUCCESS" if success else "FAIL",
                        ep_dict.get("steps", 0),
                    )
                    self._update_progress(item_idx + 1, total_items, collector.error_count)
                    close_recorder(ep_dict, "success" if success else "fail")
                    continue
                except ConnectionError as exc:
                    logger.error(
                        "  [%d/%d] %s ep%d: server unreachable, aborting benchmark",
                        item_idx + 1,
                        total_items,
                        task_name,
                        ep,
                    )
                    fail = record_failure("server_unreachable", str(exc))
                    close_recorder(fail, "error")
                    return self._finalize_benchmark(
                        collector, cfg, safe_name, partial=True, server_info=conn.server_info
                    )
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
                    fail = record_failure("connection_closed", f"code={close_code} reason={close_reason}")
                    close_recorder(fail, "error")
                    try:
                        await conn.reconnect()
                    except Exception:
                        logger.exception("Reconnect failed, aborting benchmark")
                        return self._finalize_benchmark(
                            collector, cfg, safe_name, partial=True, server_info=conn.server_info
                        )
                    continue
                except TimeoutError as exc:
                    logger.warning(
                        "  [%d/%d] %s ep%d: TimeoutError (act timeout=%ss)",
                        item_idx + 1,
                        total_items,
                        task_name,
                        ep,
                        self._server_cfg.timeout,
                    )
                    fail = record_failure("timeout", f"timeout={self._server_cfg.timeout}s: {exc}")
                    close_recorder(fail, "error")
                    try:
                        await conn.reconnect()
                    except Exception:
                        logger.exception("Reconnect failed, aborting benchmark")
                        return self._finalize_benchmark(
                            collector, cfg, safe_name, partial=True, server_info=conn.server_info
                        )
                    continue
                except Exception:
                    logger.exception(
                        "  [%d/%d] %s ep%d: ERROR",
                        item_idx + 1,
                        total_items,
                        task_name,
                        ep,
                    )
                    fail = record_failure("exception", traceback.format_exc())
                    close_recorder(fail, "error")
                    continue
        finally:
            watchdog.pet(f"{safe_name} cleanup")
            benchmark.cleanup()
            await conn.close()

        return self._finalize_benchmark(collector, cfg, safe_name, partial=False, server_info=conn.server_info)

    def _build_recorder(
        self,
        rec_cfg: dict[str, Any] | None,
        task: dict[str, Any],
        bench_eval_id: str,
        benchmark_safe_name: str,
        task_idx: int,
        episode_id: int,
        benchmark: Any,
    ) -> EpisodeRecorder:
        """Build per-episode recorder from YAML config + task dict, or Null if recording is off.

        ``step_fields`` is read from the benchmark config (per-benchmark
        ``params.step_fields``) and validated against ``benchmark._ALL_RECORD_FIELDS``
        by the recorder.
        """
        if self._store is None or rec_cfg is None:
            return NullEpisodeRecorder()
        eid = str(uuid.uuid4())
        allowed = getattr(benchmark, "_ALL_RECORD_FIELDS", None)
        return EpisodeRecorder(
            store=self._store,
            sid=self._sid,
            eid=eid,
            eval_id=bench_eval_id,
            output_dir=rec_cfg.get("output_dir") or str(self._output_dir / "episodes"),
            filename_stem=rec_cfg.get("filename_stem") or DEFAULT_FILENAME_STEM,
            context=serializable_task_kwargs(task),
            filename_context=recording_filename_context(
                task, benchmark_safe_name=benchmark_safe_name, task_idx=task_idx, episode_id=episode_id
            ),
            record_video=bool(rec_cfg.get("record_video", False)),
            record_step=bool(rec_cfg.get("record_step", True)),
            video_fps=int(rec_cfg.get("video_fps", 20)),
            step_fields=rec_cfg.get("step_fields"),
            allowed_fields=allowed,
        )

    def _validate_filename_stem(
        self,
        rec_cfg: dict[str, Any],
        first_task: dict[str, Any],
        benchmark_safe_name: str,
        task_idx: int,
        episode_id: int,
    ) -> None:
        """Dry-render the template so YAML key typos fail before any episode runs."""
        stem = rec_cfg.get("filename_stem") or DEFAULT_FILENAME_STEM
        # episode_idx is injected per-iteration by the run loop, not present on first_task itself.
        probe = recording_filename_context(
            {**first_task, "episode_idx": 0},
            benchmark_safe_name=benchmark_safe_name,
            task_idx=task_idx,
            episode_id=episode_id,
        )
        probe["status"] = "success"
        try:
            stem.format(**probe)
        except KeyError as exc:
            raise ValueError(
                f"recording.filename_stem={stem!r} references key {exc} "
                f"that's not in the task dict (available: {sorted(probe)})."
            ) from None
        except (IndexError, ValueError) as exc:
            raise ValueError(f"recording.filename_stem={stem!r} is malformed: {exc}") from None

    def _finalize_benchmark(
        self,
        collector: ResultCollector,
        cfg: EvalConfig,
        safe_name: str,
        *,
        partial: bool,
        server_info: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Print summary and return the in-memory benchmark result for the caller."""
        collector.print_summary()

        output: dict[str, Any] = {**collector.get_benchmark_result(config=cfg.to_dict())}
        if server_info is not None:
            output["server_info"] = server_info
        if partial:
            output["partial"] = True
        if self.num_shards is not None and self.shard_id is not None:
            output["shard"] = {"id": self.shard_id, "total": self.num_shards}

        if self._progress_path is not None and self._progress_path.exists():
            self._progress_path.unlink()
        return output
