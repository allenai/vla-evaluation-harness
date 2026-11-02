"""Orchestrator: coordinates benchmark evaluation runs."""

from __future__ import annotations

import inspect
import json
import logging
import math
import random
import re
import traceback
import uuid
from collections.abc import Callable, Collection, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import websockets

from vla_eval import __version__, watchdog
from vla_eval.benchmarks.base import VectorStepBenchmark
from vla_eval.config import EvalConfig, ServerConfig
from vla_eval.connection import Connection
from vla_eval.protocol.image_codec import ImageFormat
from vla_eval.protocol.numpy_codec import set_image_format
from vla_eval.recording import (
    DEFAULT_FILENAME_STEM,
    EpisodeRecorder,
    EpisodeStatus,
    NullEpisodeRecorder,
    RecordingError,
    RecordingStore,
    db_path_for_eval,
    recording_filename_context,
    serializable_task_kwargs,
)
from vla_eval.registry import resolve_import_string
from vla_eval.render import apply_render_mode, normalize_render_mode
from vla_eval.specs import DimSpec, check_specs
from vla_eval.results.collector import EpisodeResult, ResultCollector
from vla_eval.runners.base import episode_status, failure_reason
from vla_eval.runners.live_runner import LiveEpisodeRunner
from vla_eval.runners.clock import Clock
from vla_eval.runners.sync_runner import SyncEpisodeRunner
from vla_eval.runners.vector_runner import VectorEpisode, VectorEpisodeRunner
from vla_eval.tracking import Tracker, call_each, get_reporting_trackers

logger = logging.getLogger(__name__)

UNHEALTHY_AFTER = 3  # errors from a shard's first episode on, with no success in between


class UnhealthyShardError(RuntimeError):
    pass


@dataclass
class _Entry:
    """One benchmark entry's run state, shared by the one-at-a-time and the vector loop."""

    cfg: EvalConfig
    name: str
    safe_name: str
    eval_id: str
    work_items: list[tuple[int, Any, int]]
    dynamic: bool
    collector: ResultCollector
    rec_cfg: dict[str, Any] | None
    metadata: dict[str, Any]
    max_steps: int


_SAFE_NAME_RE = re.compile(r"[^\w\-.]")
_DEFAULT_RECORDING_CONFIG: dict[str, Any] = {"record_step": True, "record_video": False}


def _effective_recording_config(raw: dict[str, Any] | None, *, no_save: bool) -> dict[str, Any] | None:
    """Recorder policy for one benchmark entry: on by default, ``recording:`` overrides, ``--no-save`` disables."""
    if no_save:
        return None
    return {**_DEFAULT_RECORDING_CONFIG, **(raw or {})}


_SHARD_SHUFFLE_SEED = 42


def _shard_work_items(work_items: list[Any], num_shards: int, shard_id: int) -> list[Any]:
    """Fixed-seed shuffle so a shard never collects one episode index across every task
    (gcd(num_shards, episodes_per_task) > 1 does that); re-sort by task to keep env rebuilds rare."""
    items = list(work_items)
    random.Random(_SHARD_SHUFFLE_SEED).shuffle(items)
    mine = [w for i, w in enumerate(items) if i % num_shards == shard_id]
    mine.sort(key=lambda w: w[0])
    return mine


def _accepted_init_params(benchmark_cls: type[Any]) -> set[str]:
    """Named ``__init__`` params across the MRO; stops where ``**kwargs`` no longer flows upward."""
    names: set[str] = set()
    for klass in benchmark_cls.__mro__:
        if klass is object or "__init__" not in klass.__dict__:
            continue
        params = list(inspect.signature(klass.__init__).parameters.values())
        names.update(p.name for p in params if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY))
        if not any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params):
            break
    names.discard("self")
    return names


def _merge_observation_params(
    benchmark_cls: type[Any],
    configured_params: Mapping[str, Any],
    observation_params: Mapping[str, Any],
) -> dict[str, Any]:
    """Merge server ``observation_params`` into constructor params. Eval config wins;
    keys no class in the MRO names are dropped with a warning."""
    accepted = _accepted_init_params(benchmark_cls)
    merged = dict(configured_params)

    for key, value in observation_params.items():
        if key in merged:
            continue
        if key in accepted:
            merged[key] = value
            logger.info("Auto-configured from model server: %s=%s", key, value)
        else:
            logger.warning(
                "Model-server observation parameter %r cannot be forwarded to %s; "
                "the benchmark constructor does not accept it",
                key,
                benchmark_cls.__qualname__,
            )

    return merged


class Orchestrator:
    """Coordinates evaluation: creates benchmarks, runners, connections, and runs episodes.

    Execution flow:
        1. For each benchmark in config, resolve the import string to a class.
        2. Instantiate the benchmark with ``params`` from config.
        3. Determine ``max_steps``: if config omits it, the benchmark's
           ``get_metadata()["max_steps"]`` is used.
        4. Build a flat list of (task, episode) work items.
        5. If sharding is enabled, shuffle the list with a fixed seed, take this
           shard's round-robin slice (``item_index % num_shards == shard_id``),
           and re-sort it by task so env rebuilds stay rare.
        6. Run each work item via the runner. Recording goes through SQLite.
           Failures are isolated per episode.  A :class:`VectorStepBenchmark`
           runs up to ``num_envs`` items at once over one connection.

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
        requeue_unhealthy: bool = False,
    ) -> None:
        self.config = config
        self._server_cfg = ServerConfig.from_dict(config.get("server"))
        set_image_format(cast(ImageFormat, self._server_cfg.image_format))
        self._render_mode = normalize_render_mode(config.get("render"))
        self.shard_id = shard_id
        self.num_shards = num_shards
        self.no_save = no_save
        self.requeue_unhealthy = requeue_unhealthy
        self._episodes_ok = 0
        self._errors_from_start = 0
        self._failed_items: list[tuple[str, int]] = []  # (entry eval_id, queue item)
        self._episode_errored = False
        self._eval_id = eval_id or str(uuid.uuid4())
        self._sid = str(uuid.uuid4())  # one per shard process
        self._progress_path: Path | None = None
        self._progress_last: tuple[int, int, int] | None = None
        self._store: RecordingStore | None = None

        # Trackers are instantiated on every shard so config errors surface
        # fast, but live emission only fires when there's a single writer —
        # ``vla-eval export`` owns the aggregate push for sharded runs.
        self._trackers: list[Tracker] = get_reporting_trackers((config.get("tracking") or {}).get("report_to"))
        self._live_tracking = num_shards is None
        if self._trackers and not self._live_tracking:
            logger.warning(
                "tracking.report_to set with sharding active; per-episode and eval-end "
                "emission deferred to `vla-eval export`."
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
        if not self.no_save:
            self._store = RecordingStore(db_path_for_eval(self._output_dir, self._eval_id))
            self._store.set_run_metadata(self._eval_id, self.config)

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

    def _check_health(self, bench_eval_id: str | None, item: int, in_flight: Collection[int] = ()) -> None:
        """Exit a sharded run's shard whose first UNHEALTHY_AFTER episodes all errored (a dead GPU).
        ``in_flight``: queue items still running (vector benchmarks), requeued with the failed ones."""
        if self.num_shards is None or self._episodes_ok or not self._episode_errored:
            return
        self._errors_from_start += 1
        if bench_eval_id is not None:
            self._failed_items.append((bench_eval_id, item))
        if self._errors_from_start < UNHEALTHY_AFTER:
            return
        release = self._failed_items + ([(bench_eval_id, i) for i in in_flight] if bench_eval_id is not None else [])
        if self.requeue_unhealthy and self._store is not None and release:
            self._store.release(release)
        raise UnhealthyShardError(
            f"shard {self.shard_id}: its first {UNHEALTHY_AFTER} episodes all errored"
            + ("; the items went back to the queue" if self.requeue_unhealthy and release else "")
        )

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
        if self._store is not None:
            self._store.set_shard_complete(
                f"{self._eval_id}-{safe_name}", self.shard_id or 0, self.num_shards or 1, complete=False
            )

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
        # Before construction: the renderer binds at the first simulator import.
        try:
            render_env = apply_render_mode(benchmark_cls, self._render_mode, name)
        except Exception:
            await conn.close()
            raise
        sig = inspect.signature(benchmark_cls.__init__)

        obs_params = conn.server_info.get("observation_params") or {}
        merged_params = _merge_observation_params(benchmark_cls, cfg.params, obs_params)

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

        live = cfg.mode.startswith("live")
        if live and isinstance(benchmark, VectorStepBenchmark):
            benchmark.cleanup()
            await conn.close()
            raise ValueError(f"{name}: a VectorStepBenchmark runs in sync mode only (mode={cfg.mode!r})")
        if live:
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

        tasks = benchmark.get_tasks()
        if cfg.tasks:
            tasks = [t for t in tasks if t.get("suite") in cfg.tasks or t.get("name") in cfg.tasks]
        if cfg.max_tasks:
            tasks = tasks[: cfg.max_tasks]

        work_items = [
            (task_idx, task, ep) for task_idx, task in enumerate(tasks) for ep in range(cfg.episodes_per_task)
        ]
        # Shards with a recording DB take items from a shared queue as they free up; without one, a fixed split.
        dynamic = self._store is not None and self.num_shards is not None and self.shard_id is not None
        if self.num_shards is not None and self.shard_id is not None and not dynamic:
            work_items = _shard_work_items(work_items, self.num_shards, self.shard_id)
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
            # applied can differ from requested (e.g. RoboMME's auto lavapipe probe)
            "render": {"requested": self._render_mode, "applied_env": render_env},
        }
        rec_cfg = _effective_recording_config(cfg.recording, no_save=self.no_save)
        if self._store is not None and rec_cfg is not None:
            self._store.upsert_eval_metadata(bench_eval_id, safe_name, bench_metadata)
            if work_items:
                task_idx, first_task, ep = work_items[0]
                self._validate_filename_stem(rec_cfg, first_task, safe_name, task_idx, ep)

        entry = _Entry(
            cfg, name, safe_name, bench_eval_id, work_items, dynamic, collector, rec_cfg, metadata, max_steps
        )
        if isinstance(benchmark, VectorStepBenchmark):
            return await self._run_vector(benchmark, conn, entry)
        runner: SyncEpisodeRunner | LiveEpisodeRunner = SyncEpisodeRunner()
        if live:
            runner = LiveEpisodeRunner(hz=cfg.hz, clock=Clock(pace=1.0 if cfg.paced else math.inf))
        return await self._run_one_at_a_time(benchmark, runner, conn, entry)

    def _episode_ended(
        self,
        e: _Entry,
        recorder: EpisodeRecorder,
        task_name: str,
        ep: int,
        result: Mapping[str, Any] | None,
        error: BaseException | None,
        done: Callable[[], int],
    ) -> None:
        """Book one episode's end (``result`` xor ``error``) for every runner.  ``done()`` runs once the recording
        is durable (finish the queue item there) and returns the entry's finished-episode count."""
        status: EpisodeStatus
        ep_dict: dict[str, Any]
        if error is None:
            assert result is not None
            ep_dict = {**result, "episode_id": ep}
            status = episode_status(ep_dict)
        else:
            reason, detail = self._failure(error)
            ep_dict = {"episode_id": ep, "metrics": {"success": False}}
            ep_dict.update(failure_reason=reason, failure_detail=detail)
            status = "error"
        e.collector.record(task_name, cast(EpisodeResult, ep_dict))
        self._close_recorder(recorder, ep_dict, status, e.name, task_name, ep)
        n, total = done(), len(e.work_items)
        if error is None:
            outcome = "SUCCESS" if status == "success" else "FAIL"
            logger.info("  [%d/%d] %s ep%d: %s (steps=%d)", n, total, task_name, ep, outcome, ep_dict.get("steps", 0))
            self._episodes_ok += 1
        else:
            logger.warning("  [%d/%d] %s ep%d: %s: %s", n, total, task_name, ep, ep_dict["failure_reason"], error)
        self._update_progress(n, total, e.collector.error_count)
        self._episode_errored = error is not None

    @staticmethod
    def _episode_task(e: _Entry, task: Any, ep: int) -> tuple[str, Any]:
        """The task's display name, and the task with this episode's ``episode_idx``."""
        max_ep = e.metadata.get("max_episodes_per_task")
        episode_idx = ep % max_ep if e.cfg.throughput_mode and max_ep is not None else ep
        return task.get("name", str(task)), {**task, "episode_idx": episode_idx}

    async def _run_one_at_a_time(
        self, benchmark: Any, runner: SyncEpisodeRunner | LiveEpisodeRunner, conn: Connection, e: _Entry
    ) -> dict[str, Any]:
        def my_items() -> Iterator[tuple[int, tuple[int, Any, int]]]:
            if not e.dynamic:
                for i, w in enumerate(e.work_items):
                    yield i, w
                    self._check_health(None, i)
                return
            assert self.shard_id is not None and self.num_shards is not None
            store, shard = self._queue(), self.shard_id
            store.seed_queue(e.eval_id, [t for t, _, _ in e.work_items])
            task_idx = None
            while (item := store.claim(e.eval_id, shard, task_idx, self.num_shards)) is not None:
                yield store.queue_progress(e.eval_id)[0], e.work_items[item]
                store.finish(e.eval_id, item)  # not reached when the loop aborts: a rerun redoes it
                self._check_health(e.eval_id, item)
                task_idx = e.work_items[item][0]

        try:
            for item_idx, (task_idx, task, ep) in my_items():
                self._episode_errored = False
                task_name, task = self._episode_task(e, task, ep)
                watchdog.pet(f"{e.safe_name} {task_name} ep{ep}")
                recorder: EpisodeRecorder = NullEpisodeRecorder()
                result: dict[str, Any] | None = None
                error: Exception | None = None
                try:
                    recorder = self._build_recorder(e.rec_cfg, task, e.eval_id, e.safe_name, task_idx, ep, benchmark)
                    result = await runner.run_episode(benchmark, task, conn, max_steps=e.max_steps, recorder=recorder)
                except RecordingError:
                    raise
                except Exception as exc:
                    error = exc
                self._episode_ended(e, recorder, task_name, ep, result, error, lambda: item_idx + 1)
                if isinstance(error, ConnectionError):
                    logger.error("%s: server unreachable, aborting benchmark", e.name)
                    return self._finalize_benchmark(
                        e.collector, e.cfg, e.safe_name, partial=True, server_info=conn.server_info
                    )
                if isinstance(error, (websockets.exceptions.ConnectionClosed, TimeoutError)):
                    try:
                        await conn.reconnect()
                    except Exception:
                        logger.exception("Reconnect failed, aborting benchmark")
                        return self._finalize_benchmark(
                            e.collector, e.cfg, e.safe_name, partial=True, server_info=conn.server_info
                        )
        finally:
            watchdog.pet(f"{e.safe_name} cleanup")
            benchmark.cleanup()
            await conn.close()

        return self._finalize_benchmark(e.collector, e.cfg, e.safe_name, partial=False, server_info=conn.server_info)

    async def _run_vector(self, benchmark: VectorStepBenchmark, conn: Connection, e: _Entry) -> dict[str, Any]:
        """Run the entry's items up to ``num_envs`` at once on ``conn``.  Running episodes share one task;
        the next task starts once they all ended.  Each episode has its own recording ``sid``."""
        if not conn.server_info.get("capabilities", {}).get("vector_step"):
            benchmark.cleanup()
            await conn.close()
            raise RuntimeError(
                f"{e.name}: the model server does not take vector steps (vla-eval too old, or continuous inference)"
            )
        claimed: set[int] = set()  # queue items in flight
        held: tuple[int | None, tuple[int, Any, int]] | None = None  # the next task's first item, waiting
        group: int | None = None  # the running episodes' task_idx
        active = 0
        static_items = iter(e.work_items)
        ended = 0
        if e.dynamic:
            self._queue().seed_queue(e.eval_id, [t for t, _, _ in e.work_items])

        def end(
            episode: VectorEpisode,
            result: dict[str, Any] | None,
            error: BaseException | None,
            *,
            aborted: bool = False,
        ) -> None:
            nonlocal active
            item, ep, task_name = episode.ref
            active -= 1

            def done() -> int:
                nonlocal ended
                ended += 1
                if item is None:
                    return ended
                claimed.discard(item)
                if not aborted:  # an aborted run's item is redone on rerun
                    self._queue().finish(e.eval_id, item)
                return self._queue().queue_progress(e.eval_id)[0]

            self._episode_ended(e, episode.recorder, task_name, ep, result, error, done)
            if not aborted:  # the server went away: a partial result, as in the one-at-a-time loop
                self._check_health(e.eval_id if e.dynamic else None, ended - 1 if item is None else item, claimed)

        def take() -> tuple[int | None, tuple[int, Any, int]] | None:
            nonlocal held
            if held is not None:
                nxt, held = held, None
                return nxt
            if not e.dynamic:
                work = next(static_items, None)
                return None if work is None else (None, work)
            assert self.shard_id is not None and self.num_shards is not None
            item = self._queue().claim(e.eval_id, self.shard_id, group, self.num_shards, exclude=claimed)
            if item is None:
                return None
            claimed.add(item)
            return item, e.work_items[item]

        def next_episode(slot: int) -> VectorEpisode | None:
            nonlocal held, group, active
            while (nxt := take()) is not None:
                item, (task_idx, task, ep) = nxt
                if active and task_idx != group:
                    held = nxt  # leads the next group, once this one has ended
                    return None
                group = task_idx
                task_name, task = self._episode_task(e, task, ep)
                watchdog.pet(f"{e.safe_name} {task_name} ep{ep}")
                active += 1
                try:
                    recorder = self._build_recorder(
                        e.rec_cfg, task, e.eval_id, e.safe_name, task_idx, ep, benchmark, sid=str(uuid.uuid4())
                    )
                except RecordingError:
                    raise
                except Exception as exc:
                    end(VectorEpisode(task, NullEpisodeRecorder(), (item, ep, task_name)), None, exc)
                    continue
                return VectorEpisode(task, recorder, (item, ep, task_name))
            return None

        runner = VectorEpisodeRunner()
        try:
            logger.info("%s: %d environments, one server session each", e.name, benchmark.num_envs)
            await runner.run(benchmark, conn, next_episode, end, max_steps=e.max_steps)
        except ConnectionError:
            logger.error("%s: server unreachable, aborting benchmark", e.name)
            return self._finalize_benchmark(
                e.collector, e.cfg, e.safe_name, partial=True, server_info=conn.server_info
            )
        finally:
            watchdog.pet(f"{e.safe_name} cleanup")
            benchmark.cleanup()
            await conn.close()

        return self._finalize_benchmark(e.collector, e.cfg, e.safe_name, partial=False, server_info=conn.server_info)

    def _queue(self) -> RecordingStore:
        assert self._store is not None
        return self._store

    def _failure(self, error: BaseException) -> tuple[str, str]:
        """``(failure_reason, failure_detail)`` of an episode that ended in ``error``."""
        reason = failure_reason(error)
        if reason == "server_unreachable":
            return reason, str(error)
        if isinstance(error, websockets.exceptions.ConnectionClosed):
            rcvd = error.rcvd
            return reason, f"code={rcvd.code if rcvd else None} reason={rcvd.reason if rcvd else None}"
        if reason == "timeout":
            return reason, f"timeout={self._server_cfg.timeout}s: {error}"
        detail = "".join(traceback.format_exception(type(error), error, error.__traceback__))  # 3.8: no 1-arg form
        return reason, detail

    def _close_recorder(
        self,
        recorder: EpisodeRecorder,
        ep_dict: dict[str, Any],
        status: EpisodeStatus,
        name: str,
        task_name: str,
        ep: int,
    ) -> None:
        recorder.close(
            status=status,
            metrics=ep_dict.get("metrics") or {},
            task_name=task_name,
            episode_id=int(ep_dict.get("episode_id", ep)),
            steps=int(ep_dict.get("steps", 0)),
            elapsed_sec=float(ep_dict.get("elapsed_sec", 0.0)),
            failure_reason=ep_dict.get("failure_reason"),
            failure_detail=ep_dict.get("failure_detail"),
            rt_metrics=ep_dict.get("rt_metrics"),
        )
        # Fire from this site so error terminations (status != "success") reach
        # trackers too — collector.record() misses the reconnect paths.
        if self._live_tracking:
            call_each(self._trackers, "on_episode_end", name, task_name, ep_dict, status)

    def _build_recorder(
        self,
        rec_cfg: dict[str, Any] | None,
        task: dict[str, Any],
        bench_eval_id: str,
        benchmark_safe_name: str,
        task_idx: int,
        episode_id: int,
        benchmark: Any,
        sid: str | None = None,
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
            sid=sid or self._sid,
            eid=eid,
            eval_id=bench_eval_id,
            output_dir=rec_cfg.get("output_dir") or str(self._output_dir / "episodes"),
            filename_stem=rec_cfg.get("filename_stem") or DEFAULT_FILENAME_STEM,
            context={**serializable_task_kwargs(task), "task_idx": task_idx},
            filename_context=recording_filename_context(
                benchmark_safe_name=benchmark_safe_name, task_idx=task_idx, episode_id=episode_id
            ),
            record_video=bool(rec_cfg["record_video"]),
            record_step=bool(rec_cfg["record_step"]),
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
        probe = {
            **serializable_task_kwargs(first_task),
            "episode_idx": 0,
            **recording_filename_context(
                benchmark_safe_name=benchmark_safe_name, task_idx=task_idx, episode_id=episode_id
            ),
            "status": "success",
        }
        try:
            stem.format(**probe)
        except KeyError as exc:
            raise ValueError(
                f"recording.filename_stem={stem!r} references unknown key {exc} (available: {sorted(probe)})."
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

        if self._store is not None:
            self._store.set_shard_complete(
                f"{self._eval_id}-{safe_name}", self.shard_id or 0, self.num_shards or 1, complete=not partial
            )

        if self._progress_path is not None and self._progress_path.exists():
            self._progress_path.unlink()
        return output
