# /// script
# requires-python = "~=3.11"
# dependencies = [
#     "vla-eval",
#     "jax[cuda12]==0.4.35",
#     "flax==0.10.2",
#     "einops",
#     "numpy==1.26.4",
# ]
#
# [tool.uv.sources]
# vla-eval = { path = "../../..", editable = true }
#
# [tool.uv]
# exclude-newer = "2026-11-02T00:20:00Z"
#
# [tool.ty.rules]  # ty checks scripts on their own, against the host env without their deps
# unresolved-import = "ignore"
# ///
"""Real-Time Chunking (arXiv:2506.07339) on Kinetix, with the official sampling code.

``KinetixRTCServer`` executes the official ``eval_flow.py`` chunk protocol, sync or live: inference every ``horizon``
(s) steps from that step's observation, the previous chunk covering the first ``delay`` (d) steps.  Sampling is the
official ``model.FlowPolicy`` (``action`` / ``realtime_action`` / ``bid_action``) imported from ``rtc_src_path``, else
from a ``Physical-Intelligence/real-time-chunking-kinetix`` clone at ``_RTC_REV``.  Checkpoints: the public
``gs://rtc-assets/bc/<epoch>/policies``, fetched on first use, or a local directory.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import pickle
import sys
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

# nvidia-cuda-nvcc-cu12 is a namespace package (__file__ None), which breaks jax 0.4.35's CUDA lookup.
try:
    import nvidia.cuda_nvcc as _cuda_nvcc

    if _cuda_nvcc.__file__ is None and _cuda_nvcc.__path__:
        import os as _os

        _cuda_nvcc.__file__ = _os.path.join(_cuda_nvcc.__path__[0], "__init__.py")
except ImportError:
    pass

import anyio
import numpy as np
from anyio.to_thread import run_sync as _run_in_thread

from vla_eval.model_servers.base import Rows
from vla_eval.model_servers.daemons import DaemonModelServer, EpisodeContext, ObservationRequest
from vla_eval.recording import StepRecorder
from vla_eval.specs import LANGUAGE, RAW, DimSpec
from vla_eval.types import EpisodeResult

logger = logging.getLogger(__name__)

Method = Literal["naive", "rtc", "rtc_hard", "bid"]
Latency = Literal["steps", "wall", "wall_exact"]
H = 8  # the checkpoints' prediction horizon
_RTC_REPO = "https://github.com/Physical-Intelligence/real-time-chunking-kinetix.git"
_RTC_REV = "9296f31d62d5bfeb5779dcb2f9bcf71ca37f448b"


def _rtc_src() -> str:
    """The official repo's ``src/``, cloned at ``_RTC_REV`` (one directory per revision)."""
    from vla_eval.dirs import ensure_git_clone

    clone = ensure_git_clone(name=f"rtc-kinetix-{_RTC_REV[:12]}", repo=_RTC_REPO, rev=_RTC_REV, shallow=False)
    return str(clone / "src")


def _local_dir(path: str) -> str:
    """A local directory as-is; ``gs://<bucket>/<prefix>`` (public) has its ``*.pkl`` fetched once, md5-checked."""
    if not path.startswith("gs://"):
        return path
    from vla_eval.dirs import assets_cache

    bucket, _, prefix = path.removeprefix("gs://").strip("/").partition("/")
    target = assets_cache(f"{bucket}/{prefix}")
    if (target / ".complete").exists():
        return str(target)
    api = f"https://storage.googleapis.com/storage/v1/b/{bucket}/o"
    query = urllib.parse.urlencode({"prefix": f"{prefix}/", "fields": "items(name,md5Hash)"})
    with urllib.request.urlopen(f"{api}?{query}", timeout=60) as resp:
        items = [it for it in json.load(resp).get("items", []) if it["name"].endswith(".pkl")]
    if not items:
        raise FileNotFoundError(f"no *.pkl under {path}")
    target.mkdir(parents=True, exist_ok=True)
    for it in items:
        dest = target / it["name"].rsplit("/", 1)[-1]
        if dest.exists():
            continue
        logger.info("Fetching %s/%s -> %s", path, dest.name, dest)
        url = f"{api}/{urllib.parse.quote(it['name'], safe='')}?alt=media"
        tmp = dest.with_name(f"{dest.name}.tmp-{os.getpid()}")
        try:
            with urllib.request.urlopen(url, timeout=60) as resp, tmp.open("wb") as fh:
                md5 = hashlib.md5()
                while chunk := resp.read(1 << 20):
                    md5.update(chunk)
                    fh.write(chunk)
            if base64.b64encode(md5.digest()).decode() != it["md5Hash"]:
                raise OSError(f"md5 mismatch for {url}")
            tmp.replace(dest)
        finally:
            tmp.unlink(missing_ok=True)
    (target / ".complete").touch()
    return str(target)


def _filter_none(d: dict) -> dict:
    """Drop None leaves (older flax checkpoints)."""
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            if sub := _filter_none(v):
                out[k] = sub
        elif v is not None:
            out[k] = v
    return out


@dataclass(eq=False)
class _Inference:
    k: int  # boundary index; -1 is the initial chunk
    t: int  # the boundary step (its observation's step)
    prev_k: int  # the chunk it was guided by
    done: anyio.Event = field(default_factory=anyio.Event)
    chunk: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), np.float32))
    infer_ms: float = 0.0
    landed_wall: float = 0.0


@dataclass(eq=False)
class _RowState:
    level: str
    seed: int
    recorder: StepRecorder | None
    key: Any = None  # JAX PRNG key, made on the model thread


class KinetixRTCServer(DaemonModelServer):
    """Official RTC / BID / naive chunk execution for the 12 Kinetix levels (one checkpoint per level).

    Each observation is answered with the plan's action for its step (``reply`` in sync, ``publish`` in live), so
    the step index, not a time grid, decides which slot a tick gets: the benchmark's ``env_step`` when the observation
    carries one (Kinetix holds its env until the first action), else its index.  A boundary's chunk lands ``delay``
    steps later (``latency="steps"``) or after ``(delay - 1/2) / hz`` (``"wall"``) / ``delay / hz`` (``"wall_exact"``) seconds
    since its observation's admission, compute included (live only).  The initial chunk (official: computed with
    no delay before the loop) is exempt unless ``first_chunk_delayed``, in which case the first steps get no action
    (needs a benchmark that steps on holds).

    Args:
        checkpoint_dir: ``worlds_l_<level>.pkl`` files, local or ``gs://`` (the paper evaluates ``bc/31/policies``).
        rtc_src_path: The official repo's ``src/`` (``import model``); default: a clone at ``_RTC_REV``.
        weak_checkpoint_dir: BID's weak policy (paper: the 8-epoch ``bc/7/policies``); needed when ``bid_k`` is set.
        method: ``naive``, ``rtc`` (soft masking, ``exp``), ``rtc_hard`` (``zeros``) or ``bid``.
        delay / horizon: d and s, ``0 <= d <= s``, ``s + d <= 8``.
        bid_n / bid_k: BID samples and K (``bid_k=None``: backward loss only); beta: RTC's max guidance weight.
        hz: ``1 / step_dt`` when ``EPISODE_START`` carries no ``step_dt``.
    """

    def __init__(
        self,
        checkpoint_dir: str,
        rtc_src_path: str | None = None,
        *,
        weak_checkpoint_dir: str | None = None,
        method: Method = "rtc",
        delay: int = 0,
        horizon: int = 1,
        latency: Latency = "steps",
        first_chunk_delayed: bool = False,
        bid_n: int = 32,
        bid_k: int | None = 3,
        beta: float = 5.0,
        flow_steps: int = 5,
        hz: float = 10.0,
        obs_dim: int = 679,
        action_dim: int = 6,
        max_rows: int | None = None,
    ) -> None:
        if method not in ("naive", "rtc", "rtc_hard", "bid"):
            raise ValueError(f"unknown method {method!r}")
        if latency not in ("steps", "wall", "wall_exact"):
            raise ValueError(f"unknown latency {latency!r}")
        if not (0 <= delay <= horizon and horizon >= 1 and horizon + delay <= H):
            raise ValueError(f"need 0 <= delay <= horizon, horizon >= 1, horizon + delay <= {H}")
        if method == "bid" and bid_k is not None and weak_checkpoint_dir is None:
            raise ValueError("BID with bid_k needs weak_checkpoint_dir")
        super().__init__(backlog=64, step_dt=1.0 / hz, max_rows=max_rows)
        self.checkpoint_dir, self.rtc_src_path, self.weak_checkpoint_dir = (
            checkpoint_dir,
            rtc_src_path,
            weak_checkpoint_dir,
        )
        self.method, self.delay, self.horizon, self.latency = method, delay, horizon, latency
        self.first_chunk_delayed = first_chunk_delayed
        self.bid_n, self.bid_k, self.beta, self.flow_steps = bid_n, bid_k, beta, flow_steps
        self.obs_dim, self.action_dim = obs_dim, action_dim
        self._states: dict[str, Any] = {}  # level -> nnx.State
        self._weak: dict[str, Any] = {}
        self._eps_state: dict[int, _RowState] = {}
        self._init_fn: Any = None
        self._sample_fn: Any = None

    def get_action_spec(self) -> dict[str, DimSpec]:
        return {"actions": RAW}

    def get_observation_spec(self) -> dict[str, DimSpec]:
        return {"state": RAW, "language": LANGUAGE}

    # -- model thread ----------------------------------------------------------------------------------------------

    def _load(self, directory: str) -> tuple[Any, dict[str, Any]]:
        import flax.nnx as nnx
        import jax

        model = self._import_model()
        files = sorted(Path(_local_dir(directory)).glob("worlds_l_*.pkl"))
        if not files:
            raise FileNotFoundError(f"no worlds_l_*.pkl in {directory}")
        policy = model.FlowPolicy(
            obs_dim=self.obs_dim, action_dim=self.action_dim, config=model.ModelConfig(), rngs=nnx.Rngs(0)
        )
        graphdef, template = nnx.split(policy)
        states = {}
        for f in files:
            state = jax.tree.map(lambda x: x, template)
            with f.open("rb") as fh:
                state.replace_by_pure_dict(_filter_none(pickle.load(fh)))
            states[f.stem.removeprefix("worlds_l_")] = state
        return graphdef, states

    def _import_model(self) -> Any:
        src = self.rtc_src_path or _rtc_src()
        if not (Path(src) / "model.py").is_file():
            raise ImportError(f"RTC model.py not found in {src}: pass rtc_src_path=<real-time-chunking-kinetix>/src")
        if src not in sys.path:
            sys.path.insert(0, src)
        import model

        return model

    def _warm(self) -> None:
        """Load every level and compile once (model thread, before the first EPISODE_READY)."""
        import flax.nnx as nnx
        import jax

        graphdef, self._states = self._load(self.checkpoint_dir)
        if self.method == "bid" and self.bid_k is not None:
            assert self.weak_checkpoint_dir is not None
            self._weak = self._load(self.weak_checkpoint_dir)[1]
        d, s, n = self.delay, self.horizon, self.flow_steps
        method, beta, bid_n, bid_k = self.method, self.beta, self.bid_n, self.bid_k

        def init(state: Any, key: Any, obs: Any) -> Any:
            return nnx.merge(graphdef, state).action(key, obs, n)

        def sample(state: Any, weak: Any, key: Any, obs: Any, prev: Any) -> Any:
            policy = nnx.merge(graphdef, state)
            if method == "naive":
                return policy.action(key, obs, n)
            if method in ("rtc", "rtc_hard"):
                schedule = "exp" if method == "rtc" else "zeros"
                return policy.realtime_action(key, obs, n, prev, d, H - s, schedule, beta)
            weak_policy = None if bid_k is None else nnx.merge(graphdef, weak)
            return policy.bid_action(key, obs, n, prev, d, H - s, bid_n, bid_weak_policy=weak_policy, bid_k=bid_k)

        self._init_fn, self._sample_fn = jax.jit(init), jax.jit(sample)
        level = next(iter(self._states))
        obs = np.zeros((1, self.obs_dim), np.float32)
        prev = np.zeros((1, H, self.action_dim), np.float32)
        key = jax.random.split(jax.random.PRNGKey(0))[0]  # split compiles too
        self._call_init(level, key, obs)
        self._call_sample(level, key, obs, prev)
        logger.info(
            "RTC %s d=%d s=%d: %d levels loaded and compiled on %s",
            method,
            d,
            s,
            len(self._states),
            jax.devices()[0].platform,
        )

    def _call_init(self, level: str, key: Any, obs: np.ndarray) -> np.ndarray:
        return np.asarray(self._init_fn(self._states[level], key, obs))

    def _call_sample(self, level: str, key: Any, obs: np.ndarray, prev: np.ndarray) -> np.ndarray:
        return np.asarray(self._sample_fn(self._states[level], self._weak.get(level), key, obs, prev))

    def reset_episodes(self, rows: Rows) -> dict[int, Exception] | None:
        if not self._states:
            self._warm()
        failed: dict[int, Exception] = {}
        for i, (rid, episode) in enumerate(zip(rows.ids.tolist(), rows.episodes)):
            level = (episode.config.get("task") or {}).get("level")
            if not isinstance(level, str) or level not in self._states:
                failed[i] = ValueError(f"no checkpoint for level {level!r}; have {sorted(self._states)}")
                continue
            if self.latency != "steps" and episode.mode != "live":
                failed[i] = ValueError(f"latency={self.latency!r} needs a live episode")
                continue
            seed = int(hashlib.sha256(episode.episode_id.encode()).hexdigest(), 16) % 2**31
            rec = None
            if episode.recording_db_path:
                rec = StepRecorder(episode.recording_db_path, episode.session_id, episode.episode_id)
            self._eps_state[rid] = _RowState(level, seed, rec)
        return failed

    def _infer(self, st: _RowState, obs: np.ndarray, prev: np.ndarray | None) -> tuple[np.ndarray, float]:
        import jax

        st.key, key = jax.random.split(jax.random.PRNGKey(st.seed) if st.key is None else st.key)
        t0 = time.perf_counter()
        chunk = self._call_init(st.level, key, obs) if prev is None else self._call_sample(st.level, key, obs, prev)
        return chunk[0], (time.perf_counter() - t0) * 1e3

    def finish_episodes(self, rows: Rows, results: list[EpisodeResult]) -> None:
        for rid in rows.ids.tolist():
            self._eps_state.pop(rid, None)

    # -- event loop ------------------------------------------------------------------------------------------------

    async def run(self, ep: EpisodeContext) -> None:
        st = self._eps_state[ep.row_id]
        d, s = self.delay, self.horizon
        assert ep.step_dt is not None  # hz always gives a default
        land_after = {"steps": None, "wall": (d - 0.5) * ep.step_dt, "wall_exact": d * ep.step_dt}[self.latency]
        plan: _Inference | None = None  # the landed chunk in effect: plan.chunk[i] is the action for step plan.t + i
        pending: dict[int, _Inference] = {}  # boundaries the loop lands itself (step latency, or d = 0), by step
        dispensed_k: int | None = None

        def land(inf: _Inference) -> None:
            nonlocal plan
            if plan is None or inf.k > plan.k:
                inf.landed_wall = time.monotonic()
                plan = inf

        async def infer(inf: _Inference, req: ObservationRequest, prev: np.ndarray | None) -> None:
            obs = np.asarray(req.obs["state"], np.float32)[None]
            inf.chunk, inf.infer_ms = await ep.call(self._infer, st, obs, prev)
            if land_after is not None and d > 0 and inf.k >= 0:  # injected latency, off the model thread
                while (wait := anchors[inf.t] + land_after - time.monotonic()) > 0:
                    await anyio.sleep(wait)  # a repeated observation of step t moves the anchor
                land(inf)
            inf.done.set()

        anchors: dict[int, float] = {}  # boundary step -> admission of its latest observation
        last_j: int | None = None
        async for req in ep.observations():
            # The benchmark's env step, when it sends one: a gated start repeats step 0 until the first action.
            admitted = req.admitted_at if ep.can_publish else time.monotonic()  # live: WallClock, monotonic
            j = int(req.obs.get("env_step", req.row.step))
            row: dict[str, Any] = {"step": req.row.step, "p": j, "admitted_wall": admitted}
            if j == last_j:  # the env did not move: same slot again, latency counted from this observation
                if plan is None:  # a delayed first chunk would never land: the env waits for it
                    raise ValueError(
                        "first_chunk_delayed needs a benchmark that steps on holds (Kinetix: "
                        "start_on_first_action=false)"
                    )
                anchors[j] = admitted
                row["repeat"] = True
            else:
                last_j = j
                anchors[j] = admitted
                if j == 0 and not self.first_chunk_delayed:
                    init = _Inference(-1, 0, -1)
                    await infer(init, req, None)  # official: no delay for the initial chunk
                    land(init)
                if land_after is None and d > 0 and (inf := pending.pop(j - d, None)) is not None:
                    await inf.done.wait()
                    land(inf)
                if j % s == 0:
                    prev = np.zeros((1, H, self.action_dim), np.float32)
                    if plan is not None:
                        tail = plan.chunk[j - plan.t :]
                        prev[0, : len(tail)] = tail  # the official frame shift, zero-padded
                    inf = _Inference(j // s, j, -2 if plan is None else plan.k)
                    if land_after is None or d == 0:
                        pending[j] = inf
                    ep.start_soon(infer, inf, req, prev)
                    if d == 0:
                        await inf.done.wait()
                        land(pending.pop(j))
                    row.update(infer_k=inf.k, prev_k=inf.prev_k)
            age = None if plan is None else j - plan.t
            action = np.zeros(self.action_dim, np.float32)  # nothing planned for this step
            if plan is not None and j - plan.t < H:
                action = plan.chunk[j - plan.t]
            row.update(intended_action=action.tolist(), chunk_k=None if plan is None else plan.k, chunk_age=age)
            if plan is not None and plan.k != dispensed_k:
                dispensed_k = plan.k
                row.update(infer_ms=plan.infer_ms, landed_wall=plan.landed_wall)
            row["dispense_wall"] = time.monotonic()
            if st.recorder is not None:
                st.recorder.record(row)
            if ep.can_publish:
                if plan is not None:  # before the first chunk the client holds
                    await ep.publish({"actions": action}, source=req)
            else:
                await req.reply({"actions": action})

    async def on_episode_end(self, rows: Rows, results: list[EpisodeResult]) -> None:
        recorders = [st.recorder for rid in rows.ids.tolist() if (st := self._eps_state.get(rid)) and st.recorder]
        await super().on_episode_end(rows, results)
        for rec in recorders:  # SQLite I/O off the model thread
            try:
                await _run_in_thread(rec.close)
            except Exception:
                logger.exception("server step rows were not saved")


if __name__ == "__main__":
    from vla_eval.model_servers.serve import run_server

    run_server(KinetixRTCServer)
