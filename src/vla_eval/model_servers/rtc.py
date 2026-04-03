# /// script
# requires-python = "~=3.11"
# dependencies = [
#     "vla-eval",
#     "jax[cuda12]==0.4.35",
#     "flax==0.10.0",
#     "distrax",
#     "einops",
#     "numpy>=1.24,<2",
# ]
#
# [tool.uv.sources]
# vla-eval = { path = "../../..", editable = true }
#
# [tool.uv]
# exclude-newer = "2026-02-24T00:00:00Z"
# ///
"""RTC (Real-Time Chunking) diffusion policy model server.

Loads trained RTC flow-matching MLP-Mixer checkpoints and runs inference
for Kinetix environments. The RTC policy takes symbolic observations
(~679-dim flat vector) and outputs action chunks
(prediction horizon H=8, action dim=6).

Each Kinetix level has its own checkpoint (``worlds_l_<level>.pkl``).
The server loads all checkpoints from a directory at startup and selects
the appropriate model on each ``episode_start`` based on the task's level.

Checkpoint availability:
    RTC checkpoints are stored on Google Cloud Storage at
    ``gs://rtc-assets/bc/`` (imitation learning policies). Download with::

        mkdir -p /path/to/checkpoint_dir/
        gsutil -m cp -r gs://rtc-assets/bc/0/policies/* /path/to/checkpoint_dir/

    The checkpoint directory should contain files named
    ``worlds_l_<level_name>.pkl``.

CPU-only (no GPU)::

        JAX_PLATFORMS=cpu uv run src/vla_eval/model_servers/rtc.py ...

Reference:
    Real-Time Execution of Action Chunking Flow Policies
    (arXiv:2506.07339, Physical Intelligence)
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import pickle
import sys
import time
from pathlib import Path
from typing import Any

from vla_eval.specs import LANGUAGE, RAW, DimSpec
from vla_eval.types import Action, Observation

# Workaround: nvidia-cuda-nvcc-cu12 exposes __file__=None (namespace package),
# which crashes jax 0.4.35's CUDA path detection.  Patch before any jax import.
try:
    import nvidia.cuda_nvcc as _cuda_nvcc

    if _cuda_nvcc.__file__ is None and _cuda_nvcc.__path__:
        import os as _os

        _cuda_nvcc.__file__ = _os.path.join(_cuda_nvcc.__path__[0], "__init__.py")
except ImportError:
    pass

import numpy as np

from vla_eval.model_servers.base import SessionContext
from vla_eval.model_servers.predict import PredictModelServer
from vla_eval.model_servers.serve import serve

logger = logging.getLogger(__name__)

# RTC default configuration (matching the paper)
DEFAULT_ACTION_DIM = 6
DEFAULT_PREDICTION_HORIZON = 8
DEFAULT_DENOISING_STEPS = 5


def _filter_none(d: dict) -> dict:
    """Recursively remove None values from a nested dict (old flax ckpt compat)."""
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            filtered = _filter_none(v)
            if filtered:
                out[k] = filtered
        elif v is not None:
            out[k] = v
    return out


class RTCModelServer(PredictModelServer):
    """RTC flow-matching diffusion policy model server.

    Loads per-level RTC checkpoints from ``checkpoint_dir`` and selects
    the correct model on each episode based on the task's ``level`` field.

    The Kinetix environment already stacks observation history internally,
    so ``obs_history`` defaults to 1 (no additional stacking needed).

    Args:
        checkpoint_dir: Directory containing ``worlds_l_<level>.pkl`` files.
        rtc_src_path: Path to the RTC repo ``src/`` directory (added to
            ``sys.path`` so ``import model`` works).
        action_dim: Action space dimensionality (default 6 for Kinetix large).
        obs_dim: Observation dimensionality (default 679 for Kinetix large
            symbolic flat). All RTC levels share the same obs_dim.
        prediction_horizon: Number of future actions to predict (default 8).
        denoising_steps: Number of flow-matching denoising steps (default 5).
        obs_history: Number of observation history frames to stack (default 1,
            since the env already provides stacked observations).
        chunk_size: Action chunk size for the harness (default 8 = prediction_horizon).
        action_ensemble: Ensemble strategy for overlapping chunks.
        inference_delay: Artificial delay in seconds added before each predict()
            call, for benchmarking inference latency sensitivity.
    """

    def __init__(
        self,
        checkpoint_dir: str | None = None,
        rtc_src_path: str | None = None,
        action_dim: int = DEFAULT_ACTION_DIM,
        obs_dim: int = 679,
        prediction_horizon: int = DEFAULT_PREDICTION_HORIZON,
        denoising_steps: int = DEFAULT_DENOISING_STEPS,
        obs_history: int = 1,
        inference_delay: float = 0.0,
        *,
        chunk_size: int = DEFAULT_PREDICTION_HORIZON,
        action_ensemble: str = "newest",
        **kwargs: Any,
    ) -> None:
        super().__init__(chunk_size=chunk_size, action_ensemble=action_ensemble, **kwargs)
        self.checkpoint_dir = checkpoint_dir
        self.rtc_src_path = rtc_src_path
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.prediction_horizon = prediction_horizon
        self.denoising_steps = denoising_steps
        self.obs_history = obs_history
        self.inference_delay = inference_delay

        # Per-level model cache: level_name -> (policy, jit_action_fn)
        self._models: dict[str, Any] = {}
        self._current_model: Any = None
        self._current_jit_action: Any = None
        self._obs_buffer: dict[str, list[np.ndarray]] = {}
        self._rtc_model_module: Any = None
        # Per-session RNG state: split on each predict() call for stochastic policy
        self._session_rng: dict[str, Any] = {}

    def get_action_spec(self) -> dict[str, DimSpec]:
        return {"actions": RAW}

    def get_observation_spec(self) -> dict[str, DimSpec]:
        return {"state": RAW, "language": LANGUAGE}

    def _ensure_rtc_import(self) -> Any:
        """Import the RTC model module, adding rtc_src_path to sys.path."""
        if self._rtc_model_module is not None:
            return self._rtc_model_module

        if self.rtc_src_path is not None and self.rtc_src_path not in sys.path:
            sys.path.insert(0, self.rtc_src_path)

        # flax >=0.10 removed nnx.List; RTC model.py still references it.
        import flax.nnx as _nnx

        if not hasattr(_nnx, "List"):
            _nnx.List = list

        import importlib.util

        if importlib.util.find_spec("model") is None:
            raise ImportError(
                "RTC model package not found. Either set --rtc_src_path to the RTC repo src/ directory, "
                "or: git clone https://github.com/Physical-Intelligence/real-time-chunking-kinetix && "
                "pass --rtc_src_path /path/to/real-time-chunking-kinetix/src"
            )

        import model as rtc_model

        self._rtc_model_module = rtc_model
        return rtc_model

    def _load_policy(self, level: str, obs_dim: int) -> Any:
        """Load a single level's FlowPolicy from pickle checkpoint."""
        import flax.nnx as nnx

        rtc_model = self._ensure_rtc_import()

        assert self.checkpoint_dir is not None
        ckpt_path = Path(self.checkpoint_dir) / f"worlds_l_{level}.pkl"
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"RTC checkpoint not found: {ckpt_path}")

        config = rtc_model.ModelConfig()
        policy = rtc_model.FlowPolicy(obs_dim=obs_dim, action_dim=self.action_dim, config=config, rngs=nnx.Rngs(0))

        with open(ckpt_path, "rb") as f:
            state_dict = _filter_none(pickle.load(f))

        graphdef, state = nnx.split(policy)
        state.replace_by_pure_dict(state_dict)
        return nnx.merge(graphdef, state)

    def _load_all_models(self) -> None:
        """Load all checkpoint files from checkpoint_dir.

        Each file is named ``worlds_l_<level>.pkl``. We probe obs_dim from
        a test environment reset.
        """
        if self.checkpoint_dir is None:
            raise ValueError(
                "checkpoint_dir not provided. Download checkpoints from gs://rtc-assets/bc/ "
                "and pass --checkpoint_dir /path/to/policies/"
            )

        import jax

        ckpt_dir = Path(self.checkpoint_dir)
        if not ckpt_dir.is_dir():
            raise FileNotFoundError(f"checkpoint_dir not found: {ckpt_dir}")

        pkl_files = sorted(ckpt_dir.glob("worlds_l_*.pkl"))
        if not pkl_files:
            raise FileNotFoundError(f"No worlds_l_*.pkl files in {ckpt_dir}")

        logger.info("Loading %d RTC checkpoints from %s", len(pkl_files), ckpt_dir)

        obs_dim = self.obs_dim

        for pkl_path in pkl_files:
            # Extract level name: worlds_l_<level>.pkl -> <level>
            level = pkl_path.stem.removeprefix("worlds_l_")

            policy = self._load_policy(level, obs_dim)
            jit_action = jax.jit(lambda rng, obs, p=policy: p.action(rng, obs, self.denoising_steps))

            # Warmup JIT
            dummy_obs = np.zeros((1, obs_dim), dtype=np.float32)
            _ = jit_action(jax.random.PRNGKey(0), dummy_obs)
            jax.block_until_ready(_)

            self._models[level] = (policy, jit_action)
            logger.info("  Loaded level=%s (obs_dim=%d)", level, obs_dim)

        logger.info("All %d models loaded.", len(self._models))

    async def on_episode_start(self, config: dict[str, Any], ctx: SessionContext) -> None:
        """Select the correct per-level model based on task metadata."""
        import jax
        from anyio.to_thread import run_sync

        # Load all models on first episode (in thread to avoid blocking the event loop
        # during JAX JIT warmup, which can take 30s+ and cause ping timeout disconnects).
        if not self._models:
            await run_sync(self._load_all_models)

        task = config.get("task", {})
        level = task.get("level")
        if level is None:
            raise ValueError("Task must have a 'level' field for RTC model selection")

        if level not in self._models:
            raise ValueError(f"No RTC model loaded for level={level!r}. Available: {list(self._models.keys())}")

        _policy, jit_action = self._models[level]
        self._current_jit_action = jit_action

        # Create a unique RNG per episode from the episode_id hash.
        # This ensures the diffusion policy's noise sampling varies across episodes,
        # producing different trajectories even when the env initial state is identical.
        episode_seed = int(hashlib.sha256(ctx.episode_id.encode()).hexdigest(), 16) % (2**31)
        self._session_rng[ctx.session_id] = jax.random.PRNGKey(episode_seed)
        logger.info("Selected model for level=%s session=%s (episode_seed=%d)", level, ctx.session_id, episode_seed)

        await super().on_episode_start(config, ctx)

    def _get_obs_with_history(self, obs: Observation, ctx: SessionContext) -> np.ndarray:
        """Build observation, optionally with history stacking.

        When ``obs_history == 1`` (default), returns the state vector directly
        with no stacking — the Kinetix env already handles observation history.
        """
        state = obs.get("state")
        if state is None:
            for v in obs.values():
                if isinstance(v, np.ndarray) and v.ndim == 1:
                    state = v
                    break
        if state is None:
            raise ValueError("RTC requires symbolic state in observation (key 'state' or a 1D array)")

        state = np.asarray(state, dtype=np.float32)

        if self.obs_history <= 1:
            return state

        sid = ctx.session_id
        if ctx.is_first or sid not in self._obs_buffer:
            self._obs_buffer[sid] = [state.copy() for _ in range(self.obs_history)]
        else:
            self._obs_buffer[sid].append(state.copy())
            if len(self._obs_buffer[sid]) > self.obs_history:
                self._obs_buffer[sid] = self._obs_buffer[sid][-self.obs_history :]

        return np.concatenate(self._obs_buffer[sid])

    def predict(self, obs: Observation, ctx: SessionContext) -> Action:
        import jax

        if self.inference_delay > 0:
            time.sleep(self.inference_delay)

        if self._current_jit_action is None:
            raise RuntimeError("No model selected — on_episode_start must be called first")

        obs_vec = self._get_obs_with_history(obs, ctx)

        # Split the per-episode RNG so each step gets a unique key.
        # This makes the diffusion policy stochastic across episodes.
        # Fallback: if OBSERVATION arrives before EPISODE_START (e.g. reconnect),
        # initialise a deterministic RNG so predict() doesn't crash.
        if ctx.session_id not in self._session_rng:
            logger.warning(
                "predict() called without prior EPISODE_START session=%s — using fallback RNG", ctx.session_id[:8]
            )
            self._session_rng[ctx.session_id] = jax.random.PRNGKey(0)
        rng = self._session_rng[ctx.session_id]
        rng, step_rng = jax.random.split(rng)
        self._session_rng[ctx.session_id] = rng

        # Run flow-matching inference: expects (batch, obs_dim)
        actions = self._current_jit_action(step_rng, obs_vec[None])[0]
        actions = np.asarray(actions, dtype=np.float32)

        # Ensure correct shape: (prediction_horizon, action_dim)
        if actions.ndim == 1:
            actions = actions.reshape(self.prediction_horizon, self.action_dim)

        return {"actions": actions}

    async def on_episode_end(self, result: dict[str, Any], ctx: SessionContext) -> None:
        """Clear observation history buffer and RNG state on episode end."""
        self._obs_buffer.pop(ctx.session_id, None)
        self._session_rng.pop(ctx.session_id, None)
        await super().on_episode_end(result, ctx)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RTC diffusion policy model server")
    parser.add_argument("--checkpoint_dir", default=None, help="Directory with worlds_l_<level>.pkl files")
    parser.add_argument("--rtc_src_path", default=None, help="Path to RTC repo src/ directory")
    parser.add_argument("--action_dim", type=int, default=DEFAULT_ACTION_DIM)
    parser.add_argument("--obs_dim", type=int, default=679, help="Observation dimensionality (679 for Kinetix large)")
    parser.add_argument("--prediction_horizon", type=int, default=DEFAULT_PREDICTION_HORIZON)
    parser.add_argument("--denoising_steps", type=int, default=DEFAULT_DENOISING_STEPS)
    parser.add_argument("--obs_history", type=int, default=1)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--chunk_size", type=int, default=DEFAULT_PREDICTION_HORIZON)
    parser.add_argument("--action_ensemble", default="newest")
    parser.add_argument("--inference_delay", type=float, default=0.0, help="Artificial inference delay in seconds")
    parser.add_argument("--ci", action="store_true", help="Enable Continuous Inference")
    parser.add_argument("--laas", action="store_true", help="Enable LAAS (requires --ci)")
    parser.add_argument("--hz", type=float, default=10.0)
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )

    if args.laas and not args.ci:
        parser.error("--laas requires --ci")

    server = RTCModelServer(
        checkpoint_dir=args.checkpoint_dir,
        rtc_src_path=args.rtc_src_path,
        action_dim=args.action_dim,
        obs_dim=args.obs_dim,
        prediction_horizon=args.prediction_horizon,
        denoising_steps=args.denoising_steps,
        obs_history=args.obs_history,
        inference_delay=args.inference_delay,
        chunk_size=args.chunk_size,
        action_ensemble=args.action_ensemble,
        continuous_inference=args.ci,
        laas=args.laas,
        hz=args.hz,
    )

    if args.checkpoint_dir:
        logger.info("Pre-loading all models...")
        server._load_all_models()

    logger.info("Starting RTC model server on ws://%s:%d", args.host, args.port)
    serve(server, host=args.host, port=args.port)
