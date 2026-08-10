"""RoboMME benchmark implementation using ManiSkill3 fork + SAPIEN.

Creates a fresh environment per episode via BenchmarkEnvBuilder.
Each episode produces a conditioning video (via motion planning) that
is sent to the model server as ``video_history`` on the first observation.
"""

from __future__ import annotations

import logging
import os
import re
import sys
from typing import Any, Literal

import numpy as np

from vla_eval.benchmarks.base import StepBenchmark, StepResult
from vla_eval.render import apply_env, lavapipe_cpu_env, resolve_lavapipe_icd
from vla_eval.specs import IMAGE_RGB, LANGUAGE, RAW, DimSpec
from vla_eval.types import Action, EpisodeResult, Observation, Task

logger = logging.getLogger(__name__)

# A grounded subgoal that hasn't had its `<obj_center>`-style placeholders
# substituted with image coords still has bracketed identifiers (alpha/_).
# Filled coords look like `<70, 84>` — the first char inside `<` is a digit.
_UNFILLED_PLACEHOLDER_RE = re.compile(r"<[A-Za-z_]")

_DEFAULT_TASK_LIST = [
    "PickXtimes",
    "StopCube",
    "SwingXtimes",
    "BinFill",
    "VideoUnmaskSwap",
    "VideoUnmask",
    "ButtonUnmaskSwap",
    "ButtonUnmask",
    "VideoRepick",
    "VideoPlaceButton",
    "VideoPlaceOrder",
    "PickHighlight",
    "InsertPeg",
    "MoveCube",
    "PatternLock",
    "RouteStick",
]


class RoboMMEBenchmark(StepBenchmark):
    """RoboMME (Memory-Augmented Manipulation Evaluation) benchmark.

    16 tasks across 4 cognitive suites (Counting, Permanence, Reference,
    Imitation).  Built on a ManiSkill3 fork with SAPIEN rendering.

    Non-obvious behaviors:
        - **Conditioning video**: On ``reset``, the environment runs motion
          planning to produce a demonstration trajectory.  These frames are
          sent as ``video_history`` in the first observation only.
        - **Fresh env per episode**: ``BenchmarkEnvBuilder.make_env_for_episode``
          creates a full wrapper chain for each episode.
        - **Error obs**: ``FailAwareWrapper`` returns ``obs=None`` on exception;
          ``EndeffectorDemonstrationWrapper`` returns ``obs={}`` on IK failure.
          Both are handled gracefully in ``make_obs``.
        - **Torch scalars**: ``reward``, ``terminated``, ``truncated`` may be
          torch tensors — always cast with ``float()`` / ``bool()``.

    Args:
        tasks: Subset of task names to evaluate.  ``None`` runs all 16.
        action_space: ``"joint_angle"`` (8D) or ``"ee_pose"`` (7D).
        dataset: Dataset split — ``"test"``, ``"val"``, or ``"train"``.
        max_steps: Maximum steps per episode (paper default: 1300).
        send_wrist_image: Include wrist camera in observations.
        send_state: Include proprioceptive state in observations.
        send_video_history: Send conditioning video on the first observation.
        send_subgoal: Attach the per-step subgoal text to ``obs["subgoal"]``.
        subgoal_mode: ``"grounded"`` sends ``info['grounded_subgoal_online']``
            (subgoal with image-coord placeholders filled, e.g. ``"pick up the
            green cube at <77, 170>"``); ``"simple"`` sends
            ``info['simple_subgoal_online']`` (no coords).  Both come from
            ``DemonstrationWrapper`` in the upstream robomme env.  ``"grounded"``
            falls back to simple if grounded is empty.
    """

    _ALL_RECORD_FIELDS = frozenset(
        {"simple_subgoal_online", "grounded_subgoal_online", "reward", "state_fq", "terminated"}
    )

    # Env applied by whichever call bound the renderer: None until bound, {} on the
    # native GPU path, non-empty iff lavapipe is engaged.
    _render_env: dict[str, str] | None = None

    # Software Vulkan via lavapipe on cpu (the shipped-config default); gpu opts
    # into SAPIEN's native NVIDIA path, which hangs on a small subset of hosts.
    render_backends = frozenset({"gpu", "cpu"})

    @classmethod
    def _mark_render_configured(cls, env: dict[str, str]) -> dict[str, str]:
        RoboMMEBenchmark._render_env = dict(env)
        return dict(env)

    @classmethod
    def configure_render(cls, mode: str) -> dict[str, str]:
        """``cpu`` (the shipped-config default) engages lavapipe (~5-10x slower);
        ``gpu`` opts into SAPIEN's native NVIDIA path.

        cpu is the default because the native path hangs at the first capture on
        some hosts (SAPIEN #290, hardware dependent) and no startup probe can
        certify a host: a render-only check passes where the combined CUDA plus
        rendering workload later hangs (issue #112).

        Called once per benchmark entry; only the first call binds the
        per-process renderer, the rest replay what it applied.
        """
        legacy = os.environ.get("ROBOMME_USE_LAVAPIPE")
        if legacy is not None:
            raise RuntimeError(
                "ROBOMME_USE_LAVAPIPE={!r} is set, but the variable has been removed. "
                "Rendering is now selected by the run-level 'render' key: the shipped RoboMME "
                "configs default to 'render: cpu' (lavapipe, completes on every host); pass "
                "--render gpu to opt into the native NVIDIA path on a known-good host.".format(legacy)
            )
        if RoboMMEBenchmark._render_env is not None:
            if mode == "cpu" and not RoboMMEBenchmark._render_env:
                # Renderer already bound to the native GPU path; the Vulkan ICD is
                # loaded at first `import sapien.render` and cannot be re-bound.
                raise RuntimeError(
                    "render: cpu requested after this process already initialised SAPIEN on the "
                    "native GPU path; lavapipe can no longer be engaged."
                )
            return dict(RoboMMEBenchmark._render_env)
        if mode != "cpu":
            logger.warning(
                "RoboMME render: gpu uses SAPIEN's native NVIDIA Vulkan path, which hangs at the "
                "first image capture on a small subset of hosts (hardware dependent). If this run "
                "stalls at episode start with 100%% GPU utilisation, switch back to the default "
                "render: cpu."
            )
            return cls._mark_render_configured({})
        applied = cls._engage_lavapipe()
        if applied is None:
            raise RuntimeError(
                "render: cpu needs lavapipe software Vulkan, which could not be engaged. "
                "See earlier log lines for the reason (sapien.render already imported, ICD missing, etc.)."
            )
        logger.warning(
            "RoboMME is software-rendering through lavapipe (~4-12 fps at 256x256, roughly 5-10x "
            "slower than the native NVIDIA path). On a host where the native path is known good, "
            "pass --render gpu to speed up the simulator."
        )
        return cls._mark_render_configured(applied)

    def __init__(
        self,
        tasks: list[str] | None = None,
        action_space: str = "joint_angle",
        dataset: str = "test",
        max_steps: int = 1300,
        send_wrist_image: bool = True,
        send_state: bool = True,
        send_video_history: bool = True,
        send_subgoal: bool = False,
        subgoal_mode: Literal["grounded", "simple"] = "grounded",
    ) -> None:
        super().__init__()
        if subgoal_mode not in ("grounded", "simple"):
            raise ValueError(f"subgoal_mode must be 'grounded' or 'simple', got {subgoal_mode!r}")
        self.tasks = tasks or list(_DEFAULT_TASK_LIST)
        self.action_space = action_space
        self.dataset = dataset
        self.max_steps = max_steps
        self.send_wrist_image = send_wrist_image
        self.send_state = send_state
        self.send_video_history = send_video_history
        self.send_subgoal = send_subgoal
        self.subgoal_mode = subgoal_mode

        self._env: Any = None
        self._task: Task | None = None
        self._task_description: str = ""
        self._video_frames: list[np.ndarray] = []
        self._wrist_video_frames: list[np.ndarray] = []
        self._current_subgoal: str = ""

    def get_tasks(self) -> list[Task]:
        return [{"name": t, "env_id": t} for t in self.tasks]

    @staticmethod
    def _engage_lavapipe() -> dict[str, str] | None:
        """Apply the three-piece lavapipe patch + perf-tuning env vars.

        Must be called BEFORE ``import sapien.render`` in this process for
        ``VK_ICD_FILENAMES`` and ``LP_NUM_THREADS`` to take effect (Vulkan
        ICD is loaded at first ``import sapien.render``).

        Returns the applied env on success, None if the patch could not be
        applied (sapien.render already imported, lavapipe ICD missing, etc.).
        The caller is responsible for treating None as fatal — silently
        continuing on the native path would hang on affected hosts.
        """
        if "sapien.render" in sys.modules:
            logger.error(
                "Cannot engage lavapipe: sapien.render is already imported. "
                "VK_ICD_FILENAMES / LP_NUM_THREADS only take effect on first "
                "Vulkan init. Configure render: cpu before any sapien import."
            )
            return None

        lavapipe_icd = resolve_lavapipe_icd("ROBOMME_LAVAPIPE_ICD")
        if lavapipe_icd is None:
            logger.error("Lavapipe ICD not found; cannot engage lavapipe rendering")
            return None

        applied = apply_env(lavapipe_cpu_env(lavapipe_icd))
        logger.info("SAPIEN rendering: using lavapipe software Vulkan (%s)", lavapipe_icd)

        import sapien.render as sr

        _OrigRenderSystem = sr.RenderSystem

        def _lavapipe_render_system(*args, **kwargs):
            return _OrigRenderSystem()

        sr.RenderSystem = _lavapipe_render_system

        # Patch parse_sim_and_render_backend in BOTH places: the source module
        # (so any later imports get the patched version) AND already-imported
        # `mani_skill.envs.sapien_env` (which captured the unpatched reference
        # at its own import time via `from ... import parse_sim_and_render_backend`).
        try:
            from mani_skill.envs.utils.system import backend as _backend_mod

            _orig_parse = _backend_mod.parse_sim_and_render_backend

            def _patched_parse(sim_backend, render_backend):
                result = _orig_parse(sim_backend, render_backend)
                if result.render_backend == "sapien_cuda":
                    result.render_backend = "sapien_cpu"
                return result

            _backend_mod.parse_sim_and_render_backend = _patched_parse

            import mani_skill.envs.sapien_env

            mani_skill.envs.sapien_env.parse_sim_and_render_backend = _patched_parse
        except Exception as e:
            logger.warning("Could not patch mani_skill render backend to sapien_cpu: %s", e)

        return applied

    def reset(self, task: Task) -> Any:
        # Bind the renderer if the orchestrator's configure_render hasn't already:
        # direct users of this class get the native path (plus its host warning).
        if RoboMMEBenchmark._render_env is None:
            self.configure_render("gpu")
        import robomme.robomme_env  # noqa: F401 — registers gym environments
        from robomme.env_record_wrapper import BenchmarkEnvBuilder

        # Close previous env — fresh env per episode
        if self._env is not None:
            try:
                self._env.close()
            except Exception:
                pass

        episode_idx = task.get("episode_idx", 0)
        self._task = task
        builder = BenchmarkEnvBuilder(
            env_id=task["env_id"],
            dataset=self.dataset,
            action_space=self.action_space,
            gui_render=False,
            max_steps=self.max_steps,
        )
        self._env = builder.make_env_for_episode(episode_idx)
        obs_batch, info_flat = self._env.reset()

        # Store conditioning video frames (demo trajectory, excluding final init frame)
        self._video_frames = list(obs_batch["front_rgb_list"][:-1])
        if self.send_wrist_image:
            self._wrist_video_frames = list(obs_batch.get("wrist_rgb_list", [])[:-1])

        # Extract task description
        task_goal = info_flat["task_goal"]
        self._task_description = task_goal[0] if isinstance(task_goal, list) else str(task_goal)

        if self.send_subgoal:
            self._current_subgoal = self._extract_subgoal(info_flat)

        self._recorder.record_video(self._extract_frame(obs_batch))
        return obs_batch

    def step(self, action: Action) -> StepResult:
        raw_action = action.get("actions", action.get("action"))
        if raw_action is None:
            raise ValueError("Action dict must contain 'actions' or 'action' key")
        if hasattr(raw_action, "flatten"):
            raw_action = raw_action.flatten().tolist()
        elif not isinstance(raw_action, list):
            raw_action = list(raw_action)

        assert self._env is not None
        obs, reward, terminated, truncated, info = self._env.step(raw_action)

        if self.send_subgoal:
            self._current_subgoal = self._extract_subgoal(info)

        terminated = bool(terminated)
        truncated = bool(truncated)
        reward = float(reward)
        done = terminated or truncated or info.get("status") == "error"

        row: dict[str, Any] = {
            "simple_subgoal_online": info.get("simple_subgoal_online", ""),
            "grounded_subgoal_online": info.get("grounded_subgoal_online", ""),
            "reward": reward,
            "terminated": terminated,
        }
        if isinstance(obs, dict):
            state = obs.get("state_fq")
            if state is not None:
                row["state_fq"] = state.tolist() if hasattr(state, "tolist") else list(state)
        self._recorder.record_video(self._extract_frame(obs))
        self._recorder.record_step(**row)

        return StepResult(obs=obs, reward=reward, done=done, info=info)

    @staticmethod
    def _extract_frame(raw_obs: Any) -> np.ndarray | None:
        if not isinstance(raw_obs, dict):
            return None
        front_list = raw_obs.get("front_rgb_list", [])
        if not front_list:
            return None
        return np.asarray(front_list[-1])

    def _extract_subgoal(self, info: dict[str, Any]) -> str:
        """Pick the configured subgoal text from the env's info dict.

        ``DemonstrationWrapper`` always populates ``simple_subgoal_online`` and
        ``grounded_subgoal_online``; grounded may be empty OR may still hold
        the raw placeholder template (e.g. ``"pick up the green cube at
        <obj_center>"``) when segmentation hasn't been computed for the
        current frame. Fall back to simple in either case so the model never
        sees an unfilled template.
        """
        if self.subgoal_mode == "grounded":
            grounded = str(info.get("grounded_subgoal_online") or "")
            if grounded and not _UNFILLED_PLACEHOLDER_RE.search(grounded):
                return grounded
        return str(info.get("simple_subgoal_online") or "")

    def make_obs(self, raw_obs: Any, task: Task) -> Observation:
        # Handle error cases (FailAwareWrapper → None, IK failure → {})
        if not raw_obs:
            return {"images": {}, "task_description": self._task_description}

        front_list = raw_obs.get("front_rgb_list", [])
        if not front_list:
            return {"images": {}, "task_description": self._task_description}

        front = front_list[-1]

        obs: dict[str, Any] = {
            "images": {"agentview": front},
            "task_description": self._task_description,
        }

        if self.send_wrist_image:
            wrist_list = raw_obs.get("wrist_rgb_list")
            if wrist_list:
                obs["images"]["wrist"] = wrist_list[-1]

        if self.send_state:
            joint = np.asarray(raw_obs["joint_state_list"][-1], dtype=np.float64)
            gripper = np.asarray(raw_obs["gripper_state_list"][-1], dtype=np.float64)[:1]
            obs["states"] = np.concatenate([joint, gripper]).astype(np.float32)

        if self.send_video_history and self._video_frames:
            obs["video_history"] = list(self._video_frames)
            if self.send_wrist_image and self._wrist_video_frames:
                obs["wrist_video_history"] = list(self._wrist_video_frames)
            obs["episode_restart"] = True
            # Clear — sent only once per episode
            self._video_frames = []
            self._wrist_video_frames = []

        if self.send_subgoal:
            obs["subgoal"] = self._current_subgoal

        return obs

    def check_done(self, step_result: StepResult) -> bool:
        return step_result.done

    def get_step_result(self, step_result: StepResult) -> EpisodeResult:
        success = step_result.info.get("status") == "success"
        return {"success": success}

    def get_metadata(self) -> dict[str, Any]:
        return {"max_steps": self.max_steps, "action_space": self.action_space}

    def get_action_spec(self) -> dict[str, DimSpec]:
        return {"action": RAW}

    def get_observation_spec(self) -> dict[str, DimSpec]:
        spec: dict[str, DimSpec] = {
            "agentview": IMAGE_RGB,
            "language": LANGUAGE,
        }
        if self.send_wrist_image:
            spec["wrist"] = IMAGE_RGB
        if self.send_state:
            spec["state"] = RAW
        if self.send_subgoal:
            spec["subgoal"] = LANGUAGE
        return spec

    def cleanup(self) -> None:
        if self._env is not None:
            try:
                self._env.close()
            except Exception:
                pass
            self._env = None
        self._video_frames = []
        self._wrist_video_frames = []
        self._task = None
