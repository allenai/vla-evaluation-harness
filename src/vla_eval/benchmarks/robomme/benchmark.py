"""RoboMME benchmark implementation using ManiSkill3 fork + SAPIEN.

Creates a fresh environment per episode via BenchmarkEnvBuilder.
Each episode produces a conditioning video (via motion planning) that
is sent to the model server as ``video_history`` on the first observation.
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
import sys
from typing import Any, Literal

import numpy as np

from vla_eval.benchmarks.base import StepBenchmark, StepResult
from vla_eval.benchmarks.data_recording import EpisodeRecorder, RecordingConfig
from vla_eval.specs import IMAGE_RGB, LANGUAGE, RAW, DimSpec
from vla_eval.types import Action, EpisodeResult, Observation, Task

logger = logging.getLogger(__name__)

# A grounded subgoal that hasn't had its `<obj_center>`-style placeholders
# substituted with image coords still has bracketed identifiers (alpha/_).
# Filled coords look like `<70, 84>` — the first char inside `<` is a digit.
_UNFILLED_PLACEHOLDER_RE = re.compile(r"<[A-Za-z_]")

# Probe script for ROBOMME_USE_LAVAPIPE=auto. Runs in a child process so a hang
# in SAPIEN's Vulkan instance creation can be timed out without poisoning the
# parent process (SAPIEN's VkInstance is created at import time and cannot be
# reset in-process). subprocess.run's timeout is the watchdog — the child has
# no internal timer.
_NATIVE_PROBE = """
import sapien
import sapien.render
import sapien.physx as physx
rs = sapien.render.RenderSystem('cuda:0')
scene = sapien.Scene([physx.PhysxCpuSystem(), rs])
cam = scene.add_camera(name='t', width=64, height=64, fovy=1.0, near=0.01, far=10.0)
cam.set_pose(sapien.Pose([0, 0, 1]))
scene.step()
scene.update_render()
cam.take_picture()
cam.get_picture('Color')
"""


def native_render_path_works(timeout_s: int = 15) -> bool:
    """Probe whether SAPIEN's native NVIDIA Vulkan path works on this host.

    Spawns a child Python that does ``RenderSystem('cuda:0')`` → scene → camera
    → ``take_picture`` → ``get_picture('Color')``. The hang on affected hosts
    shows up at ``get_picture`` (Vulkan fence wait that never signals) —
    ``subprocess.run`` times out and SIGKILLs the child while we abort it.

    Returns True if the child completes successfully within the timeout, False
    if it hangs, crashes, or any unexpected error occurs. Conservative: any
    non-zero exit means "fall back to lavapipe".

    Public so launchers can health-check a host before scheduling work on it.
    """
    try:
        result = subprocess.run(
            [sys.executable, "-c", _NATIVE_PROBE],
            timeout=timeout_s,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    except Exception as e:
        logger.warning("Native render-path probe error: %s", e)
        return False


def _resolve_lavapipe_icd() -> str | None:
    """Find the lavapipe ICD path. Returns None if not available.

    Honors ``ROBOMME_LAVAPIPE_ICD`` if explicitly set — falling back silently
    when an explicit user setting points to a missing file would be surprising,
    so that case logs an error and returns None instead. The implicit default
    paths (``/opt/lavapipe/lvp_icd.json`` then the Mesa-shipped
    ``/usr/share/vulkan/icd.d/lvp_icd.x86_64.json``) are tried in order.
    """
    user_icd = os.environ.get("ROBOMME_LAVAPIPE_ICD")
    if user_icd:
        if os.path.isfile(user_icd):
            return user_icd
        logger.error(
            "ROBOMME_LAVAPIPE_ICD=%s does not exist; refusing to silently fall back to a different ICD path",
            user_icd,
        )
        return None
    for candidate in ("/opt/lavapipe/lvp_icd.json", "/usr/share/vulkan/icd.d/lvp_icd.x86_64.json"):
        if os.path.isfile(candidate):
            return candidate
    return None


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
        recording: A ``RecordingConfig`` dict (or ``None`` to disable).
            Controls per-episode video + JSONL data recording.
    """

    _ALL_RECORD_FIELDS = frozenset({"gt_subgoal", "grounded_subgoal", "reward", "robot_state", "terminated"})

    _rendering_configured: bool = False

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
        recording: dict[str, Any] | None = None,
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
        rec = RecordingConfig(**recording) if recording else None
        if rec and rec.step_fields:
            unknown = set(rec.step_fields) - self._ALL_RECORD_FIELDS
            if unknown:
                raise ValueError(f"Unknown step_fields: {unknown}. Valid: {sorted(self._ALL_RECORD_FIELDS)}")
        self._record_fields: set[str] = (
            set(rec.step_fields) if rec and rec.step_fields else set(self._ALL_RECORD_FIELDS)
        )
        self._recorder: EpisodeRecorder | None = (
            EpisodeRecorder(output_dir=rec.output_dir, record_video=rec.record_video, record_step=rec.record_step)
            if rec
            else None
        )
        self._step_counter: int = 0

        self._env: Any = None
        self._task: Task | None = None
        self._task_description: str = ""
        self._video_frames: list[np.ndarray] = []
        self._wrist_video_frames: list[np.ndarray] = []
        self._current_subgoal: str = ""

    def get_tasks(self) -> list[Task]:
        return [{"name": t, "env_id": t} for t in self.tasks]

    @staticmethod
    def _setup_rendering() -> None:
        """Optionally switch SAPIEN to lavapipe software Vulkan.

        On a small subset of hosts, SAPIEN's ``RenderSystem("cuda:0")`` path
        hangs at the first ``take_picture`` — observed as 100%% GPU util with
        no progress past ``_setup_scene``. Symptom matches SAPIEN #290
        (closed as host-specific). Empirically, the kernel module flavor is
        not the discriminator: across our DGX-H100 fleet, both the closed
        ``NVIDIA`` and the open ``Dual MIT/GPL`` modules reproduce the hang
        on some nodes and not others, with byte-identical packages. The
        difference appears to be hardware/firmware-level.

        ``ROBOMME_USE_LAVAPIPE`` (env var) controls the workaround:
            - unset / ``0`` / ``false`` (default): native NVIDIA path,
              ~30–80 fps at 256×256. Will hang on affected hosts.
            - ``1`` / ``true`` / ``yes``: always engage lavapipe.
            - ``auto``: probe the native path in a child process (with
              a watchdog timeout). If it returns successfully, leave SAPIEN
              alone. If it hangs, engage lavapipe in this process.
              ``auto`` adds ~5–10 s of startup probe time on healthy hosts
              but lets a single launcher work across the whole fleet.

        Decision is cached per-process via ``_rendering_configured``;
        changing ``ROBOMME_USE_LAVAPIPE`` after the first ``reset()`` has no
        effect (Vulkan ICD is loaded at the first ``import sapien.render``
        and cannot be re-bound in-process).

        Lavapipe path (when engaged) is ~5–10× slower than the native path
        (~4–12 fps vs ~33–80 fps). When engaged we also set
        ``LP_NUM_THREADS=4`` and ``OMP_NUM_THREADS=1`` (empirical sweet spot
        for Mesa lavapipe with 256×256 frames; recovers ~30%% over the
        unset default). Three-piece patch:

        1. ``VK_ICD_FILENAMES`` → lavapipe (so Vulkan dispatch goes to
           the Mesa software renderer, bypassing the broken interop path).
        2. Wrap ``sapien.render.RenderSystem`` to drop the ``device``
           positional arg — lavapipe has no CUDA backend, so calling
           ``RenderSystem("cuda:0")`` raises "Failed to find a supported
           physical device".
        3. Patch ``mani_skill ... parse_sim_and_render_backend`` so the
           render backend resolves to ``sapien_cpu``, matching the device
           that lavapipe actually uses.
        """
        if RoboMMEBenchmark._rendering_configured:
            return

        mode = os.environ.get("ROBOMME_USE_LAVAPIPE", "").strip().lower()
        if mode == "auto":
            if native_render_path_works():
                logger.info("SAPIEN auto-detect: native NVIDIA Vulkan path works, skipping lavapipe")
                RoboMMEBenchmark._rendering_configured = True
                return
            logger.warning(
                "SAPIEN auto-detect: native render path hung within watchdog timeout; engaging lavapipe fallback"
            )
        elif mode not in ("1", "true", "yes", "on"):
            RoboMMEBenchmark._rendering_configured = True
            return

        # Engagement requested (mode in {"1","true","yes","on","auto"+hang}).
        # If the patch can't take effect, the next take_picture would silently
        # hang — fail loudly so the launcher knows to either fix the host or
        # set ROBOMME_USE_LAVAPIPE before any sapien import.
        if not RoboMMEBenchmark._engage_lavapipe():
            raise RuntimeError(
                "Lavapipe rendering requested (ROBOMME_USE_LAVAPIPE={!r}) but could not "
                "be engaged. Check earlier log lines for the specific reason "
                "(sapien.render already imported, ICD missing, etc.); continuing on the "
                "native NVIDIA path would hang on affected hosts.".format(mode)
            )
        RoboMMEBenchmark._rendering_configured = True

    @staticmethod
    def _engage_lavapipe() -> bool:
        """Apply the three-piece lavapipe patch + perf-tuning env vars.

        Must be called BEFORE ``import sapien.render`` in this process for
        ``VK_ICD_FILENAMES`` and ``LP_NUM_THREADS`` to take effect (Vulkan
        ICD is loaded at first ``import sapien.render``).

        Returns True on success, False if the patch could not be applied
        (sapien.render already imported, lavapipe ICD missing, etc.). The
        caller is responsible for treating False as fatal — silently
        continuing on the native path would hang on affected hosts.
        """
        if "sapien.render" in sys.modules:
            logger.error(
                "Cannot engage lavapipe: sapien.render is already imported. "
                "VK_ICD_FILENAMES / LP_NUM_THREADS only take effect on first "
                "Vulkan init. Set ROBOMME_USE_LAVAPIPE before any sapien import."
            )
            return False

        lavapipe_icd = _resolve_lavapipe_icd()
        if lavapipe_icd is None:
            logger.error("Lavapipe ICD not found; cannot engage lavapipe rendering")
            return False

        # Mesa lavapipe perf tuning. Empirical sweep on A100 with RoboMME's
        # 256×256 frames: LP=4, OMP=1 is the sweet spot (~4.8 fps vs ~3.3 fps
        # at LP=1 vs ~3.7 fps unset). Don't override if user already set them.
        os.environ.setdefault("LP_NUM_THREADS", "4")
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")

        os.environ["VK_ICD_FILENAMES"] = lavapipe_icd
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

        return True

    def reset(self, task: Task) -> Any:
        self._setup_rendering()
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

        if self._recorder is not None:
            self._recorder.start({"env_id": task["env_id"], "episode_idx": episode_idx})
            front_list = obs_batch.get("front_rgb_list", [])
            if front_list:
                self._recorder.record_frame(front_list[-1])
        self._step_counter = 0

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

        if self._recorder is not None and obs:
            front_list = obs.get("front_rgb_list", [])
            if front_list:
                self._recorder.record_frame(front_list[-1])

        terminated = bool(terminated)
        truncated = bool(truncated)
        reward = float(reward)
        done = terminated or truncated or info.get("status") == "error"

        if self._recorder is not None and self._recorder.active:
            fields = self._record_fields
            row: dict[str, Any] = {"step": self._step_counter}
            if "gt_subgoal" in fields:
                row["gt_subgoal"] = info.get("simple_subgoal_online", "")
            if "grounded_subgoal" in fields:
                row["grounded_subgoal"] = info.get("grounded_subgoal_online", "")
            if "reward" in fields:
                row["reward"] = reward
            if "robot_state" in fields and obs:
                state = obs.get("state_fq")
                if state is not None:
                    row["robot_state"] = state.tolist() if hasattr(state, "tolist") else list(state)
            if "terminated" in fields:
                row["terminated"] = terminated
            self._recorder.record_step(row)
        self._step_counter += 1

        return StepResult(obs=obs, reward=reward, done=done, info=info)

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
        status = "success" if success else "fail"
        if self._recorder is not None:
            self._recorder.save(status=status)
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
        if self._recorder is not None:
            self._recorder.discard()
        self._video_frames = []
        self._wrist_video_frames = []
        self._task = None
