"""VSArena remote stacking benchmark for vla-eval.

Unlike Docker-local sim benchmarks, VSArena runs physics on a hosted WebSocket
harness (Rapier 60 Hz). This adapter implements :class:`StepBenchmark` by
bridging vla-eval's model-server loop to the VSArena ``state → action → result``
protocol.

Live matches require ``VSARENA_API_KEY`` (from https://vsarena.vercel.app/account).
Smoke tests use ``dry_run=True`` (no network, no API key).
"""

from __future__ import annotations

import base64
import json
import logging
import os
import uuid
from typing import Any

import numpy as np

from vla_eval.benchmarks.base import StepBenchmark, StepResult, repeat_last_hold
from vla_eval.specs import GRIPPER_CLOSE_POS, IMAGE_RGB, LANGUAGE, POSITION_DELTA, RAW, DimSpec
from vla_eval.types import Action, EpisodeResult, Observation, Task

logger = logging.getLogger(__name__)

DEFAULT_WS_URL = "wss://vsarena-harness.onrender.com"
STACK_INSTRUCTION = (
    "Stack the three cubes into a tower on the pad: cyan base, orange middle, magenta on top."
)
JOINT_KEYS = ("joint_1", "joint_2", "joint_3", "joint_4")
ACTION_DIM = 7  # LIBERO-style: dx dy dz dax day daz grip
_MOCK_RGB_B64 = base64.b64encode(bytes([28, 32, 40] * (8 * 8))).decode("ascii")


def _mock_state(tick: int, match_id: str, *, mode: str = "vla") -> dict[str, Any]:
    state: dict[str, Any] = {
        "type": "state",
        "match_id": match_id,
        "tick": tick,
        "timestamp_ms": tick * 100,
        "observation_mode": mode,
        "instruction": STACK_INSTRUCTION,
        "scene": {
            "gripper_pose": [0.2, 1.1, 0.0, 0.0, 0.0, 0.0, 1.0],
            "blocks": [],
            "joint_states": {
                "joint_1": 0.0,
                "joint_2": 0.55,
                "joint_3": -1.15,
                "joint_4": -0.35,
            },
            "grasped_block_id": None,
        },
    }
    if mode == "vla":
        state["images"] = {
            "scene": {"mime": "image/rgb8", "width": 8, "height": 8, "b64": _MOCK_RGB_B64},
        }
    return state


def _decode_scene_image(state: dict[str, Any]) -> np.ndarray | None:
    images = state.get("images") or {}
    scene = images.get("scene")
    if not isinstance(scene, dict):
        return None
    raw = base64.b64decode(scene["b64"])
    width = int(scene["width"])
    height = int(scene["height"])
    arr = np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3)
    return arr.copy()


def _action_to_vsarena(action: Action) -> dict[str, Any]:
    """Map vla-eval model output to VSArena harness action payload."""
    if isinstance(action.get("vsarena_action"), dict):
        payload = dict(action["vsarena_action"])
        gripper = payload.get("gripper_state", "open")
        if gripper not in ("open", "closed"):
            raise ValueError("gripper_state must be 'open' or 'closed'")
        return payload

    raw = action.get("actions", action.get("action"))
    if raw is None:
        raise ValueError("action must include 'actions' or 'vsarena_action'")

    vec = np.atleast_1d(np.asarray(raw, dtype=np.float32))
    if vec.shape[-1] >= 7:
        gripper = "closed" if float(vec[6]) > 0 else "open"
        return {
            "ee_delta": {"dx": float(vec[0]), "dy": float(vec[1]), "dz": float(vec[2])},
            "gripper_state": gripper,
        }
    if vec.shape[-1] >= 4:
        return {
            "joint_targets": {key: float(vec[i]) for i, key in enumerate(JOINT_KEYS)},
            "gripper_state": "closed" if vec.shape[-1] >= 5 and float(vec[4]) > 0 else "open",
        }
    raise ValueError(f"unsupported action dimension: {vec.shape[-1]} (expected 4, 5, or 7)")


class VSArenaBenchmark(StepBenchmark):
    """Remote VSArena block-stacking benchmark (hosted harness).

    Non-obvious behaviors:
        - **Remote env**: Physics runs on the VSArena harness server, not in this
          container. The Docker image only needs network egress + ``websockets``.
        - **dry_run**: Offline mock loop for ``vla-eval test`` smoke runs (no API key).
        - **Action mapping**: Default 7-D LIBERO-style deltas map to ``ee_delta`` +
          gripper open/closed. Pass through ``vsarena_action`` for full control.
        - **Public ELO**: Live harness may write ELO on ingest; vla-eval still records
          ``success`` and VSArena scores in episode metrics.

    Args:
        ws_url: Harness WebSocket URL.
        api_key: API key; defaults to ``VSARENA_API_KEY`` env var.
        agent_name: Leaderboard label for live matches.
        task: Harness task id (MVP: ``block_stacking`` only).
        mode: ``vla`` (RGB + instruction) or ``state`` (privileged debug track).
        dry_run: Offline mock — no socket, no API key.
        max_episode_steps: Safety cap when harness does not terminate.
        tasks: Optional subset of task names to run.
    """

    _ALL_RECORD_FIELDS = frozenset(
        {"reward", "done", "success", "spatial_accuracy", "task_completion_score", "elo_delta"}
    )

    render_backends = frozenset({"cpu"})

    @classmethod
    def configure_render(cls, mode: str) -> dict[str, str]:
        return {}

    def __init__(
        self,
        ws_url: str = DEFAULT_WS_URL,
        api_key: str | None = None,
        agent_name: str | None = None,
        task: str = "block_stacking",
        mode: str = "vla",
        dry_run: bool = False,
        max_episode_steps: int = 500,
        tasks: list[str] | None = None,
    ) -> None:
        super().__init__()
        if task != "block_stacking":
            raise ValueError("VSArena MVP only supports task='block_stacking'")
        if mode not in ("vla", "state"):
            raise ValueError("mode must be 'vla' or 'state'")

        self._ws_url = ws_url
        self._api_key = api_key or os.environ.get("VSARENA_API_KEY")
        self._agent_name = agent_name or os.environ.get("VSARENA_AGENT_NAME")
        self._task = task
        self._mode = mode
        self._dry_run = dry_run
        self._max_episode_steps = max_episode_steps
        self._task_filter = set(tasks) if tasks is not None else None

        self._socket: Any = None
        self._match_id: str | None = None
        self._tick = 0
        self._step_count = 0
        self._last_state: dict[str, Any] | None = None
        self._result: dict[str, Any] | None = None

    def cleanup(self) -> None:
        if self._socket is not None:
            try:
                self._socket.close()
            except Exception:
                pass
            self._socket = None

    def get_tasks(self) -> list[Task]:
        tasks = [{"name": "VSArena block stacking", "task": self._task, "mode": self._mode}]
        if self._task_filter is not None:
            return [t for t in tasks if t["name"] in self._task_filter]
        return tasks

    def reset(self, task: Task) -> Any:
        self._step_count = 0
        self._tick = 0
        self._result = None
        self._match_id = f"vla-eval-{uuid.uuid4()}"

        if self._dry_run:
            self._last_state = _mock_state(0, self._match_id, mode=self._mode)
            self._recorder.record_video(_decode_scene_image(self._last_state))
            return self._last_state

        if not self._api_key:
            raise ValueError("live VSArena eval requires api_key= or VSARENA_API_KEY")

        from websockets.sync.client import connect

        self.cleanup()
        self._socket = connect(self._ws_url, open_timeout=30, close_timeout=5)
        hello: dict[str, Any] = {
            "type": "hello",
            "api_key": self._api_key,
            "task": task.get("task", self._task),
            "mode": task.get("mode", self._mode),
        }
        if self._agent_name:
            hello["agent"] = self._agent_name
        self._socket.send(json.dumps(hello))
        self._last_state = self._recv_until_state()
        self._recorder.record_video(_decode_scene_image(self._last_state))
        return self._last_state

    def _recv_until_state(self) -> dict[str, Any]:
        assert self._socket is not None
        while True:
            raw = self._socket.recv(timeout=120)
            message = json.loads(raw)
            kind = message.get("type")
            if kind == "error":
                raise RuntimeError(message.get("message", message))
            if kind == "result":
                self._result = message
                return self._last_state or _mock_state(self._tick, self._match_id or "done", mode=self._mode)
            if kind == "state":
                self._tick = int(message.get("tick", 0))
                return message
            logger.debug("ignoring harness message type=%s", kind)

    def _send_action_and_advance(self, vsarena_action: dict[str, Any]) -> StepResult:
        if self._dry_run:
            self._step_count += 1
            self._tick += 1
            done = self._step_count >= min(12, self._max_episode_steps)
            if done:
                self._result = {
                    "status": "completed",
                    "scores": {
                        "spatial_accuracy": 0.0,
                        "task_completion_score": 0.0,
                    },
                    "elo_delta": 0,
                    "dry_run": True,
                }
            else:
                self._last_state = _mock_state(self._tick, self._match_id or "dry", mode=self._mode)
            frame = _decode_scene_image(self._last_state) if self._last_state else None
            self._recorder.record_video(frame)
            success = bool(self._result and self._result.get("status") == "completed")
            self._recorder.record_step(reward=0.0, done=done, success=success)
            return StepResult(obs=self._last_state, reward=0.0, done=done, info={"result": self._result})

        assert self._socket is not None and self._last_state is not None
        payload = {
            "type": "action",
            "match_id": self._last_state["match_id"],
            "tick": self._last_state["tick"],
            "action": vsarena_action,
        }
        self._socket.send(json.dumps(payload))

        self._step_count += 1
        message = self._recv_until_state()
        done = self._result is not None or self._step_count >= self._max_episode_steps

        if self._result is None and isinstance(message, dict) and message.get("type") == "state":
            self._last_state = message
        elif self._result is not None:
            done = True

        frame = _decode_scene_image(self._last_state) if self._last_state else None
        self._recorder.record_video(frame)

        scores = (self._result or {}).get("scores") or {}
        completion = float(scores.get("task_completion_score", 0.0))
        success = bool(self._result and self._result.get("status") == "completed" and completion >= 0.5)
        self._recorder.record_step(reward=completion, done=done, success=success)

        return StepResult(
            obs=self._last_state,
            reward=completion,
            done=done,
            info={"result": self._result, "harness_status": (self._result or {}).get("status")},
        )

    def step(self, action: Action) -> StepResult:
        vsarena_action = _action_to_vsarena(action)
        return self._send_action_and_advance(vsarena_action)

    def make_obs(self, raw_obs: Any, task: Task) -> Observation:
        state = raw_obs if isinstance(raw_obs, dict) else {}
        instruction = state.get("instruction", STACK_INSTRUCTION)
        obs: dict[str, Any] = {"task_description": instruction}

        img = _decode_scene_image(state)
        if img is not None:
            obs["images"] = {"scene": img}

        scene = state.get("scene") or {}
        joints = scene.get("joint_states") or {}
        if joints:
            obs["proprio"] = np.array([float(joints.get(k, 0.0)) for k in JOINT_KEYS], dtype=np.float32)

        return obs

    def check_done(self, step_result: StepResult) -> bool:
        return step_result.done

    def get_step_result(self, step_result: StepResult) -> EpisodeResult:
        result = step_result.info.get("result") or {}
        scores = result.get("scores") or {}
        status = result.get("status")
        completion = float(scores.get("task_completion_score", step_result.reward))
        success = status == "completed" and completion >= 0.5
        if self._dry_run and result.get("dry_run"):
            success = False

        episode: EpisodeResult = {"success": success}
        if "spatial_accuracy" in scores:
            episode["spatial_accuracy"] = float(scores["spatial_accuracy"])
        if "task_completion_score" in scores:
            episode["task_completion_score"] = float(scores["task_completion_score"])
        if "elo_delta" in result:
            episode["elo_delta"] = float(result["elo_delta"])
        return episode

    def get_metadata(self) -> dict[str, Any]:
        return {
            "max_steps": self._max_episode_steps,
            "action_dim": ACTION_DIM,
            "ws_url": self._ws_url,
            "dry_run": self._dry_run,
            "observation_mode": self._mode,
        }

    def get_metric_keys(self) -> dict[str, str]:
        return {
            "success": "mean",
            "spatial_accuracy": "mean",
            "task_completion_score": "mean",
            "elo_delta": "mean",
        }

    def get_hold_action(self, last_action: Action | None) -> Action:
        return repeat_last_hold(last_action, ACTION_DIM)

    def get_action_spec(self) -> dict[str, DimSpec]:
        return {
            "action": POSITION_DELTA,
            "gripper": GRIPPER_CLOSE_POS,
            "vsarena_action": RAW,
        }

    def get_observation_spec(self) -> dict[str, DimSpec]:
        return {
            "scene": IMAGE_RGB,
            "language": LANGUAGE,
        }
