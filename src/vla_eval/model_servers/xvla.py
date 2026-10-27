# /// script
# requires-python = "~=3.11"
# dependencies = [
#     "vla-eval",
#     "torch>=2.2",
#     "transformers>=4.44,<=4.51.3",
#     "numpy>=1.24",
#     "pillow>=9.0",
#     "opencv-python-headless",
#     "fastapi",
#     "json-numpy",
#     "uvicorn",
#     "einops",
#     "timm",
# ]
#
# [tool.uv.sources]
# vla-eval = { path = "../../..", editable = true }
#
# [tool.uv]
# exclude-newer = "2026-02-24T00:00:00Z"
# ///
"""X-VLA model server.

Loads an X-VLA checkpoint from HuggingFace and runs flow-matching
inference directly via ``model.generate_actions()``.  No external
server required.

Action conversion (``output_action_dim=7``):
    X-VLA uses a unified 20-D dual-arm ``EE6DActionSpace``.  For
    single-arm benchmarks (LIBERO, SimplerEnv, CALVIN) the model server
    extracts the first arm (10-D), converts the 6-D rotation to
    axis-angle via Gram-Schmidt orthogonalisation, and applies sigmoid +
    threshold to the gripper, yielding the standard 7-D format::

        [pos_x, pos_y, pos_z, aa_x, aa_y, aa_z, gripper]

    RoboTwin (``output_action_dim=16``) decodes both arms as the official
    ``evaluation/robotwin-2.0/client.py`` does, into RoboTwin's ``take_action(..., "ee")`` layout.

Proprioceptive state (closed-loop feedback):
    On the **first** inference of an episode, accepts ``obs["state"]``
    or ``obs["states"]`` as a flat array (e.g. ``[pos3, axisangle3,
    gripper*]``) and converts it to the 20-D format expected by X-VLA
    (``[pos3, rot6d6, 0.0, zeros10]``).  RoboTwin sends its 16-D end pose instead.

    On **subsequent** inferences (when the action chunk buffer is
    drained), the model server feeds the **last predicted action's**
    ``[pos3, rot6d6, gripper]`` as proprioception instead of the
    environment state.  This matches the official X-VLA evaluation loop,
    which updates ``proprio[:10] = action[-1, :10]`` after each call.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from vla_eval.specs import (
    GRIPPER_CLOSE_POS,
    IMAGE_RGB,
    LANGUAGE,
    POSITION_DELTA,
    RAW,
    ROTATION_AA,
    ROTATION_EULER,
    STATE_ROT6D_PROPRIO_20D,
    EE_POSE_DUAL_WXYZ,
    STATE_EE_POSE_DUAL_WXYZ,
    DimSpec,
)
from vla_eval.types import Action, Observation

from vla_eval.model_servers.base import SessionContext
from vla_eval.model_servers.predict import PredictModelServer

from vla_eval.rotation import (
    axisangle_to_matrix,
    axisangle_to_rot6d_contiguous as _axisangle_to_rot6d_contig,
    axisangle_to_rot6d_interleaved as _axisangle_to_rot6d_inter,
    euler_xyz_to_rot6d_contiguous as _euler_to_rot6d_contig,
    euler_xyz_to_rot6d_interleaved as _euler_to_rot6d_inter,
    matrix_to_euler_xyz,
    matrix_to_quat as _mat_to_quat,
    quat_to_axisangle as _quat_to_axisangle,
    quat_to_matrix as _quat_to_matrix,
    quat_to_rot6d_interleaved,
    rot6d_contiguous_to_matrix as _rot6d_to_matrix_contig,
    rot6d_interleaved_to_matrix as _rot6d_to_matrix_inter,
    rot6d_interleaved_to_quat,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _XVLABenchmarkProfile:
    image_keys: tuple[str, ...]
    predicted_proprio_dims: int | None
    use_predicted_proprio: bool
    gripper_threshold: float  # sigmoid threshold for gripper binarization
    gripper_close_above: bool  # True: >threshold=close(+1), False: <threshold=close(+1)
    output_action_dim: int | None = None
    instruction_from_task_name: bool = False  # official RoboTwin client prompts with the task name
    unflip_wrist: bool = False  # un-flip wrist image (benchmark sends it flipped)
    euler_state: bool = False  # True: state[3:6] is euler XYZ, not axis-angle
    rot6d_convention: str = "interleaved"  # "interleaved" or "contiguous"
    gripper_threshold_map: dict[str, float] | None = None  # per-task gripper thresholds


_BENCHMARK_PROFILES: dict[str, _XVLABenchmarkProfile] = {
    "libero": _XVLABenchmarkProfile(
        image_keys=("agentview", "wrist"),
        predicted_proprio_dims=10,
        use_predicted_proprio=True,
        gripper_threshold=0.5,
        gripper_close_above=True,
        output_action_dim=7,
        unflip_wrist=True,
        rot6d_convention="contiguous",
    ),
    "calvin": _XVLABenchmarkProfile(
        image_keys=("rgb_static", "rgb_gripper"),
        predicted_proprio_dims=10,
        use_predicted_proprio=True,
        gripper_threshold=0.8,
        gripper_close_above=True,
        output_action_dim=7,
        euler_state=True,
    ),
    "simpler": _XVLABenchmarkProfile(
        image_keys=("primary",),
        predicted_proprio_dims=10,
        use_predicted_proprio=True,
        gripper_threshold=0.25,
        gripper_close_above=True,
        output_action_dim=7,
    ),
    "simpler_widowx": _XVLABenchmarkProfile(
        image_keys=("primary",),
        predicted_proprio_dims=10,
        use_predicted_proprio=True,
        gripper_threshold=0.7,
        gripper_close_above=False,
        output_action_dim=7,
        gripper_threshold_map={
            "widowx_stack_cube": 0.91,
            "widowx_spoon_on_towel": 0.7,
            "widowx_carrot_on_plate": 0.95,
            "widowx_put_eggplant_in_basket": 0.8,
        },
    ),
    "vlabench": _XVLABenchmarkProfile(
        image_keys=("primary", "front", "wrist"),
        predicted_proprio_dims=10,
        use_predicted_proprio=False,
        gripper_threshold=0.5,
        gripper_close_above=True,
    ),
    "robotwin": _XVLABenchmarkProfile(
        image_keys=("head_camera", "left_camera", "right_camera"),
        predicted_proprio_dims=None,  # proprio is the commanded end pose (_robotwin_proprio)
        use_predicted_proprio=True,
        gripper_threshold=0.7,
        gripper_close_above=True,
        output_action_dim=16,
        instruction_from_task_name=True,
    ),
}


def _get_profile(name: str) -> _XVLABenchmarkProfile:
    try:
        return _BENCHMARK_PROFILES[name]
    except KeyError as exc:
        choices = ", ".join(sorted(_BENCHMARK_PROFILES))
        raise ValueError(f"Unsupported X-VLA benchmark_profile {name!r}. Expected one of: {choices}") from exc


_PROFILE_OBS_PARAMS: dict[str, dict[str, Any]] = {
    "libero": {"send_wrist_image": True, "send_state": True, "absolute_action": True},
    "calvin": {"send_wrist_image": True, "send_state": True, "absolute_action": True, "ep_len": 720},
    "simpler": {
        "send_state": True,
        "success_mode": "early_stop",
        "max_episode_steps": 160,
        "control_mode": "arm_pd_ee_base_pose_align_interpolate_by_planner_gripper_pd_joint_target_delta_pos_interpolate_by_planner",
    },
    "simpler_widowx": {
        "send_state": True,
        "max_episode_steps": 1200,
        "success_mode": "early_stop",
        "control_mode": "arm_pd_ee_target_base_pose_gripper_pd_joint_pos",
    },
    "vlabench": {"send_wrist_image": True, "send_state": True},
    "robotwin": {"send_state": True, "action_type": "ee"},
}


def _compute_ee_pos_wrt_base(base_pose: np.ndarray, tcp_pose: np.ndarray) -> np.ndarray:
    """Compute EE position relative to robot base (matches official X-VLA SimplerEnv eval).

    ``base_pose`` and ``tcp_pose`` are 7-D ``[pos3, quat4]`` from ManiSkill2
    (quaternion in ``[w, x, y, z]`` order).  Returns 3-D position only.
    """
    # Quaternion inverse: q_inv = conjugate / norm^2 (unit quat → just conjugate)
    bq = base_pose[3:7]  # [w, x, y, z]
    bq_inv = np.array([bq[0], -bq[1], -bq[2], -bq[3]])  # conjugate
    # Rotate (tcp_pos - base_pos) by base_quat_inv
    dp = tcp_pose[:3] - base_pose[:3]
    # Quaternion rotation: q * v * q_inv (using matrix form for simplicity)
    bmat = _quat_to_matrix(np.array([bq_inv[1], bq_inv[2], bq_inv[3], bq_inv[0]]))  # [x,y,z,w] for our func
    return (bmat @ dp).astype(np.float32)


def _obs_state_array(obs: dict[str, Any]) -> np.ndarray | None:
    """Read proprioceptive state from observation.

    Prefers ``controller_states`` (LIBERO: from robot.controller, matches
    X-VLA training data) over ``states`` (from raw_obs quaternion, different
    coordinate frame). Falls back to ``states``/``state`` for benchmarks
    that don't provide controller state (CALVIN, SimplerEnv, etc.).
    """
    raw_state = obs.get("controller_states")
    if raw_state is None:
        raw_state = obs.get("states")
    if raw_state is None:
        raw_state = obs.get("state")
    if raw_state is None:
        return None
    return np.asarray(raw_state, dtype=np.float32).flatten()


def _ordered_images(obs: dict[str, Any], image_keys: tuple[str, ...]) -> list[np.ndarray]:
    images_dict = obs.get("images", {})
    if not isinstance(images_dict, dict):
        return []

    if image_keys:
        ordered = [np.asarray(images_dict[key], dtype=np.uint8) for key in image_keys if key in images_dict]
        if ordered:
            return ordered

    return [np.asarray(img, dtype=np.uint8) for img in images_dict.values()]


def _default_predicted_proprio_dims(output_action_dim: int | None) -> int | None:
    return 10 if output_action_dim is not None else None


def _rot6d_to_axisangle(rot6d: np.ndarray, rot6d_to_matrix=_rot6d_to_matrix_inter) -> np.ndarray:
    """6-D rotation → axis-angle (3-D)."""
    return _quat_to_axisangle(_mat_to_quat(rot6d_to_matrix(rot6d)))


def _convert_ee6d_to_7d(
    actions: np.ndarray,
    gripper_threshold: float = 0.5,
    gripper_close_above: bool = True,
    rot6d_to_matrix=_rot6d_to_matrix_inter,
) -> np.ndarray:
    """Convert X-VLA EE6D 20-D actions → 7-D ``[pos3, axisangle3, gripper]``.

    Extracts arm-1, converts rot6d → axis-angle, and binarizes the gripper
    using the configured threshold and direction.

    Note: ``generate_actions()`` already applies sigmoid to the gripper
    via ``postprocess()``, so we threshold directly without re-applying
    sigmoid.
    """
    single = actions.ndim == 1
    if single:
        actions = actions[np.newaxis]
    out = np.zeros((len(actions), 7), dtype=np.float32)
    for i in range(len(actions)):
        out[i, :3] = actions[i, :3]
        out[i, 3:6] = _rot6d_to_axisangle(actions[i, 3:9], rot6d_to_matrix)
        # Gripper binarization using profile-configured threshold and direction
        sig = float(actions[i, 9])
        if gripper_close_above:
            out[i, 6] = 1.0 if sig > gripper_threshold else -1.0
        else:
            out[i, 6] = 1.0 if sig < gripper_threshold else -1.0
    return out[0] if single else out


def _ee6d_to_robotwin_ee16(actions: np.ndarray, gripper_threshold: float) -> np.ndarray:
    """``(T, 20)`` EE6D → ``(T, 16)`` RoboTwin ee actions, as the official client: interleaved rot6d, the quaternion
    left in scipy's xyzw order, gripper ``1 - 2 * (sigmoid > threshold)`` (RoboTwin clips it after interpolating)."""
    out = np.zeros((len(actions), 16), dtype=np.float64)
    for i, a in enumerate(np.asarray(actions, dtype=np.float64)):
        for arm in range(2):
            src, dst = a[10 * arm : 10 * arm + 10], out[i, 8 * arm : 8 * arm + 8]
            dst[:3] = src[:3]
            dst[3:7] = rot6d_interleaved_to_quat(src[3:9])
            dst[7] = 1.0 - 2.0 * float(src[9] > gripper_threshold)
    return out


def _robotwin_proprio(state: np.ndarray, commanded: np.ndarray | None) -> np.ndarray:
    """16-D RoboTwin end pose → 20-D proprio, as the official client: each arm ``[pos3,
    rot6d(quat read as xyzw), 1 - 2 * gripper]``; after the first chunk the poses are the last commanded ones."""
    ee = np.asarray(state, dtype=np.float64).copy()
    if commanded is not None:
        ee[0:7], ee[8:15] = commanded[0:7], commanded[8:15]
    proprio = np.zeros(20, dtype=np.float32)
    for arm in range(2):
        src = ee[8 * arm : 8 * arm + 8]
        proprio[10 * arm : 10 * arm + 3] = src[:3]
        proprio[10 * arm + 3 : 10 * arm + 9] = quat_to_rot6d_interleaved(src[3:7] / np.linalg.norm(src[3:7]))
        proprio[10 * arm + 9] = 1.0 - 2.0 * src[7]
    return proprio


def _state_to_xvla_proprio(
    state: np.ndarray,
    dim: int = 20,
    euler_state: bool = False,
    axisangle_to_rot6d=_axisangle_to_rot6d_inter,
    euler_to_rot6d=_euler_to_rot6d_inter,
) -> np.ndarray:
    """Convert ``[pos3, axisangle3, gripper*]`` → X-VLA proprio (20-D).

    Matches the official eval format: ``[pos3, rot6d6, 0.0, zeros10]``.
    When ``euler_state=True``, state[3:6] is interpreted as XYZ Euler angles.
    """
    proprio = np.zeros(dim, dtype=np.float32)
    if len(state) >= 6:
        proprio[:3] = state[:3]
        if euler_state:
            proprio[3:9] = euler_to_rot6d(state[3:6])
        else:
            proprio[3:9] = axisangle_to_rot6d(state[3:6])
    return proprio


class XVLAModelServer(PredictModelServer):
    """X-VLA model server using HuggingFace AutoModel."""

    def __init__(
        self,
        model_path: str = "2toINF/X-VLA-Libero",
        domain_id: int = 0,
        denoising_steps: int = 10,
        *,
        benchmark_profile: str | None = None,
        chunk_size: int = 30,
        action_ensemble: str = "newest",
        output_action_dim: int | None = None,
        use_predicted_proprio: bool | None = None,
        euler_offset: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(chunk_size=chunk_size, action_ensemble=action_ensemble, **kwargs)
        profile = _get_profile(benchmark_profile) if benchmark_profile is not None else None
        if output_action_dim is None and profile is not None:
            output_action_dim = profile.output_action_dim
        if use_predicted_proprio is None and profile is not None:
            use_predicted_proprio = profile.use_predicted_proprio
        if use_predicted_proprio is None:
            use_predicted_proprio = True

        self.model_path = model_path
        self.domain_id = domain_id
        self.denoising_steps = denoising_steps
        self.benchmark_profile = benchmark_profile
        self.output_action_dim = output_action_dim
        self.use_predicted_proprio = use_predicted_proprio
        self._image_keys = profile.image_keys if profile is not None else ()
        self._predicted_proprio_dims = (
            profile.predicted_proprio_dims
            if profile is not None
            else _default_predicted_proprio_dims(output_action_dim)
        )
        self._instruction_from_task_name = profile.instruction_from_task_name if profile is not None else False
        self._unflip_wrist = profile.unflip_wrist if profile is not None else False
        self._euler_state = profile.euler_state if profile is not None else False
        self._gripper_threshold = profile.gripper_threshold if profile is not None else 0.5
        self._gripper_close_above = profile.gripper_close_above if profile is not None else True
        self._gripper_threshold_map = profile.gripper_threshold_map if profile is not None else None
        rot6d_conv = profile.rot6d_convention if profile is not None else "interleaved"
        if rot6d_conv == "contiguous":
            self._rot6d_to_matrix = _rot6d_to_matrix_contig
            self._axisangle_to_rot6d = _axisangle_to_rot6d_contig
            self._euler_to_rot6d = _euler_to_rot6d_contig
        else:
            self._rot6d_to_matrix = _rot6d_to_matrix_inter
            self._axisangle_to_rot6d = _axisangle_to_rot6d_inter
            self._euler_to_rot6d = _euler_to_rot6d_inter
        self._euler_offset: np.ndarray | None = None
        if euler_offset is not None:
            self._euler_offset = np.array([float(x) for x in euler_offset.split(",")], dtype=np.float32)
        # Closed-loop proprio: store raw 20-D actions per session so the
        # next predict() call can feed the model its own last prediction.
        # Disabled when use_predicted_proprio=False (e.g. VLABench, which
        # always uses fresh env state in the official eval).
        self._last_raw_actions: dict[str, np.ndarray] = {}
        self._current_xyz: dict[str, np.ndarray] = {}
        self._last_commanded: dict[str, np.ndarray] = {}  # RoboTwin: last executed 16-D ee action
        self._task_prompts: dict[str, str] = {}

        import torch
        from transformers import AutoConfig, AutoModel, AutoProcessor

        logger.info("Loading X-VLA from %s", self.model_path)
        self._processor = AutoProcessor.from_pretrained(
            self.model_path,
            trust_remote_code=True,
        )

        # Load config and force eager attention on the Florence2 sub-config
        # to work around a @property/_supports_sdpa incompatibility with
        # transformers >= 4.46 (the property accesses self.language_model
        # which doesn't exist yet during __init__).
        config = AutoConfig.from_pretrained(
            self.model_path,
            trust_remote_code=True,
        )
        if hasattr(config, "florence_config"):
            config.florence_config._attn_implementation_internal = "eager"

        # Force float32 — the official X-VLA deploy.py explicitly casts to
        # float32.  The 10-step denoising process is sensitive to precision;
        # float16/bfloat16 can cause numerical drift that degrades actions.
        self._model = AutoModel.from_pretrained(
            self.model_path,
            config=config,
            trust_remote_code=True,
            attn_implementation="eager",
            torch_dtype=torch.float32,
        )
        self._model.to(device="cuda:0", dtype=torch.float32).eval()
        logger.info(
            "X-VLA model loaded on cuda:0 (float32, profile=%s)",
            self.benchmark_profile or "custom",
        )

    async def on_episode_start(self, config: dict[str, Any], ctx: SessionContext) -> None:
        self._last_raw_actions.pop(ctx.session_id, None)
        self._current_xyz.pop(ctx.session_id, None)
        self._last_commanded.pop(ctx.session_id, None)
        if self._instruction_from_task_name:
            self._task_prompts[ctx.session_id] = config["task"]["name"].replace("_", " ")
        await super().on_episode_start(config, ctx)

    async def on_episode_end(self, result: dict[str, Any], ctx: SessionContext) -> None:
        self._last_raw_actions.pop(ctx.session_id, None)
        self._current_xyz.pop(ctx.session_id, None)
        self._last_commanded.pop(ctx.session_id, None)
        self._task_prompts.pop(ctx.session_id, None)
        await super().on_episode_end(result, ctx)

    def get_observation_params(self) -> dict[str, Any]:
        if self.benchmark_profile and self.benchmark_profile in _PROFILE_OBS_PARAMS:
            return dict(_PROFILE_OBS_PARAMS[self.benchmark_profile])
        return {}

    def get_action_spec(self) -> dict[str, DimSpec]:
        if self.output_action_dim == 7:
            rotation = ROTATION_EULER if self._euler_offset is not None else ROTATION_AA
            return {"position": POSITION_DELTA, "rotation": rotation, "gripper": GRIPPER_CLOSE_POS}
        if self.output_action_dim == 16:
            return {"ee_pose": EE_POSE_DUAL_WXYZ}
        return {"actions": RAW}

    def get_observation_spec(self) -> dict[str, DimSpec]:
        spec: dict[str, DimSpec] = {}
        for key in self._image_keys:
            spec[key] = IMAGE_RGB
        if not self._image_keys:
            spec["image"] = IMAGE_RGB
        if self.output_action_dim == 16:
            spec["ee_state"] = STATE_EE_POSE_DUAL_WXYZ
        elif self.use_predicted_proprio:
            spec["state"] = STATE_ROT6D_PROPRIO_20D
        spec["language"] = LANGUAGE
        return spec

    def predict(self, obs: Observation, ctx: SessionContext) -> Action:
        import torch
        from PIL import Image

        if obs.get("episode_restart"):
            self._last_raw_actions.pop(ctx.session_id, None)
            self._last_commanded.pop(ctx.session_id, None)

        raw_images = _ordered_images(obs, self._image_keys)
        # Un-flip wrist image: benchmark sends all images flipped [::-1,::-1],
        # but X-VLA was trained with unflipped wrist images.
        if self._unflip_wrist and len(raw_images) >= 2:
            raw_images[1] = raw_images[1][::-1, ::-1].copy()
        pil_images = [Image.fromarray(img) for img in raw_images]

        task_desc = obs.get("task_description", "")
        if self._instruction_from_task_name:
            task_desc = self._task_prompts[ctx.session_id]

        # Process with XVLAProcessor
        inputs = self._processor(
            images=pil_images if pil_images else None,
            language_instruction=task_desc,
        )
        device = next(self._model.parameters()).device
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

        # Proprioceptive state
        # - LIBERO/CALVIN: feed model its own last prediction (closed-loop).
        # - VLABench: always use fresh env state (matches official eval).
        dim_proprio = self._model.action_space.dim_action
        last_actions = self._last_raw_actions.get(ctx.session_id) if self.use_predicted_proprio else None
        if self.output_action_dim == 16:
            state = _obs_state_array(obs)
            if state is None or len(state) != 16:
                raise ValueError(
                    "RoboTwin profile needs the 16-D end pose in obs['states']: send_state, and render_every == chunk_size"
                )
            commanded = self._last_commanded.get(ctx.session_id)
            proprio = torch.tensor(_robotwin_proprio(state, commanded), device=device).unsqueeze(0)
        elif last_actions is not None:
            proprio_np = np.zeros(dim_proprio, dtype=np.float32)
            n = self._predicted_proprio_dims or dim_proprio
            n = min(n, dim_proprio, last_actions.shape[-1])
            proprio_np[:n] = last_actions[-1, :n]
            proprio = torch.tensor(proprio_np, device=device).unsqueeze(0)
        else:
            # Try base-relative EE pose (SimplerEnv: base_pose + tcp_pose)
            base_pose = obs.get("base_pose")
            tcp_pose = obs.get("tcp_pose")
            if base_pose is not None and tcp_pose is not None:
                bp = np.asarray(base_pose, dtype=np.float64)
                tp = np.asarray(tcp_pose, dtype=np.float64)
                ee_pos = _compute_ee_pos_wrt_base(bp, tp)
                if self.benchmark_profile == "simpler":
                    # Google Robot: zero proprio on first step (matches official eval)
                    proprio_np = np.zeros(dim_proprio, dtype=np.float32)
                else:
                    proprio_np = np.zeros(dim_proprio, dtype=np.float32)
                    proprio_np[:3] = ee_pos
                    proprio_np[3:10] = [1, 0, 0, 1, 0, 0, 0]  # identity rot6d + gripper
                proprio = torch.tensor(proprio_np, device=device).unsqueeze(0)
            else:
                raw = _obs_state_array(obs)
                if raw is not None:
                    if len(raw) == dim_proprio:
                        # 20D state already in X-VLA format [pos3, rot6d6, 0, zeros10]
                        proprio = torch.tensor(raw, device=device).unsqueeze(0)
                    else:
                        # Legacy 8D state [pos3, axisangle3, gripper2] — convert
                        proprio_np = _state_to_xvla_proprio(
                            raw,
                            dim_proprio,
                            euler_state=self._euler_state,
                            axisangle_to_rot6d=self._axisangle_to_rot6d,
                            euler_to_rot6d=self._euler_to_rot6d,
                        )
                        proprio = torch.tensor(proprio_np, device=device).unsqueeze(0)
                else:
                    proprio = torch.zeros(1, dim_proprio, dtype=torch.float32, device=device)

        domain_id = torch.tensor([self.domain_id], dtype=torch.long, device=device)

        with torch.no_grad():
            actions = self._model.generate_actions(
                **inputs,
                domain_id=domain_id,
                proprio=proprio,
                steps=self.denoising_steps,
            )

        # [B, num_actions, action_dim] → [num_actions, action_dim]
        raw_actions = actions[0].cpu().numpy()

        # Store raw 20-D actions for closed-loop proprio on next call
        self._last_raw_actions[ctx.session_id] = raw_actions.copy()

        if self.output_action_dim == 16:
            # the whole chunk runs before the next query, so its last action is the next commanded pose
            ee16 = _ee6d_to_robotwin_ee16(raw_actions[: self.chunk_size], self._gripper_threshold)
            self._last_commanded[ctx.session_id] = ee16[-1]
            return {"actions": ee16}

        # Convert EE6D 20-D → 7-D when requested
        if self.output_action_dim == 7 and raw_actions.shape[-1] == 20:
            # Resolve per-task gripper threshold (e.g. X-VLA SimplerEnv WidowX)
            threshold = self._gripper_threshold
            if self._gripper_threshold_map is not None:
                task_name = obs.get("task_name", "")
                threshold = self._gripper_threshold_map.get(task_name, threshold)
            converted = _convert_ee6d_to_7d(
                raw_actions,
                threshold,
                self._gripper_close_above,
                self._rot6d_to_matrix,
            )
            # Apply euler offset if configured (convert axis-angle → euler → +offset)
            if self._euler_offset is not None:
                single = converted.ndim == 1
                if single:
                    converted = converted[np.newaxis]
                for i in range(len(converted)):
                    euler = matrix_to_euler_xyz(axisangle_to_matrix(converted[i, 3:6]))
                    converted[i, 3:6] = euler + self._euler_offset
                if single:
                    converted = converted[0]

            # Google Robot: action stride + position accumulation.
            # Matches official X-VLA eval: stride=2, max_chunk=10,
            # positions accumulated to absolute for base_pose controller.
            if self.benchmark_profile == "simpler":
                converted = converted[::2][:10]
                base_pose = obs.get("base_pose")
                tcp_pose = obs.get("tcp_pose")
                if ctx.session_id not in self._current_xyz and base_pose is not None and tcp_pose is not None:
                    self._current_xyz[ctx.session_id] = _compute_ee_pos_wrt_base(
                        np.asarray(base_pose, dtype=np.float64),
                        np.asarray(tcp_pose, dtype=np.float64),
                    )
                cur = self._current_xyz.get(ctx.session_id)
                if cur is not None:
                    converted[:, :3] += cur
                    self._current_xyz[ctx.session_id] = converted[-1, :3].copy()

            return {"actions": converted}

        return {"actions": raw_actions}


if __name__ == "__main__":
    from vla_eval.model_servers.serve import run_server

    run_server(XVLAModelServer)
