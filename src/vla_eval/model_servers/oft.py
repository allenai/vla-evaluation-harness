# /// script
# requires-python = "~=3.11"
# dependencies = [
#     "vla-eval",
#     "openvla-oft",
#     "torch>=2.2",
#     "transformers>=4.40,<5",
#     "pillow>=9.0",
#     "numpy>=1.24",
#     "accelerate",
#     "timm",
# ]
#
# [tool.uv.sources]
# vla-eval = { path = "../../.." }
# openvla-oft = { git = "https://github.com/moojink/openvla-oft.git", rev = "e4287e94541f459edc4feabc4e181f537cd569a8" }
#
# [tool.uv]
# exclude-newer = "2026-02-24T00:00:00Z"
# ///
"""OpenVLA-OFT model server.

Uses the openvla-oft repo for fine-tuned OpenVLA checkpoints with
action chunking and parallel decoding (26× faster, 3× lower latency).
"""

from __future__ import annotations

import argparse
import logging
from typing import Any

import numpy as np

from vla_eval.types import Action, Observation

from vla_eval.model_servers.base import SessionContext
from vla_eval.model_servers.predict import PredictModelServer
from vla_eval.model_servers.serve import serve

logger = logging.getLogger(__name__)


class OFTModelServer(PredictModelServer):
    """OpenVLA-OFT model server with action chunking.

    Loads a fine-tuned OpenVLA-OFT checkpoint and generates action
    chunks using L1 regression or diffusion action heads.
    """

    def __init__(
        self,
        pretrained_checkpoint: str,
        unnorm_key: str = "",
        use_l1_regression: bool = True,
        use_diffusion: bool = False,
        use_film: bool = False,
        num_images_in_input: int = 1,
        use_proprio: bool = True,
        center_crop: bool = True,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        *,
        chunk_size: int = 10,
        action_ensemble: str = "newest",
        **kwargs: Any,
    ) -> None:
        super().__init__(chunk_size=chunk_size, action_ensemble=action_ensemble, **kwargs)
        self.pretrained_checkpoint = pretrained_checkpoint
        self.unnorm_key = unnorm_key
        self.use_l1_regression = use_l1_regression
        self.use_diffusion = use_diffusion
        self.use_film = use_film
        self.num_images_in_input = num_images_in_input
        self.use_proprio = use_proprio
        self.center_crop = center_crop
        self.load_in_8bit = load_in_8bit
        self.load_in_4bit = load_in_4bit
        self._vla = None
        self._processor = None
        self._action_head = None
        self._proprio_projector = None
        self._cfg = None

    def _load_model(self) -> None:
        if self._vla is not None:
            return
        import types

        from experiments.robot.openvla_utils import get_action_head, get_processor, get_proprio_projector, get_vla
        from prismatic.vla.constants import NUM_ACTIONS_CHUNK, PROPRIO_DIM

        logger.info("Loading OpenVLA-OFT from %s", self.pretrained_checkpoint)
        # Use SimpleNamespace instead of GenerateConfig to avoid
        # importing libero (transitive dep of run_libero_eval.py).
        self._cfg = types.SimpleNamespace(
            pretrained_checkpoint=self.pretrained_checkpoint,
            model_family="openvla",
            use_l1_regression=self.use_l1_regression,
            use_diffusion=self.use_diffusion,
            num_diffusion_steps_train=50,
            num_diffusion_steps_inference=50,
            use_film=self.use_film,
            num_images_in_input=self.num_images_in_input,
            use_proprio=self.use_proprio,
            center_crop=self.center_crop,
            load_in_8bit=self.load_in_8bit,
            load_in_4bit=self.load_in_4bit,
            num_open_loop_steps=NUM_ACTIONS_CHUNK,
            unnorm_key=self.unnorm_key,
            lora_rank=32,
        )
        self._vla = get_vla(self._cfg)
        self._processor = get_processor(self._cfg)
        self._action_head = get_action_head(self._cfg, llm_dim=self._vla.llm_dim)
        if self.use_proprio:
            self._proprio_projector = get_proprio_projector(
                self._cfg, llm_dim=self._vla.llm_dim, proprio_dim=PROPRIO_DIM
            )
        logger.info("OpenVLA-OFT model loaded.")

    def predict(self, obs: Observation, ctx: SessionContext) -> Action:
        from experiments.robot.openvla_utils import get_vla_action
        from prismatic.vla.constants import PROPRIO_DIM

        self._load_model()
        assert self._vla is not None and self._cfg is not None

        # Build OFT observation dict — get_vla_action expects numpy arrays (uint8 HWC)
        images_dict = obs.get("images", {})
        keys = list(images_dict.keys()) if isinstance(images_dict, dict) else []
        primary_img = (
            np.asarray(images_dict[keys[0]], dtype=np.uint8) if keys else np.zeros((256, 256, 3), dtype=np.uint8)
        )
        oft_obs: dict[str, Any] = {"full_image": primary_img}

        if len(keys) > 1:
            oft_obs["wrist_image"] = np.asarray(images_dict[keys[1]], dtype=np.uint8)

        # Provide proprio state; fall back to zeros when missing
        # LIBERO sends "states" (plural), other benchmarks may use "state" (singular)
        raw_state = obs.get("states", obs.get("state"))
        if raw_state is not None:
            oft_obs["state"] = np.asarray(raw_state, dtype=np.float64)
        elif self.use_proprio:
            oft_obs["state"] = np.zeros(PROPRIO_DIM, dtype=np.float64)

        task_desc = obs.get("task_description", "")
        actions = get_vla_action(
            self._cfg,
            self._vla,
            self._processor,
            oft_obs,
            task_desc,
            self._action_head,
            self._proprio_projector,
        )
        # Gripper: RLDS [0=close,1=open] → robosuite [-1=open,+1=close]
        actions_arr = np.asarray(actions, dtype=np.float32)
        actions_arr[..., -1] = -np.sign(2 * actions_arr[..., -1] - 1)
        return {"actions": actions_arr}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OpenVLA-OFT model server (uv script)")
    parser.add_argument(
        "--pretrained_checkpoint",
        required=True,
        help="HuggingFace model ID (e.g. moojink/openvla-7b-oft-finetuned-libero-spatial)",
    )
    parser.add_argument("--unnorm_key", default="", help="Unnormalization key")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--chunk_size", type=int, default=10)
    parser.add_argument("--action_ensemble", default="newest")
    parser.add_argument(
        "--num_images_in_input", type=int, default=1, help="Number of input images (1=full_image only, 2=full+wrist)"
    )
    parser.add_argument("--use_diffusion", action="store_true")
    parser.add_argument("--no_proprio", action="store_true")
    parser.add_argument("--load_in_8bit", action="store_true")
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )

    server = OFTModelServer(
        pretrained_checkpoint=args.pretrained_checkpoint,
        unnorm_key=args.unnorm_key,
        use_l1_regression=not args.use_diffusion,
        use_diffusion=args.use_diffusion,
        num_images_in_input=args.num_images_in_input,
        use_proprio=not args.no_proprio,
        chunk_size=args.chunk_size,
        action_ensemble=args.action_ensemble,
        load_in_8bit=args.load_in_8bit,
        load_in_4bit=args.load_in_4bit,
    )

    logger.info("Pre-loading model...")
    server._load_model()
    logger.info("Model ready, starting server on ws://%s:%d", args.host, args.port)
    serve(server, host=args.host, port=args.port)
