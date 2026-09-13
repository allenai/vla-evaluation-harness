"""Train LeRobot's Diffusion Policy on Push-T; evaluate with vla-eval every N steps.

Push-T is not a vla-eval benchmark: ``benchmark/`` holds the adapter, the YAML naming it,
and a Dockerfile. The adapter runs in this process, or with ``--docker`` in the image the
YAML builds. Training is LeRobot's Push-T example; the addition is one call inside the loop::

    results = vla_eval.evaluate(PolicyServer(policy, ...), "benchmark/eval.yaml", ...)
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import vla_eval
from lerobot.configs import FeatureType
from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies import make_pre_post_processors
from lerobot.policies.diffusion import DiffusionConfig, DiffusionPolicy
from lerobot.policies.utils import prepare_observation_for_inference
from lerobot.utils.feature_utils import dataset_to_policy_features
from vla_eval.model_servers.predict import PredictModelServer
from vla_eval.specs import IMAGE_RGB, DimSpec

from benchmark.pusht import ACTION_XY_ABSOLUTE, STATE_AGENT_POS

HERE = Path(__file__).resolve().parent


class PolicyServer(PredictModelServer):
    """Serves the live policy. ``select_action`` keeps the policy's own observation
    history and action queue, so one action comes back per observation."""

    def __init__(self, policy: DiffusionPolicy, preprocess: Any, postprocess: Any, device: torch.device) -> None:
        super().__init__()
        self.policy, self.preprocess, self.postprocess, self.device = policy, preprocess, postprocess, device

    def predict(self, obs: dict[str, Any], ctx: Any) -> dict[str, Any]:
        frame = {
            "observation.image": np.ascontiguousarray(obs["images"]["agentview"], dtype=np.uint8),
            "observation.state": np.asarray(obs["state"], dtype=np.float32),
        }
        frame = self.preprocess(prepare_observation_for_inference(frame, self.device))
        with torch.inference_mode():
            action = self.postprocess(self.policy.select_action(frame))
        return {"actions": action.squeeze(0).cpu().numpy().astype(np.float32)}

    async def on_episode_start(self, config: dict[str, Any], ctx: Any) -> None:
        self.policy.reset()  # clear the history/action queues between episodes
        await super().on_episode_start(config, ctx)

    def get_action_spec(self) -> dict[str, DimSpec]:
        return {"position": ACTION_XY_ABSOLUTE}

    def get_observation_spec(self) -> dict[str, DimSpec]:
        return {"image": IMAGE_RGB, "state": STATE_AGENT_POS}


def evaluate(server: PolicyServer, args: argparse.Namespace) -> dict[str, float]:
    server.policy.eval()
    try:
        results = vla_eval.evaluate(
            server,
            HERE / "benchmark" / "eval.yaml",
            docker=args.docker,
            runtime=args.runtime,
            no_save=not args.docker,  # container runs report through the recording on disk
            output_dir=args.output_dir / "eval",
            benchmark_overrides={"episodes_per_task": args.eval_episodes, "max_steps": args.eval_max_steps},
        )
    finally:
        server.policy.train()
    bench = results[0]
    if bench.get("num_errors"):
        raise RuntimeError(f"{bench['num_errors']} eval episodes errored; see the harness log above")
    return {"success": float(bench["mean_success"]), "coverage": float(bench["mean_coverage"])}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--eval-every", type=int, default=2000)
    p.add_argument("--eval-episodes", type=int, default=20)
    p.add_argument("--eval-max-steps", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--output-dir", type=Path, default=Path("outputs"))
    p.add_argument(
        "--docker", action="store_true", help="run the benchmark in the image built from benchmark/Dockerfile"
    )
    p.add_argument(
        "--runtime", choices=["docker", "charliecloud"], default=None, help="container runtime for --docker"
    )
    args = p.parse_args()

    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # LeRobot's Push-T setup.
    meta = LeRobotDatasetMetadata("lerobot/pusht")
    features = dataset_to_policy_features(meta.features)
    output_features = {k: f for k, f in features.items() if f.type is FeatureType.ACTION}
    input_features = {k: f for k, f in features.items() if k not in output_features}
    cfg = DiffusionConfig(input_features=input_features, output_features=output_features, device=str(device))
    policy = DiffusionPolicy(cfg).to(device)
    preprocess, postprocess = make_pre_post_processors(
        cfg, dataset_stats=meta.stats, preprocessor_overrides={"device_processor": {"device": str(device)}}
    )
    delta_timestamps = {
        "observation.image": [i / meta.fps for i in cfg.observation_delta_indices],
        "observation.state": [i / meta.fps for i in cfg.observation_delta_indices],
        "action": [i / meta.fps for i in cfg.action_delta_indices],
    }
    dataset = LeRobotDataset("lerobot/pusht", delta_timestamps=delta_timestamps, video_backend="pyav")
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=True,
        pin_memory=device.type == "cuda",
    )
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)

    server = PolicyServer(policy, preprocess, postprocess, device)
    curve = args.output_dir / "curve.jsonl"
    t0, step = time.time(), 0
    policy.train()
    while step < args.steps:
        for batch in loader:
            loss, _ = policy.forward(preprocess(batch))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            step += 1
            if step % args.eval_every == 0 or step == args.steps:
                metrics = evaluate(server, args)
                row = {"step": step, "loss": loss.item(), "elapsed_s": round(time.time() - t0), **metrics}
                print(
                    f"step {step}  loss {row['loss']:.4f}  success {metrics['success']:.0%}  coverage {metrics['coverage']:.3f}"
                )
                with curve.open("a") as f:
                    f.write(json.dumps(row) + "\n")
            if step >= args.steps:
                break

    policy.save_pretrained(args.output_dir / "policy")
    preprocess.save_pretrained(args.output_dir / "policy")
    postprocess.save_pretrained(args.output_dir / "policy")


if __name__ == "__main__":
    main()
