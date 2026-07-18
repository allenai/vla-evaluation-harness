"""Run a short in-process RoboCasa365 + GR00T vla-eval integration check."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import anyio

from vla_eval.model_servers.robocasa_groot import RoboCasaGR00TN15ModelServer
from vla_eval.model_servers.serve import serve_async
from vla_eval.orchestrator import Orchestrator
from vla_eval.results.merge import merge_eval


async def run(args: argparse.Namespace) -> dict[str, Any]:
    server = RoboCasaGR00TN15ModelServer(
        model_path=str(args.checkpoint),
        checkpoint_revision=args.checkpoint_revision,
        seed=args.seed,
    )
    config = {
        "server": {"url": f"ws://127.0.0.1:{args.port}", "timeout": 300},
        "output_dir": str(args.output_dir),
        "benchmarks": [
            {
                "benchmark": "vla_eval.benchmarks.robocasa.benchmark:RoboCasaBenchmark",
                "name": args.name,
                "episodes_per_task": args.episodes,
                "max_steps": args.max_steps,
                "params": {
                    "tasks": [args.task],
                    "split": args.split,
                    "seed": args.seed,
                    "camera_size": 256,
                    "enable_render": True,
                },
                "recording": {"record_step": True, "record_video": False},
            }
        ],
    }
    orchestrator = Orchestrator(config, eval_id=args.eval_id)
    async with anyio.create_task_group() as task_group:
        task_group.start_soon(serve_async, server, "127.0.0.1", args.port)
        await anyio.sleep(0.25)
        try:
            results = await orchestrator.run()
        finally:
            task_group.cancel_scope.cancel()

    aggregates = merge_eval(args.output_dir, orchestrator.eval_id)
    result = results[0]
    episodes = result["tasks"][0]["episodes"]
    smoke_pass = bool(
        len(episodes) == args.episodes
        and not any(episode.get("failure_reason") for episode in episodes)
        and result["server_info"]["model_metadata"]["checkpoint_revision"]
        == args.checkpoint_revision
        and result["server_info"]["model_metadata"]["policy_seed"] == args.seed
        and result["benchmark_metadata"]["upstream"]["robocasa"]["revision"]
        == "b4684e6ee37d377cc392e98302a6b916d588b415"
    )
    summary = {
        "pass": smoke_pass,
        "task": args.task,
        "split": args.split,
        "seed": args.seed,
        "episodes": args.episodes,
        "max_steps": args.max_steps,
        "result": result,
        "aggregate": aggregates[0],
    }
    (args.output_dir / "integration_summary.json").write_text(
        json.dumps(summary, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--checkpoint-revision", required=True)
    parser.add_argument("--task", default="KettleBoiling")
    parser.add_argument("--split", choices=["pretrain", "target"], default="pretrain")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--eval-id", default="robocasa-groot-smoke")
    parser.add_argument("--name", default="RoboCasa365GR00TSmoke")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary = anyio.run(run, args)
    if not summary["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
