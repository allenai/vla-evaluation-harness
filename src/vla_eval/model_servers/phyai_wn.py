"""Torchrun entrypoint for pi0.5 PhyAI data-parallel LIBERO serving."""

from __future__ import annotations

import argparse
import logging
import os

from vla_eval.model_servers.phyai import PhyAIModelServer
from vla_eval.model_servers.serve import serve

logger = logging.getLogger(__name__)


def _rank_env() -> tuple[int, int, int]:
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    return rank, local_rank, world_size


def _resolve_device(device: str, local_rank: int) -> str:
    if device == "cuda":
        return f"cuda:{local_rank}"
    return device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--params_dtype", default="bfloat16")
    parser.add_argument("--attn_backend", default="flashinfer")
    parser.add_argument("--norm_backend", default="phyai-kernel")
    parser.add_argument("--linear_backend", default="flashinfer")
    parser.add_argument("--flashinfer_workspace_bytes", type=int, default=512 * 1024 * 1024)
    parser.add_argument("--chunk_size", type=int, default=10)
    parser.add_argument("--policy_chunk_size", type=int, default=None)
    parser.add_argument("--max_batch_size", type=int, default=32)
    parser.add_argument("--max_wait_time", type=float, default=0.05)
    parser.add_argument("--send_action_chunks", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--dp_size", type=int, default=None)
    parser.add_argument("--use_cuda_graph", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s [rank=%(process)d]: %(message)s",
    )

    rank, local_rank, world_size = _rank_env()
    dp_size = int(args.dp_size or world_size)
    if world_size != dp_size:
        raise SystemExit(f"WORLD_SIZE={world_size} must equal --dp_size={dp_size} for pi05_wn")

    device = _resolve_device(args.device, local_rank)
    server = PhyAIModelServer(
        checkpoint_path=args.checkpoint_path,
        device=device,
        params_dtype=args.params_dtype,
        use_cuda_graph=args.use_cuda_graph,
        attn_backend=args.attn_backend,
        norm_backend=args.norm_backend,
        linear_backend=args.linear_backend,
        flashinfer_workspace_bytes=args.flashinfer_workspace_bytes,
        chunk_size=args.chunk_size,
        policy_chunk_size=args.policy_chunk_size,
        max_batch_size=args.max_batch_size,
        max_wait_time=args.max_wait_time,
        send_action_chunks=args.send_action_chunks,
        engine_plugin="pi05_wn",
        world_size=world_size,
        dp_size=dp_size,
    )

    logger.info(
        "Loading pi05_wn server rank=%d local_rank=%d world_size=%d device=%s batch=%d",
        rank,
        local_rank,
        world_size,
        device,
        args.max_batch_size,
    )
    server._load_model()

    try:
        if rank == 0:
            logger.info("Starting rank0 WebSocket server on ws://%s:%d", args.host, args.port)
            serve(server, host=args.host, port=args.port)
        else:
            logger.info("Rank %d entering distributed worker loop", rank)
            server.run_distributed_worker()
    finally:
        if rank == 0:
            server.stop_distributed_workers()
        server.close()


if __name__ == "__main__":
    main()
