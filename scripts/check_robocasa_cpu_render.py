#!/usr/bin/env python3
"""Create, step, and render the three RC365 cameras with Mesa llvmpipe."""

from __future__ import annotations

import json
import os

os.environ["VLA_EVAL_RENDER"] = "cpu"
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np

from vla_eval.benchmarks.robocasa.rc365 import ACTION_DIM, VIDEO_KEYS, RoboCasa365Benchmark
from vla_eval.recording import NullEpisodeRecorder


def main() -> None:
    benchmark = RoboCasa365Benchmark(
        tasks=["RinseSinkBasin"],
        split="pretrain",
        seed=0,
        max_steps=16,
    )
    benchmark._recorder = NullEpisodeRecorder()
    try:
        task = {"name": "RinseSinkBasin", "episode_idx": 0}
        raw = benchmark.reset(task)
        first = benchmark.make_obs(raw, task)
        stepped = benchmark.step({"actions": np.zeros(ACTION_DIM, dtype=np.float32)})
        second = benchmark.make_obs(stepped.obs, task)
        for observation in (first, second):
            images = observation["images"]
            missing = [key for key in VIDEO_KEYS if key not in images]
            if missing:
                raise RuntimeError(f"missing rendered cameras: {missing}")
            for key in VIDEO_KEYS:
                image = np.asarray(images[key])
                if image.shape != (256, 256, 3) or image.dtype != np.uint8:
                    raise RuntimeError(f"unexpected {key} image: shape={image.shape}, dtype={image.dtype}")
        print(
            json.dumps(
                {
                    "camera_shapes": {key: list(np.asarray(second["images"][key]).shape) for key in VIDEO_KEYS},
                    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                    "egl_platform": os.environ.get("EGL_PLATFORM"),
                    "mujoco_gl": os.environ.get("MUJOCO_GL"),
                    "pyopengl_platform": os.environ.get("PYOPENGL_PLATFORM"),
                    "pass": True,
                },
                sort_keys=True,
            )
        )
    finally:
        benchmark.cleanup()


if __name__ == "__main__":
    main()
