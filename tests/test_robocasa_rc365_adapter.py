"""CPU-only tests for RoboCasa365 reset ownership."""

from __future__ import annotations

import random
from types import SimpleNamespace

import numpy as np

from vla_eval.benchmarks.robocasa.benchmark import RoboCasaBenchmark, configure_robocasa_rendering
from vla_eval.benchmarks.robocasa.rc365 import ACTION_DIM, RoboCasa365Benchmark
from vla_eval.rc365_s2b_qualification import RoboCasaS2BQualificationBenchmark
from vla_eval.recording import NullEpisodeRecorder


def test_robocasa_reset_owns_episode_and_process_seeds(monkeypatch):
    class FakeEnv:
        def __init__(self) -> None:
            self.samples: list[tuple[float, float]] = []
            self.closed = False

        def reset(self):
            self.samples.append((random.random(), float(np.random.random())))
            return {}

        @staticmethod
        def get_ep_meta():
            return {}

        def close(self):
            self.closed = True

    envs: list[FakeEnv] = []
    constructions: list[tuple[str, int | None]] = []
    benchmark = RoboCasaBenchmark(
        tasks=["Task"],
        seed=5,
        max_steps=32,
    )
    benchmark._recorder = NullEpisodeRecorder()

    def make_env(task_name, *, episode_seed):
        constructions.append((task_name, episode_seed))
        env = FakeEnv()
        envs.append(env)
        return env

    monkeypatch.setattr(benchmark, "_make_env", make_env)
    task = {"name": "Task", "episode_idx": 3}
    benchmark.reset(task)
    benchmark.reset(task)
    benchmark.reset({"name": "Task", "episode_idx": 4})

    assert constructions == [("Task", 8), ("Task", 9)]
    assert envs[0].samples[0] == envs[0].samples[1]
    assert envs[0].closed is True


def test_rc365_rebuilds_environment_when_episode_seed_changes(monkeypatch):
    class FakeEnv:
        def __init__(self) -> None:
            self.reset_seeds: list[int | None] = []
            self.closed = False

        def reset(self, *, seed):
            self.reset_seeds.append(seed)
            return {}, {}

        def close(self):
            self.closed = True

    envs: list[FakeEnv] = []
    constructions: list[tuple[str, int | None]] = []
    benchmark = RoboCasa365Benchmark(tasks=["Task"], seed=5, max_steps=32)
    benchmark._recorder = NullEpisodeRecorder()

    def make_env(task_name, *, episode_seed):
        constructions.append((task_name, episode_seed))
        env = FakeEnv()
        envs.append(env)
        return env

    monkeypatch.setattr(benchmark, "_make_env", make_env)
    benchmark.reset({"name": "Task", "episode_idx": 3})
    benchmark.reset({"name": "Task", "episode_idx": 3})
    benchmark.reset({"name": "Task", "episode_idx": 4})

    assert constructions == [("Task", 8), ("Task", 9)]
    assert envs[0].reset_seeds == [8, 8]
    assert envs[0].closed is True
    assert envs[1].reset_seeds == [9]


def test_cpu_render_toggle_selects_osmesa_and_clears_egl_bindings():
    environ = {
        "VLA_EVAL_RENDER": "cpu",
        "EGL_PLATFORM": "device",
        "MUJOCO_EGL_DEVICE_ID": "3",
    }

    assert configure_robocasa_rendering(environ) == "cpu"
    assert environ["MUJOCO_GL"] == "osmesa"
    assert environ["PYOPENGL_PLATFORM"] == "osmesa"
    assert environ["LIBGL_ALWAYS_SOFTWARE"] == "1"
    assert "EGL_PLATFORM" not in environ
    assert "MUJOCO_EGL_DEVICE_ID" not in environ


def test_empty_render_toggle_uses_gpu_default():
    environ = {"VLA_EVAL_RENDER": ""}

    assert configure_robocasa_rendering(environ) == "gpu"
    assert environ["MUJOCO_GL"] == "egl"
    assert environ["EGL_PLATFORM"] == "device"


def test_qualification_checks_success_each_step_and_stops_at_chunk_boundary(monkeypatch):
    class FakeEnv:
        def __init__(self) -> None:
            self.steps = 0
            self.unwrapped = SimpleNamespace(env=self)

        def reset(self, *, seed):
            assert seed == 7
            self.steps = 0
            return {}, {}

        def step(self, action):
            assert set(action) == {
                "action.base_motion",
                "action.control_mode",
                "action.end_effector_position",
                "action.end_effector_rotation",
                "action.gripper_close",
            }
            self.steps += 1
            return {}, 0.0, self.steps == 2, False, {}

        def _check_success(self):
            return self.steps >= 3

        @staticmethod
        def get_ep_meta():
            return {"lang": "complete the task"}

        def close(self):
            pass

    env = FakeEnv()
    benchmark = RoboCasaS2BQualificationBenchmark(
        tasks=["Task"],
        seed=7,
        max_steps=20,
        qualification_condition="global-s1",
    )
    benchmark._recorder = NullEpisodeRecorder()
    monkeypatch.setattr(benchmark, "_make_env", lambda *args, **kwargs: env)
    benchmark.reset({"name": "Task", "episode_idx": 0})

    results = [benchmark.step({"actions": np.zeros(ACTION_DIM)}) for _ in range(16)]
    details = benchmark.get_step_result(results[-1])["_rc365_s2b"]

    assert [result.done for result in results] == [False] * 15 + [True]
    assert details["success_first_step"] == 3
    assert details["chunks"] == [
        {
            "index": 0,
            "step_start": 0,
            "step_end": 16,
            "steps": 16,
            "strict_success": True,
            "became_successful": True,
            "env_terminated": True,
            "env_truncated": False,
        }
    ]
