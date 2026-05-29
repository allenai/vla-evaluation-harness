"""Tests for vla_eval.docker_resources."""

from unittest.mock import patch

from vla_eval.docker_resources import (
    _detect_gpu_ids_rocm,
    _detect_runtime,
    _format_cpuset,
    gpu_docker_flag,
    gpu_visibility_env,
    parse_cpus,
    parse_gpus,
    shard_docker_flags,
)


# ---------------------------------------------------------------------------
# _format_cpuset
# ---------------------------------------------------------------------------


class TestFormatCpuset:
    def test_contiguous(self):
        assert _format_cpuset([0, 1, 2, 3]) == "0-3"

    def test_single(self):
        assert _format_cpuset([5]) == "5"

    def test_non_contiguous(self):
        assert _format_cpuset([0, 1, 2, 8, 9, 10]) == "0-2,8-10"

    def test_mixed(self):
        assert _format_cpuset([0, 2, 3, 4, 7]) == "0,2-4,7"

    def test_unsorted_input(self):
        assert _format_cpuset([3, 1, 2, 0]) == "0-3"


# ---------------------------------------------------------------------------
# parse_cpus
# ---------------------------------------------------------------------------


class TestParseCpus:
    def test_range(self):
        assert parse_cpus("0-3") == [0, 1, 2, 3]

    def test_multi_range(self):
        assert parse_cpus("0-2,8-10") == [0, 1, 2, 8, 9, 10]

    def test_individual(self):
        assert parse_cpus("0,2,4") == [0, 2, 4]

    def test_dedup_and_sort(self):
        assert parse_cpus("3,1,2,1") == [1, 2, 3]

    def test_none_returns_all(self):
        with patch("vla_eval.docker_resources.os.cpu_count", return_value=4):
            assert parse_cpus(None) == [0, 1, 2, 3]


# ---------------------------------------------------------------------------
# parse_gpus
# ---------------------------------------------------------------------------


class TestParseGpus:
    @patch("vla_eval.docker_resources._detect_gpu_ids", return_value=["0", "1"])
    def test_none(self, _mock):
        assert parse_gpus(None) == ["0", "1"]

    @patch("vla_eval.docker_resources._detect_gpu_ids", return_value=["0", "1"])
    def test_all(self, _mock):
        assert parse_gpus("all") == ["0", "1"]
        assert parse_gpus("ALL") == ["0", "1"]

    def test_single(self):
        assert parse_gpus("0") == ["0"]

    def test_multi(self):
        assert parse_gpus("0,1,2") == ["0", "1", "2"]


# ---------------------------------------------------------------------------
# runtime detection
# ---------------------------------------------------------------------------


class TestDetectRuntime:
    def teardown_method(self):
        _detect_runtime.cache_clear()

    @patch("vla_eval.docker_resources.subprocess.check_output", return_value="")
    def test_detects_rocm_when_rocm_smi_succeeds(self, _mock):
        _detect_runtime.cache_clear()
        assert _detect_runtime() == "rocm"

    @patch("vla_eval.docker_resources.subprocess.check_output", return_value="")
    def test_detection_is_cached(self, mock_check_output):
        _detect_runtime.cache_clear()
        assert _detect_runtime() == "rocm"
        assert _detect_runtime() == "rocm"
        assert mock_check_output.call_count == 1

    @patch("vla_eval.docker_resources.subprocess.check_output", side_effect=FileNotFoundError)
    def test_defaults_to_nvidia_when_rocm_smi_missing(self, _mock):
        _detect_runtime.cache_clear()
        assert _detect_runtime() == "nvidia"

    @patch("vla_eval.docker_resources.subprocess.check_output", side_effect=PermissionError)
    def test_defaults_to_nvidia_when_rocm_smi_is_not_executable(self, _mock):
        _detect_runtime.cache_clear()
        assert _detect_runtime() == "nvidia"


class TestDetectRocmGpuIds:
    @patch(
        "vla_eval.docker_resources.subprocess.check_output",
        return_value="GPU[0] : GPU ID: 0x740f\nGPU[1] : GPU ID: 0x740f\n",
    )
    def test_parses_rocm_smi_gpu_indices(self, _mock):
        assert _detect_gpu_ids_rocm() == ["0", "1"]

    @patch(
        "vla_eval.docker_resources.subprocess.check_output",
        return_value="GPU[0] : GPU ID: 0x740f\nGPU[0] : Unique ID: 0xabc\nGPU[1] : GPU ID: 0x740f\n",
    )
    def test_deduplicates_rocm_smi_gpu_indices(self, _mock):
        assert _detect_gpu_ids_rocm() == ["0", "1"]

    @patch("vla_eval.docker_resources.glob", return_value=["/dev/dri/renderD128", "/dev/dri/renderD129"])
    @patch("vla_eval.docker_resources.subprocess.check_output", side_effect=FileNotFoundError)
    def test_falls_back_to_render_node_count(self, _subprocess_mock, _glob_mock):
        assert _detect_gpu_ids_rocm() == ["0", "1"]

    @patch("vla_eval.docker_resources.glob", return_value=[])
    @patch("vla_eval.docker_resources.subprocess.check_output", side_effect=FileNotFoundError)
    def test_falls_back_to_zero_when_no_rocm_query_works(self, _subprocess_mock, _glob_mock):
        assert _detect_gpu_ids_rocm() == ["0"]


# ---------------------------------------------------------------------------
# gpu_docker_flag
# ---------------------------------------------------------------------------


class TestGpuDockerFlag:
    @patch("vla_eval.docker_resources._detect_runtime", return_value="nvidia")
    def test_none(self, _runtime_mock):
        assert gpu_docker_flag(None) == ["--gpus", "all"]

    @patch("vla_eval.docker_resources._detect_runtime", return_value="nvidia")
    def test_all(self, _runtime_mock):
        assert gpu_docker_flag("all") == ["--gpus", "all"]

    @patch("vla_eval.docker_resources._detect_runtime", return_value="nvidia")
    def test_specific(self, _runtime_mock):
        assert gpu_docker_flag("0") == ["--gpus", "device=0"]

    @patch("vla_eval.docker_resources._detect_runtime", return_value="nvidia")
    def test_multi(self, _runtime_mock):
        assert gpu_docker_flag("0,1") == ["--gpus", "device=0,1"]


class TestRocmGpuDockerFlag:
    @patch("vla_eval.docker_resources._detect_runtime", return_value="rocm")
    def test_none_leaves_all_rocm_devices_visible(self, _runtime_mock):
        flags = gpu_docker_flag(None)
        assert flags == ["--device=/dev/kfd", "--device=/dev/dri", "--group-add", "video"]
        assert "--ipc=host" not in flags
        assert "--security-opt" not in flags

    @patch("vla_eval.docker_resources._detect_runtime", return_value="rocm")
    def test_all_leaves_all_rocm_devices_visible(self, _runtime_mock):
        flags = gpu_docker_flag("all")
        assert flags == ["--device=/dev/kfd", "--device=/dev/dri", "--group-add", "video"]

    @patch("vla_eval.docker_resources._detect_runtime", return_value="rocm")
    def test_specific_sets_hip_visible_devices(self, _runtime_mock):
        flags = gpu_docker_flag("0,1")
        assert flags == [
            "--device=/dev/kfd",
            "--device=/dev/dri",
            "--group-add",
            "video",
            "-e",
            "HIP_VISIBLE_DEVICES=0,1",
        ]


class TestGpuVisibilityEnv:
    def test_none(self):
        assert gpu_visibility_env(None) == {}

    @patch("vla_eval.docker_resources._detect_runtime", return_value="nvidia")
    def test_nvidia_uses_cuda_visible_devices(self, _runtime_mock):
        assert gpu_visibility_env("1") == {"CUDA_VISIBLE_DEVICES": "1"}

    @patch("vla_eval.docker_resources._detect_runtime", return_value="rocm")
    def test_rocm_uses_hip_visible_devices(self, _runtime_mock):
        assert gpu_visibility_env("1") == {"HIP_VISIBLE_DEVICES": "1"}


# ---------------------------------------------------------------------------
# shard_docker_flags
# ---------------------------------------------------------------------------


class TestShardDockerFlags:
    @patch("vla_eval.docker_resources._detect_runtime", return_value="nvidia")
    @patch("vla_eval.docker_resources._detect_gpu_ids", return_value=["0", "1"])
    def test_single_shard_all_gpus(self, _gpu_mock, _runtime_mock):
        flags = shard_docker_flags(0, 1, gpus="all")
        assert "--gpus" in flags
        idx = flags.index("--gpus")
        assert flags[idx + 1] == "device=0"
        # No cpuset for single shard
        assert "--cpuset-cpus" not in flags
        # OMP always set
        assert "OMP_NUM_THREADS=1" in flags
        assert "MKL_NUM_THREADS=1" in flags

    @patch("vla_eval.docker_resources._detect_runtime", return_value="nvidia")
    @patch("vla_eval.docker_resources._detect_gpu_ids", return_value=["0"])
    @patch("vla_eval.docker_resources.os.cpu_count", return_value=16)
    def test_multi_shard_cpu_partition(self, _cpu_mock, _gpu_mock, _runtime_mock):
        flags = shard_docker_flags(0, 4)
        assert "--cpuset-cpus" in flags
        idx = flags.index("--cpuset-cpus")
        assert flags[idx + 1] == "0-3"

        flags1 = shard_docker_flags(1, 4)
        idx = flags1.index("--cpuset-cpus")
        assert flags1[idx + 1] == "4-7"

        flags3 = shard_docker_flags(3, 4)
        idx = flags3.index("--cpuset-cpus")
        assert flags3[idx + 1] == "12-15"

    @patch("vla_eval.docker_resources._detect_runtime", return_value="nvidia")
    def test_gpu_round_robin(self, _runtime_mock):
        flags0 = shard_docker_flags(0, 4, gpus="0,1")
        flags1 = shard_docker_flags(1, 4, gpus="0,1")
        flags2 = shard_docker_flags(2, 4, gpus="0,1")
        flags3 = shard_docker_flags(3, 4, gpus="0,1")

        def _gpu(f):
            return f[f.index("--gpus") + 1]

        assert _gpu(flags0) == "device=0"
        assert _gpu(flags1) == "device=1"
        assert _gpu(flags2) == "device=0"
        assert _gpu(flags3) == "device=1"

    @patch("vla_eval.docker_resources._detect_runtime", return_value="nvidia")
    @patch("vla_eval.docker_resources._detect_gpu_ids", return_value=["0"])
    def test_explicit_cpu_range(self, _gpu_mock, _runtime_mock):
        flags = shard_docker_flags(0, 2, cpus="8-15")
        idx = flags.index("--cpuset-cpus")
        assert flags[idx + 1] == "8-11"

        flags1 = shard_docker_flags(1, 2, cpus="8-15")
        idx = flags1.index("--cpuset-cpus")
        assert flags1[idx + 1] == "12-15"

    @patch("vla_eval.docker_resources._detect_runtime", return_value="nvidia")
    @patch("vla_eval.docker_resources._detect_gpu_ids", return_value=["0"])
    def test_non_contiguous_cpu_range(self, _gpu_mock, _runtime_mock):
        flags = shard_docker_flags(0, 2, cpus="0-2,12-14")
        idx = flags.index("--cpuset-cpus")
        assert flags[idx + 1] == "0-2"

        flags1 = shard_docker_flags(1, 2, cpus="0-2,12-14")
        idx = flags1.index("--cpuset-cpus")
        assert flags1[idx + 1] == "12-14"

    @patch("vla_eval.docker_resources._detect_runtime", return_value="nvidia")
    @patch("vla_eval.docker_resources._detect_gpu_ids", return_value=["0"])
    @patch("vla_eval.docker_resources.os.cpu_count", return_value=4)
    def test_more_shards_than_cpus_skips_cpuset(self, _cpu_mock, _gpu_mock, _runtime_mock):
        flags = shard_docker_flags(0, 8)
        assert "--cpuset-cpus" not in flags
        # OMP still set
        assert "OMP_NUM_THREADS=1" in flags


class TestRocmShardDockerFlags:
    @patch("vla_eval.docker_resources._detect_runtime", return_value="rocm")
    @patch("vla_eval.docker_resources._detect_gpu_ids", return_value=["0", "1"])
    def test_rocm_shards_round_robin_with_hip_visible_devices(self, _gpu_mock, _runtime_mock):
        flags0 = shard_docker_flags(0, 2, gpus="all", cpus="0-7")
        flags1 = shard_docker_flags(1, 2, gpus="all", cpus="0-7")

        assert "--gpus" not in flags0
        assert "--device=/dev/kfd" in flags0
        assert "--device=/dev/dri" in flags0
        assert flags0[flags0.index("--group-add") + 1] == "video"
        assert "--ipc=host" not in flags0
        assert "--security-opt" not in flags0
        assert "HIP_VISIBLE_DEVICES=0" in flags0
        assert "HIP_VISIBLE_DEVICES=1" in flags1

        assert flags0[flags0.index("--cpuset-cpus") + 1] == "0-3"
        assert flags1[flags1.index("--cpuset-cpus") + 1] == "4-7"
        assert "OMP_NUM_THREADS=1" in flags0
        assert "MKL_NUM_THREADS=1" in flags0

    @patch("vla_eval.docker_resources._detect_runtime", return_value="rocm")
    def test_rocm_explicit_gpu_round_robin(self, _runtime_mock):
        flags = shard_docker_flags(2, 4, gpus="2,3")
        assert "HIP_VISIBLE_DEVICES=2" in flags
