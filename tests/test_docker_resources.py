"""Tests for vla_eval.docker_resources."""

from unittest.mock import patch

from vla_eval.docker_resources import (
    _format_cpuset,
    gpu_docker_flag,
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
# gpu_docker_flag
# ---------------------------------------------------------------------------


class TestGpuDockerFlag:
    def test_none(self):
        assert gpu_docker_flag(None) == ["--gpus", "all"]

    def test_all(self):
        assert gpu_docker_flag("all") == ["--gpus", "all"]

    def test_specific(self):
        assert gpu_docker_flag("0") == ["--gpus", "device=0"]

    def test_multi(self):
        assert gpu_docker_flag("0,1") == ["--gpus", "device=0,1"]


# ---------------------------------------------------------------------------
# shard_docker_flags
# ---------------------------------------------------------------------------


class TestShardDockerFlags:
    @patch("vla_eval.docker_resources._detect_gpu_ids", return_value=["0", "1"])
    def test_single_shard_all_gpus(self, _mock):
        flags = shard_docker_flags(0, 1, gpus="all")
        assert "--gpus" in flags
        idx = flags.index("--gpus")
        assert flags[idx + 1] == "device=0"
        # No cpuset for single shard
        assert "--cpuset-cpus" not in flags
        # OMP always set
        assert "OMP_NUM_THREADS=1" in flags
        assert "MKL_NUM_THREADS=1" in flags

    @patch("vla_eval.docker_resources._detect_gpu_ids", return_value=["0"])
    @patch("vla_eval.docker_resources.os.cpu_count", return_value=16)
    def test_multi_shard_cpu_partition(self, _cpu_mock, _gpu_mock):
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

    def test_gpu_round_robin(self):
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

    def test_explicit_cpu_range(self):
        flags = shard_docker_flags(0, 2, cpus="8-15")
        idx = flags.index("--cpuset-cpus")
        assert flags[idx + 1] == "8-11"

        flags1 = shard_docker_flags(1, 2, cpus="8-15")
        idx = flags1.index("--cpuset-cpus")
        assert flags1[idx + 1] == "12-15"

    def test_non_contiguous_cpu_range(self):
        flags = shard_docker_flags(0, 2, cpus="0-2,12-14")
        idx = flags.index("--cpuset-cpus")
        assert flags[idx + 1] == "0-2"

        flags1 = shard_docker_flags(1, 2, cpus="0-2,12-14")
        idx = flags1.index("--cpuset-cpus")
        assert flags1[idx + 1] == "12-14"

    @patch("vla_eval.docker_resources._detect_gpu_ids", return_value=["0"])
    @patch("vla_eval.docker_resources.os.cpu_count", return_value=4)
    def test_more_shards_than_cpus_skips_cpuset(self, _cpu_mock, _gpu_mock):
        flags = shard_docker_flags(0, 8)
        assert "--cpuset-cpus" not in flags
        # OMP still set
        assert "OMP_NUM_THREADS=1" in flags
