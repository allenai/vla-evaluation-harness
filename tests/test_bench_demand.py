"""Tests for demand-benchmark resource monitoring helpers."""

import json
from unittest.mock import patch

from experiments.bench_demand import ResourceMonitor


class TestResourceMonitorGpuStats:
    @patch("vla_eval.docker_resources._detect_runtime", return_value="nvidia")
    @patch(
        "experiments.bench_demand.subprocess.check_output",
        return_value="10, 1024, 8192\n75, 2048, 8192\n",
    )
    def test_nvidia_gpu_stats(self, _check_output_mock, _runtime_mock):
        assert ResourceMonitor._gpu_stats() == {
            "gpu_util_pct": 75.0,
            "gpu_mem_used_gb": 3.0,
            "gpu_mem_total_gb": 16.0,
        }

    @patch("vla_eval.docker_resources._detect_runtime", return_value="rocm")
    @patch(
        "experiments.bench_demand.subprocess.check_output",
        return_value=json.dumps(
            {
                "card0": {
                    "GPU use (%)": "15%",
                    "VRAM Total Memory (B)": str(16 * 1024**3),
                    "VRAM Total Used Memory (B)": str(2 * 1024**3),
                },
                "card1": {
                    "GPU use (%)": "80%",
                    "VRAM Total Memory (B)": str(32 * 1024**3),
                    "VRAM Total Used Memory (B)": str(int(3.5 * 1024**3)),
                },
            }
        ),
    )
    def test_rocm_gpu_stats(self, _check_output_mock, _runtime_mock):
        assert ResourceMonitor._gpu_stats() == {
            "gpu_util_pct": 80.0,
            "gpu_mem_used_gb": 5.5,
            "gpu_mem_total_gb": 48.0,
        }

    @patch("vla_eval.docker_resources._detect_runtime", return_value="rocm")
    @patch("experiments.bench_demand.subprocess.check_output", side_effect=FileNotFoundError)
    def test_rocm_gpu_stats_falls_back_to_zero(self, _check_output_mock, _runtime_mock):
        assert ResourceMonitor._gpu_stats() == {
            "gpu_util_pct": 0.0,
            "gpu_mem_used_gb": 0.0,
            "gpu_mem_total_gb": 0.0,
        }

    def test_rocm_parser_handles_mib_values(self):
        assert ResourceMonitor._parse_rocm_card_stats(
            {
                "GPU use (%)": "42%",
                "VRAM Total Memory (MiB)": "65536",
                "VRAM Total Used Memory (MiB)": "12288",
            }
        ) == (42.0, 12.0, 64.0)
