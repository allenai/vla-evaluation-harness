"""Harness errors stay in the success denominator but are surfaced at the top level."""

from __future__ import annotations

from typing import Any, cast

from vla_eval.results.collector import EpisodeResult, ResultCollector


def _collector(*failure_reasons: str | None) -> ResultCollector:
    c = ResultCollector(benchmark_name="B", mode="sync", metric_keys={"success": "mean"})
    for i, reason in enumerate(failure_reasons):
        ep: dict[str, Any] = {"episode_id": i, "metrics": {"success": reason is None}}
        if reason:
            ep["failure_reason"] = reason
        c.record("task", cast(EpisodeResult, ep))
    return c


def test_num_errors_promoted_to_benchmark_level() -> None:
    result = _collector(None, "timeout", None, "exception").get_benchmark_result()
    assert result["num_errors"] == 2
    assert result.get("mean_success") == 0.5  # errors still count as failures


def test_num_errors_present_when_zero() -> None:
    assert _collector(None, None).get_benchmark_result()["num_errors"] == 0
