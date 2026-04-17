"""Deterministic-logic tests for leaderboard/scripts/refine.py.

Covers the parts that do NOT need an LLM: URL parsing, aggregation-rule
math, the row_type decision table, collapse cross-check, reported_avg
recovery, and leaderboard merging.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# The scripts are UV-run-style (# /// script) with typer as an inline dep
# rather than a project dependency, so typer is not installed in the
# pytest venv. Stub it out so the module-level imports succeed — we do
# not exercise the CLI surface from these tests.
sys.modules.setdefault("typer", MagicMock())

_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS))

import refine  # noqa: E402


# ---------------------------------------------------------------------------
# URL helpers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "url, expected",
    [
        ("https://arxiv.org/abs/2410.12345", "2410.12345"),
        ("http://arxiv.org/abs/2410.12345", "2410.12345"),
        ("https://arxiv.org/abs/2410.12345v2", "2410.12345"),
        ("https://arxiv.org/abs/2410.1234", "2410.1234"),
        ("https://github.com/foo/bar", None),
        ("https://proceedings.mlr.press/v1/paper.pdf", None),
        ("", None),
        (None, None),
    ],
)
def test_arxiv_id_of(url, expected):
    assert refine._arxiv_id_of(url) == expected


@pytest.mark.parametrize(
    "name, expected",
    [
        ("OpenVLA", "openvla"),
        ("PMP (Ours)", "pmp"),
        ("  PointMapPolicy  ", "pointmappolicy"),
        ("3D Diffuser Actor", "3ddiffuseractor"),
        ("X-VLA", "xvla"),
    ],
)
def test_norm_name(name, expected):
    assert refine._norm_name(name) == expected


# ---------------------------------------------------------------------------
# Aggregation rule arithmetic
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_rules(monkeypatch):
    """Stub _aggregation_rules so tests do not depend on benchmarks.json state."""
    rules = {
        "libero": {
            "container": "suite_scores",
            "keys": ["libero_spatial", "libero_object", "libero_goal", "libero_10"],
        },
        "mikasa": {
            "container": "task_scores",
            "keys": [
                "ShellGameTouch",
                "InterceptMedium",
                "RememberColor3",
                "RememberColor5",
                "RememberColor9",
            ],
        },
        "simpler_env": "forbidden",
        # calvin: absent → no rule
    }
    monkeypatch.setattr(refine, "_aggregation_rules", lambda: rules)
    yield


def test_compute_overall_libero_complete(mock_rules):
    suite = {"libero_spatial": 80, "libero_object": 85, "libero_goal": 90, "libero_10": 75}
    assert refine._compute_overall("libero", suite, {}) == 82.5


def test_compute_overall_libero_partial_returns_none(mock_rules):
    suite = {"libero_spatial": 80, "libero_object": 85, "libero_goal": 90}
    assert refine._compute_overall("libero", suite, {}) is None


def test_compute_overall_simpler_env_forbidden(mock_rules):
    suite = {"google_robot_vm": 50, "google_robot_va": 45, "widowx_vm": 30}
    assert refine._compute_overall("simpler_env", suite, {}) is None


def test_compute_overall_calvin_no_rule(mock_rules):
    assert refine._compute_overall("calvin", {"1_task": 90}, {}) is None


def test_compute_overall_mikasa_task_container(mock_rules):
    task = {
        "ShellGameTouch": 50,
        "InterceptMedium": 60,
        "RememberColor3": 70,
        "RememberColor5": 65,
        "RememberColor9": 55,
    }
    assert refine._compute_overall("mikasa", {}, task) == 60.0


# ---------------------------------------------------------------------------
# Row classification decision table
# ---------------------------------------------------------------------------


def _classify(**kwargs):
    defaults = {
        "citing_url": "https://arxiv.org/abs/2501.11111",
        "is_score_original": "original",
        "model_paper": None,
        "cited_paper": None,
        "benchmark": "libero",
        "name_in_paper": "TestMethod",
        "row_index": set(),
    }
    defaults.update(kwargs)
    return refine._classify_row(**defaults)


def test_classify_first_party_citing_equals_model_paper():
    row_type, reported, cited = _classify(
        citing_url="https://arxiv.org/abs/2406.09246",
        is_score_original="original",
        model_paper="https://arxiv.org/abs/2406.09246",
        cited_paper=None,
    )
    assert row_type == "first_party"
    assert reported == "https://arxiv.org/abs/2406.09246"
    assert cited is None


def test_classify_original_but_citing_neq_model_paper_is_third_party():
    # is_score_original='original' but citing paper is not the model's paper
    # → paper ran the method; credit the citing paper as measurer.
    row_type, reported, _ = _classify(
        citing_url="https://arxiv.org/abs/2501.11111",
        is_score_original="original",
        model_paper="https://arxiv.org/abs/2406.09246",
        cited_paper=None,
    )
    assert row_type == "third_party"
    assert reported == "https://arxiv.org/abs/2501.11111"


def test_classify_collapse_hit_drops_row():
    # cited_paper's arxiv == model_paper's arxiv; original paper's extraction
    # contains the matching row → this row is redundant and drops.
    row_index = {("2406.09246", "libero", "openvla")}
    row_type, _, _ = _classify(
        is_score_original="cited_baseline",
        model_paper="https://arxiv.org/abs/2406.09246",
        cited_paper="https://arxiv.org/abs/2406.09246",
        name_in_paper="OpenVLA",
        row_index=row_index,
    )
    assert row_type == "drop"


def test_classify_collapse_miss_demotes_and_nulls_cited():
    # cited arxiv == model arxiv but original paper's extraction lacks the row
    # → cite unverified. Demote to third_party crediting citing paper;
    # null out cited_paper so downstream does not carry the unverified link.
    row_type, reported, cited = _classify(
        citing_url="https://arxiv.org/abs/2501.11111",
        is_score_original="cited_baseline",
        model_paper="https://arxiv.org/abs/2406.09246",
        cited_paper="https://arxiv.org/abs/2406.09246",
        name_in_paper="OpenVLA",
        row_index=set(),
    )
    assert row_type == "third_party"
    assert reported == "https://arxiv.org/abs/2501.11111"
    assert cited is None


def test_classify_third_party_via_different_arxiv_cite():
    row_type, reported, cited = _classify(
        citing_url="https://arxiv.org/abs/2501.11111",
        is_score_original="cited_baseline",
        model_paper="https://arxiv.org/abs/2406.09246",
        cited_paper="https://arxiv.org/abs/2410.00000",
        name_in_paper="OpenVLA",
    )
    assert row_type == "third_party"
    assert reported == "https://arxiv.org/abs/2410.00000"
    assert cited == "https://arxiv.org/abs/2410.00000"


def test_classify_pi0_github_case_stays_citing_attributed():
    # π₀ scenario: citing paper attributes a LIBERO score to π₀'s official
    # github (non-arxiv). π₀'s own paper has no such LIBERO score — we MUST
    # NOT collapse to π₀ paper. Row stays third_party crediting citing.
    row_type, reported, cited = _classify(
        citing_url="https://arxiv.org/abs/2501.11111",
        is_score_original="cited_baseline",
        model_paper="https://arxiv.org/abs/2410.24164",
        cited_paper="https://github.com/physical-intelligence/openpi",
        name_in_paper="pi_0",
    )
    assert row_type == "third_party"
    assert reported == "https://arxiv.org/abs/2501.11111"
    assert cited == "https://github.com/physical-intelligence/openpi"


def test_classify_reproduction_is_third_party():
    row_type, reported, _ = _classify(
        citing_url="https://arxiv.org/abs/2501.11111",
        is_score_original="reproduction",
        model_paper="https://arxiv.org/abs/2406.09246",
        cited_paper=None,
        name_in_paper="OpenVLA",
    )
    assert row_type == "third_party"
    assert reported == "https://arxiv.org/abs/2501.11111"


def test_classify_unknown_with_no_cite_is_third_party():
    row_type, reported, _ = _classify(
        is_score_original="unknown",
        model_paper="https://arxiv.org/abs/2406.09246",
        cited_paper=None,
    )
    assert row_type == "third_party"


# ---------------------------------------------------------------------------
# Merge helper
# ---------------------------------------------------------------------------


def test_merge_leaderboard_replaces_only_touched_benchmarks(tmp_path):
    existing = {
        "last_updated": "2026-01-01",
        "results": [
            {"model": "a", "benchmark": "libero"},
            {"model": "b", "benchmark": "calvin"},
            {"model": "c", "benchmark": "libero"},
        ],
    }
    output = tmp_path / "leaderboard.json"
    output.write_text(json.dumps(existing))
    new_entries = [{"model": "x", "benchmark": "libero"}]
    refine._merge_leaderboard(new_entries, ["libero"], output)

    result = json.loads(output.read_text())
    libero_models = sorted(r["model"] for r in result["results"] if r["benchmark"] == "libero")
    calvin_models = sorted(r["model"] for r in result["results"] if r["benchmark"] == "calvin")
    assert libero_models == ["x"]
    assert calvin_models == ["b"]
    # Sorted by (benchmark, model)
    bms = [(r["benchmark"], r["model"]) for r in result["results"]]
    assert bms == sorted(bms)


def test_merge_leaderboard_on_missing_existing_file(tmp_path):
    output = tmp_path / "leaderboard.json"
    refine._merge_leaderboard(
        [{"model": "m", "benchmark": "libero"}],
        ["libero"],
        output,
    )
    data = json.loads(output.read_text())
    assert len(data["results"]) == 1
    assert data["last_updated"]  # today


# ---------------------------------------------------------------------------
# build_candidates end-to-end with synthetic extractions
# ---------------------------------------------------------------------------


def _make_row(
    name: str,
    *,
    is_score_original: str = "original",
    model_paper: str | None = None,
    cited_paper: str | None = None,
    match: str = "yes",
    overall: float | None = None,
    suite: dict | None = None,
    task: dict | None = None,
) -> dict:
    return {
        "name_in_paper": name,
        "weight_type": "shared",
        "is_score_original": is_score_original,
        "model_paper": model_paper,
        "cited_paper": cited_paper,
        "scores": {
            "overall_score": overall,
            "suite_scores": {k: {"value": v, "quote": ""} for k, v in (suite or {}).items()},
            "task_scores": {k: {"value": v, "quote": ""} for k, v in (task or {}).items()},
        },
        "protocol": {"matches_standard": match, "rationale": "fixture"},
    }


def _make_extraction(arxiv_id: str, benchmark: str, rows: list[dict]) -> dict:
    return {
        "arxiv_id": arxiv_id,
        "extracted_at": "2026-04-17T00:00:00Z",
        "paper_hash": f"sha256:{arxiv_id}",
        "extraction_scope": [benchmark],
        "benchmarks": [{"benchmark": benchmark, "models": rows}],
    }


def _write_extractions(tmp_path: Path, monkeypatch, extractions: list[dict]) -> None:
    ext_dir = tmp_path / "extractions"
    ext_dir.mkdir()
    for ext in extractions:
        (ext_dir / f"{ext['arxiv_id']}.json").write_text(json.dumps(ext))
    monkeypatch.setattr(refine, "EXTRACTIONS_DIR", ext_dir)


def test_build_candidates_yes_complete_suite_computes_overall(tmp_path, monkeypatch, mock_rules):
    ext = _make_extraction(
        "2500.00001",
        "libero",
        [
            _make_row(
                "TestMethod",
                model_paper="https://arxiv.org/abs/2500.00001",
                overall=85.7,
                suite={"libero_spatial": 80, "libero_object": 85, "libero_goal": 90, "libero_10": 75},
            )
        ],
    )
    _write_extractions(tmp_path, monkeypatch, [ext])

    candidates, stats = refine.build_candidates()
    assert len(candidates) == 1
    c = candidates[0]
    assert c["overall_score"] == 82.5  # computed mean, not the paper's 85.7
    assert c["row_type"] == "first_party"


def test_build_candidates_yes_partial_preserves_raw_in_reported_avg(tmp_path, monkeypatch, mock_rules):
    # Regression for defect B: match=yes + aggregation rule + partial
    # components + raw overall → overall_score=None, raw lands in
    # task_scores.reported_avg.
    ext = _make_extraction(
        "2500.00001",
        "libero",
        [
            _make_row(
                "TestMethod",
                model_paper="https://arxiv.org/abs/2500.00001",
                overall=85.7,
                suite={"libero_spatial": 80, "libero_object": 85, "libero_goal": 90},
            )
        ],
    )
    _write_extractions(tmp_path, monkeypatch, [ext])

    candidates, _ = refine.build_candidates()
    assert len(candidates) == 1
    c = candidates[0]
    assert c["overall_score"] is None
    assert c["task_scores"].get("reported_avg") == 85.7
    assert "libero_spatial" in c["suite_scores"]


def test_build_candidates_match_no_with_overall_preserves_reported_avg(tmp_path, monkeypatch, mock_rules):
    ext = _make_extraction(
        "2500.00001",
        "libero",
        [
            _make_row(
                "TestMethod",
                model_paper="https://arxiv.org/abs/2500.00001",
                match="no",
                overall=47.2,
            )
        ],
    )
    _write_extractions(tmp_path, monkeypatch, [ext])

    candidates, stats = refine.build_candidates()
    assert len(candidates) == 1
    assert candidates[0]["overall_score"] is None
    assert candidates[0]["task_scores"].get("reported_avg") == 47.2
    assert stats["rows_match_no_kept_null"] == 1


def test_build_candidates_collapse_drops_when_original_contains_row(tmp_path, monkeypatch, mock_rules):
    # Original paper + citing paper. Citing is cited_baseline quoting the
    # original. Original has the same (benchmark, name) row → citing drops.
    original = _make_extraction(
        "2406.09246",
        "libero",
        [
            _make_row(
                "OpenVLA",
                model_paper="https://arxiv.org/abs/2406.09246",
                overall=84.0,
                suite={"libero_spatial": 80, "libero_object": 85, "libero_goal": 90, "libero_10": 81},
            )
        ],
    )
    citing = _make_extraction(
        "2501.11111",
        "libero",
        [
            _make_row(
                "OpenVLA",
                is_score_original="cited_baseline",
                model_paper="https://arxiv.org/abs/2406.09246",
                cited_paper="https://arxiv.org/abs/2406.09246",
                overall=84.0,
                suite={"libero_spatial": 80, "libero_object": 85, "libero_goal": 90, "libero_10": 81},
            )
        ],
    )
    _write_extractions(tmp_path, monkeypatch, [original, citing])

    candidates, stats = refine.build_candidates()
    assert len(candidates) == 1
    assert candidates[0]["row_type"] == "first_party"
    assert stats["rows_dropped_collapse"] == 1


def test_build_candidates_collapse_miss_demotes_to_third_party_with_null_cite(tmp_path, monkeypatch, mock_rules):
    # Citing cites the original, but the original's extraction exists WITHOUT
    # this row (citing is making a broken attribution). Demote to third_party,
    # reported_paper=citing, cited_paper=None.
    original = _make_extraction("2406.09246", "libero", [])  # no rows
    citing = _make_extraction(
        "2501.11111",
        "libero",
        [
            _make_row(
                "OpenVLA",
                is_score_original="cited_baseline",
                model_paper="https://arxiv.org/abs/2406.09246",
                cited_paper="https://arxiv.org/abs/2406.09246",
                overall=84.0,
                suite={"libero_spatial": 80, "libero_object": 85, "libero_goal": 90, "libero_10": 81},
            )
        ],
    )
    _write_extractions(tmp_path, monkeypatch, [original, citing])

    candidates, _ = refine.build_candidates()
    assert len(candidates) == 1
    c = candidates[0]
    assert c["row_type"] == "third_party"
    assert c["reported_paper"] == "https://arxiv.org/abs/2501.11111"
    assert c["cited_paper"] is None


def test_build_candidates_pi0_github_cite_not_collapsed(tmp_path, monkeypatch, mock_rules):
    # Citing paper cites π₀/libero from github. Must stay third_party
    # credited to citing — the number must NOT be filed under π₀'s paper.
    citing = _make_extraction(
        "2501.11111",
        "libero",
        [
            _make_row(
                "pi_0",
                is_score_original="cited_baseline",
                model_paper="https://arxiv.org/abs/2410.24164",
                cited_paper="https://github.com/physical-intelligence/openpi",
                overall=85.7,
                suite={"libero_spatial": 80, "libero_object": 85, "libero_goal": 90, "libero_10": 88},
            )
        ],
    )
    _write_extractions(tmp_path, monkeypatch, [citing])

    candidates, _ = refine.build_candidates()
    assert len(candidates) == 1
    c = candidates[0]
    assert c["row_type"] == "third_party"
    assert c["reported_paper"] == "https://arxiv.org/abs/2501.11111"
    assert c["cited_paper"] == "https://github.com/physical-intelligence/openpi"


def test_build_candidates_benchmark_filter(tmp_path, monkeypatch, mock_rules):
    libero_full = {
        "libero_spatial": 80,
        "libero_object": 85,
        "libero_goal": 90,
        "libero_10": 81,
    }
    mikasa_full = {
        "ShellGameTouch": 50,
        "InterceptMedium": 60,
        "RememberColor3": 70,
        "RememberColor5": 65,
        "RememberColor9": 55,
    }
    ext = {
        "arxiv_id": "2500.00001",
        "extracted_at": "2026-04-17T00:00:00Z",
        "paper_hash": "sha256:x",
        "extraction_scope": ["libero", "mikasa"],
        "benchmarks": [
            {
                "benchmark": "libero",
                "models": [
                    _make_row(
                        "Libero1",
                        model_paper="https://arxiv.org/abs/2500.00001",
                        suite=libero_full,
                    )
                ],
            },
            {
                "benchmark": "mikasa",
                "models": [
                    _make_row(
                        "Mikasa1",
                        model_paper="https://arxiv.org/abs/2500.00001",
                        task=mikasa_full,
                    )
                ],
            },
        ],
    }
    _write_extractions(tmp_path, monkeypatch, [ext])

    mikasa_only, _ = refine.build_candidates(benchmark_filter="mikasa")
    assert all(c["benchmark"] == "mikasa" for c in mikasa_only)
    assert len(mikasa_only) == 1
