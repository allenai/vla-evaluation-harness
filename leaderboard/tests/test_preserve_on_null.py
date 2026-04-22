"""Unit tests for refine's preserve-on-null merge.

When refine re-runs a benchmark, the LLM re-emits every row for that
benchmark. Ideally the LLM carries curator-set metadata (``model_paper``,
corrected ``notes``, etc.) forward because the prompt instructs it to
read existing rows. But LLMs are non-deterministic and occasionally null
a field they shouldn't. The ``_preserve_on_null`` post-step is the
deterministic safety net: for each new row, any field in the preserve
list that's null-in-new-but-set-in-old gets restored from the old row.
"""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from refine import PRESERVE_ON_NULL_FIELDS, preserve_on_null  # noqa: E402


def _row(**kw) -> dict:
    base = {"benchmark": "libero", "model": "x", "model_paper": None, "display_name": None, "notes": None}
    base.update(kw)
    return base


def test_null_field_restored_from_existing():
    """LLM re-emitted the row but dropped model_paper — restore from existing."""
    old = [_row(benchmark="libero", model="foo", model_paper="https://arxiv.org/abs/2024.00001")]
    new = [_row(benchmark="libero", model="foo", model_paper=None)]
    preserve_on_null(new, old)
    assert new[0]["model_paper"] == "https://arxiv.org/abs/2024.00001"


def test_non_null_llm_value_wins():
    """When LLM emits a non-null value, it overrides existing — LLM is authoritative."""
    old = [_row(benchmark="libero", model="foo", model_paper="https://arxiv.org/abs/2024.00001")]
    new = [_row(benchmark="libero", model="foo", model_paper="https://arxiv.org/abs/2024.99999")]
    preserve_on_null(new, old)
    assert new[0]["model_paper"] == "https://arxiv.org/abs/2024.99999"


def test_new_row_passes_through():
    """Row not present in existing — nothing to preserve from, row flows untouched."""
    old: list[dict] = []
    new = [_row(benchmark="libero", model="new", model_paper=None, display_name=None)]
    preserve_on_null(new, old)
    assert new[0]["model_paper"] is None
    assert new[0]["display_name"] is None


def test_only_matching_key_preserves():
    """Match is on (benchmark, model). Different benchmark or model → no preserve."""
    old = [
        _row(benchmark="libero", model="foo", model_paper="https://arxiv.org/abs/2024.00001"),
        _row(benchmark="calvin", model="bar", model_paper="https://arxiv.org/abs/2024.00002"),
    ]
    new = [
        _row(benchmark="libero", model="bar", model_paper=None),  # same model, different bench
        _row(benchmark="calvin", model="foo", model_paper=None),  # same bench, different model
    ]
    preserve_on_null(new, old)
    assert new[0]["model_paper"] is None
    assert new[1]["model_paper"] is None


def test_scores_not_preserved():
    """Score fields are NOT in the preserve list — the LLM's number is authoritative."""
    # Build from scratch so we can include score fields
    old = [{"benchmark": "libero", "model": "foo", "overall_score": 95.0, "suite_scores": {"a": 95.0}}]
    new = [{"benchmark": "libero", "model": "foo", "overall_score": None, "suite_scores": None}]
    preserve_on_null(new, old)
    # Both score fields remain null — they were not in PRESERVE_ON_NULL_FIELDS
    assert new[0]["overall_score"] is None
    assert new[0]["suite_scores"] is None


def test_all_preserve_fields_covered():
    """Every declared preserve field should actually be restored when null."""
    old_row = {
        "benchmark": "libero",
        "model": "foo",
        "model_paper": "https://arxiv.org/abs/2024.00001",
        "display_name": "Foo Method",
        "params": "7B",
        "notes": "curator-added context",
        "weight_type": "finetuned",
    }
    new_row = {"benchmark": "libero", "model": "foo"}
    for f in PRESERVE_ON_NULL_FIELDS:
        new_row[f] = None
    preserve_on_null([new_row], [old_row])
    for f in PRESERVE_ON_NULL_FIELDS:
        assert new_row[f] == old_row[f], f"field {f} not restored"
