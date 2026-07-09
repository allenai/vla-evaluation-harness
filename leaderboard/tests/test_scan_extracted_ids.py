"""Unit tests for scan's extraction-id discovery.

``papers_reviewed`` is the scan pool intersected with the set of papers that
have an extraction record. Records live in two places: the authoritative packed
``data/extractions.json`` that ``extract.py`` writes, and the transient
``.cache/extractions/`` dir. Reading only one silently under-counts — extracting
a single paper into a fresh clone leaves a one-entry cache beside a full packed
file, which used to zero out coverage entirely.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import scan  # noqa: E402


def _setup(monkeypatch, tmp_path: Path, packed: list[dict] | None = None, cache_ids: list[str] | None = None) -> None:
    packed_path = tmp_path / "extractions.json"
    cache_dir = tmp_path / "cache"
    if packed is not None:
        packed_path.write_text(json.dumps(packed))
    if cache_ids is not None:
        cache_dir.mkdir()
        for aid in cache_ids:
            (cache_dir / f"{aid}.json").write_text("{}")
    monkeypatch.setattr(scan, "EXTRACTIONS_PACKED_PATH", packed_path)
    monkeypatch.setattr(scan, "EXTRACTIONS_CACHE_DIR", cache_dir)


def test_packed_only(monkeypatch, tmp_path):
    """No cache dir — ids come from the packed file."""
    _setup(monkeypatch, tmp_path, packed=[{"arxiv_id": "1111.1111"}, {"arxiv_id": "2222.2222"}])
    assert scan._extracted_arxiv_ids() == {"1111.1111", "2222.2222"}


def test_cache_only(monkeypatch, tmp_path):
    """No packed file — ids come from the cache dir."""
    _setup(monkeypatch, tmp_path, cache_ids=["3333.3333"])
    assert scan._extracted_arxiv_ids() == {"3333.3333"}


def test_sparse_cache_does_not_hide_packed(monkeypatch, tmp_path):
    """Regression: a one-entry cache used to shadow the full packed file, zeroing coverage."""
    _setup(
        monkeypatch,
        tmp_path,
        packed=[{"arxiv_id": "1111.1111"}, {"arxiv_id": "2222.2222"}],
        cache_ids=["9999.9999"],
    )
    assert scan._extracted_arxiv_ids() == {"1111.1111", "2222.2222", "9999.9999"}


def test_neither_source(monkeypatch, tmp_path):
    """Nothing extracted anywhere — empty set, no crash."""
    _setup(monkeypatch, tmp_path)
    assert scan._extracted_arxiv_ids() == set()


def test_malformed_packed_still_reads_cache(monkeypatch, tmp_path):
    """A corrupt packed file must not drop the cache's ids on the floor."""
    packed_path = tmp_path / "extractions.json"
    packed_path.write_text("{not json")
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    (cache_dir / "4444.4444.json").write_text("{}")
    monkeypatch.setattr(scan, "EXTRACTIONS_PACKED_PATH", packed_path)
    monkeypatch.setattr(scan, "EXTRACTIONS_CACHE_DIR", cache_dir)
    assert scan._extracted_arxiv_ids() == {"4444.4444"}


def test_packed_entries_without_arxiv_id_ignored(monkeypatch, tmp_path):
    """Records missing arxiv_id are skipped rather than raising."""
    _setup(monkeypatch, tmp_path, packed=[{"arxiv_id": "1111.1111"}, {"paper_hash": "x"}])
    assert scan._extracted_arxiv_ids() == {"1111.1111"}
