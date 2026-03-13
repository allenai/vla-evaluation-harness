"""Tests for resolve_import_string."""

from __future__ import annotations

import pytest

from vla_eval.registry import resolve_import_string


def test_resolve_valid_import_string():
    cls = resolve_import_string("vla_eval.benchmarks.base:Benchmark")
    assert cls.__name__ == "Benchmark"


def test_resolve_invalid_module():
    with pytest.raises(Exception):
        resolve_import_string("nonexistent.module:SomeClass")


def test_resolve_invalid_attr():
    with pytest.raises(Exception):
        resolve_import_string("vla_eval.benchmarks.base:NonexistentClass")
