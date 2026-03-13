"""Import string resolution for benchmarks and model servers.

Uses lazyregistry's ImportString to resolve "module.path:ClassName" strings
from config files into actual Python classes.
"""

from __future__ import annotations

from typing import Any

from lazyregistry import ImportString


def resolve_import_string(import_path: str) -> Any:
    """Resolve a "module.path:ClassName" string to the actual object.

    >>> cls = resolve_import_string("vla_eval.benchmarks.base:Benchmark")
    >>> cls.__name__
    'Benchmark'
    """
    return ImportString(import_path).load()
