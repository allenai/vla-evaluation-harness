"""Evaluation harness for Vision-Language-Action models."""

from vla_eval._version import __version__, __version_tuple__


def __getattr__(name: str):  # lazy: keep ``import vla_eval`` cheap for model-server scripts
    if name in ("evaluate", "run", "serve_background", "ServerHandle"):
        from vla_eval import api

        return getattr(api, name)
    raise AttributeError(f"module 'vla_eval' has no attribute {name!r}")


__all__ = ["ServerHandle", "__version__", "__version_tuple__", "evaluate", "run", "serve_background"]
