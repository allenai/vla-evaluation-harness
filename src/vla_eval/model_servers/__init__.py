"""Model server base classes, utilities, and implementations."""

import os
import sys

from vla_eval.model_servers.base import Episode, ModelServer, Row, Rows
from vla_eval.model_servers.daemons import DaemonModelServer, EpisodeContext, EpisodeDaemons
from vla_eval.model_servers.engine import PredictionEngine
from vla_eval.model_servers.predict import PredictModelServer

# `uv run <server>.py` puts this dir on sys.path, where siblings (lerobot.py, ...) shadow the
# same-named packages that model code imports; drop it and its subdirs before any model import.
_here = os.path.realpath(os.path.dirname(__file__)) + os.sep
sys.path[:] = [p for p in sys.path if not (os.path.realpath(p) + os.sep).startswith(_here)]

__all__ = [
    "DaemonModelServer",
    "Episode",
    "EpisodeContext",
    "EpisodeDaemons",
    "ModelServer",
    "PredictModelServer",
    "PredictionEngine",
    "Row",
    "Rows",
]
