"""PredictModelServer: override ``predict(obs, row)`` (one episode's observation), or ``predict_batch(obs, rows)``
(column batch, a row per episode) when batched inference beats per-row; the default calls ``predict`` per row."""

from __future__ import annotations

from anyio.abc import TaskGroup

from vla_eval.types import Action, Batch, EpisodeResult, Observation
from vla_eval.model_servers.base import ModelServer, Row, Rows
from vla_eval.model_servers.engine import PredictionEngine


class PredictModelServer(ModelServer):
    """Args:
    chunk_size: Actions per inference; the rest of each ``[T, D]`` chunk is served from a per-episode buffer.
    max_batch_size: Most rows per ``predict_batch`` call (``None``: no limit); raise it to batch episodes.
    max_wait_time: Seconds to wait for other connections' requests before inferring a partial batch.
    max_rows: Most concurrent episodes (``None``: no limit); 1 for a model whose state is not per episode.

    The hooks below run on one model thread, in arrival order; ``__init__`` runs on the caller's thread, so
    create thread-bound state (CUDA graphs, JAX) in the first ``reset_episodes`` / ``predict``.
    """

    assumes_whole_chunks: bool = False
    """``predict`` assumes the first ``chunk_size`` actions of its previous chunk all ran (e.g. it tracks the last
    commanded pose).  ``DelayedChunkServer``, which skips a chunk's first actions, refuses such a server."""

    def __init__(
        self,
        *,
        chunk_size: int | None = None,
        max_batch_size: int | None = 1,
        max_wait_time: float = 0.01,
        max_rows: int | None = None,
    ) -> None:
        super().__init__()
        self.max_rows = max_rows
        batched = type(self).predict_batch is not PredictModelServer.predict_batch
        self._engine = PredictionEngine(
            self,
            predict=self.predict,
            predict_batch=self.predict_batch if batched else None,
            reset=self.reset_episodes,
            finish=self.finish_episodes,
            chunk_size=chunk_size,
            max_batch_size=max_batch_size,
            max_wait_time=max_wait_time,
        )

    # The engine's settings (GET /config sets max_batch_size / max_wait_time).
    @property
    def chunk_size(self) -> int | None:
        return self._engine.chunk_size

    @chunk_size.setter
    def chunk_size(self, value: int | None) -> None:
        self._engine.chunk_size = value

    @property
    def max_batch_size(self) -> int | None:
        return self._engine.max_batch_size

    @max_batch_size.setter
    def max_batch_size(self, value: int | None) -> None:
        self._engine.max_batch_size = value

    @property
    def max_wait_time(self) -> float:
        return self._engine.max_wait_time

    @max_wait_time.setter
    def max_wait_time(self, value: float) -> None:
        self._engine.max_wait_time = value

    # ------------------------------------------------------------------
    # Override these
    # ------------------------------------------------------------------

    def reset_episodes(self, rows: Rows) -> dict[int, Exception] | None:
        """Reset the state of ``rows`` (``episode.config``: the ``EPISODE_START`` payload).  Blocking.
        Return ``{position: exception}`` for starts that failed, or raise to fail them all."""
        return None

    def finish_episodes(self, rows: Rows, results: list[EpisodeResult]) -> None:
        """Episodes in ``rows`` ended (``results[i]``: the ``EPISODE_END`` payload).  Blocking."""

    def predict(self, obs: Observation, row: Row) -> Action:
        """One observation of the episode in ``row``.  Blocking."""
        raise NotImplementedError(f"{type(self).__name__} must override predict() or predict_batch()")

    def predict_batch(self, obs: Batch, rows: Rows) -> Batch:
        """Observation columns of ``rows`` to action columns, a row each.  Blocking.  Not overridden, the engine
        runs ``predict`` per row."""
        raise NotImplementedError(f"{type(self).__name__} does not override predict_batch()")

    # ------------------------------------------------------------------
    # ModelServer callbacks
    # ------------------------------------------------------------------

    async def on_serve_start(self, tg: TaskGroup) -> None:
        await super().on_serve_start(tg)
        await self._engine.on_serve_start(tg)

    async def on_episode_start(self, rows: Rows) -> dict[int, Exception] | None:
        return await self._engine.on_episode_start(rows)

    async def on_observation(self, obs: Batch, rows: Rows) -> None:
        await self._engine.on_observation(obs, rows)

    async def on_episode_end(self, rows: Rows, results: list[EpisodeResult]) -> None:
        await self._engine.on_episode_end(rows, results)
