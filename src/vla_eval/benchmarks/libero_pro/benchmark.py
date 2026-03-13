"""LIBERO-Pro benchmark implementation.

LIBERO-Pro is a perturbation-based extension of the standard LIBERO benchmark.
It registers additional suites that apply systematic perturbations to the
original LIBERO tasks, enabling robustness evaluation along five axes:

    - **swap** (position): Object positions are swapped relative to the
      original task layout.
    - **object**: Target or distractor objects are replaced with novel ones.
    - **lan** (language): Task instructions are paraphrased while preserving
      semantics.
    - **task**: The task goal is redesigned (different success condition).
    - **env** (environment): The surrounding environment is replaced.

Each perturbation suite is named ``{base_suite}_{perturbation}``, e.g.
``libero_spatial_swap``.  The observation and action spaces are identical to
standard LIBERO (7-D actions, 256×256 images), and the same per-suite
``max_steps`` limits apply.
"""

from __future__ import annotations

from typing import Any

from vla_eval.benchmarks.libero.benchmark import MAX_STEP_MAPPING, LIBEROBenchmark

# Canonical perturbation names and accepted aliases.
_PERTURBATION_ALIASES: dict[str, str] = {
    "swap": "swap",
    "position": "swap",
    "object": "object",
    "lan": "lan",
    "language": "lan",
    "task": "task",
    "env": "env",
    "environment": "env",
}


class LIBEROProBenchmark(LIBEROBenchmark):
    """LIBERO-Pro perturbation benchmark.

    Thin wrapper around :class:`LIBEROBenchmark` that constructs the
    perturbation-specific suite name (e.g. ``libero_spatial_swap``) and
    delegates all environment logic to the parent class.

    The LIBERO-Pro library registers its suites under the same
    ``libero.libero.benchmark`` registry, so no import changes are needed —
    only the suite name differs.

    Supports two usage patterns:

    1. **Direct suite name** — pass the full registered suite name via
       ``suite`` with ``perturbation=None`` (default).  Works for any suite
       that LIBERO-Pro registers, e.g. ``libero_spatial_with_mug``.

    2. **Base + perturbation** — pass a base suite (``libero_spatial``) plus
       a perturbation type (``swap``).  The class concatenates them into
       ``libero_spatial_swap``.  Note: perturbation suites require
       pre-generated BDDL & init-state files (see ``perturbation.py``).

    Args:
        suite: LIBERO-Pro suite name.  Either a full name (e.g.
            ``"libero_spatial_with_mug"``) or a base suite when combined
            with *perturbation*.
        perturbation: Perturbation type — one of ``"swap"`` / ``"position"``,
            ``"object"``, ``"lan"`` / ``"language"``, ``"task"``,
            ``"env"`` / ``"environment"``, or ``None`` / ``"none"`` to use
            *suite* as-is.
        seed: Random seed for environment initialization.
        num_steps_wait: Dummy action steps at episode start (default 10).
        send_wrist_image: Include wrist camera image in observations.
        send_state: Include proprioceptive state.
    """

    def __init__(
        self,
        suite: str = "libero_spatial_with_mug",
        perturbation: str | None = None,
        seed: int = 7,
        num_steps_wait: int = 10,
        send_wrist_image: bool = False,
        send_state: bool = False,
    ) -> None:
        self._perturbation = self._resolve_perturbation(perturbation)

        # Build the full suite name for LIBERO-Pro (e.g. "libero_spatial_swap")
        if self._perturbation:
            full_suite = f"{suite}_{self._perturbation}"
        else:
            full_suite = suite

        # Infer the base suite for max_steps lookup.
        self._base_suite = self._infer_base_suite(suite)

        super().__init__(
            suite=full_suite,
            seed=seed,
            num_steps_wait=num_steps_wait,
            send_wrist_image=send_wrist_image,
            send_state=send_state,
        )

    @staticmethod
    def _resolve_perturbation(perturbation: str | None) -> str:
        """Normalise perturbation name, returning canonical suffix or ``""``."""
        if perturbation is None or perturbation.lower() == "none":
            return ""
        key = perturbation.lower().strip()
        if key not in _PERTURBATION_ALIASES:
            valid = sorted(set(_PERTURBATION_ALIASES.values()))
            raise ValueError(
                f"Unknown perturbation {perturbation!r}. "
                f"Valid values: {valid} (or aliases {list(_PERTURBATION_ALIASES)})"
            )
        return _PERTURBATION_ALIASES[key]

    @staticmethod
    def _infer_base_suite(suite: str) -> str:
        """Find the matching base LIBERO suite for *max_steps* lookup."""
        for base in MAX_STEP_MAPPING:
            if suite == base or suite.startswith(f"{base}_"):
                return base
        return suite

    def get_metadata(self) -> dict[str, Any]:
        """Return metadata using the *base* suite's max_steps."""
        return {
            "max_steps": MAX_STEP_MAPPING.get(self._base_suite, 300),
            "suite": self.suite,
            "base_suite": self._base_suite,
            "perturbation": self._perturbation or "none",
        }
