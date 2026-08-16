"""Architecture/seed stability check, specific to neural discovery methods.

A single trained neural network is one sample of a random process, not a
stable fact about the data. Passing downstream statistical validation on
ONE trained model's output is necessary but not sufficient: the underlying
non-linear structure itself has to reappear across independent training
runs (>= 3 seeds/splits) before it's trusted.

Phase 2 only has a deterministic classical method (hierarchical
clustering), which has no seed to retrain across, so this module currently
only implements the applicability gate the gate.py orchestrator needs to
know when to skip this check versus run it. The actual multi-seed retrain-
and-compare logic is deferred to Phase 3, where the first neural method
(autoencoder) exists to validate it against; NotImplementedError is raised
rather than silently no-opping if it's called on a method that needs it.
"""

from __future__ import annotations

from dataclasses import dataclass

NEURAL_METHODS: set[str] = {
    "autoencoder",
    "variational_autoencoder",
    "deepsurv",
    "attention",
}


def requires_stability_check(method: str) -> bool:
    return method in NEURAL_METHODS


@dataclass
class StabilityResult:
    applicable: bool
    method: str
    reason: str


def stability_check(method: str) -> StabilityResult:
    if not requires_stability_check(method):
        return StabilityResult(
            applicable=False,
            method=method,
            reason="deterministic/classical method; seed/architecture retraining stability does not apply",
        )
    raise NotImplementedError(
        f"seed/architecture stability check for neural method {method!r} is not implemented yet "
        "(lands in Phase 3 with the first neural discovery method); a neural candidate must not be "
        "reported as validated without it"
    )
