"""Architecture/seed stability check, specific to neural discovery methods.

A single trained neural network is one sample of a random process, not a
stable fact about the data. Passing downstream statistical validation on
ONE trained model's output is necessary but not sufficient: the underlying
non-linear structure itself has to reappear across independent training
runs (>= 3 seeds/splits) before it's trusted (skill rule 2).

The discovery method (autoencoder.py) does the actual multi-seed retraining
and cross-seed matching, and attaches the raw measured result to every
candidate it emits, in provenance["stability"]:
    {
        "n_seeds_total": int,           # other seeds compared against
        "n_seeds_matched": int,         # how many exceeded the match threshold
        "matched_seed_fractions": [...],# best |correlation| against each other seed
        "min_abs_correlation_across_seeds": float,
    }
Every unit becomes a Candidate regardless of match outcome (skill rule 1:
log every candidate, not just the best-looking one). This module only
turns that raw measurement into a calibration band and a pass/fail, the
same way bootstrap.half_split_consistency reports a rate and lets the gate
interpret it, rather than the discovery method silently deciding pass/fail
via a hard threshold before the candidate exists.
"""

from __future__ import annotations

from dataclasses import dataclass

from discovery_engine.campaign.candidate import Candidate
from discovery_engine.validation.bootstrap import calibration_band

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
    stability_fraction: float | None = None
    calibration: str | None = None
    passed: bool = True


def stability_check(candidate: Candidate) -> StabilityResult:
    if not requires_stability_check(candidate.method):
        return StabilityResult(
            applicable=False,
            method=candidate.method,
            reason="deterministic/classical method; seed/architecture retraining stability does not apply",
        )

    info = candidate.provenance.get("stability")
    if info is None:
        raise ValueError(
            f"neural candidate {candidate.id!r} (method={candidate.method!r}) is missing "
            "provenance['stability']; the discovery method must attach multi-seed retrain "
            "results for every candidate it emits (skill rule 2), it cannot be computed "
            "after the fact from the candidate alone"
        )

    n_total = info["n_seeds_total"]
    n_matched = info["n_seeds_matched"]
    fraction = n_matched / n_total if n_total else 0.0
    calibration = calibration_band(fraction)

    return StabilityResult(
        applicable=True,
        method=candidate.method,
        reason=(
            f"matched {n_matched}/{n_total} other seeds "
            f"(min |r| across matches={info['min_abs_correlation_across_seeds']:.2f}) -> {calibration}"
        ),
        stability_fraction=fraction,
        calibration=calibration,
        passed=calibration != "noise",
    )
