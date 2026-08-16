"""Seed/algorithm-internal-randomness stability check.

Every discovery method with internal stochasticity (model init, train/val
split, algorithm-internal tie-breaking) falls into exactly one of three
categories with respect to this module. Stated explicitly here because the
category a method belongs to is a real, load-bearing distinction, not an
implementation detail -- "deterministic" and "stochastic but empirically
verified stable" are different claims resting on different evidence, and
conflating them (or leaving no place in the design for the third case) is
exactly the kind of gap this project's methodology exists to catch.

1. DETERMINISTIC: no internal randomness at all (clustering.py's
   scipy.cluster.hierarchy.linkage; no seed parameter exists because there
   is nothing to seed). Not in either set/dict below. discovery.base's
   OutcomeBlindDiscoveryMethod docstring states this is the only category
   its protocol assumes by default; anything stochastic must say so.

2. STABILITY_REQUIRED_METHODS: internal randomness that is NOT exempted --
   either because skill rule 2 mandates the check structurally regardless
   of empirical results (every method in here today is neural: a single
   trained network is one sample of a random process, not a stable fact
   about the data, full stop), or because a method WAS empirically checked
   and the check came back mixed/unstable, so it needs the identical
   per-candidate provenance and gate treatment even though it isn't
   neural. Nothing here is neural-specific in mechanism; the set is named
   for what's actually in it today, not for an assumption about why.
   Candidates from these methods MUST carry provenance["stability"]:
       {
           "n_seeds_total": int,           # other seeds compared against
           "n_seeds_matched": int,         # how many exceeded the match threshold
           "matched_seed_fractions": [...],# best |correlation| against each other seed
           "min_abs_correlation_across_seeds": float,
       }
   attached by the discovery method itself (see autoencoder.py, deepsurv.py,
   attention.py), one per candidate, every candidate regardless of match
   outcome (skill rule 1: log everything, not just the best-looking one).
   This module only turns that raw measurement into a calibration band and
   a pass/fail, the same way bootstrap.half_split_consistency reports a
   rate and lets the gate interpret it, never deciding pass/fail itself
   before the candidate exists.

3. EMPIRICALLY_EXEMPTED_METHODS: has internal randomness, was actually
   measured (not assumed) via match_units_across_seeds on a realistic
   config, including an adversarial condition chosen because the relevant
   literature flags it as the likely failure mode (small ensembles / no
   real signal for RSF; large, dense graphs for Louvain), and the
   cross-seed correlation cleared the match threshold comfortably in every
   condition tested. Mapped to a short citation of that evidence, not just
   a bare method name, so the claim stays checkable rather than becoming
   folklore. No provenance["stability"] required for these; the exemption
   IS the accountability mechanism, in place of a per-candidate check.

A new method with any internal randomness must be placed into (2) or (3)
explicitly -- there is no default. This is a manually-maintained
convention, not an auto-discovered one (matching how CandidateRegistry and
every other method-name-keyed set in this project already works); a method
that's stochastic and unaccounted-for in both is a bug to fix, not a gap
to silently ship.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from discovery_engine.campaign.candidate import Candidate
from discovery_engine.validation.bootstrap import calibration_band

STABILITY_REQUIRED_METHODS: set[str] = {
    "autoencoder",
    "variational_autoencoder",
    "deepsurv",
    "attention",
}

EMPIRICALLY_EXEMPTED_METHODS: dict[str, str] = {
    "random_survival_forest": (
        "match_units_across_seeds, 3 seeds, smallest-n_estimators sweep-grid config: "
        "cross-seed |r| 0.71-0.99 across small-forest (n_estimators=10) and no-signal "
        "(pure-noise cohort) configs, both comfortably above the 0.7 match threshold; "
        "see rsf.py module docstring for full numbers"
    ),
    "louvain": (
        "match_units_across_seeds on community-membership module scores, 3 seeds, across "
        "4 configs (small graph with a real planted module, small graph with no signal, a "
        "finer resolution, and a larger/denser 200-feature graph -- the last chosen because "
        "the literature flags large/dense graphs as Louvain's likely instability regime, not "
        "a repeat of favorable conditions): cross-seed |r| ~1.0 in every condition tested; "
        "see louvain.py module docstring for full numbers"
    ),
}


def requires_stability_check(method: str) -> bool:
    return method in STABILITY_REQUIRED_METHODS


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
        if candidate.method in EMPIRICALLY_EXEMPTED_METHODS:
            reason = f"empirically exempted: {EMPIRICALLY_EXEMPTED_METHODS[candidate.method]}"
        else:
            reason = "deterministic method; no internal randomness to check"
        return StabilityResult(applicable=False, method=candidate.method, reason=reason)

    info = candidate.provenance.get("stability")
    if info is None:
        raise ValueError(
            f"candidate {candidate.id!r} (method={candidate.method!r}) is in "
            "STABILITY_REQUIRED_METHODS but is missing provenance['stability']; the "
            "discovery method must attach multi-seed retrain results for every candidate "
            "it emits (skill rule 2), it cannot be computed after the fact from the "
            "candidate alone"
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


@dataclass
class TrainedRun:
    """One independent trained run's per-sample scores, for cross-seed
    stability matching. `latent_scores` is (n_samples, n_units): n_units is
    the autoencoder's latent_dim, or exactly 1 for single-output methods
    like DeepSurv (a risk score is a "latent space" of dimension 1, so the
    exact same matching logic below applies unchanged, not a special-cased
    pass/fail shortcut for the single-unit case).
    """

    latent_scores: np.ndarray


def match_units_across_seeds(runs: list[TrainedRun], threshold: float) -> list[dict]:
    """For every unit in runs[0] (the reference seed), find its best-matching
    unit, by |Pearson r| over the shared samples, in each OTHER run. Returns
    one raw measurement dict per reference unit, in unit order; does not
    decide pass/fail (that's stability_check's job, from the resulting
    Candidate's provenance, not this function's).

    Used in two ways: (1) embedded directly in the discover() of every
    STABILITY_REQUIRED_METHODS method (autoencoder.py, deepsurv.py,
    attention.py) to build every candidate's provenance["stability"], and
    (2) as a one-off empirical verification tool -- run manually against a
    realistic config before deciding whether a method belongs in
    EMPIRICALLY_EXEMPTED_METHODS instead (random_survival_forest, louvain),
    where the result is cited as evidence in that method's own module
    docstring rather than embedded in its production discover() call.
    Moved here from autoencoder.py once a second method needed the
    identical logic.
    """
    ref = runs[0].latent_scores
    n_units = ref.shape[1]
    other_runs = runs[1:]

    results = []
    for unit_idx in range(n_units):
        ref_col = ref[:, unit_idx]
        fractions = [
            max(
                (abs(_safe_corr(ref_col, other.latent_scores[:, j])) for j in range(other.latent_scores.shape[1])),
                default=0.0,
            )
            for other in other_runs
        ]
        n_total = len(other_runs)
        n_matched = sum(1 for f in fractions if f >= threshold)
        results.append(
            {
                "n_seeds_total": n_total,
                "n_seeds_matched": n_matched,
                "matched_seed_fractions": fractions,
                "min_abs_correlation_across_seeds": min(fractions) if fractions else 0.0,
            }
        )
    return results


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])
