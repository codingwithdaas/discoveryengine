"""Campaign-level multiple-testing correction.

The core rule this whole project is built around: the number of "shots on
goal" is every candidate ever generated across the full campaign, not the
count within any one method. A within-method q-value is not sufficient.

Implementation: studentized maxT (Westfall & Young 1993). This is needed,
not a plain maxT over raw statistics, because candidates from different
methods carry different statistic types (a Spearman rho from an autoencoder
latent, a Cox Wald z from a survival forest score) that are not directly
comparable in scale. Studentizing each candidate's statistic against its own
permutation-derived null mean/sd puts every candidate on a common scale
before taking the pool-wide max, which is what makes the max valid across a
heterogeneous candidate pool.

For outcome-blind candidates (the large majority: clustering, NMF,
autoencoder latents, community modules), recomputing the statistic against a
permuted outcome is exact, since the candidate's definition never depended
on the outcome to begin with. For outcome-informed candidates (RSF risk
scores, DeepSurv risk scores, attention weights, explicit interaction
search), `Candidate.refit` is called each permutation to actually rerun the
outcome-dependent step, i.e. a full pipeline rerun under permutation, not a
shortcut.

(Earlier draft of this docstring said "RSF importances" here. RSF's
candidates are the forest's aggregate predicted risk score per
hyperparameter config, not per-feature importances treated as separate
candidates -- see discovery/rsf.py's module docstring for why: using
importance-ranking to nominate which features become candidates would
need the *selection* step re-run under every permutation too, not just a
refit of one fixed feature's association, which `Candidate.refit` isn't
shaped for.)

Every refit call is handed a `np.random.Generator` derived deterministically
from this module's own seeded rng, and refit is contractually required to
source all of its internal randomness (model init, split shuffles, dropout)
from it rather than its own unseeded RNG. That's what makes a full campaign
run, including every retrained outcome-informed candidate, bit-reproducible
given the same seed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from discovery_engine.campaign.candidate import Candidate
from discovery_engine.data.cohort import CohortData
from discovery_engine.outcomes import OutcomeSpec, association_statistic

_MIN_SD = 1e-8


class RefitDidNotChangeError(RuntimeError):
    """Raised when an outcome-informed candidate's refit() returns scores
    identical to the original real-outcome candidate's scores under a
    permuted outcome.

    This means the permutation is being silently ignored, i.e. refit() is
    returning cached/stale values instead of genuinely rerunning the
    outcome-dependent discovery step, which would make campaign correction
    under-correct without any visible sign that it happened. Fails loudly,
    with the candidate id and permutation index, so a break in a long real
    campaign (RSF/DeepSurv refits across 1000 permutations) doesn't require
    re-running the whole thing to locate.
    """


@dataclass
class CorrectionResult:
    candidate_id: str
    method: str
    observed_statistic: float
    studentized_statistic: float
    campaign_p_value: float
    n_permutations: int


def run_campaign_correction(
    candidates: list[Candidate],
    data: CohortData,
    outcome: OutcomeSpec,
    n_permutations: int = 1000,
    seed: int = 0,
) -> dict[str, CorrectionResult]:
    if not candidates:
        return {}

    rng = np.random.default_rng(seed)
    n = len(candidates)

    observed = np.array([association_statistic(outcome, c.scores) for c in candidates])
    null_stats = _null_statistics(candidates, data, outcome, n_permutations, rng)

    null_mean = null_stats.mean(axis=0)
    null_sd = np.maximum(null_stats.std(axis=0, ddof=1), _MIN_SD)

    studentized_null = (null_stats - null_mean) / null_sd
    pool_max_per_permutation = np.max(np.abs(studentized_null), axis=1)

    observed_studentized = (observed - null_mean) / null_sd

    results: dict[str, CorrectionResult] = {}
    for i, c in enumerate(candidates):
        exceed = np.sum(pool_max_per_permutation >= np.abs(observed_studentized[i]))
        p_value = (1 + exceed) / (n_permutations + 1)
        results[c.id] = CorrectionResult(
            candidate_id=c.id,
            method=c.method,
            observed_statistic=float(observed[i]),
            studentized_statistic=float(observed_studentized[i]),
            campaign_p_value=float(p_value),
            n_permutations=n_permutations,
        )
    return results


def naive_within_method_bh_fdr(
    candidates: list[Candidate],
    data: CohortData,
    outcome: OutcomeSpec,
    n_permutations: int = 1000,
    seed: int = 0,
) -> dict[str, float]:
    """The comparison baseline: permutation p-value computed against each
    candidate's OWN null only, then Benjamini-Hochberg FDR correction applied
    separately within each method's candidates. This is what campaign-level
    correction is meant to be more conservative than; it is not a valid
    substitute for it because it never lets one method's candidates compete
    against another method's in the same correction.
    """
    rng = np.random.default_rng(seed)
    n = len(candidates)
    observed = np.array([association_statistic(outcome, c.scores) for c in candidates])
    null_stats = _null_statistics(candidates, data, outcome, n_permutations, rng)

    raw_p = np.array(
        [(1 + np.sum(np.abs(null_stats[:, i]) >= np.abs(observed[i]))) / (n_permutations + 1) for i in range(n)]
    )

    q_values = np.empty(n)
    methods = {c.method for c in candidates}
    for method in methods:
        idx = [i for i, c in enumerate(candidates) if c.method == method]
        q_values[idx] = _bh_fdr(raw_p[idx])

    return {candidates[i].id: float(q_values[i]) for i in range(n)}


def _null_statistics(
    candidates: list[Candidate],
    data: CohortData,
    outcome: OutcomeSpec,
    n_permutations: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Shared permutation loop used by both correction functions. For each
    permutation, outcome-blind candidates are re-scored against the shuffled
    outcome directly (exact, since their definition never depended on
    outcome); outcome-informed candidates are rederived from scratch via
    refit(data, permuted_outcome), and the guard below raises immediately if
    that rerun suspiciously produced the exact same scores as the original
    real-outcome fit.
    """
    n = len(candidates)
    null_stats = np.empty((n_permutations, n))

    for b in range(n_permutations):
        permuted_outcome = outcome.permute(rng)
        for i, c in enumerate(candidates):
            if c.outcome_blind:
                null_stats[b, i] = association_statistic(permuted_outcome, c.scores)
            else:
                # Derived from the outer seeded rng, never left to refit() to
                # source its own randomness, so a fixed campaign seed gives
                # bit-identical results across reruns, including for every
                # retrained outcome-informed candidate.
                refit_rng = np.random.default_rng(rng.integers(0, 2**63 - 1))
                refit_candidate = c.refit(data, permuted_outcome, refit_rng)
                if refit_candidate.scores == c.scores:
                    raise RefitDidNotChangeError(
                        f"candidate {c.id!r} (method={c.method!r}) refit at permutation "
                        f"index {b}/{n_permutations} returned scores identical to the "
                        "original real-outcome candidate's scores; refit() must genuinely "
                        "rerun the outcome-dependent discovery step on the permuted "
                        "outcome, not return cached/stale values"
                    )
                null_stats[b, i] = association_statistic(permuted_outcome, refit_candidate.scores)

    return null_stats


def _bh_fdr(p_values: np.ndarray) -> np.ndarray:
    m = len(p_values)
    order = np.argsort(p_values)
    ranked = p_values[order] * m / (np.arange(m) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    q = np.empty(m)
    q[order] = np.clip(ranked, 0, 1)
    return q
