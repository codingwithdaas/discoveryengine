"""Two distinct reproducibility checks that must not be conflated:

- half_split_consistency: repeated random splitting WITHOUT replacement
  (>= 30 splits). This is the primary reproducibility metric per the
  project's methodology: does the direction of the association reproduce
  on an independent half of the cohort.
- bootstrap_confidence_interval: true bootstrap, resampling WITH
  replacement, for a confidence interval on the statistic itself. A
  separate, also-needed method; calling either one "bootstrap" when it's
  actually the other is exactly the mislabeling the project's methodology
  flags as a defect.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from discovery_engine.campaign.candidate import Candidate
from discovery_engine.outcomes import OutcomeSpec, OutcomeType, association_statistic

CALIBRATION_BANDS = (
    (0.5, "noise"),
    (0.7, "weak"),
    (0.9, "moderate"),
    (float("inf"), "robust"),
)


def _calibration_band(consistency: float) -> str:
    for upper, label in CALIBRATION_BANDS:
        if consistency < upper:
            return label
    return "robust"


@dataclass
class HalfSplitResult:
    n_splits: int
    n_usable_splits: int
    consistency_fraction: float
    calibration: str
    full_cohort_statistic: float


def half_split_consistency(
    candidate: Candidate,
    outcome: OutcomeSpec,
    n_splits: int = 30,
    seed: int = 0,
) -> HalfSplitResult:
    """Split-half reliability: each round, split the cohort into two
    DISJOINT random halves and compare their directions to EACH OTHER, not
    to the full-cohort statistic.

    Comparing a half against the full cohort's own statistic is a self-
    inclusion bug, not a stricter test: the full statistic is computed from
    data that includes that very half, so the two are not independent even
    under pure noise, and "consistency" against it is inflated well above
    chance regardless of whether the candidate is real. Two disjoint halves
    share no data, which is what makes ~50% the correct EXPECTED agreement
    rate for a true null, averaged over many independent (candidate,
    outcome) realizations.

    That expectation does NOT hold per fixed dataset. A single fixed pure-
    noise realization has its own fixed, if small, empirical correlation
    (a specific artifact of how that realization's values happen to pair
    up), and resampling that ONE dataset tends to reproduce ITS OWN fixed
    sign reliably; observed single-realization consistency under a true
    null has been seen anywhere from ~0.03 to 1.0 across seeds in this
    project's own testing (see test_bootstrap.py). Do not read a single
    candidate's consistency_fraction as calibrated against a clean 50%
    baseline; the CALIBRATION_BANDS are a heuristic read on THAT candidate's
    own reproducibility, not a per-dataset null-hypothesis guarantee.
    """
    if n_splits < 30:
        raise ValueError("half_split_consistency requires at least 30 splits per project methodology")

    common_ids = [sid for sid in outcome.sample_ids if sid in candidate.scores]
    full_stat = association_statistic(outcome, candidate.scores)

    rng = np.random.default_rng(seed)
    agree = 0
    usable = 0
    for _ in range(n_splits):
        shuffled = rng.permutation(common_ids)
        midpoint = len(shuffled) // 2
        half_a, half_b = list(shuffled[:midpoint]), list(shuffled[midpoint:])
        if len(half_a) < 3 or len(half_b) < 3:
            continue
        try:
            stat_a = association_statistic(
                outcome.subset(half_a), {sid: candidate.scores[sid] for sid in half_a}
            )
            stat_b = association_statistic(
                outcome.subset(half_b), {sid: candidate.scores[sid] for sid in half_b}
            )
        except ValueError:
            continue
        usable += 1
        if stat_a != 0 and stat_b != 0 and np.sign(stat_a) == np.sign(stat_b):
            agree += 1

    consistency = agree / usable if usable else 0.0
    return HalfSplitResult(
        n_splits=n_splits,
        n_usable_splits=usable,
        consistency_fraction=consistency,
        calibration=_calibration_band(consistency),
        full_cohort_statistic=full_stat,
    )


@dataclass
class BootstrapCIResult:
    n_bootstrap: int
    point_estimate: float
    ci_low: float
    ci_high: float
    ci_level: float


def bootstrap_confidence_interval(
    candidate: Candidate,
    outcome: OutcomeSpec,
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
    seed: int = 0,
) -> BootstrapCIResult:
    common_ids = [sid for sid in outcome.sample_ids if sid in candidate.scores]
    point_estimate = association_statistic(outcome, candidate.scores)

    idx = {sid: i for i, sid in enumerate(outcome.sample_ids)}
    n = len(common_ids)
    rng = np.random.default_rng(seed)

    boot_stats = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        drawn = rng.choice(common_ids, size=n, replace=True)
        # Resampling WITH replacement means an id can appear more than once;
        # OutcomeSpec.subset assumes unique sample ids, so we build the
        # resampled OutcomeSpec directly here with synthetic unique ids
        # rather than routing duplicate ids back through .subset().
        unique_ids = [f"{sid}__{i}" for i, sid in enumerate(drawn)]
        positions = [idx[sid] for sid in drawn]
        resampled_outcome = OutcomeSpec(outcome.type, unique_ids, outcome.values[positions])
        resampled_scores = {uid: candidate.scores[sid] for uid, sid in zip(unique_ids, drawn)}
        try:
            boot_stats[b] = association_statistic(resampled_outcome, resampled_scores)
        except ValueError:
            boot_stats[b] = np.nan

    boot_stats = boot_stats[~np.isnan(boot_stats)]
    alpha = 1 - ci_level
    ci_low, ci_high = np.percentile(boot_stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])

    return BootstrapCIResult(
        n_bootstrap=n_bootstrap,
        point_estimate=point_estimate,
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        ci_level=ci_level,
    )
