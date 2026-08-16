"""Median-split vs extreme-groups design crosscheck.

A median split can under-power a real effect that only shows up at the
extremes. An extreme-groups design can also surface a pure confound that a
median split was too diluted to detect. Both designs are run; when they
disagree on significance, and a suspected confound is supplied, the SAME
two designs are rerun with the confound standing in for the candidate (no
candidate involved at all) to see whether the confound alone reproduces
whichever design flagged significant, which is what tells you which design
was telling the truth.

Reuses outcomes.permutation_test by treating each design's group assignment
(0/1) as if it were a candidate's scores: testing "does outcome differ
between these two groups" is exactly testing association between a binary
grouping variable and the outcome, which permutation_test already handles
for all three outcome types (survival treats the 0/1 group as a Cox
covariate, which is precisely a two-group hazard comparison).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from discovery_engine.campaign.candidate import Candidate
from discovery_engine.outcomes import OutcomeSpec, permutation_test


@dataclass
class DesignResult:
    design: str
    n_group_low: int
    n_group_high: int
    observed_statistic: float
    p_value: float


@dataclass
class DesignCrosscheckResult:
    median_split: DesignResult
    extreme_groups: DesignResult
    disagreement: bool
    confound_median_split: DesignResult | None
    confound_extreme_groups: DesignResult | None


def median_split_vs_extreme_groups(
    candidate: Candidate,
    outcome: OutcomeSpec,
    extreme_fraction: float = 0.25,
    n_permutations: int = 500,
    seed: int = 0,
    alpha: float = 0.05,
    confound: dict[str, float] | None = None,
) -> DesignCrosscheckResult:
    median_result = _run_design(_median_split_labels(candidate.scores), outcome, "median_split", n_permutations, seed)
    extreme_result = _run_design(
        _extreme_group_labels(candidate.scores, extreme_fraction), outcome, "extreme_groups", n_permutations, seed
    )

    disagreement = (median_result.p_value < alpha) != (extreme_result.p_value < alpha)

    confound_median = confound_extreme = None
    if disagreement and confound is not None:
        confound_median = _run_design(
            _median_split_labels(confound), outcome, "median_split_confound_alone", n_permutations, seed
        )
        confound_extreme = _run_design(
            _extreme_group_labels(confound, extreme_fraction), outcome, "extreme_groups_confound_alone", n_permutations, seed
        )

    return DesignCrosscheckResult(median_result, extreme_result, disagreement, confound_median, confound_extreme)


def _median_split_labels(scores: dict[str, float]) -> dict[str, float]:
    median = float(np.median(list(scores.values())))
    return {sid: (1.0 if v > median else 0.0) for sid, v in scores.items()}


def _extreme_group_labels(scores: dict[str, float], fraction: float) -> dict[str, float]:
    items = sorted(scores.items(), key=lambda kv: kv[1])
    n = len(items)
    k = max(1, int(round(n * fraction)))
    labels: dict[str, float] = {}
    for sid, _ in items[:k]:
        labels[sid] = 0.0
    for sid, _ in items[-k:]:
        labels[sid] = 1.0
    return labels


def _run_design(
    labels: dict[str, float],
    outcome: OutcomeSpec,
    design_name: str,
    n_permutations: int,
    seed: int,
) -> DesignResult:
    common_ids = [sid for sid in outcome.sample_ids if sid in labels]
    sub_outcome = outcome.subset(common_ids)
    sub_labels = {sid: labels[sid] for sid in common_ids}
    stat, p = permutation_test(sub_outcome, sub_labels, n_permutations, seed)
    n_low = sum(1 for sid in common_ids if labels[sid] == 0.0)
    n_high = sum(1 for sid in common_ids if labels[sid] == 1.0)
    return DesignResult(design_name, n_low, n_high, stat, p)
