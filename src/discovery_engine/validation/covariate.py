"""Grade/covariate-independence checks.

Two complementary methods, both required, neither a substitute for the
other:
  - covariate_stratified_test: recompute the association within each level
    of a clinical covariate held constant, so a real effect should survive
    stratification even though power drops in each stratum.
  - freedman_lane_test: a formal covariate-adjusted permutation test (see
    Freedman & Lane 1983) that tests whether the candidate adds anything to
    the outcome once the covariates' own contribution is regressed out.

freedman_lane_test is implemented for CONTINUOUS and BINARY outcomes via
OLS residualization (for BINARY this is the standard "linear probability
model" simplification, not a logistic GLM; documented here rather than
silently treated as exact). SURVIVAL is not yet supported by this specific
method and raises NotImplementedError rather than silently returning a
wrong answer; use covariate_stratified_test for survival outcomes instead,
which works for all three outcome types.

freedman_lane_test pre-filters to covariate-complete patients before the
linear algebra solve, the same way covariate_stratified_test already does
implicitly (pandas groupby silently drops NaN groups). This was NOT true
of an earlier version: a NaN in the covariate design matrix made
np.linalg.lstsq raise a generic LinAlgError ("SVD did not converge"),
confirmed directly, not assumed -- real, not a rare edge case, since real
clinical covariates (e.g. GRADE on the first TCGA-BLCA cohort) commonly
have a handful of missing values. Fails loudly with a specific message
naming the actual dropped count if filtering leaves too few patients to
fit the design at all, reusing the same "5 + k" minimum-sample convention
validation.multivariate already uses for its own OLS design (there,
k = number of comparator signatures; here, k = number of covariate
columns) rather than inventing a new threshold.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from discovery_engine.campaign.candidate import Candidate
from discovery_engine.data.cohort import CohortData
from discovery_engine.outcomes import OutcomeSpec, OutcomeType, permutation_test

logger = logging.getLogger(__name__)


@dataclass
class StratumResult:
    covariate_value: object
    n_samples: int
    observed_statistic: float
    p_value: float


@dataclass
class CovariateStratifiedResult:
    covariate_name: str
    full_cohort_statistic: float
    full_cohort_p_value: float
    strata: list[StratumResult]


_MIN_STRATUM_SIZE = 6


def covariate_stratified_test(
    candidate: Candidate,
    data: CohortData,
    outcome: OutcomeSpec,
    covariate_col: str,
    n_permutations: int = 500,
    seed: int = 0,
) -> CovariateStratifiedResult:
    if data.covariates is None or covariate_col not in data.covariates.columns:
        raise ValueError(f"covariate {covariate_col!r} not found in cohort covariates")

    full_stat, full_p = permutation_test(outcome, candidate.scores, n_permutations, seed)

    strata: list[StratumResult] = []
    for level, group in data.covariates.groupby(covariate_col):
        level_ids = [sid for sid in group.index if sid in candidate.scores and sid in outcome.sample_ids]
        if len(level_ids) < _MIN_STRATUM_SIZE:
            strata.append(StratumResult(level, len(level_ids), float("nan"), float("nan")))
            continue
        sub_outcome = outcome.subset(level_ids)
        sub_scores = {sid: candidate.scores[sid] for sid in level_ids}
        stat, p = permutation_test(sub_outcome, sub_scores, n_permutations, seed)
        strata.append(StratumResult(level, len(level_ids), stat, p))

    return CovariateStratifiedResult(covariate_col, full_stat, full_p, strata)


@dataclass
class FreedmanLaneResult:
    covariates_used: list[str]
    observed_partial_statistic: float
    p_value: float
    n_permutations: int


def freedman_lane_test(
    candidate: Candidate,
    data: CohortData,
    outcome: OutcomeSpec,
    covariate_cols: list[str],
    n_permutations: int = 1000,
    seed: int = 0,
) -> FreedmanLaneResult:
    if outcome.type is OutcomeType.SURVIVAL:
        raise NotImplementedError(
            "freedman_lane_test covers CONTINUOUS and BINARY outcomes via OLS residualization; "
            "SURVIVAL is not yet implemented here. Use covariate_stratified_test for survival "
            "outcomes, which is covariate-adjusted via stratification and works for all outcome types."
        )
    if data.covariates is None:
        raise ValueError("cohort.covariates required for freedman_lane_test")

    common_ids = [sid for sid in outcome.sample_ids if sid in candidate.scores]
    complete = data.covariates.loc[common_ids, covariate_cols].notna().all(axis=1)
    dropped = [sid for sid, ok in zip(common_ids, complete) if not ok]
    common_ids = [sid for sid, ok in zip(common_ids, complete) if ok]
    if dropped:
        logger.info(
            "freedman_lane_test: dropping %d/%d patient(s) with missing %s: %s",
            len(dropped), len(dropped) + len(common_ids), covariate_cols, dropped[:10],
        )
    min_size = 5 + len(covariate_cols)
    if len(common_ids) < min_size:
        raise ValueError(
            f"only {len(common_ids)} patients have complete {covariate_cols} coverage "
            f"(dropped {len(dropped)} with missing values); need at least {min_size} to fit "
            "Freedman-Lane's design (5 + number of covariate columns, matching "
            "validation.multivariate's identical convention for its own OLS design)"
        )

    sub_outcome = outcome.subset(common_ids)
    y = sub_outcome.values.astype(float)
    x = np.array([candidate.scores[sid] for sid in common_ids])
    cov = data.covariates.loc[common_ids, covariate_cols].to_numpy(dtype=float)
    z = np.column_stack([np.ones(len(common_ids)), cov])

    observed, p_value = freedman_lane_core(x, y, z, n_permutations, seed)
    return FreedmanLaneResult(list(covariate_cols), observed, p_value, n_permutations)


def freedman_lane_core(
    x: np.ndarray,
    y: np.ndarray,
    z_with_intercept: np.ndarray,
    n_permutations: int,
    seed: int,
) -> tuple[float, float]:
    """The residualize-permute-refit engine behind freedman_lane_test,
    exposed so validation.multivariate can reuse it with a different
    adjustment matrix (other accepted signatures' scores instead of
    clinical covariates) rather than duplicating the permutation loop.
    """
    beta_reduced, *_ = np.linalg.lstsq(z_with_intercept, y, rcond=None)
    fitted = z_with_intercept @ beta_reduced
    residuals = y - fitted

    observed = _partial_coefficient(x, y, z_with_intercept)

    rng = np.random.default_rng(seed)
    null = np.empty(n_permutations)
    for b in range(n_permutations):
        permuted_residuals = rng.permutation(residuals)
        y_star = fitted + permuted_residuals
        null[b] = _partial_coefficient(x, y_star, z_with_intercept)

    p_value = (1 + np.sum(np.abs(null) >= np.abs(observed))) / (n_permutations + 1)
    return float(observed), float(p_value)


def _partial_coefficient(x: np.ndarray, y: np.ndarray, z_with_intercept: np.ndarray) -> float:
    design = np.column_stack([x, z_with_intercept])
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    return float(beta[0])
