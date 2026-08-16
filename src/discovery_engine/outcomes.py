"""Outcome types and their association statistics / permutation rules.

Every discovery method and every campaign-correction step must go through
this module to compute "how strongly does this candidate associate with the
outcome" and "what does a permuted version of the outcome look like". Keeping
both in one place means every method is tested against the outcome the same
way, and survival permutation (which must shuffle time and event jointly)
can't accidentally be done wrong in one call site and right in another.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from scipy.stats import pointbiserialr, spearmanr


class OutcomeType(Enum):
    BINARY = "binary"
    CONTINUOUS = "continuous"
    SURVIVAL = "survival"


@dataclass(frozen=True)
class OutcomeSpec:
    """Outcome values for a cohort, indexed by sample id.

    For BINARY/CONTINUOUS, `values` holds one number per sample.
    For SURVIVAL, `values` holds (time, event) pairs per sample; `event` is
    1 = event observed, 0 = censored.
    """

    type: OutcomeType
    sample_ids: list[str]
    values: np.ndarray  # shape (n,) for binary/continuous, (n, 2) for survival

    def __post_init__(self) -> None:
        n = len(self.sample_ids)
        expected_shape = (n, 2) if self.type is OutcomeType.SURVIVAL else (n,)
        if self.values.shape != expected_shape:
            raise ValueError(
                f"OutcomeSpec values shape {self.values.shape} does not match "
                f"expected {expected_shape} for {self.type} with {n} samples"
            )

    def subset(self, sample_ids: list[str]) -> "OutcomeSpec":
        idx = {sid: i for i, sid in enumerate(self.sample_ids)}
        missing = [sid for sid in sample_ids if sid not in idx]
        if missing:
            raise KeyError(f"sample ids not in outcome: {missing[:5]}...")
        positions = [idx[sid] for sid in sample_ids]
        return OutcomeSpec(self.type, list(sample_ids), self.values[positions])

    def permute(self, rng: np.random.Generator) -> "OutcomeSpec":
        """Shuffle outcome values across samples, keeping sample_ids fixed.

        For SURVIVAL this shuffles (time, event) pairs as a unit, never
        decoupling time from event.
        """
        perm = rng.permutation(len(self.sample_ids))
        return OutcomeSpec(self.type, self.sample_ids, self.values[perm])


def association_statistic(outcome: OutcomeSpec, scores: dict[str, float]) -> float:
    """Signed association statistic between candidate scores and outcome.

    Larger |statistic| = stronger association, in all three cases, so the
    same studentization/maxT logic in campaign.correction can treat them
    uniformly:
      BINARY     -> point-biserial correlation coefficient
      CONTINUOUS -> Spearman rho
      SURVIVAL   -> Cox PH Wald z-statistic for the score as a single covariate
    """
    common_ids = [sid for sid in outcome.sample_ids if sid in scores]
    if len(common_ids) < 3:
        raise ValueError("need at least 3 overlapping samples to compute an association statistic")
    sub_outcome = outcome.subset(common_ids)
    x = np.array([scores[sid] for sid in common_ids], dtype=float)

    if sub_outcome.type is OutcomeType.BINARY:
        stat, _ = pointbiserialr(sub_outcome.values.astype(float), x)
        return float(stat)

    if sub_outcome.type is OutcomeType.CONTINUOUS:
        stat, _ = spearmanr(sub_outcome.values, x)
        return float(stat)

    if sub_outcome.type is OutcomeType.SURVIVAL:
        return _cox_wald_z(sub_outcome, x)

    raise AssertionError(f"unhandled outcome type {sub_outcome.type}")


def permutation_test(
    outcome: OutcomeSpec,
    scores: dict[str, float],
    n_permutations: int = 1000,
    seed: int = 0,
) -> tuple[float, float]:
    """Observed association_statistic plus a two-sided permutation p-value
    against it, permuting the outcome n_permutations times. Shared by every
    validation-gate check that needs "is this association more extreme than
    chance", so the permutation mechanics (in particular, survival's
    time+event pairs staying joined) live in exactly one place.
    """
    rng = np.random.default_rng(seed)
    observed = association_statistic(outcome, scores)
    null = np.empty(n_permutations)
    for b in range(n_permutations):
        null[b] = association_statistic(outcome.permute(rng), scores)
    p_value = (1 + np.sum(np.abs(null) >= np.abs(observed))) / (n_permutations + 1)
    return observed, float(p_value)


def _cox_wald_z(outcome: OutcomeSpec, x: np.ndarray) -> float:
    time = outcome.values[:, 0]
    event = outcome.values[:, 1]
    if np.std(x) == 0:
        return 0.0
    df = pd.DataFrame({"time": time, "event": event, "score": x})
    try:
        cph = CoxPHFitter()
        cph.fit(df, duration_col="time", event_col="event")
        coef = cph.params_["score"]
        se = cph.standard_errors_["score"]
        if se == 0 or not np.isfinite(se):
            return 0.0
        return float(coef / se)
    except Exception:
        # Non-convergence (e.g. under a degenerate permutation) is a valid
        # null draw of "no detectable association", not a pipeline error.
        return 0.0
