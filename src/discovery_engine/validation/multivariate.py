""""Adds value beyond X" test: a candidate and an already-accepted signature
can both be individually significant while one is fully redundant with the
other. This tests the candidate jointly against every already-accepted
signature in the registry, not just alone.

CONTINUOUS/BINARY outcomes reuse the same Freedman-Lane residualize-permute
engine as validation.covariate, with other signatures' scores standing in
for clinical covariates. SURVIVAL outcomes use a multivariable Cox model
(lifelines handles multiple covariates directly) and report the candidate's
Wald p-value from that joint fit; this is a Wald test, not a permutation
test, and is labeled as such (test_kind="cox_wald") rather than implied to
be the same kind of result as the other two outcome types.

Why survival is handled differently here, honestly: this is not purely a
compute-cost tradeoff. Freedman-Lane's residualize-permute-refit trick
depends on OLS residuals (y - fitted), which Cox's partial likelihood
doesn't give you in an equally clean, standard form; a permutation-based
multivariable Cox analog is a genuinely harder, less-standardized problem,
not just an expensive one. That difficulty is compounded by the fact that
lifelines hands back an asymptotic Wald p-value from a multivariable fit
for free, and outcomes._cox_wald_z already uses the same asymptotic
Wald-z-without-permutation pattern for the single-covariate case, with
permutation layered on separately (outcomes.permutation_test,
campaign.correction) rather than built into the Cox computation itself.
This choice follows that existing pattern rather than inventing a new one,
but the harder-problem point is the real reason a permutation analog isn't
here yet, not just convenience.

Compatibility with campaign-level correction: today this function's output
is NEVER fed into campaign.correction.run_campaign_correction. That
function only ever consumes Candidate.scores through
outcomes.association_statistic (a single-covariate statistic per
candidate); multivariate_adds_value is a validation.gate-only diagnostic
that feeds GateResult.passed/failure_reasons, structurally disconnected
from the correction pool. So there is no live bug today. If this result
were ever pooled into a corrected campaign in the future, the two
test_kind values would need different treatment: a
"freedman_lane_permutation" p_value is already permutation-derived and
close to poolable in spirit, but still isn't a raw statistic with its own
per-permutation trajectory the way correction.py's studentized maxT needs.
A "cox_wald" p_value is an asymptotic p-value, not a statistic with a
permutation null at all, and could not simply be dropped into that pool.
Either would need a Candidate.refit that reruns the full joint model under
each of correction.py's permuted outcomes and returns the resulting
coefficient/z from THAT permutation's fit (the Phase 1 refit contract
already supports this), or would need to stay excluded from pool-based
correction and be reported as its own separately-labeled, uncorrected
diagnostic, which is what happens today.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter

from discovery_engine.campaign.candidate import Candidate
from discovery_engine.outcomes import OutcomeSpec, OutcomeType
from discovery_engine.validation.covariate import freedman_lane_core


@dataclass
class MultivariateAddsValueResult:
    comparator_ids: list[str]
    test_kind: str  # "freedman_lane_permutation" or "cox_wald"
    candidate_coefficient: float
    p_value: float
    n_permutations: int | None


def multivariate_adds_value(
    candidate: Candidate,
    outcome: OutcomeSpec,
    other_signatures: list[Candidate],
    n_permutations: int = 1000,
    seed: int = 0,
) -> MultivariateAddsValueResult:
    common_ids = [
        sid
        for sid in outcome.sample_ids
        if sid in candidate.scores and all(sid in s.scores for s in other_signatures)
    ]
    if len(common_ids) < 5 + len(other_signatures):
        raise ValueError("not enough overlapping samples to fit a joint model with this many comparators")

    x = np.array([candidate.scores[sid] for sid in common_ids])
    comparator_ids = [s.id for s in other_signatures]

    if outcome.type is OutcomeType.SURVIVAL:
        return _cox_multivariate(candidate, outcome, other_signatures, common_ids, x, comparator_ids)

    sub_outcome = outcome.subset(common_ids)
    y = sub_outcome.values.astype(float)
    if other_signatures:
        z_cov = np.column_stack([[s.scores[sid] for sid in common_ids] for s in other_signatures])
    else:
        z_cov = np.empty((len(common_ids), 0))
    z = np.column_stack([np.ones(len(common_ids)), z_cov])

    observed, p_value = freedman_lane_core(x, y, z, n_permutations, seed)
    return MultivariateAddsValueResult(comparator_ids, "freedman_lane_permutation", observed, p_value, n_permutations)


def _cox_multivariate(
    candidate: Candidate,
    outcome: OutcomeSpec,
    other_signatures: list[Candidate],
    common_ids: list[str],
    x: np.ndarray,
    comparator_ids: list[str],
) -> MultivariateAddsValueResult:
    sub_outcome = outcome.subset(common_ids)
    df = pd.DataFrame({"time": sub_outcome.values[:, 0], "event": sub_outcome.values[:, 1], "candidate": x})
    for s in other_signatures:
        df[s.id] = [s.scores[sid] for sid in common_ids]

    cph = CoxPHFitter()
    cph.fit(df, duration_col="time", event_col="event")
    coef = float(cph.params_["candidate"])
    p_value = float(cph.summary.loc["candidate", "p"])
    return MultivariateAddsValueResult(comparator_ids, "cox_wald", coef, p_value, None)
