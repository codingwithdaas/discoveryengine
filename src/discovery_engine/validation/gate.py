"""The validation gate: every candidate from every method passes through
this, identically, before campaign-level correction (campaign.correction)
is applied as the separate, final, pool-wide step.

"passed" here is an advisory aggregate over the checks below, not the
campaign-level significance call; a candidate can pass this gate and still
fail to survive campaign-wide correction, and campaign correction can only
run after every candidate (passed or failed here) has been through it, so
nothing is silently dropped before the full pool is assembled.

Aggregate pass rule (documented so it's a stated decision, not an implicit
one): a candidate FAILS this gate if any of the following hold:
  - half-split consistency calibration is "noise"
  - a Freedman-Lane covariate-adjusted p-value (when computable) is >= 0.05
  - it is "fully_redundant" against an already-validated signature
  - it does not add value in a joint model against existing signatures
    (only evaluated when there are existing signatures to compare against)
  - no interpretability report could be produced
  - it is a neural-method candidate whose seed/architecture stability check
    calibrates as "noise" (applicable neural methods only; see validation.stability)
Every individual check's result is kept regardless of pass/fail, so a
failed candidate's specific reasons are always in the record.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from discovery_engine.campaign.candidate import Candidate
from discovery_engine.data.cohort import CohortData
from discovery_engine.outcomes import OutcomeSpec
from discovery_engine.validation.bootstrap import BootstrapCIResult, HalfSplitResult, bootstrap_confidence_interval, half_split_consistency
from discovery_engine.validation.covariate import CovariateStratifiedResult, FreedmanLaneResult, covariate_stratified_test, freedman_lane_test
from discovery_engine.validation.design_crosscheck import DesignCrosscheckResult, median_split_vs_extreme_groups
from discovery_engine.validation.interpretability import InterpretabilityReport, correlation_interpretability_report
from discovery_engine.validation.multivariate import MultivariateAddsValueResult, multivariate_adds_value
from discovery_engine.validation.novelty import NoveltyResult, novelty_check
from discovery_engine.validation.stability import StabilityResult, stability_check


@dataclass
class GateResult:
    candidate_id: str
    half_split: HalfSplitResult
    bootstrap_ci: BootstrapCIResult
    covariate_stratified: CovariateStratifiedResult | None
    freedman_lane: FreedmanLaneResult | None
    design_crosscheck: DesignCrosscheckResult
    multivariate: MultivariateAddsValueResult
    stability: StabilityResult
    novelty: NoveltyResult
    interpretability: InterpretabilityReport
    passed: bool
    failure_reasons: list[str] = field(default_factory=list)


def run_validation_gate(
    candidate: Candidate,
    data: CohortData,
    outcome: OutcomeSpec,
    existing_signatures: list[Candidate],
    covariate_col: str | None = None,
    confound_for_crosscheck: dict[str, float] | None = None,
    n_permutations: int = 500,
    seed: int = 0,
) -> GateResult:
    reasons: list[str] = []

    half_split = half_split_consistency(candidate, outcome, n_splits=30, seed=seed)
    if half_split.calibration == "noise":
        reasons.append(f"half-split consistency calibrated as noise ({half_split.consistency_fraction:.2f})")

    bootstrap_ci = bootstrap_confidence_interval(candidate, outcome, n_bootstrap=1000, seed=seed)

    covariate_stratified = None
    freedman_lane = None
    if covariate_col is not None and data.covariates is not None:
        covariate_stratified = covariate_stratified_test(
            candidate, data, outcome, covariate_col, n_permutations=n_permutations, seed=seed
        )
        try:
            freedman_lane = freedman_lane_test(
                candidate, data, outcome, [covariate_col], n_permutations=n_permutations, seed=seed
            )
            if freedman_lane.p_value >= 0.05:
                reasons.append(
                    f"Freedman-Lane p-value not significant after adjusting for "
                    f"{covariate_col} ({freedman_lane.p_value:.3f})"
                )
        except NotImplementedError:
            pass  # e.g. survival outcome; covariate_stratified_test above already covers it

    design_crosscheck = median_split_vs_extreme_groups(
        candidate, outcome, n_permutations=n_permutations, seed=seed, confound=confound_for_crosscheck
    )

    multivariate = multivariate_adds_value(
        candidate, outcome, existing_signatures, n_permutations=n_permutations, seed=seed
    )
    if existing_signatures and multivariate.p_value >= 0.05:
        reasons.append(
            f"does not add value beyond {len(existing_signatures)} already-validated "
            f"signature(s) (p={multivariate.p_value:.3f})"
        )

    stability = stability_check(candidate)
    if stability.applicable and not stability.passed:
        reasons.append(f"seed/architecture stability calibrated as noise ({stability.reason})")

    novelty = novelty_check(candidate, existing_signatures)
    if novelty.overall_classification == "fully_redundant":
        reasons.append("fully redundant with an already-validated signature")

    interpretability = correlation_interpretability_report(candidate, data)
    if not interpretability.top_features:
        reasons.append("no interpretability report could be produced")

    return GateResult(
        candidate_id=candidate.id,
        half_split=half_split,
        bootstrap_ci=bootstrap_ci,
        covariate_stratified=covariate_stratified,
        freedman_lane=freedman_lane,
        design_crosscheck=design_crosscheck,
        multivariate=multivariate,
        stability=stability,
        novelty=novelty,
        interpretability=interpretability,
        passed=len(reasons) == 0,
        failure_reasons=reasons,
    )
