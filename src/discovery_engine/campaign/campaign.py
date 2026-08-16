"""Top-level campaign orchestrator: ties discovery output, the per-candidate
validation gate, pool-wide campaign correction, and the persistent
SignatureRegistry into one call. This is what proves the whole pipeline
works end to end, not just each piece in isolation.

Order of operations matters and is fixed here:
  0. Every candidate's (method, outcome.type) pair is checked against
     discovery.base.UNVERIFIED_OUTCOME_TYPES before anything else runs --
     an outcome-blind method's outcome-blindness guarantee only covers
     candidate GENERATION, not whether every downstream statistic against
     every outcome type is trustworthy (interaction_search vs SURVIVAL is
     the concrete case this caught).
  1. Campaign correction runs over the FULL candidate pool first (every
     candidate this campaign produced, pass or fail on the gate), since the
     correction's null distribution needs the whole pool regardless of any
     individual candidate's gate outcome.
  2. Each candidate then goes through the per-candidate validation gate,
     checked for novelty/added-value against every signature already in the
     registry PLUS every signature already accepted earlier in this same
     campaign (not just signatures from prior campaigns), so two near-
     duplicate candidates surfaced by different methods in the same run
     don't both get reported as independently novel.
  3. A candidate is accepted into the signature registry only if it both
     passed the gate and survived campaign-level correction at `alpha`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from discovery_engine.campaign.candidate import Candidate
from discovery_engine.campaign.correction import CorrectionResult, run_campaign_correction
from discovery_engine.data.cohort import CohortData
from discovery_engine.discovery.base import assert_outcome_compatible
from discovery_engine.outcomes import OutcomeSpec
from discovery_engine.signatures.registry import SignatureRegistry
from discovery_engine.validation.gate import GateResult, run_validation_gate


@dataclass
class CampaignCandidateReport:
    candidate_id: str
    method: str
    gate: GateResult
    correction: CorrectionResult
    accepted: bool


@dataclass
class CampaignReport:
    campaign_id: str
    cohort_id: str
    alpha: float
    candidate_reports: list[CampaignCandidateReport] = field(default_factory=list)

    @property
    def accepted(self) -> list[CampaignCandidateReport]:
        return [r for r in self.candidate_reports if r.accepted]

    @property
    def rejected(self) -> list[CampaignCandidateReport]:
        return [r for r in self.candidate_reports if not r.accepted]


def run_campaign(
    campaign_id: str,
    cohort_id: str,
    candidates: list[Candidate],
    data: CohortData,
    outcome: OutcomeSpec,
    signature_registry: SignatureRegistry,
    covariate_col: str | None = None,
    confound_for_crosscheck: dict[str, float] | None = None,
    correction_n_permutations: int = 1000,
    gate_n_permutations: int = 500,
    gate_bootstrap_n_resamples: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> CampaignReport:
    for c in candidates:
        assert_outcome_compatible(c.method, outcome.type)

    correction_results = run_campaign_correction(
        candidates, data, outcome, n_permutations=correction_n_permutations, seed=seed
    )

    accepted_so_far = signature_registry.all_as_candidates(cohort_id=cohort_id)
    report = CampaignReport(campaign_id=campaign_id, cohort_id=cohort_id, alpha=alpha)

    for candidate in candidates:
        gate_result = run_validation_gate(
            candidate,
            data,
            outcome,
            accepted_so_far,
            covariate_col=covariate_col,
            confound_for_crosscheck=confound_for_crosscheck,
            n_permutations=gate_n_permutations,
            bootstrap_n_resamples=gate_bootstrap_n_resamples,
            seed=seed,
        )
        correction_result = correction_results[candidate.id]
        accepted = gate_result.passed and correction_result.campaign_p_value < alpha

        report.candidate_reports.append(
            CampaignCandidateReport(candidate.id, candidate.method, gate_result, correction_result, accepted)
        )

        if accepted:
            signature_registry.add(
                candidate,
                cohort_id=cohort_id,
                campaign_id=campaign_id,
                validation_summary={
                    "half_split_consistency": gate_result.half_split.consistency_fraction,
                    "half_split_calibration": gate_result.half_split.calibration,
                    "campaign_p_value": correction_result.campaign_p_value,
                    "bootstrap_ci": [gate_result.bootstrap_ci.ci_low, gate_result.bootstrap_ci.ci_high],
                },
            )
            accepted_so_far.append(candidate)

    return report
