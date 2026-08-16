import numpy as np
import pandas as pd
import pytest

from discovery_engine.campaign.candidate import Candidate
from discovery_engine.campaign.correction import run_campaign_correction
from discovery_engine.campaign.registry import CandidateRegistry
from discovery_engine.data.cohort import CohortData
from discovery_engine.discovery.rsf import RandomSurvivalForestDiscovery
from discovery_engine.outcomes import OutcomeSpec, OutcomeType, association_statistic
from discovery_engine.validation.stability import requires_stability_check, stability_check

_SMALL_CONFIG = {"n_estimators": 25, "max_depth": 5, "min_samples_leaf": 5}


def _planted_survival_cohort(n=150, seed=0, beta=2.0):
    rng = np.random.default_rng(seed)
    sample_ids = [f"S{i:04d}" for i in range(n)]
    latent = rng.normal(size=n)
    rna = pd.DataFrame(
        {f"gene_{i}": latent + rng.normal(scale=0.2, size=n) for i in range(10)}, index=sample_ids
    )
    protein = pd.DataFrame(
        {f"prot_{i}": latent + rng.normal(scale=0.2, size=n) for i in range(10)}, index=sample_ids
    )
    data = CohortData(modalities={"rnaseq": rna, "protein": protein})

    baseline_rate = 0.1
    hazard = baseline_rate * np.exp(beta * latent)
    true_time = rng.exponential(1.0 / hazard)
    censor_time = rng.exponential(scale=1.0 / baseline_rate * 3)
    time = np.minimum(true_time, censor_time)
    event = (true_time <= censor_time).astype(float)
    outcome = OutcomeSpec(OutcomeType.SURVIVAL, sample_ids, np.column_stack([time, event]))
    return data, outcome


def test_random_survival_forest_is_exempt_from_stability_check():
    """RSF is empirically exempted, not in STABILITY_REQUIRED_METHODS (see
    rsf.py's module docstring and validation.stability.EMPIRICALLY_EXEMPTED_METHODS
    for the evidence backing this)."""
    assert requires_stability_check("random_survival_forest") is False
    candidate = Candidate(
        id="c1", method="random_survival_forest", method_params={}, outcome_blind=False,
        modality_touch=["m1"], sample_ids=["S0", "S1", "S2"],
        scores={"S0": 0.1, "S1": 0.2, "S2": 0.3}, provenance={},
    )
    result = stability_check(candidate)
    assert result.applicable is False


def test_discover_is_bit_reproducible_given_same_seed():
    data, outcome = _planted_survival_cohort(n=80, seed=2)
    disc = RandomSurvivalForestDiscovery()
    kwargs = dict(modalities=["rnaseq", "protein"], forest_configs=[_SMALL_CONFIG])
    c1 = disc.discover(data, outcome, np.random.default_rng(7), **kwargs)
    c2 = disc.discover(data, outcome, np.random.default_rng(7), **kwargs)
    assert len(c1) == len(c2) == 1
    assert c1[0].scores == c2[0].scores


def test_discover_recovers_planted_survival_signal():
    data, outcome = _planted_survival_cohort(n=150, seed=1, beta=2.0)
    disc = RandomSurvivalForestDiscovery()
    candidates = disc.discover(
        data, outcome, np.random.default_rng(3),
        modalities=["rnaseq", "protein"], forest_configs=[_SMALL_CONFIG],
    )
    assert len(candidates) == 1
    stat = association_statistic(outcome, candidates[0].scores)
    assert abs(stat) > 3.0  # Cox Wald z for a strongly planted beta=2.0 signal


def test_rsf_candidate_is_outcome_informed_with_working_refit():
    data, outcome = _planted_survival_cohort(n=80, seed=4)
    disc = RandomSurvivalForestDiscovery()
    candidates = disc.discover(
        data, outcome, np.random.default_rng(5),
        modalities=["rnaseq", "protein"], forest_configs=[_SMALL_CONFIG],
    )
    candidate = candidates[0]
    assert candidate.outcome_blind is False
    assert candidate.refit is not None
    assert "stability" not in candidate.provenance

    registry = CandidateRegistry()
    registry.add(candidate)
    assert len(registry) == 1


def test_refit_reruns_from_scratch_under_permutation():
    data, outcome = _planted_survival_cohort(n=80, seed=6)
    disc = RandomSurvivalForestDiscovery()
    candidate = disc.discover(
        data, outcome, np.random.default_rng(9),
        modalities=["rnaseq", "protein"], forest_configs=[_SMALL_CONFIG],
    )[0]

    permuted = outcome.permute(np.random.default_rng(11))
    refit_result = candidate.refit(data, permuted, np.random.default_rng(12))

    assert refit_result.scores != candidate.scores
    assert "stability" not in refit_result.provenance


def test_rsf_candidate_survives_real_campaign_correction_without_raising():
    data, outcome = _planted_survival_cohort(n=60, seed=8, beta=2.0)
    disc = RandomSurvivalForestDiscovery()
    candidate = disc.discover(
        data, outcome, np.random.default_rng(13),
        modalities=["rnaseq", "protein"], forest_configs=[_SMALL_CONFIG],
    )[0]

    results = run_campaign_correction([candidate], data, outcome, n_permutations=5, seed=0)
    assert candidate.id in results
    assert 0.0 < results[candidate.id].campaign_p_value <= 1.0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
