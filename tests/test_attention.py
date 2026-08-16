import numpy as np
import pandas as pd
import pytest

from discovery_engine.campaign.candidate import Candidate
from discovery_engine.campaign.correction import run_campaign_correction
from discovery_engine.campaign.registry import CandidateRegistry
from discovery_engine.data.cohort import CohortData
from discovery_engine.discovery.attention import AttentionDiscovery, _AttentionNet
from discovery_engine.outcomes import OutcomeSpec, OutcomeType
from discovery_engine.validation.stability import TrainedRun, match_units_across_seeds, stability_check

_SMALL_CONFIG = {"d_model": 8, "n_heads": 2}


def _planted_survival_cohort(n=150, seed=0, beta=2.0):
    rng = np.random.default_rng(seed)
    sample_ids = [f"S{i:04d}" for i in range(n)]
    latent = rng.normal(size=n)
    rna = pd.DataFrame({f"gene_{i}": latent + rng.normal(scale=0.2, size=n) for i in range(10)}, index=sample_ids)
    protein = pd.DataFrame({f"prot_{i}": latent + rng.normal(scale=0.2, size=n) for i in range(10)}, index=sample_ids)
    data = CohortData(modalities={"rnaseq": rna, "protein": protein})
    baseline_rate = 0.1
    hazard = baseline_rate * np.exp(beta * latent)
    true_time = rng.exponential(1.0 / hazard)
    censor_time = rng.exponential(scale=1.0 / baseline_rate * 3)
    time = np.minimum(true_time, censor_time)
    event = (true_time <= censor_time).astype(float)
    outcome = OutcomeSpec(OutcomeType.SURVIVAL, sample_ids, np.column_stack([time, event]))
    return data, outcome


def _planted_binary_cohort(n=150, seed=0):
    rng = np.random.default_rng(seed)
    sample_ids = [f"S{i:04d}" for i in range(n)]
    latent = rng.normal(size=n)
    rna = pd.DataFrame({f"gene_{i}": latent + rng.normal(scale=0.2, size=n) for i in range(10)}, index=sample_ids)
    protein = pd.DataFrame({f"prot_{i}": latent + rng.normal(scale=0.2, size=n) for i in range(10)}, index=sample_ids)
    data = CohortData(modalities={"rnaseq": rna, "protein": protein})
    y = (latent > np.median(latent)).astype(float)
    outcome = OutcomeSpec(OutcomeType.BINARY, sample_ids, y)
    return data, outcome


def _signal_vs_noise_continuous_cohort(n=200, seed=0, n_signal=5, n_noise=15):
    rng = np.random.default_rng(seed)
    sample_ids = [f"S{i:04d}" for i in range(n)]
    latent = rng.normal(size=n)
    signal_cols = {f"signal_{i}": latent + rng.normal(scale=0.2, size=n) for i in range(n_signal)}
    noise_cols = {f"noise_{i}": rng.normal(size=n) for i in range(n_noise)}
    rna = pd.DataFrame({**signal_cols, **noise_cols}, index=sample_ids)
    data = CohortData(modalities={"rnaseq": rna})
    y = latent + rng.normal(scale=0.3, size=n)
    outcome = OutcomeSpec(OutcomeType.CONTINUOUS, sample_ids, y)
    return data, outcome, list(signal_cols.keys()), list(noise_cols.keys())


def test_d_model_not_divisible_by_n_heads_raises():
    with pytest.raises(ValueError):
        _AttentionNet(n_features=10, d_model=7, n_heads=2)


def test_discover_is_bit_reproducible_given_same_seed():
    data, outcome = _planted_survival_cohort(n=80, seed=2)
    disc = AttentionDiscovery()
    kwargs = dict(modalities=["rnaseq", "protein"], head_configs=[_SMALL_CONFIG], n_seeds=2, max_epochs=15, patience=4)
    c1 = disc.discover(data, outcome, np.random.default_rng(7), **kwargs)
    c2 = disc.discover(data, outcome, np.random.default_rng(7), **kwargs)
    assert len(c1) == len(c2) == 1
    assert c1[0].scores == c2[0].scores
    assert c1[0].provenance["stability"] == c2[0].provenance["stability"]
    assert c1[0].provenance["attention_weights"] == c2[0].provenance["attention_weights"]
    assert c1[0].provenance["attention_weight_stability"] == c2[0].provenance["attention_weight_stability"]


def test_discover_supports_survival_binary_and_continuous_outcomes():
    disc = AttentionDiscovery()
    kwargs = dict(modalities=["rnaseq", "protein"], head_configs=[_SMALL_CONFIG], n_seeds=1, max_epochs=10, patience=3)

    data_s, outcome_s = _planted_survival_cohort(n=60, seed=1)
    assert len(disc.discover(data_s, outcome_s, np.random.default_rng(0), **kwargs)) == 1

    data_b, outcome_b = _planted_binary_cohort(n=60, seed=1)
    assert len(disc.discover(data_b, outcome_b, np.random.default_rng(0), **kwargs)) == 1

    data_c, outcome_c, _, _ = _signal_vs_noise_continuous_cohort(n=60, seed=1)
    kwargs_c = dict(modalities=["rnaseq"], head_configs=[_SMALL_CONFIG], n_seeds=1, max_epochs=10, patience=3)
    assert len(disc.discover(data_c, outcome_c, np.random.default_rng(0), **kwargs_c)) == 1


def test_attention_identifies_planted_signal_features_over_noise_features():
    """The interpretability claim itself, tested directly: attention weight
    on the features that actually drive the outcome should exceed weight on
    pure-noise features, not just "a report was produced".
    """
    data, outcome, signal_features, noise_features = _signal_vs_noise_continuous_cohort(n=200, seed=3)
    disc = AttentionDiscovery()
    candidates = disc.discover(
        data, outcome, np.random.default_rng(5),
        modalities=["rnaseq"], head_configs=[{"d_model": 16, "n_heads": 2}],
        n_seeds=1, max_epochs=100, patience=15,
    )
    weights = candidates[0].provenance["attention_weights"]
    signal_prefixed = [f"rnaseq:{f}" for f in signal_features]
    noise_prefixed = [f"rnaseq:{f}" for f in noise_features]
    mean_signal = np.mean([weights[f] for f in signal_prefixed])
    mean_noise = np.mean([weights[f] for f in noise_prefixed])
    assert mean_signal > mean_noise


def test_attention_weight_stability_direct_shape_and_partial_match():
    """Direct test of the new attention_weight_stability measurement on its
    actual (n_features, 1) shape, mirroring the DeepSurv partial-match
    closure test: 1-of-2 seeds matching must land in "weak", not a
    pass/fail shortcut, and this is measured, not gated, at this layer.
    """
    n_features = 200
    ramp = np.arange(n_features, dtype=float).reshape(-1, 1)
    ref = TrainedRun(latent_scores=ramp)
    run1 = TrainedRun(latent_scores=ramp.copy())
    rng = np.random.default_rng(31)
    run2 = TrainedRun(latent_scores=rng.normal(size=(n_features, 1)))

    matches = match_units_across_seeds([ref, run1, run2], threshold=0.7)
    assert matches[0]["n_seeds_total"] == 2
    assert matches[0]["n_seeds_matched"] == 1

    candidate = Candidate(
        id="attn_stability_test", method="attention", method_params={}, outcome_blind=False,
        modality_touch=["m1"], sample_ids=["S0"], scores={"S0": 0.0},
        provenance={"stability": matches[0]},  # reusing stability_check's generic interpretation
    )
    result = stability_check(candidate)
    assert result.calibration == "weak"
    assert result.passed is True


def test_attention_candidate_is_outcome_informed_with_working_refit():
    data, outcome = _planted_survival_cohort(n=80, seed=4)
    disc = AttentionDiscovery()
    candidate = disc.discover(
        data, outcome, np.random.default_rng(6),
        modalities=["rnaseq", "protein"], head_configs=[_SMALL_CONFIG], n_seeds=1, max_epochs=15, patience=4,
    )[0]
    assert candidate.outcome_blind is False
    assert candidate.refit is not None

    registry = CandidateRegistry()
    registry.add(candidate)
    assert len(registry) == 1


def test_refit_reruns_from_scratch_and_carries_no_stability_keys():
    data, outcome = _planted_survival_cohort(n=80, seed=8)
    disc = AttentionDiscovery()
    candidate = disc.discover(
        data, outcome, np.random.default_rng(10),
        modalities=["rnaseq", "protein"], head_configs=[_SMALL_CONFIG], n_seeds=1, max_epochs=15, patience=4,
    )[0]

    permuted = outcome.permute(np.random.default_rng(11))
    refit_result = candidate.refit(data, permuted, np.random.default_rng(12))

    assert refit_result.scores != candidate.scores
    assert "stability" not in refit_result.provenance
    assert "attention_weight_stability" not in refit_result.provenance
    assert "attention_weights" not in refit_result.provenance


def test_attention_candidate_survives_real_campaign_correction_without_raising():
    data, outcome = _planted_survival_cohort(n=60, seed=9, beta=2.0)
    disc = AttentionDiscovery()
    candidate = disc.discover(
        data, outcome, np.random.default_rng(14),
        modalities=["rnaseq", "protein"], head_configs=[_SMALL_CONFIG], n_seeds=1, max_epochs=15, patience=4,
    )[0]

    results = run_campaign_correction([candidate], data, outcome, n_permutations=5, seed=0)
    assert candidate.id in results
    assert 0.0 < results[candidate.id].campaign_p_value <= 1.0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
