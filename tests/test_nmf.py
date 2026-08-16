import numpy as np
import pandas as pd
import pytest

from discovery_engine.campaign.candidate import Candidate
from discovery_engine.data.cohort import CohortData
from discovery_engine.discovery.nmf import NMFDiscovery, _min_max_scale, _train_one_seed
from discovery_engine.validation.stability import requires_stability_check, stability_check


def _cohort_with_planted_module(n_samples=100, n_module_features=6, n_noise_features=20, seed=0):
    """Deliberately uses standard-normal (signed) values, not pre-shifted
    non-negative data, so this also exercises min-max scaling's handling
    of a raw modality that isn't already non-negative.
    """
    rng = np.random.default_rng(seed)
    sample_ids = [f"S{i}" for i in range(n_samples)]
    latent = rng.normal(size=n_samples)
    module_features = {
        f"module_feat_{i}": latent + rng.normal(scale=0.3, size=n_samples) for i in range(n_module_features)
    }
    noise_features = {f"noise_{i}": rng.normal(size=n_samples) for i in range(n_noise_features)}
    df = pd.DataFrame({**module_features, **noise_features}, index=sample_ids)
    return CohortData(modalities={"rnaseq": df}), sample_ids, list(module_features.keys()), latent


def _nonneg_cohort_with_planted_module(n_samples=120, n_module_features=6, n_noise_features=25, seed=0):
    """Genuinely non-negative, parts-based generative structure (Gamma
    latent, Exponential noise), matching what NMF's decomposition actually
    assumes and what real expression/abundance data looks like -- unlike
    _cohort_with_planted_module's signed Gaussian data forced non-negative
    via min-max scaling, which is fine for testing the SCALING MECHANICS
    but is not the regime NMF's "which features load onto the real factor"
    claim can be expected to hold in (measured directly below: the signed-
    data version of this exact test failed, mean noise loading exceeded
    mean signal loading, because min-max squashing signed, symmetric,
    same-shaped signal and noise features leaves little for NMF's
    non-negative "high value = characteristic of this factor" structure to
    actually latch onto).
    """
    rng = np.random.default_rng(seed)
    sample_ids = [f"S{i}" for i in range(n_samples)]
    latent = rng.gamma(shape=2.0, scale=1.0, size=n_samples)
    module_features = {
        f"module_feat_{i}": latent + rng.exponential(scale=0.5, size=n_samples) for i in range(n_module_features)
    }
    noise_features = {f"noise_{i}": rng.exponential(scale=1.0, size=n_samples) for i in range(n_noise_features)}
    df = pd.DataFrame({**module_features, **noise_features}, index=sample_ids)
    return CohortData(modalities={"rnaseq": df}), sample_ids, list(module_features.keys()), latent


def test_min_max_scale_handles_signed_input_and_produces_zero_to_one_range():
    df = pd.DataFrame({"a": [-5.0, 0.0, 5.0, 10.0], "b": [1.0, 1.0, 1.0, 1.0], "c": [-2.0, -1.0, 0.0, 1.0]})
    scaled, feature_names = _min_max_scale(df)
    assert feature_names == ["a", "c"]  # "b" (zero-range) dropped
    assert scaled.min() >= 0.0
    assert scaled.max() <= 1.0
    assert (scaled >= 0).all()


def test_production_init_is_genuinely_stochastic_not_silently_deterministic():
    """Insurance against exactly the failure mode the stability wiring
    depends on not happening: if init ever silently changes from "random"
    to a deterministic scheme (e.g. "nndsvd"), two different random_state
    values would produce BIT-IDENTICAL component matrices, and the whole
    multi-seed stability check would silently stop testing anything real
    while still reporting perfect "stability". Calls the actual production
    _train_one_seed function, not a reimplementation.
    """
    data, sample_ids, _, _ = _cohort_with_planted_module(n_samples=60)
    x, _ = _min_max_scale(data.modalities["rnaseq"])
    run_a = _train_one_seed(x, n_components=3, max_iter=200, rng=np.random.default_rng(1))
    run_b = _train_one_seed(x, n_components=3, max_iter=200, rng=np.random.default_rng(2))
    assert not np.array_equal(run_a.W, run_b.W)


def test_discover_is_bit_reproducible_given_same_outer_seed():
    data, sample_ids, _, _ = _cohort_with_planted_module(n_samples=60)
    disc = NMFDiscovery()
    kwargs = dict(modality="rnaseq", n_components_options=[3], n_seeds=2, max_iter=200, seed=7)
    c1 = disc.discover(data, **kwargs)
    c2 = disc.discover(data, **kwargs)
    assert len(c1) == len(c2) == 3
    assert [c.scores for c in c1] == [c.scores for c in c2]
    assert [c.provenance["stability"] for c in c1] == [c.provenance["stability"] for c in c2]


def test_every_component_at_every_rank_becomes_a_candidate():
    data, sample_ids, _, _ = _cohort_with_planted_module(n_samples=80)
    n_components_options = [2, 4]
    candidates = NMFDiscovery().discover(
        data, modality="rnaseq", n_components_options=n_components_options, n_seeds=2, max_iter=200, seed=1
    )
    assert len(candidates) == sum(n_components_options)
    ranks_seen = {c.method_params["n_components"] for c in candidates}
    assert ranks_seen == set(n_components_options)


def test_recovers_planted_module_at_some_rank():
    data, sample_ids, module_features, latent = _nonneg_cohort_with_planted_module(n_samples=100, seed=2)
    candidates = NMFDiscovery().discover(
        data, modality="rnaseq", n_components_options=[2, 4], n_seeds=1, max_iter=300, seed=3
    )
    best_corr = max(
        abs(np.corrcoef([c.scores[sid] for sid in sample_ids], latent)[0, 1]) for c in candidates
    )
    assert best_corr > 0.7


def test_component_loadings_identify_planted_signal_features_over_noise():
    data, sample_ids, module_features, latent = _nonneg_cohort_with_planted_module(
        n_samples=120, n_module_features=6, n_noise_features=25, seed=4
    )
    candidates = NMFDiscovery().discover(
        data, modality="rnaseq", n_components_options=[3], n_seeds=1, max_iter=300, seed=5
    )
    best = max(
        candidates, key=lambda c: abs(np.corrcoef([c.scores[sid] for sid in sample_ids], latent)[0, 1])
    )
    loadings = best.provenance["component_loadings"]
    noise_features = [f for f in loadings if f not in module_features]
    mean_signal = np.mean([loadings[f] for f in module_features])
    mean_noise = np.mean([loadings[f] for f in noise_features])
    assert mean_signal > mean_noise


def test_nmf_is_in_stability_required_methods():
    assert requires_stability_check("nmf") is True
    bare_candidate = Candidate(
        id="c1", method="nmf", method_params={}, outcome_blind=True,
        modality_touch=["m1"], sample_ids=["S0", "S1", "S2"], scores={"S0": 0.1, "S1": 0.2, "S2": 0.3},
        provenance={},
    )
    with pytest.raises(ValueError):
        stability_check(bare_candidate)


def test_overparameterized_rank_surfaces_real_instability_through_the_gate():
    """Closes the loop: confirms the wiring actually catches the
    instability measured before this module was built, not just that
    low-rank configs look fine. A high-rank config on a small, noisy
    cohort should produce at least one candidate whose real stability_check
    calibration is NOT "robust" -- the wiring surfacing a real, measured
    phenomenon end to end, not merely existing structurally.
    """
    rng = np.random.default_rng(9)
    n_samples, n_features = 60, 40
    sample_ids = [f"S{i}" for i in range(n_samples)]
    df = pd.DataFrame(rng.normal(size=(n_samples, n_features)), columns=[f"f{i}" for i in range(n_features)], index=sample_ids)
    data = CohortData(modalities={"rnaseq": df})

    candidates = NMFDiscovery().discover(
        data, modality="rnaseq", n_components_options=[15], n_seeds=3, max_iter=200, seed=11
    )
    calibrations = [stability_check(c).calibration for c in candidates]
    assert any(cal != "robust" for cal in calibrations), (
        f"expected at least one non-robust component at an overparameterized rank, got {calibrations}"
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
