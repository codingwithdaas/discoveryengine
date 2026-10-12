import inspect

import numpy as np
import pandas as pd
import pytest

from discovery_engine.campaign.correction import run_campaign_correction
from discovery_engine.data.cohort import CohortData
from discovery_engine.discovery.louvain import LouvainDiscovery
from discovery_engine.outcomes import OutcomeSpec, OutcomeType
from discovery_engine.validation.stability import stability_check


def _cohort_with_planted_module(n_samples=100, n_module_features=8, n_noise_features=30, seed=0):
    rng = np.random.default_rng(seed)
    sample_ids = [f"S{i}" for i in range(n_samples)]
    latent = rng.normal(size=n_samples)
    module_features = {
        f"module_feat_{i}": latent + rng.normal(scale=0.2, size=n_samples) for i in range(n_module_features)
    }
    noise_features = {f"noise_{i}": rng.normal(size=n_samples) for i in range(n_noise_features)}
    df = pd.DataFrame({**module_features, **noise_features}, index=sample_ids)
    return CohortData(modalities={"rnaseq": df}), list(module_features.keys())


def test_discover_signature_has_no_outcome_parameter():
    params = inspect.signature(LouvainDiscovery.discover).parameters
    assert "outcome" not in params


def test_all_candidates_flagged_outcome_blind_with_no_refit():
    cohort, _ = _cohort_with_planted_module()
    candidates = LouvainDiscovery().discover(cohort, modality="rnaseq", resolutions=[1.0])
    assert len(candidates) > 0
    assert all(c.outcome_blind for c in candidates)
    assert all(c.refit is None for c in candidates)


def test_every_resolution_in_sweep_produces_logged_candidates():
    """Direct answer to the question this design turn was about: every
    resolution value contributes its own candidates, not just whichever
    resolution looks best by some internal comparison.
    """
    cohort, _ = _cohort_with_planted_module()
    resolutions = [0.5, 1.0, 2.0]
    candidates = LouvainDiscovery().discover(cohort, modality="rnaseq", resolutions=resolutions)
    resolutions_seen = {c.method_params["resolution"] for c in candidates}
    assert resolutions_seen == set(resolutions), "every resolution must contribute candidates, not just the 'best' one"


def test_recovers_the_planted_module_at_some_resolution():
    cohort, module_feature_names = _cohort_with_planted_module()
    candidates = LouvainDiscovery().discover(cohort, modality="rnaseq", resolutions=[0.5, 1.0, 1.5, 2.0, 3.0])

    best_overlap = max(
        len(set(c.provenance["member_features"]) & set(module_feature_names)) / len(module_feature_names)
        for c in candidates
    )
    assert best_overlap >= 0.8, "expected at least one community to recover most of the planted correlated block"


def test_discover_is_reproducible_given_same_seed():
    cohort, _ = _cohort_with_planted_module()
    kwargs = dict(modality="rnaseq", resolutions=[0.5, 1.0, 2.0], seed=42)
    c1 = LouvainDiscovery().discover(cohort, **kwargs)
    c2 = LouvainDiscovery().discover(cohort, **kwargs)
    assert [c.provenance["member_features"] for c in c1] == [c.provenance["member_features"] for c in c2]
    assert [c.scores for c in c1] == [c.scores for c in c2]


def test_louvain_candidates_carry_cross_seed_stability_and_are_checked():
    """Louvain was exempted on synthetic evidence; on real TCGA-BLCA RNA-seq
    its communities changed with the seed, so every candidate now carries a
    per-candidate stability record and the gate checks it."""
    cohort, _ = _cohort_with_planted_module()
    candidates = LouvainDiscovery().discover(cohort, modality="rnaseq", resolutions=[1.0], n_seeds=3)
    assert candidates
    for c in candidates:
        info = c.provenance["stability"]
        assert info["n_seeds_total"] == 2
        assert len(info["matched_seed_fractions"]) == 2
        assert len(info["best_member_jaccard_per_other_seed"]) == 2
        assert all(0.0 <= j <= 1.0 for j in info["best_member_jaccard_per_other_seed"])
    result = stability_check(candidates[0])
    assert result.applicable is True
    assert result.calibration is not None


def test_planted_module_is_stable_across_seeds():
    cohort, planted = _cohort_with_planted_module()
    candidates = LouvainDiscovery().discover(cohort, modality="rnaseq", resolutions=[1.0], n_seeds=3)
    best = max(candidates, key=lambda c: len(set(c.provenance["member_features"]) & set(planted)))
    assert stability_check(best).passed


def test_unstable_partition_is_flagged_as_noise():
    """Pure-noise features above the edge threshold give arbitrary
    communities; at least one of them should fail to reproduce across seeds."""
    rng = np.random.default_rng(3)
    base = rng.normal(size=(80, 1))
    # weakly, uniformly correlated block: many near-equivalent partitions
    X = base * 0.45 + rng.normal(size=(80, 60))
    df = pd.DataFrame(X, index=[f"s{i}" for i in range(80)], columns=[f"g{j}" for j in range(60)])
    candidates = LouvainDiscovery().discover(CohortData({"rnaseq": df}), modality="rnaseq",
                                             resolutions=[1.0], min_abs_correlation=0.1, n_seeds=3)
    jaccards = [min(c.provenance["stability"]["best_member_jaccard_per_other_seed"]) for c in candidates]
    assert min(jaccards) < 1.0


def test_n_seeds_below_two_is_rejected():
    cohort, _ = _cohort_with_planted_module()
    with pytest.raises(ValueError, match="n_seeds"):
        LouvainDiscovery().discover(cohort, modality="rnaseq", resolutions=[1.0], n_seeds=1)


def test_louvain_candidate_survives_real_campaign_correction():
    cohort, _ = _cohort_with_planted_module(n_samples=60)
    candidates = LouvainDiscovery().discover(cohort, modality="rnaseq", resolutions=[1.0])
    assert len(candidates) > 0

    rng = np.random.default_rng(0)
    outcome = OutcomeSpec(OutcomeType.CONTINUOUS, list(cohort.modalities["rnaseq"].index), rng.normal(size=60))
    results = run_campaign_correction(candidates, cohort, outcome, n_permutations=10, seed=0, allow_low_permutations=True)
    assert len(results) == len(candidates)
    for r in results.values():
        assert 0.0 < r.campaign_p_value <= 1.0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
