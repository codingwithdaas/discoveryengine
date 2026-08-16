import inspect

import numpy as np
import pandas as pd

from discovery_engine.data.cohort import CohortData
from discovery_engine.discovery.clustering import HierarchicalClusteringDiscovery


def _cohort_with_planted_module(n_samples=80, n_module_features=6, n_noise_features=20, seed=0):
    rng = np.random.default_rng(seed)
    sample_ids = [f"S{i}" for i in range(n_samples)]
    latent = rng.normal(size=n_samples)

    module_features = {
        f"module_feat_{i}": latent + rng.normal(scale=0.3, size=n_samples) for i in range(n_module_features)
    }
    noise_features = {f"noise_{i}": rng.normal(size=n_samples) for i in range(n_noise_features)}

    df = pd.DataFrame({**module_features, **noise_features}, index=sample_ids)
    return CohortData(modalities={"rnaseq": df}), list(module_features.keys())


def test_discover_signature_has_no_outcome_parameter():
    """Structural proof of outcome-blindness: there is no parameter this
    method could even read outcome data through.
    """
    params = inspect.signature(HierarchicalClusteringDiscovery.discover).parameters
    assert "outcome" not in params


def test_all_candidates_flagged_outcome_blind_and_never_touch_outcome_type():
    cohort, _ = _cohort_with_planted_module()
    candidates = HierarchicalClusteringDiscovery().discover(cohort, modality="rnaseq", k_values=[2, 4, 6])
    assert len(candidates) > 0
    assert all(c.outcome_blind for c in candidates)
    assert all(c.refit is None for c in candidates)


def test_every_k_in_grid_produces_logged_candidates():
    cohort, _ = _cohort_with_planted_module()
    k_values = [2, 3, 5, 8]
    candidates = HierarchicalClusteringDiscovery().discover(cohort, modality="rnaseq", k_values=k_values)
    ks_seen = {c.method_params["k"] for c in candidates}
    assert ks_seen == set(k_values), "every K in the grid must contribute candidates, not just the 'best' one"


def test_recovers_the_planted_module_at_some_k():
    cohort, module_feature_names = _cohort_with_planted_module()
    candidates = HierarchicalClusteringDiscovery().discover(
        cohort, modality="rnaseq", k_values=[2, 3, 4, 5, 6, 8, 10]
    )

    best_overlap = max(
        len(set(c.provenance["member_features"]) & set(module_feature_names)) / len(module_feature_names)
        for c in candidates
    )
    assert best_overlap >= 0.8, "expected at least one module to recover most of the planted correlated block"


def test_zero_variance_feature_dropped_not_crashed(caplog):
    """Real-data finding, not a hypothetical: a Spearman correlation
    against a constant column is NaN, which scipy's linkage() correctly
    refuses. 292 of the real 20,518-gene TCGA-BLCA RNA-seq matrix have
    zero variance across the real cohort (e.g. TSSK2, unexpressed in
    bladder tissue) -- this must be dropped before the distance matrix is
    built, not crash discovery.
    """
    import logging

    n_samples = 80
    rng = np.random.default_rng(4)
    sample_ids = [f"S{i}" for i in range(n_samples)]
    latent = rng.normal(size=n_samples)
    module_features = {f"feat_{i}": latent + rng.normal(scale=0.2, size=n_samples) for i in range(5)}
    df = pd.DataFrame({**module_features, "constant_gene": np.zeros(n_samples)}, index=sample_ids)
    cohort = CohortData(modalities={"rnaseq": df})

    with caplog.at_level(logging.INFO):
        candidates = HierarchicalClusteringDiscovery().discover(cohort, modality="rnaseq", k_values=[2])

    assert len(candidates) > 0
    assert all("constant_gene" not in c.provenance["member_features"] for c in candidates)
    assert any("zero-variance" in r.message for r in caplog.records)


def test_module_score_correlates_with_underlying_latent():
    n_samples = 80
    rng = np.random.default_rng(1)
    sample_ids = [f"S{i}" for i in range(n_samples)]
    latent = rng.normal(size=n_samples)
    module_features = {f"feat_{i}": latent + rng.normal(scale=0.2, size=n_samples) for i in range(5)}
    df = pd.DataFrame(module_features, index=sample_ids)
    cohort = CohortData(modalities={"m": df})

    candidates = HierarchicalClusteringDiscovery().discover(cohort, modality="m", k_values=[1])
    assert len(candidates) == 1
    scores = np.array([candidates[0].scores[sid] for sid in sample_ids])
    corr = np.corrcoef(scores, latent)[0, 1]
    assert abs(corr) > 0.8


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-q"]))
