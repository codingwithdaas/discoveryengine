"""Real bug, caught against real TCGA-BLCA data: every discover() method
built its candidate id from the CohortData modality KEY (e.g. "rnaseq"),
never from what was actually inside that modality's DataFrame. Two discover()
calls against different gene subsets of the same modality (variance-top-1000
vs Hallmark) with the same structural parameters (same k, same n_components,
same hidden_dims, ...) therefore produced IDENTICAL candidate ids.

That silently broke run_campaign_correction's `results: dict[str,
CorrectionResult]` (correction.py), keyed by id: the second candidate's
result overwrote the first's, and run_campaign's per-candidate lookup
(`correction_results[candidate.id]`) then handed BOTH candidates the same
(wrong, for one of them) campaign_p_value -- discovered only after a full
~113-minute real campaign run, by noticing duplicate-id rows in the report
with suspiciously identical p-values.

Every discover() method now accepts an optional `feature_subset: str | None`
that gets folded into the id (default None: id unchanged, so no existing
caller/test needed to change). This test proves each fixed method actually
produces distinct ids when given distinct feature_subset labels, same
structural config otherwise -- the exact scenario that broke silently.
"""

import numpy as np
import pandas as pd

from discovery_engine.data.cohort import CohortData
from discovery_engine.discovery.attention import AttentionDiscovery
from discovery_engine.discovery.autoencoder import AutoencoderDiscovery
from discovery_engine.discovery.clustering import HierarchicalClusteringDiscovery
from discovery_engine.discovery.deepsurv import DeepSurvDiscovery
from discovery_engine.discovery.interaction import InteractionSearchDiscovery
from discovery_engine.discovery.louvain import LouvainDiscovery
from discovery_engine.discovery.nmf import NMFDiscovery
from discovery_engine.discovery.rsf import RandomSurvivalForestDiscovery
from discovery_engine.outcomes import OutcomeSpec, OutcomeType

_SMALL_NEURAL = dict(n_seeds=2, max_epochs=5, patience=2)


def _cohort(seed=0, n=60):
    rng = np.random.default_rng(seed)
    sample_ids = [f"S{i}" for i in range(n)]
    latent = rng.normal(size=n)
    df = pd.DataFrame(
        {f"gene_{i}": latent + rng.normal(scale=0.2, size=n) for i in range(8)}, index=sample_ids
    )
    return CohortData(modalities={"rnaseq": df}), sample_ids


def _survival_outcome(sample_ids, seed=1):
    rng = np.random.default_rng(seed)
    n = len(sample_ids)
    time = rng.exponential(1.0, size=n)
    event = rng.integers(0, 2, size=n).astype(float)
    return OutcomeSpec(OutcomeType.SURVIVAL, sample_ids, np.column_stack([time, event]))


def _assert_disjoint_ids(candidates_a, candidates_b):
    ids_a = {c.id for c in candidates_a}
    ids_b = {c.id for c in candidates_b}
    assert ids_a, "subset A produced no candidates"
    assert ids_b, "subset B produced no candidates"
    assert ids_a.isdisjoint(ids_b), f"id collision across feature subsets: {ids_a & ids_b}"


def test_hierarchical_clustering_ids_distinct_across_feature_subsets():
    data, _ = _cohort()
    disc = HierarchicalClusteringDiscovery()
    a = disc.discover(data, modality="rnaseq", k_values=[2], feature_subset="variance_top1000")
    b = disc.discover(data, modality="rnaseq", k_values=[2], feature_subset="hallmark_4384")
    _assert_disjoint_ids(a, b)


def test_louvain_ids_distinct_across_feature_subsets():
    data, _ = _cohort()
    disc = LouvainDiscovery()
    a = disc.discover(data, modality="rnaseq", resolutions=[1.0], min_abs_correlation=0.1, seed=1, feature_subset="variance_top1000")
    b = disc.discover(data, modality="rnaseq", resolutions=[1.0], min_abs_correlation=0.1, seed=1, feature_subset="hallmark_4384")
    _assert_disjoint_ids(a, b)


def test_nmf_ids_distinct_across_feature_subsets():
    data, _ = _cohort()
    disc = NMFDiscovery()
    a = disc.discover(data, modality="rnaseq", n_components_options=[2], n_seeds=2, max_iter=50, seed=2, feature_subset="variance_top1000")
    b = disc.discover(data, modality="rnaseq", n_components_options=[2], n_seeds=2, max_iter=50, seed=2, feature_subset="hallmark_4384")
    _assert_disjoint_ids(a, b)


def test_interaction_search_ids_distinct_across_feature_subsets():
    data, _ = _cohort()
    disc = InteractionSearchDiscovery()
    pairs = [("gene_0", "gene_1")]
    a = disc.discover(data, modality="rnaseq", feature_pairs=pairs, feature_subset="variance_top1000")
    b = disc.discover(data, modality="rnaseq", feature_pairs=pairs, feature_subset="hallmark_4384")
    _assert_disjoint_ids(a, b)


def test_autoencoder_ids_distinct_across_feature_subsets():
    data, _ = _cohort()
    disc = AutoencoderDiscovery()
    a = disc.discover(data, modalities=["rnaseq"], latent_dims=[2], seed=3, feature_subset="variance_top1000", **_SMALL_NEURAL)
    b = disc.discover(data, modalities=["rnaseq"], latent_dims=[2], seed=3, feature_subset="hallmark_4384", **_SMALL_NEURAL)
    _assert_disjoint_ids(a, b)


def test_rsf_ids_distinct_across_feature_subsets():
    data, sample_ids = _cohort()
    outcome = _survival_outcome(sample_ids)
    disc = RandomSurvivalForestDiscovery()
    config = [{"n_estimators": 10, "max_depth": 3, "min_samples_leaf": 5}]
    a = disc.discover(data, outcome, np.random.default_rng(4), modalities=["rnaseq"], forest_configs=config, feature_subset="variance_top1000")
    b = disc.discover(data, outcome, np.random.default_rng(4), modalities=["rnaseq"], forest_configs=config, feature_subset="hallmark_4384")
    _assert_disjoint_ids(a, b)


def test_deepsurv_ids_distinct_across_feature_subsets():
    data, sample_ids = _cohort()
    outcome = _survival_outcome(sample_ids)
    disc = DeepSurvDiscovery()
    a = disc.discover(data, outcome, np.random.default_rng(5), modalities=["rnaseq"], hidden_dims_options=[[4]], feature_subset="variance_top1000", **_SMALL_NEURAL)
    b = disc.discover(data, outcome, np.random.default_rng(5), modalities=["rnaseq"], hidden_dims_options=[[4]], feature_subset="hallmark_4384", **_SMALL_NEURAL)
    _assert_disjoint_ids(a, b)


def test_attention_ids_distinct_across_feature_subsets():
    data, sample_ids = _cohort()
    outcome = _survival_outcome(sample_ids)
    disc = AttentionDiscovery()
    config = [{"d_model": 8, "n_heads": 2}]
    a = disc.discover(data, outcome, np.random.default_rng(6), modalities=["rnaseq"], head_configs=config, feature_subset="variance_top1000", **_SMALL_NEURAL)
    b = disc.discover(data, outcome, np.random.default_rng(6), modalities=["rnaseq"], head_configs=config, feature_subset="hallmark_4384", **_SMALL_NEURAL)
    _assert_disjoint_ids(a, b)


def test_feature_subset_none_leaves_id_unchanged():
    """Default behavior for every existing caller: no feature_subset given
    means the id is exactly what it was before this fix.
    """
    data, _ = _cohort()
    candidates = HierarchicalClusteringDiscovery().discover(data, modality="rnaseq", k_values=[2])
    assert all("_fs" not in c.id for c in candidates)


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-q"]))
