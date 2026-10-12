import numpy as np
import pandas as pd
import pytest

from discovery_engine.campaign.campaign import run_campaign
from discovery_engine.data.cohort import CohortData
from discovery_engine.discovery.base import assert_outcome_compatible
from discovery_engine.discovery.clustering import HierarchicalClusteringDiscovery
from discovery_engine.discovery.interaction import InteractionSearchDiscovery
from discovery_engine.outcomes import OutcomeSpec, OutcomeType
from discovery_engine.signatures.registry import SignatureRegistry


def _cohort(n=60, seed=0):
    rng = np.random.default_rng(seed)
    sample_ids = [f"S{i}" for i in range(n)]
    df = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)}, index=sample_ids)
    return CohortData(modalities={"m": df}), sample_ids


def test_assert_outcome_compatible_raises_for_interaction_search_survival():
    with pytest.raises(ValueError, match="not currently trusted"):
        assert_outcome_compatible("interaction_search", OutcomeType.SURVIVAL)


def test_assert_outcome_compatible_allows_interaction_search_continuous_and_binary():
    assert_outcome_compatible("interaction_search", OutcomeType.CONTINUOUS)
    assert_outcome_compatible("interaction_search", OutcomeType.BINARY)


def test_assert_outcome_compatible_allows_other_methods_against_survival():
    assert_outcome_compatible("hierarchical_clustering", OutcomeType.SURVIVAL)
    assert_outcome_compatible("deepsurv", OutcomeType.SURVIVAL)


def test_run_campaign_raises_before_doing_any_work_for_interaction_search_survival(tmp_path):
    data, sample_ids = _cohort()
    candidates = InteractionSearchDiscovery().discover(data, modality="m", feature_pairs=[("a", "b")])
    rng = np.random.default_rng(1)
    time = rng.exponential(5, size=len(sample_ids))
    event = rng.integers(0, 2, size=len(sample_ids)).astype(float)
    outcome = OutcomeSpec(OutcomeType.SURVIVAL, sample_ids, np.column_stack([time, event]))
    registry = SignatureRegistry(tmp_path / "sig.db")

    with pytest.raises(ValueError, match="not currently trusted"):
        run_campaign(
            campaign_id="camp", cohort_id="cohort1", candidates=candidates, data=data, outcome=outcome,
            signature_registry=registry, correction_n_permutations=5, gate_n_permutations=5, allow_low_permutations=True,
        )


def test_run_campaign_proceeds_normally_for_interaction_search_continuous(tmp_path):
    data, sample_ids = _cohort()
    candidates = InteractionSearchDiscovery().discover(data, modality="m", feature_pairs=[("a", "b")])
    rng = np.random.default_rng(1)
    outcome = OutcomeSpec(OutcomeType.CONTINUOUS, sample_ids, rng.normal(size=len(sample_ids)))
    registry = SignatureRegistry(tmp_path / "sig.db")

    report = run_campaign(
        campaign_id="camp", cohort_id="cohort1", candidates=candidates, data=data, outcome=outcome,
        signature_registry=registry, correction_n_permutations=5, gate_n_permutations=5, allow_low_permutations=True,
    )
    assert len(report.candidate_reports) == len(candidates)


def test_run_campaign_proceeds_normally_for_other_methods_against_survival(tmp_path):
    data, sample_ids = _cohort()
    candidates = HierarchicalClusteringDiscovery().discover(data, modality="m", k_values=[2])
    rng = np.random.default_rng(1)
    time = rng.exponential(5, size=len(sample_ids))
    event = rng.integers(0, 2, size=len(sample_ids)).astype(float)
    outcome = OutcomeSpec(OutcomeType.SURVIVAL, sample_ids, np.column_stack([time, event]))
    registry = SignatureRegistry(tmp_path / "sig.db")

    report = run_campaign(
        campaign_id="camp", cohort_id="cohort1", candidates=candidates, data=data, outcome=outcome,
        signature_registry=registry, correction_n_permutations=5, gate_n_permutations=5, allow_low_permutations=True,
    )
    assert len(report.candidate_reports) == len(candidates)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
