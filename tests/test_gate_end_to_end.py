"""Phase 2 proof: the whole pipeline, discovery -> per-candidate validation
gate -> campaign-wide correction -> signature registry, working end to end
on synthetic data with a known planted signal, using the one classical
discovery method (hierarchical clustering) built so far.
"""

import numpy as np
import pandas as pd
import pytest

from discovery_engine.campaign.campaign import run_campaign
from discovery_engine.data.cohort import CohortData
from discovery_engine.discovery.clustering import HierarchicalClusteringDiscovery
from discovery_engine.outcomes import OutcomeSpec, OutcomeType
from discovery_engine.signatures.registry import SignatureRegistry


def _build_cohort_and_outcome(effect_size: float, seed: int, n_samples: int = 200):
    rng = np.random.default_rng(seed)
    sample_ids = [f"S{i:04d}" for i in range(n_samples)]
    latent = rng.normal(size=n_samples)

    module_features = {
        f"module_gene_{i}": latent + rng.normal(scale=0.3, size=n_samples) for i in range(6)
    }
    noise_features = {f"noise_gene_{i}": rng.normal(size=n_samples) for i in range(35)}
    df = pd.DataFrame({**module_features, **noise_features}, index=sample_ids)

    grade = rng.integers(1, 4, size=n_samples)  # irrelevant clinical covariate
    covariates = pd.DataFrame({"grade": grade}, index=sample_ids)

    data = CohortData(modalities={"rnaseq": df}, covariates=covariates)

    residual_scale = float(np.sqrt(max(1e-6, 1 - effect_size**2)))
    y = effect_size * latent + residual_scale * rng.normal(size=n_samples)
    outcome = OutcomeSpec(OutcomeType.CONTINUOUS, sample_ids, y)

    return data, outcome, list(module_features.keys())


def test_end_to_end_pipeline_recovers_planted_module_and_records_every_candidate(tmp_path):
    data, outcome, true_module_features = _build_cohort_and_outcome(effect_size=0.65, seed=0)

    candidates = HierarchicalClusteringDiscovery().discover(data, modality="rnaseq", k_values=[2, 4, 6, 8])
    assert len(candidates) > 5, "expected a real campaign-scale pool, not a single candidate"

    registry = SignatureRegistry(tmp_path / "signatures.db")
    report = run_campaign(
        campaign_id="camp1",
        cohort_id="synthetic_cohort",
        candidates=candidates,
        data=data,
        outcome=outcome,
        signature_registry=registry,
        covariate_col="grade",
        correction_n_permutations=300,
        gate_n_permutations=150,
        seed=1,
    )

    # Nothing silently dropped: every candidate discovery produced has a report.
    assert len(report.candidate_reports) == len(candidates)
    assert {r.candidate_id for r in report.candidate_reports} == {c.id for c in candidates}

    assert len(report.accepted) >= 1, "expected the planted module to survive the full pipeline"

    best = max(
        report.accepted,
        key=lambda r: len(
            set(next(c for c in candidates if c.id == r.candidate_id).provenance["member_features"])
            & set(true_module_features)
        ),
    )
    best_candidate = next(c for c in candidates if c.id == best.candidate_id)
    overlap = len(set(best_candidate.provenance["member_features"]) & set(true_module_features)) / len(
        true_module_features
    )
    assert overlap >= 0.8, "the accepted signature should substantially recover the planted module"

    top_attributed = {a.feature for a in best.gate.interpretability.top_features}
    assert top_attributed & set(true_module_features), "interpretability report should surface the real module genes"

    assert len(registry) == len(report.accepted)


def test_end_to_end_pipeline_on_pure_noise_accepts_almost_nothing(tmp_path):
    """Same pipeline, same K-grid campaign scale, but the outcome is
    independent of every feature. Campaign correction plus the gate should
    keep false discoveries rare, not merely "not everything".
    """
    rng = np.random.default_rng(42)
    n_samples = 200
    sample_ids = [f"S{i:04d}" for i in range(n_samples)]
    noise_df = pd.DataFrame(
        {f"gene_{i}": rng.normal(size=n_samples) for i in range(40)}, index=sample_ids
    )
    grade = rng.integers(1, 4, size=n_samples)
    data = CohortData(modalities={"rnaseq": noise_df}, covariates=pd.DataFrame({"grade": grade}, index=sample_ids))
    outcome = OutcomeSpec(OutcomeType.CONTINUOUS, sample_ids, rng.normal(size=n_samples))

    candidates = HierarchicalClusteringDiscovery().discover(data, modality="rnaseq", k_values=[2, 4, 6, 8])
    registry = SignatureRegistry(tmp_path / "signatures.db")
    report = run_campaign(
        campaign_id="camp_noise",
        cohort_id="synthetic_noise_cohort",
        candidates=candidates,
        data=data,
        outcome=outcome,
        signature_registry=registry,
        covariate_col="grade",
        correction_n_permutations=300,
        gate_n_permutations=150,
        seed=2,
    )

    assert len(report.accepted) <= max(1, len(candidates) // 10)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
