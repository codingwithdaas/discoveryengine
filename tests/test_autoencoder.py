import logging

import numpy as np
import pandas as pd
import pytest

from discovery_engine.data.cohort import CohortData
from discovery_engine.discovery.autoencoder import AutoencoderDiscovery
from discovery_engine.validation.stability import TrainedRun, match_units_across_seeds, stability_check


def _planted_multimodal_cohort(n=150, seed=0):
    rng = np.random.default_rng(seed)
    sample_ids = [f"S{i:04d}" for i in range(n)]
    latent = rng.normal(size=n)
    rna = pd.DataFrame(
        {f"gene_{i}": latent + rng.normal(scale=0.2, size=n) for i in range(10)}, index=sample_ids
    )
    protein = pd.DataFrame(
        {f"prot_{i}": latent + rng.normal(scale=0.2, size=n) for i in range(10)}, index=sample_ids
    )
    return CohortData(modalities={"rnaseq": rna, "protein": protein})


def test_match_units_across_seeds_direct():
    """Isolated test of the cross-seed matching logic itself, with hand-built
    latent score arrays (no neural training involved), so this doesn't
    depend on training convergence at all.
    """
    ramp = np.arange(200, dtype=float).reshape(-1, 1)
    ref = TrainedRun(latent_scores=ramp)
    run1 = TrainedRun(latent_scores=ramp.copy())  # perfect match
    rng = np.random.default_rng(42)
    run2 = TrainedRun(latent_scores=rng.normal(size=(200, 1)))  # ~uncorrelated

    matches = match_units_across_seeds([ref, run1, run2], threshold=0.7)
    assert len(matches) == 1
    m = matches[0]
    assert m["n_seeds_total"] == 2
    assert m["matched_seed_fractions"][0] > 0.99
    assert m["matched_seed_fractions"][1] < 0.5
    assert m["n_seeds_matched"] == 1
    assert m["min_abs_correlation_across_seeds"] == pytest.approx(m["matched_seed_fractions"][1])


def test_discover_is_bit_reproducible_given_same_seed():
    data = _planted_multimodal_cohort(n=80, seed=2)
    disc = AutoencoderDiscovery()
    kwargs = dict(
        modalities=["rnaseq", "protein"], latent_dims=[2], n_seeds=2,
        hidden_dims=[8], max_epochs=20, patience=5, seed=7,
    )
    c1 = disc.discover(data, **kwargs)
    c2 = disc.discover(data, **kwargs)
    assert len(c1) == len(c2)
    for a, b in zip(c1, c2):
        assert a.scores == b.scores
        assert a.provenance["stability"] == b.provenance["stability"]


def test_discover_recovers_planted_multimodal_factor_with_stable_unit():
    data = _planted_multimodal_cohort(n=150, seed=1)
    disc = AutoencoderDiscovery(variational=False)
    candidates = disc.discover(
        data, modalities=["rnaseq", "protein"], latent_dims=[2],
        n_seeds=2, hidden_dims=[8], max_epochs=60, patience=10, seed=3,
    )
    assert len(candidates) == 2

    best = max(candidates, key=lambda c: c.provenance["stability"]["min_abs_correlation_across_seeds"])
    stability = best.provenance["stability"]
    assert stability["n_seeds_matched"] == stability["n_seeds_total"]
    assert stability["min_abs_correlation_across_seeds"] > 0.7


def test_stability_provenance_integrates_with_gate_stability_check():
    data = _planted_multimodal_cohort(n=80, seed=4)
    disc = AutoencoderDiscovery()
    candidates = disc.discover(
        data, modalities=["rnaseq", "protein"], latent_dims=[2],
        n_seeds=2, hidden_dims=[8], max_epochs=20, patience=5, seed=9,
    )
    for c in candidates:
        result = stability_check(c)
        assert result.applicable is True
        assert result.calibration in ("noise", "weak", "moderate", "robust")


def test_discover_drops_samples_missing_from_any_modality_and_logs_it(caplog):
    data = _planted_multimodal_cohort(n=60, seed=5)
    protein = data.modalities["protein"].iloc[1:]  # drop one sample from this modality only
    data = CohortData(modalities={"rnaseq": data.modalities["rnaseq"], "protein": protein})

    disc = AutoencoderDiscovery()
    with caplog.at_level(logging.INFO):
        candidates = disc.discover(
            data, modalities=["rnaseq", "protein"], latent_dims=[1],
            n_seeds=1, hidden_dims=[8], max_epochs=5, patience=2, seed=1,
        )
    assert len(candidates[0].sample_ids) == 59
    assert any("dropped" in r.message for r in caplog.records)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
