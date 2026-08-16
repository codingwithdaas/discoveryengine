import inspect

import numpy as np
import pandas as pd
import pytest

from discovery_engine.data.cohort import CohortData
from discovery_engine.discovery.interaction import InteractionSearchDiscovery
from discovery_engine.outcomes import OutcomeSpec, OutcomeType, permutation_test


def test_discover_signature_has_no_outcome_parameter():
    params = inspect.signature(InteractionSearchDiscovery.discover).parameters
    assert "outcome" not in params


def test_all_candidates_flagged_outcome_blind_with_no_refit():
    rng = np.random.default_rng(0)
    n = 60
    sample_ids = [f"S{i}" for i in range(n)]
    df = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n), "c": rng.normal(size=n)}, index=sample_ids)
    data = CohortData(modalities={"m": df})

    candidates = InteractionSearchDiscovery().discover(data, modality="m", feature_pairs=[("a", "b"), ("a", "c")])
    assert len(candidates) == 2
    assert all(c.outcome_blind for c in candidates)
    assert all(c.refit is None for c in candidates)


def test_every_requested_pair_becomes_its_own_candidate():
    rng = np.random.default_rng(1)
    n = 60
    sample_ids = [f"S{i}" for i in range(n)]
    df = pd.DataFrame({f: rng.normal(size=n) for f in ["a", "b", "c", "d"]}, index=sample_ids)
    data = CohortData(modalities={"m": df})

    pairs = [("a", "b"), ("a", "c"), ("b", "d"), ("c", "d")]
    candidates = InteractionSearchDiscovery().discover(data, modality="m", feature_pairs=pairs)
    pairs_seen = {tuple(c.provenance["feature_pair"]) for c in candidates}
    assert pairs_seen == set(pairs), "every requested pair must contribute a candidate, not a filtered subset"


def test_degenerate_self_pair_raises():
    rng = np.random.default_rng(2)
    n = 40
    df = pd.DataFrame({"a": rng.normal(size=n)}, index=[f"S{i}" for i in range(n)])
    data = CohortData(modalities={"m": df})
    with pytest.raises(ValueError):
        InteractionSearchDiscovery().discover(data, modality="m", feature_pairs=[("a", "a")])


def test_zero_variance_feature_pair_skipped_not_crashed(caplog):
    import logging

    n = 40
    sample_ids = [f"S{i}" for i in range(n)]
    rng = np.random.default_rng(3)
    df = pd.DataFrame({"a": rng.normal(size=n), "const": np.ones(n)}, index=sample_ids)
    data = CohortData(modalities={"m": df})
    with caplog.at_level(logging.INFO):
        candidates = InteractionSearchDiscovery().discover(data, modality="m", feature_pairs=[("a", "const")])
    assert candidates == []
    assert any("skipping pair" in r.message for r in caplog.records)


def test_discover_is_deterministic_across_repeated_calls():
    rng = np.random.default_rng(4)
    n = 80
    sample_ids = [f"S{i}" for i in range(n)]
    df = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)}, index=sample_ids)
    data = CohortData(modalities={"m": df})

    c1 = InteractionSearchDiscovery().discover(data, modality="m", feature_pairs=[("a", "b")])
    c2 = InteractionSearchDiscovery().discover(data, modality="m", feature_pairs=[("a", "b")])
    assert c1[0].scores == c2[0].scores


def _true_interaction_cohort(n=300, seed=0, beta=2.0):
    """outcome depends on the INTERACTION z_a*z_b only, no main effects."""
    rng = np.random.default_rng(seed)
    sample_ids = [f"S{i}" for i in range(n)]
    a = rng.normal(size=n)
    b = rng.normal(size=n)
    df = pd.DataFrame({"a": a, "b": b}, index=sample_ids)
    z_a = (a - a.mean()) / a.std()
    z_b = (b - b.mean()) / b.std()
    y = beta * (z_a * z_b) + rng.normal(scale=0.5, size=n)
    return CohortData(modalities={"m": df}), sample_ids, OutcomeSpec(OutcomeType.CONTINUOUS, sample_ids, y)


def _skewed_correlated_main_effects_only_cohort(n=300, seed=11, beta_a=2.0, beta_b=2.0):
    """The actual adversarial cohort that caught the real gap in this
    method's design (see interaction.py's module docstring): correlated
    (shared latent factor) AND skewed (Gamma/Exponential) features, no
    true interaction in the outcome, main effects only. Independent or
    jointly-Gaussian correlated features do NOT expose this failure mode
    (verified separately, see the module docstring); skew is required.
    """
    rng = np.random.default_rng(seed)
    sample_ids = [f"S{i}" for i in range(n)]
    shared = rng.gamma(shape=2.0, scale=1.0, size=n)
    a = shared + rng.exponential(scale=0.5, size=n)
    b = shared + rng.exponential(scale=0.5, size=n)
    df = pd.DataFrame({"a": a, "b": b}, index=sample_ids)
    z_a = (a - a.mean()) / a.std()
    z_b = (b - b.mean()) / b.std()
    linear_part = beta_a * z_a + beta_b * z_b
    y_continuous = linear_part + rng.normal(scale=0.5, size=n)
    y_binary = (linear_part + rng.normal(scale=0.5, size=n) > np.median(linear_part)).astype(float)
    return (
        CohortData(modalities={"m": df}), sample_ids,
        OutcomeSpec(OutcomeType.CONTINUOUS, sample_ids, y_continuous),
        OutcomeSpec(OutcomeType.BINARY, sample_ids, y_binary),
        z_a, z_b,
    )


def test_residualized_score_detects_a_true_interaction():
    data, sample_ids, outcome = _true_interaction_cohort(seed=10)
    candidate = InteractionSearchDiscovery().discover(data, modality="m", feature_pairs=[("a", "b")])[0]
    _, p_value = permutation_test(outcome, candidate.scores, n_permutations=500, seed=0)
    assert p_value < 0.01


def test_residualized_score_is_specific_on_skewed_correlated_main_effects_only():
    """The core verification this method's actual claim needs, mirroring
    the standard applied to attention weights and NMF component loadings.
    This is the SECOND version of this test: the first (independent
    Gaussian features) passed trivially with no real contrast; investigating
    why led to the skewed/correlated cohort below, which then FAILED against
    the original raw-z-score residualization (Spearman rho=-0.38, p=6e-12
    on a main-effects-only cohort) and drove the switch to rank-based
    residualization in interaction.py. This test is what that fix has to
    keep passing, on both CONTINUOUS (Spearman) and BINARY (point-biserial),
    the two outcome types association_statistic actually uses for this
    method (SURVIVAL is not covered by this method's typical use and not
    separately verified here).
    """
    data, sample_ids, outcome_cont, outcome_bin, z_a, z_b = _skewed_correlated_main_effects_only_cohort()
    candidate = InteractionSearchDiscovery().discover(data, modality="m", feature_pairs=[("a", "b")])[0]

    _, p_continuous = permutation_test(outcome_cont, candidate.scores, n_permutations=500, seed=0)
    _, p_binary = permutation_test(outcome_bin, candidate.scores, n_permutations=500, seed=0)
    assert p_continuous > 0.05, f"expected no spurious CONTINUOUS association, got p={p_continuous}"
    assert p_binary > 0.05, f"expected no spurious BINARY association, got p={p_binary}"

    # The historically accurate regression check: a raw, non-residualized
    # product term was NOT actually the spurious one in this cohort (naive
    # p came back 0.80, not significant, contradicting the original
    # assumption) -- the real bug was in the OLD raw-z-score residualization
    # itself. Reproduce that exact prior result here as the "before" side
    # of the contrast, so this test would catch a regression back to it.
    old_design = np.column_stack([np.ones_like(z_a), z_a, z_b])
    old_beta, *_ = np.linalg.lstsq(old_design, z_a * z_b, rcond=None)
    old_residual = z_a * z_b - old_design @ old_beta
    old_scores = {sid: float(old_residual[i]) for i, sid in enumerate(sample_ids)}
    _, old_residualized_p = permutation_test(outcome_cont, old_scores, n_permutations=500, seed=0)
    assert old_residualized_p < 0.01, (
        f"expected to reproduce the OLD raw-z-score residualization's spurious result on this "
        f"exact adversarial cohort (originally p=6.0e-12), got p={old_residualized_p}"
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
