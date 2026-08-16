import numpy as np

from discovery_engine.outcomes import OutcomeSpec, OutcomeType, association_statistic


def test_continuous_association_recovers_correlated_signal():
    rng = np.random.default_rng(0)
    n = 200
    ids = [str(i) for i in range(n)]
    x = rng.normal(size=n)
    y = 0.8 * x + rng.normal(scale=0.3, size=n)
    outcome = OutcomeSpec(OutcomeType.CONTINUOUS, ids, y)
    scores = dict(zip(ids, x))
    stat = association_statistic(outcome, scores)
    assert stat > 0.5


def test_binary_association_recovers_correlated_signal():
    rng = np.random.default_rng(1)
    n = 200
    ids = [str(i) for i in range(n)]
    x = rng.normal(size=n)
    y = (x + rng.normal(scale=0.5, size=n) > 0).astype(float)
    outcome = OutcomeSpec(OutcomeType.BINARY, ids, y)
    scores = dict(zip(ids, x))
    stat = association_statistic(outcome, scores)
    assert stat > 0.3


def test_survival_association_recovers_correlated_signal():
    rng = np.random.default_rng(2)
    n = 150
    ids = [str(i) for i in range(n)]
    x = rng.normal(size=n)
    risk = np.exp(0.9 * x)
    time = rng.exponential(scale=1.0 / risk)
    censor = rng.exponential(scale=np.median(time) * 2)
    event = (time <= censor).astype(float)
    observed = np.minimum(time, censor)
    outcome = OutcomeSpec(OutcomeType.SURVIVAL, ids, np.column_stack([observed, event]))
    scores = dict(zip(ids, x))
    stat = association_statistic(outcome, scores)
    assert abs(stat) > 2.0  # Wald z, |z| > 2 ~ p < 0.05 for a real planted effect


def test_permute_survival_keeps_time_event_pairs_joined():
    ids = [str(i) for i in range(20)]
    time = np.arange(20, dtype=float)
    event = np.array([i % 2 for i in range(20)], dtype=float)
    outcome = OutcomeSpec(OutcomeType.SURVIVAL, ids, np.column_stack([time, event]))
    rng = np.random.default_rng(0)
    permuted = outcome.permute(rng)

    original_pairs = {tuple(row) for row in outcome.values}
    permuted_pairs = {tuple(row) for row in permuted.values}
    assert original_pairs == permuted_pairs


def test_subset_reorders_by_requested_sample_ids():
    outcome = OutcomeSpec(OutcomeType.CONTINUOUS, ["a", "b", "c"], np.array([1.0, 2.0, 3.0]))
    sub = outcome.subset(["c", "a"])
    assert sub.sample_ids == ["c", "a"]
    assert list(sub.values) == [3.0, 1.0]


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-q"]))
