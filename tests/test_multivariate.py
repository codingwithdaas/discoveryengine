import numpy as np
import pytest

from discovery_engine.campaign.candidate import Candidate
from discovery_engine.outcomes import OutcomeSpec, OutcomeType
from discovery_engine.validation.multivariate import multivariate_adds_value


def _base(n=200, seed=0):
    rng = np.random.default_rng(seed)
    ids = [f"S{i}" for i in range(n)]
    latent = rng.normal(size=n)
    y = 0.7 * latent + np.sqrt(1 - 0.7**2) * rng.normal(size=n)
    outcome = OutcomeSpec(OutcomeType.CONTINUOUS, ids, y)
    return ids, rng, latent, outcome


def _candidate(cid, ids, scores):
    return Candidate(id=cid, method="m", method_params={}, outcome_blind=True, modality_touch=[], sample_ids=ids, scores=scores)


def test_fully_redundant_candidate_adds_nothing_beyond_existing_signature():
    ids, rng, latent, outcome = _base()
    existing = _candidate("existing", ids, dict(zip(ids, latent + rng.normal(scale=0.1, size=len(ids)))))
    redundant = _candidate("redundant", ids, dict(zip(ids, latent + rng.normal(scale=0.1, size=len(ids)))))

    result = multivariate_adds_value(redundant, outcome, [existing], n_permutations=300, seed=1)
    assert result.p_value > 0.05


def test_independent_candidate_adds_value_beyond_existing_signature():
    ids, rng, latent, outcome = _base()
    existing_latent = rng.normal(size=len(ids))
    y2 = 0.5 * latent + 0.5 * existing_latent + rng.normal(scale=0.3, size=len(ids))
    outcome2 = OutcomeSpec(OutcomeType.CONTINUOUS, ids, y2)

    existing = _candidate("existing", ids, dict(zip(ids, existing_latent + rng.normal(scale=0.1, size=len(ids)))))
    independent = _candidate("independent", ids, dict(zip(ids, latent + rng.normal(scale=0.1, size=len(ids)))))

    result = multivariate_adds_value(independent, outcome2, [existing], n_permutations=300, seed=2)
    assert result.p_value < 0.05


def test_no_comparators_still_returns_a_result():
    ids, rng, latent, outcome = _base()
    candidate = _candidate("solo", ids, dict(zip(ids, latent + rng.normal(scale=0.2, size=len(ids)))))
    result = multivariate_adds_value(candidate, outcome, [], n_permutations=200, seed=0)
    assert result.comparator_ids == []
    assert result.p_value < 0.05


def test_survival_outcome_uses_cox_multivariate():
    n = 200
    rng = np.random.default_rng(5)
    ids = [f"S{i}" for i in range(n)]
    latent = rng.normal(size=n)
    risk = np.exp(0.8 * latent)
    time = rng.exponential(scale=1.0 / risk)
    censor = rng.exponential(scale=np.median(time) * 2)
    event = (time <= censor).astype(float)
    observed = np.minimum(time, censor)
    outcome = OutcomeSpec(OutcomeType.SURVIVAL, ids, np.column_stack([observed, event]))

    candidate = _candidate("c", ids, dict(zip(ids, latent + rng.normal(scale=0.1, size=n))))
    result = multivariate_adds_value(candidate, outcome, [], n_permutations=100, seed=0)
    assert result.test_kind == "cox_wald"
    assert result.n_permutations is None
    assert result.p_value < 0.05


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
