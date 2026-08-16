import numpy as np
import pytest

from discovery_engine.campaign.candidate import Candidate
from discovery_engine.validation.novelty import novelty_check


def _c(cid, sample_ids, scores, member_features=None):
    return Candidate(
        id=cid, method="m", method_params={}, outcome_blind=True,
        modality_touch=[], sample_ids=sample_ids, scores=scores,
        provenance={"member_features": member_features} if member_features else {},
    )


def test_no_existing_signatures_is_novel():
    ids = [f"S{i}" for i in range(20)]
    candidate = _c("c", ids, {sid: 1.0 for sid in ids})
    result = novelty_check(candidate, [])
    assert result.overall_classification == "novel"
    assert result.comparisons == []


def test_highly_correlated_candidate_is_fully_redundant():
    rng = np.random.default_rng(0)
    ids = [f"S{i}" for i in range(100)]
    base = rng.normal(size=100)
    existing = _c("existing", ids, dict(zip(ids, base)))
    near_copy = _c("copy", ids, dict(zip(ids, base + rng.normal(scale=0.05, size=100))))

    result = novelty_check(near_copy, [existing])
    assert result.overall_classification == "fully_redundant"
    assert abs(result.comparisons[0].correlation) >= 0.8


def test_uncorrelated_and_no_feature_overlap_is_novel():
    rng = np.random.default_rng(1)
    ids = [f"S{i}" for i in range(100)]
    existing = _c("existing", ids, dict(zip(ids, rng.normal(size=100))), member_features=["a", "b", "c"])
    independent = _c("indep", ids, dict(zip(ids, rng.normal(size=100))), member_features=["x", "y", "z"])

    result = novelty_check(independent, [existing])
    assert result.overall_classification == "novel"


def test_identical_feature_sets_are_redundant_even_with_low_score_correlation():
    rng = np.random.default_rng(2)
    ids = [f"S{i}" for i in range(50)]
    existing = _c("existing", ids, dict(zip(ids, rng.normal(size=50))), member_features=["g1", "g2", "g3"])
    same_features = _c("same_feat", ids, dict(zip(ids, rng.normal(size=50))), member_features=["g1", "g2", "g3"])

    result = novelty_check(same_features, [existing])
    assert result.overall_classification == "fully_redundant"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
