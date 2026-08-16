import pytest

from discovery_engine.campaign.candidate import Candidate
from discovery_engine.validation.stability import requires_stability_check, stability_check


def _candidate(method: str, stability: dict | None = None) -> Candidate:
    return Candidate(
        id="c1", method=method, method_params={}, outcome_blind=True,
        modality_touch=["m1"], sample_ids=["S0", "S1", "S2"],
        scores={"S0": 0.1, "S1": 0.2, "S2": 0.3},
        provenance={"stability": stability} if stability is not None else {},
    )


def test_classical_method_not_applicable():
    result = stability_check(_candidate("hierarchical_clustering"))
    assert result.applicable is False


def test_neural_candidate_missing_provenance_raises():
    assert requires_stability_check("autoencoder") is True
    with pytest.raises(ValueError):
        stability_check(_candidate("autoencoder"))


def test_neural_candidate_matched_all_other_seeds_is_robust():
    result = stability_check(_candidate("autoencoder", {
        "n_seeds_total": 2, "n_seeds_matched": 2,
        "matched_seed_fractions": [0.95, 0.88],
        "min_abs_correlation_across_seeds": 0.88,
    }))
    assert result.applicable is True
    assert result.calibration == "robust"
    assert result.passed is True


def test_neural_candidate_matched_no_other_seeds_is_noise_and_fails():
    result = stability_check(_candidate("autoencoder", {
        "n_seeds_total": 2, "n_seeds_matched": 0,
        "matched_seed_fractions": [0.1, 0.2],
        "min_abs_correlation_across_seeds": 0.1,
    }))
    assert result.calibration == "noise"
    assert result.passed is False


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
