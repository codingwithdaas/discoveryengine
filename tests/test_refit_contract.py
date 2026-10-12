import numpy as np

from discovery_engine.campaign.candidate import Candidate
from discovery_engine.campaign.correction import (
    RefitDidNotChangeError,
    naive_within_method_bh_fdr,
    run_campaign_correction,
)
from discovery_engine.data.cohort import CohortData
from discovery_engine.outcomes import OutcomeSpec, OutcomeType

_SAMPLE_IDS = [f"S{i}" for i in range(30)]
_EMPTY_DATA = CohortData(modalities={})


def _real_outcome() -> OutcomeSpec:
    rng = np.random.default_rng(42)
    return OutcomeSpec(OutcomeType.CONTINUOUS, _SAMPLE_IDS, rng.normal(size=len(_SAMPLE_IDS)))


def _genuine_refit(data: CohortData, outcome: OutcomeSpec, rng: np.random.Generator) -> Candidate:
    """A refit that actually uses both the (permuted) outcome and the passed
    rng to derive scores, the way a real RSF/DeepSurv/attention refit would
    depend on whatever labels and random init it's handed.
    """
    scores = {
        sid: float(v) + 0.01 + 0.001 * float(rng.normal())
        for sid, v in zip(outcome.sample_ids, outcome.values)
    }
    return Candidate(
        id="genuine",
        method="method_genuine",
        method_params={},
        outcome_blind=False,
        modality_touch=[],
        sample_ids=outcome.sample_ids,
        scores=scores,
    )


def _broken_refit_ignores_outcome(fixed_scores: dict):
    def _refit(data: CohortData, outcome: OutcomeSpec, rng: np.random.Generator) -> Candidate:
        return Candidate(
            id="broken",
            method="method_broken",
            method_params={},
            outcome_blind=False,
            modality_touch=[],
            sample_ids=outcome.sample_ids,
            scores=dict(fixed_scores),
        )

    return _refit


def test_genuine_refit_scores_differ_from_original_and_across_permutations():
    outcome = _real_outcome()
    original = _genuine_refit(_EMPTY_DATA, outcome, np.random.default_rng(0))

    rng = np.random.default_rng(1)
    permuted_a = outcome.permute(rng)
    permuted_b = outcome.permute(rng)

    result_a = _genuine_refit(_EMPTY_DATA, permuted_a, np.random.default_rng(2))
    result_b = _genuine_refit(_EMPTY_DATA, permuted_b, np.random.default_rng(3))

    assert result_a.scores != original.scores
    assert result_b.scores != original.scores
    assert result_a.scores != result_b.scores, "refit output should change with a different permutation"


def test_broken_refit_that_ignores_permuted_outcome_raises():
    outcome = _real_outcome()
    fixed_scores = {sid: float(v) for sid, v in zip(outcome.sample_ids, outcome.values)}

    candidate = Candidate(
        id="broken",
        method="method_broken",
        method_params={},
        outcome_blind=False,
        modality_touch=[],
        sample_ids=outcome.sample_ids,
        scores=dict(fixed_scores),
        refit=_broken_refit_ignores_outcome(fixed_scores),
    )

    try:
        run_campaign_correction([candidate], _EMPTY_DATA, outcome, n_permutations=5, seed=0, allow_low_permutations=True)
        assert False, "expected RefitDidNotChangeError"
    except RefitDidNotChangeError as e:
        assert "broken" in str(e)
        assert "permutation index" in str(e)


def test_broken_refit_raises_in_naive_baseline_too():
    outcome = _real_outcome()
    fixed_scores = {sid: float(v) for sid, v in zip(outcome.sample_ids, outcome.values)}

    candidate = Candidate(
        id="broken",
        method="method_broken",
        method_params={},
        outcome_blind=False,
        modality_touch=[],
        sample_ids=outcome.sample_ids,
        scores=dict(fixed_scores),
        refit=_broken_refit_ignores_outcome(fixed_scores),
    )

    try:
        naive_within_method_bh_fdr([candidate], _EMPTY_DATA, outcome, n_permutations=5, seed=0, allow_low_permutations=True)
        assert False, "expected RefitDidNotChangeError"
    except RefitDidNotChangeError:
        pass


def test_genuine_refit_passes_through_correction_without_raising():
    outcome = _real_outcome()
    candidate = Candidate(
        id="genuine",
        method="method_genuine",
        method_params={},
        outcome_blind=False,
        modality_touch=[],
        sample_ids=outcome.sample_ids,
        scores=_genuine_refit(_EMPTY_DATA, outcome, np.random.default_rng(0)).scores,
        refit=_genuine_refit,
    )

    results = run_campaign_correction([candidate], _EMPTY_DATA, outcome, n_permutations=10, seed=0, allow_low_permutations=True)
    assert "genuine" in results


def test_run_campaign_correction_is_bit_reproducible_with_fixed_seed():
    """The whole point of threading rng through refit(): a fixed campaign
    seed must give bit-identical results across two full reruns, including
    for outcome-informed candidates whose refit draws its own randomness.
    """
    outcome = _real_outcome()
    candidate = Candidate(
        id="genuine",
        method="method_genuine",
        method_params={},
        outcome_blind=False,
        modality_touch=[],
        sample_ids=outcome.sample_ids,
        scores=_genuine_refit(_EMPTY_DATA, outcome, np.random.default_rng(0)).scores,
        refit=_genuine_refit,
    )

    results_a = run_campaign_correction([candidate], _EMPTY_DATA, outcome, n_permutations=25, seed=7, allow_low_permutations=True)
    results_b = run_campaign_correction([candidate], _EMPTY_DATA, outcome, n_permutations=25, seed=7, allow_low_permutations=True)

    assert results_a["genuine"].observed_statistic == results_b["genuine"].observed_statistic
    assert results_a["genuine"].studentized_statistic == results_b["genuine"].studentized_statistic
    assert results_a["genuine"].campaign_p_value == results_b["genuine"].campaign_p_value


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-q"]))
