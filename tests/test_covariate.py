import numpy as np
import pandas as pd
import pytest

from discovery_engine.campaign.candidate import Candidate
from discovery_engine.data.cohort import CohortData
from discovery_engine.outcomes import OutcomeSpec, OutcomeType
from discovery_engine.validation.covariate import covariate_stratified_test, freedman_lane_test


def _cohort_and_outcome(n=200, seed=0):
    rng = np.random.default_rng(seed)
    ids = [f"S{i}" for i in range(n)]
    grade = rng.integers(1, 4, size=n)  # 1, 2, 3 -- an irrelevant-to-signal covariate
    covariates = pd.DataFrame({"grade": grade}, index=ids)
    data = CohortData(modalities={}, covariates=covariates)
    return data, ids, rng


def test_real_signal_survives_covariate_stratification_when_covariate_is_irrelevant():
    data, ids, rng = _cohort_and_outcome(n=240)
    latent = rng.normal(size=240)
    y = 0.7 * latent + np.sqrt(1 - 0.7**2) * rng.normal(size=240)
    scores = dict(zip(ids, latent + rng.normal(scale=0.2, size=240)))
    outcome = OutcomeSpec(OutcomeType.CONTINUOUS, ids, y)
    candidate = Candidate(
        id="c", method="m", method_params={}, outcome_blind=True,
        modality_touch=[], sample_ids=ids, scores=scores,
    )

    result = covariate_stratified_test(candidate, data, outcome, "grade", n_permutations=300, seed=1)
    assert result.full_cohort_p_value < 0.05
    significant_strata = [s for s in result.strata if not np.isnan(s.p_value) and s.p_value < 0.1]
    assert len(significant_strata) >= 2, "a real signal should survive in most strata of an irrelevant covariate"


def test_freedman_lane_confound_fully_explained_by_covariate_is_not_significant():
    """If the 'candidate' is literally a noisy copy of the covariate itself,
    Freedman-Lane adjustment for that covariate should wash out the effect.
    """
    data, ids, rng = _cohort_and_outcome(n=200)
    grade = data.covariates["grade"].to_numpy(dtype=float)
    y = grade + rng.normal(scale=0.5, size=200)
    scores = dict(zip(ids, grade + rng.normal(scale=0.1, size=200)))
    outcome = OutcomeSpec(OutcomeType.CONTINUOUS, ids, y)
    candidate = Candidate(
        id="confound", method="m", method_params={}, outcome_blind=True,
        modality_touch=[], sample_ids=ids, scores=scores,
    )

    result = freedman_lane_test(candidate, data, outcome, ["grade"], n_permutations=300, seed=2)
    assert result.p_value > 0.05


def test_freedman_lane_independent_signal_remains_significant_after_adjustment():
    data, ids, rng = _cohort_and_outcome(n=200)
    latent = rng.normal(size=200)  # independent of grade
    y = 0.7 * latent + np.sqrt(1 - 0.7**2) * rng.normal(size=200)
    scores = dict(zip(ids, latent + rng.normal(scale=0.2, size=200)))
    outcome = OutcomeSpec(OutcomeType.CONTINUOUS, ids, y)
    candidate = Candidate(
        id="real", method="m", method_params={}, outcome_blind=True,
        modality_touch=[], sample_ids=ids, scores=scores,
    )

    result = freedman_lane_test(candidate, data, outcome, ["grade"], n_permutations=300, seed=3)
    assert result.p_value < 0.05


def test_freedman_lane_raises_clear_error_not_linalgerror_when_too_few_covariate_complete_patients():
    """Insurance against exactly the failure mode discovered building the
    first real TCGA-BLCA loader: a NaN in the covariate design matrix makes
    np.linalg.lstsq raise a generic LinAlgError ("SVD did not converge"),
    confirmed directly before this fix existed. The pre-filter must turn
    that into a specific, actionable ValueError instead.
    """
    n = 10
    ids = [f"S{i}" for i in range(n)]
    rng = np.random.default_rng(5)
    grade = np.array([1.0] * 3 + [np.nan] * 7)  # only 3 covariate-complete, below the 5+1=6 minimum
    covariates = pd.DataFrame({"grade": grade}, index=ids)
    data = CohortData(modalities={}, covariates=covariates)
    y = rng.normal(size=n)
    scores = dict(zip(ids, rng.normal(size=n)))
    outcome = OutcomeSpec(OutcomeType.CONTINUOUS, ids, y)
    candidate = Candidate(
        id="c", method="m", method_params={}, outcome_blind=True,
        modality_touch=[], sample_ids=ids, scores=scores,
    )

    with pytest.raises(ValueError, match="complete .* coverage"):
        freedman_lane_test(candidate, data, outcome, ["grade"], n_permutations=50, seed=0)


def test_freedman_lane_runs_cleanly_with_a_small_number_of_missing_covariates():
    """Mirrors the real TCGA-BLCA N=367 case (2/367 missing GRADE) at the
    same ratio: a small number of NaN covariates among plenty of complete
    ones should be silently dropped (logged) and the test should run
    cleanly, not raise at all.
    """
    n = 367
    ids = [f"S{i}" for i in range(n)]
    rng = np.random.default_rng(6)
    grade = rng.integers(0, 2, size=n).astype(float)
    grade[[10, 200]] = np.nan  # exactly 2 missing, matching the real cohort's ratio
    covariates = pd.DataFrame({"grade_high": grade}, index=ids)
    data = CohortData(modalities={}, covariates=covariates)
    latent = rng.normal(size=n)
    y = 0.6 * latent + np.sqrt(1 - 0.36) * rng.normal(size=n)
    scores = dict(zip(ids, latent + rng.normal(scale=0.3, size=n)))
    outcome = OutcomeSpec(OutcomeType.CONTINUOUS, ids, y)
    candidate = Candidate(
        id="c", method="m", method_params={}, outcome_blind=True,
        modality_touch=[], sample_ids=ids, scores=scores,
    )

    result = freedman_lane_test(candidate, data, outcome, ["grade_high"], n_permutations=200, seed=7)
    assert 0.0 <= result.p_value <= 1.0
    assert not np.isnan(result.observed_partial_statistic)


def test_freedman_lane_raises_not_implemented_for_survival():
    data, ids, rng = _cohort_and_outcome(n=100)
    time = rng.exponential(size=100)
    event = rng.integers(0, 2, size=100).astype(float)
    outcome = OutcomeSpec(OutcomeType.SURVIVAL, ids, np.column_stack([time, event]))
    scores = dict(zip(ids, rng.normal(size=100)))
    candidate = Candidate(
        id="c", method="m", method_params={}, outcome_blind=True,
        modality_touch=[], sample_ids=ids, scores=scores,
    )
    with pytest.raises(NotImplementedError):
        freedman_lane_test(candidate, data, outcome, ["grade"], n_permutations=50, seed=0)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
