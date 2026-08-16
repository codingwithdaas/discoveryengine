import numpy as np
import pytest

from discovery_engine.campaign.candidate import Candidate
from discovery_engine.outcomes import OutcomeSpec, OutcomeType
from discovery_engine.validation.design_crosscheck import median_split_vs_extreme_groups


def _extremes_only_effect_scenario():
    """An effect that only exists in the top/bottom ~8% of the score
    distribution; the middle 84% of samples carry no relationship to
    outcome at all. A median split mixes each half with mostly-unrelated
    middle samples and loses power; extreme-groups isolates the true
    effect cleanly.
    """
    n = 150
    rng = np.random.default_rng(11)
    ids = [f"S{i}" for i in range(n)]
    score = rng.normal(size=n)
    lo_thresh, hi_thresh = np.percentile(score, [8, 92])
    y = rng.normal(scale=3.0, size=n)
    y = np.where(score >= hi_thresh, y + 3.0, y)
    y = np.where(score <= lo_thresh, y - 3.0, y)
    scores = dict(zip(ids, score))
    outcome = OutcomeSpec(OutcomeType.CONTINUOUS, ids, y)
    candidate = Candidate(
        id="c", method="m", method_params={}, outcome_blind=True,
        modality_touch=[], sample_ids=ids, scores=scores,
    )
    return candidate, outcome


def test_extreme_groups_detects_effect_that_median_split_misses():
    candidate, outcome = _extremes_only_effect_scenario()
    result = median_split_vs_extreme_groups(
        candidate, outcome, extreme_fraction=0.08, n_permutations=300, seed=1
    )
    assert result.median_split.p_value > 0.05
    assert result.extreme_groups.p_value < 0.05
    assert result.disagreement is True


def test_confound_check_populated_on_disagreement_and_rules_out_an_unrelated_confound():
    candidate, outcome = _extremes_only_effect_scenario()
    rng = np.random.default_rng(99)
    unrelated_confound = dict(zip(outcome.sample_ids, rng.normal(size=len(outcome.sample_ids))))

    result = median_split_vs_extreme_groups(
        candidate, outcome, extreme_fraction=0.08, n_permutations=300, seed=1, confound=unrelated_confound
    )
    assert result.confound_median_split is not None
    assert result.confound_extreme_groups is not None
    # An unrelated confound should NOT reproduce the extreme-groups significance,
    # which is evidence the candidate's own extreme-groups signal isn't just
    # riding on this particular confound.
    assert result.confound_extreme_groups.p_value > 0.05


def test_confound_check_not_populated_when_designs_agree():
    n = 300
    rng = np.random.default_rng(3)
    ids = [f"S{i}" for i in range(n)]
    latent = rng.normal(size=n)
    y = 0.8 * latent + np.sqrt(1 - 0.8**2) * rng.normal(size=n)
    scores = dict(zip(ids, latent + rng.normal(scale=0.2, size=n)))
    outcome = OutcomeSpec(OutcomeType.CONTINUOUS, ids, y)
    candidate = Candidate(
        id="c", method="m", method_params={}, outcome_blind=True,
        modality_touch=[], sample_ids=ids, scores=scores,
    )
    confound = dict(zip(ids, rng.normal(size=n)))

    result = median_split_vs_extreme_groups(
        candidate, outcome, extreme_fraction=0.25, n_permutations=300, seed=0, confound=confound
    )
    assert result.disagreement is False
    assert result.confound_median_split is None
    assert result.confound_extreme_groups is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
