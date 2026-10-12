import numpy as np

from discovery_engine.campaign.candidate import Candidate
from discovery_engine.campaign.correction import run_campaign_correction
from discovery_engine.data.synthetic import make_synthetic_candidate_pool, make_synthetic_cohort
from discovery_engine.outcomes import OutcomeType


def _refit_random_noise(data, outcome, rng):
    scores = dict(zip(outcome.sample_ids, rng.normal(size=len(outcome.sample_ids))))
    return Candidate(
        id="refit_noise",
        method="method_refit",
        method_params={},
        outcome_blind=False,
        modality_touch=[],
        sample_ids=outcome.sample_ids,
        scores=scores,
    )


def test_campaign_correction_recovers_planted_signal_and_suppresses_noise():
    cohort, outcome, real_feature = make_synthetic_cohort(
        n_samples=150, n_noise_features=30, effect_size=0.6, outcome_type=OutcomeType.CONTINUOUS, seed=0
    )
    candidates = make_synthetic_candidate_pool(cohort, real_feature, n_methods=4, n_noise_candidates_per_method=8, seed=1)

    refit_candidate = Candidate(
        id="refit_noise",
        method="method_refit",
        method_params={},
        outcome_blind=False,
        modality_touch=[],
        sample_ids=outcome.sample_ids,
        scores=dict(zip(outcome.sample_ids, np.random.default_rng(3).normal(size=len(outcome.sample_ids)))),
        refit=_refit_random_noise,
    )
    candidates.append(refit_candidate)

    results = run_campaign_correction(candidates, cohort, outcome, n_permutations=500, seed=2, allow_low_permutations=True)

    assert len(results) == len(candidates)

    real_p = results["real_signal"].campaign_p_value
    assert real_p < 0.05

    all_p = {cid: r.campaign_p_value for cid, r in results.items()}
    assert min(all_p, key=all_p.get) == "real_signal"

    noise_p_values = [r.campaign_p_value for cid, r in results.items() if cid != "real_signal"]
    false_positives = sum(1 for p in noise_p_values if p < 0.05)
    assert false_positives <= 3, f"too many noise candidates falsely significant: {false_positives}/{len(noise_p_values)}"


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-q"]))
