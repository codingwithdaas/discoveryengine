from discovery_engine.campaign.correction import naive_within_method_bh_fdr, run_campaign_correction
from discovery_engine.data.synthetic import make_synthetic_candidate_pool, make_synthetic_cohort
from discovery_engine.outcomes import OutcomeType


def test_campaign_correction_is_more_conservative_than_naive_within_method_pooling():
    """The central architectural claim: correcting across the FULL campaign
    pool must be at least as conservative as, and typically more conservative
    than, applying BH-FDR separately within each method and pooling the
    results. Many small per-method pools each under-correct relative to the
    true campaign-wide number of shots on goal.
    """
    cohort, outcome, real_feature = make_synthetic_cohort(
        n_samples=150, n_noise_features=30, effect_size=0.6, outcome_type=OutcomeType.CONTINUOUS, seed=0
    )
    candidates = make_synthetic_candidate_pool(
        cohort, real_feature, n_methods=6, n_noise_candidates_per_method=10, seed=1
    )

    campaign_results = run_campaign_correction(candidates, cohort, outcome, n_permutations=300, seed=2)
    naive_q = naive_within_method_bh_fdr(candidates, cohort, outcome, n_permutations=300, seed=2)

    noise_ids = [c.id for c in candidates if c.id != "real_signal"]
    campaign_p = [campaign_results[cid].campaign_p_value for cid in noise_ids]
    naive = [naive_q[cid] for cid in noise_ids]

    campaign_false_positives = sum(1 for p in campaign_p if p < 0.05)
    naive_false_positives = sum(1 for q in naive if q < 0.05)
    assert campaign_false_positives <= naive_false_positives, (
        f"campaign correction called MORE noise candidates significant ({campaign_false_positives}) "
        f"than naive within-method BH-FDR ({naive_false_positives}); it should never be less conservative"
    )

    mean_campaign_p = sum(campaign_p) / len(campaign_p)
    mean_naive_q = sum(naive) / len(naive)
    assert mean_campaign_p >= mean_naive_q, (
        f"expected campaign-corrected p-values (mean={mean_campaign_p:.3f}) to run at least as "
        f"conservative on average as naive within-method q-values (mean={mean_naive_q:.3f})"
    )

    # The real signal must still survive full campaign-wide correction.
    assert campaign_results["real_signal"].campaign_p_value < 0.05


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-q"]))
