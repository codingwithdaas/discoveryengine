"""Guards added after the real-data campaign review:

1. Minimum permutation count. Every acceptance in the real TCGA-BLCA
   campaigns came from runs with 50 permutations, where each candidate's
   null SD (used for studentization) was underestimated. Rerun at 10,000
   permutations, none survived. Correction now refuses fewer than
   MIN_CAMPAIGN_PERMUTATIONS unless explicitly allowed (tests, canaries).
2. Duplicate candidate ids. Results are keyed by id, so a duplicate would
   silently overwrite another candidate's result. This happened once on real
   data (feature-subset id collision); correction now fails loudly instead.
"""
import logging

import numpy as np
import pytest

from discovery_engine.campaign.campaign import run_campaign
from discovery_engine.campaign.candidate import Candidate
from discovery_engine.campaign.correction import (
    MIN_CAMPAIGN_PERMUTATIONS,
    DuplicateCandidateIdError,
    TooFewPermutationsError,
    check_permutation_count,
    naive_within_method_bh_fdr,
    run_campaign_correction,
)
from discovery_engine.data.synthetic import make_synthetic_candidate_pool, make_synthetic_cohort
from discovery_engine.outcomes import OutcomeType
from discovery_engine.signatures.registry import SignatureRegistry


def _pool():
    cohort, outcome, real_feature = make_synthetic_cohort(
        n_samples=60, n_noise_features=5, effect_size=0.6, outcome_type=OutcomeType.CONTINUOUS, seed=0
    )
    candidates = make_synthetic_candidate_pool(cohort, real_feature, n_methods=2, n_noise_candidates_per_method=2, seed=1)
    return cohort, outcome, candidates


def test_minimum_is_at_least_1000():
    assert MIN_CAMPAIGN_PERMUTATIONS >= 1000


def test_correction_refuses_too_few_permutations_by_default():
    cohort, outcome, candidates = _pool()
    with pytest.raises(TooFewPermutationsError, match="below the campaign minimum"):
        run_campaign_correction(candidates, cohort, outcome, n_permutations=50, seed=0)


def test_naive_baseline_refuses_too_few_permutations_by_default():
    cohort, outcome, candidates = _pool()
    with pytest.raises(TooFewPermutationsError):
        naive_within_method_bh_fdr(candidates, cohort, outcome, n_permutations=50, seed=0)


def test_run_campaign_refuses_too_few_permutations_by_default(tmp_path):
    cohort, outcome, candidates = _pool()
    registry = SignatureRegistry(tmp_path / "sig.db")
    with pytest.raises(TooFewPermutationsError):
        run_campaign(
            campaign_id="c", cohort_id="k", candidates=candidates, data=cohort, outcome=outcome,
            signature_registry=registry, correction_n_permutations=50, gate_n_permutations=5,
        )


def test_explicit_opt_out_runs_and_warns(caplog):
    cohort, outcome, candidates = _pool()
    with caplog.at_level(logging.WARNING, logger="discovery_engine.campaign.correction"):
        results = run_campaign_correction(
            candidates, cohort, outcome, n_permutations=20, seed=0, allow_low_permutations=True
        )
    assert len(results) == len(candidates)
    assert any("do not report these p-values" in r.message for r in caplog.records)


def test_minimum_or_more_passes_the_check():
    check_permutation_count(MIN_CAMPAIGN_PERMUTATIONS, allow_low_permutations=False)
    check_permutation_count(MIN_CAMPAIGN_PERMUTATIONS * 10, allow_low_permutations=False)


def test_duplicate_candidate_ids_fail_loudly_before_any_permutation():
    cohort, outcome, candidates = _pool()
    a = candidates[0]
    clone = Candidate(
        id=a.id, method=a.method, method_params={}, outcome_blind=True, modality_touch=a.modality_touch,
        sample_ids=a.sample_ids, scores={k: float(v) + 1.0 for k, v in a.scores.items()},
    )
    with pytest.raises(DuplicateCandidateIdError, match=a.id):
        run_campaign_correction(candidates + [clone], cohort, outcome, n_permutations=20, seed=0,
                                allow_low_permutations=True)


def test_unique_ids_are_not_flagged():
    cohort, outcome, candidates = _pool()
    assert len({c.id for c in candidates}) == len(candidates)
    run_campaign_correction(candidates, cohort, outcome, n_permutations=10, seed=0, allow_low_permutations=True)
