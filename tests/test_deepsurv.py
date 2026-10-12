import numpy as np
import pandas as pd
import pytest
import torch

from discovery_engine.campaign.candidate import Candidate
from discovery_engine.campaign.correction import RefitDidNotChangeError, run_campaign_correction
from discovery_engine.campaign.registry import CandidateRegistry
from discovery_engine.data.cohort import CohortData
from discovery_engine.discovery.deepsurv import DeepSurvDiscovery, _cox_loss
from discovery_engine.outcomes import OutcomeSpec, OutcomeType
from discovery_engine.validation.stability import TrainedRun, match_units_across_seeds, stability_check


def _planted_survival_cohort(n=150, seed=0, beta=2.0):
    rng = np.random.default_rng(seed)
    sample_ids = [f"S{i:04d}" for i in range(n)]
    latent = rng.normal(size=n)
    rna = pd.DataFrame(
        {f"gene_{i}": latent + rng.normal(scale=0.2, size=n) for i in range(10)}, index=sample_ids
    )
    protein = pd.DataFrame(
        {f"prot_{i}": latent + rng.normal(scale=0.2, size=n) for i in range(10)}, index=sample_ids
    )
    data = CohortData(modalities={"rnaseq": rna, "protein": protein})

    baseline_rate = 0.1
    hazard = baseline_rate * np.exp(beta * latent)
    true_time = rng.exponential(1.0 / hazard)
    censor_time = rng.exponential(scale=1.0 / baseline_rate * 3)
    time = np.minimum(true_time, censor_time)
    event = (true_time <= censor_time).astype(float)
    outcome = OutcomeSpec(OutcomeType.SURVIVAL, sample_ids, np.column_stack([time, event]))
    return data, outcome


def test_cox_loss_prefers_risk_ordering_that_matches_event_times():
    """Sanity check on the loss implementation itself: risk scores that
    correctly rank "who fails first" should get a lower (better) Cox loss
    than risk scores with the opposite ranking, on the same event times.
    """
    time = torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0])
    event = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0])
    # correctly-ordered: highest risk for earliest time (time=1 -> risk=5)
    good_risk = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    bad_risk = torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0])

    good_loss = _cox_loss(good_risk, time, event).item()
    bad_loss = _cox_loss(bad_risk, time, event).item()
    assert good_loss < bad_loss


def test_cox_loss_returns_nan_when_batch_has_zero_events():
    time = torch.tensor([1.0, 2.0, 3.0])
    event = torch.tensor([0.0, 0.0, 0.0])
    risk = torch.tensor([0.1, 0.2, 0.3])
    assert torch.isnan(_cox_loss(risk, time, event))


def test_discover_is_bit_reproducible_given_same_seed():
    data, outcome = _planted_survival_cohort(n=80, seed=2)
    disc = DeepSurvDiscovery()
    kwargs = dict(
        modalities=["rnaseq", "protein"], hidden_dims_options=[[8]],
        n_seeds=2, max_epochs=15, patience=4,
    )
    c1 = disc.discover(data, outcome, np.random.default_rng(7), **kwargs)
    c2 = disc.discover(data, outcome, np.random.default_rng(7), **kwargs)
    assert len(c1) == len(c2) == 1
    assert c1[0].scores == c2[0].scores
    assert c1[0].provenance["stability"] == c2[0].provenance["stability"]


def test_discover_recovers_planted_survival_signal_with_stable_config():
    data, outcome = _planted_survival_cohort(n=150, seed=1, beta=2.0)
    disc = DeepSurvDiscovery()
    candidates = disc.discover(
        data, outcome, np.random.default_rng(3),
        modalities=["rnaseq", "protein"], hidden_dims_options=[[8]],
        n_seeds=2, max_epochs=60, patience=10,
    )
    assert len(candidates) == 1
    stability = candidates[0].provenance["stability"]
    assert stability["n_seeds_matched"] == stability["n_seeds_total"]
    assert stability["min_abs_correlation_across_seeds"] > 0.7


def test_deepsurv_candidate_is_outcome_informed_with_working_refit():
    data, outcome = _planted_survival_cohort(n=80, seed=4)
    disc = DeepSurvDiscovery()
    candidates = disc.discover(
        data, outcome, np.random.default_rng(5),
        modalities=["rnaseq", "protein"], hidden_dims_options=[[6]],
        n_seeds=1, max_epochs=15, patience=4,
    )
    candidate = candidates[0]
    assert candidate.outcome_blind is False
    assert candidate.refit is not None

    registry = CandidateRegistry()
    registry.add(candidate)  # would raise if refit were missing
    assert len(registry) == 1


def test_refit_reruns_from_scratch_and_carries_no_stability_key():
    data, outcome = _planted_survival_cohort(n=80, seed=6)
    disc = DeepSurvDiscovery()
    candidate = disc.discover(
        data, outcome, np.random.default_rng(9),
        modalities=["rnaseq", "protein"], hidden_dims_options=[[6]],
        n_seeds=1, max_epochs=15, patience=4,
    )[0]

    permuted = outcome.permute(np.random.default_rng(11))
    refit_result = candidate.refit(data, permuted, np.random.default_rng(12))

    assert refit_result.scores != candidate.scores
    assert "stability" not in refit_result.provenance


def test_deepsurv_partial_seed_match_produces_weak_calibration_and_passes():
    """Closes the loop explicitly for DeepSurv's single-unit (n_samples, 1)
    risk-score shape specifically, running the real match_units_across_seeds
    function (not a hand-typed provenance dict) through stability_check:
    1-of-2 other seeds matching must land in "weak", distinct from the
    matched-all "robust" and matched-none "noise" cases, and must still
    pass the gate. This is the same underlying function the autoencoder's
    partial-match test already covers, but this confirms the (n, 1) shape
    specifically produces the identical behavior, not inferred from reuse.
    """
    n = 200
    ramp = np.arange(n, dtype=float).reshape(-1, 1)
    ref = TrainedRun(latent_scores=ramp)
    run1 = TrainedRun(latent_scores=ramp.copy())  # matches
    rng = np.random.default_rng(21)
    run2 = TrainedRun(latent_scores=rng.normal(size=(n, 1)))  # doesn't match

    matches = match_units_across_seeds([ref, run1, run2], threshold=0.7)
    assert len(matches) == 1
    stability_info = matches[0]
    assert stability_info["n_seeds_total"] == 2
    assert stability_info["n_seeds_matched"] == 1

    candidate = Candidate(
        id="deepsurv_partial_match_test", method="deepsurv", method_params={}, outcome_blind=False,
        modality_touch=["m1"], sample_ids=[f"S{i}" for i in range(n)],
        scores={f"S{i}": 0.0 for i in range(n)},
        provenance={"stability": stability_info},
    )
    result = stability_check(candidate)
    assert result.applicable is True
    assert result.stability_fraction == pytest.approx(0.5)
    assert result.calibration == "weak"
    assert result.calibration not in ("robust", "noise")
    assert result.passed is True


def test_deepsurv_candidate_survives_real_campaign_correction_without_raising():
    """The actual proof: an outcome-informed NEURAL candidate flowing
    through campaign.correction's real permutation loop -- refit gets
    called once per permutation, must never raise RefitDidNotChangeError,
    and must produce a valid CorrectionResult. Small n_permutations to
    keep this test's wall-clock cost bounded; the mechanism being tested
    doesn't depend on how many permutations are run.
    """
    data, outcome = _planted_survival_cohort(n=60, seed=8, beta=2.0)
    disc = DeepSurvDiscovery()
    candidate = disc.discover(
        data, outcome, np.random.default_rng(13),
        modalities=["rnaseq", "protein"], hidden_dims_options=[[6]],
        n_seeds=1, max_epochs=15, patience=4,
    )[0]

    results = run_campaign_correction([candidate], data, outcome, n_permutations=5, seed=0, allow_low_permutations=True)
    assert candidate.id in results
    result = results[candidate.id]
    assert 0.0 < result.campaign_p_value <= 1.0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
