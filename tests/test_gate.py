import numpy as np
import pytest

from discovery_engine.campaign.candidate import Candidate
from discovery_engine.data.cohort import CohortData
from discovery_engine.outcomes import OutcomeSpec, OutcomeType
from discovery_engine.validation.gate import run_validation_gate


def test_bootstrap_n_resamples_is_independently_configurable_from_n_permutations():
    """Real-data motivated: bootstrap CI's cost is disproportionate to every
    other gate sub-check at real Cox-fit scale (see gate.py's module
    docstring), so it needs its own knob, not just gate_n_permutations
    controlling everything. Confirms the threading actually works, not just
    that the parameter exists.
    """
    n = 60
    rng = np.random.default_rng(0)
    ids = [f"S{i}" for i in range(n)]
    scores = dict(zip(ids, rng.normal(size=n)))
    outcome = OutcomeSpec(OutcomeType.CONTINUOUS, ids, rng.normal(size=n))
    data = CohortData(modalities={"m": __import__("pandas").DataFrame({"f": rng.normal(size=n)}, index=ids)})
    candidate = Candidate(
        id="c", method="hierarchical_clustering", method_params={}, outcome_blind=True,
        modality_touch=["m"], sample_ids=ids, scores=scores, provenance={"member_features": ["f"]},
    )

    result = run_validation_gate(
        candidate, data, outcome, existing_signatures=[],
        n_permutations=20, bootstrap_n_resamples=17, seed=0,
    )
    assert result.bootstrap_ci.n_bootstrap == 17
    assert result.bootstrap_ci.ci_low <= result.bootstrap_ci.ci_high

    default_result = run_validation_gate(
        candidate, data, outcome, existing_signatures=[], n_permutations=20, seed=0,
    )
    assert default_result.bootstrap_ci.n_bootstrap == 1000, "default must stay 1000 for existing callers"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))


def _survival_setup(n=60, seed=0):
    import pandas as pd
    rng = np.random.default_rng(seed)
    ids = [f"S{i}" for i in range(n)]
    x = rng.normal(size=n)
    time = rng.exponential(scale=np.exp(-0.8 * x)) * 10 + 0.1
    event = (rng.random(n) < 0.7).astype(float)
    outcome = OutcomeSpec(OutcomeType.SURVIVAL, ids, np.column_stack([time, event]))
    data = CohortData(
        modalities={"m": pd.DataFrame({"f": x}, index=ids)},
        covariates=pd.DataFrame({"grade": rng.integers(0, 2, size=n).astype(float)}, index=ids),
    )
    candidate = Candidate(
        id="c", method="hierarchical_clustering", method_params={}, outcome_blind=True,
        modality_touch=["m"], sample_ids=ids, scores=dict(zip(ids, x)), provenance={"member_features": ["f"]},
    )
    return candidate, data, outcome


def test_survival_freedman_lane_skip_is_recorded_not_silently_passed():
    candidate, data, outcome = _survival_setup()
    result = run_validation_gate(
        candidate, data, outcome, existing_signatures=[], covariate_col="grade",
        n_permutations=20, bootstrap_n_resamples=20, seed=0,
    )
    assert result.freedman_lane is None
    assert any(s.startswith("freedman_lane: not implemented for survival") for s in result.skipped_checks)


def test_missing_known_signatures_is_recorded_as_skipped():
    candidate, data, outcome = _survival_setup()
    result = run_validation_gate(
        candidate, data, outcome, existing_signatures=[], n_permutations=20, bootstrap_n_resamples=20, seed=0,
    )
    assert any(s.startswith("adds_value_and_novelty") for s in result.skipped_checks)
    assert any(s.startswith("freedman_lane: no covariate") for s in result.skipped_checks)


def test_advisory_checks_are_named_and_never_fail_a_candidate_alone():
    candidate, data, outcome = _survival_setup()
    result = run_validation_gate(
        candidate, data, outcome, existing_signatures=[], n_permutations=20, bootstrap_n_resamples=20, seed=0,
    )
    assert set(result.advisory_checks) == {"bootstrap_ci", "covariate_stratified", "design_crosscheck"}
    for reason in result.failure_reasons:
        assert "bootstrap" not in reason and "stratified" not in reason and "extreme" not in reason
