import numpy as np
import pandas as pd
import pytest

from discovery_engine.campaign.candidate import Candidate
from discovery_engine.data.cohort import CohortData
from discovery_engine.validation.interpretability import correlation_interpretability_report


def test_top_features_ranked_by_absolute_correlation():
    n = 100
    rng = np.random.default_rng(0)
    ids = [f"S{i}" for i in range(n)]
    latent = rng.normal(size=n)

    df = pd.DataFrame(
        {
            "strong_pos": latent + rng.normal(scale=0.1, size=n),
            "strong_neg": -latent + rng.normal(scale=0.1, size=n),
            "weak": latent + rng.normal(scale=3.0, size=n),
            "noise_1": rng.normal(size=n),
            "noise_2": rng.normal(size=n),
        },
        index=ids,
    )
    data = CohortData(modalities={"rnaseq": df})
    candidate = Candidate(
        id="c", method="m", method_params={}, outcome_blind=True,
        modality_touch=["rnaseq"], sample_ids=ids, scores=dict(zip(ids, latent)),
    )

    report = correlation_interpretability_report(candidate, data, top_n=3)
    assert report.method == "correlation"
    top_names = [a.feature for a in report.top_features]
    assert set(top_names[:2]) == {"strong_pos", "strong_neg"}
    assert abs(report.top_features[0].correlation) > 0.8


def test_constant_feature_is_dropped_not_nan():
    n = 50
    ids = [f"S{i}" for i in range(n)]
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"constant": [5.0] * n, "varies": rng.normal(size=n)}, index=ids)
    data = CohortData(modalities={"m": df})
    candidate = Candidate(
        id="c", method="m", method_params={}, outcome_blind=True,
        modality_touch=["m"], sample_ids=ids, scores=dict(zip(ids, rng.normal(size=n))),
    )
    report = correlation_interpretability_report(candidate, data)
    names = [a.feature for a in report.top_features]
    assert "constant" not in names
    assert all(np.isfinite(a.correlation) for a in report.top_features)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
