"""Feature attribution report: no candidate is reported as a finding without
one. Correlation-with-the-candidate-score, per original feature, by
modality, is the baseline attribution every method supports (SHAP and
attention-weight-based attribution get added alongside the methods that
produce them, in later phases; a black-box score with no attribution is a
lead, not a finding).

Spearman (rank) correlation, matching this project's existing convention of
non-parametric, distribution-agnostic statistics; vectorized via rank
transform + covariance rather than a per-feature Python loop, since real
modalities here (RNA-seq etc) can have thousands of features.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import rankdata

from discovery_engine.campaign.candidate import Candidate
from discovery_engine.data.cohort import CohortData


@dataclass
class FeatureAttribution:
    feature: str
    modality: str
    correlation: float


@dataclass
class InterpretabilityReport:
    method: str
    top_features: list[FeatureAttribution]


def correlation_interpretability_report(
    candidate: Candidate,
    data: CohortData,
    top_n: int = 15,
) -> InterpretabilityReport:
    attributions: list[FeatureAttribution] = []
    for modality in candidate.modality_touch:
        df = data.modalities[modality]
        common_ids = [sid for sid in df.index if sid in candidate.scores]
        if len(common_ids) < 3:
            continue
        sub = df.loc[common_ids]
        y = np.array([candidate.scores[sid] for sid in common_ids], dtype=float)
        attributions.extend(_ranked_correlations(sub, y, modality))

    attributions.sort(key=lambda a: abs(a.correlation), reverse=True)
    return InterpretabilityReport(method="correlation", top_features=attributions[:top_n])


def _ranked_correlations(sub, y: np.ndarray, modality: str) -> list[FeatureAttribution]:
    x = sub.to_numpy(dtype=float)
    x_ranks = np.apply_along_axis(rankdata, 0, x)
    y_ranks = rankdata(y)

    x_centered = x_ranks - x_ranks.mean(axis=0)
    y_centered = y_ranks - y_ranks.mean()

    numerator = x_centered.T @ y_centered
    x_ss = (x_centered**2).sum(axis=0)
    y_ss = (y_centered**2).sum()
    denominator = np.sqrt(x_ss * y_ss)

    with np.errstate(invalid="ignore", divide="ignore"):
        correlations = numerator / denominator

    return [
        FeatureAttribution(feature, modality, float(corr))
        for feature, corr in zip(sub.columns, correlations)
        if np.isfinite(corr)
    ]
