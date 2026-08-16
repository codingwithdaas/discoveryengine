"""Synthetic cohorts and candidate pools with known ground truth.

Used to prove the campaign registry / correction framework works before any
real discovery method exists: we know exactly which candidate is real signal
and which are noise, so the pipeline's output can be checked against the
answer instead of just "looking plausible".
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from discovery_engine.campaign.candidate import Candidate
from discovery_engine.data.cohort import CohortData
from discovery_engine.outcomes import OutcomeSpec, OutcomeType


def make_synthetic_cohort(
    n_samples: int = 120,
    n_noise_features: int = 50,
    effect_size: float = 0.6,
    outcome_type: OutcomeType = OutcomeType.CONTINUOUS,
    seed: int = 0,
) -> tuple[CohortData, OutcomeSpec, str]:
    """One modality with one real (outcome-correlated) feature and many
    pure-noise features. Returns (cohort, outcome, real_feature_name).
    """
    rng = np.random.default_rng(seed)
    sample_ids = [f"S{i:04d}" for i in range(n_samples)]

    latent = rng.normal(size=n_samples)
    real_feature = latent + rng.normal(scale=1.0, size=n_samples)

    noise = rng.normal(size=(n_samples, n_noise_features))
    columns = ["real_feature"] + [f"noise_{i}" for i in range(n_noise_features)]
    matrix = np.column_stack([real_feature, noise])
    df = pd.DataFrame(matrix, index=sample_ids, columns=columns)
    cohort = CohortData(modalities={"synthetic_modality": df})

    outcome = _make_outcome(sample_ids, latent, effect_size, outcome_type, rng)
    return cohort, outcome, "real_feature"


def _make_outcome(
    sample_ids: list[str],
    latent: np.ndarray,
    effect_size: float,
    outcome_type: OutcomeType,
    rng: np.random.Generator,
) -> OutcomeSpec:
    n = len(sample_ids)
    residual_scale = float(np.sqrt(max(1e-6, 1 - effect_size**2)))
    driven = effect_size * latent + residual_scale * rng.normal(size=n)

    if outcome_type is OutcomeType.CONTINUOUS:
        return OutcomeSpec(OutcomeType.CONTINUOUS, sample_ids, driven)

    if outcome_type is OutcomeType.BINARY:
        binary = (driven > np.median(driven)).astype(float)
        return OutcomeSpec(OutcomeType.BINARY, sample_ids, binary)

    if outcome_type is OutcomeType.SURVIVAL:
        # Higher `driven` -> shorter time-to-event (higher risk), a standard
        # simulated proportional-hazards setup.
        risk = np.exp(driven)
        time = rng.exponential(scale=1.0 / risk)
        censor_time = rng.exponential(scale=np.median(time) * 1.5)
        event = (time <= censor_time).astype(float)
        observed_time = np.minimum(time, censor_time)
        return OutcomeSpec(OutcomeType.SURVIVAL, sample_ids, np.column_stack([observed_time, event]))

    raise AssertionError(f"unhandled outcome type {outcome_type}")


def make_synthetic_candidate_pool(
    cohort: CohortData,
    real_feature_name: str,
    n_methods: int = 5,
    n_noise_candidates_per_method: int = 8,
    seed: int = 1,
) -> list[Candidate]:
    """A campaign-scale candidate pool: one real candidate plus many noise
    candidates spread across several simulated "discovery methods", mimicking
    e.g. 5 methods each contributing several latent dims / modules. All
    outcome-blind, since their scores are fixed feature/random values that
    never read the outcome.
    """
    rng = np.random.default_rng(seed)
    df = cohort.modalities["synthetic_modality"]
    sample_ids = list(df.index)

    candidates = [
        Candidate(
            id="real_signal",
            method="method_0_real",
            method_params={},
            outcome_blind=True,
            modality_touch=["synthetic_modality"],
            sample_ids=sample_ids,
            scores=df[real_feature_name].to_dict(),
        )
    ]

    for m in range(n_methods):
        for k in range(n_noise_candidates_per_method):
            noise_scores = dict(zip(sample_ids, rng.normal(size=len(sample_ids))))
            candidates.append(
                Candidate(
                    id=f"method_{m}_noise_{k}",
                    method=f"method_{m}",
                    method_params={"k": k},
                    outcome_blind=True,
                    modality_touch=["synthetic_modality"],
                    sample_ids=sample_ids,
                    scores=noise_scores,
                )
            )

    return candidates
