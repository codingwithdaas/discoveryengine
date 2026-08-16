"""NMF (non-negative matrix factorization) discovery: outcome-blind, like
clustering.py and louvain.py, but the first classical method placed in
validation.stability.STABILITY_REQUIRED_METHODS rather than
EMPIRICALLY_EXEMPTED_METHODS -- on evidence, not architecture.

The empirical check (run before this module existed, per this project's
established practice of measuring before deciding, not assuming): the
literal "smallest rank" reading came back clean (min |r| = 1.0 at
n_components=2, random init, both real signal and pure noise). Pushed
further to the actual "underdetermined" regime -- rank approaching
min(n_samples, n_features), more free parameters than the data can
uniquely pin down, the standard linear-algebra meaning, not just the
smallest number in a sweep grid -- and found real instability there: at
n_components=20 on a 150-sample/64-feature cohort, most components matched
fine but some did not (real signal: one component with n_seeds_matched=0,
min |r|=0.54; pure noise: two components with n_seeds_matched=0, min
|r| as low as 0.24, several more at partial match). Mixed, not clean, so
this method carries real per-candidate stability wiring rather than an
exemption -- the same standard applied to autoencoder.py, deepsurv.py, and
attention.py, reusing the identical match_units_across_seeds mechanism on
the W matrix (samples x components), which is exactly AE's shape.

Two NMF-specific design points this raised, both resolved here rather than
left implicit:

1. NMF requires non-negative input (X >= 0), which the project's standard
   z-score input construction (multimodal_input.build_multimodal_input_matrix,
   used by every neural method so far) violates outright -- z-scored
   features are centered at 0 and go negative for below-average samples.
   This module does NOT reuse that function. Input is min-max scaled to
   [0, 1] per feature instead, which is non-negative by construction
   regardless of the raw modality's original sign or scale, and is the
   standard, principled choice for NMF specifically (not an arbitrary
   shift-and-hope). Single-modality for now (`modality: str`), matching
   the classical-method precedent (clustering.py, louvain.py), not the
   multi-modal shape of the neural methods -- NMF's classic use case
   (metagene-style decomposition of one expression matrix) doesn't need
   multi-modal concatenation the way a shared-latent-space neural method
   does; flag if multi-modal NMF is ever wanted.

2. The stability check is only a meaningful check if production discover()
   actually uses init="random". sklearn's nndsvd-family inits are
   deterministic: every "different seed" would produce the IDENTICAL
   decomposition, so a multi-seed check against a deterministic init would
   silently stop testing anything real while still reporting perfect
   "stability" -- exactly the failure mode the measured instability above
   makes clearly real, not hypothetical. Guarded by
   test_production_init_is_genuinely_stochastic_not_silently_deterministic,
   which calls the actual _train_one_seed function (not a mock) with two
   different seeds and asserts the resulting component matrices are NOT
   bit-identical; that test fails immediately if init is ever silently
   changed to a deterministic scheme.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import NMF

from discovery_engine.campaign.candidate import Candidate
from discovery_engine.data.cohort import CohortData
from discovery_engine.validation.stability import TrainedRun, match_units_across_seeds


@dataclass
class _NMFRun:
    W: np.ndarray  # (n_samples, n_components)
    H: np.ndarray  # (n_components, n_features)


class NMFDiscovery:
    name = "nmf"

    def discover(
        self,
        data: CohortData,
        modality: str,
        n_components_options: list[int],
        n_seeds: int = 3,
        max_iter: int = 300,
        stability_threshold: float = 0.7,
        seed: int = 0,
    ) -> list[Candidate]:
        df = data.modalities[modality]
        sample_ids = list(df.index)
        x, feature_names = _min_max_scale(df)

        rng = np.random.default_rng(seed)
        candidates: list[Candidate] = []
        for n_components in n_components_options:
            runs = [
                _train_one_seed(x, n_components, max_iter, np.random.default_rng(rng.integers(0, 2**63 - 1)))
                for _ in range(n_seeds)
            ]
            matches = match_units_across_seeds([TrainedRun(latent_scores=r.W) for r in runs], stability_threshold)
            ref = runs[0]

            for unit_idx in range(n_components):
                scores = {sid: float(ref.W[i, unit_idx]) for i, sid in enumerate(sample_ids)}
                loadings = {fname: float(ref.H[unit_idx, j]) for j, fname in enumerate(feature_names)}
                candidates.append(
                    Candidate(
                        id=f"nmf_{modality}_r{n_components}_c{unit_idx}",
                        method=self.name,
                        method_params={
                            "n_components": n_components, "component_idx": unit_idx,
                            "modality": modality, "n_seeds": n_seeds,
                        },
                        outcome_blind=True,
                        modality_touch=[modality],
                        sample_ids=sample_ids,
                        scores=scores,
                        provenance={
                            "feature_names": feature_names,
                            "component_loadings": loadings,
                            "stability": matches[unit_idx],
                        },
                    )
                )
        return candidates


def _min_max_scale(df) -> tuple[np.ndarray, list[str]]:
    mn = df.min(axis=0)
    mx = df.max(axis=0)
    span = mx - mn
    usable = span[span > 0].index
    scaled = (df[usable] - mn[usable]) / span[usable]
    return scaled.to_numpy().astype(np.float64), list(usable)


def _train_one_seed(x: np.ndarray, n_components: int, max_iter: int, rng: np.random.Generator) -> _NMFRun:
    random_state = int(rng.integers(0, 2**31 - 1))
    nmf = NMF(n_components=n_components, init="random", random_state=random_state, max_iter=max_iter)
    w = nmf.fit_transform(x)
    return _NMFRun(W=w, H=nmf.components_)
