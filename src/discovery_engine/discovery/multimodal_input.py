"""Shared multi-modal input construction for neural discovery methods.

Moved out of autoencoder.py once deepsurv.py needed the identical logic:
z-score each modality, then divide by sqrt(n_features) so a high-dimensional
modality (e.g. ~20k-gene RNA-seq) can't drown a low-dimensional one (e.g. a
~200-feature copy-number block) purely by having more columns. Samples not
present in every requested modality are dropped and logged, never silently
included with imputed/zero values.
"""

from __future__ import annotations

import logging

import numpy as np

from discovery_engine.data.cohort import CohortData

logger = logging.getLogger(__name__)


def build_multimodal_input_matrix(
    data: CohortData, modalities: list[str]
) -> tuple[np.ndarray, list[str], list[str], dict]:
    dfs = [data.modalities[m] for m in modalities]
    common_ids = sorted(set.intersection(*(set(df.index) for df in dfs)))
    if len(common_ids) < 10:
        raise ValueError(f"only {len(common_ids)} samples shared across modalities {modalities}, too few to train")

    all_ids: set[str] = set()
    for df in dfs:
        all_ids.update(df.index)
    dropped = sorted(all_ids - set(common_ids))
    if dropped:
        logger.info(
            "multi-modal input: %d sample(s) dropped, not present in every requested "
            "modality %s: %s", len(dropped), modalities, dropped[:10],
        )

    blocks: list[np.ndarray] = []
    feature_names: list[str] = []
    modality_info: dict = {}
    for m, df in zip(modalities, dfs):
        sub = df.loc[common_ids]
        mean = sub.mean(axis=0)
        std = sub.std(axis=0, ddof=0)
        usable = std[std > 0].index
        z = (sub[usable] - mean[usable]) / std[usable]
        z = z / np.sqrt(max(1, len(usable)))
        blocks.append(z.to_numpy())
        feature_names.extend(f"{m}:{f}" for f in usable)
        modality_info[m] = {
            "n_features_used": len(usable),
            "n_features_dropped_zero_variance": len(sub.columns) - len(usable),
        }

    x = np.concatenate(blocks, axis=1).astype(np.float32)
    return x, common_ids, feature_names, modality_info
