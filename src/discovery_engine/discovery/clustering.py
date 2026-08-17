"""Outcome-blind hierarchical clustering discovery.

The classical baseline every other discovery method gets compared against.
Feature-feature correlation structure only, computed from a single
modality's DataFrame, k never chosen by looking at outcome performance:
every K in the requested grid is run and EVERY resulting module at EVERY K
becomes its own candidate. That's what lets campaign correction absorb
K-selection as its own layer of multiple comparisons (skill rule 7) instead
of it being a separate, unaccounted-for choice made before correction runs.

Zero-variance features are dropped before the correlation/distance matrix
is built, not after -- a real gap found against real data, not a
hypothetical: a Spearman correlation against a constant column is NaN,
which scipy's linkage() correctly refuses ("condensed distance matrix
must contain only finite values"), and this module had no guard for it
at all, unlike module_scores below, multimodal_input.py, nmf.py, and
interaction.py, which all already drop zero-variance features for exactly
this reason. Not a Hallmark-gene-set-specific quirk either: 292 of the
full 20,518-gene real RNA-seq matrix have zero variance across the first
real TCGA-BLCA cohort (e.g. TSSK2, simply unexpressed in bladder tissue),
so any real feature subset was always one unlucky draw away from hitting
this, not just the curated-pathway one that happened to surface it first.
louvain.py was checked too (it also computes Spearman correlation on raw
modality data) and confirmed safe without a fix: its edge-threshold
comparison (abs(r) >= min_abs_correlation) is False for NaN by construction,
so a constant feature just ends up correctly isolated with no edges,
never crashing -- verified directly, not assumed from the code shape.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from discovery_engine.campaign.candidate import Candidate
from discovery_engine.data.cohort import CohortData

logger = logging.getLogger(__name__)


class HierarchicalClusteringDiscovery:
    name = "hierarchical_clustering"

    def discover(
        self,
        data: CohortData,
        modality: str,
        k_values: list[int],
        linkage_method: str = "average",
        min_module_size: int = 2,
        feature_subset: str | None = None,
    ) -> list[Candidate]:
        df = data.modalities[modality]
        sample_ids = list(df.index)

        std = df.std(axis=0, ddof=0)
        usable = std[std > 0].index
        dropped = [f for f in df.columns if f not in usable]
        if dropped:
            logger.info(
                "hierarchical_clustering: dropping %d zero-variance feature(s) from modality %r: %s",
                len(dropped), modality, dropped[:10],
            )
        df = df[usable]
        feature_names = list(df.columns)

        corr = df.corr(method="spearman").to_numpy()
        distance = 1.0 - corr
        np.fill_diagonal(distance, 0.0)
        distance = np.clip(distance, 0.0, None)  # guard tiny negative floating-point noise
        condensed = squareform(distance, checks=False)
        z = linkage(condensed, method=linkage_method)

        candidates: list[Candidate] = []
        for k in k_values:
            labels = fcluster(z, t=k, criterion="maxclust")
            for cluster_id in sorted(set(labels)):
                members = [f for f, lab in zip(feature_names, labels) if lab == cluster_id]
                if len(members) < min_module_size:
                    continue
                scores = module_scores(df[members], sample_ids)
                candidates.append(
                    Candidate(
                        id=f"hierarchical_{modality}_k{k}_c{cluster_id}" + (f"_fs{feature_subset}" if feature_subset else ""),
                        method=self.name,
                        method_params={
                            "k": k,
                            "cluster_id": int(cluster_id),
                            "linkage": linkage_method,
                            "modality": modality,
                            "n_members": len(members),
                        },
                        outcome_blind=True,
                        modality_touch=[modality],
                        sample_ids=sample_ids,
                        scores=scores,
                        provenance={"member_features": members},
                    )
                )
        return candidates


def module_scores(member_df, sample_ids: list[str]) -> dict[str, float]:
    """Per-sample module score: mean z-score across member features.

    Constant features (std 0, would divide by zero) are dropped from the
    average rather than propagating NaN into the module score. Shared with
    louvain.py, which needs the identical feature-group-to-score conversion
    for its community-detection output; made public once a second method
    needed it (same pattern as match_units_across_seeds and
    build_multimodal_input_matrix).
    """
    mean = member_df.mean(axis=0)
    std = member_df.std(axis=0, ddof=0)
    usable = std[std > 0].index
    if len(usable) == 0:
        return {sid: 0.0 for sid in sample_ids}
    z = (member_df[usable] - mean[usable]) / std[usable]
    module_score = z.mean(axis=1)
    return {sid: float(module_score.loc[sid]) for sid in sample_ids}
