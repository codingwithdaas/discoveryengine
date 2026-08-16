"""Outcome-blind hierarchical clustering discovery.

The classical baseline every other discovery method gets compared against.
Feature-feature correlation structure only, computed from a single
modality's DataFrame, k never chosen by looking at outcome performance:
every K in the requested grid is run and EVERY resulting module at EVERY K
becomes its own candidate. That's what lets campaign correction absorb
K-selection as its own layer of multiple comparisons (skill rule 7) instead
of it being a separate, unaccounted-for choice made before correction runs.
"""

from __future__ import annotations

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from discovery_engine.campaign.candidate import Candidate
from discovery_engine.data.cohort import CohortData


class HierarchicalClusteringDiscovery:
    name = "hierarchical_clustering"

    def discover(
        self,
        data: CohortData,
        modality: str,
        k_values: list[int],
        linkage_method: str = "average",
        min_module_size: int = 2,
    ) -> list[Candidate]:
        df = data.modalities[modality]
        sample_ids = list(df.index)
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
                        id=f"hierarchical_{modality}_k{k}_c{cluster_id}",
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
