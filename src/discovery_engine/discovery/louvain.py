"""Louvain graph community detection: outcome-blind, like clustering.py,
built from the same feature-feature Spearman correlation structure, but a
graph-modularity method rather than agglomerative clustering.

Uses networkx's built-in louvain_communities (networkx.algorithms.community)
-- already an installed dependency, pulled in transitively by torch, so
this needs no new package. True Leiden (leidenalg + python-igraph) is a
deliberate, documented deferral, not a silent one: Leiden mainly fixes a
rare disconnected-community edge case Louvain has on pathological graphs;
not worth two new dependencies given the stability evidence below, but a
straightforward swap if ever needed.

Every resolution's every community becomes its own Candidate, exactly the
same nested-loop shape as clustering.py's k_values: no internal "pick the
best resolution by modularity score" step. Resolution-selection is exactly
K-selection's own layer of multiple comparisons (skill rule 7),
generalized to a graph-granularity knob instead of a cluster count, and
gets absorbed into campaign correction the identical way.

The graph is thresholded (min_abs_correlation), not a dense complete graph:
sparsification is standard practice for co-expression networks and keeps
this tractable at real gene-count scale, where a complete graph over
thousands of features would be untenable.

Deterministic vs stochastic, stated explicitly per discovery.base's
OutcomeBlindDiscoveryMethod docstring: Louvain's local-moving optimization
has real internal randomness (node-visitation order can affect which local
modularity optimum it converges to), unlike clustering.py's scipy.linkage
which has none. That randomness is NOT threaded through any refit (this
method has none -- outcome-blind candidates are re-scored directly against
permuted outcomes in campaign.correction, never re-discovered), so a plain
`seed: int` is enough; per-resolution seeds are derived from an internal
np.random.default_rng(seed) so different resolutions in one sweep don't
coincidentally share a Louvain seed.

This method is registered in validation.stability's
EMPIRICALLY_EXEMPTED_METHODS, not STABILITY_REQUIRED_METHODS. Measured,
not assumed: match_units_across_seeds run on 3 independently-seeded
Louvain runs (converting each run's discovered community to a per-sample
module score via clustering.module_scores, then correlating those score
vectors across seeds -- exactly the same technique used for every neural
method's stability check, just without any outcome involved) across four
configs:
    small graph (48 features), real planted module, resolution=1.0: |r| = 1.0
    small graph (48 features), NO signal (pure noise), resolution=1.0: |r| = 1.0
    small graph (48 features), real module, finer resolution=2.0:     |r| ~= 1.0
    larger graph (200 features, 145 edges), real module, res=1.0:     |r| ~= 1.0
The larger/denser graph was tested specifically because the literature
flags that regime, not small/sparse graphs, as Louvain's actual
instability risk -- stress-testing the failure mode that's plausible, not
just repeating favorable conditions. Every condition landed at or
essentially at perfect reproducibility, well past the 0.7 match threshold
used elsewhere in this project, so a per-candidate stability check would
add cost for no signal here.
"""

from __future__ import annotations

import networkx as nx
import numpy as np

from discovery_engine.campaign.candidate import Candidate
from discovery_engine.data.cohort import CohortData
from discovery_engine.discovery.clustering import module_scores


class LouvainDiscovery:
    name = "louvain"

    def discover(
        self,
        data: CohortData,
        modality: str,
        resolutions: list[float],
        min_abs_correlation: float = 0.2,
        min_module_size: int = 2,
        seed: int = 0,
        feature_subset: str | None = None,
    ) -> list[Candidate]:
        df = data.modalities[modality]
        sample_ids = list(df.index)
        graph = _build_correlation_graph(df, min_abs_correlation)

        rng = np.random.default_rng(seed)
        candidates: list[Candidate] = []
        for resolution in resolutions:
            louvain_seed = int(rng.integers(0, 2**31 - 1))
            communities = nx.algorithms.community.louvain_communities(
                graph, weight="weight", resolution=resolution, seed=louvain_seed
            )
            for community_idx, community in enumerate(communities):
                members = sorted(community)
                if len(members) < min_module_size:
                    continue
                scores = module_scores(df[members], sample_ids)
                candidates.append(
                    Candidate(
                        id=f"louvain_{modality}_res{resolution}_c{community_idx}" + (f"_fs{feature_subset}" if feature_subset else ""),
                        method=self.name,
                        method_params={
                            "resolution": resolution,
                            "community_idx": community_idx,
                            "modality": modality,
                            "min_abs_correlation": min_abs_correlation,
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


def _build_correlation_graph(df, min_abs_correlation: float) -> nx.Graph:
    corr = df.corr(method="spearman").to_numpy()
    features = list(df.columns)

    graph = nx.Graph()
    graph.add_nodes_from(features)
    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            r = corr[i, j]
            if abs(r) >= min_abs_correlation:
                graph.add_edge(features[i], features[j], weight=float(abs(r)))
    return graph
