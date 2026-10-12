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

Stability (revised after real data): this method is now in
validation.stability's STABILITY_REQUIRED_METHODS. It was previously
exempted on synthetic evidence (3 seeds, four configs including a
200-feature graph, cross-seed |r| ~1.0 in every case). On the real
TCGA-BLCA variance-top-1000 RNA-seq graph that did not hold: the same
luminal community came out with 289 to 378 member genes depending on the
seed, and the community partition shifted. So every candidate now carries
provenance["stability"] from n_seeds independent Louvain runs, matched by
module-score correlation (match_units_across_seeds, the same technique
used for NMF and the neural methods), plus the best member-set Jaccard
overlap against each other seed, so membership drift is visible even when
the module scores stay correlated.

The first run at each resolution uses the same seed derivation as before,
so existing candidates (ids, members, scores) are unchanged; the extra
stability runs draw their seeds from a separate generator.
"""

from __future__ import annotations

import networkx as nx
import numpy as np

from discovery_engine.campaign.candidate import Candidate
from discovery_engine.data.cohort import CohortData
from discovery_engine.discovery.clustering import module_scores
from discovery_engine.validation.stability import TrainedRun, match_units_across_seeds


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
        n_seeds: int = 3,
        stability_threshold: float = 0.7,
    ) -> list[Candidate]:
        if n_seeds < 2:
            raise ValueError("n_seeds must be at least 2 so every candidate gets a cross-seed stability check")
        df = data.modalities[modality]
        sample_ids = list(df.index)
        graph = _build_correlation_graph(df, min_abs_correlation)

        rng = np.random.default_rng(seed)
        stability_rng = np.random.default_rng([seed, 7919])
        candidates: list[Candidate] = []
        for resolution in resolutions:
            louvain_seed = int(rng.integers(0, 2**31 - 1))
            extra_seeds = [int(stability_rng.integers(0, 2**31 - 1)) for _ in range(n_seeds - 1)]
            runs = [
                _run_louvain(graph, resolution, s, min_module_size)
                for s in [louvain_seed] + extra_seeds
            ]
            communities = runs[0]["all"]
            ref_kept = runs[0]["kept"]
            score_runs = [
                TrainedRun(latent_scores=_score_matrix(df, run["kept"], sample_ids)) for run in runs
            ]
            matches = match_units_across_seeds(score_runs, stability_threshold) if ref_kept else []
            match_by_idx = dict(zip([idx for idx, _ in ref_kept], matches))
            for community_idx, community in enumerate(communities):
                members = sorted(community)
                if len(members) < min_module_size:
                    continue
                stability = dict(match_by_idx[community_idx])
                stability["best_member_jaccard_per_other_seed"] = [
                    max((_jaccard(set(members), set(other)) for _, other in run["kept"]), default=0.0)
                    for run in runs[1:]
                ]
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
                        provenance={"member_features": members, "stability": stability},
                    )
                )
        return candidates


def _run_louvain(graph: nx.Graph, resolution: float, louvain_seed: int, min_module_size: int) -> dict:
    communities = nx.algorithms.community.louvain_communities(
        graph, weight="weight", resolution=resolution, seed=louvain_seed
    )
    kept = [(idx, sorted(c)) for idx, c in enumerate(communities) if len(c) >= min_module_size]
    return {"all": communities, "kept": kept}


def _score_matrix(df, kept: list[tuple[int, list[str]]], sample_ids: list[str]) -> np.ndarray:
    if not kept:
        return np.zeros((len(sample_ids), 0))
    cols = [module_scores(df[members], sample_ids) for _, members in kept]
    return np.column_stack([[col[sid] for sid in sample_ids] for col in cols])


def _jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if (a or b) else 0.0


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
