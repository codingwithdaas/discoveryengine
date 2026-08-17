"""Random Survival Forest discovery: outcome-informed like DeepSurv, but
classical (a tree ensemble, not a neural network). Built second, on purpose,
to test whether the refit/permutation contract from Phase 1 generalizes to a
model that isn't torch-shaped -- confirmed: OutcomeInformedDiscoveryMethod
needed no changes, and refit here is meaningfully simpler than DeepSurv's
(one .fit() call, no train/val split, no early-stopping loop, since RSF
isn't iteratively gradient-trained).

Candidate = the forest's aggregate predicted risk score for one
hyperparameter config (n_estimators/max_depth/min_samples_leaf), exactly
parallel to DeepSurv's one-config-one-candidate shape. This is a deliberate
departure from an earlier draft (correction.py's Phase 1 docstring said "RSF
importances"): using per-feature importance-ranking to NOMINATE which
features become candidates has a real double-dipping problem, since the
true candidates would be selected by ranking against the TRUE outcome, but
a valid permutation null would need that SELECTION step re-run under every
permutation too, not just a refit of one fixed feature's association --
Candidate.refit isn't shaped for "rerun a selection procedure and return
whichever feature comes out on top this time". Sidestepped entirely by
using the forest's own risk score as the candidate, the same way DeepSurv
does. (scikit-survival's RandomSurvivalForest also doesn't implement
feature_importances_ at all -- confirmed via NotImplementedError, not
assumed -- so this design has no fallback dependency on it. Per-feature
interpretability for an accepted RSF candidate uses the same generic,
method-agnostic correlation_interpretability_report every other candidate
type already uses; no RSF-specific interpretability code needed.)

No stability check, and this is now an empirically checked decision, not
just the variance-reduction argument on its own. match_units_across_seeds
was run on 3 independently-random_state-seeded RandomSurvivalForest fits
(same config, same true outcome) for the smallest-n_estimators configs in
the planned sweep grid, including a pure-noise (no planted signal) cohort
as the harder case:
    n_estimators=25, real signal:  cross-seed |r| = 0.986, 0.992
    n_estimators=10, real signal:  cross-seed |r| = 0.973, 0.966
    n_estimators=10, NO signal:    cross-seed |r| = 0.739, 0.713
    n_estimators=25, NO signal:    cross-seed |r| = 0.870, 0.847
Every measured correlation cleared the 0.7 match threshold used elsewhere
in this project, including the deliberately adversarial small-forest/
no-signal case, so a per-candidate multi-seed stability check would do
nothing but add cost here: the ensemble's internal averaging (hundreds, or
even just tens, of trees) already suppresses the run-to-run variance that
a single neural network's init-sensitivity requires stability checking to
catch. This method is registered in validation.stability's
EMPIRICALLY_EXEMPTED_METHODS with the numbers above (not
STABILITY_REQUIRED_METHODS -- see that module's docstring for the general
three-category framework this decision fits into).
"""

from __future__ import annotations

import numpy as np
from sksurv.ensemble import RandomSurvivalForest

from discovery_engine.campaign.candidate import Candidate
from discovery_engine.data.cohort import CohortData
from discovery_engine.discovery.multimodal_input import build_multimodal_input_matrix
from discovery_engine.outcomes import OutcomeSpec, OutcomeType


class RandomSurvivalForestDiscovery:
    name = "random_survival_forest"

    def discover(
        self,
        data: CohortData,
        outcome: OutcomeSpec,
        rng: np.random.Generator,
        modalities: list[str],
        forest_configs: list[dict],
        feature_subset: str | None = None,
    ) -> list[Candidate]:
        if outcome.type is not OutcomeType.SURVIVAL:
            raise ValueError("RandomSurvivalForestDiscovery requires a SURVIVAL outcome")

        x, sample_ids, feature_names, modality_info = build_multimodal_input_matrix(data, modalities)
        sub_outcome = outcome.subset(sample_ids)
        y = _to_sksurv_labels(sub_outcome)

        candidates: list[Candidate] = []
        for config in forest_configs:
            random_state = int(rng.integers(0, 2**31 - 1))
            rsf = RandomSurvivalForest(random_state=random_state, **config)
            rsf.fit(x, y)
            risk = rsf.predict(x)

            scores = {sid: float(risk[i]) for i, sid in enumerate(sample_ids)}
            candidates.append(
                Candidate(
                    id=f"random_survival_forest_{'_'.join(modalities)}_{_config_id(config)}" + (f"_fs{feature_subset}" if feature_subset else ""),
                    method=self.name,
                    method_params={"modalities": list(modalities), **config},
                    outcome_blind=False,
                    modality_touch=list(modalities),
                    sample_ids=sample_ids,
                    scores=scores,
                    provenance={"feature_names": feature_names, "modality_info": modality_info},
                    refit=_make_refit(modalities, config),
                )
            )
        return candidates


def _make_refit(modalities: list[str], config: dict):
    def _refit(data: CohortData, outcome: OutcomeSpec, rng: np.random.Generator) -> Candidate:
        if outcome.type is not OutcomeType.SURVIVAL:
            raise ValueError("RandomSurvivalForest refit requires a SURVIVAL outcome")
        x, sample_ids, _, _ = build_multimodal_input_matrix(data, modalities)
        sub_outcome = outcome.subset(sample_ids)
        y = _to_sksurv_labels(sub_outcome)

        random_state = int(rng.integers(0, 2**31 - 1))
        rsf = RandomSurvivalForest(random_state=random_state, **config)
        rsf.fit(x, y)
        risk = rsf.predict(x)

        scores = {sid: float(risk[i]) for i, sid in enumerate(sample_ids)}
        return Candidate(
            id="random_survival_forest_refit",
            method="random_survival_forest",
            method_params={"modalities": list(modalities), **config},
            outcome_blind=False,
            modality_touch=list(modalities),
            sample_ids=sample_ids,
            scores=scores,
            provenance={},
        )

    return _refit


def _to_sksurv_labels(outcome: OutcomeSpec) -> np.ndarray:
    time = outcome.values[:, 0]
    event = outcome.values[:, 1].astype(bool)
    return np.array(list(zip(event, time)), dtype=[("event", bool), ("time", float)])


def _config_id(config: dict) -> str:
    return "_".join(f"{k}{v}" for k, v in sorted(config.items()))
