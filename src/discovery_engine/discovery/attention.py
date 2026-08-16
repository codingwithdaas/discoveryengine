"""Attention-based feature-attribution discovery: neural AND outcome-
informed like DeepSurv (both the Phase 3 stability contract and the Phase 1
refit contract apply, reusing the identical orthogonal-loops design
DeepSurv established -- see deepsurv.py's module docstring for that
reasoning, not repeated here).

What a Candidate is: one (d_model, n_heads) config's single CLS-pooled
prediction, exactly DeepSurv's shape, not one candidate per attention head
or per feature. Per-feature/per-head candidates would repeat RSF's
"importances as candidates" mistake (see rsf.py): features/heads would be
implicitly selected by how they look against the TRUE outcome, and a valid
permutation null would need that selection re-run every permutation, which
Candidate.refit isn't shaped for. One aggregate prediction sidesteps it,
same as DeepSurv and RSF.

Architecture: each feature is a token (its scalar value linearly projected,
plus a learned per-feature identity embedding so the network can tell
features apart, not just react to magnitude). A single learnable CLS query
token cross-attends over the feature tokens (deliberately NOT a full
multi-layer self-attention block among all tokens -- that would make
"attention weight per feature" ambiguous to interpret, and this data
regime, small-to-moderate cohorts, doesn't obviously benefit from the
extra capacity anyway). The CLS token's pooled output feeds a linear
prediction head. Generalized across all three outcome types (binary/
continuous/survival), not survival-only like DeepSurv, reusing
deepsurv._cox_loss directly for the survival case rather than
reimplementing it.

Interpretability (skill rule 4) IS this method's native signal, not a
correlation report layered on afterward: the CLS token's attention weights
over feature tokens, averaged over heads (torch's average_attn_weights=True)
and over samples, are stored directly in provenance["attention_weights"].
Scope boundary, stated explicitly rather than left implicit: Phase 4 (this
module) guarantees that provenance exists and is well-formed for every
attention candidate -- that's what makes rule 4 structurally satisfiable.
Phase 5 (reporting layer, already planned) decides how to surface
attention_weights PREFERENTIALLY over the generic correlation-based report
for this method type. Nothing is silently missing in the meantime: the
existing generic correlation_interpretability_report still runs against
any attention candidate's .scores and still satisfies the gate's pass/fail
check today; what Phase 5 adds is which signal a human report shows as the
primary explanation, not whether the gate enforces one exists.

A prediction can be reproducible across independent retrains while its
EXPLANATION isn't: attention weights could converge on the same output via
different features each time, a documented failure mode in attention-
interpretability research generally, not a hypothetical invented for this
project. So attention-weight reproducibility gets its own, separate
measurement, not folded into prediction stability: the same
match_units_across_seeds function used for provenance["stability"] (score
reproducibility) is reused, unchanged, on the F-length per-feature
attention vectors from the same n_seeds runs -- it's shape-agnostic, a
per-feature weight vector is just another kind of "unit" to correlate
across seeds. Stored in provenance["attention_weight_stability"], kept
separate from provenance["stability"] (which stays scored on prediction
only). Deliberately NOT gated pass/fail here: whether/how to fail a
candidate whose prediction is stable but whose explanation isn't, and how
to phrase that in a report (rule 12's language discipline: "consistently
identified" vs "identified in this fit, reproducibility across retrains
not confirmed"), is a Phase 5 decision. This is a stated deferral, not an
omission, the same way DeepSurv's stability-vs-refit boundary was spelled
out rather than assumed.

Refit candidates carry neither provenance["stability"] nor
provenance["attention_weight_stability"]: a permutation-null retrain is a
single run with nothing to cross-seed-match against, and (per DeepSurv's
established design) is never itself stability-checked or gate-checked --
confirmed structurally in campaign.correction._null_statistics, which
feeds refit output straight into association_statistic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from discovery_engine.campaign.candidate import Candidate
from discovery_engine.data.cohort import CohortData
from discovery_engine.discovery.deepsurv import _cox_loss
from discovery_engine.discovery.multimodal_input import build_multimodal_input_matrix
from discovery_engine.outcomes import OutcomeSpec, OutcomeType
from discovery_engine.validation.stability import TrainedRun, match_units_across_seeds


class _AttentionNet(nn.Module):
    def __init__(self, n_features: int, d_model: int, n_heads: int):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")
        self.value_proj = nn.Linear(1, d_model)
        self.feature_identity = nn.Parameter(torch.randn(n_features, d_model) * 0.02)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch = x.shape[0]
        tokens = self.value_proj(x.unsqueeze(-1)) + self.feature_identity.unsqueeze(0)  # (batch, n_features, d_model)
        cls = self.cls_token.expand(batch, -1, -1)  # (batch, 1, d_model)
        pooled, attn_weights = self.attn(cls, tokens, tokens, need_weights=True, average_attn_weights=True)
        pred = self.head(pooled.squeeze(1)).squeeze(-1)  # (batch,)
        return pred, attn_weights.squeeze(1)  # attn_weights: (batch, n_features)


def _loss(pred: torch.Tensor, outcome_type: OutcomeType, y: torch.Tensor) -> torch.Tensor:
    if outcome_type is OutcomeType.SURVIVAL:
        return _cox_loss(pred, y[:, 0], y[:, 1])
    if outcome_type is OutcomeType.BINARY:
        return nn.functional.binary_cross_entropy_with_logits(pred, y)
    return nn.functional.mse_loss(pred, y)


@dataclass
class _AttentionRun:
    predictions: np.ndarray  # (n_samples, 1)
    attention_weights: np.ndarray  # (n_features, 1), mean over heads and samples


class AttentionDiscovery:
    name = "attention"

    def discover(
        self,
        data: CohortData,
        outcome: OutcomeSpec,
        rng: np.random.Generator,
        modalities: list[str],
        head_configs: list[dict],
        n_seeds: int = 3,
        val_fraction: float = 0.2,
        max_epochs: int = 200,
        patience: int = 15,
        stability_threshold: float = 0.7,
    ) -> list[Candidate]:
        x, sample_ids, feature_names, modality_info = build_multimodal_input_matrix(data, modalities)
        sub_outcome = outcome.subset(sample_ids)
        y = sub_outcome.values

        candidates: list[Candidate] = []
        for config in head_configs:
            d_model, n_heads = config["d_model"], config["n_heads"]
            runs = [
                _train_one_seed(
                    x, outcome.type, y, d_model, n_heads, val_fraction, max_epochs, patience,
                    np.random.default_rng(rng.integers(0, 2**63 - 1)),
                )
                for _ in range(n_seeds)
            ]
            score_matches = match_units_across_seeds(
                [TrainedRun(latent_scores=r.predictions) for r in runs], stability_threshold
            )[0]
            attn_matches = match_units_across_seeds(
                [TrainedRun(latent_scores=r.attention_weights) for r in runs], stability_threshold
            )[0]

            ref = runs[0]
            scores = {sid: float(ref.predictions[i, 0]) for i, sid in enumerate(sample_ids)}
            attention_weights = {fname: float(ref.attention_weights[i, 0]) for i, fname in enumerate(feature_names)}

            candidates.append(
                Candidate(
                    id=f"attention_{'_'.join(modalities)}_d{d_model}_h{n_heads}",
                    method=self.name,
                    method_params={
                        "modalities": list(modalities), "d_model": d_model, "n_heads": n_heads, "n_seeds": n_seeds,
                    },
                    outcome_blind=False,
                    modality_touch=list(modalities),
                    sample_ids=sample_ids,
                    scores=scores,
                    provenance={
                        "feature_names": feature_names,
                        "modality_info": modality_info,
                        "stability": score_matches,
                        "attention_weights": attention_weights,
                        "attention_weight_stability": attn_matches,
                    },
                    refit=_make_refit(modalities, outcome.type, d_model, n_heads, val_fraction, max_epochs, patience),
                )
            )
        return candidates


def _make_refit(
    modalities: list[str], outcome_type: OutcomeType, d_model: int, n_heads: int,
    val_fraction: float, max_epochs: int, patience: int,
):
    def _refit(data: CohortData, outcome: OutcomeSpec, rng: np.random.Generator) -> Candidate:
        if outcome.type is not outcome_type:
            raise ValueError(f"attention refit requires outcome type {outcome_type}, got {outcome.type}")
        x, sample_ids, _, _ = build_multimodal_input_matrix(data, modalities)
        sub_outcome = outcome.subset(sample_ids)
        y = sub_outcome.values

        # One retrain, same training procedure as discovery/stability, on
        # the permuted outcome. Deliberately no "stability" or
        # "attention_weight_stability" provenance: see module docstring.
        run = _train_one_seed(x, outcome_type, y, d_model, n_heads, val_fraction, max_epochs, patience, rng)
        scores = {sid: float(run.predictions[i, 0]) for i, sid in enumerate(sample_ids)}
        return Candidate(
            id="attention_refit",
            method="attention",
            method_params={"modalities": list(modalities), "d_model": d_model, "n_heads": n_heads},
            outcome_blind=False,
            modality_touch=list(modalities),
            sample_ids=sample_ids,
            scores=scores,
            provenance={},
        )

    return _refit


def _train_one_seed(
    x: np.ndarray,
    outcome_type: OutcomeType,
    y: np.ndarray,
    d_model: int,
    n_heads: int,
    val_fraction: float,
    max_epochs: int,
    patience: int,
    rng: np.random.Generator,
) -> _AttentionRun:
    torch.manual_seed(int(rng.integers(0, 2**31 - 1)))

    n, n_features = x.shape
    perm = rng.permutation(n)
    n_val = max(1, int(n * val_fraction))
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    x_t = torch.from_numpy(x)
    y_t = torch.from_numpy(y.astype(np.float32))
    x_train, x_val = x_t[train_idx], x_t[val_idx]
    y_train, y_val = y_t[train_idx], y_t[val_idx]

    model = _AttentionNet(n_features, d_model, n_heads)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_val = float("inf")
    best_state = None
    stale_epochs = 0

    for _ in range(max_epochs):
        model.train()
        optimizer.zero_grad()
        pred_train, _ = model(x_train)
        loss = _loss(pred_train, outcome_type, y_train)
        if not torch.isnan(loss):
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            pred_val, _ = model(x_val)
            val_loss = _loss(pred_val, outcome_type, y_val).item()

        if not math.isnan(val_loss) and val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        pred_full, attn_full = model(x_t)
        predictions = pred_full.numpy().reshape(-1, 1)
        attention_weights = attn_full.mean(dim=0).numpy().reshape(-1, 1)

    return _AttentionRun(predictions=predictions, attention_weights=attention_weights)
