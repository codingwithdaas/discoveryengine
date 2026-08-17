"""DeepSurv discovery: the first method that is both neural (Phase 3's
stability contract applies) AND outcome-informed (Phase 1's refit contract
applies). Nothing before this combined the two, so the design decisions
here are stated explicitly rather than left implicit:

Stability and refit are two orthogonal retraining loops, not one:
  - Stability (discovery time, runs once per candidate): retrain the same
    architecture n_seeds times on the TRUE outcome, cross-seed match the
    resulting risk scores by correlation (validation.stability). Answers
    "is this structure reproducible", independent of permutation testing.
  - Refit (campaign-correction time, runs once per permutation): retrain
    the same architecture ONCE on the PERMUTED outcome. Answers "how
    extreme is the observed association versus this same fitting
    procedure applied to a random rearrangement of the outcome". A
    refit-produced candidate is NEVER itself stability-checked -- running
    n_seeds retrains per permutation would multiply an already-expensive
    neural refit by n_seeds for no statistical benefit, since
    campaign.correction only needs one observed statistic per permutation
    draw. Refit candidates deliberately carry no provenance["stability"]
    key, and structurally never reach validation.gate (confirmed by
    reading campaign.correction._null_statistics: refit output goes
    straight into association_statistic, never through run_validation_gate
    or stability_check).

Both loops call the exact same _train_one_seed function -- same train/val
split procedure, same early-stopping-on-held-out-Cox-loss, same patience --
so refit cannot silently diverge from the discovery fit's training
procedure (that would bias the permutation null against the observed
statistic's fitting depth). The only things that differ between a
discovery run, a stability retrain, and a refit call are which outcome
(true vs permuted) and which rng seeds it.

The refit's train/val split is freshly drawn from the per-permutation rng
every call, never reused from the original discovery fit's split. This
follows directly from Candidate.refit's existing contract (which already
requires internal randomness, explicitly including "data-split shuffles",
to be sourced from the passed rng), and it's also what makes each
permutation draw a genuinely independent replicate of the whole fitting
procedure rather than sharing one frozen, arbitrary partition's quirks
across every null draw.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

from discovery_engine.campaign.candidate import Candidate
from discovery_engine.data.cohort import CohortData
from discovery_engine.discovery.multimodal_input import build_multimodal_input_matrix
from discovery_engine.outcomes import OutcomeSpec, OutcomeType
from discovery_engine.validation.stability import TrainedRun, match_units_across_seeds


class _DeepSurvNet(nn.Module):
    def __init__(self, n_features: int, hidden_dims: list[int], dropout: float):
        super().__init__()
        layers: list[nn.Module] = []
        prev = n_features
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def _cox_loss(risk: torch.Tensor, time: torch.Tensor, event: torch.Tensor) -> torch.Tensor:
    """Negative Cox partial log-likelihood (Breslow approximation for ties),
    full-batch. Risk set for sample i is every j with time_j >= time_i;
    sorting by time descending makes that risk set exactly the running
    prefix, so a single logcumsumexp gives every sample's risk-set term.

    Returns NaN if the batch has zero observed events (undefined, not
    "perfect": a caller comparing this against a running best-so-far must
    skip NaN epochs rather than let an artificial 0.0 masquerade as the
    best model, which realistically happens with small val splits on
    survival data since events are typically the minority class).
    """
    order = torch.argsort(time, descending=True)
    risk = risk[order]
    event = event[order]
    events_mask = event.bool()
    if events_mask.sum() == 0:
        return torch.tensor(float("nan"))
    log_cumsum = torch.logcumsumexp(risk, dim=0)
    return -(risk - log_cumsum)[events_mask].mean()


class DeepSurvDiscovery:
    name = "deepsurv"

    def discover(
        self,
        data: CohortData,
        outcome: OutcomeSpec,
        rng: np.random.Generator,
        modalities: list[str],
        hidden_dims_options: list[list[int]],
        n_seeds: int = 3,
        dropout: float = 0.0,
        val_fraction: float = 0.2,
        max_epochs: int = 200,
        patience: int = 15,
        stability_threshold: float = 0.7,
        feature_subset: str | None = None,
    ) -> list[Candidate]:
        if outcome.type is not OutcomeType.SURVIVAL:
            raise ValueError("DeepSurvDiscovery requires a SURVIVAL outcome")

        x, sample_ids, feature_names, modality_info = build_multimodal_input_matrix(data, modalities)
        sub_outcome = outcome.subset(sample_ids)
        time = sub_outcome.values[:, 0].astype(np.float32)
        event = sub_outcome.values[:, 1].astype(np.float32)

        candidates: list[Candidate] = []
        for hidden_dims in hidden_dims_options:
            hidden_dims = list(hidden_dims)
            runs = [
                _train_one_seed(
                    x, time, event, hidden_dims, dropout, val_fraction, max_epochs, patience,
                    np.random.default_rng(rng.integers(0, 2**63 - 1)),
                )
                for _ in range(n_seeds)
            ]
            matches = match_units_across_seeds(runs, stability_threshold)
            ref_scores = runs[0].latent_scores

            scores = {sid: float(ref_scores[i, 0]) for i, sid in enumerate(sample_ids)}
            candidates.append(
                Candidate(
                    id=f"deepsurv_{'_'.join(modalities)}_h{'-'.join(map(str, hidden_dims))}" + (f"_fs{feature_subset}" if feature_subset else ""),
                    method=self.name,
                    method_params={
                        "modalities": list(modalities),
                        "hidden_dims": hidden_dims,
                        "dropout": dropout,
                        "n_seeds": n_seeds,
                    },
                    outcome_blind=False,
                    modality_touch=list(modalities),
                    sample_ids=sample_ids,
                    scores=scores,
                    provenance={
                        "feature_names": feature_names,
                        "modality_info": modality_info,
                        "stability": matches[0],
                    },
                    refit=_make_refit(modalities, hidden_dims, dropout, val_fraction, max_epochs, patience),
                )
            )
        return candidates


def _make_refit(
    modalities: list[str], hidden_dims: list[int], dropout: float,
    val_fraction: float, max_epochs: int, patience: int,
):
    def _refit(data: CohortData, outcome: OutcomeSpec, rng: np.random.Generator) -> Candidate:
        if outcome.type is not OutcomeType.SURVIVAL:
            raise ValueError("DeepSurv refit requires a SURVIVAL outcome")
        x, sample_ids, _, _ = build_multimodal_input_matrix(data, modalities)
        sub_outcome = outcome.subset(sample_ids)
        time = sub_outcome.values[:, 0].astype(np.float32)
        event = sub_outcome.values[:, 1].astype(np.float32)

        # One retrain, same training procedure as discovery/stability, on
        # the permuted outcome. Deliberately no stability check and no
        # "stability" provenance key: see module docstring.
        run = _train_one_seed(x, time, event, hidden_dims, dropout, val_fraction, max_epochs, patience, rng)
        scores = {sid: float(run.latent_scores[i, 0]) for i, sid in enumerate(sample_ids)}
        return Candidate(
            id="deepsurv_refit",
            method="deepsurv",
            method_params={"modalities": list(modalities), "hidden_dims": hidden_dims, "dropout": dropout},
            outcome_blind=False,
            modality_touch=list(modalities),
            sample_ids=sample_ids,
            scores=scores,
            provenance={},
        )

    return _refit


def _train_one_seed(
    x: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
    hidden_dims: list[int],
    dropout: float,
    val_fraction: float,
    max_epochs: int,
    patience: int,
    rng: np.random.Generator,
) -> TrainedRun:
    torch.manual_seed(int(rng.integers(0, 2**31 - 1)))

    n = x.shape[0]
    perm = rng.permutation(n)
    n_val = max(1, int(n * val_fraction))
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    x_t = torch.from_numpy(x)
    time_t, event_t = torch.from_numpy(time), torch.from_numpy(event)
    x_train, time_train, event_train = x_t[train_idx], time_t[train_idx], event_t[train_idx]
    x_val, time_val, event_val = x_t[val_idx], time_t[val_idx], event_t[val_idx]

    model = _DeepSurvNet(x.shape[1], hidden_dims, dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_val = float("inf")
    best_state = None
    stale_epochs = 0

    for _ in range(max_epochs):
        model.train()
        optimizer.zero_grad()
        loss = _cox_loss(model(x_train), time_train, event_train)
        if not torch.isnan(loss):
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = _cox_loss(model(x_val), time_val, event_val).item()

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
        risk_full = model(x_t).numpy().reshape(-1, 1)

    return TrainedRun(latent_scores=risk_full)
