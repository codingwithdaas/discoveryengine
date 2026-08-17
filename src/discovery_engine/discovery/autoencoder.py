"""Multi-modal autoencoder / VAE discovery: the first stochastic (neural)
discovery method in this project. Outcome-blind (discover() takes only
CohortData, same OutcomeBlindDiscoveryMethod protocol as clustering.py).

Two things make this different from the classical baseline:

1. Randomness inside discovery itself (weight init, train/val split,
   VAE reparameterization) must be sourced from the passed rng, never from
   torch's own unseeded global RNG. Same rng-threading contract Phase 1
   established for outcome-informed refit closures: derive a per-run seed
   via `rng.integers(...)`, hand it to `torch.manual_seed`, never call
   `torch.manual_seed()` with a hardcoded literal or skip seeding.

2. A single trained network is one sample of a random process (skill rule
   2): the latent structure has to reappear across >= 3 independent
   training runs before it's trusted. Latent units are not identifiable
   across independent runs (arbitrary permutation, arbitrary sign flip), so
   "the same unit reappearing" is decided by cross-seed correlation
   matching, not by unit index. Every unit still becomes its own Candidate
   regardless of match outcome (skill rule 1: log everything, not just the
   best-looking one); the raw match measurement travels in
   provenance["stability"] and validation.stability.stability_check turns
   it into a calibration band / pass-fail at gate time. This module never
   decides pass/fail itself.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
from torch import nn

from discovery_engine.campaign.candidate import Candidate
from discovery_engine.data.cohort import CohortData
from discovery_engine.discovery.multimodal_input import build_multimodal_input_matrix
from discovery_engine.validation.stability import TrainedRun, match_units_across_seeds

logger = logging.getLogger(__name__)


class _AutoencoderNet(nn.Module):
    def __init__(self, n_features: int, latent_dim: int, hidden_dims: list[int], variational: bool):
        super().__init__()
        self.variational = variational

        encoder_layers: list[nn.Module] = []
        prev = n_features
        for h in hidden_dims:
            encoder_layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        self.encoder_trunk = nn.Sequential(*encoder_layers)
        self.to_latent = nn.Linear(prev, latent_dim * 2 if variational else latent_dim)

        decoder_layers: list[nn.Module] = []
        prev = latent_dim
        for h in reversed(hidden_dims):
            decoder_layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        decoder_layers.append(nn.Linear(prev, n_features))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        h = self.encoder_trunk(x)
        out = self.to_latent(h)
        if self.variational:
            mu, logvar = out.chunk(2, dim=-1)
            return mu, logvar
        return out, None

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encode(x)
        if self.variational:
            std = torch.exp(0.5 * logvar)
            z = mu + std * torch.randn_like(std)
        else:
            z = mu
        recon = self.decoder(z)
        return recon, mu, logvar


def _loss(recon: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor | None, beta: float) -> torch.Tensor:
    recon_loss = nn.functional.mse_loss(recon, x, reduction="mean")
    if logvar is None:
        return recon_loss
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl


class AutoencoderDiscovery:
    def __init__(self, variational: bool = False):
        self.variational = variational
        self.name = "variational_autoencoder" if variational else "autoencoder"

    def discover(
        self,
        data: CohortData,
        modalities: list[str],
        latent_dims: list[int],
        n_seeds: int = 3,
        hidden_dims: list[int] | None = None,
        beta: float = 1.0,
        val_fraction: float = 0.2,
        max_epochs: int = 200,
        patience: int = 15,
        stability_threshold: float = 0.7,
        seed: int = 0,
        feature_subset: str | None = None,
    ) -> list[Candidate]:
        hidden_dims = list(hidden_dims) if hidden_dims is not None else [128, 64]
        x, sample_ids, feature_names, modality_info = build_multimodal_input_matrix(data, modalities)

        rng = np.random.default_rng(seed)
        candidates: list[Candidate] = []
        for latent_dim in latent_dims:
            runs = [
                _train_one_seed(
                    x, latent_dim, self.variational, hidden_dims, beta,
                    val_fraction, max_epochs, patience,
                    np.random.default_rng(rng.integers(0, 2**63 - 1)),
                )
                for _ in range(n_seeds)
            ]
            matches = match_units_across_seeds(runs, stability_threshold)
            ref_scores = runs[0].latent_scores

            for unit_idx in range(latent_dim):
                scores = {sid: float(ref_scores[i, unit_idx]) for i, sid in enumerate(sample_ids)}
                candidates.append(
                    Candidate(
                        id=f"{self.name}_{'_'.join(modalities)}_ld{latent_dim}_u{unit_idx}" + (f"_fs{feature_subset}" if feature_subset else ""),
                        method=self.name,
                        method_params={
                            "latent_dim": latent_dim,
                            "unit": unit_idx,
                            "modalities": list(modalities),
                            "n_seeds": n_seeds,
                            "hidden_dims": hidden_dims,
                            "beta": beta if self.variational else None,
                        },
                        outcome_blind=True,
                        modality_touch=list(modalities),
                        sample_ids=sample_ids,
                        scores=scores,
                        provenance={
                            "feature_names": feature_names,
                            "modality_info": modality_info,
                            "stability": matches[unit_idx],
                        },
                    )
                )
        return candidates


def _train_one_seed(
    x: np.ndarray,
    latent_dim: int,
    variational: bool,
    hidden_dims: list[int],
    beta: float,
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
    x_train, x_val = x_t[train_idx], x_t[val_idx]

    model = _AutoencoderNet(x.shape[1], latent_dim, hidden_dims, variational)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_val = float("inf")
    best_state = None
    stale_epochs = 0

    for _ in range(max_epochs):
        model.train()
        optimizer.zero_grad()
        recon, mu, logvar = model(x_train)
        loss = _loss(recon, x_train, mu, logvar, beta)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            recon_val, mu_val, logvar_val = model(x_val)
            val_loss = _loss(recon_val, x_val, mu_val, logvar_val, beta).item()

        if val_loss < best_val - 1e-6:
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
        mu_full, _ = model.encode(x_t)
        latent_scores = mu_full.numpy()

    return TrainedRun(latent_scores=latent_scores)
