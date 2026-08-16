"""The Candidate: the one unit every discovery method must emit.

A discovery method may produce many of these per run (every latent dimension,
every K, every module) and must log all of them, not just the best-looking
one, into a CandidateRegistry. That full pool is what campaign-level
correction runs over.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Optional

import numpy as np

from discovery_engine.data.cohort import CohortData
from discovery_engine.outcomes import OutcomeSpec


@dataclass
class Candidate:
    id: str
    method: str
    method_params: dict
    outcome_blind: bool
    modality_touch: list[str]
    sample_ids: list[str]
    scores: dict[str, float]
    provenance: dict = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    # Not serialized. Required when outcome_blind is False (enforced by
    # CandidateRegistry.add, not here, since a Candidate loaded back from
    # disk for reporting has no live refit function and that's fine).
    #
    # Contract: refit MUST source all of its internal randomness from the
    # passed-in `np.random.Generator` (e.g. model init, data-split shuffles,
    # dropout) and must never instantiate its own RNG (np.random.default_rng()
    # with no seed, torch's global RNG, etc). The permutation loop in
    # campaign.correction derives this generator deterministically from its
    # own seeded RNG, so a real campaign run is bit-reproducible end to end,
    # including every retrained RSF/DeepSurv/attention refit. A refit that
    # reaches for its own unseeded randomness breaks that reproducibility in
    # a way that's invisible until someone tries to rerun a campaign and gets
    # different p-values.
    refit: Optional[Callable[[CohortData, OutcomeSpec, np.random.Generator], "Candidate"]] = field(
        default=None, repr=False, compare=False
    )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "method": self.method,
            "method_params": self.method_params,
            "outcome_blind": self.outcome_blind,
            "modality_touch": self.modality_touch,
            "sample_ids": self.sample_ids,
            "scores": self.scores,
            "provenance": self.provenance,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Candidate":
        return cls(
            id=d["id"],
            method=d["method"],
            method_params=d["method_params"],
            outcome_blind=d["outcome_blind"],
            modality_touch=d["modality_touch"],
            sample_ids=d["sample_ids"],
            scores=d["scores"],
            provenance=d.get("provenance", {}),
            created_at=d.get("created_at", ""),
            refit=None,
        )


def tag_feature_subset(candidates: list["Candidate"], subsets: dict[str, str | None]) -> list["Candidate"]:
    """Stamps provenance["feature_subset"] on every candidate in place, as
    a dict keyed by modality name (restricted to that candidate's own
    modality_touch), not a single flat string -- returns the same list for
    chaining.

    A flat string is wrong for multi-modal candidates (autoencoder,
    DeepSurv, attention), which touch more than one modality per
    candidate and can have a DIFFERENT subset status per modality in the
    very same candidate (e.g. CMB searched in full, RNA-seq restricted to
    a variance-selected set, together in one autoencoder candidate). A
    flat string would either mislabel the fully-searched modality as
    subsetted too, or -- if multi-modal candidates were left untagged to
    avoid that -- make them indistinguishable from a candidate that
    genuinely searched everything, the exact ambiguity this field exists
    to prevent. Caught by review before the real count-gathering compute
    ran, not after.

    `subsets` maps modality name -> subset name (None means "searched in
    full"); only keys present in a given candidate's modality_touch are
    applied to it, so one `subsets` dict covering every modality this
    campaign ever touches is safe to reuse across every discover() call,
    single- or multi-modal alike, rather than needing a separate call per
    modality combination.
    """
    for c in candidates:
        c.provenance["feature_subset"] = {m: subsets.get(m) for m in c.modality_touch}
    return candidates
