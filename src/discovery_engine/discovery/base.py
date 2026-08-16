"""DiscoveryMethod protocols, for the two kinds of discovery this project
has to support.

OutcomeBlindDiscoveryMethod: the signature IS the outcome-blindness proof.
`discover` takes only CohortData, no outcome parameter exists to read
from. A method conforming to this protocol cannot leak outcome into
candidate generation because there is nothing to leak from.

That guarantee is about outcome-blindness ONLY -- this protocol does not
claim or imply determinism, and conflating the two would be a real gap,
not a stylistic one. clustering.py's scipy.cluster.hierarchy.linkage
genuinely has no internal randomness at all, so it takes no seed. Other
conforming methods (louvain.py) DO have algorithm-internal stochasticity
(e.g. node-visitation order) and take an explicit `seed: int` via
**params; "outcome-blind" says nothing about whether such a method's
output is reproducible across seeds, only that the outcome never touched
it. "Deterministic" and "stochastic but empirically verified stable" are
different claims resting on different evidence, and a method with
internal randomness must say which one applies, and on what evidence --
see validation.stability's module docstring for the three-category
framework (deterministic / STABILITY_REQUIRED_METHODS / empirically
exempted via EMPIRICALLY_EXEMPTED_METHODS) every such method must place
itself into explicitly. There is no default category; silence is a bug,
not an exemption.

OutcomeInformedDiscoveryMethod (Phase 4, first implemented by deepsurv.py):
takes an outcome and an rng, and is exempt from the outcome-blindness
guarantee by construction, not by convention, which is why it's a
different protocol rather than an optional parameter on the same one.
Every candidate it emits must be outcome_blind=False with a working
`refit` (campaign.registry.CandidateRegistry.add enforces this), and the
rng it receives must be threaded through to every source of internal
randomness (model init, train/val split, dropout) exactly like
Candidate.refit's own contract, since discovery here and refit under
permutation are literally the same training procedure run on different
(outcome, rng) inputs.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from discovery_engine.campaign.candidate import Candidate
from discovery_engine.data.cohort import CohortData
from discovery_engine.outcomes import OutcomeSpec


class OutcomeBlindDiscoveryMethod(Protocol):
    name: str

    def discover(self, data: CohortData, **params) -> list[Candidate]: ...


class OutcomeInformedDiscoveryMethod(Protocol):
    name: str

    def discover(
        self, data: CohortData, outcome: OutcomeSpec, rng: np.random.Generator, **params
    ) -> list[Candidate]: ...
