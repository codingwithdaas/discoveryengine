"""DiscoveryMethod protocol for outcome-blind methods.

The signature IS the outcome-blindness proof: `discover` takes only
CohortData, no outcome parameter exists to read from. A method conforming
to this protocol cannot leak outcome into candidate generation because
there is nothing to leak from. Outcome-informed methods (RSF, DeepSurv,
attention, interaction search; Phase 4) need a different protocol that
takes an outcome and an rng, and are exempt from this guarantee by
construction, not by convention, which is why they're handled separately.
"""

from __future__ import annotations

from typing import Protocol

from discovery_engine.campaign.candidate import Candidate
from discovery_engine.data.cohort import CohortData


class OutcomeBlindDiscoveryMethod(Protocol):
    name: str

    def discover(self, data: CohortData, **params) -> list[Candidate]: ...
