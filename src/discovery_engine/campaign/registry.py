"""CandidateRegistry: the append-only log of every candidate a campaign
has ever produced. Correction runs over this full pool, not per-method
subsets, so nothing here is allowed to be quietly dropped.
"""

from __future__ import annotations

import json
from pathlib import Path

from discovery_engine.campaign.candidate import Candidate


class CandidateRegistry:
    def __init__(self) -> None:
        self._candidates: dict[str, Candidate] = {}

    def add(self, candidate: Candidate) -> None:
        if candidate.id in self._candidates:
            raise ValueError(f"duplicate candidate id: {candidate.id!r}")
        if not candidate.outcome_blind and candidate.refit is None:
            raise ValueError(
                f"candidate {candidate.id!r} is outcome-informed but has no refit "
                "function; campaign correction cannot validly recompute its "
                "significance under permutation without rerunning the "
                "outcome-dependent discovery step"
            )
        self._candidates[candidate.id] = candidate

    def all(self) -> list[Candidate]:
        return list(self._candidates.values())

    def __len__(self) -> int:
        return len(self._candidates)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        with path.open("w", encoding="utf-8") as f:
            for c in self._candidates.values():
                f.write(json.dumps(c.to_dict()) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> "CandidateRegistry":
        reg = cls()
        with Path(path).open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                reg._candidates[d["id"]] = Candidate.from_dict(d)
        return reg
