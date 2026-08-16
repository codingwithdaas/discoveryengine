"""SignatureRegistry: the persistent, cross-campaign, cross-cohort record of
every ACCEPTED signature (as opposed to campaign.registry.CandidateRegistry,
which is the ephemeral, per-campaign log of every candidate generated,
accepted or not, used for correction). "Accepted" means it passed the gate
and survived campaign-level correction, not "validated" in any deeper
epistemic sense -- see reporting/language.py for why that word is avoided
throughout this project's user-facing output.

SQLite (stdlib) rather than a flat file: this one needs real queries
(novelty checks against everything accepted so far, optionally scoped to a
cohort) as the registry grows across cohorts and campaigns, which a
directory of JSON files would need to reinvent indexing for.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from discovery_engine.campaign.candidate import Candidate

_SCHEMA = """
CREATE TABLE IF NOT EXISTS signatures (
    id TEXT PRIMARY KEY,
    cohort_id TEXT NOT NULL,
    campaign_id TEXT NOT NULL,
    method TEXT NOT NULL,
    outcome_blind INTEGER NOT NULL,
    method_params TEXT NOT NULL,
    modality_touch TEXT NOT NULL,
    sample_ids TEXT NOT NULL,
    scores TEXT NOT NULL,
    provenance TEXT NOT NULL,
    validation_summary TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""

_COLUMNS = (
    "id", "method", "outcome_blind", "method_params", "modality_touch",
    "sample_ids", "scores", "provenance", "created_at",
)


class SignatureRegistry:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._conn = sqlite3.connect(str(self.path))
        self._conn.execute(_SCHEMA)
        self._conn.commit()

    def add(self, candidate: Candidate, cohort_id: str, campaign_id: str, validation_summary: dict) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO signatures "
            "(id, cohort_id, campaign_id, method, outcome_blind, method_params, "
            " modality_touch, sample_ids, scores, provenance, validation_summary, created_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                candidate.id,
                cohort_id,
                campaign_id,
                candidate.method,
                int(candidate.outcome_blind),
                json.dumps(candidate.method_params),
                json.dumps(candidate.modality_touch),
                json.dumps(candidate.sample_ids),
                json.dumps(candidate.scores),
                json.dumps(candidate.provenance),
                json.dumps(validation_summary),
                candidate.created_at,
            ),
        )
        self._conn.commit()

    def all_as_candidates(self, cohort_id: str | None = None) -> list[Candidate]:
        query = f"SELECT {', '.join(_COLUMNS)} FROM signatures"
        params: tuple = ()
        if cohort_id is not None:
            query += " WHERE cohort_id = ?"
            params = (cohort_id,)
        rows = self._conn.execute(query, params).fetchall()
        return [_row_to_candidate(row) for row in rows]

    def __len__(self) -> int:
        (count,) = self._conn.execute("SELECT COUNT(*) FROM signatures").fetchone()
        return count

    def close(self) -> None:
        self._conn.close()


def _row_to_candidate(row: tuple) -> Candidate:
    cid, method, outcome_blind, method_params, modality_touch, sample_ids, scores, provenance, created_at = row
    return Candidate(
        id=cid,
        method=method,
        method_params=json.loads(method_params),
        outcome_blind=bool(outcome_blind),
        modality_touch=json.loads(modality_touch),
        sample_ids=json.loads(sample_ids),
        scores=json.loads(scores),
        provenance=json.loads(provenance),
        created_at=created_at,
        refit=None,
    )
