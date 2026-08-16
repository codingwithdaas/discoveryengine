"""Novelty scoring against everything already validated.

A newly discovered candidate can be individually significant against an
outcome while being fully redundant with an already-known signature.
Classification uses both score correlation and, when both candidates
expose a discrete contributing feature set (e.g. clustering modules),
feature-set Jaccard overlap; a candidate is flagged if EITHER signal says
so, since a high-correlation-but-different-features candidate and a
low-correlation-but-same-features candidate are both cases worth a human
looking twice at.

Existing validated signatures are represented as plain Candidate objects
(what the SignatureRegistry stores) rather than inventing a parallel type.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from discovery_engine.campaign.candidate import Candidate

CORRELATION_REDUNDANT = 0.8
CORRELATION_PARTIAL = 0.4
JACCARD_REDUNDANT = 0.7
JACCARD_PARTIAL = 0.3

_RANK = {"novel": 0, "partially_overlapping": 1, "fully_redundant": 2}


@dataclass
class NoveltyComparison:
    signature_id: str
    correlation: float | None
    n_overlapping_samples: int
    feature_overlap_jaccard: float | None
    classification: str


@dataclass
class NoveltyResult:
    comparisons: list[NoveltyComparison]
    overall_classification: str


def novelty_check(candidate: Candidate, existing_signatures: list[Candidate]) -> NoveltyResult:
    if not existing_signatures:
        return NoveltyResult(comparisons=[], overall_classification="novel")

    comparisons = [_compare(candidate, existing) for existing in existing_signatures]
    overall = max((c.classification for c in comparisons), key=lambda cls: _RANK[cls])
    return NoveltyResult(comparisons=comparisons, overall_classification=overall)


def _compare(candidate: Candidate, existing: Candidate) -> NoveltyComparison:
    common_ids = [sid for sid in candidate.scores if sid in existing.scores]
    correlation = None
    if len(common_ids) >= 3:
        a = np.array([candidate.scores[sid] for sid in common_ids])
        b = np.array([existing.scores[sid] for sid in common_ids])
        if np.std(a) > 0 and np.std(b) > 0:
            correlation = float(np.corrcoef(a, b)[0, 1])

    jaccard = _feature_jaccard(candidate, existing)

    is_redundant = (correlation is not None and abs(correlation) >= CORRELATION_REDUNDANT) or (
        jaccard is not None and jaccard >= JACCARD_REDUNDANT
    )
    is_partial = (correlation is not None and abs(correlation) >= CORRELATION_PARTIAL) or (
        jaccard is not None and jaccard >= JACCARD_PARTIAL
    )

    if is_redundant:
        classification = "fully_redundant"
    elif is_partial:
        classification = "partially_overlapping"
    else:
        classification = "novel"

    return NoveltyComparison(existing.id, correlation, len(common_ids), jaccard, classification)


def _feature_jaccard(candidate: Candidate, existing: Candidate) -> float | None:
    a = candidate.provenance.get("member_features")
    b = existing.provenance.get("member_features")
    if not a or not b:
        return None
    a, b = set(a), set(b)
    union = a | b
    if not union:
        return None
    return len(a & b) / len(union)
