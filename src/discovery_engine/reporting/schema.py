"""CampaignReportDocument: the one canonical, fully-resolved report
representation. render_json.py and render_markdown.py both consume THIS,
never CampaignReport independently a second time, so the two output
formats cannot drift apart from each other.

Every candidate the campaign produced appears in `.candidates`, accepted
and rejected together in one list, not two -- negative results are not an
appendix (skill rule 5). Each entry embeds the FULL original GateResult
and CorrectionResult objects unmodified, not a flattened summary: every
sub-check's every number survives into the report, which is what makes a
rejected candidate's record independently useful later rather than a bare
pass/fail flag (dataclasses.asdict, in render_json.py, recurses through
all of it for free, since every validation result type is already a plain
dataclass).

`build_report_document` needs the original `candidates` list, not just
`campaign_report`: CampaignCandidateReport only stores candidate_id/
method/gate/correction/accepted, not the Candidate object itself, and
this module needs each candidate's `.provenance` (attention_weights,
component_loadings) to pick the preferred interpretability signal.
Deliberately never embeds a raw Candidate object into the report itself
though -- its `refit` field is a closure, not JSON-serializable; only
specific already-serializable fields are pulled out of it.

PREFERRED_INTERPRETABILITY_SIGNAL is a fixed, manually-maintained,
method-name-keyed lookup, the same shape as validation.stability's
STABILITY_REQUIRED_METHODS and EMPIRICALLY_EXEMPTED_METHODS -- not a new
pattern. Falls back to the generic correlation-based report for any
method not listed. The generic correlation signal is never dropped even
when a preferred signal exists: it's always present as
`secondary_correlation_signal`, the same "keep both, don't replace"
precedent as half-split-and-bootstrap-CI, median-split-and-extreme-groups.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from discovery_engine.campaign.campaign import CampaignReport
from discovery_engine.campaign.candidate import Candidate
from discovery_engine.campaign.correction import CorrectionResult
from discovery_engine.reporting import language
from discovery_engine.validation.gate import GateResult

PREFERRED_INTERPRETABILITY_SIGNAL: dict[str, str] = {
    "attention": "attention_weights",
    "nmf": "component_loadings",
}

_TOP_N = 15


@dataclass
class MethodSummary:
    method: str
    n_candidates: int
    n_accepted: int


@dataclass
class CandidateReportEntry:
    candidate_id: str
    method: str
    accepted: bool
    gate: GateResult
    correction: CorrectionResult
    rejection_reasons: list[str]
    primary_interpretability_source: str
    primary_interpretability_signal: list[dict]
    secondary_correlation_signal: list[dict]
    explanation_reproducibility_note: str | None
    narrative_summary: str


@dataclass
class CampaignReportDocument:
    campaign_id: str
    cohort_id: str
    alpha: float
    generated_at: str
    n_candidates_total: int
    n_accepted: int
    n_rejected: int
    method_summaries: list[MethodSummary] = field(default_factory=list)
    candidates: list[CandidateReportEntry] = field(default_factory=list)


def build_report_document(campaign_report: CampaignReport, candidates: list[Candidate]) -> CampaignReportDocument:
    candidates_by_id = {c.id: c for c in candidates}

    entries = [
        _build_entry(cr, candidates_by_id[cr.candidate_id], campaign_report.alpha)
        for cr in campaign_report.candidate_reports
    ]
    entries.sort(key=lambda e: (e.method, e.correction.campaign_p_value))

    return CampaignReportDocument(
        campaign_id=campaign_report.campaign_id,
        cohort_id=campaign_report.cohort_id,
        alpha=campaign_report.alpha,
        generated_at=datetime.now(timezone.utc).isoformat(),
        n_candidates_total=len(entries),
        n_accepted=sum(1 for e in entries if e.accepted),
        n_rejected=sum(1 for e in entries if not e.accepted),
        method_summaries=_summarize_methods(entries),
        candidates=entries,
    )


def _build_entry(cr, candidate: Candidate, alpha: float) -> CandidateReportEntry:
    rejection_reasons: list[str] = []
    if not cr.accepted:
        rejection_reasons = list(cr.gate.failure_reasons)
        if cr.correction.campaign_p_value >= alpha:
            rejection_reasons.append(
                f"campaign-corrected p-value {cr.correction.campaign_p_value:.4f} >= alpha {alpha}"
            )

    primary_source, primary_signal = _select_primary_signal(candidate, cr.gate)
    secondary = [{"feature": a.feature, "value": a.correlation} for a in cr.gate.interpretability.top_features]
    explanation_note = _explanation_reproducibility_note(candidate)

    narrative = (
        language.describe_acceptance(cr.accepted, cr.correction.campaign_p_value, alpha)
        + "; "
        + language.describe_reproducibility(cr.gate.half_split.calibration, cr.gate.half_split.consistency_fraction)
    )

    return CandidateReportEntry(
        candidate_id=cr.candidate_id,
        method=cr.method,
        accepted=cr.accepted,
        gate=cr.gate,
        correction=cr.correction,
        rejection_reasons=rejection_reasons,
        primary_interpretability_source=primary_source,
        primary_interpretability_signal=primary_signal,
        secondary_correlation_signal=secondary,
        explanation_reproducibility_note=explanation_note,
        narrative_summary=narrative,
    )


def _select_primary_signal(candidate: Candidate, gate: GateResult) -> tuple[str, list[dict]]:
    preferred_key = PREFERRED_INTERPRETABILITY_SIGNAL.get(candidate.method)
    if preferred_key and preferred_key in candidate.provenance:
        raw: dict = candidate.provenance[preferred_key]
        ranked = sorted(raw.items(), key=lambda kv: abs(kv[1]), reverse=True)[:_TOP_N]
        return preferred_key, [{"feature": f, "value": v} for f, v in ranked]

    top = gate.interpretability.top_features[:_TOP_N]
    return "correlation", [{"feature": a.feature, "value": a.correlation} for a in top]


def _explanation_reproducibility_note(candidate: Candidate) -> str | None:
    stability_info = candidate.provenance.get("attention_weight_stability")
    if stability_info is None:
        return None
    return language.describe_explanation_reproducibility(stability_info)


def _summarize_methods(entries: list[CandidateReportEntry]) -> list[MethodSummary]:
    methods = sorted({e.method for e in entries})
    return [
        MethodSummary(
            method=m,
            n_candidates=sum(1 for e in entries if e.method == m),
            n_accepted=sum(1 for e in entries if e.method == m and e.accepted),
        )
        for m in methods
    ]
