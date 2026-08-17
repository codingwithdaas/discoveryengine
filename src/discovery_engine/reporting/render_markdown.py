"""Markdown rendering of a CampaignReportDocument, built from the SAME
resolved document render_json.py consumes -- never re-derived from
CampaignReport independently, so JSON and Markdown cannot drift apart.

Negative results are not an appendix: every candidate, accepted or
rejected, appears in ONE unified table ("All candidates") before any
per-candidate narrative detail, per skill rule 5 -- a campaign that
tried many configurations and validated none reports that as clearly as
one that found something. All descriptive language comes from
reporting.language; this module only ever interpolates numbers and
already-resolved strings into markdown structure.
"""

from __future__ import annotations

from pathlib import Path

from discovery_engine.reporting.schema import CampaignReportDocument


def to_markdown(document: CampaignReportDocument) -> str:
    lines: list[str] = []
    lines.append(f"# Campaign {document.campaign_id} - Cohort {document.cohort_id}")
    lines.append("")
    lines.append(
        f"Generated {document.generated_at}. {document.n_candidates_total} candidates tested, "
        f"{document.n_accepted} accepted, {document.n_rejected} rejected, alpha = {document.alpha}."
    )
    lines.append("")

    if document.scope_limitations:
        lines.append("## Scope limitations")
        lines.append("")
        lines.append(
            "Method x modality x feature-subset combinations deliberately NOT attempted this "
            "campaign, stated explicitly rather than left to be inferred from their absence below:"
        )
        lines.append("")
        lines.append("| Method | Modality | Feature subset | Reason |")
        lines.append("|---|---|---|---|")
        for lim in document.scope_limitations:
            lines.append(f"| {lim.method} | {lim.modality} | {lim.feature_subset} | {lim.reason} |")
        lines.append("")

    lines.append("## Candidates by method")
    lines.append("")
    lines.append("| Method | Candidates | Accepted |")
    lines.append("|---|---|---|")
    for m in document.method_summaries:
        lines.append(f"| {m.method} | {m.n_candidates} | {m.n_accepted} |")
    lines.append("")

    lines.append("## All candidates")
    lines.append("")
    lines.append("| Candidate | Method | Feature subset | Accepted | Campaign p | Reproducibility | Reason(s) if rejected |")
    lines.append("|---|---|---|---|---|---|---|")
    for e in document.candidates:
        reasons = "; ".join(e.rejection_reasons) if e.rejection_reasons else ""
        lines.append(
            f"| {e.candidate_id} | {e.method} | {_format_feature_subset(e.feature_subset)} | {e.accepted} | "
            f"{e.correction.campaign_p_value:.4f} | {e.gate.half_split.calibration} | {reasons} |"
        )
    lines.append("")

    accepted_entries = [e for e in document.candidates if e.accepted]
    if accepted_entries:
        lines.append("## Accepted signatures")
        lines.append("")
        for e in accepted_entries:
            lines.append(f"### {e.candidate_id}")
            lines.append("")
            lines.append(e.narrative_summary + ".")
            lines.append("")
            if e.explanation_reproducibility_note:
                lines.append(e.explanation_reproducibility_note + ".")
                lines.append("")
            lines.append(f"Primary interpretability signal ({e.primary_interpretability_source}):")
            lines.append("")
            lines.append("| Feature | Value |")
            lines.append("|---|---|")
            for item in e.primary_interpretability_signal[:10]:
                lines.append(f"| {item['feature']} | {item['value']:.3f} |")
            lines.append("")

    return "\n".join(lines)


def _format_feature_subset(feature_subset: dict[str, str | None] | None) -> str:
    if feature_subset is None:
        return "(untagged)"
    if not feature_subset:
        return "(no modalities recorded)"
    return "; ".join(f"{modality}={subset or 'full'}" for modality, subset in sorted(feature_subset.items()))


def write_markdown(document: CampaignReportDocument, path: str | Path) -> None:
    Path(path).write_text(to_markdown(document), encoding="utf-8")
