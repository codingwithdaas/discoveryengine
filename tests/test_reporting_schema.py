import pytest

from discovery_engine.campaign.campaign import CampaignCandidateReport, CampaignReport
from discovery_engine.campaign.candidate import Candidate
from discovery_engine.campaign.correction import CorrectionResult
from discovery_engine.reporting.schema import build_report_document
from discovery_engine.validation.bootstrap import BootstrapCIResult, HalfSplitResult
from discovery_engine.validation.covariate import CovariateStratifiedResult
from discovery_engine.validation.design_crosscheck import DesignCrosscheckResult, DesignResult
from discovery_engine.validation.gate import GateResult
from discovery_engine.validation.interpretability import FeatureAttribution, InterpretabilityReport
from discovery_engine.validation.multivariate import MultivariateAddsValueResult
from discovery_engine.validation.novelty import NoveltyResult
from discovery_engine.validation.stability import StabilityResult


def _design_result(name="median_split"):
    return DesignResult(design=name, n_group_low=10, n_group_high=10, observed_statistic=0.2, p_value=0.1)


def _make_gate_result(
    candidate_id="c1",
    calibration="robust",
    consistency_fraction=0.95,
    novelty_classification="novel",
    top_features=None,
    passed=True,
    failure_reasons=None,
    stability_applicable=False,
):
    top_features = top_features if top_features is not None else [FeatureAttribution("gene_1", "rnaseq", 0.7)]
    return GateResult(
        candidate_id=candidate_id,
        half_split=HalfSplitResult(30, 30, consistency_fraction, calibration, 0.5),
        bootstrap_ci=BootstrapCIResult(1000, 0.5, 0.3, 0.7, 0.95),
        covariate_stratified=CovariateStratifiedResult("grade", 0.5, 0.02, []),
        freedman_lane=None,
        design_crosscheck=DesignCrosscheckResult(_design_result(), _design_result("extreme_groups"), False, None, None),
        multivariate=MultivariateAddsValueResult([], "freedman_lane_permutation", 0.4, 0.01, 500),
        stability=StabilityResult(applicable=stability_applicable, method="m", reason="n/a"),
        novelty=NoveltyResult([], novelty_classification),
        interpretability=InterpretabilityReport("correlation", top_features),
        passed=passed,
        failure_reasons=failure_reasons or [],
    )


def _make_candidate(cid, method="hierarchical_clustering", provenance=None):
    ids = [f"S{i}" for i in range(10)]
    return Candidate(
        id=cid, method=method, method_params={}, outcome_blind=True,
        modality_touch=["rnaseq"], sample_ids=ids, scores={sid: 0.1 for sid in ids},
        provenance=provenance or {},
    )


def _make_report(entries: list[tuple[str, str, bool, GateResult, float]]) -> tuple[CampaignReport, list[Candidate]]:
    """entries: list of (candidate_id, method, accepted, gate_result, campaign_p_value)"""
    candidates = []
    candidate_reports = []
    for cid, method, accepted, gate, p in entries:
        candidates.append(_make_candidate(cid, method))
        correction = CorrectionResult(cid, method, 2.0, 2.0, p, 500)
        candidate_reports.append(CampaignCandidateReport(cid, method, gate, correction, accepted))
    report = CampaignReport(campaign_id="camp1", cohort_id="cohort1", alpha=0.05, candidate_reports=candidate_reports)
    return report, candidates


def test_every_candidate_appears_accepted_and_rejected_together():
    accepted_gate = _make_gate_result("c1", passed=True)
    rejected_gate = _make_gate_result(
        "c2", calibration="noise", consistency_fraction=0.3, passed=False,
        failure_reasons=["half-split consistency calibrated as noise (0.30)"],
    )
    report, candidates = _make_report([
        ("c1", "hierarchical_clustering", True, accepted_gate, 0.001),
        ("c2", "hierarchical_clustering", False, rejected_gate, 0.001),
    ])
    doc = build_report_document(report, candidates)
    assert doc.n_candidates_total == 2
    assert doc.n_accepted == 1
    assert doc.n_rejected == 1
    ids_seen = {e.candidate_id for e in doc.candidates}
    assert ids_seen == {"c1", "c2"}, "both accepted and rejected candidates must appear in the same list"


def test_rejected_candidate_gets_both_gate_and_correction_reasons():
    gate = _make_gate_result(
        "c1", calibration="noise", consistency_fraction=0.3, passed=False,
        failure_reasons=["half-split consistency calibrated as noise (0.30)"],
    )
    report, candidates = _make_report([("c1", "hierarchical_clustering", False, gate, 0.4)])
    doc = build_report_document(report, candidates)
    entry = doc.candidates[0]
    assert "half-split consistency calibrated as noise (0.30)" in entry.rejection_reasons
    assert any("campaign-corrected p-value" in r for r in entry.rejection_reasons)


def test_accepted_candidate_has_no_rejection_reasons():
    gate = _make_gate_result("c1", passed=True)
    report, candidates = _make_report([("c1", "hierarchical_clustering", True, gate, 0.001)])
    doc = build_report_document(report, candidates)
    assert doc.candidates[0].rejection_reasons == []


def test_full_gate_and_correction_objects_are_preserved_unmodified():
    gate = _make_gate_result("c1", calibration="moderate", consistency_fraction=0.8)
    report, candidates = _make_report([("c1", "hierarchical_clustering", True, gate, 0.001)])
    doc = build_report_document(report, candidates)
    entry = doc.candidates[0]
    assert entry.gate is gate
    assert entry.gate.half_split.consistency_fraction == 0.8
    assert entry.correction.campaign_p_value == 0.001


def test_attention_candidate_uses_attention_weights_as_primary_signal():
    gate = _make_gate_result("c1", top_features=[FeatureAttribution("gene_x", "rnaseq", 0.2)])
    provenance = {"attention_weights": {"gene_a": 0.9, "gene_b": 0.1, "gene_c": 0.5}}
    candidates = [_make_candidate("c1", method="attention", provenance=provenance)]
    correction = CorrectionResult("c1", "attention", 2.0, 2.0, 0.001, 500)
    report = CampaignReport("camp1", "cohort1", 0.05, [CampaignCandidateReport("c1", "attention", gate, correction, True)])

    doc = build_report_document(report, candidates)
    entry = doc.candidates[0]
    assert entry.primary_interpretability_source == "attention_weights"
    top_feature_names = [item["feature"] for item in entry.primary_interpretability_signal]
    assert top_feature_names[0] == "gene_a"  # highest |value|
    # generic correlation signal must still be present, not dropped
    assert entry.secondary_correlation_signal == [{"feature": "gene_x", "value": 0.2}]


def test_nmf_candidate_uses_component_loadings_as_primary_signal():
    gate = _make_gate_result("c1")
    provenance = {"component_loadings": {"gene_a": 0.3, "gene_b": -0.9}}
    candidates = [_make_candidate("c1", method="nmf", provenance=provenance)]
    correction = CorrectionResult("c1", "nmf", 2.0, 2.0, 0.001, 500)
    report = CampaignReport("camp1", "cohort1", 0.05, [CampaignCandidateReport("c1", "nmf", gate, correction, True)])

    doc = build_report_document(report, candidates)
    entry = doc.candidates[0]
    assert entry.primary_interpretability_source == "component_loadings"
    assert entry.primary_interpretability_signal[0]["feature"] == "gene_b"  # highest |value| (-0.9)


def test_non_preferred_method_falls_back_to_correlation_signal():
    gate = _make_gate_result("c1", top_features=[FeatureAttribution("gene_z", "rnaseq", 0.6)])
    report, candidates = _make_report([("c1", "hierarchical_clustering", True, gate, 0.001)])
    doc = build_report_document(report, candidates)
    entry = doc.candidates[0]
    assert entry.primary_interpretability_source == "correlation"
    assert entry.primary_interpretability_signal == [{"feature": "gene_z", "value": 0.6}]


def test_attention_weight_stability_note_present_only_when_provenance_has_it():
    gate = _make_gate_result("c1")
    provenance = {"attention_weights": {"g": 0.5}, "attention_weight_stability": {"n_seeds_total": 2, "n_seeds_matched": 2}}
    candidates = [_make_candidate("c1", method="attention", provenance=provenance)]
    correction = CorrectionResult("c1", "attention", 2.0, 2.0, 0.001, 500)
    report = CampaignReport("camp1", "cohort1", 0.05, [CampaignCandidateReport("c1", "attention", gate, correction, True)])
    doc = build_report_document(report, candidates)
    assert doc.candidates[0].explanation_reproducibility_note is not None
    assert "consistently identified" in doc.candidates[0].explanation_reproducibility_note

    gate2 = _make_gate_result("c2")
    candidates2 = [_make_candidate("c2", method="hierarchical_clustering")]
    correction2 = CorrectionResult("c2", "hierarchical_clustering", 2.0, 2.0, 0.001, 500)
    report2 = CampaignReport("camp2", "cohort1", 0.05, [CampaignCandidateReport("c2", "hierarchical_clustering", gate2, correction2, True)])
    doc2 = build_report_document(report2, candidates2)
    assert doc2.candidates[0].explanation_reproducibility_note is None


def test_method_summaries_include_zero_accepted_methods():
    """Negative-result reporting per skill rule 5: a method with zero
    accepted candidates must still appear in method_summaries, not be
    silently absent because nothing survived.
    """
    gate = _make_gate_result("c1", calibration="noise", consistency_fraction=0.2, passed=False, failure_reasons=["x"])
    report, candidates = _make_report([("c1", "nmf", False, gate, 0.9)])
    doc = build_report_document(report, candidates)
    assert len(doc.method_summaries) == 1
    assert doc.method_summaries[0].method == "nmf"
    assert doc.method_summaries[0].n_candidates == 1
    assert doc.method_summaries[0].n_accepted == 0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
