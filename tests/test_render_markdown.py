import pytest

from discovery_engine.reporting.render_markdown import to_markdown
from discovery_engine.reporting.schema import ScopeLimitation, build_report_document
from tests.test_reporting_schema import _make_gate_result, _make_report

_BANNED = ["validated", "establishes", "proves", "statistically independent", "recapitulates", "the dominant signal"]


def test_unified_table_includes_accepted_and_rejected_with_no_separate_failures_section():
    accepted_gate = _make_gate_result("c1")
    rejected_gate = _make_gate_result(
        "c2", calibration="noise", consistency_fraction=0.2, passed=False,
        failure_reasons=["half-split consistency calibrated as noise (0.20)"],
    )
    report, candidates = _make_report([
        ("c1", "hierarchical_clustering", True, accepted_gate, 0.001),
        ("c2", "hierarchical_clustering", False, rejected_gate, 0.9),
    ])
    doc = build_report_document(report, candidates)
    md = to_markdown(doc)

    assert "## All candidates" in md
    assert "c1" in md and "c2" in md
    assert "half-split consistency calibrated as noise (0.20)" in md
    # a candidate that tried and failed is not relegated to a separate section
    assert "## Failures" not in md
    assert "## Appendix" not in md


def test_method_with_zero_accepted_still_reported():
    gate = _make_gate_result("c1", calibration="noise", consistency_fraction=0.1, passed=False, failure_reasons=["x"])
    report, candidates = _make_report([("c1", "nmf", False, gate, 0.9)])
    doc = build_report_document(report, candidates)
    md = to_markdown(doc)
    assert "| nmf | 1 | 0 |" in md


def test_no_banned_language_in_rendered_markdown_output():
    """Extra insurance on actual generated output, on top of language.py's
    own exhaustive-output test: this exercises the full render path, not
    just the phrase functions in isolation.
    """
    accepted_gate = _make_gate_result("c1")
    rejected_gate = _make_gate_result("c2", calibration="noise", consistency_fraction=0.1, passed=False, failure_reasons=["x"])
    report, candidates = _make_report([
        ("c1", "hierarchical_clustering", True, accepted_gate, 0.001),
        ("c2", "hierarchical_clustering", False, rejected_gate, 0.9),
    ])
    doc = build_report_document(report, candidates)
    md = to_markdown(doc).lower()
    for phrase in _BANNED:
        assert phrase not in md
    assert "—" not in to_markdown(doc)


def test_scope_limitations_rendered_prominently_before_candidate_table():
    gate = _make_gate_result("c1")
    report, candidates = _make_report([("c1", "hierarchical_clustering", True, gate, 0.001)])
    limitation = ScopeLimitation(
        method="attention", modality="rnaseq", feature_subset="hallmark_4384",
        reason="did not converge within the shared epoch budget at ~4450 features",
    )
    doc = build_report_document(report, candidates, scope_limitations=[limitation])
    md = to_markdown(doc)

    assert "## Scope limitations" in md
    assert "attention" in md and "hallmark_4384" in md
    assert "did not converge within the shared epoch budget" in md
    # must appear before the candidate table, not buried after it
    assert md.index("## Scope limitations") < md.index("## All candidates")


def test_no_scope_limitations_section_when_none_recorded():
    gate = _make_gate_result("c1")
    report, candidates = _make_report([("c1", "hierarchical_clustering", True, gate, 0.001)])
    doc = build_report_document(report, candidates)
    md = to_markdown(doc)
    assert "## Scope limitations" not in md


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
