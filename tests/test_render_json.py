import json

import pytest

from discovery_engine.reporting.render_json import to_json
from discovery_engine.reporting.schema import build_report_document
from tests.test_reporting_schema import _make_candidate, _make_gate_result, _make_report


def test_json_round_trips_and_contains_full_gate_detail():
    gate = _make_gate_result("c1", calibration="moderate", consistency_fraction=0.8)
    report, candidates = _make_report([("c1", "hierarchical_clustering", True, gate, 0.001)])
    doc = build_report_document(report, candidates)

    parsed = json.loads(to_json(doc))
    entry = parsed["candidates"][0]
    assert entry["gate"]["half_split"]["consistency_fraction"] == 0.8
    assert entry["gate"]["half_split"]["calibration"] == "moderate"
    assert entry["correction"]["campaign_p_value"] == 0.001
    assert entry["gate"]["multivariate"]["p_value"] == 0.01


def test_json_never_embeds_raw_candidate_or_refit():
    gate = _make_gate_result("c1")
    report, candidates = _make_report([("c1", "hierarchical_clustering", True, gate, 0.001)])
    doc = build_report_document(report, candidates)
    parsed = json.loads(to_json(doc))
    entry = parsed["candidates"][0]
    assert "refit" not in json.dumps(entry)


def test_json_includes_both_accepted_and_rejected_in_one_list():
    accepted_gate = _make_gate_result("c1")
    rejected_gate = _make_gate_result("c2", calibration="noise", consistency_fraction=0.2, passed=False, failure_reasons=["x"])
    report, candidates = _make_report([
        ("c1", "hierarchical_clustering", True, accepted_gate, 0.001),
        ("c2", "hierarchical_clustering", False, rejected_gate, 0.9),
    ])
    doc = build_report_document(report, candidates)
    parsed = json.loads(to_json(doc))
    ids = {c["candidate_id"] for c in parsed["candidates"]}
    assert ids == {"c1", "c2"}


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
