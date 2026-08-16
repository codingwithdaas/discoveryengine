"""Phase 5 proof: build_report_document, to_json, and to_markdown against
a REAL CampaignReport from an actual campaign run, not just hand-built
GateResult/CorrectionResult fixtures (test_reporting_schema.py already
covers those in isolation). Mirrors test_gate_end_to_end.py's planted-
module pattern.
"""

import json

import numpy as np
import pandas as pd
import pytest

from discovery_engine.campaign.campaign import run_campaign
from discovery_engine.data.cohort import CohortData
from discovery_engine.discovery.clustering import HierarchicalClusteringDiscovery
from discovery_engine.outcomes import OutcomeSpec, OutcomeType
from discovery_engine.reporting.render_json import to_json
from discovery_engine.reporting.render_markdown import to_markdown
from discovery_engine.reporting.schema import build_report_document
from discovery_engine.signatures.registry import SignatureRegistry

_BANNED = ["validated", "establishes", "proves", "statistically independent", "recapitulates", "the dominant signal"]


def test_report_document_against_a_real_campaign_run(tmp_path):
    rng = np.random.default_rng(0)
    n_samples = 150
    sample_ids = [f"S{i:04d}" for i in range(n_samples)]
    latent = rng.normal(size=n_samples)
    module_features = {f"module_gene_{i}": latent + rng.normal(scale=0.3, size=n_samples) for i in range(6)}
    noise_features = {f"noise_gene_{i}": rng.normal(size=n_samples) for i in range(20)}
    df = pd.DataFrame({**module_features, **noise_features}, index=sample_ids)
    data = CohortData(modalities={"rnaseq": df})

    outcome = OutcomeSpec(
        OutcomeType.CONTINUOUS, sample_ids, 0.6 * latent + np.sqrt(1 - 0.36) * rng.normal(size=n_samples)
    )

    candidates = HierarchicalClusteringDiscovery().discover(data, modality="rnaseq", k_values=[2, 4, 6])
    registry = SignatureRegistry(tmp_path / "signatures.db")
    campaign_report = run_campaign(
        campaign_id="camp_report_test", cohort_id="cohort1", candidates=candidates, data=data, outcome=outcome,
        signature_registry=registry, correction_n_permutations=100, gate_n_permutations=100, seed=1,
    )

    doc = build_report_document(campaign_report, candidates)

    # nothing silently dropped: every discovered candidate has a report entry
    assert doc.n_candidates_total == len(candidates)
    assert {e.candidate_id for e in doc.candidates} == {c.id for c in candidates}
    assert doc.n_accepted + doc.n_rejected == doc.n_candidates_total

    # JSON: full round trip, every candidate present, gate detail intact
    parsed = json.loads(to_json(doc))
    assert len(parsed["candidates"]) == len(candidates)
    assert all("half_split" in c["gate"] for c in parsed["candidates"])

    # Markdown: unified table, no banned language, method summary present
    md = to_markdown(doc)
    assert "## All candidates" in md
    assert "hierarchical_clustering" in md
    md_lower = md.lower()
    for phrase in _BANNED:
        assert phrase not in md_lower


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
