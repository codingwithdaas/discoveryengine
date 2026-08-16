import pytest

from discovery_engine.campaign.candidate import Candidate
from discovery_engine.signatures.registry import SignatureRegistry


def _candidate(cid):
    return Candidate(
        id=cid, method="hierarchical_clustering", method_params={"k": 4},
        outcome_blind=True, modality_touch=["rnaseq"], sample_ids=["a", "b", "c"],
        scores={"a": 1.0, "b": 2.0, "c": 3.0}, provenance={"member_features": ["g1", "g2"]},
    )


def test_add_and_retrieve_roundtrip(tmp_path):
    reg = SignatureRegistry(tmp_path / "sig.db")
    reg.add(_candidate("sig1"), cohort_id="blca", campaign_id="camp1", validation_summary={"passed": True})
    assert len(reg) == 1

    candidates = reg.all_as_candidates()
    assert len(candidates) == 1
    c = candidates[0]
    assert c.id == "sig1"
    assert c.scores == {"a": 1.0, "b": 2.0, "c": 3.0}
    assert c.provenance == {"member_features": ["g1", "g2"]}
    assert c.outcome_blind is True


def test_scoped_by_cohort(tmp_path):
    reg = SignatureRegistry(tmp_path / "sig.db")
    reg.add(_candidate("sig_blca"), cohort_id="blca", campaign_id="c1", validation_summary={})
    reg.add(_candidate("sig_luad"), cohort_id="luad", campaign_id="c2", validation_summary={})

    assert len(reg.all_as_candidates()) == 2
    blca_only = reg.all_as_candidates(cohort_id="blca")
    assert [c.id for c in blca_only] == ["sig_blca"]


def test_persists_across_reopen(tmp_path):
    path = tmp_path / "sig.db"
    reg = SignatureRegistry(path)
    reg.add(_candidate("sig1"), cohort_id="blca", campaign_id="c1", validation_summary={})
    reg.close()

    reg2 = SignatureRegistry(path)
    assert len(reg2) == 1


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
