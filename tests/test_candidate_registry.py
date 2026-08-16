from discovery_engine.campaign.candidate import Candidate
from discovery_engine.campaign.registry import CandidateRegistry


def _blind_candidate(cid: str) -> Candidate:
    return Candidate(
        id=cid,
        method="test_method",
        method_params={},
        outcome_blind=True,
        modality_touch=["m1"],
        sample_ids=["a", "b", "c"],
        scores={"a": 1.0, "b": 2.0, "c": 3.0},
    )


def test_add_and_retrieve():
    reg = CandidateRegistry()
    reg.add(_blind_candidate("c1"))
    assert len(reg) == 1
    assert reg.all()[0].id == "c1"


def test_duplicate_id_rejected():
    reg = CandidateRegistry()
    reg.add(_blind_candidate("c1"))
    try:
        reg.add(_blind_candidate("c1"))
        assert False, "expected ValueError on duplicate id"
    except ValueError:
        pass


def test_outcome_informed_without_refit_rejected():
    reg = CandidateRegistry()
    c = _blind_candidate("c1")
    c.outcome_blind = False
    c.refit = None
    try:
        reg.add(c)
        assert False, "expected ValueError: outcome-informed candidate needs a refit fn"
    except ValueError:
        pass


def test_save_and_load_roundtrip(tmp_path):
    reg = CandidateRegistry()
    reg.add(_blind_candidate("c1"))
    reg.add(_blind_candidate("c2"))
    path = tmp_path / "registry.jsonl"
    reg.save(path)

    loaded = CandidateRegistry.load(path)
    assert len(loaded) == 2
    ids = {c.id for c in loaded.all()}
    assert ids == {"c1", "c2"}
    assert loaded.all()[0].scores == {"a": 1.0, "b": 2.0, "c": 3.0}


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-q"]))
