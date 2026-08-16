import pytest

from discovery_engine.validation.stability import requires_stability_check, stability_check


def test_classical_method_not_applicable():
    result = stability_check("hierarchical_clustering")
    assert result.applicable is False


def test_neural_method_requires_check_but_not_yet_implemented():
    assert requires_stability_check("autoencoder") is True
    with pytest.raises(NotImplementedError):
        stability_check("autoencoder")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
