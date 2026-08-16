import pytest

from discovery_engine.reporting import language

_BANNED = ["validated", "establishes", "proves", "statistically independent", "recapitulates", "the dominant signal"]


def test_no_banned_language_in_any_actual_template_or_generated_output():
    """Belt-and-suspenders insurance on top of the structural design (see
    language.py's module docstring).

    Originally scanned inspect.getsource(language) for banned phrases, but
    that false-positived on the module's own docstring, which legitimately
    NAMES the banned words to explain what's excluded (a real lesson from
    running this, not assumed away). Scanning source text conflates
    "documentation that mentions a word" with "a template that could emit
    it". Fixed by enumerating the actual, finite space of what every
    describe_* function can output (every calibration band, both
    acceptance outcomes, every novelty classification, every seed-match
    count that changes wording) and scanning ONLY that -- content that
    could actually reach a report, exhaustively, not a sample.
    """
    generated: list[str] = []
    generated += list(language.CALIBRATION_PHRASES.values())
    generated += list(language.NOVELTY_PHRASES.values())
    generated += [language.describe_reproducibility(band, 0.5) for band in language.CALIBRATION_PHRASES]
    generated += [language.describe_acceptance(accepted, 0.01, 0.05) for accepted in (True, False)]
    generated += [language.describe_novelty(cls) for cls in language.NOVELTY_PHRASES]
    generated += [
        language.describe_explanation_reproducibility({"n_seeds_total": 2, "n_seeds_matched": n})
        for n in (0, 1, 2)
    ]

    combined = " ".join(generated).lower()
    for phrase in _BANNED:
        assert phrase not in combined, f"banned phrase {phrase!r} found in generated output"
    assert "—" not in " ".join(generated), "em dash found in generated output"


def test_describe_reproducibility_bands():
    assert "not distinguishable from chance" in language.describe_reproducibility("noise", 0.42)
    assert "highly reproducible" in language.describe_reproducibility("robust", 0.95)
    assert "42%" in language.describe_reproducibility("noise", 0.42)


def test_describe_acceptance_accepted_vs_rejected():
    accepted = language.describe_acceptance(True, 0.001, 0.05)
    rejected = language.describe_acceptance(False, 0.42, 0.05)
    assert "reproducible candidate" in accepted
    assert "did not survive" in rejected
    for text in (accepted, rejected):
        for phrase in _BANNED:
            assert phrase not in text.lower()


def test_describe_novelty_all_three_classifications():
    assert "no detectable overlap" in language.describe_novelty("novel")
    assert "partial overlap" in language.describe_novelty("partially_overlapping")
    assert "substantial overlap" in language.describe_novelty("fully_redundant")
    with pytest.raises(KeyError):
        language.describe_novelty("not_a_real_classification")


def test_describe_explanation_reproducibility_robust_vs_weak():
    robust = language.describe_explanation_reproducibility(
        {"n_seeds_total": 2, "n_seeds_matched": 2, "min_abs_correlation_across_seeds": 0.95}
    )
    weak = language.describe_explanation_reproducibility(
        {"n_seeds_total": 2, "n_seeds_matched": 1, "min_abs_correlation_across_seeds": 0.3}
    )
    assert "consistently identified" in robust
    assert "reproducibility not confirmed" not in robust  # robust case shouldn't hedge
    assert "identified in this fit" in weak
    assert "1/2 other seeds matched" in weak


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
