"""The only place in this project's reporting layer where evidentiary
language (words describing HOW STRONG or HOW REAL a finding is) gets
written. Every other reporting module calls these functions and only ever
interpolates NUMBERS into their output; there is no free-text composition
path elsewhere that could do this module's job differently.

Skill rule 12's banned vocabulary is deliberately absent from every
template here: "validated", "establishes", "proves", "statistically
independent" (use "no detectable overlap/rank correlation" instead),
"recapitulates". "The dominant signal" has no template at all, not
filtered out, simply never written: no check in this project's validation
gate formally establishes dominance over other accepted signatures, and
rule 12 only permits that phrase when dominance is formally shown. No em
dashes in generated text, matching the project's own documentation
convention.

Precision about what kind of guarantee this is, stated once here rather
than assumed at every call site: this is NOT the same category of
guarantee as discovery.base's outcome-blindness (enforced by Python's own
call semantics -- there is no `outcome` parameter, so no code path can
leak one) or Candidate.refit's contract (enforced by CandidateRegistry.add
structurally rejecting outcome-informed candidates with no refit). Python
has no compile-time way to forbid some other function, anywhere in this
codebase, from hand-writing "this candidate is validated" as a string
literal. What this module actually gives you: there is no free-text
GENERATION step to begin with -- no function that takes a candidate and
returns prose describing it in the way an LLM or a template-composing
report writer would. A banned word entering a report requires someone to
hand-edit one of the small dicts below and add it there, in one small,
auditable file, not something that could vary per-candidate or drift in
silently as reporting code grows elsewhere. test_language.py adds a
banned-word scan as belt-and-suspenders insurance on top of the
structural design, not a replacement for it. It enumerates the actual,
finite output of every describe_* function across its full input space
(every calibration band, both acceptance outcomes, every novelty
classification), not this module's raw source text -- a first version
scanned source text and false-positived on this very docstring, which
legitimately names the banned words to explain what's excluded. Scanning
real, exhaustively-enumerable output is both more accurate and a
fundamentally smaller, safer thing to check than linting free-generated
prose after the fact, which is the (a) design this project chose against.
"""

from __future__ import annotations

from discovery_engine.validation.bootstrap import calibration_band

CALIBRATION_PHRASES: dict[str, str] = {
    "noise": "not distinguishable from chance reproducibility",
    "weak": "weakly reproducible",
    "moderate": "moderately reproducible",
    "robust": "highly reproducible",
}

NOVELTY_PHRASES: dict[str, str] = {
    "novel": "no detectable overlap with previously accepted signatures",
    "partially_overlapping": "partial overlap with at least one previously accepted signature",
    "fully_redundant": "substantial overlap with a previously accepted signature",
}


def describe_reproducibility(calibration: str, consistency_fraction: float) -> str:
    return f"{CALIBRATION_PHRASES[calibration]} ({consistency_fraction:.0%} half-split agreement)"


def describe_acceptance(accepted: bool, campaign_p_value: float, alpha: float) -> str:
    if accepted:
        return (
            f"reproducible candidate, consistent with an association with the outcome "
            f"(campaign-corrected p = {campaign_p_value:.4f} < alpha {alpha})"
        )
    return (
        f"did not survive campaign-level correction "
        f"(campaign-corrected p = {campaign_p_value:.4f} >= alpha {alpha})"
    )


def describe_novelty(overall_classification: str) -> str:
    return NOVELTY_PHRASES[overall_classification]


def describe_explanation_reproducibility(attention_weight_stability: dict) -> str:
    """Resolves the Phase 5 deferral named in attention.py's module
    docstring: a prediction can be reproducible across seeds while its
    EXPLANATION isn't. Deliberately a separate phrase from
    describe_reproducibility, built from attention_weight_stability's
    calibration, not the candidate's own prediction-level half_split
    calibration.
    """
    n_total = attention_weight_stability["n_seeds_total"]
    n_matched = attention_weight_stability["n_seeds_matched"]
    fraction = n_matched / n_total if n_total else 0.0
    calibration = calibration_band(fraction)
    if calibration == "robust":
        return "the feature attribution was consistently identified across independent retrains"
    return (
        f"the feature attribution was identified in this fit; reproducibility across "
        f"independent retrains was {CALIBRATION_PHRASES[calibration]} "
        f"({n_matched}/{n_total} other seeds matched)"
    )
