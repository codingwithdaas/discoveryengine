"""Explicit interaction search: outcome-blind (like clustering.py,
louvain.py, and nmf.py), fully deterministic (like clustering.py -- no
seed parameter exists because there is nothing stochastic to seed, and
nothing was measured because there was nothing to measure; this is a
different claim from EMPIRICALLY_EXEMPTED_METHODS' "measured and cleared
the bar", it's simply "no randomness exists here at all").

The central design question this method raised: interaction-with-outcome
sounds inherently outcome-informed, so the natural first assumption is
that it needs OutcomeInformedDiscoveryMethod, refit, and a per-candidate
permutation-cost budget the way DeepSurv/RSF/attention do. It doesn't.
Score construction (below) never touches outcome, so this is genuinely
outcome-blind: no refit exists, no permutation-cost budget to manage,
outcome-blind candidates are already recomputed for free in
campaign.correction._null_statistics via direct re-scoring against the
permuted outcome.

SCORE CONSTRUCTION, and the two wrong mental models corrected in sequence
before landing here (the specificity test caught both, empirically, before
anything was committed -- worth keeping the record, not just the fix):

For a fixed pair (a, b), the candidate's score is meant to isolate the
INTERACTION between a and b, not a main-effect artifact leaking through a
raw product term. The intended mechanism is residualization: fit the
interaction term against both main effects (plus intercept) via closed-
form OLS, keep the residual. By the Frisch-Waugh-Lovell theorem, this
residual is EXACTLY Pearson-orthogonal to both main effects (a hard
linear-algebra fact about OLS residuals: they are orthogonal to the
column space of whatever design matrix produced them, regardless of
collinearity).

Attempt 1 (wrong assumption): built a "main effects only, no true
interaction" adversarial cohort with INDEPENDENT Gaussian features, and
neither a naive nor a residualized product term showed spurious
association -- no contrast to observe. Investigated: for jointly Gaussian
mean-zero (z_a, z_b), Cov(z_a*z_b, z_a) = 0 EXACTLY regardless of any
correlation between a and b (Isserlis' theorem: all third moments of a
mean-zero Gaussian vector vanish). The classic "naive product term picks
up main-effect bias" failure mode does not occur for independent-or-
correlated GAUSSIAN features at all; it requires non-Gaussian (skewed)
structure, which is also the realistic case for genomic data (expression
counts, abundance measures are commonly right-skewed).

Attempt 2 (the real gap): rebuilt the adversarial cohort with correlated,
SKEWED features (Gamma/Exponential, shared-latent-factor correlated,
correlation ~0.84). This is where residualizing raw z-scored values broke
down, verified directly, not inferred:
    Pearson corr(residual, z_a):               -1.3e-16  (exactly zero, as OLS guarantees)
    Pearson corr(residual, z_b):                -8.5e-17  (exactly zero, as OLS guarantees)
    Pearson corr(residual, y_cont with noise):    0.014   (negligible, consistent with the above)
    Spearman rho(residual, y_cont):              -0.383, p = 6.0e-12   (STRONGLY significant)
The residual was doing exactly what OLS guarantees -- exact Pearson
orthogonality to the main effects. But outcomes.association_statistic's
CONTINUOUS branch uses Spearman (rank-based, nonlinear), and OLS's
orthogonality guarantee is a fact about LINEAR correlation; it says
nothing about rank correlation. On skewed, correlated features the
residual carried a strong nonlinear/rank relationship to a main-effects-
only outcome that Pearson orthogonality never touched. The original
docstring here called this "well-motivated preprocessing, not an exact
algebraic identity" -- true, but understated: this is a concrete,
demonstrable false-positive risk under realistic conditions, not a rare
edge case or a theoretical footnote.

THE FIX, and its PRECISE (not assumed) revised guarantee: residualize on
RANKS instead of raw z-scores. Each feature is rank-transformed
(scipy.stats.rankdata, then standardized) before the product and the OLS
residualization step, so the design matrix Z-side is [1, rank(a), rank(b)]
throughout. This gives an EXACT, provable claim of the same kind as
before: the raw residual is Pearson-orthogonal to rank(a) and rank(b),
exactly, by the identical OLS projection argument, just applied on the
rank scale instead of the raw-value scale. This is NOT, by itself, a
proof that Spearman(residual, outcome) = 0 in general: outcomes.
association_statistic's CONTINUOUS branch calls scipy's spearmanr, which
RANKS the residual again internally before correlating; rank(residual) is
a further nonlinear transform of a quantity only the RAW residual is
guaranteed orthogonal to rank(a)/rank(b), and re-ranking has no automatic
orthogonality guarantee of its own. What actually stands behind this
fix's specificity claim is empirical verification against the SAME
adversarial cohort that broke the raw-value version (correlated, skewed
features), not a second mathematical proof:
    Pearson corr(residual, rank(a)):                1.1e-16   (exact, as guaranteed)
    Pearson corr(residual, rank(b)):                1.2e-17   (exact, as guaranteed)
    Spearman rho(residual, y_cont, main-effects-only):  -0.00055, p = 0.992   (clean)
    permutation_test p-value (CONTINUOUS, main-effects-only): 0.992
    permutation_test p-value (BINARY, main-effects-only):     0.852
    permutation_test p-value (CONTINUOUS, true interaction present): 0.002
A real, large practical improvement, verified with real numbers on the
exact case that failed before, not a certificate covering every possible
data-generating process.

SURVIVAL: checked, and it fails. This was originally left as "a stated
gap, not an assumed pass" -- checked afterward, before real data, per
explicit instruction not to carry an unresolved gap forward. Same
adversarial cohort (correlated, skewed features), SURVIVAL outcome (Cox
proportional hazards simulation, main effects only, no true interaction),
production rank-residualized score:
    single realization (seed=11):      Cox Wald z = 2.62, permutation p = 0.008
    20 independent realizations:       3/20 significant at alpha=0.05
                                        (expect ~1/20=5% under a calibrated null),
                                        two of the three hit the permutation floor
The rank-based fix was motivated by aligning with Spearman's rank-
invariance (CONTINUOUS); there was never a real reason to expect it to
transfer to Cox's partial likelihood, which depends on actual covariate
VALUES within each risk set, not just their ranks, and is not invariant
to monotonic transforms of the covariate the way Spearman is. Smaller in
magnitude than the original raw-value bug (~15% false-positive rate here
vs an essentially deterministic failure there), but real, not a rounding
error, and not something campaign-level correction can fix on its own
(correction handles multiple comparisons across candidates; it doesn't
repair a single candidate's own miscalibrated null).

Because this method is outcome-blind by design, discover() never sees
the outcome and cannot structurally refuse to run against SURVIVAL the
way an outcome-informed method could validate outcome.type directly.
Guarded instead at the one place candidates and outcome actually meet:
discovery.base.UNVERIFIED_OUTCOME_TYPES + assert_outcome_compatible,
called from campaign.campaign.run_campaign before anything else runs.
Do not remove "interaction_search" from that dict until a SURVIVAL-
specific fix is verified to the same standard as CONTINUOUS/BINARY were
here, with real numbers, not by assumption.

No gate-level re-check of "does this add value beyond its own main
effects" is needed: the residualization already targets that at score-
construction time, not something validation.multivariate needs to
separately re-verify against ad hoc comparators.

Every pair in `feature_pairs` becomes its own logged Candidate feeding the
real campaign correction pool -- identical to every K, every resolution,
every rank in every other sweep-based method built so far. No post-hoc
pruning after computing outcome-association (that would repeat RSF's
importances-as-candidates double-dipping problem, see rsf.py). The
combinatorial-scale problem this method could otherwise create (all pairs
over thousands of features is untenable) is NOT solved by filtering after
the fact; it's solved the same way every other method already bounds its
sweep: the caller supplies a bounded, explicit list up front, matching
clustering.py's k_values, louvain.py's resolutions, nmf.py's
n_components_options.

CALLER OBLIGATION, stated explicitly because this module cannot enforce it
structurally: `feature_pairs` must be constructed without reference to the
outcome. Every other outcome-blindness guarantee in this project is
provable from a function signature alone (no `outcome` parameter exists to
leak from). This one is different and weaker: the pair SELECTION happens
entirely outside this module, before discover() is ever called, so there
is no signature that could prove the list wasn't built by eyeballing which
pairs "look interesting" against the true outcome first. If that happened,
the double-dipping problem RSF's design avoided would reappear here
through the back door: the true candidates would have benefited from an
outcome-informed selection step that no permutation replicate re-runs.
This module trusts the caller on this exactly once, in the docstring, not
silently -- an auto-generation convenience (e.g. an outcome-blind
correlation- or variance-based heuristic for producing `feature_pairs`)
was deliberately not built here, to avoid taking on that trust
implicitly; pass an explicit list built from outcome-blind criteria
(modality co-membership, a feature-feature correlation threshold, prior
biological pathway membership, etc).

Single-modality scope (`modality: str`, not `modalities: list[str]`),
matching the classical-method precedent (clustering.py, louvain.py,
nmf.py), not the multi-modal shape of the neural methods.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.stats import rankdata

from discovery_engine.campaign.candidate import Candidate
from discovery_engine.data.cohort import CohortData

logger = logging.getLogger(__name__)


class InteractionSearchDiscovery:
    name = "interaction_search"

    def discover(
        self,
        data: CohortData,
        modality: str,
        feature_pairs: list[tuple[str, str]],
    ) -> list[Candidate]:
        df = data.modalities[modality]
        sample_ids = list(df.index)

        candidates: list[Candidate] = []
        for a, b in feature_pairs:
            if a == b:
                raise ValueError(f"interaction pair ({a!r}, {a!r}) is degenerate: a feature paired with itself")

            r_a, ok_a = _rank_z(df[a])
            r_b, ok_b = _rank_z(df[b])
            if not (ok_a and ok_b):
                logger.info("interaction search: skipping pair (%s, %s), a constant (zero-variance) feature", a, b)
                continue

            residual = _residualize_interaction(r_a, r_b)
            scores = {sid: float(residual[i]) for i, sid in enumerate(sample_ids)}
            candidates.append(
                Candidate(
                    id=f"interaction_{modality}_{a}_{b}",
                    method=self.name,
                    method_params={"modality": modality, "feature_a": a, "feature_b": b},
                    outcome_blind=True,
                    modality_touch=[modality],
                    sample_ids=sample_ids,
                    scores=scores,
                    provenance={"feature_pair": [a, b], "modality": modality},
                )
            )
        return candidates


def _rank_z(series) -> tuple[np.ndarray, bool]:
    """Rank-transform then standardize, rather than z-score raw values --
    see module docstring for why: the downstream association test is
    rank-based (Spearman) for CONTINUOUS outcomes, and residualizing on
    ranks is what makes the OLS orthogonality guarantee land on the scale
    that actually matters, instead of a scale (raw values) the downstream
    test doesn't use.
    """
    ranks = rankdata(series.to_numpy(dtype=float))
    std = ranks.std(ddof=0)
    if std == 0:
        return ranks, False
    return (ranks - ranks.mean()) / std, True


def _residualize_interaction(r_a: np.ndarray, r_b: np.ndarray) -> np.ndarray:
    """The Frisch-Waugh-Lovell step, on ranks: residualize the interaction
    term r_a * r_b against both rank-transformed main effects (plus
    intercept). Exactly Pearson-orthogonal to r_a and r_b by construction
    (OLS residuals are always orthogonal to their own design matrix's
    column space); NOT a proof that the downstream Spearman test is
    exactly zero under a true null, since spearmanr ranks this residual
    again -- see module docstring for the empirical verification standing
    behind that claim instead.
    """
    product = r_a * r_b
    design = np.column_stack([np.ones_like(r_a), r_a, r_b])
    beta, *_ = np.linalg.lstsq(design, product, rcond=None)
    fitted = design @ beta
    return product - fitted
