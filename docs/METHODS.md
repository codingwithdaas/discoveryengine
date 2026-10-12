# Discovery Engine: Methods and Design

This is the detailed reference. For a plain-language overview, see the main [README](../README.md). For a hands-on walkthrough, see [HOW_TO_USE](../HOW_TO_USE.md).

A discovery-first, validation-gated framework for finding real biomarker signatures in high-dimensional, correlated genomics and histology data (computational morphometric biomarkers, RNA-seq, RPPA protein, and beyond).

Built from methodological standards developed empirically across an extended real analysis project on TCGA-BLCA data, where the cost of getting small-sample, high-dimensional statistics wrong was learned the hard way, repeatedly, before this engine existed to prevent it structurally.

## Philosophy

This is a **discovery** engine, not a validation-only tool. Its job is to actively search for real, previously unknown structure using many different methods, linear and non-linear, classical and neural, not just to check candidates a human already proposed. But every single candidate that comes out of discovery, however it was found, must survive a rigorous, standardized validation gate before being reported as a finding.

**Discovery is allowed to be exploratory and aggressive. Reporting must never be.**

The single most important architectural principle: if a campaign runs multiple discovery methods, each producing multiple candidates, the true multiple-testing burden is the *full pool* of candidates across the entire campaign, not the count within any one method. Every candidate ever generated is tracked in one registry, and statistical correction is applied across that full pool before anything is reported as significant, not method by method.

## What's inside

### Campaign correction (`campaign/`)
- `candidate.py`: the unit of discovery output; a `Candidate` carries an optional `refit` closure for outcome-informed methods, so a valid permutation null can genuinely retrain the candidate under permutation rather than just re-score a fixed fit.
- `correction.py`: studentized maxT / Westfall-Young correction across the full candidate pool. Fully bit-reproducible given a fixed seed, including every retrained outcome-informed candidate. Refuses fewer than 1,000 permutations (studentization needs a well-estimated null SD; at 50 permutations real-data runs produced false acceptances) and refuses duplicate candidate ids.
- `registry.py`: append-only candidate log. Structurally rejects outcome-informed candidates with no refit closure; they cannot be validly corrected without one.
- `campaign.py`: top-level orchestrator: correction over the full pool first, then the per-candidate gate, with outcome-type compatibility checked before anything runs.

### Validation gate (`validation/`)
Every candidate, from every method, goes through the same checks. Not all of them decide pass/fail; each candidate's gate result records which checks failed, which were skipped and why, and which are advisory.

**Gating** (can reject a candidate): half-split reproducibility; Freedman-Lane covariate adjustment (continuous and binary outcomes only); adds-value beyond known signatures and novelty (only when known signatures are loaded in the registry); seed stability (stability-required methods); interpretability.

**Advisory** (recorded, never reject on their own): bootstrap confidence interval; covariate-stratified test; median-split vs extreme-groups crosscheck.

**Skipped** checks are listed with a reason, for example Freedman-Lane on survival outcomes, which is not implemented. A skipped check is not a passed check.

The checks:
- Bootstrap direction-consistency (≥30 independent random half-splits) and true bootstrap confidence intervals, kept as two distinct, correctly-labeled methods, not conflated.
- Grade/covariate independence, via stratified testing and Freedman-Lane covariate-adjusted permutation.
- Median-split vs. extreme-groups design crosscheck, with a confound-only rerun when the two designs disagree.
- "Adds value beyond X": tested jointly against every already-accepted signature, not just alone.
- Seed/architecture stability for stochastic (neural) discovery methods, a single trained model is one sample of a random process, not a fact about the data; structure must reproduce across independently retrained runs, measured and recorded even for partial matches.
- Novelty scoring against every previously accepted signature.
- Interpretability: no candidate is reported without a feature-attribution signal, native (attention weights, NMF component loadings) where the method produces one, correlation-based otherwise.

### Discovery methods (`discovery/`)
Seven methods, each explicitly categorized as outcome-blind or outcome-informed, deterministic or stochastic, with empirical evidence backing every category placement:

| Method | Type | Stability |
|---|---|---|
| Hierarchical clustering | Outcome-blind | Deterministic |
| Louvain community detection | Outcome-blind | Stability-required (exempted on synthetic evidence, then measured unstable on real RNA-seq: membership changed with the seed) |
| NMF | Outcome-blind | Stability-required (measured *unstable* at high rank, evidence, not architecture, decided this) |
| Explicit interaction search | Outcome-blind | Deterministic (closed-form residualization) |
| Autoencoder / VAE | Outcome-blind | Stability-required |
| DeepSurv | Outcome-informed | Stability-required |
| Random Survival Forest | Outcome-informed | Empirically exempted |
| Attention | Outcome-informed | Stability-required (prediction and explanation reproducibility measured separately) |

### Reporting (`reporting/`)
- Structural, not linted, language discipline: a closed set of phrase templates is the *only* place evidentiary language is written. There is no free-text generation path for claims like "validated" or "statistically independent" to leak through.
- Every candidate a campaign ever produced appears in one unified report, accepted and rejected together, negative results are part of the record, not an appendix.
- Deliberately unattempted method × modality × feature-subset combinations are recorded explicitly (`scope_limitations`), never left to be inferred from absence.
- One canonical `CampaignReportDocument` feeds both JSON and Markdown renderers, so the two formats cannot drift apart from each other.

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate   # or source .venv/bin/activate on macOS/Linux
pip install -e .
```

## Data

**This repository intentionally contains no data.** Raw and real-data-derived outputs (cohort files, campaign reports, logs) are excluded from version control by design (`.gitignore`), not by convention alone, see `data/` and `outputs/` patterns. Point loaders such as `data/tcga_blca.py` at your own locally-governed copy of the relevant cohort data; nothing here assumes or requires a specific data location to be committed to the repo.

## Status

Research software, not a clinical tool. Any signature it reports is a discovery-stage candidate from a single cohort until independently replicated.

All core phases are built and tested against synthetic data with known planted signals before ever touching real data. The engine has since been run against real TCGA-BLCA data (CMB histology + RNA-seq, disease-specific survival outcome), including catching and fixing a genuine candidate-ID collision bug mid-campaign. Those campaigns were run with 50 permutations; rerun with 10,000, none of their accepted candidates survived correction. The engine now refuses fewer than 1,000 permutations.

## Methodology standards

The codebase is built against an explicit set of rules: outcome-blind discovery, tracked K-selection as its own layer of multiple comparisons, bootstrap reproducibility over single p-values, cross-modality validation, structural (not linted) language discipline, and more, each one tied to a real finding from the analysis work this engine grew out of.

## License

Not yet specified.
