# How to Use the Discovery Engine

A practical guide to running a real discovery campaign, written so someone new to this codebase can actually do it, not just read about the architecture. If you want the "why," see `README.md` and `SKILL.md`. This document is about the "how."

## What this tool actually does, in plain terms

You give it a cohort of patients with some combination of data types (histology features, gene expression, protein levels, whatever you have) and an outcome you care about (survival, treatment response, anything). It then:

1. **Searches** for patterns using several different methods at once, some simple (grouping correlated features together), some complex (neural networks).
2. **Tests every single pattern it finds** against a strict checklist: does it reproduce on a different subset of patients, does it hold up once you account for things like tumor grade, does it tell you anything new beyond what you already know, can it be explained in terms of real, named features rather than being a black box.
3. **Corrects for the fact that it tried a lot of things.** If you search 50 different ways, some will look interesting purely by chance. The tool accounts for the total number of things it tried, not just the one that looked best, before telling you anything is real.

Only patterns that survive all of that get reported as findings. Everything else, including every failed attempt, still gets recorded, so you can see exactly what was tried and why it didn't hold up.

## Before you start: install it

```bash
git clone https://github.com/codingwithdaas/discoveryengine.git
cd discoveryengine
python -m venv .venv
.venv\Scripts\activate      # macOS/Linux: source .venv/bin/activate
pip install -e .
```

## Step 1: Get your data into the shape the tool expects

The tool works with a `CohortData` object: a set of feature tables (one per data type, called a "modality"), all indexed by the same patient ID, plus an outcome and optionally some covariates you want to check for (like tumor grade).

```python
import pandas as pd
from discovery_engine.data.cohort import CohortData
from discovery_engine.outcomes import OutcomeSpec, OutcomeType

# Each modality is just a table: rows = patients, columns = features
histology_features = pd.read_excel("my_histology_data.xlsx", index_col="PatientID")
gene_expression = pd.read_csv("my_rnaseq_data.txt", sep="\t", index_col=0).T

cohort = CohortData(
    modalities={
        "histology": histology_features,
        "rnaseq": gene_expression,
    },
    covariates=clinical_df[["grade_high"]],  # optional, for checking confounds
)

outcome = OutcomeSpec(
    type=OutcomeType.SURVIVAL,           # or BINARY, or CONTINUOUS
    sample_ids=list(clinical_df.index),
    values=clinical_df[["time_months", "event"]].values,
)
```

**This step is the one most likely to have real data-quality issues, and they will not announce themselves loudly.** Every real cohort this tool has been run against so far has needed at least one fix here: patient IDs that don't match exactly across files (a common one, especially between histology/clinical data at the patient level and expression data at the sample level, which often carries an extra suffix), genes or features that never vary across your cohort, and duplicate feature names that silently overwrite each other unless de-duplicated. **Check your loaded data before trusting it, don't assume a clean read means a clean dataset.**

## Step 2: A quick sanity check before running anything real

Before committing to a full campaign, run one discovery method by itself and look at what comes out:

```python
from discovery_engine.discovery.clustering import HierarchicalClusteringDiscovery

candidates = HierarchicalClusteringDiscovery().discover(
    cohort, modality="histology", k_values=[4, 6, 8]
)
print(f"{len(candidates)} candidates found")
```

If this runs in a few seconds and gives you a reasonable number of candidates, your data is probably loaded correctly. If it errors out or gives you something obviously wrong (zero candidates, or a huge number instantly), stop and check your data before going further, this is a cheap check, not a step to skip.

## Step 3: Estimate the real cost before running a full campaign

**Do this every time, even if you've run a campaign before on similar data.** Different cohorts and modality combinations have genuinely different costs, and guessing wrong means either wasting hours or, worse, launching something so slow it never finishes.

```python
import time

# Time a single statistical test on your actual data, at your actual sample size
start = time.time()
# ... run one representative check, e.g. one Cox model fit if using survival ...
per_fit_seconds = time.time() - start

# Multiply by roughly how many such tests the validation gate will run per candidate
# (bootstrap CI, half-split checks, design crosscheck, etc. — expect several hundred per candidate)
```

Run discovery alone first (no validation, no correction) to get a real candidate count. Multiply that by your measured per-fit cost. If the estimate comes out to more than a few minutes, don't launch it as one long blocking call, run it in the background and check on it periodically rather than waiting with no visibility into progress.

## Step 4: Run the actual campaign

```python
from discovery_engine.campaign.campaign import run_campaign
from discovery_engine.discovery.clustering import HierarchicalClusteringDiscovery
from discovery_engine.discovery.louvain import LouvainDiscovery
# ... import whichever discovery methods you want to use ...

all_candidates = []
all_candidates += HierarchicalClusteringDiscovery().discover(cohort, modality="histology", k_values=[4, 6, 8])
all_candidates += LouvainDiscovery().discover(cohort, modality="histology", resolutions=[0.5, 1.0, 1.5])
# ... run every discovery method you want included ...

report = run_campaign(
    candidates=all_candidates,
    data=cohort,
    outcome=outcome,
    correction_n_permutations=1000,
    alpha=0.05,
)

print(f"{len(report.accepted)} accepted, {len(report.rejected)} rejected, out of {len(all_candidates)} total")
```

**One rule that matters more than any other: if you're combining multiple discovery methods or multiple feature subsets of the same data, run correction once, over everything together, at the end.** Never run it separately per method or per subset. Splitting correction defeats the entire reason this framework exists, it's supposed to account for the total number of things you tried, and it can't do that if you hide part of the search from it.

## Step 5: Write the report

```python
from discovery_engine.reporting.schema import build_report_document
from discovery_engine.reporting.render_json import to_json
from discovery_engine.reporting.render_markdown import to_markdown

doc = build_report_document(report, all_candidates)

with open("my_campaign_report.md", "w") as f:
    f.write(to_markdown(doc))
```

The report lists every candidate, accepted or rejected, with the specific reason for its outcome. Read the rejected ones too, not just the accepted ones, a method that consistently fails one specific check tells you something real about your data.

## What to actually check when something gets accepted

A candidate surviving the gate and correction is a real, statistically defensible result. It is not automatically a finished, publication-ready finding. Before trusting it further:

1. **Check what it's actually made of.** Look at the top contributing features (in the report's interpretability section). Does the pattern make sense, or does it look like noise that happened to survive?
2. **Check whether it's actually new.** The novelty score tells you if it overlaps with anything you already knew. A candidate that's 90% correlated with something you've already validated isn't a new finding.
3. **Check it against an obvious confound directly**, don't just trust that the automated check passed. If you have a covariate you're worried about (grade, stage, batch, anything), look at the actual numbers, not just the pass/fail flag.

## Practical lessons learned running this for real (read this before your first real campaign)

These are not hypothetical warnings, every one of them is something that actually happened running this tool against real data:

- **Real data will have zero-variance features.** A gene that's simply never expressed in your tissue type, a feature that's constant across your whole cohort, these break naive correlation-based methods (they produce undefined results). The discovery methods in this codebase handle this, but if you write your own, check for it.
- **If you run the same method against multiple different feature subsets of the same modality** (say, testing both a small curated gene set and a larger one), **make sure your candidate IDs are unique across subsets.** Two structurally identical candidates from different subsets can otherwise silently overwrite each other's results. Pass a `feature_subset` label to keep them distinct.
- **Neural methods (autoencoders, attention-based models) can fail to converge on some feature combinations for reasons that aren't always obvious from feature count alone.** Don't assume a smaller input is automatically safer than one that worked before. If you're using one of these methods, check its actual training behavior (did the loss plateau, or was it still improving when training stopped) before trusting its output, not just whether it finished without crashing.
- **Long-running background jobs need visible progress logging from the start.** Without it, you have no way to tell "still working" from "silently stuck" except waiting and hoping. Turn on progress logging before launching anything that will take more than a few minutes.
- **Before trusting a long run's output, run something small and fast through the exact same code path first** (a "canary"). It's cheap insurance against discovering, after an hour of computation, that something fundamental was set up wrong.

## A note on data

**Never commit real patient data, or anything derived from it (reports, figures, logs), to version control, even in a private repository.** This applies regardless of whether any individual file looks like it contains identifiable information. Keep your data and your campaign outputs in a local folder outside the repository, and check that your `.gitignore` actually excludes them, don't just assume it does.

## Where to go for more detail

- `README.md` — what's actually in the codebase, module by module.
- `SKILL.md` — the full methodological rulebook this tool enforces, and the real findings that led to each rule.
- The `tests/` directory — every method and every validation check has tests that double as working examples of how to call it.
