# Discovery Engine

**Software that finds patterns in cancer data, then tries hard to prove they are fake.**

Built to search patient tumor data for biomarkers (measurable signs that predict how a disease will go) while guarding against the most common mistake in this field: mistaking luck for a discovery.

**New here?** Read this page for the overview. For the statistical methods and design details, see the [`docs/`](docs/) folder, starting with [`docs/METHODS.md`](docs/METHODS.md).

---

## The problem, in one example

Imagine buying 1,000 lottery tickets and then announcing that ticket #437 is "special" because it won. It won only because you bought so many.

Cancer datasets work the same way. A single tumor sample has thousands of measurements, and there are only a few hundred patients. If you search enough measurements, something will look like it predicts survival purely by chance. Many published biomarkers have failed to replicate for exactly this reason.

## What this project does

1. **Searches broadly.** Seven different methods, from classical statistics to neural networks, look for groups of measurements that move together and might relate to patient survival.
2. **Counts everything it tried.** Every candidate from every method goes into one list. The statistics are corrected for the *whole* list, not just the winner, which is the lottery-ticket fix.
3. **Tries to break each survivor.** A candidate must reproduce on random re-splits of the patients, hold up after adjusting for known clinical factors like tumor grade, and prove it isn't just a rediscovery of something already known.
4. **Reports honestly.** The report writer uses a fixed set of phrases, so it cannot call something "validated" when the evidence only supports "a candidate worth following up." Rejected candidates are reported too.

```
 many methods  ->  one pooled list  ->  corrected for  ->  validation  ->  honest
 find patterns     of ALL candidates    everything tried     checks         report
```

## How it was tested

Before touching real patient data, the engine was run on **synthetic data with signals planted on purpose**, to check that it finds real signals and rejects fake ones. The test suite (164 tests) is in `tests/`.

It was then run on public TCGA bladder cancer data. The real run exposed bugs that synthetic tests had missed (for example, two candidates silently sharing an ID), and those were fixed and committed. That history is visible in the commit log. No patient data or results are included in this repository.

## Where to look

For the full methods and design (architecture, the validation checks and which of them decide pass/fail, how each method is categorized), see [`docs/METHODS.md`](docs/METHODS.md).


| If you want to see... | Open |
|---|---|
| The multiple-testing correction (the core idea) | `src/discovery_engine/campaign/correction.py` |
| The checks every candidate must pass | `src/discovery_engine/validation/gate.py` |
| The seven discovery methods | `src/discovery_engine/discovery/` |
| How wording is kept honest | `src/discovery_engine/reporting/language.py` |
| How I know it works | `tests/` |

## Run it

```bash
python -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -e .
pytest                        # runs the test suite on synthetic data
```

For a full walkthrough, see `HOW_TO_USE.md`.

## My role and how it was built

I designed the project: the question, the statistical rules (what must be corrected, what counts as reproducible), and the decisions at each step. I built it with AI assistance (Claude Code), directing the work, reviewing results, and checking claims against the data. The AI contributor appears in the commit history. I'm noting that here so there is no ambiguity about how it was made.

This grew out of reviewing a bladder cancer lab's analyses, where I kept running into the question of whether a "finding" was real. The engine is my attempt to answer that question systematically.

## Limits

Research software, not a clinical tool. Anything it reports is a discovery-stage candidate from one dataset until independently replicated. Real patient data is not included and is governed by its own access rules.

## Data

This repository contains no data. Point the loaders in `src/discovery_engine/data/` at your own locally-held copy.
