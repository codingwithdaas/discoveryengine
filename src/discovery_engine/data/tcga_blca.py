"""Loader for the first real (non-synthetic) campaign: TCGA-BLCA CMB
histology + RNA-seq, DSS outcome, GRADE covariate.

Deliberately narrow and dataset-specific (hardcoded column names, cBioPortal
comment-header skiprows, this cohort's confirmed ID conventions) rather than
a generic multi-cohort loader -- there is exactly one real cohort in scope
right now (campaign one: CMB + RNA-seq; RPPA is campaign two, once this one
is confirmed to behave correctly end to end, per explicit build-order
decision). Every join-key and drop-rate claim below was verified empirically
against the actual files before this was written, not assumed from general
TCGA convention:

- Sample IDs across data_clinical_sample.SAMPLE_ID, both case_lists files,
  and both data files' column headers are uniformly PATIENT_ID + "-01",
  zero exceptions, zero patients with more than one sample column in either
  the RNA-seq or RPPA data. A blind "keep first 3 barcode segments"
  truncation is safe for THIS cohort -- checked, not a general TCGA
  guarantee, and would need re-verifying if this loader is ever pointed at
  a different TCGA cohort where multi-sample patients are common.
- The real four-way intersection (CMB + clinical + RNA-seq + RPPA) is 318
  of the CMB matrix's 386 patients (17.6% drop), driven almost entirely by
  RPPA's incomplete coverage (343/411 patients cohort-wide). CMB + RNA-seq
  alone drops almost nothing (386 -> 381 after also requiring the clinical
  file). This is why campaign one is CMB + RNA-seq only: the first time any
  of this project's discovery methods touch real data is not the moment to
  also introduce RPPA's much larger drop rate in the same run -- if
  something looks off, there would be no way to isolate which changed
  variable caused it.
- Requiring a complete DSS outcome (both DSS_STATUS and DSS_MONTHS
  non-null) drops the 381 further, to 367.

Known, confirmed (not hypothetical) validation-layer gap this loading
surfaced: 2 of the 367 final patients have a missing GRADE. validation.
covariate.covariate_stratified_test handles this fine (pandas groupby
silently drops NaN groups). validation.covariate.freedman_lane_test does
not filter on covariate nullness before casting to float for a linear
algebra solve -- checked directly: a NaN-containing design matrix makes
np.linalg.lstsq raise LinAlgError (fails loudly, not silently corrupting
results), but that's an opaque crash, not a clear message, for any real
campaign that calls freedman_lane_test with covariate_col="grade_high"
without first restricting to grade-complete patients. Not fixed here
(a validation-layer decision, out of this loader's scope) -- flagged for
a real decision before freedman_lane_test is relied on with this
covariate on this cohort.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from discovery_engine.data.cohort import CohortData
from discovery_engine.outcomes import OutcomeSpec, OutcomeType

logger = logging.getLogger(__name__)


def load_tcga_blca_cmb_rnaseq_cohort(
    tcga_dir: str | Path,
    cmb_xlsx_path: str | Path,
    outcome_endpoint: str = "DSS",
) -> tuple[CohortData, OutcomeSpec]:
    tcga_dir = Path(tcga_dir)

    cmb = pd.read_excel(cmb_xlsx_path).set_index("PatientID")

    rnaseq = pd.read_csv(tcga_dir / "data_mrna_seq_v2_rsem.txt", sep="\t")
    rnaseq = _clean_rnaseq(rnaseq)

    clinical = pd.read_csv(tcga_dir / "data_clinical_patient.txt", sep="\t", skiprows=4).set_index("PATIENT_ID")
    sample = pd.read_csv(tcga_dir / "data_clinical_sample.txt", sep="\t", skiprows=4).set_index("PATIENT_ID")

    status_col = f"{outcome_endpoint}_STATUS"
    months_col = f"{outcome_endpoint}_MONTHS"
    if status_col not in clinical.columns or months_col not in clinical.columns:
        raise ValueError(f"outcome_endpoint {outcome_endpoint!r} has no {status_col}/{months_col} in clinical file")

    candidates = sorted(set(cmb.index) & set(clinical.index) & set(rnaseq.index))
    logger.info(
        "tcga_blca loader: CMB (%d) & clinical (%d) & RNA-seq (%d) -> %d patients before outcome filtering",
        len(cmb), len(clinical), len(rnaseq), len(candidates),
    )

    clinical_sub = clinical.loc[candidates]
    valid_outcome = clinical_sub[status_col].notna() & clinical_sub[months_col].notna()
    dropped_for_outcome = sorted(set(candidates) - set(clinical_sub.index[valid_outcome]))
    if dropped_for_outcome:
        logger.info(
            "tcga_blca loader: dropping %d patient(s) with missing %s/%s: %s",
            len(dropped_for_outcome), status_col, months_col, dropped_for_outcome[:10],
        )
    final_patients = sorted(clinical_sub.index[valid_outcome])

    cmb_final = cmb.loc[final_patients]
    rnaseq_final = rnaseq.loc[final_patients]

    grade = sample.reindex(final_patients)["GRADE"]
    n_missing_grade = grade.isna().sum()
    if n_missing_grade:
        logger.info("tcga_blca loader: %d/%d patients missing GRADE (kept in cohort, NaN covariate)",
                     n_missing_grade, len(final_patients))
    covariates = pd.DataFrame({"grade_high": (grade == "High Grade").astype(float).where(grade.notna())}, index=final_patients)

    data = CohortData(modalities={"cmb_histology": cmb_final, "rnaseq": rnaseq_final}, covariates=covariates)

    event = clinical_sub.loc[final_patients, status_col].str.split(":").str[0].astype(float).to_numpy()
    months = clinical_sub.loc[final_patients, months_col].to_numpy(dtype=float)
    outcome = OutcomeSpec(OutcomeType.SURVIVAL, final_patients, np.column_stack([months, event]))

    logger.info("tcga_blca loader: final cohort N=%d, event rate=%.3f", len(final_patients), event.mean())
    return data, outcome


def _clean_rnaseq(rnaseq: pd.DataFrame) -> pd.DataFrame:
    """Drop blank Hugo_Symbol rows (can't be a column name); disambiguate
    duplicated Hugo_Symbol rows with their Entrez_Gene_Id, transpose to
    sample-indexed rows x gene columns, and truncate sample IDs to the bare
    PatientID (confirmed safe for this cohort, see module docstring).

    13 blank Hugo_Symbol rows, 7 duplicated Hugo_Symbol values, of 20531
    total -- checked directly, not assumed. The Entrez_Gene_Id suffix alone
    is NOT sufficient: checked directly, and 6 of those 7 duplicate pairs
    have IDENTICAL Hugo_Symbol *and* IDENTICAL Entrez_Gene_Id (e.g. ELMOD1,
    Entrez 55531, appears twice), so appending Entrez doesn't disambiguate
    them -- this was a real bug in an earlier version of this function,
    caught by checking data.modalities["rnaseq"].columns.duplicated().sum()
    on the actual loaded cohort rather than assuming the Entrez suffix
    covered every case. Also checked whether these same-symbol-same-Entrez
    row pairs are copy-paste duplicates (safe to drop one) or genuinely
    different data: their expression VALUES differ substantially across
    samples (e.g. ELMOD1 row 5533 vs 10081: 1.1662 vs 0.0 for the first
    sample), so this is real, distinct data under a colliding label (likely
    a collapsed-isoform/annotation artifact upstream in RSEM), not a
    duplicate row -- dropping either would silently discard real
    measurements, which is why the fix is a positional uniqueness suffix
    (keeps both), not a de-duplication drop (keeps only one).
    """
    rnaseq = rnaseq[rnaseq["Hugo_Symbol"].notna()].copy()
    dup_mask = rnaseq["Hugo_Symbol"].duplicated(keep=False)
    rnaseq.loc[dup_mask, "Hugo_Symbol"] = (
        rnaseq.loc[dup_mask, "Hugo_Symbol"] + "|" + rnaseq.loc[dup_mask, "Entrez_Gene_Id"].astype(str)
    )
    # Final guarantee, regardless of cause: some symbol+Entrez pairs still
    # collide (see docstring), so append a positional occurrence suffix to
    # any name still duplicated after the Entrez pass.
    still_dup = rnaseq["Hugo_Symbol"].duplicated(keep=False)
    occurrence = rnaseq.groupby("Hugo_Symbol").cumcount()
    rnaseq.loc[still_dup, "Hugo_Symbol"] = (
        rnaseq.loc[still_dup, "Hugo_Symbol"] + "#" + (occurrence[still_dup] + 1).astype(str)
    )
    assert not rnaseq["Hugo_Symbol"].duplicated().any(), "gene column names must be unique after disambiguation"

    rnaseq = rnaseq.drop(columns=["Entrez_Gene_Id"]).set_index("Hugo_Symbol")
    transposed = rnaseq.T
    transposed.index = ["-".join(sid.split("-")[:3]) for sid in transposed.index]
    return transposed
