"""PPMI CSV discovery/loading and clinical table construction.

Reconstructed from ``ParkinsonsVoice_Detection_Pipeline.ipynb`` cells 2, 4,
5, 6, 7, 8, 12, and 17 (the notebook's indentation was destroyed on upload;
logic below follows the docstrings/variable names/obvious block structure
that survived).

Data access
-----------
PPMI (Parkinson's Progression Markers Initiative) data is **gated**: it
requires a study application and a signed Data Use Agreement, and is not
redistributed here. See:
https://www.ppmi-info.org/access-data-specimens/download-data

Expected local layout (matches what a PPMI portal download unpacks to)::

    ./PPMI_raw/                     # raw PPMI CSV exports (see find_file)
    ./processed/                    # cleaned tables produced upstream, e.g.
                                     #   clinical_clean*.csv
                                     #   updrs_longitudinal*.csv
                                     #   voice_features*.csv

Nothing in this module invents a downloader for PPMI data - callers must
supply files locally after completing the PPMI data-access process.
"""

from __future__ import annotations

import logging
import os
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Clinical tables referenced in the PPMI Data User Guide that the original
# notebook attempted to load by partial filename match.
CORE_TABLE_NAMES = (
    "Participant_Status",
    "Demographics",
    "MDS-UPDRS_Part_I",
    "MDS-UPDRS_Part_I_Patient_Questionnaire",
    "MDS-UPDRS_Part_II__Patient_Questionnaire",
    "MDS-UPDRS_Part_III",
    "LEDD_Concomitant_Medication_Log",
    "Concomitant_Medication_Log",
    "UPSIT",
    "Codes",
)


class PPMIDataNotFoundError(FileNotFoundError):
    """Raised when a required PPMI file/directory is missing locally."""


def find_file(partial_name: str, directory: str | os.PathLike) -> str | None:
    """Return the first CSV whose filename contains ``partial_name``.

    PPMI downloads include a date suffix in filenames, so an exact match is
    unreliable; this does a glob on ``*<partial_name>*.csv``.

    Returns None (and logs a warning) if no file matches, rather than
    raising, so batch-loading a list of optional tables can proceed.
    """
    directory = Path(directory)
    pattern = str(directory / f"*{partial_name}*.csv")
    matches = sorted(glob(pattern))
    if not matches:
        logger.warning("No file matches for '%s' in %s", partial_name, directory)
        return None
    if len(matches) > 1:
        logger.info("Multiple matches for '%s', using first: %s", partial_name, matches[0])
    return matches[0]


def require_data_dir(directory: str | os.PathLike) -> Path:
    """Raise a clear, actionable error if ``directory`` does not exist.

    Used at pipeline entry points instead of silently operating on an empty
    directory, since PPMI data is not bundled and a missing directory
    almost always means the user has not completed the PPMI data-access
    process yet.
    """
    path = Path(directory)
    if not path.is_dir():
        raise PPMIDataNotFoundError(
            f"PPMI data directory not found: {path}. PPMI data requires a study "
            "application and Data Use Agreement and is not bundled with this repo. "
            "Apply for access at https://www.ppmi-info.org/access-data-specimens/download-data "
            f"and place downloaded CSVs under {path}."
        )
    return path


def load_csv_by_partial(
    partial_name: str,
    directory: str | os.PathLike,
    dtype: dict | None = None,
    parse_dates: list[str] | None = None,
) -> pd.DataFrame | None:
    """Load a CSV matched by partial filename. Returns None if missing/unreadable."""
    fp = find_file(partial_name, directory)
    if fp is None:
        return None
    try:
        df = pd.read_csv(fp, dtype=dtype, parse_dates=parse_dates, low_memory=False)
        logger.info("Loaded '%s' -> %s %s", partial_name, os.path.basename(fp), df.shape)
        return df
    except Exception as exc:  # noqa: BLE001 - surfaced via logging, caller gets None
        logger.error("Failed to load %s: %s", fp, exc)
        return None


def load_core_tables(directory: str | os.PathLike) -> dict[str, pd.DataFrame | None]:
    """Load the standard set of PPMI clinical tables by partial name.

    Any table not present in ``directory`` is returned as None rather than
    raising, matching the notebook's "warn but don't crash" behavior for
    optional tables.
    """
    return {name: load_csv_by_partial(name, directory) for name in CORE_TABLE_NAMES}


def build_participant_master(
    participant_status_df: pd.DataFrame | None,
    demographics_df: pd.DataFrame | None,
    codes_df: pd.DataFrame | None = None,
) -> pd.DataFrame | None:
    """Build a participant-level master table from Participant_Status + Demographics.

    Keeps participants with ENROLL_STATUS in {enrolled, withdrew, complete},
    per the PPMI Data User Guide. ``codes_df`` is accepted for future code
    decoding but not currently applied (matches the original notebook, which
    left this as a documented extension point).
    """
    if participant_status_df is None:
        logger.error("Participant_Status table is required to build master table")
        return None

    ps = participant_status_df.copy()

    if "ENROLL_STATUS" in ps.columns:
        ps["ENROLL_STATUS"] = ps["ENROLL_STATUS"].astype(str).str.lower()
        valid_status = ["enrolled", "withdrew", "complete"]
        ps = ps[ps["ENROLL_STATUS"].isin(valid_status)].copy()
    else:
        logger.warning("ENROLL_STATUS missing; proceeding without filtering by status")

    if demographics_df is not None:
        merged = ps.merge(demographics_df, on="PATNO", how="left", suffixes=("", "_demo"))
    else:
        merged = ps

    return merged


def cohort_summary(participant_status_df: pd.DataFrame | None) -> pd.DataFrame | None:
    """Return a COHORT_DEFINITION x ENROLL_STATUS crosstab for quick EDA."""
    if participant_status_df is None:
        logger.warning("Participant_Status not available")
        return None
    ps = participant_status_df.copy()
    ps["ENROLL_STATUS"] = ps["ENROLL_STATUS"].astype(str).str.lower()
    return ps.groupby(["COHORT_DEFINITION", "ENROLL_STATUS"]).size().unstack(fill_value=0)


def build_ledd_timeline(ledd_df: pd.DataFrame | None) -> pd.DataFrame | None:
    """Aggregate LEDD (Levodopa Equivalent Daily Dose) rows into a per-PATNO timeline.

    A conservative, month-level aggregation, per the notebook's comment that
    the PPMI guide's LEDD tracking is at month granularity.
    """
    if ledd_df is None:
        logger.warning("LEDD table not available")
        return None

    df = ledd_df.copy()

    if "STARTDT" in df.columns:
        df["STARTDT"] = pd.to_datetime(df["STARTDT"], errors="coerce")

    led_cols = [c for c in df.columns if c.upper() in ("LEDD", "LEDDSUM", "LD")]
    if led_cols:
        df["LEDD_VAL"] = pd.to_numeric(df[led_cols[0]], errors="coerce")
    else:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            df["LEDD_VAL"] = df[numeric_cols[0]]
        else:
            logger.warning("No LEDD-like column found")
            df["LEDD_VAL"] = np.nan

    df = df.sort_values(["PATNO", "STARTDT"]) if "STARTDT" in df.columns else df.sort_values(["PATNO"])
    df = df[~df["LEDD_VAL"].isnull()]

    out = df.groupby(["PATNO", "STARTDT"])["LEDD_VAL"].sum().reset_index()
    return out.sort_values(["PATNO", "STARTDT"])


def load_processed(name: str, directory: str | os.PathLike) -> pd.DataFrame | None:
    """Load a previously-cleaned table (e.g. 'clinical_clean') from processed_dir."""
    fp = find_file(name, directory)
    if fp is None:
        return None
    return pd.read_csv(fp, low_memory=False)


def load_processed_sources(processed_dir: str | os.PathLike) -> dict[str, pd.DataFrame | None]:
    """Load the three processed tables the ML pipeline expects: clinical, updrs, voice.

    NOTE (reconstruction judgment call): in the original notebook these three
    tables are loaded but only ``processed_clinical`` is ever passed into the
    modeling cells (``prepare_ml_data`` / ``baseline_train_eval``) - there is
    no code anywhere in the notebook that merges ``processed_voice`` (or
    ``processed_updrs``) into the classifier's feature matrix, despite the
    notebook's filename ("...Voice_Detection_Pipeline"). This function
    preserves that structure faithfully (all three are loaded and returned
    separately) rather than inventing a merge step the original author never
    wrote. If your clinical table already has voice-derived acoustic
    features joined in upstream, `processed_clinical` alone is sufficient for
    `features.prepare_ml_data`.
    """
    return {
        "clinical": load_processed("clinical_clean", processed_dir),
        "updrs": load_processed("updrs_longitudinal", processed_dir),
        "voice": load_processed("voice_features", processed_dir),
    }


def save_sample(df: pd.DataFrame | None, output_dir: str | os.PathLike, filename: str, n: int = 200) -> str | None:
    """Save the first ``n`` rows of ``df`` to ``output_dir/filename`` for traceability."""
    if df is None:
        logger.warning("Nothing to save for %s (dataframe is None)", filename)
        return None
    os.makedirs(output_dir, exist_ok=True)
    out_fp = os.path.join(output_dir, filename)
    df.head(n).to_csv(out_fp, index=False)
    logger.info("Saved sample to %s", out_fp)
    return out_fp
