#!/usr/bin/env python3
"""Raw scatters of Sedline PSi and cerebral StO2 against time from induction.

Two stacked panels sharing one x-axis, where TIME = 0 is anesthesia induction:

    top     PSi (or another Sedline signal)   -- from --filepaths
    bottom  cerebral StO2                      -- from --sto2-filepaths

Every valid raw sample from every patient is drawn as one dot. Nothing is
averaged, binned or smoothed before plotting. Because the two panels share the
x-axis and the same induction anchor, a change in one lines up vertically with
a change in the other, which is the point: it shows whether StO2 moves with
PSi, and how long after.

On top of the dots an optional trend curve is drawn. It is fitted SEPARATELY
before and after induction so no smoothing bleeds across TIME = 0, and it is
blanked wherever fewer than --min-patients patients contribute data, so one
patient's record can never masquerade as a cohort trend. Its purpose is the LAG
readout printed to the terminal: how long after the induction drug is pushed
each signal starts to move, and how long it takes to reach its nadir.

Data sources
------------
  * Sedline files, one per patient, listed one path per line in --filepaths.
    Columns used (matched by prefix, so stray spaces are fine):
        Date, Time            -> timestamp   (Epoch Time used as a fallback)
        PSi (Sedline) Value   -> the signal  (or SEFL / SEFR / SR / EMG)
        ARTF % (Sedline) Value-> optional artifact filter
  * StO2 files, one per patient, listed in --sto2-filepaths. Columns used:
        Time                  -> timestamp
        StO2_CH1..CH4         -> the signal, averaged over channels whose
        valid_CH1..CH4        -> valid flag equals 1
  * REDCap labeled export (--redcap) supplying, per patient:
        "What is the name of the file ([IU][HOSPITAL...]" -> research ID
        "Date of Surgery"                                 -> anchors clock times
        "What time was INDUCTION (when primary induction med pushed) (TIME = 0)?"
        "What time did the patient enter the OR?"

Research IDs
------------
  Three repairs run, in order, and every one of them is printed:

  1. Separators are stripped, so IUMH_2026030301 becomes IUMH2026030301.
  2. REDCAP_ID_FIXES, the hardcoded table at the top of this file. Entries keyed
     by (typed ID, Date of Surgery) fire only on that date, which is how
     IUMH2026010601 can be corrected to IUMH2026010501 for the 2026-01-05 case
     without touching the real IUMH2026010601 patient operated on 2026-01-06.
     Add new typos there.
  3. Anything left over is checked against the date embedded in the ID itself
     (IU + hospital + YYYYMMDD + patient of the day); where it disagrees with
     Date of Surgery the date wins. A repair is refused if it would land on an
     ID another row already holds legitimately. Disable with --no-id-repair.

  REDCap splits one patient across several rows when instruments repeat, and
  only the row owning the ID field carries the ID. The ID and Date of Surgery
  are therefore filled across each record block before anything else, otherwise
  an OR-entry time living on a sibling row is invisible.

Output
------
  One PNG. Everything else -- per-patient windows, exclusions, ID repairs, the
  lag readout and the summary statistics -- is printed to the terminal.

Usage
-----
    python psi_vs_time_regression.py
    python psi_vs_time_regression.py --panels psi          # PSi alone
    python psi_vs_time_regression.py --min-minutes -20 --max-minutes 30
    python psi_vs_time_regression.py --fit none            # raw dots only
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Hardcoded research-ID corrections.
#
#   key   (ID exactly as typed into REDCap, Date of Surgery as YYYY-MM-DD)
#         Use None for the date to apply the fix on every row with that ID --
#         only safe when the typed ID is not also a real patient's ID.
#   value the true research ID.
#
# Add new typos here. Anything listed is applied before the automatic
# date-based check, and each one is reported when it fires.
# --------------------------------------------------------------------------- #
REDCAP_ID_FIXES: dict[tuple[str, str | None], str] = {
    # Real patient IUMH2026010601 exists (surgery 2026-01-06), so this typo can
    # only be corrected on its own surgery date.
    ("IUMH2026010601", "2026-01-05"): "IUMH2026010501",
    # 13 characters -- a digit was dropped from the day. Not a real ID.
    ("IUMH202601601", None): "IUMH2026011601",
}


# --------------------------------------------------------------------------- #
# Signals this script can plot.
# --------------------------------------------------------------------------- #
SERIES = {
    "PSi": {
        "kind": "sedline", "column_prefix": "psi",
        "label": "PSi (Sedline)", "valid_range": (0.0, 100.0),
        "color": "#4c78a8",
    },
    "SEFL": {
        "kind": "sedline", "column_prefix": "sefl",
        "label": "SEF Left (Hz)", "valid_range": (0.0, 30.0),
        "color": "#4c78a8",
    },
    "SEFR": {
        "kind": "sedline", "column_prefix": "sefr",
        "label": "SEF Right (Hz)", "valid_range": (0.0, 30.0),
        "color": "#4c78a8",
    },
    "SR": {
        "kind": "sedline", "column_prefix": "sr %",
        "label": "Suppression Ratio (%)", "valid_range": (0.0, 100.0),
        "color": "#4c78a8",
    },
    "EMG": {
        "kind": "sedline", "column_prefix": "emg",
        "label": "EMG (%)", "valid_range": (0.0, 100.0),
        "color": "#4c78a8",
    },
    "StO2": {
        "kind": "sto2", "column_prefix": None,
        "label": "Cerebral StO2 (%)", "valid_range": (0.0, 100.0),
        "color": "#2f7d4f",
    },
}

ARTF_PREFIX = "artf"
STO2_CHANNELS = (1, 2, 3, 4)
DEFAULT_LAMBDA = 100.0          # penalty used when --fit spline is selected

# Research ID: IU + hospital + surgery date + patient number of that day.
ID_PATTERN = re.compile(r"IU(?:MH|UH)\d+", re.IGNORECASE)
ID_STRUCTURE = re.compile(r"^IU(MH|UH)(\d{8})(\d{2})$")
ID_LOOSE = re.compile(r"^IU(MH|UH)\d*?(\d{2})$")

REDCAP_ID_PREFIX = "what is the name of the file"
INDUCTION_PREFIX = "what time was induction"
OR_ENTRY_PREFIX = "what time did the patient enter the or"
SURGERY_DATE_PREFIX = "date of surgery"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    base = Path("/N/project/Analgesia_BDproject/PR/scripts_PR/9-16 Regression")
    parser.add_argument(
        "--filepaths", type=Path, default=base / "sedline_filepaths.csv",
        help="CSV listing one Sedline file path per line.")
    parser.add_argument(
        "--sto2-filepaths", type=Path, default=base / "sto2_filepaths.csv",
        help="CSV listing one cerebral-StO2 file path per line.")
    parser.add_argument(
        "--redcap", type=Path,
        default=Path("/N/project/Analgesia_BDproject/data/00_raw/"
                     "BDPostInductionHemod_DATA_LABELS_2026-08-21_1723.csv"),
        help="REDCap labeled export holding induction and OR-entry times.")
    parser.add_argument(
        "--outdir", type=Path, default=base / "output",
        help="Where the figure is written. No CSVs are produced.")
    parser.add_argument(
        "--panels", choices=["both", "psi", "sto2"], default="both",
        help="Which panels to draw (default both, stacked and sharing x).")
    parser.add_argument(
        "--signal", choices=[k for k, v in SERIES.items() if v["kind"] == "sedline"],
        default="PSi",
        help="Which Sedline signal fills the top panel (default PSi).")
    parser.add_argument(
        "--min-minutes", type=float, default=-10.0,
        help="Left edge, minutes before induction (default -10).")
    parser.add_argument(
        "--max-minutes", type=float, default=15.0,
        help="Right edge, minutes after induction (default 15).")
    parser.add_argument(
        "--max-artf", type=float, default=None,
        help="Optional: drop Sedline samples whose ARTF %% exceeds this. Off by "
             "default; the run reports what a threshold of 20 would cost.")
    parser.add_argument(
        "--fit", choices=["lowess", "linear", "spline", "none"],
        default="lowess",
        help="Trend curve drawn over the raw dots, fitted separately either "
             "side of induction. 'lowess' (default) is a local curve, which is "
             "what the lag readout needs; 'linear' is a straight least-squares "
             "line per side; 'none' plots the raw dots only.")
    parser.add_argument(
        "--lowess-frac", type=float, default=0.08,
        help="LOWESS smoothing span as a fraction of the segment (default "
             "0.08). Larger = smoother.")
    parser.add_argument(
        "--n-splines", type=int, default=80,
        help="Basis size when --fit spline is used.")
    parser.add_argument(
        "--min-patients", type=int, default=5,
        help="The trend curve is not drawn where fewer than this many patients "
             "have data (default 5). Stops a single long record from being "
             "mistaken for a cohort trend at the edges.")
    parser.add_argument(
        "--band", action="store_true",
        help="Also shade a 95%% band around the trend curve, obtained by "
             "resampling PATIENTS. Off by default because it obscures the raw "
             "dots this plot exists to show.")
    parser.add_argument(
        "--n-boot", type=int, default=100,
        help="Patient-level bootstrap replicates used for --band.")
    parser.add_argument(
        "--no-id-repair", action="store_true",
        help="Skip the automatic Date-of-Surgery ID check. The hardcoded "
             "REDCAP_ID_FIXES table still applies.")
    parser.add_argument(
        "--seed", type=int, default=20260916)
    return parser.parse_args()


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #

def read_filepath_list(master: Path) -> list[str]:
    """One path per line; tolerate a header, quotes and extra columns."""
    paths: list[str] = []
    with open(master, "r", newline="") as handle:
        for raw in handle:
            first = raw.strip().replace("\t", ",").split(",")[0]
            first = first.strip().strip('"').strip("'")
            if not first:
                continue
            if not first.lower().endswith((".csv", ".txt")) and "/" not in first:
                continue  # header row such as "filepath"
            paths.append(first)
    return paths


def patient_id_from_path(path: str) -> str | None:
    """Research ID embedded in the file path, e.g. IUMH2026030501.

    File-path IDs are taken as ground truth. Typos are corrected on the REDCap
    side instead.
    """
    match = ID_PATTERN.search(path)
    return match.group(0).upper() if match else None


def find_column(frame: pd.DataFrame, prefix: str) -> str | None:
    """Locate a column by case-insensitive prefix after stripping whitespace.

    The Sedline exports ship a leading space on ' Time', so exact names break,
    and the REDCap labeled export uses the full question text as the header.
    """
    prefix = prefix.strip().lower()
    for column in frame.columns:
        if str(column).strip().lower().startswith(prefix):
            return column
    return None


def repair_subject_ids(raw_ids: pd.Series, surgery_dates: pd.Series,
                       auto_repair: bool = True) -> tuple[pd.Series, list[str]]:
    """Resolve each REDCap research ID: separators, hardcoded table, then date.

    The ID carries the surgery date inside it, so the two fields are redundant
    and can check each other. When they disagree the date is believed and the ID
    is rebuilt, keeping the hospital and the patient-of-the-day number. A repair
    is refused if it would land on an ID that some other row already holds
    legitimately, so a typo can never overwrite a real patient.
    """
    text = raw_ids.astype("string").str.strip().str.upper()
    # Strip separators first: IUMH_2026030301 and IUMH-2026030301 are just
    # IUMH2026030301 typed with a stray character.
    cleaned = text.str.replace(r"[^A-Z0-9]", "", regex=True)

    notes: list[str] = []
    for original, tidy in zip(text, cleaned):
        if pd.notna(original) and original != tidy:
            notes.append(f"{original} -> {tidy}  (separator stripped)")
            break  # one example is enough; the count is reported separately
    separator_fixes = int((text.notna() & text.ne(cleaned)).sum())

    # Hardcoded table, applied before anything automatic.
    resolved: list[str | None] = []
    hardcoded_hits: list[str] = []
    pending: list[bool] = []
    for value, date in zip(cleaned, surgery_dates):
        if pd.isna(value) or not value:
            resolved.append(None)
            pending.append(False)
            continue
        stamp = None if pd.isna(date) else f"{date:%Y-%m-%d}"
        fix = REDCAP_ID_FIXES.get((value, stamp), REDCAP_ID_FIXES.get((value, None)))
        if fix is not None:
            resolved.append(fix)
            pending.append(False)
            hardcoded_hits.append(
                f"{value} -> {fix}  (hardcoded"
                + (f", Date of Surgery {stamp}" if (value, stamp) in REDCAP_ID_FIXES
                   else "") + ")"
            )
        else:
            resolved.append(value)
            pending.append(True)
    notes.extend(sorted(set(hardcoded_hits)))

    if separator_fixes:
        notes.append(f"{separator_fixes} ID(s) had separators stripped")

    if not auto_repair:
        return pd.Series(resolved, index=raw_ids.index, dtype="object"), notes

    # Rows whose ID already agrees with their own surgery date are trusted and
    # are never a valid repair target.
    trusted: set[str] = set()
    for value, date in zip(resolved, surgery_dates):
        if value is None or pd.isna(date):
            continue
        match = ID_STRUCTURE.match(value)
        if match and match.group(2) == f"{date:%Y%m%d}":
            trusted.add(value)

    automatic: list[str] = []
    final: list[str | None] = []
    for value, date, still_open in zip(resolved, surgery_dates, pending):
        if value is None or not still_open:
            final.append(value)
            continue

        structured = ID_STRUCTURE.match(value)
        if pd.isna(date):
            # Nothing to check against: keep a well-formed ID, drop a broken one.
            final.append(value if structured else None)
            if not structured:
                automatic.append(f"{value}: malformed and no Date of Surgery — dropped")
            continue

        if structured:
            hospital, datepart, sequence = structured.groups()
            if datepart == f"{date:%Y%m%d}":
                final.append(value)
                continue
        else:
            loose = ID_LOOSE.match(value)
            if loose is None:
                final.append(None)
                automatic.append(f"{value}: unrecognised ID format — dropped")
                continue
            hospital, sequence = loose.groups()

        candidate = f"IU{hospital}{date:%Y%m%d}{sequence}"
        if candidate in trusted:
            final.append(value if structured else None)
            automatic.append(
                f"{value}: Date of Surgery {date:%Y-%m-%d} implies {candidate}, "
                f"but that ID is already held by a row whose own date agrees — "
                f"left unchanged"
            )
            continue

        final.append(candidate)
        automatic.append(f"{value} -> {candidate}  (Date of Surgery {date:%Y-%m-%d})")

    notes.extend(sorted(set(automatic)))
    return pd.Series(final, index=raw_ids.index, dtype="object"), notes


def find_record_key(redcap: pd.DataFrame) -> str | None:
    """The column identifying which REDCap record a row belongs to.

    Normally the export's first column (Record ID / Screening ID). Accepted only
    if it is populated on every row and genuinely repeats, which is what marks it
    as a record key rather than a data field.
    """
    for column in redcap.columns[:3]:
        values = redcap[column]
        if values.notna().all() and 1 < values.nunique() < len(redcap):
            return column
    return None


def fill_across_record_rows(redcap: pd.DataFrame,
                            columns: list[str]) -> tuple[int, str]:
    """Carry the ID and surgery date across a record's repeating-instrument rows.

    REDCap writes one row per instrument. Only the row owning a field carries its
    value, so a patient's OR-entry time frequently sits on a row whose ID cell is
    blank. Filling both directions inside each record block makes those rows
    addressable; without it the time is simply invisible and the patient looks
    like it has no OR entry recorded.

    Returns (cells filled, how).
    """
    record_key = find_record_key(redcap)
    if record_key is None and redcap[columns[0]].notna().all():
        return 0, "not needed (one row per patient)"

    filled = 0
    for column in columns:
        before = int(redcap[column].notna().sum())
        if record_key is not None:
            key = redcap[record_key]
            redcap[column] = redcap.groupby(key, sort=False)[column].ffill()
            redcap[column] = redcap.groupby(key, sort=False)[column].bfill()
        else:
            # No usable record key: fall back to carrying the last seen value
            # downwards, which is the layout REDCap produces anyway.
            redcap[column] = redcap[column].ffill()
        filled += int(redcap[column].notna().sum()) - before

    how = (f"grouped by '{record_key}'" if record_key is not None
           else "forward-filled (no record-key column found)")
    return filled, how


def load_event_times(redcap_path: Path,
                     auto_repair: bool = True) -> tuple[pd.DataFrame, list[str]]:
    """Per-patient induction and OR-entry instants from the REDCap export."""
    redcap = pd.read_csv(redcap_path, low_memory=False)

    # Prefer the named ID question; fall back to whichever column holds the most
    # IU-format values if the export is renamed.
    id_column = find_column(redcap, REDCAP_ID_PREFIX)
    if id_column is None:
        best = 0
        for column in redcap.columns:
            # NB: match against the pattern as written - upper-casing it would
            # turn \d into \D and silently match nothing.
            hits = int(
                redcap[column].astype("string").str.strip()
                .str.fullmatch(r"IU(?:MH|UH)\d+", case=False, na=False).sum()
            )
            if hits > best:
                id_column, best = column, hits
    if id_column is None:
        raise ValueError("No research-ID column (IUMH.../IUUH...) found in REDCap.")

    induction_column = find_column(redcap, INDUCTION_PREFIX)
    or_entry_column = find_column(redcap, OR_ENTRY_PREFIX)
    if induction_column is None:
        raise ValueError("REDCap has no 'What time was INDUCTION...' column.")
    if or_entry_column is None:
        raise ValueError("REDCap has no 'What time did the patient enter the OR?' column.")

    date_column = find_column(redcap, SURGERY_DATE_PREFIX)
    if date_column is None:
        raise ValueError("REDCap has no 'Date of Surgery' column; it is needed "
                         "both to timestamp the clock times and to repair IDs.")

    notes: list[str] = []
    filled, how = fill_across_record_rows(redcap, [id_column, date_column])
    if filled:
        notes.append(f"{filled} blank ID / Date-of-Surgery cell(s) filled from "
                     f"sibling rows of the same REDCap record, {how}")

    surgery_date = pd.to_datetime(redcap[date_column], errors="coerce")

    def resolve(values: pd.Series) -> pd.Series:
        """Attach Date of Surgery to bare HH:MM[:SS] entries.

        Clock-only strings go through an explicit vectorized path rather than
        dateutil, which would parse them one element at a time (slow, and it
        warns) and date them to today. Anything else falls through to the
        general parser and is re-anchored if it came back dated today.
        """
        text = values.astype("string").str.strip()
        parsed = pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns]")

        clock = text.str.fullmatch(r"\d{1,2}:\d{2}(?::\d{2})?", na=False)
        if clock.any():
            padded = text.loc[clock].str.replace(
                r"^(\d{1,2}):(\d{2})$", r"\1:\2:00", regex=True)
            parsed.loc[clock] = (surgery_date.loc[clock].dt.normalize()
                                 + pd.to_timedelta(padded))

        other = ~clock & text.notna() & text.ne("")
        if other.any():
            loose = pd.to_datetime(text.loc[other], errors="coerce", format="mixed")
            parsed.loc[other] = loose
            # "8:30 AM" and friends parse to TODAY's date; re-anchor those.
            today = pd.Timestamp.today().normalize()
            stale = (parsed.notna() & parsed.dt.normalize().eq(today)
                     & surgery_date.notna() & other)
            parsed.loc[stale] = (surgery_date.loc[stale].dt.normalize()
                                 + (parsed.loc[stale] - today))
        return parsed

    subject_ids, id_notes = repair_subject_ids(redcap[id_column], surgery_date,
                                               auto_repair=auto_repair)
    notes.extend(id_notes)

    induction_at = resolve(redcap[induction_column])
    or_entry_at = resolve(redcap[or_entry_column])

    # A late-evening case crosses midnight: both clock times get stamped with the
    # same Date of Surgery, which puts induction apparently ~23 h BEFORE OR entry.
    # Roll induction onto the next day when that happens.
    crossed = (induction_at.notna() & or_entry_at.notna()
               & ((or_entry_at - induction_at) > pd.Timedelta(hours=12)))
    induction_at.loc[crossed] = induction_at.loc[crossed] + pd.Timedelta(days=1)
    if int(crossed.sum()):
        notes.append(f"{int(crossed.sum())} case(s) crossed midnight between OR "
                     f"entry and induction; induction rolled to the next day")

    events = pd.DataFrame({
        "subject_id": subject_ids,
        "induction": induction_at,
        "or_entry": or_entry_at,
    })

    events = events.loc[events["subject_id"].notna() & events["subject_id"].ne("")]
    # Collapse to one row per patient taking the first NON-MISSING value of each
    # time, so an induction time on one form row and an OR-entry time on another
    # are not lost to each other.
    events = (
        events.sort_values("subject_id")
        .groupby("subject_id", as_index=True)
        .agg(induction=("induction", "first"), or_entry=("or_entry", "first"))
    )
    return events, notes


def load_sedline_file(path: str, series: str) -> pd.DataFrame | None:
    """Timestamped, valid samples of one Sedline signal from one patient."""
    frame = pd.read_csv(path, low_memory=False)
    spec = SERIES[series]

    value_column = find_column(frame, spec["column_prefix"])
    if value_column is None:
        print(f"WARNING: {Path(path).name} has no '{series}' column; skipped")
        return None

    date_column = find_column(frame, "date")
    time_column = find_column(frame, "time")          # ' Time' has a leading space
    epoch_column = find_column(frame, "epoch")

    timestamp = pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns]")
    if date_column and time_column and date_column != time_column:
        timestamp = pd.to_datetime(
            frame[date_column].astype("string").str.strip() + " "
            + frame[time_column].astype("string").str.strip(),
            errors="coerce", format="mixed",
        )
    if timestamp.isna().all() and epoch_column:
        # Epoch Time is milliseconds since the unix epoch in these exports.
        timestamp = pd.to_datetime(
            pd.to_numeric(frame[epoch_column], errors="coerce"),
            unit="ms", errors="coerce",
        )
    if timestamp.isna().all():
        print(f"WARNING: {Path(path).name} has no parseable timestamps; skipped")
        return None

    # "-" means the monitor had no value; to_numeric turns it into a real NaN.
    value = pd.to_numeric(frame[value_column], errors="coerce")

    artf_column = find_column(frame, ARTF_PREFIX)
    artf = (pd.to_numeric(frame[artf_column], errors="coerce")
            if artf_column else pd.Series(np.nan, index=frame.index))

    return pd.DataFrame({"timestamp": timestamp, "value": value, "artf": artf})


def load_sto2_file(path: str, series: str) -> pd.DataFrame | None:
    """Cerebral StO2 averaged over the channels flagged valid on each row.

    A channel only counts when its valid_CHn flag is 1, which is how the
    oximetry export marks a probe that was actually reading.
    """
    frame = pd.read_csv(path, low_memory=False)

    time_column = find_column(frame, "time")
    if time_column is None:
        print(f"WARNING: {Path(path).name} has no Time column; skipped")
        return None
    timestamp = pd.to_datetime(frame[time_column].astype("string").str.strip(),
                               errors="coerce", format="mixed")

    columns_present = 0
    total = pd.Series(0.0, index=frame.index)
    count = pd.Series(0, index=frame.index)
    for channel in STO2_CHANNELS:
        value_column = find_column(frame, f"sto2_ch{channel}")
        valid_column = find_column(frame, f"valid_ch{channel}")
        if value_column is None:
            continue
        columns_present += 1
        value = pd.to_numeric(frame[value_column], errors="coerce")
        if valid_column is not None:
            usable = pd.to_numeric(frame[valid_column], errors="coerce").eq(1)
            value = value.where(usable)
        total = total.add(value.fillna(0.0))
        count = count.add(value.notna().astype(int))

    if columns_present == 0:
        print(f"WARNING: {Path(path).name} has no StO2_CH* columns; skipped")
        return None

    mean = (total / count).where(count > 0)
    return pd.DataFrame({"timestamp": timestamp, "value": mean,
                         "artf": np.nan})


LOADERS = {"sedline": load_sedline_file, "sto2": load_sto2_file}


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #

def build_points(filepaths: Path, series: str, args: argparse.Namespace,
                 events: pd.DataFrame):
    """Pool every valid sample from every patient onto minutes-from-induction."""
    spec = SERIES[series]
    low, high = spec["valid_range"]
    loader = LOADERS[spec["kind"]]

    chunks: list[pd.DataFrame] = []
    audit: list[dict] = []

    for path in read_filepath_list(filepaths):
        subject_id = patient_id_from_path(path)
        record = {"path": path, "subject_id": subject_id}

        if subject_id is None:
            record["status"] = "no research ID in file path"
            audit.append(record)
            continue
        if not Path(path).is_file():
            record["status"] = "file not found"
            audit.append(record)
            continue
        if subject_id not in events.index:
            record["status"] = "not in REDCap export"
            audit.append(record)
            continue

        induction = events.loc[subject_id, "induction"]
        or_entry = events.loc[subject_id, "or_entry"]
        if pd.isna(induction):
            record["status"] = "no REDCap induction time"
            audit.append(record)
            continue

        try:
            frame = loader(path, series)
        except Exception as exc:
            record["status"] = f"read error: {exc}"
            audit.append(record)
            continue
        if frame is None or frame.empty:
            record["status"] = "no usable rows"
            audit.append(record)
            continue

        record["rows_in_file"] = int(len(frame))
        record["rows_missing_value"] = int(frame["value"].isna().sum())

        keep = frame["value"].between(low, high) & frame["timestamp"].notna()
        record["rows_out_of_range"] = int(
            (frame["value"].notna() & ~frame["value"].between(low, high)).sum()
        )

        # How much a conventional artifact threshold would cost, reported even
        # when the filter is off so the choice is visible rather than implicit.
        record["rows_artf_gt_20"] = int((frame["artf"] > 20).sum())
        if args.max_artf is not None and spec["kind"] == "sedline":
            keep &= frame["artf"].isna() | frame["artf"].le(args.max_artf)

        frame = frame.loc[keep].copy()
        if frame.empty:
            record["status"] = "no valid samples"
            audit.append(record)
            continue

        frame["minutes"] = (
            (frame["timestamp"] - induction).dt.total_seconds() / 60.0
        )
        # The full span of the record relative to induction, before any gating,
        # so a patient who falls outside the window can be diagnosed rather than
        # silently dropped.
        record["record_start_min"] = round(float(frame["minutes"].min()), 2)
        record["record_end_min"] = round(float(frame["minutes"].max()), 2)

        # Left edge: the first sample at or after OR entry.
        if pd.notna(or_entry):
            gated = frame.loc[frame["timestamp"] >= or_entry]
            if gated.empty:
                # OR entry sits past the end of the record -- almost always a
                # clock mismatch. Keeping the patient beats discarding them; the
                # window bounds still apply.
                record["or_entry_used"] = "NO — OR entry is after the record ends"
            else:
                frame = gated
                record["or_entry_used"] = "yes"
        else:
            record["or_entry_used"] = "NO — no REDCap OR-entry time"

        record["first_minute_raw"] = round(float(frame["minutes"].min()), 2)
        inside = frame["minutes"].between(args.min_minutes, args.max_minutes)
        record["points_clipped_left"] = int((frame["minutes"] < args.min_minutes).sum())
        frame = frame.loc[inside]
        if frame.empty:
            record["status"] = "no samples inside the plotted window"
            audit.append(record)
            continue

        frame["subject_id"] = subject_id
        chunks.append(frame[["subject_id", "minutes", "value"]])

        record["status"] = "included"
        record["points_plotted"] = int(len(frame))
        record["minutes_min"] = round(float(frame["minutes"].min()), 2)
        record["minutes_max"] = round(float(frame["minutes"].max()), 2)
        audit.append(record)

    points = (pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame())
    return points, pd.DataFrame(audit)


# --------------------------------------------------------------------------- #
# Trend curve, fitted separately either side of induction
# --------------------------------------------------------------------------- #

def _fit_segment(minutes: np.ndarray, values: np.ndarray, grid: np.ndarray,
                 method: str, frac: float, n_splines: int) -> np.ndarray | None:
    """Fit one side of induction. Returns values on `grid`, or None."""
    if minutes.size < 10 or grid.size == 0:
        return None

    if method == "linear":
        slope, intercept = np.polyfit(minutes, values, 1)
        return intercept + slope * grid

    if method == "lowess":
        try:
            from statsmodels.nonparametric.smoothers_lowess import lowess
            span = float(minutes.max() - minutes.min())
            fitted = lowess(values, minutes, frac=frac, it=0,
                            delta=0.01 * span, return_sorted=True)
            return np.interp(grid, fitted[:, 0], fitted[:, 1])
        except Exception:
            pass
    elif method == "spline":
        try:
            from pygam import LinearGAM, s
            gam = LinearGAM(s(0, n_splines=n_splines),
                            lam=float(DEFAULT_LAMBDA)).fit(
                minutes.reshape(-1, 1), values)
            return gam.predict(grid.reshape(-1, 1))
        except Exception:
            pass

    # Last resort: binned medians joined up.
    edges = np.linspace(minutes.min(), minutes.max(), 40)
    index = np.clip(np.digitize(minutes, edges[1:-1]), 0, len(edges) - 2)
    centers, medians = [], []
    for bucket in range(len(edges) - 1):
        mask = index == bucket
        if mask.sum() >= 5:
            centers.append(float(np.median(minutes[mask])))
            medians.append(float(np.median(values[mask])))
    if len(centers) < 3:
        return None
    return np.interp(grid, centers, medians)


def fit_curve(minutes: np.ndarray, values: np.ndarray, grid: np.ndarray,
              args: argparse.Namespace) -> np.ndarray | None:
    """Trend curve on `grid`, fitted independently before and after induction.

    Induction is a near-discontinuity. A single curve spanning it either rings
    (a global spline basis throws waves across the flat pre-induction stretch)
    or smears the drop backwards in time, which would invent the very lag this
    plot is meant to measure. Fitting each side separately avoids both.
    """
    if args.fit == "none":
        return None

    fitted = np.full(grid.shape, np.nan)
    for point_side, grid_side in ((minutes < 0, grid < 0),
                                  (minutes >= 0, grid >= 0)):
        segment = _fit_segment(minutes[point_side], values[point_side],
                               grid[grid_side], args.fit,
                               args.lowess_frac, args.n_splines)
        if segment is not None:
            fitted[grid_side] = segment
    return None if np.isnan(fitted).all() else fitted


def patient_coverage(points: pd.DataFrame, grid: np.ndarray,
                     n_bins: int = 200) -> np.ndarray:
    """How many distinct patients have data near each grid point.

    Used to blank the trend curve where the cohort thins out. A stretch held up
    by one patient is that patient's record, not a trend, and drawing a line
    through it invites exactly the wrong reading.
    """
    edges = np.linspace(grid.min(), grid.max(), n_bins + 1)
    minutes = points["minutes"].to_numpy(float)
    codes, uniques = pd.factorize(points["subject_id"])
    bucket = np.clip(np.digitize(minutes, edges[1:-1]), 0, n_bins - 1)

    seen = np.zeros((len(uniques), n_bins), dtype=bool)
    seen[codes, bucket] = True
    per_bin = seen.sum(axis=0)
    return per_bin[np.clip(np.digitize(grid, edges[1:-1]), 0, n_bins - 1)]


def bootstrap_band(points: pd.DataFrame, grid: np.ndarray,
                   args: argparse.Namespace):
    """95% band from resampling PATIENTS, not rows.

    A patient contributes thousands of correlated samples, so a row bootstrap
    would produce an interval far narrower than the data supports.
    """
    minutes = points["minutes"].to_numpy(float)
    values = points["value"].to_numpy(float)
    rng = np.random.default_rng(args.seed)
    subject_ids = points["subject_id"].to_numpy()
    unique_ids = np.unique(subject_ids)
    by_subject = {sid: np.flatnonzero(subject_ids == sid) for sid in unique_ids}

    curves = []
    for replicate in range(args.n_boot):
        draw = rng.choice(unique_ids, size=len(unique_ids), replace=True)
        index = np.concatenate([by_subject[sid] for sid in draw])
        curve = fit_curve(minutes[index], values[index], grid, args)
        if curve is not None:
            curves.append(curve)

    if not curves:
        return None, None
    stacked = np.vstack(curves)
    return (np.nanpercentile(stacked, 2.5, axis=0),
            np.nanpercentile(stacked, 97.5, axis=0))


def lag_readout(grid: np.ndarray, fitted: np.ndarray,
                points: pd.DataFrame) -> list[str]:
    """When the signal actually starts moving after the drug is pushed.

    The direction is read from the data, not assumed: PSi falls after induction
    while StO2 typically rises, so the response is whichever extreme of the
    post-induction curve lies furthest from baseline.

    Baseline is the median of the raw pre-induction samples, not a fitted value,
    so the reference point does not depend on the smoother. The wobble of the
    fitted curve over the flat pre-induction stretch is used as a noise floor --
    a post-induction excursion smaller than that is the smoother breathing, not
    a response, and is reported as such.
    """
    pre = points.loc[points["minutes"] < 0, "value"]
    after = grid >= 0
    if pre.empty or not after.any() or np.isnan(fitted[after]).all():
        return ["Not enough data either side of induction for a lag readout."]

    baseline = float(pre.median())
    grid_after, curve_after = grid[after], fitted[after]
    usable = ~np.isnan(curve_after)
    grid_after, curve_after = grid_after[usable], curve_after[usable]

    extreme_index = int(np.argmax(np.abs(curve_after - baseline)))
    extreme = float(curve_after[extreme_index])
    change = extreme - baseline
    rising = change > 0
    word = "peak" if rising else "nadir"

    lines = [f"Pre-induction baseline (median of raw samples): {baseline:.1f}"]

    pre_curve = fitted[(grid < 0) & ~np.isnan(fitted)]
    noise = float(pre_curve.max() - pre_curve.min()) if pre_curve.size else 0.0
    if abs(change) <= max(noise, 1e-9):
        lines.append(
            f"No response detected: the curve moves {abs(change):.2f} units "
            f"after induction, within its own pre-induction wobble of "
            f"{noise:.2f}. Lag times are not meaningful here."
        )
        return lines

    lines.append(f"Curve {word} after induction: {extreme:.1f} at "
                 f"{grid_after[extreme_index]:.2f} min "
                 f"({'rise' if rising else 'fall'} of {abs(change):.1f} units)")

    for label, fraction in (("10%", 0.10), ("50%", 0.50), ("90%", 0.90)):
        target = baseline + fraction * change
        reached = np.flatnonzero(curve_after >= target if rising
                                 else curve_after <= target)
        if reached.size:
            moment = float(grid_after[reached[0]])
            lines.append(f"Time to {label} of the total change: {moment:.2f} min "
                         f"({moment * 60:.0f} s)")
        else:
            lines.append(f"Time to {label} of the total change: not reached in window")
    lines.append(f"Pre-induction wobble of the curve (noise floor): {noise:.2f} units")
    return lines


# --------------------------------------------------------------------------- #
# Figure
# --------------------------------------------------------------------------- #

def draw_panel(axis, panel: dict, args: argparse.Namespace) -> None:
    spec = SERIES[panel["series"]]
    points = panel["points"]
    n_points = len(points)

    axis.scatter(
        points["minutes"], points["value"],
        s=4, alpha=0.08 if n_points > 50_000 else 0.18,
        color=spec["color"], edgecolors="none", rasterized=True, zorder=2,
        label=f"raw samples (n={n_points:,}, "
              f"{points['subject_id'].nunique()} patients)",
    )

    fitted = panel.get("fitted")
    if fitted is not None:
        if panel.get("band_low") is not None:
            axis.fill_between(
                panel["grid"], panel["band_low"], panel["band_high"],
                color="#d1495b", alpha=0.20, zorder=4,
                label="95% band: refit after resampling patients",
            )
        fit_label = {
            "lowess": f"LOWESS local trend, span={args.lowess_frac}",
            "linear": "least-squares straight line",
            "spline": f"penalized spline, k={args.n_splines}",
        }.get(args.fit, args.fit)
        axis.plot(panel["grid"], fitted, color="#d1495b", lw=2.8, zorder=5,
                  label=f"trend through the dots — {fit_label}, fitted "
                        f"separately each side of induction")

    axis.axvline(0, color="black", ls="--", lw=1.5, zorder=3)
    axis.set_ylabel(spec["label"])
    axis.grid(True, color="#d9d9d9", lw=0.6, alpha=0.8)
    axis.set_axisbelow(True)
    legend = axis.legend(loc="lower left", framealpha=0.95, fontsize=8.5,
                         markerscale=3)
    for handle in legend.legend_handles:
        try:
            handle.set_alpha(1.0)
        except Exception:
            pass


def make_figure(panels: list[dict], args: argparse.Namespace,
                output_path: Path) -> None:
    figure, axes = plt.subplots(
        len(panels), 1, sharex=True, squeeze=False,
        figsize=(13, 5.0 * len(panels)),
    )
    axes = axes.ravel()

    for axis, panel in zip(axes, panels):
        draw_panel(axis, panel, args)

    axes[0].annotate(
        "INDUCTION", xy=(0, 1.0), xycoords=("data", "axes fraction"),
        xytext=(4, -10), textcoords="offset points",
        fontsize=10, fontweight="bold", va="top", color="black",
    )
    axes[0].set_xlim(args.min_minutes, args.max_minutes)
    axes[-1].set_xlabel("Time from induction (minutes)   —   "
                        "0 = primary induction med pushed")

    names = " and ".join(SERIES[panel["series"]]["label"] for panel in panels)
    figure.suptitle(
        f"{names} versus time from induction\n"
        f"every valid raw sample plotted, nothing averaged — "
        f"panels share the induction anchor, so changes line up vertically",
        fontsize=13,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


# --------------------------------------------------------------------------- #
# Terminal reporting
# --------------------------------------------------------------------------- #

def banner(title: str) -> None:
    print("\n" + title)
    print("-" * max(len(title), 60))


def report_panel(panel: dict, args: argparse.Namespace) -> None:
    series, points, audit = panel["series"], panel["points"], panel["audit"]
    listed = len(audit)

    banner(f"=== {series} ===")
    print(f"{listed} file(s) listed; {points['subject_id'].nunique()} patients "
          f"plotted, {len(points):,} raw samples")
    print(f"Window: {args.min_minutes:g} to {args.max_minutes:g} minutes from "
          f"induction")

    if (args.max_artf is None and SERIES[series]["kind"] == "sedline"
            and "rows_artf_gt_20" in audit):
        would_drop = int(audit["rows_artf_gt_20"].fillna(0).sum())
        if would_drop:
            print(f"Note: --max-artf 20 would drop {would_drop:,} more samples "
                  f"(artifact filter is currently OFF)")

    included = audit.loc[audit["status"] == "included"]
    no_or_entry = included.loc[included["or_entry_used"] != "yes"]
    if len(no_or_entry):
        banner(f"{series}: {len(no_or_entry)} included patient(s) without a "
               f"usable OR-entry time")
        print(no_or_entry[["subject_id", "record_start_min", "or_entry_used"]]
              .sort_values("subject_id").to_string(index=False))
        print("\nThese keep every sample inside the plotted window instead. "
              "Since the window starts at "
              f"{args.min_minutes:g} min that changes little, but the REDCap "
              "OR-entry cell is worth checking for them.")

    clipped = included.loc[included["points_clipped_left"] > 0]
    if len(clipped):
        banner(f"{series}: {len(clipped)} patient(s) with data left of "
               f"{args.min_minutes:g} min (clipped, not dropped)")
        print(clipped[["subject_id", "record_start_min", "first_minute_raw",
                       "points_clipped_left", "or_entry_used"]]
              .sort_values("first_minute_raw").head(10).to_string(index=False))

    excluded = audit.loc[audit["status"] != "included"]
    if excluded.empty:
        print(f"\n{series}: every listed file was included.")
    else:
        banner(f"{series}: {len(excluded)} file(s) excluded")
        for reason, group in excluded.groupby("status"):
            print(f"  {len(group):>3}  {reason}")
            columns = ["subject_id"]
            for extra in ("record_start_min", "record_end_min"):
                if extra in group and group[extra].notna().any():
                    columns.append(extra)
            print(group[columns].to_string(index=False, header=False,
                                           na_rep="-"))
        if "no samples inside the plotted window" in set(excluded["status"]):
            print("\n'No samples inside the plotted window' shows each record's "
                  "true span in minutes from induction. A span sitting hours "
                  "away means the monitor clock and the REDCap clock disagree "
                  "for that patient, not that data is missing.")


def summary_row(series: str, label: str, subset: pd.DataFrame) -> dict | None:
    if len(subset) <= 2:
        return None
    slope, _intercept = np.polyfit(subset["minutes"], subset["value"], 1)
    return {
        "signal": series,
        "window": label,
        "n_points": len(subset),
        "n_patients": subset["subject_id"].nunique(),
        "slope_per_min": round(float(slope), 4),
        "pearson_vs_time": round(float(subset["minutes"].corr(subset["value"])), 4),
        "spearman_vs_time": round(
            float(subset["minutes"].corr(subset["value"], method="spearman")), 4),
        "median_value": round(float(subset["value"].median()), 2),
    }


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #

def main() -> int:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    wanted: list[tuple[str, Path]] = []
    if args.panels in ("both", "psi"):
        wanted.append((args.signal, args.filepaths))
    if args.panels in ("both", "sto2"):
        wanted.append(("StO2", args.sto2_filepaths))

    for _series, path in wanted:
        if not path.is_file():
            sys.stderr.write(f"Input file does not exist: {path}\n")
            return 1
    if not args.redcap.is_file():
        sys.stderr.write(f"Input file does not exist: {args.redcap}\n")
        return 1
    if args.min_minutes >= args.max_minutes:
        sys.stderr.write("--min-minutes must be less than --max-minutes\n")
        return 1

    events, notes = load_event_times(args.redcap,
                                     auto_repair=not args.no_id_repair)
    if notes:
        banner("REDCap repairs")
        for note in notes:
            print(f"  {note}")
    print(f"\nREDCap: {len(events)} patients; induction time for "
          f"{int(events['induction'].notna().sum())}, OR-entry for "
          f"{int(events['or_entry'].notna().sum())}")

    grid = np.linspace(args.min_minutes, args.max_minutes, 600)
    panels: list[dict] = []
    for series, filepaths in wanted:
        points, audit = build_points(filepaths, series, args, events)
        if points.empty:
            banner(f"=== {series} ===")
            print("No plottable samples.")
            if not audit.empty:
                print(audit["status"].value_counts().to_string())
            continue
        panels.append({"series": series, "points": points, "audit": audit,
                       "grid": grid})

    if not panels:
        sys.stderr.write("\nNothing to plot.\n")
        return 1

    for panel in panels:
        report_panel(panel, args)

    for panel in panels:
        points = panel["points"]
        if args.fit == "none":
            continue
        fitted = fit_curve(points["minutes"].to_numpy(float),
                           points["value"].to_numpy(float), grid, args)
        if fitted is None:
            continue
        coverage = patient_coverage(points, grid)
        thin = coverage < args.min_patients
        fitted = np.where(thin, np.nan, fitted)
        panel["fitted"] = fitted
        panel["coverage"] = coverage
        if thin.any():
            edges = grid[~thin]
            print(f"\n{panel['series']}: trend curve drawn only from "
                  f"{edges.min():.1f} to {edges.max():.1f} min, where at least "
                  f"{args.min_patients} patients have data.")
        if args.band and args.n_boot > 0:
            print(f"{panel['series']}: resampling patients for the 95% band "
                  f"({args.n_boot} replicates)...")
            low, high = bootstrap_band(points, grid, args)
            if low is not None:
                panel["band_low"] = np.where(thin, np.nan, low)
                panel["band_high"] = np.where(thin, np.nan, high)

    for panel in panels:
        if panel.get("fitted") is None:
            continue
        banner(f"{panel['series']}: lag readout after the drug is pushed")
        for line in lag_readout(grid, panel["fitted"], panel["points"]):
            print(f"  {line}")

    rows = []
    for panel in panels:
        points = panel["points"]
        for label, subset in (
            ("pre-induction", points.loc[points["minutes"] < 0]),
            ("post-induction", points.loc[points["minutes"] >= 0]),
            ("all", points),
        ):
            row = summary_row(panel["series"], label, subset)
            if row:
                rows.append(row)
    banner("Summary")
    print(pd.DataFrame(rows).to_string(index=False))

    stem = "_".join(panel["series"].lower() for panel in panels)
    figure_path = args.outdir / f"{stem}_vs_time_scatter.png"
    make_figure(panels, args, figure_path)
    print(f"\nFigure: {figure_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
