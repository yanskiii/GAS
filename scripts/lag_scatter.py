#!/usr/bin/env python3
"""Per-patient lag scatters: PSi and cerebral StO2 paired one dot per patient.

This answers the question "does one signal turn before the other?" in the form
it was asked: each patient contributes ONE number per signal, and the two are
plotted against each other.

For every patient we take two things from each of their two traces:

    PSi   -> the LOWEST value they reached, and the minute it happened
    StO2  -> the HIGHEST value they reached, and the minute it happened

(the direction is fixed per signal, because PSi falls and StO2 rises after
induction). Those four numbers split into two pairs, which are the two scatter
plots asked for:

    Panel A   value pair   lowest PSi        vs  highest StO2
    Panel B   timing pair  minute PSi lowest vs  minute StO2 highest
    Panel C   timing pair, alternative definition: minutes each signal took to
              travel half of its own change. Panel B can be jumpy when a trace
              sits near its floor for several minutes, because then "the minute
              of the lowest value" can land anywhere along that flat stretch.
              The half-way crossing barely moves, so C answers B's question
              with steadier numbers.

Panels B and C carry a y = x line. A patient above it is one whose StO2 event
came later than their PSi event. That line, and how the dots sit around it, is
the lag.

Why one dot per patient matters
-------------------------------
  A correlation is only legitimate when the points are independent. On a
  sample-level scatter one patient contributes thousands of correlated readings,
  so an r computed there is inflated by pseudo-replication and means little.
  Collapsing each patient to a single point removes that, which is why the
  confidence intervals here come from a plain bootstrap over the dots.

Where the extreme is looked for
------------------------------
  Within 20 minutes of induction by default (--search-minutes). This excludes
  nobody: every patient still gets a dot, the search is simply not allowed to
  wander past the induction response.

  That bound is not cosmetic. Searching whole records on the real cohort put 59
  of 89 patients' extremes more than an hour after induction -- the lowest PSi
  of a three-hour case is usually a deep-maintenance episode, and the highest
  StO2 is wherever it had drifted by the end. The timing panels then largely
  measure how long each case ran. Two thirds of a cohort is not an outlier
  problem, so the window is the honest lever, not --exclude.

  --search-minutes 0 restores the whole-record search and warns about it.

The lag itself
--------------
  The panels show the spread; the paired statistic is the answer. For both
  timing definitions the run reports the within-patient difference (StO2 minus
  PSi) with a bootstrap CI over patients and a Wilcoxon signed-rank test, and
  puts that sentence directly in the panel subtitle.

  The two definitions can legitimately disagree in sign. A slow drifting signal
  can start moving BEFORE a fast one yet reach its extreme AFTER it, so an
  earlier half-way crossing alongside a later peak is one coherent story, not a
  contradiction. The run says so explicitly when it happens.

Cross-correlation: the same lag, measured without choosing an event
------------------------------------------------------------------
  The panels above all depend on picking an event to time, and panels B and C
  show that the answer can depend on which one. The second figure avoids the
  choice entirely: each patient's two traces are resampled onto a shared grid
  and slid against each other, and the shift that best aligns them is that
  patient's lag. One number per patient, then a histogram.

  PSi falls while StO2 rises, so the two are anti-correlated and the best
  alignment is the most NEGATIVE correlation. Sign convention matches the rest
  of the script: positive lag means StO2 follows PSi.

  If this agrees in direction with the half-way crossing, the lag has survived a
  method that never picks an event at all, which is a genuinely independent
  check rather than a restatement. The run says whether they agree.

Three-signal timing map
-----------------------
  With beat-to-beat MAP added, the third figure places all three signals on one
  timing axis within each patient: a ladder of every patient's three event
  times, and a forest of the three pairwise differences with intervals. This is
  what connects the lag work to the project's actual question, since hypotension
  is the outcome and MAP is where it shows up.

  MAP is optional. If no MAP file list is found the other two figures are
  produced exactly as before.

Data sources
------------
  * Sedline files, one path per line in --filepaths
  * StO2 files, one path per line in --sto2-filepaths
  * Beat-to-beat MAP files in --map-filepaths (optional; databad == 1 rows are
    dropped, and clock-only timestamps are re-dated onto the surgery day)
  * REDCap labeled export (--redcap) for induction time, OR entry and
    Date of Surgery

Output
------
  Three PNGs:
    psi_sto2_lag_scatter.png       the paired per-patient scatters
    psi_sto2_crosscorrelation.png  the waveform-based lag, as a histogram
    psi_sto2_map_timing_map.png    all three signals on one timing axis
  Plus a full per-patient inclusion report on the terminal naming every patient
  that did not make a figure and exactly why. No CSVs.

Usage
-----
    python lag_scatter.py
    python lag_scatter.py --map-filepaths /path/to/btb_filepaths.csv
    python lag_scatter.py --search-minutes 30
    python lag_scatter.py --smooth-samples 1     # no smoothing at all
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
# --------------------------------------------------------------------------- #
REDCAP_ID_FIXES: dict[tuple[str, str | None], str] = {
    # Real patient IUMH2026010601 exists (surgery 2026-01-06), so this typo can
    # only be corrected on its own surgery date.
    ("IUMH2026010601", "2026-01-05"): "IUMH2026010501",
    # 13 characters -- a digit was dropped from the day. Not a real ID.
    ("IUMH202601601", None): "IUMH2026011601",
}

# The two signals, and which end of each trace is the response.
SERIES = {
    "PSi": {
        "kind": "sedline", "column_prefix": "psi", "direction": "min",
        "label": "PSi (Sedline)", "valid_range": (0.0, 100.0),
        "extreme_word": "lowest", "event": "PSi was at its lowest",
        "half_word": "to fall halfway",
    },
    "StO2": {
        "kind": "sto2", "column_prefix": None, "direction": "max",
        "label": "Cerebral StO2 (%)", "valid_range": (0.0, 100.0),
        "extreme_word": "highest", "event": "StO2 was at its highest",
        "half_word": "to rise halfway",
    },
    "MAP": {
        "kind": "map", "column_prefix": "meanarterial", "direction": "min",
        "label": "Mean arterial pressure (mmHg)", "valid_range": (20.0, 160.0),
        "extreme_word": "lowest", "event": "MAP was at its lowest",
        "half_word": "to fall halfway",
    },
}

# Colour per signal, used by the three-signal timing figure.
SIGNAL_COLORS = {"PSi": "#4c78a8", "StO2": "#2f7d4f", "MAP": "#b5651d"}

# Filenames tried for the beat-to-beat MAP list when --map-filepaths is left
# at its default, since this list has been called several things.
MAP_LIST_FALLBACKS = ("map_filepaths.csv", "btb_filepaths.csv",
                      "b2b_filepaths.csv", "b2b_filepath.csv",
                      "btb_filepath.csv")

STO2_CHANNELS = (1, 2, 3, 4)
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
    base = Path("/N/project/Analgesia_BDproject/PR/scripts_PR/9-23 Lag Scatter")
    data = Path("/N/project/Analgesia_BDproject/PR/data")
    parser.add_argument(
        "--filepaths", type=Path, default=data / "sedline_filepaths.csv",
        help="CSV listing one Sedline file path per line.")
    parser.add_argument(
        "--sto2-filepaths", type=Path, default=data / "sto2_filepaths.csv",
        help="CSV listing one cerebral-StO2 file path per line.")
    parser.add_argument(
        "--map-filepaths", type=Path, default=None,
        help="CSV listing one beat-to-beat MAP file path per line. Left "
             "unset, several usual names are tried inside the data folder; if "
             "none is found the two MAP figures are skipped and the rest of "
             "the run is unaffected.")
    parser.add_argument(
        "--redcap", type=Path,
        default=data / ("PR_6.16.26.FIXED-TYPOS-BDPostInductionHemod_"
                        "DATA_LABELS_2026-06-16_1657.csv"),
        help="REDCap labeled export holding induction and OR-entry times.")
    parser.add_argument(
        "--outdir", type=Path, default=base / "output",
        help="Where the figure is written. No CSVs are produced.")
    parser.add_argument(
        "--search-minutes", type=float, default=20.0,
        help="Look for each patient's extreme within this many minutes after "
             "induction (default 20). This does NOT exclude patients -- every "
             "patient still gets a dot; it only stops the search wandering "
             "hours past induction into the maintenance phase, which is a "
             "different event from the one being measured. Pass 0 to search "
             "the whole record.")
    parser.add_argument(
        "--baseline-minutes", type=float, default=None,
        help="Optional: use only the last N minutes before induction as the "
             "baseline. Off by default -- every pre-induction sample from OR "
             "entry onward is used. Baseline only affects panel C.")
    parser.add_argument(
        "--smooth-samples", type=int, default=15,
        help="Rolling-median width applied to each patient's trace before "
             "finding their extreme (default 15 samples, ~30 s at 2 s). Stops "
             "one stray reading defining a patient's lowest value. Pass 1 to "
             "use the raw values untouched.")
    parser.add_argument(
        "--min-samples", type=int, default=10,
        help="Post-induction samples a patient needs to be timed (default 10).")
    parser.add_argument(
        "--exclude", nargs="*", default=[], metavar="ID",
        help="Research IDs to drop from the figure entirely, e.g. "
             "--exclude IUMH2026011501. Every exclusion is named in the "
             "report, so the figure never hides who is missing.")
    parser.add_argument(
        "--max-lag", type=float, default=5.0,
        help="Widest lead or lag tested by the cross-correlation, in minutes "
             "(default 5). The search runs from -max-lag to +max-lag.")
    parser.add_argument(
        "--ccf-step-seconds", type=float, default=10.0,
        help="Grid spacing both traces are resampled onto before "
             "cross-correlating (default 10 s). Also the resolution of the "
             "lag estimate.")
    parser.add_argument(
        "--ccf-lead-minutes", type=float, default=3.0,
        help="Minutes of pre-induction record included in the "
             "cross-correlation (default 3). Some baseline anchors the "
             "'before' state; too much dilutes the response with flat signal.")
    parser.add_argument(
        "--no-id-repair", action="store_true",
        help="Skip the automatic Date-of-Surgery ID check. The hardcoded "
             "REDCAP_ID_FIXES table still applies.")
    parser.add_argument("--seed", type=int, default=20260923)
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
    """Research ID embedded in the file path, e.g. IUMH2026030501."""
    match = ID_PATTERN.search(path)
    return match.group(0).upper() if match else None


def find_column(frame: pd.DataFrame, prefix: str) -> str | None:
    """Locate a column by case-insensitive prefix after stripping whitespace.

    Sedline exports ship a leading space on ' Time', and the REDCap labeled
    export uses the full question text as the header.
    """
    prefix = prefix.strip().lower()
    for column in frame.columns:
        if str(column).strip().lower().startswith(prefix):
            return column
    return None


def repair_subject_ids(raw_ids: pd.Series, surgery_dates: pd.Series,
                       auto_repair: bool = True) -> tuple[pd.Series, list[str]]:
    """Resolve each REDCap research ID: separators, hardcoded table, then date.

    The ID carries the surgery date inside it (IU + hospital + YYYYMMDD +
    patient of the day), so the two fields check each other. When they disagree
    the date is believed. A repair is refused if it would land on an ID another
    row already holds legitimately, so a typo cannot overwrite a real patient.
    """
    text = raw_ids.astype("string").str.strip().str.upper()
    cleaned = text.str.replace(r"[^A-Z0-9]", "", regex=True)

    notes: list[str] = []
    separator_fixes = int((text.notna() & text.ne(cleaned)).sum())
    for original, tidy in zip(text, cleaned):
        if pd.notna(original) and original != tidy:
            notes.append(f"{original} -> {tidy}  (separator stripped)")
            break

    resolved: list[str | None] = []
    pending: list[bool] = []
    hardcoded: list[str] = []
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
            hardcoded.append(f"{value} -> {fix}  (hardcoded"
                             + (f", Date of Surgery {stamp}"
                                if (value, stamp) in REDCAP_ID_FIXES else "") + ")")
        else:
            resolved.append(value)
            pending.append(True)
    notes.extend(sorted(set(hardcoded)))
    if separator_fixes:
        notes.append(f"{separator_fixes} ID(s) had separators stripped")

    if not auto_repair:
        return pd.Series(resolved, index=raw_ids.index, dtype="object"), notes

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
                f"left unchanged")
            continue
        final.append(candidate)
        automatic.append(f"{value} -> {candidate}  (Date of Surgery {date:%Y-%m-%d})")

    notes.extend(sorted(set(automatic)))
    return pd.Series(final, index=raw_ids.index, dtype="object"), notes


def find_record_key(redcap: pd.DataFrame) -> str | None:
    """The column identifying which REDCap record a row belongs to."""
    for column in redcap.columns[:3]:
        values = redcap[column]
        if values.notna().all() and 1 < values.nunique() < len(redcap):
            return column
    return None


def fill_across_record_rows(redcap: pd.DataFrame,
                            columns: list[str]) -> tuple[int, str]:
    """Carry the ID and surgery date across a record's repeating-instrument rows.

    REDCap writes one row per instrument, and only the row owning a field
    carries its value, so a patient's OR-entry time often sits on a row whose ID
    cell is blank. Without this the time is invisible and the patient looks like
    it has none recorded.
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
            redcap[column] = redcap[column].ffill()
        filled += int(redcap[column].notna().sum()) - before
    how = (f"grouped by '{record_key}'" if record_key is not None
           else "forward-filled (no record-key column found)")
    return filled, how


def load_event_times(redcap_path: Path,
                     auto_repair: bool = True) -> tuple[pd.DataFrame, list[str]]:
    """Per-patient induction and OR-entry instants from the REDCap export."""
    redcap = pd.read_csv(redcap_path, low_memory=False)

    id_column = find_column(redcap, REDCAP_ID_PREFIX)
    if id_column is None:
        best = 0
        for column in redcap.columns:
            # NB: match the pattern as written - upper-casing it would turn \d
            # into \D and silently match nothing.
            hits = int(redcap[column].astype("string").str.strip()
                       .str.fullmatch(r"IU(?:MH|UH)\d+", case=False, na=False).sum())
            if hits > best:
                id_column, best = column, hits
    if id_column is None:
        raise ValueError("No research-ID column (IUMH.../IUUH...) found in REDCap.")

    induction_column = find_column(redcap, INDUCTION_PREFIX)
    or_entry_column = find_column(redcap, OR_ENTRY_PREFIX)
    date_column = find_column(redcap, SURGERY_DATE_PREFIX)
    for name, column in (("What time was INDUCTION...", induction_column),
                         ("What time did the patient enter the OR?", or_entry_column),
                         ("Date of Surgery", date_column)):
        if column is None:
            raise ValueError(f"REDCap has no '{name}' column.")

    notes: list[str] = []
    filled, how = fill_across_record_rows(redcap, [id_column, date_column])
    if filled:
        notes.append(f"{filled} blank ID / Date-of-Surgery cell(s) filled from "
                     f"sibling rows of the same REDCap record, {how}")

    surgery_date = pd.to_datetime(redcap[date_column], errors="coerce")

    def resolve(values: pd.Series) -> pd.Series:
        """Attach Date of Surgery to bare HH:MM[:SS] entries."""
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
            parsed.loc[other] = pd.to_datetime(text.loc[other], errors="coerce",
                                               format="mixed")
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

    # A late-evening case crosses midnight: both clock times get stamped with
    # the same Date of Surgery, putting induction ~23 h BEFORE OR entry.
    crossed = (induction_at.notna() & or_entry_at.notna()
               & ((or_entry_at - induction_at) > pd.Timedelta(hours=12)))
    induction_at.loc[crossed] = induction_at.loc[crossed] + pd.Timedelta(days=1)
    if int(crossed.sum()):
        notes.append(f"{int(crossed.sum())} case(s) crossed midnight between OR "
                     f"entry and induction; induction rolled to the next day")

    events = pd.DataFrame({"subject_id": subject_ids, "induction": induction_at,
                           "or_entry": or_entry_at})
    events = events.loc[events["subject_id"].notna() & events["subject_id"].ne("")]
    events = (events.sort_values("subject_id")
              .groupby("subject_id", as_index=True)
              .agg(induction=("induction", "first"), or_entry=("or_entry", "first")))
    return events, notes


def load_sedline_file(path: str, series: str,
                      induction: pd.Timestamp | None = None) -> pd.DataFrame | None:
    """Timestamped, valid samples of one Sedline signal from one patient."""
    frame = pd.read_csv(path, low_memory=False)
    value_column = find_column(frame, SERIES[series]["column_prefix"])
    if value_column is None:
        return None

    date_column = find_column(frame, "date")
    time_column = find_column(frame, "time")        # ' Time' has a leading space
    epoch_column = find_column(frame, "epoch")

    timestamp = pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns]")
    if date_column and time_column and date_column != time_column:
        timestamp = pd.to_datetime(
            frame[date_column].astype("string").str.strip() + " "
            + frame[time_column].astype("string").str.strip(),
            errors="coerce", format="mixed")
    if timestamp.isna().all() and epoch_column:
        timestamp = pd.to_datetime(pd.to_numeric(frame[epoch_column], errors="coerce"),
                                   unit="ms", errors="coerce")
    if timestamp.isna().all():
        return None

    # "-" means the monitor had no value; to_numeric turns it into a real NaN.
    return pd.DataFrame({"timestamp": timestamp,
                         "value": pd.to_numeric(frame[value_column], errors="coerce")})


def load_sto2_file(path: str, series: str,
                   induction: pd.Timestamp | None = None) -> pd.DataFrame | None:
    """Cerebral StO2 averaged over the channels flagged valid on each row."""
    frame = pd.read_csv(path, low_memory=False)
    time_column = find_column(frame, "time")
    if time_column is None:
        return None
    timestamp = pd.to_datetime(frame[time_column].astype("string").str.strip(),
                               errors="coerce", format="mixed")

    present = 0
    total = pd.Series(0.0, index=frame.index)
    count = pd.Series(0, index=frame.index)
    for channel in STO2_CHANNELS:
        value_column = find_column(frame, f"sto2_ch{channel}")
        if value_column is None:
            continue
        present += 1
        value = pd.to_numeric(frame[value_column], errors="coerce")
        valid_column = find_column(frame, f"valid_ch{channel}")
        if valid_column is not None:
            value = value.where(pd.to_numeric(frame[valid_column],
                                              errors="coerce").eq(1))
        total = total.add(value.fillna(0.0))
        count = count.add(value.notna().astype(int))
    if present == 0:
        return None
    return pd.DataFrame({"timestamp": timestamp,
                         "value": (total / count).where(count > 0)})


def anchor_clock_only(timestamp: pd.Series, induction: pd.Timestamp) -> pd.Series:
    """Re-date bare clock times onto the surgery day, and unwrap midnight.

    The beat-to-beat exports often carry a time of day with no date, which
    pandas dates to today. Re-anchoring to the induction date fixes that, and a
    record that runs past midnight then shows a ~24 h backward jump, which is
    undone by adding a day to everything after each wrap.
    """
    if timestamp.isna().all() or pd.isna(induction):
        return timestamp
    today = pd.Timestamp.today().normalize()
    on_today = timestamp.dt.normalize().eq(today)
    if on_today.mean() > 0.9:
        timestamp = induction.normalize() + (timestamp - today)
    wrapped = timestamp.diff() < pd.Timedelta(hours=-12)
    if wrapped.any():
        timestamp = timestamp + pd.to_timedelta(wrapped.cumsum(), unit="D")
    return timestamp


def load_map_file(path: str, series: str,
                  induction: pd.Timestamp | None = None) -> pd.DataFrame | None:
    """Beat-to-beat MAP, with the monitor's own bad-data rows dropped."""
    frame = pd.read_csv(path, low_memory=False, skipinitialspace=True)
    frame.columns = [str(column).strip() for column in frame.columns]

    value_column = (find_column(frame, SERIES[series]["column_prefix"])
                    or find_column(frame, "map"))
    time_column = find_column(frame, "time")
    if value_column is None or time_column is None:
        return None

    # databad == 1 marks a beat the monitor itself flagged as unreliable.
    bad_column = find_column(frame, "databad")
    keep = pd.Series(True, index=frame.index)
    if bad_column is not None:
        keep &= pd.to_numeric(frame[bad_column], errors="coerce").ne(1)

    timestamp = pd.to_datetime(frame[time_column].astype("string").str.strip(),
                               errors="coerce", format="mixed")
    timestamp = anchor_clock_only(timestamp, induction)
    return pd.DataFrame({
        "timestamp": timestamp,
        "value": pd.to_numeric(frame[value_column], errors="coerce"),
    }).loc[keep]


LOADERS = {"sedline": load_sedline_file, "sto2": load_sto2_file,
           "map": load_map_file}


def resolve_map_list(explicit: Path | None, data_dir: Path) -> Path | None:
    """The MAP file list, if one can be found. None means skip MAP quietly."""
    if explicit is not None:
        return explicit if explicit.is_file() else None
    for name in MAP_LIST_FALLBACKS:
        candidate = data_dir / name
        if candidate.is_file():
            return candidate
    return None


# --------------------------------------------------------------------------- #
# One row per patient
# --------------------------------------------------------------------------- #

def ccf_grid(args: argparse.Namespace) -> np.ndarray:
    """Shared time axis every patient's trace is resampled onto.

    One grid for everybody is what makes the per-patient curves stackable and
    the lag axis mean the same thing for each patient.
    """
    step = float(args.ccf_step_seconds) / 60.0
    high = float(args.search_minutes) if args.search_minutes else 60.0
    return np.arange(-abs(args.ccf_lead_minutes), high + step / 2.0, step)


def measure_patients(filepaths: Path, series: str, args: argparse.Namespace,
                     events: pd.DataFrame,
                     grid: np.ndarray | None = None
                     ) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """For each patient: their extreme value, when it happened, and their T50.

    No response-magnitude test is applied. The question asked is simply "what
    was the lowest PSi / highest StO2, and when", which every patient with
    post-induction data can answer. Screening on the size of the response would
    quietly drop the weak responders, and they are part of the distribution the
    plot exists to show.
    """
    spec = SERIES[series]
    low, high = spec["valid_range"]
    loader = LOADERS[spec["kind"]]
    want_min = spec["direction"] == "min"

    rows: list[dict] = []
    audit: list[dict] = []
    traces: dict[str, np.ndarray] = {}

    for path in read_filepath_list(filepaths):
        subject_id = patient_id_from_path(path)
        record = {"subject_id": subject_id, "path": Path(path).name}

        if subject_id is None:
            record["status"] = "no research ID in the file path"
            audit.append(record); continue
        if not Path(path).is_file():
            record["status"] = "file not found"
            audit.append(record); continue
        if subject_id not in events.index:
            record["status"] = "not in the REDCap export"
            audit.append(record); continue

        induction = events.loc[subject_id, "induction"]
        or_entry = events.loc[subject_id, "or_entry"]
        if pd.isna(induction):
            record["status"] = "no REDCap induction time"
            audit.append(record); continue

        try:
            frame = loader(path, series, induction)
        except Exception as exc:
            record["status"] = f"read error: {exc}"
            audit.append(record); continue
        if frame is None or frame.empty:
            record["status"] = "no readable rows or timestamps"
            audit.append(record); continue

        frame = frame.loc[frame["value"].between(low, high)
                          & frame["timestamp"].notna()].copy()
        if frame.empty:
            record["status"] = "no valid values"
            audit.append(record); continue

        frame["minutes"] = (frame["timestamp"] - induction).dt.total_seconds() / 60.0
        frame = frame.sort_values("minutes")
        record["record_start_min"] = round(float(frame["minutes"].min()), 1)
        record["record_end_min"] = round(float(frame["minutes"].max()), 1)

        # Smooth once, then split, so the baseline's scatter and the response
        # are measured the same way.
        width = max(int(args.smooth_samples), 1)
        smoothed = frame["value"].rolling(width, center=True, min_periods=1).median()

        after = frame["minutes"] >= 0.0
        if args.search_minutes:
            after &= frame["minutes"] <= args.search_minutes
        if int(after.sum()) < args.min_samples:
            record["status"] = (f"fewer than {args.min_samples} samples after "
                                f"induction (record runs "
                                f"{record['record_start_min']} to "
                                f"{record['record_end_min']} min)")
            audit.append(record); continue

        post_values = smoothed.loc[after].to_numpy(float)
        post_times = frame.loc[after, "minutes"].to_numpy(float)
        index = int(np.argmin(post_values) if want_min else np.argmax(post_values))
        record["extreme_value"] = round(float(post_values[index]), 2)
        record["t_extreme"] = round(float(post_times[index]), 2)

        # Baseline, used only for the half-way crossing in panel C. Bounded
        # below by OR entry so ward monitoring cannot leak into it.
        before = frame["minutes"] < 0.0
        if pd.notna(or_entry):
            in_or = frame["timestamp"] >= or_entry
            if int((before & in_or).sum()) > 0:
                before &= in_or
                record["baseline_from"] = "OR entry to induction"
            else:
                record["baseline_from"] = "record start (OR entry unusable)"
        else:
            record["baseline_from"] = "record start (no OR-entry time)"
        if args.baseline_minutes is not None:
            before &= frame["minutes"] >= -abs(args.baseline_minutes)

        record["baseline_n"] = int(before.sum())
        if record["baseline_n"] > 0:
            baseline = float(frame.loc[before, "value"].median())
            record["baseline"] = round(baseline, 2)
            change = record["extreme_value"] - baseline
            record["change"] = round(change, 2)
            if abs(change) > 1e-9:
                target = baseline + 0.5 * change
                reached = np.flatnonzero(post_values <= target if change < 0
                                         else post_values >= target)
                if reached.size:
                    record["t50"] = round(float(post_times[reached[0]]), 3)

        # How close the extreme sits to the end of the record. An extreme that
        # lands on the final samples usually means the trace was still drifting
        # when recording stopped, so "the highest value of the case" is really
        # "wherever it had got to by the end" -- a different quantity from a
        # response to induction.
        record["min_before_record_end"] = round(
            float(post_times[-1] - post_times[index]), 2)

        if grid is not None:
            # Resample onto the shared grid for the cross-correlation. Points
            # outside the record become NaN rather than being extrapolated, so
            # a short record contributes only where it actually has data.
            traces[subject_id] = np.interp(
                grid, frame["minutes"].to_numpy(float), smoothed.to_numpy(float),
                left=np.nan, right=np.nan)

        record["status"] = "included"
        rows.append({k: record.get(k) for k in
                     ("subject_id", "baseline", "extreme_value", "t_extreme",
                      "t50", "change", "record_start_min", "record_end_min",
                      "min_before_record_end")})
        audit.append(record)

    return pd.DataFrame(rows), pd.DataFrame(audit), traces


# --------------------------------------------------------------------------- #
# Figure
# --------------------------------------------------------------------------- #

def correlation_with_ci(x: np.ndarray, y: np.ndarray, seed: int,
                        n_boot: int = 2000) -> dict:
    """Pearson and Spearman with a bootstrap CI.

    Each point is one patient, so resampling points IS resampling patients.
    That is what makes a correlation defensible here and not on a sample-level
    scatter.
    """
    frame = pd.DataFrame({"x": x, "y": y}).dropna()
    result = {"n": len(frame)}
    if len(frame) < 5:
        return result
    result["pearson"] = float(frame["x"].corr(frame["y"]))
    result["spearman"] = float(frame["x"].corr(frame["y"], method="spearman"))
    rng = np.random.default_rng(seed)
    values = frame.to_numpy()
    estimates = []
    for row in rng.integers(0, len(frame), size=(n_boot, len(frame))):
        sample = values[row]
        if np.std(sample[:, 0]) < 1e-12 or np.std(sample[:, 1]) < 1e-12:
            continue
        estimates.append(np.corrcoef(sample[:, 0], sample[:, 1])[0, 1])
    if estimates:
        result["ci"] = tuple(np.percentile(estimates, [2.5, 97.5]))
    return result


def paired_difference(frame: pd.DataFrame, x_column: str, y_column: str,
                      seed: int, n_boot: int = 2000) -> dict | None:
    """Within-patient difference in event time, StO2 minus PSi.

    Pairing on the patient is what turns a cloud of dots into an answer: each
    patient acts as their own control, so the wide between-patient spread in
    absolute timing cancels and only the within-patient ordering is left.
    """
    data = frame[[x_column, y_column]].dropna()
    if len(data) < 5:
        return None
    difference = (data[y_column] - data[x_column]).to_numpy(float)
    rng = np.random.default_rng(seed)
    draws = rng.choice(difference, size=(n_boot, len(difference)), replace=True)
    low, high = np.percentile(np.median(draws, axis=1), [2.5, 97.5])
    result = {
        "n": len(data),
        "median_x": float(data[x_column].median()),
        "median_y": float(data[y_column].median()),
        "difference": float(np.median(difference)),
        "ci": (float(low), float(high)),
    }
    try:
        from scipy.stats import wilcoxon
        result["p"] = float(wilcoxon(difference).pvalue)
    except Exception:
        pass
    return result


def difference_sentence(stats: dict | None, event: str) -> str:
    """Plain-English verdict for a panel subtitle."""
    if stats is None:
        return "too few paired patients for a lag estimate"
    gap = stats["difference"]
    low, high = stats["ci"]
    interval = f"(paired, 95% CI {low:+.1f} to {high:+.1f} min)"
    # An interval straddling zero means the ordering is not established, and
    # the subtitle must not imply one.
    if low <= 0.0 <= high:
        return (f"no clear ordering: StO2 {event} {abs(gap):.1f} min "
                f"{'later' if gap > 0 else 'earlier'},\nbut the interval "
                f"spans zero {interval}")
    return (f"StO2 {event} {abs(gap):.1f} min "
            f"{'LATER' if gap > 0 else 'EARLIER'} than PSi\n"
            f"in the same patient {interval}")


def draw_pair_panel(axis, frame: pd.DataFrame, x_column: str, y_column: str,
                    x_label: str, y_label: str, title: str, subtitle: str,
                    seed: int, identity: bool) -> tuple[dict, str]:
    """Draw the dots and lines only.

    Nothing is written inside the axes: the numbers are returned as a caption
    and placed under the panel by the caller. Boxes floating over a scatter
    cover the very dots the reader is trying to judge, and on a projector the
    covered corner is usually where the interesting patients are.
    """
    data = frame[[x_column, y_column]].dropna()
    axis.scatter(data[x_column], data[y_column], s=46, alpha=0.75,
                 color="#3b6ea5", edgecolors="white", linewidths=0.8, zorder=3)
    stats = correlation_with_ci(data[x_column].to_numpy(float),
                                data[y_column].to_numpy(float), seed)

    lines: list[str] = []
    if identity and len(data):
        lo = float(min(data[x_column].min(), data[y_column].min()))
        hi = float(max(data[x_column].max(), data[y_column].max()))
        pad = 0.05 * (hi - lo or 1.0)
        axis.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="#777777",
                  ls="--", lw=1.3, zorder=2)
        axis.set_xlim(lo - pad, hi + pad)
        axis.set_ylim(lo - pad, hi + pad)
        later = int((data[y_column] > data[x_column]).sum())
        lines.append(f"{later} of {len(data)} patients above the line")
        lines.append("(StO2 event came later)")

    if len(data) >= 3:
        slope, intercept = np.polyfit(data[x_column], data[y_column], 1)
        span = np.linspace(data[x_column].min(), data[x_column].max(), 50)
        axis.plot(span, intercept + slope * span, color="#d1495b", lw=2.2,
                  zorder=4)

    # One short line each: three columns of captions sit side by side under the
    # figure, so anything wide collides with the neighbouring panel's text.
    head = [f"n = {stats['n']} patients"]
    if "pearson" in stats:
        pearson = f"Pearson r = {stats['pearson']:+.2f}"
        if "ci" in stats:
            pearson += f"  (95% CI {stats['ci'][0]:+.2f} to {stats['ci'][1]:+.2f})"
        head.append(pearson)
        head.append(f"Spearman rho = {stats['spearman']:+.2f}")
        # Pearson is driven by distance, Spearman only by rank, so a wide gap
        # between them means a handful of far-out dots are carrying the
        # correlation. Say so rather than letting r read as a cohort result.
        if abs(stats["pearson"] - stats["spearman"]) > 0.3:
            stats["leverage"] = True
            lines.append("! r and rho disagree — a few outlying")
            lines.append("patients are driving r; trust rho")
    lines[:0] = head

    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)
    axis.set_title(f"{title}\n{subtitle}", fontsize=11)
    axis.grid(True, color="#e0e0e0", lw=0.6)
    axis.set_axisbelow(True)
    return stats, "\n".join(lines)


def make_figure(merged: pd.DataFrame, args: argparse.Namespace,
                output_path: Path) -> dict:
    figure, axes = plt.subplots(1, 3, figsize=(17, 7.8))
    stats = {}
    gaps = {
        "t_extreme": paired_difference(merged, "t_extreme_psi",
                                       "t_extreme_sto2", args.seed),
        "t50": paired_difference(merged, "t50_psi", "t50_sto2", args.seed),
    }

    captions: list[str] = []
    stats["value"], caption = draw_pair_panel(
        axes[0], merged, "extreme_value_psi", "extreme_value_sto2",
        "Lowest PSi reached", "Highest cerebral StO2 reached (%)",
        "A. Value pair",
        "how far down PSi went vs how far up StO2 went",
        args.seed, identity=False)
    captions.append(caption)
    stats["t_extreme"], caption = draw_pair_panel(
        axes[1], merged, "t_extreme_psi", "t_extreme_sto2",
        "Minutes after induction that PSi was lowest",
        "Minutes after induction that StO2 was highest",
        "B. Timing pair — when each signal hit its extreme",
        difference_sentence(gaps["t_extreme"], "peaks"),
        args.seed, identity=True)
    # Dots pinned to the search boundary are not turning points; say how many
    # rather than letting them read as data.
    if args.search_minutes:
        edge = float(args.search_minutes)
        pinned = int(((merged["t_extreme_psi"] >= edge - 0.5)
                      | (merged["t_extreme_sto2"] >= edge - 0.5)).sum())
        if pinned:
            caption += (f"\n{pinned} dot(s) sit on the {edge:g} min edge —\n"
                        f"still moving when the search stopped")
    captions.append(caption)
    stats["t50"], caption = draw_pair_panel(
        axes[2], merged, "t50_psi", "t50_sto2",
        "Minutes for PSi to fall halfway",
        "Minutes for StO2 to rise halfway",
        "C. Timing pair — time to half the change",
        difference_sentence(gaps["t50"], "gets halfway"),
        args.seed, identity=True)
    captions.append(caption)
    stats["gaps"] = gaps

    figure.suptitle(
        "PSi and cerebral StO2 paired within patient — one dot per patient\n"
        "each patient contributes one number per signal, so these correlations "
        "are statistically legitimate",
        fontsize=13)
    # Leave the bottom third of the canvas empty: the captions stack under
    # each panel and the one shared legend sits below all of them, so nothing
    # is ever written over the data.
    figure.tight_layout(rect=(0, 0.32, 1, 0.91))

    for axis, caption in zip(axes, captions):
        box = axis.get_position()
        figure.text(box.x0 + box.width / 2.0, 0.275, caption, ha="center",
                    va="top", fontsize=9, linespacing=1.6, color="#333333")

    from matplotlib.lines import Line2D
    handles = [
        Line2D([], [], marker="o", ls="none", color="#3b6ea5", markersize=8,
               markeredgecolor="white", label="one patient"),
        Line2D([], [], color="#d1495b", lw=2.2, label="least-squares fit"),
        Line2D([], [], color="#777777", ls="--", lw=1.3,
               label="y = x — the same time in both signals (panels B and C)"),
    ]
    figure.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
                  fontsize=10, bbox_to_anchor=(0.5, 0.02))

    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return stats


# --------------------------------------------------------------------------- #
# Cross-correlation: the lag measured from the whole waveform
# --------------------------------------------------------------------------- #

def cross_correlation(x: np.ndarray, y: np.ndarray, max_lag_steps: int,
                      min_overlap: int = 20) -> np.ndarray:
    """corr(x(t), y(t + lag)) across lags, in grid steps.

    The sign convention matters and matches the rest of this script: a POSITIVE
    lag is y shifted later, so a peak at positive lag means y's response comes
    after x's.
    """
    out = np.full(2 * max_lag_steps + 1, np.nan)
    for position, lag in enumerate(range(-max_lag_steps, max_lag_steps + 1)):
        if lag >= 0:
            left, right = x[:len(x) - lag], y[lag:]
        else:
            left, right = x[-lag:], y[:len(y) + lag]
        usable = np.isfinite(left) & np.isfinite(right)
        if int(usable.sum()) < min_overlap:
            continue
        a, b = left[usable], right[usable]
        if np.std(a) < 1e-9 or np.std(b) < 1e-9:
            continue
        out[position] = float(np.corrcoef(a, b)[0, 1])
    return out


def cross_correlation_lags(traces_x: dict, traces_y: dict, grid: np.ndarray,
                           args: argparse.Namespace) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Per-patient best lag between two signals, plus every patient's curve.

    This is the lag measured from the SHAPE of the two traces rather than from
    one chosen event, which is what makes it a genuinely independent check on
    the event-timing panels. PSi falls while StO2 rises, so the two are
    anti-correlated and the alignment we want is the most NEGATIVE correlation.
    """
    step = float(args.ccf_step_seconds) / 60.0
    max_lag_steps = max(int(round(abs(args.max_lag) / step)), 1)
    lags = np.arange(-max_lag_steps, max_lag_steps + 1) * step

    rows, curves = [], []
    for subject_id in sorted(set(traces_x) & set(traces_y)):
        curve = cross_correlation(traces_x[subject_id], traces_y[subject_id],
                                  max_lag_steps)
        if np.isnan(curve).all():
            continue
        best = int(np.nanargmin(curve))
        rows.append({"subject_id": subject_id,
                     "lag": float(lags[best]),
                     "r": float(curve[best])})
        curves.append(curve)
    return pd.DataFrame(rows), lags, (np.vstack(curves) if curves
                                      else np.empty((0, len(lags))))


def make_ccf_figure(table: pd.DataFrame, lags: np.ndarray, curves: np.ndarray,
                    args: argparse.Namespace, output_path: Path) -> dict:
    figure, axes = plt.subplots(1, 2, figsize=(14, 6.6))
    summary: dict = {"n": len(table)}
    if table.empty:
        plt.close(figure)
        return summary

    values = table["lag"].to_numpy(float)
    rng = np.random.default_rng(args.seed)
    draws = rng.choice(values, size=(2000, len(values)), replace=True)
    low, high = np.percentile(np.median(draws, axis=1), [2.5, 97.5])
    summary.update({"median": float(np.median(values)),
                    "ci": (float(low), float(high)),
                    "strong": int((table["r"].abs() >= 0.5).sum()),
                    "moderate": int((table["r"].abs() >= 0.3).sum())})
    try:
        from scipy.stats import wilcoxon
        summary["p"] = float(wilcoxon(values).pvalue)
    except Exception:
        pass

    step = float(args.ccf_step_seconds) / 60.0
    axes[0].hist(values, bins=np.arange(lags.min() - step / 2,
                                        lags.max() + step, max(step, 0.25)),
                 color="#4c78a8", edgecolor="white")
    axes[0].axvline(0, color="black", ls="--", lw=1.4)
    axes[0].axvspan(low, high, color="#d1495b", alpha=0.18)
    axes[0].axvline(summary["median"], color="#d1495b", lw=2.4)
    axes[0].set_xlabel("Lag of StO2 behind PSi (minutes)\n"
                       "negative = StO2 leads   |   positive = StO2 follows")
    axes[0].set_ylabel("Number of patients")
    axes[0].set_title("Per-patient best-fitting lag", fontsize=11)
    axes[0].grid(True, axis="y", color="#e0e0e0", lw=0.6)
    axes[0].set_axisbelow(True)

    for curve in curves:
        axes[1].plot(lags, curve, color="#4c78a8", alpha=0.12, lw=0.9)
    with np.errstate(invalid="ignore"):
        mean_curve = np.nanmean(curves, axis=0)
    axes[1].plot(lags, mean_curve, color="#d1495b", lw=2.8)
    axes[1].axvline(0, color="black", ls="--", lw=1.4)
    axes[1].axhline(0, color="#999999", lw=1.0)
    axes[1].set_xlabel("Lag applied to StO2 (minutes)")
    axes[1].set_ylabel("Correlation between PSi and StO2")
    axes[1].set_title("Every patient's correlation curve, and their mean",
                      fontsize=11)
    axes[1].grid(True, color="#e0e0e0", lw=0.6)
    axes[1].set_axisbelow(True)

    verdict = (f"median lag {summary['median']:+.2f} min "
               f"(95% CI {low:+.2f} to {high:+.2f})")
    if low <= 0.0 <= high:
        verdict += " — spans zero, so no clear lead or lag"
    elif summary["median"] < 0:
        verdict += " — StO2 LEADS PSi"
    else:
        verdict += " — StO2 FOLLOWS PSi"
    figure.suptitle(
        "Lag measured from the whole waveform, not from a chosen event\n"
        + verdict, fontsize=13)

    figure.tight_layout(rect=(0, 0.13, 1, 0.88))
    figure.text(
        0.5, 0.085,
        f"n = {summary['n']} patients   |   "
        f"{summary['moderate']} with |r| >= 0.3   |   "
        f"{summary['strong']} with |r| >= 0.5\n"
        f"Each patient's PSi and StO2 traces are slid against each other; the "
        f"lag plotted is the shift that best aligns them.\n"
        f"PSi falls while StO2 rises, so the best alignment is the most "
        f"negative correlation.",
        ha="center", va="top", fontsize=9, linespacing=1.6, color="#333333")
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return summary


# --------------------------------------------------------------------------- #
# Three-signal timing map
# --------------------------------------------------------------------------- #

def make_timing_map_figure(merged: pd.DataFrame, order: list[str],
                           args: argparse.Namespace,
                           output_path: Path) -> dict:
    """Where PSi, StO2 and MAP each turn, within the same patient.

    The ladder shows every patient's three event times joined up, so the
    ordering is visible per patient rather than only on average; the forest
    beside it gives each pairwise difference with an interval.
    """
    figure, axes = plt.subplots(2, 2, figsize=(15, 10.5))
    results: dict = {}

    # Order the columns by median half-way time, so the ladder reads
    # left-to-right in the order the signals actually turn. The same order is
    # kept in the second row even if its medians disagree, because comparing
    # the two rows is the point and a reshuffle would hide any disagreement.
    order = sorted(order, key=lambda name: merged[f"t50_{name.lower()}"].median())

    for row, (measure, nice) in enumerate((("t50", "time to half the change"),
                                           ("t_extreme", "time of the extreme"))):
        columns = [f"{measure}_{name.lower()}" for name in order]
        data = merged[["subject_id"] + columns].dropna()

        ladder = axes[row][0]
        positions = np.arange(len(order))
        for values in data[columns].to_numpy(float):
            ladder.plot(positions, values, color="#999999", alpha=0.35, lw=0.8,
                        marker="o", markersize=3, zorder=2)
        for position, name in zip(positions, order):
            column = f"{measure}_{name.lower()}"
            ladder.scatter([position], [data[column].median()], s=170,
                           color=SIGNAL_COLORS[name], zorder=4,
                           edgecolors="black", linewidths=1.0)
            ladder.vlines(position, data[column].quantile(0.25),
                          data[column].quantile(0.75),
                          color=SIGNAL_COLORS[name], lw=6, alpha=0.45, zorder=3)
        ladder.set_xticks(positions)
        ladder.set_xticklabels(order)
        ladder.set_ylabel("Minutes after induction")
        ladder.set_title(f"{'AB'[row]}1. Each patient's {nice}\n"
                         f"grey line = one patient; dot = median, bar = IQR",
                         fontsize=11)
        ladder.grid(True, axis="y", color="#e0e0e0", lw=0.6)
        ladder.set_axisbelow(True)

        forest = axes[row][1]
        pairs = [(order[i], order[j])
                 for i in range(len(order)) for j in range(i + 1, len(order))]
        labels, entries = [], []
        for first, second in pairs:
            entry = paired_difference(data, f"{measure}_{first.lower()}",
                                      f"{measure}_{second.lower()}", args.seed)
            if entry is None:
                continue
            entries.append(entry)
            labels.append(f"{second} − {first}")
        results[measure] = dict(zip(labels, entries))

        for index, entry in enumerate(entries):
            crosses = entry["ci"][0] <= 0.0 <= entry["ci"][1]
            color = "#999999" if crosses else "#d1495b"
            forest.plot(entry["ci"], [index, index], color=color, lw=3,
                        solid_capstyle="round", zorder=3)
            forest.scatter([entry["difference"]], [index], s=90, color=color,
                           zorder=4, edgecolors="black", linewidths=0.8)
        forest.axvline(0, color="black", ls="--", lw=1.4, zorder=2)
        forest.set_yticks(range(len(labels)))
        forest.set_yticklabels(labels)
        forest.invert_yaxis()
        forest.set_xlabel("Minutes (positive = the second signal came later)")
        forest.set_title(f"{'AB'[row]}2. Paired difference, {nice}\n"
                         f"grey = interval crosses zero, so no clear ordering",
                         fontsize=11)
        forest.grid(True, axis="x", color="#e0e0e0", lw=0.6)
        forest.set_axisbelow(True)

    figure.suptitle(
        "Three-signal timing map — PSi, cerebral StO2 and MAP in the same "
        "patient\nwhich signal turns first, and by how much",
        fontsize=14)
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return results


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #

def banner(title: str) -> None:
    print("\n" + title)
    print("-" * max(len(title), 62))


def report_series(series: str, table: pd.DataFrame, audit: pd.DataFrame) -> None:
    banner(f"=== {series} ===")
    print(f"{len(audit)} file(s) listed; {len(table)} patient(s) measured")
    excluded = audit.loc[audit["status"] != "included"]
    if excluded.empty:
        print("Every listed file was measured.")
        return
    banner(f"{series}: {len(excluded)} file(s) NOT measured")
    for reason, group in excluded.groupby("status"):
        print(f"  {len(group):>3}  {reason}")
        for _, row in group.iterrows():
            print(f"         {row['subject_id'] or row['path']}")


def main() -> int:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    for path in (args.filepaths, args.sto2_filepaths, args.redcap):
        if not path.is_file():
            sys.stderr.write(f"Input file does not exist: {path}\n")
            return 1

    events, notes = load_event_times(args.redcap, auto_repair=not args.no_id_repair)
    if notes:
        banner("REDCap repairs")
        for note in notes:
            print(f"  {note}")
    print(f"\nREDCap: {len(events)} patients; induction time for "
          f"{int(events['induction'].notna().sum())}, OR-entry for "
          f"{int(events['or_entry'].notna().sum())}")
    if not args.search_minutes:
        print("Search window: the whole record from induction onward (NO "
              "LIMIT).\n  Warning: over a long case the lowest PSi is usually "
              "a maintenance-phase event, not the induction response, so the "
              "timing panels will largely measure how long each case ran.")
    else:
        print(f"Search window: induction to +{args.search_minutes:g} min. "
              f"No patient is excluded by this -- it only bounds where each "
              f"patient's extreme is looked for. Pass --search-minutes 0 to "
              f"search the whole record.")

    map_list = resolve_map_list(args.map_filepaths, args.redcap.parent)
    if map_list is None:
        banner("Beat-to-beat MAP: not found — the MAP figure will be skipped")
        if args.map_filepaths is not None:
            print(f"  --map-filepaths was given but does not exist: "
                  f"{args.map_filepaths}")
        else:
            print(f"  Looked in {args.redcap.parent} for: "
                  f"{', '.join(MAP_LIST_FALLBACKS)}")
        print("  Pass --map-filepaths <file> to include MAP. Everything else "
              "runs as normal.")
    else:
        print(f"\nBeat-to-beat MAP list: {map_list}")

    grid = ccf_grid(args)
    wanted = [("PSi", args.filepaths), ("StO2", args.sto2_filepaths)]
    if map_list is not None:
        wanted.append(("MAP", map_list))

    tables, audits, traces = {}, {}, {}
    for series, filepaths in wanted:
        tables[series], audits[series], traces[series] = measure_patients(
            filepaths, series, args, events, grid)
        report_series(series, tables[series], audits[series])

    if tables["PSi"].empty or tables["StO2"].empty:
        sys.stderr.write("\nOne of the signals produced no patients.\n")
        return 1

    def tag(table: pd.DataFrame, name: str) -> pd.DataFrame:
        """Suffix every measurement column so three tables can be merged."""
        return table.rename(columns={column: f"{column}_{name.lower()}"
                                     for column in table.columns
                                     if column != "subject_id"})

    merged = tag(tables["PSi"], "PSi").merge(tag(tables["StO2"], "StO2"),
                                             on="subject_id")

    # ---- who is furthest out, so an exclusion can be chosen by name ---------
    banner("Most extreme event times (candidates for --exclude)")
    print("  A record running many hours past induction is no longer an "
          "induction response, and a single such patient can stretch panels B "
          "and C so the rest collapse into the corner.\n")
    for label, column in (("PSi lowest", "t_extreme_psi"),
                          ("StO2 highest", "t_extreme_sto2"),
                          ("PSi halfway", "t50_psi"),
                          ("StO2 halfway", "t50_sto2")):
        ranked = merged[["subject_id", column]].dropna().nlargest(5, column)
        entries = ", ".join(f"{row.subject_id} ({getattr(row, column):.0f} min)"
                            for row in ranked.itertuples())
        print(f"  latest {label:<14} {entries}")

    if args.exclude:
        wanted = {str(value).strip().upper() for value in args.exclude}
        present = wanted & set(merged["subject_id"])
        missing = wanted - present
        merged = merged.loc[~merged["subject_id"].isin(present)]
        banner(f"Excluded by --exclude ({len(present)} patient(s))")
        for subject_id in sorted(present):
            print(f"  {subject_id}")
        if missing:
            print(f"  NOT FOUND (check the spelling): {', '.join(sorted(missing))}")
        if merged.empty:
            sys.stderr.write("\nEvery patient was excluded.\n")
            return 1

    banner("Patients in the figure")
    psi_ids, sto2_ids = set(tables["PSi"]["subject_id"]), set(tables["StO2"]["subject_id"])
    print(f"  PSi measured           {len(psi_ids)}")
    print(f"  StO2 measured          {len(sto2_ids)}")
    print(f"  in BOTH                {len(psi_ids & sto2_ids)}")
    if args.exclude:
        print(f"  after --exclude        {len(merged)}")
    print(f"  PLOTTED                {len(merged)}")
    print("\nA patient needs BOTH signals to be a dot, so the figure can never "
          "exceed the smaller of the two lists above.")
    for series, ids, other in (("PSi", psi_ids, sto2_ids),
                               ("StO2", sto2_ids, psi_ids)):
        missing = sorted(ids - other)
        if missing:
            print(f"\n  {len(missing)} patient(s) have {series} but no matching "
                  f"partner file, so they cannot be plotted:")
            print(f"    {', '.join(missing)}")

    has_t50 = int(merged[["t50_psi", "t50_sto2"]].notna().all(axis=1).sum())
    if has_t50 < len(merged):
        print(f"\n  Panel C additionally needs a pre-induction baseline; "
              f"{len(merged) - has_t50} of {len(merged)} plotted patients lack "
              f"one and appear in panels A and B only.")

    banner("How late each patient's extreme occurred (minutes after induction)")
    for series, column in (("PSi lowest", "t_extreme_psi"),
                           ("StO2 highest", "t_extreme_sto2")):
        values = merged[column].dropna()
        print(f"  {series:<14} median {values.median():6.1f}   "
              f"IQR {values.quantile(.25):.1f}-{values.quantile(.75):.1f}   "
              f"max {values.max():.1f}")
    late = merged.loc[(merged["t_extreme_psi"] > 60)
                      | (merged["t_extreme_sto2"] > 60), "subject_id"]
    if len(late):
        share = 100.0 * len(late) / len(merged)
        print(f"\n  {len(late)} of {len(merged)} patients ({share:.0f}%) have "
              f"an extreme more than an hour after induction.")
        if share >= 25.0:
            print(f"  !! At {share:.0f}% this is not a handful of outliers, it "
                  f"is most of the cohort, and excluding them one by one would "
                  f"be gerrymandering. It means the search is reaching past "
                  f"induction into maintenance. Lower --search-minutes rather "
                  f"than excluding patients.")
        print(f"    {', '.join(sorted(late))}")

    edge = "searched span" if args.search_minutes else "record"
    banner(f"Is the extreme a real turning point, or just the end of the {edge}?")
    worst = 0.0
    for series, column in (("PSi", "min_before_record_end_psi"),
                           ("StO2", "min_before_record_end_sto2")):
        at_end = merged.loc[merged[column] <= 2.0, "subject_id"]
        share = 100.0 * len(at_end) / len(merged) if len(merged) else 0.0
        worst = max(worst, share)
        print(f"  {series:<5} {len(at_end):>3} of {len(merged)} patients "
              f"({share:.0f}%) have their extreme within the last 2 minutes "
              f"of the {edge}")
    print(f"\n  A patient counted here was still drifting when the {edge} "
          f"ended, so their 'extreme' is really 'wherever it had got to by "
          f"then' rather than a turning point.")
    if worst >= 25.0:
        print(f"\n  !! {worst:.0f}% is high. For those patients panel B is "
              f"partly reporting where the {edge} ends rather than when the "
              f"signal turned, which flattens the spread and drags the dots "
              f"toward one edge. Panel C does not have this problem -- a "
              f"half-way crossing happens on the way to the extreme, so it "
              f"does not care where the search stops. Prefer panel C for the "
              f"lag claim, and treat panel B as support.")

    figure_path = args.outdir / "psi_sto2_lag_scatter.png"
    stats = make_figure(merged, args, figure_path)

    banner("THE LAG — paired within patient (this is the answer to the question)")
    print("  Each patient is their own control, so the wide spread in absolute "
          "timing cancels\n  and only the within-patient ordering is left. "
          "Positive = StO2 event came later.\n")
    for key, label, event in (("t_extreme", "peak / lowest point", "peaks"),
                              ("t50", "half-way crossing", "gets halfway")):
        entry = stats["gaps"].get(key)
        if entry is None:
            print(f"  {label}: too few paired patients")
            continue
        print(f"  {label} (n={entry['n']})")
        print(f"      PSi median {entry['median_x']:.2f} min, "
              f"StO2 median {entry['median_y']:.2f} min")
        print(f"      paired difference {entry['difference']:+.2f} min "
              f"(95% CI {entry['ci'][0]:+.2f} to {entry['ci'][1]:+.2f})"
              + (f", Wilcoxon p={entry['p']:.3g}" if "p" in entry else ""))
    both = [stats["gaps"].get("t_extreme"), stats["gaps"].get("t50")]
    if all(entry is not None for entry in both):
        peak_gap, half_gap = both[0]["difference"], both[1]["difference"]
        if peak_gap * half_gap < 0:
            print(f"\n  Note: the two measures point OPPOSITE ways "
                  f"({half_gap:+.2f} min at the half-way crossing, "
                  f"{peak_gap:+.2f} min at the extreme). That is not a "
                  f"contradiction: it means StO2 starts moving "
                  f"{'before' if half_gap < 0 else 'after'} PSi but finishes "
                  f"{'after' if peak_gap > 0 else 'before'} it, which is what "
                  f"a slow drifting signal does against a fast step. Report "
                  f"both.")

    banner("Correlations (one dot per patient)")
    labels = {"value": "A  lowest PSi vs highest StO2",
              "t_extreme": "B  minute PSi lowest vs minute StO2 highest",
              "t50": "C  time to fall halfway vs time to rise halfway"}
    for key, label in labels.items():
        entry = stats[key]
        if "pearson" not in entry:
            print(f"  {label}: too few patients")
            continue
        ci = entry.get("ci")
        print(f"  {label}")
        print(f"      n={entry['n']}  Pearson r={entry['pearson']:+.3f}"
              + (f" (95% CI {ci[0]:+.3f} to {ci[1]:+.3f})" if ci else "")
              + f"  Spearman rho={entry['spearman']:+.3f}")

    written = [figure_path]

    # ---- figure 2: the lag read off the whole waveform ---------------------
    ccf_table, lags, curves = cross_correlation_lags(
        traces["PSi"], traces["StO2"], grid, args)
    ccf_table = ccf_table.loc[ccf_table["subject_id"].isin(merged["subject_id"])]
    banner("CROSS-CORRELATION — the lag measured from the whole waveform")
    if len(ccf_table) < 5:
        print("  Too few patients with overlapping traces.")
    else:
        ccf_path = args.outdir / "psi_sto2_crosscorrelation.png"
        ccf = make_ccf_figure(ccf_table, lags, curves, args, ccf_path)
        written.append(ccf_path)
        print(f"  n = {ccf['n']} patients "
              f"({ccf['moderate']} with |r| >= 0.3, {ccf['strong']} >= 0.5)")
        print(f"  median lag {ccf['median']:+.2f} min "
              f"(95% CI {ccf['ci'][0]:+.2f} to {ccf['ci'][1]:+.2f})"
              + (f", Wilcoxon p={ccf['p']:.3g}" if "p" in ccf else ""))
        print("  Negative = StO2 leads PSi. Positive = StO2 follows.")
        half = stats["gaps"].get("t50")
        if half is not None:
            agree = (ccf["median"] < 0) == (half["difference"] < 0)
            print(f"\n  Half-way crossing said {half['difference']:+.2f} min; "
                  f"this says {ccf['median']:+.2f} min — "
                  + ("they AGREE in direction, which is the point of running "
                     "both: the lag survives a method that never picks an "
                     "event at all."
                     if agree else
                     "they DISAGREE in direction. The event-based number "
                     "depends on which event is chosen; this one does not, so "
                     "treat the event-based lag with caution and show both."))

    # ---- figure 3: all three signals on one timing map ----------------------
    if map_list is not None and not tables["MAP"].empty:
        merged3 = merged.merge(tag(tables["MAP"], "MAP"), on="subject_id")
        banner("THREE-SIGNAL TIMING MAP — PSi, StO2 and MAP together")
        print(f"  {len(merged3)} patient(s) have all three signals "
              f"(PSi {len(tables['PSi'])}, StO2 {len(tables['StO2'])}, "
              f"MAP {len(tables['MAP'])})")
        if len(merged3) < 5:
            print("  Too few patients with all three signals for the figure.")
        else:
            map_path = args.outdir / "psi_sto2_map_timing_map.png"
            order = ["PSi", "StO2", "MAP"]
            timing = make_timing_map_figure(merged3, order, args, map_path)
            written.append(map_path)
            for measure, nice in (("t50", "time to half the change"),
                                  ("t_extreme", "time of the extreme")):
                print(f"\n  {nice}:")
                for name in order:
                    values = merged3[f"{measure}_{name.lower()}"].dropna()
                    if len(values):
                        print(f"      {name:<5} median {values.median():6.2f} min "
                              f"(IQR {values.quantile(.25):.2f}-"
                              f"{values.quantile(.75):.2f})")
                for label, entry in timing.get(measure, {}).items():
                    crosses = entry["ci"][0] <= 0.0 <= entry["ci"][1]
                    print(f"      {label:<14} {entry['difference']:+6.2f} min "
                          f"(95% CI {entry['ci'][0]:+.2f} to "
                          f"{entry['ci'][1]:+.2f})"
                          + ("  — crosses zero" if crosses else ""))

    banner("Figures written")
    for path in written:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
