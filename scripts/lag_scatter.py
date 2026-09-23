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

No time window
--------------
  The whole intraoperative record is searched. Nothing is cut to a window. The
  only bound is that the extreme must occur at or after induction. Use
  --search-minutes if you ever want to restrict it.

  One caveat worth knowing rather than discovering later: over a long case, the
  lowest PSi of the whole record may belong to a deep-maintenance episode hours
  after induction rather than to induction itself. The run prints how late each
  patient's extreme occurred so this is visible; --search-minutes is the lever
  if it turns out to matter.

Data sources
------------
  * Sedline files, one path per line in --filepaths
  * StO2 files, one path per line in --sto2-filepaths
  * REDCap labeled export (--redcap) for induction time, OR entry and
    Date of Surgery

Output
------
  One PNG, plus a full per-patient inclusion report on the terminal naming every
  patient that did not make the figure and exactly why. No CSVs.

Usage
-----
    python lag_scatter.py
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
}

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


def load_sedline_file(path: str, series: str) -> pd.DataFrame | None:
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


def load_sto2_file(path: str, series: str) -> pd.DataFrame | None:
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


LOADERS = {"sedline": load_sedline_file, "sto2": load_sto2_file}


# --------------------------------------------------------------------------- #
# One row per patient
# --------------------------------------------------------------------------- #

def measure_patients(filepaths: Path, series: str, args: argparse.Namespace,
                     events: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
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
            frame = loader(path, series)
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

        record["status"] = "included"
        rows.append({k: record.get(k) for k in
                     ("subject_id", "baseline", "extreme_value", "t_extreme",
                      "t50", "change", "record_start_min", "record_end_min",
                      "min_before_record_end")})
        audit.append(record)

    return pd.DataFrame(rows), pd.DataFrame(audit)


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


def draw_pair_panel(axis, frame: pd.DataFrame, x_column: str, y_column: str,
                    x_label: str, y_label: str, title: str, subtitle: str,
                    seed: int, identity: bool) -> dict:
    data = frame[[x_column, y_column]].dropna()
    axis.scatter(data[x_column], data[y_column], s=46, alpha=0.75,
                 color="#3b6ea5", edgecolors="white", linewidths=0.8, zorder=3)
    stats = correlation_with_ci(data[x_column].to_numpy(float),
                                data[y_column].to_numpy(float), seed)

    if identity and len(data):
        lo = float(min(data[x_column].min(), data[y_column].min()))
        hi = float(max(data[x_column].max(), data[y_column].max()))
        pad = 0.05 * (hi - lo or 1.0)
        axis.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="#777777",
                  ls="--", lw=1.3, zorder=2, label="y = x (same time in both)")
        axis.set_xlim(lo - pad, hi + pad)
        axis.set_ylim(lo - pad, hi + pad)
        later = int((data[y_column] > data[x_column]).sum())
        # Sits below the legend, which occupies the top-left corner.
        axis.text(0.03, 0.80,
                  f"{later} of {len(data)} above the line\n"
                  f"(StO2 event came later)",
                  transform=axis.transAxes, va="top", ha="left", fontsize=9,
                  bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc"))

    if len(data) >= 3:
        slope, intercept = np.polyfit(data[x_column], data[y_column], 1)
        span = np.linspace(data[x_column].min(), data[x_column].max(), 50)
        axis.plot(span, intercept + slope * span, color="#d1495b", lw=2.2,
                  zorder=4, label="least-squares fit")

    caption = f"n = {stats['n']} patients"
    if "pearson" in stats:
        caption += f"\nPearson r = {stats['pearson']:+.2f}"
        if "ci" in stats:
            caption += f" (95% CI {stats['ci'][0]:+.2f} to {stats['ci'][1]:+.2f})"
        caption += f"\nSpearman rho = {stats['spearman']:+.2f}"
        # Pearson is driven by distance, Spearman only by rank, so a wide gap
        # between them means a handful of far-out dots are carrying the
        # correlation. Say so on the figure rather than letting the r be read
        # as a cohort result.
        if abs(stats["pearson"] - stats["spearman"]) > 0.3:
            stats["leverage"] = True
            caption += ("\n! r and rho disagree — a few outlying\n"
                        "  patients are driving r; trust rho")
    axis.text(0.97, 0.03, caption, transform=axis.transAxes, va="bottom",
              ha="right", fontsize=9,
              bbox=dict(boxstyle="round,pad=0.4", fc="#f7f7f7", ec="#cccccc"))

    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)
    axis.set_title(f"{title}\n{subtitle}", fontsize=11)
    axis.grid(True, color="#e0e0e0", lw=0.6)
    axis.set_axisbelow(True)
    axis.legend(loc="upper left", fontsize=8, framealpha=0.95)
    return stats


def make_figure(merged: pd.DataFrame, args: argparse.Namespace,
                output_path: Path) -> dict:
    figure, axes = plt.subplots(1, 3, figsize=(17, 5.8))
    stats = {}
    stats["value"] = draw_pair_panel(
        axes[0], merged, "extreme_value_psi", "extreme_value_sto2",
        "Lowest PSi reached", "Highest cerebral StO2 reached (%)",
        "A. Value pair",
        "how far down PSi went vs how far up StO2 went",
        args.seed, identity=False)
    stats["t_extreme"] = draw_pair_panel(
        axes[1], merged, "t_extreme_psi", "t_extreme_sto2",
        "Minutes after induction that PSi was lowest",
        "Minutes after induction that StO2 was highest",
        "B. Timing pair — when each signal hit its extreme",
        "above the dashed line = StO2 peaked after PSi bottomed out",
        args.seed, identity=True)
    stats["t50"] = draw_pair_panel(
        axes[2], merged, "t50_psi", "t50_sto2",
        "Minutes for PSi to fall halfway",
        "Minutes for StO2 to rise halfway",
        "C. Timing pair — time to half the change",
        "same question as B, steadier when a trace sits near its floor",
        args.seed, identity=True)

    figure.suptitle(
        "PSi and cerebral StO2 paired within patient — one dot per patient\n"
        "each patient contributes one number per signal, so these correlations "
        "are statistically legitimate",
        fontsize=13)
    figure.tight_layout(rect=(0, 0, 1, 0.9))
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return stats


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

    tables, audits = {}, {}
    for series, filepaths in (("PSi", args.filepaths),
                              ("StO2", args.sto2_filepaths)):
        tables[series], audits[series] = measure_patients(filepaths, series,
                                                          args, events)
        report_series(series, tables[series], audits[series])

    if tables["PSi"].empty or tables["StO2"].empty:
        sys.stderr.write("\nOne of the signals produced no patients.\n")
        return 1

    merged = tables["PSi"].merge(tables["StO2"], on="subject_id",
                                 suffixes=("_psi", "_sto2"))

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

    print(f"\nFigure: {figure_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
