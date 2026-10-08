#!/usr/bin/env python3
"""MAP versus cerebral StO2, split by anesthetic depth (PSi) and cardiac output.

THE CLINICAL QUESTION
---------------------
  A healthy brain defends its own blood supply. When blood pressure falls,
  cerebral vessels dilate to keep flow -- and oxygen delivery -- roughly
  constant. That is cerebral autoregulation, and it is why a moderate drop in
  MAP usually does not harm the brain.

  Plot MAP on x and cerebral StO2 on y and the SLOPE of that line measures how
  well the defence is working:

      slope near zero   autoregulation intact. Pressure moves, brain oxygen
                        does not. The brain is protected.
      positive slope    autoregulation has failed. StO2 follows MAP, so the
                        brain is "pressure-passive" and hypotension translates
                        directly into less cerebral oxygen.

  The question this script answers is NOT "does MAP affect StO2". It is "DOES
  THE ANSWER DEPEND ON HOW DEEP THE PATIENT IS?" -- i.e. is the slope flat at
  light anesthesia and steep at deep anesthesia? If so, deep anesthesia breaks
  autoregulation, and a patient who is simultaneously deep and hypotensive is
  at real risk. That is a finding an anesthesiologist can act on.

  Statisticians call this an interaction, or effect modification: PSi modifies
  the MAP-StO2 relationship. The same question is then asked of cardiac output,
  because pressure and flow are not the same thing and a patient with poor
  output may deliver less blood at the same MAP.

TWO SLOPES, AND WHY BOTH ARE REPORTED
-------------------------------------
  The requested figure pools every sample from every patient inside a PSi band
  and fits one line. That line is easy to read but it is not a clean estimate:
  one patient contributes thousands of correlated samples, so its confidence
  interval is far too narrow, and it mixes two different things --

      BETWEEN patients  do patients who run at a higher MAP also run at a
                        higher StO2? (a question about who the patients are)
      WITHIN  patients  when THIS patient's MAP falls, does THIS patient's
                        StO2 follow? (the autoregulation question)

  Those two can even have opposite signs in the same data, which is Simpson's
  paradox. Only the within-patient slope answers the clinical question, so every
  band is reported both ways: the pooled slope that was asked for, and the
  median of the per-patient slopes with an interval obtained by resampling
  patients. Where they disagree, the within-patient number is the one to quote.

DATA SOURCES
------------
  --filepaths             Sedline files, for PSi
  --sto2-filepaths        cerebral oximetry files, for StO2
  --map-filepaths         beat-to-beat files, for MAP (databad == 1 dropped)
  --hemosphere-filepaths  Hemosphere files, for cardiac output. Columns are
                          Timestamp, Date, Time, Technology, CO, CO_SQI. Only
                          readings whose CO_SQI is at least --min-co-sqi
                          (default 3) are kept: a low SQI is the monitor saying
                          it does not trust its own number, and averaging those
                          in would blur the very bands being compared.
  --redcap                labeled export, for induction time and OR entry

  PSi, StO2 and MAP are all required of a patient. Cardiac output is optional
  and merged in only where a Hemosphere file exists, so the PSi analysis keeps
  its full cohort and only the CO split runs on the smaller subset.

OUTPUT
------
  map_sto2_by_psi.png   scatter per PSi band, plus the slope comparison
  map_sto2_by_co.png    the same split by cardiac output (only if CO is found)
  Everything else prints to the terminal. No CSVs.

USAGE
-----
    python stratified_autoregulation.py
    python stratified_autoregulation.py --psi-bin-width 20
    python stratified_autoregulation.py --min-co-sqi 4
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
# Hardcoded research-ID corrections (see the lag scatter script for the reason).
# --------------------------------------------------------------------------- #
REDCAP_ID_FIXES: dict[tuple[str, str | None], str] = {
    ("IUMH2026010601", "2026-01-05"): "IUMH2026010501",
    ("IUMH202601601", None): "IUMH2026011601",
}

VALID_RANGE = {"map": (20.0, 160.0), "sto2": (0.0, 100.0),
               "psi": (0.0, 100.0), "co": (0.5, 15.0)}

STO2_CHANNELS = (1, 2, 3, 4)
ID_PATTERN = re.compile(r"IU(?:MH|UH)\d+", re.IGNORECASE)
ID_STRUCTURE = re.compile(r"^IU(MH|UH)(\d{8})(\d{2})$")
ID_LOOSE = re.compile(r"^IU(MH|UH)\d*?(\d{2})$")

REDCAP_ID_PREFIX = "what is the name of the file"
INDUCTION_PREFIX = "what time was induction"
OR_ENTRY_PREFIX = "what time did the patient enter the or"
SURGERY_DATE_PREFIX = "date of surgery"

MAP_LIST_FALLBACKS = ("map_filepaths.csv", "btb_filepaths.csv",
                      "b2b_filepaths.csv", "b2b_filepath.csv",
                      "btb_filepath.csv")

# Column names that have meant cardiac output in these exports.
CO_PREFIXES = ("cardiac output", "cardiacoutput", "cardiac_output", "co (",
               "co_lmin", "cardiac index", "cardiacindex")

BAND_COLORS = ["#2f5597", "#548235", "#c55a11", "#7030a0", "#7f6000", "#495057"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    base = Path("/N/project/Analgesia_BDproject/PR/scripts_PR/10-8 Scatter")
    data = Path("/N/project/Analgesia_BDproject/PR/data")
    parser.add_argument("--filepaths", type=Path,
                        default=data / "sedline_filepaths.csv",
                        help="CSV listing one Sedline file path per line.")
    parser.add_argument("--sto2-filepaths", type=Path,
                        default=data / "sto2_filepaths.csv",
                        help="CSV listing one cerebral-StO2 file path per line.")
    parser.add_argument("--map-filepaths", type=Path,
                        default=data / "map_filepaths.csv",
                        help="CSV listing one beat-to-beat MAP file per line.")
    parser.add_argument("--hemosphere-filepaths", type=Path,
                        default=data / "hemosphere_filepaths.csv",
                        help="CSV listing one Hemosphere file per line. These "
                             "carry cardiac output. If the list is missing, "
                             "the PSi analysis still runs and only the CO "
                             "split is skipped.")
    parser.add_argument("--redcap", type=Path,
                        default=Path("/N/project/Analgesia_BDproject/data/"
                                     "00_raw/BDPostInductionHemod_DATA_LABELS_"
                                     "2026-09-30_1420.csv"),
                        help="REDCap labeled export with induction and OR entry.")
    parser.add_argument("--outdir", type=Path, default=base / "output",
                        help="Where the figures are written. No CSVs.")
    parser.add_argument("--min-minutes", type=float, default=-10.0,
                        help="Earliest minute relative to induction to use "
                             "(default -10).")
    parser.add_argument("--max-minutes", type=float, default=180.0,
                        help="Latest minute after induction to use (default "
                             "180). Autoregulation needs a spread of MAP "
                             "values, which takes most of a case to see.")
    parser.add_argument("--step-seconds", type=float, default=10.0,
                        help="Grid all signals are resampled onto (default 10 s).")
    parser.add_argument("--max-gap-seconds", type=float, default=60.0,
                        help="Never interpolate across a gap wider than this "
                             "(default 60 s), so a missing stretch stays "
                             "missing instead of being invented.")
    parser.add_argument("--psi-bin-width", type=float, default=25.0,
                        help="Width of the PSi bands (default 25, giving "
                             "0-25, 25-50, 50-75, 75-100).")
    parser.add_argument("--co-column", default=None,
                        help="Exact cardiac-output column name in the "
                             "Hemosphere files. Defaults to 'CO'.")
    parser.add_argument("--min-co-sqi", type=float, default=3.0,
                        help="Keep a cardiac-output reading only when its "
                             "CO_SQI signal-quality index is at least this "
                             "(default 3). Low-SQI readings are the device "
                             "saying it does not trust its own number.")
    parser.add_argument("--co-bins", type=int, default=3,
                        help="Number of cardiac-output bands, cut at quantiles "
                             "so each holds a similar number of patients "
                             "(default 3).")
    parser.add_argument("--min-cell-samples", type=int, default=60,
                        help="Samples a patient needs inside a band before "
                             "that patient gets their own slope there "
                             "(default 60 = 10 min at 10 s).")
    parser.add_argument("--min-map-range", type=float, default=10.0,
                        help="A patient's MAP must vary by at least this many "
                             "mmHg inside a band before fitting their slope "
                             "(default 10). A slope through a vertical stripe "
                             "of points is meaningless.")
    parser.add_argument("--plot-sample", type=int, default=20000,
                        help="Points drawn per panel (default 20000). All the "
                             "data is used for the statistics; this only keeps "
                             "the PNG a sane size.")
    parser.add_argument("--seed", type=int, default=20261006)
    return parser.parse_args()


# --------------------------------------------------------------------------- #
# Loading  (same machinery as the lag-scatter script, kept self-contained so
# this file can be copied to the cluster on its own)
# --------------------------------------------------------------------------- #

def read_table(path: str) -> tuple[pd.DataFrame, str]:
    """Read a monitor CSV whatever encoding it was written in.

    Monitor exports are not reliably UTF-8: the Hemosphere files are written by
    a Windows device and fail to decode as UTF-8 part-way through the first
    line. The byte-order mark settles UTF-16 and UTF-8-with-BOM outright;
    otherwise UTF-8 is tried first so nothing changes for files that were
    already fine, then cp1252, then latin-1, which cannot fail to decode. Only
    numbers and timestamps are read from these files, so a mangled character in
    a text column costs nothing.

    A file that comes back as a single column was split on the wrong character,
    so the separator is sniffed on a second pass.
    """
    with open(path, "rb") as handle:
        signature = handle.read(4)
    if signature[:2] in (b"\xff\xfe", b"\xfe\xff"):
        encodings = ["utf-16", "utf-16-le", "utf-16-be"]
    elif signature[:3] == b"\xef\xbb\xbf":
        encodings = ["utf-8-sig"]
    else:
        encodings = ["utf-8", "cp1252", "latin-1"]

    failure: Exception | None = None
    for encoding in encodings:
        try:
            frame = pd.read_csv(path, low_memory=False, skipinitialspace=True,
                                encoding=encoding)
            if frame.shape[1] <= 1:
                frame = pd.read_csv(path, sep=None, engine="python",
                                    skipinitialspace=True, encoding=encoding)
            return frame, encoding
        except Exception as exc:
            failure = exc
    raise failure if failure else RuntimeError(f"could not read {path}")


def read_filepath_list(master: Path) -> list[str]:
    paths: list[str] = []
    with open(master, "r", newline="") as handle:
        for raw in handle:
            first = raw.strip().replace("\t", ",").split(",")[0]
            first = first.strip().strip('"').strip("'")
            if not first:
                continue
            if not first.lower().endswith((".csv", ".txt")) and "/" not in first:
                continue
            paths.append(first)
    return paths


def patient_id_from_path(path: str) -> str | None:
    match = ID_PATTERN.search(path)
    return match.group(0).upper() if match else None


def find_column(frame: pd.DataFrame, prefix: str) -> str | None:
    prefix = prefix.strip().lower()
    for column in frame.columns:
        if str(column).strip().lower().startswith(prefix):
            return column
    return None


def repair_subject_ids(raw_ids: pd.Series,
                       surgery_dates: pd.Series) -> tuple[pd.Series, list[str]]:
    """Separators, then the hardcoded table, then the Date-of-Surgery check."""
    text = raw_ids.astype("string").str.strip().str.upper()
    cleaned = text.str.replace(r"[^A-Z0-9]", "", regex=True)
    notes: list[str] = []
    separator_fixes = int((text.notna() & text.ne(cleaned)).sum())

    resolved: list[str | None] = []
    pending: list[bool] = []
    hardcoded: list[str] = []
    for value, date in zip(cleaned, surgery_dates):
        if pd.isna(value) or not value:
            resolved.append(None); pending.append(False); continue
        stamp = None if pd.isna(date) else f"{date:%Y-%m-%d}"
        fix = REDCAP_ID_FIXES.get((value, stamp), REDCAP_ID_FIXES.get((value, None)))
        if fix is not None:
            resolved.append(fix); pending.append(False)
            hardcoded.append(f"{value} -> {fix}  (hardcoded)")
        else:
            resolved.append(value); pending.append(True)
    notes.extend(sorted(set(hardcoded)))
    if separator_fixes:
        notes.append(f"{separator_fixes} ID(s) had separators stripped")

    trusted = set()
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
            final.append(value); continue
        structured = ID_STRUCTURE.match(value)
        if pd.isna(date):
            final.append(value if structured else None); continue
        if structured:
            hospital, datepart, sequence = structured.groups()
            if datepart == f"{date:%Y%m%d}":
                final.append(value); continue
        else:
            loose = ID_LOOSE.match(value)
            if loose is None:
                final.append(None); continue
            hospital, sequence = loose.groups()
        candidate = f"IU{hospital}{date:%Y%m%d}{sequence}"
        if candidate in trusted:
            final.append(value if structured else None); continue
        final.append(candidate)
        automatic.append(f"{value} -> {candidate}  (Date of Surgery)")
    notes.extend(sorted(set(automatic)))
    return pd.Series(final, index=raw_ids.index, dtype="object"), notes


def find_record_key(redcap: pd.DataFrame) -> str | None:
    for column in redcap.columns[:3]:
        values = redcap[column]
        if values.notna().all() and 1 < values.nunique() < len(redcap):
            return column
    return None


def load_event_times(redcap_path: Path) -> tuple[pd.DataFrame, list[str]]:
    """Per-patient induction and OR-entry instants."""
    redcap = pd.read_csv(redcap_path, low_memory=False)

    id_column = find_column(redcap, REDCAP_ID_PREFIX)
    if id_column is None:
        best = 0
        for column in redcap.columns:
            hits = int(redcap[column].astype("string").str.strip()
                       .str.fullmatch(r"IU(?:MH|UH)\d+", case=False, na=False).sum())
            if hits > best:
                id_column, best = column, hits
    induction_column = find_column(redcap, INDUCTION_PREFIX)
    or_entry_column = find_column(redcap, OR_ENTRY_PREFIX)
    date_column = find_column(redcap, SURGERY_DATE_PREFIX)
    for name, column in (("research ID", id_column),
                         ("What time was INDUCTION...", induction_column),
                         ("What time did the patient enter the OR?", or_entry_column),
                         ("Date of Surgery", date_column)):
        if column is None:
            raise ValueError(f"REDCap has no '{name}' column.")

    notes: list[str] = []
    # REDCap splits a patient across instrument rows and only the owning row
    # carries each field, so fill the ID and date across each record block.
    record_key = find_record_key(redcap)
    if record_key is not None:
        for column in (id_column, date_column):
            before = int(redcap[column].notna().sum())
            key = redcap[record_key]
            redcap[column] = redcap.groupby(key, sort=False)[column].ffill()
            redcap[column] = redcap.groupby(key, sort=False)[column].bfill()
            gained = int(redcap[column].notna().sum()) - before
            if gained:
                notes.append(f"{gained} blank cell(s) filled in '{column[:40]}'")

    surgery_date = pd.to_datetime(redcap[date_column], errors="coerce")

    def resolve(values: pd.Series) -> pd.Series:
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

    subject_ids, id_notes = repair_subject_ids(redcap[id_column], surgery_date)
    notes.extend(id_notes)

    induction_at = resolve(redcap[induction_column])
    or_entry_at = resolve(redcap[or_entry_column])
    crossed = (induction_at.notna() & or_entry_at.notna()
               & ((or_entry_at - induction_at) > pd.Timedelta(hours=12)))
    induction_at.loc[crossed] = induction_at.loc[crossed] + pd.Timedelta(days=1)

    events = pd.DataFrame({"subject_id": subject_ids, "induction": induction_at,
                           "or_entry": or_entry_at})
    events = events.loc[events["subject_id"].notna() & events["subject_id"].ne("")]
    events = (events.sort_values("subject_id").groupby("subject_id", as_index=True)
              .agg(induction=("induction", "first"), or_entry=("or_entry", "first")))
    return events, notes


def anchor_clock_only(timestamp: pd.Series, induction: pd.Timestamp) -> pd.Series:
    """Re-date bare clock times onto the surgery day and unwrap midnight."""
    if timestamp.isna().all() or pd.isna(induction):
        return timestamp
    today = pd.Timestamp.today().normalize()
    if timestamp.dt.normalize().eq(today).mean() > 0.9:
        timestamp = induction.normalize() + (timestamp - today)
    wrapped = timestamp.diff() < pd.Timedelta(hours=-12)
    if wrapped.any():
        timestamp = timestamp + pd.to_timedelta(wrapped.cumsum(), unit="D")
    return timestamp


def load_sedline(path: str, induction: pd.Timestamp) -> pd.DataFrame | None:
    frame, _encoding = read_table(path)
    value_column = find_column(frame, "psi")
    date_column = find_column(frame, "date")
    time_column = find_column(frame, "time")
    if value_column is None:
        return None
    timestamp = pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns]")
    if date_column and time_column and date_column != time_column:
        timestamp = pd.to_datetime(
            frame[date_column].astype("string").str.strip() + " "
            + frame[time_column].astype("string").str.strip(),
            errors="coerce", format="mixed")
    if timestamp.isna().all():
        epoch_column = find_column(frame, "epoch")
        if epoch_column:
            timestamp = pd.to_datetime(
                pd.to_numeric(frame[epoch_column], errors="coerce"),
                unit="ms", errors="coerce")
    if timestamp.isna().all():
        return None
    # "-" is the monitor's missing-value mark; to_numeric makes it a real NaN.
    return pd.DataFrame({"timestamp": timestamp,
                         "psi": pd.to_numeric(frame[value_column], errors="coerce")})


def load_sto2(path: str, induction: pd.Timestamp) -> pd.DataFrame | None:
    frame, _encoding = read_table(path)
    time_column = find_column(frame, "time")
    if time_column is None:
        return None
    timestamp = pd.to_datetime(frame[time_column].astype("string").str.strip(),
                               errors="coerce", format="mixed")
    present, total, count = 0, pd.Series(0.0, index=frame.index), pd.Series(0, index=frame.index)
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
                         "sto2": (total / count).where(count > 0)})


def find_exact(frame: pd.DataFrame, *names: str) -> str | None:
    """Match a column by exact name, case- and whitespace-insensitive.

    The Hemosphere export has both 'Timestamp' and 'Time', and a prefix match
    on "time" picks up 'Timestamp' first -- so these columns must be matched
    exactly, not by prefix.
    """
    wanted = [name.strip().lower() for name in names]
    for column in frame.columns:
        if str(column).strip().lower() in wanted:
            return column
    return None


def load_hemosphere(path: str, induction: pd.Timestamp,
                    co_column: str | None,
                    min_sqi: float) -> tuple[pd.DataFrame | None, dict]:
    """Cardiac output from a Hemosphere export, filtered on its own SQI.

    Columns are Timestamp, Date, Time, Technology, CO, CO_SQI. CO_SQI is the
    monitor's confidence in its own reading, so anything below the threshold is
    dropped rather than averaged in -- a low-SQI CO is the device telling you
    not to believe it.
    """
    frame, encoding = read_table(path)
    frame.columns = [str(column).strip() for column in frame.columns]
    info: dict = {"columns": list(frame.columns), "rows": len(frame),
                  "encoding": encoding}

    value_column = (find_exact(frame, co_column) if co_column
                    else find_exact(frame, "co", "cardiac output", "co_lmin"))
    if value_column is None:
        return None, info
    info["co_column"] = value_column

    timestamp = pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns]")
    stamp_column = find_exact(frame, "timestamp")
    if stamp_column is not None:
        timestamp = pd.to_datetime(frame[stamp_column].astype("string").str.strip(),
                                   errors="coerce", format="mixed")
    date_column, time_column = find_exact(frame, "date"), find_exact(frame, "time")
    if timestamp.isna().all() and date_column and time_column:
        timestamp = pd.to_datetime(
            frame[date_column].astype("string").str.strip() + " "
            + frame[time_column].astype("string").str.strip(),
            errors="coerce", format="mixed")
    if timestamp.isna().all() and time_column:
        timestamp = pd.to_datetime(frame[time_column].astype("string").str.strip(),
                                   errors="coerce", format="mixed")
    if timestamp.isna().all():
        return None, info
    timestamp = anchor_clock_only(timestamp, induction)

    value = pd.to_numeric(frame[value_column], errors="coerce")
    info["co_present"] = int(value.notna().sum())

    sqi_column = find_exact(frame, "co_sqi", "cosqi", "co sqi")
    if sqi_column is None:
        info["sqi_column"] = None
    else:
        info["sqi_column"] = sqi_column
        sqi = pd.to_numeric(frame[sqi_column], errors="coerce")
        kept = sqi >= min_sqi
        info["co_kept"] = int((value.notna() & kept).sum())
        # A missing SQI is not evidence of a good reading, so it is dropped too.
        value = value.where(kept)

    technology = find_exact(frame, "technology")
    if technology is not None:
        info["technology"] = sorted(
            frame[technology].dropna().astype(str).str.strip().unique().tolist())[:6]
    return pd.DataFrame({"timestamp": timestamp, "co": value}), info


def load_map(path: str, induction: pd.Timestamp) -> pd.DataFrame | None:
    """Beat-to-beat MAP, with the monitor's own bad-data rows dropped."""
    frame, _encoding = read_table(path)
    frame.columns = [str(column).strip() for column in frame.columns]
    value_column = find_column(frame, "meanarterial") or find_column(frame, "map")
    time_column = find_exact(frame, "time") or find_column(frame, "time")
    if value_column is None or time_column is None:
        return None

    keep = pd.Series(True, index=frame.index)
    bad_column = find_column(frame, "databad")
    if bad_column is not None:
        keep &= pd.to_numeric(frame[bad_column], errors="coerce").ne(1)

    timestamp = pd.to_datetime(frame[time_column].astype("string").str.strip(),
                               errors="coerce", format="mixed")
    timestamp = anchor_clock_only(timestamp, induction)
    return pd.DataFrame({
        "timestamp": timestamp,
        "map": pd.to_numeric(frame[value_column], errors="coerce"),
    }).loc[keep]


def resolve_map_list(explicit: Path | None, data_dir: Path) -> Path | None:
    if explicit is not None:
        return explicit if explicit.is_file() else None
    for name in MAP_LIST_FALLBACKS:
        candidate = data_dir / name
        if candidate.is_file():
            return candidate
    return None


# --------------------------------------------------------------------------- #
# Put every signal on one clock, per patient
# --------------------------------------------------------------------------- #

def to_grid(frame: pd.DataFrame, column: str, grid: np.ndarray,
            max_gap_minutes: float) -> np.ndarray:
    """Resample one signal onto the shared grid, refusing to bridge long gaps.

    Linear interpolation will happily draw a straight line across a ten-minute
    hole in the record, inventing data that reads as real. Any grid point whose
    nearest genuine sample is further away than max_gap is set to NaN instead.
    """
    usable = frame[["minutes", column]].dropna().sort_values("minutes")
    if len(usable) < 2:
        return np.full(grid.shape, np.nan)
    times = usable["minutes"].to_numpy(float)
    values = np.interp(grid, times, usable[column].to_numpy(float),
                       left=np.nan, right=np.nan)
    nearest = np.abs(grid[:, None] - times[None, :]).min(axis=1) \
        if len(times) <= 4000 else None
    if nearest is None:
        # Large records: searchsorted gives the same answer without the matrix.
        index = np.clip(np.searchsorted(times, grid), 1, len(times) - 1)
        nearest = np.minimum(np.abs(grid - times[index - 1]),
                             np.abs(grid - times[index]))
    values[nearest > max_gap_minutes] = np.nan
    return values


def build_aligned(args: argparse.Namespace, events: pd.DataFrame,
                  map_list: Path,
                  hemo_list: Path | None) -> tuple[pd.DataFrame, dict, list[str]]:
    """One long table: subject_id, minutes, map, sto2, psi and maybe co.

    PSi, StO2 and MAP are required -- a patient missing any of them cannot
    contribute to the MAP-StO2 slope at all. Cardiac output is optional and
    merged in only where a Hemosphere file exists, so the PSi analysis keeps
    its full cohort and only the CO split runs on the smaller subset.
    """
    step = float(args.step_seconds) / 60.0
    grid = np.arange(args.min_minutes, args.max_minutes + step / 2.0, step)
    max_gap = float(args.max_gap_seconds) / 60.0

    def index_by_id(filepaths: Path) -> dict[str, str]:
        found: dict[str, str] = {}
        for path in read_filepath_list(filepaths):
            subject_id = patient_id_from_path(path)
            if subject_id and Path(path).is_file():
                found[subject_id] = path
        return found

    psi_files = index_by_id(args.filepaths)
    sto2_files = index_by_id(args.sto2_filepaths)
    map_files = index_by_id(map_list)
    hemo_files = index_by_id(hemo_list) if hemo_list is not None else {}
    shared = sorted(set(psi_files) & set(sto2_files) & set(map_files))

    chunks: list[pd.DataFrame] = []
    skipped: list[str] = []
    hemo_report: dict = {"listed": len(hemo_files), "matched": 0, "used": 0,
                         "columns": [], "co_column": None, "sqi_column": None,
                         "technology": [], "rows_seen": 0, "rows_kept": 0,
                         "errors": [], "encodings": [], "no_co": []}

    for subject_id in shared:
        if subject_id not in events.index:
            skipped.append(f"{subject_id}: not in the REDCap export")
            continue
        induction = events.loc[subject_id, "induction"]
        if pd.isna(induction):
            skipped.append(f"{subject_id}: no REDCap induction time")
            continue

        try:
            psi_frame = load_sedline(psi_files[subject_id], induction)
            sto2_frame = load_sto2(sto2_files[subject_id], induction)
            map_frame = load_map(map_files[subject_id], induction)
        except Exception as exc:
            skipped.append(f"{subject_id}: read error: {exc}")
            continue
        if psi_frame is None or sto2_frame is None or map_frame is None:
            skipped.append(f"{subject_id}: a signal had no readable values")
            continue

        sources = [(psi_frame, "psi"), (sto2_frame, "sto2"), (map_frame, "map")]

        if subject_id in hemo_files:
            hemo_report["matched"] += 1
            try:
                hemo_frame, info = load_hemosphere(
                    hemo_files[subject_id], induction, args.co_column,
                    args.min_co_sqi)
            except Exception as exc:
                hemo_frame, info = None, {}
                # A Hemosphere failure costs this patient their CO only; they
                # still have MAP, StO2 and PSi, so it must not be filed as an
                # alignment failure or the report contradicts itself.
                hemo_report["errors"].append(f"{subject_id}: {exc}")
            if info:
                encoding = info.get("encoding")
                if encoding and encoding not in hemo_report["encodings"]:
                    hemo_report["encodings"].append(encoding)
                hemo_report["columns"] = hemo_report["columns"] or info.get("columns", [])
                hemo_report["co_column"] = hemo_report["co_column"] or info.get("co_column")
                hemo_report["sqi_column"] = hemo_report["sqi_column"] or info.get("sqi_column")
                hemo_report["rows_seen"] += info.get("co_present", 0)
                hemo_report["rows_kept"] += info.get("co_kept", info.get("co_present", 0))
                for name in info.get("technology", []):
                    if name not in hemo_report["technology"]:
                        hemo_report["technology"].append(name)
            if hemo_frame is not None and hemo_frame["co"].notna().any():
                hemo_report["used"] += 1
                sources.append((hemo_frame, "co"))
            elif hemo_frame is not None:
                hemo_report["no_co"].append(subject_id)

        aligned = {"minutes": grid}
        for frame, column in sources:
            frame = frame.dropna(subset=["timestamp"]).copy()
            frame["minutes"] = (frame["timestamp"] - induction).dt.total_seconds() / 60.0
            low, high = VALID_RANGE[column]
            frame.loc[~frame[column].between(low, high), column] = np.nan
            aligned[column] = to_grid(frame, column, grid, max_gap)

        patient = pd.DataFrame(aligned).dropna(subset=["map", "sto2", "psi"])
        if patient.empty:
            skipped.append(f"{subject_id}: no overlapping MAP/StO2/PSi samples")
            continue
        patient.insert(0, "subject_id", subject_id)
        chunks.append(patient)

    combined = (pd.concat(chunks, ignore_index=True) if chunks
                else pd.DataFrame(columns=["subject_id", "minutes", "map",
                                           "sto2", "psi"]))
    return combined, hemo_report, skipped


# --------------------------------------------------------------------------- #
# The two slopes
# --------------------------------------------------------------------------- #

def pooled_slope(frame: pd.DataFrame) -> dict:
    """One line through every sample in the band -- the figure that was asked for.

    Reported, but not trusted: a patient contributes thousands of correlated
    samples, so this is pseudo-replicated and its spread is not interpretable.
    """
    data = frame[["map", "sto2"]].dropna()
    if len(data) < 10:
        return {}
    slope, intercept = np.polyfit(data["map"], data["sto2"], 1)
    return {"slope": float(slope), "intercept": float(intercept),
            "n_samples": len(data),
            "r": float(data["map"].corr(data["sto2"]))}


def within_patient_slopes(frame: pd.DataFrame, args: argparse.Namespace) -> dict:
    """Each patient's own MAP-to-StO2 slope inside this band, then the median.

    This is the autoregulation estimate. A patient only contributes if they have
    enough samples AND enough MAP movement in the band -- fitting a line through
    a near-vertical stripe of points produces a huge meaningless slope.
    """
    per_patient = []
    for subject_id, group in frame.groupby("subject_id"):
        data = group[["map", "sto2"]].dropna()
        if len(data) < args.min_cell_samples:
            continue
        spread = float(data["map"].max() - data["map"].min())
        if spread < args.min_map_range:
            continue
        slope, _intercept = np.polyfit(data["map"], data["sto2"], 1)
        per_patient.append({"subject_id": subject_id, "slope": float(slope),
                            "r": float(data["map"].corr(data["sto2"])),
                            "n": len(data), "map_range": spread})
    if len(per_patient) < 5:
        return {"n_patients": len(per_patient)}

    table = pd.DataFrame(per_patient)
    values = table["slope"].to_numpy(float)
    rng = np.random.default_rng(args.seed)
    draws = rng.choice(values, size=(2000, len(values)), replace=True)
    low, high = np.percentile(np.median(draws, axis=1), [2.5, 97.5])
    return {"n_patients": len(table), "median": float(np.median(values)),
            "ci": (float(low), float(high)),
            "median_r": float(table["r"].median()), "table": table}


def compare_bands(bands: list[dict], key: str, args: argparse.Namespace) -> dict | None:
    """Is the slope in the deepest band different from the lightest?

    Paired when it can be. A patient moves through several PSi bands during one
    case, so each serves as their own control and the comparison is clean.
    Cardiac output does not work that way -- it is near enough constant within a
    patient, so each patient sits in ONE band and there is nothing to pair.
    There the bands are compared as independent groups instead, which is a
    weaker design because the bands then differ in WHO is in them as well as in
    CO. The result records which comparison was used so the write-up can say so.
    """
    usable = [band for band in bands if "table" in band.get(key, {})]
    if len(usable) < 2:
        return None
    first, last = usable[0][key]["table"], usable[-1][key]["table"]
    labels = (usable[0]["label"], usable[-1]["label"])
    rng = np.random.default_rng(args.seed)

    merged = first.merge(last, on="subject_id", suffixes=("_a", "_b"))
    if len(merged) >= 5:
        difference = (merged["slope_b"] - merged["slope_a"]).to_numpy(float)
        draws = rng.choice(difference, size=(2000, len(difference)), replace=True)
        low, high = np.percentile(np.median(draws, axis=1), [2.5, 97.5])
        result = {"design": "paired", "n": len(merged), "labels": labels,
                  "difference": float(np.median(difference)),
                  "ci": (float(low), float(high))}
        try:
            from scipy.stats import wilcoxon
            result["p"] = float(wilcoxon(difference).pvalue)
        except Exception:
            pass
        return result

    a, b = first["slope"].to_numpy(float), last["slope"].to_numpy(float)
    if len(a) < 5 or len(b) < 5:
        return {"design": "none", "labels": labels, "n_paired": len(merged),
                "n_a": len(a), "n_b": len(b)}
    draws = (rng.choice(b, size=(2000, len(b)), replace=True).mean(axis=1)
             - rng.choice(a, size=(2000, len(a)), replace=True).mean(axis=1))
    low, high = np.percentile(draws, [2.5, 97.5])
    result = {"design": "unpaired", "labels": labels, "n_paired": len(merged),
              "n_a": len(a), "n_b": len(b),
              "difference": float(np.median(b) - np.median(a)),
              "ci": (float(low), float(high))}
    try:
        from scipy.stats import mannwhitneyu
        result["p"] = float(mannwhitneyu(b, a).pvalue)
    except Exception:
        pass
    return result


# --------------------------------------------------------------------------- #
# Figure
# --------------------------------------------------------------------------- #

def make_stratified_figure(bands: list[dict], variable: str, unit: str,
                           args: argparse.Namespace, output_path: Path) -> None:
    columns = max(len(bands), 1)
    figure = plt.figure(figsize=(4.4 * columns, 9.6))
    spec = figure.add_gridspec(2, columns, height_ratios=[1.35, 1.0],
                               hspace=0.42, wspace=0.28)
    rng = np.random.default_rng(args.seed)

    sto2_low = min((band["frame"]["sto2"].quantile(0.01) for band in bands
                    if len(band["frame"])), default=0.0)
    sto2_high = max((band["frame"]["sto2"].quantile(0.99) for band in bands
                     if len(band["frame"])), default=100.0)
    map_low = min((band["frame"]["map"].quantile(0.01) for band in bands
                   if len(band["frame"])), default=40.0)
    map_high = max((band["frame"]["map"].quantile(0.99) for band in bands
                    if len(band["frame"])), default=120.0)

    for index, band in enumerate(bands):
        axis = figure.add_subplot(spec[0, index])
        data = band["frame"]
        color = BAND_COLORS[index % len(BAND_COLORS)]
        if len(data) > args.plot_sample:
            shown = data.iloc[rng.choice(len(data), args.plot_sample, replace=False)]
        else:
            shown = data
        axis.scatter(shown["map"], shown["sto2"], s=3, alpha=0.10,
                     color=color, edgecolors="none", rasterized=True)

        pooled = band.get("pooled", {})
        if "slope" in pooled:
            span = np.linspace(map_low, map_high, 50)
            axis.plot(span, pooled["intercept"] + pooled["slope"] * span,
                      color="#d1495b", lw=2.4,
                      label=f"pooled slope {pooled['slope']:+.3f}")
        within = band.get("within", {})
        if "median" in within:
            # Anchor the within-patient line at the band's centre so the two
            # slopes can be compared by eye without an arbitrary intercept.
            centre_x = float(data["map"].median())
            centre_y = float(data["sto2"].median())
            span = np.linspace(map_low, map_high, 50)
            axis.plot(span, centre_y + within["median"] * (span - centre_x),
                      color="black", lw=2.2, ls="--",
                      label=f"within-patient {within['median']:+.3f}")

        axis.set_xlim(map_low, map_high)
        axis.set_ylim(sto2_low, sto2_high)
        axis.set_xlabel("MAP (mmHg)")
        if index == 0:
            axis.set_ylabel("Cerebral StO2 (%)")
        axis.set_title(
            f"{variable} {band['label']}{unit}\n"
            f"{len(data):,} samples, {data['subject_id'].nunique()} patients",
            fontsize=10.5)
        axis.grid(True, color="#e8e8e8", lw=0.6)
        axis.set_axisbelow(True)
        axis.legend(loc="upper left", fontsize=8, framealpha=0.9)

    summary = figure.add_subplot(spec[1, :])
    positions = np.arange(len(bands))
    pooled_values = [band.get("pooled", {}).get("slope", np.nan) for band in bands]
    summary.plot(positions, pooled_values, color="#d1495b", lw=2.2, marker="o",
                 markersize=9, label="pooled slope (every sample — "
                                     "pseudo-replicated, do not quote)")
    for index, band in enumerate(bands):
        within = band.get("within", {})
        if "median" not in within:
            continue
        summary.plot([index, index], within["ci"], color="black", lw=3,
                     solid_capstyle="round", zorder=3)
        summary.scatter([index], [within["median"]], s=95, color="black",
                        zorder=4,
                        label=("within-patient median with 95% CI "
                               "(this is the autoregulation estimate)"
                               if index == 0 else None))
    summary.axhline(0, color="#777777", ls="--", lw=1.4)
    summary.set_xticks(positions)
    summary.set_xticklabels([f"{variable}\n{band['label']}{unit}\n"
                             f"{band.get('within', {}).get('n_patients', 0)} pts"
                             for band in bands])
    summary.set_ylabel("Slope of StO2 on MAP\n(% StO2 per mmHg)")
    summary.set_title(
        "Slope by band — flat (near 0) = autoregulation intact; "
        "rising = StO2 follows MAP, so the brain is pressure-passive",
        fontsize=11)
    summary.grid(True, axis="y", color="#e8e8e8", lw=0.6)
    summary.set_axisbelow(True)
    summary.legend(loc="best", fontsize=9, framealpha=0.95)

    figure.suptitle(
        f"Cerebral StO2 versus MAP, split by {variable}\n"
        f"does the brain's protection against low blood pressure depend on "
        f"{variable}?",
        fontsize=14)
    figure.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(figure)


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #

def banner(title: str) -> None:
    print("\n" + title)
    print("-" * max(len(title), 64))


def run_stratification(aligned: pd.DataFrame, column: str, edges: np.ndarray,
                       variable: str, unit: str, args: argparse.Namespace,
                       output_path: Path) -> None:
    bands = []
    for low, high in zip(edges[:-1], edges[1:]):
        # Half-open bins, with the top band closed so the maximum is not lost.
        inside = (aligned[column] >= low) & (
            aligned[column] <= high if high == edges[-1] else aligned[column] < high)
        frame = aligned.loc[inside]
        band = {"label": f"{low:g}-{high:g}", "frame": frame}
        if len(frame):
            band["pooled"] = pooled_slope(frame)
            band["within"] = within_patient_slopes(frame, args)
        bands.append(band)

    banner(f"{variable} bands — slope of StO2 on MAP")
    print("  Flat (near 0) = the brain holds its oxygen steady while pressure "
          "moves: autoregulation intact.")
    print("  Rising        = StO2 follows MAP: the brain is pressure-passive "
          "and hypotension bites.\n")
    header = (f"  {'band':<12}{'samples':>10}{'pts':>6}{'pooled':>10}"
              f"{'within-patient (95% CI)':>30}{'median r':>10}")
    print(header)
    print("  " + "-" * (len(header) - 2))
    for band in bands:
        pooled = band.get("pooled", {})
        within = band.get("within", {})
        pooled_text = f"{pooled['slope']:+.4f}" if "slope" in pooled else "-"
        if "median" in within:
            within_text = (f"{within['median']:+.4f} "
                           f"({within['ci'][0]:+.4f} to {within['ci'][1]:+.4f})")
            r_text = f"{within['median_r']:+.2f}"
        else:
            within_text = f"too few patients ({within.get('n_patients', 0)})"
            r_text = "-"
        print(f"  {band['label']:<12}{len(band['frame']):>10,}"
              f"{band['frame']['subject_id'].nunique():>6}"
              f"{pooled_text:>10}{within_text:>30}{r_text:>10}")

    disagreements = [
        band for band in bands
        if "slope" in band.get("pooled", {}) and "median" in band.get("within", {})
        and (band["pooled"]["slope"] > 0) != (band["within"]["median"] > 0)
    ]
    if disagreements:
        print(f"\n  !! In {len(disagreements)} band(s) the pooled and "
              f"within-patient slopes have OPPOSITE signs. That is Simpson's "
              f"paradox: the pooled line is being driven by the fact that "
              f"different patients sit at different MAP and StO2 levels, not "
              f"by what happens inside a patient. Quote the within-patient "
              f"number.")

    comparison = compare_bands(bands, "within", args)
    if comparison and "difference" not in comparison:
        first, last = comparison["labels"]
        banner(f"Does the slope differ between {variable} {first} and {last}?")
        print(f"  Cannot be tested: only {comparison['n_paired']} patient(s) "
              f"appear in both bands, and the bands hold "
              f"{comparison['n_a']} and {comparison['n_b']} patients "
              f"separately — too few either way.")
    elif comparison:
        first, last = comparison["labels"]
        banner(f"Does the slope differ between {variable} {first} and {last}?")
        if comparison["design"] == "paired":
            print(f"  Compared WITHIN the {comparison['n']} patients who have "
                  f"data in both bands, so each patient is their own control "
                  f"and this is not about which patients reach each band.")
        else:
            print(f"  Only {comparison['n_paired']} patient(s) appear in both "
                  f"bands, so this is an UNPAIRED comparison of "
                  f"{comparison['n_a']} versus {comparison['n_b']} patients. "
                  f"Weaker: the bands differ in who is in them as well as in "
                  f"{variable}, so a difference here could be the patients "
                  f"rather than the {variable}.")
        print(f"  slope({last}) - slope({first}) = "
              f"{comparison['difference']:+.4f} "
              f"(95% CI {comparison['ci'][0]:+.4f} to "
              f"{comparison['ci'][1]:+.4f})"
              + (f", {'Wilcoxon' if comparison['design'] == 'paired' else 'Mann-Whitney'}"
                 f" p={comparison['p']:.3g}" if "p" in comparison else ""))
        count = comparison.get("n", min(comparison.get("n_a", 0),
                                        comparison.get("n_b", 0)))
        thin = count < 10
        # A signed-rank test on n patients cannot return a p below 2/2**n, so
        # at n=5 the smallest achievable p is 0.0625 and significance is out of
        # reach no matter how large the effect. Saying "DOES modify" off such a
        # comparison would be claiming more than the data can carry.
        floor = 2.0 / (2.0 ** count) if 0 < count < 12 else 0.0
        if comparison["ci"][0] <= 0.0 <= comparison["ci"][1]:
            print(f"  The interval includes zero, so there is NO evidence here "
                  f"that {variable} changes the MAP-StO2 relationship. That is "
                  f"a real answer: it argues against building a model around "
                  f"this interaction.")
        elif thin:
            direction = "steeper" if comparison["difference"] > 0 else "flatter"
            print(f"  !! The interval excludes zero (slope {direction} in the "
                  f"{last} band), but this rests on only {count} patient(s) and "
                  f"is NOT reliable.")
            if floor and "p" in comparison:
                print(f"     With {count} patients the smallest p the test can "
                      f"return is {floor:.4f}, so it could never reach 0.05 "
                      f"however large the effect. The reported p="
                      f"{comparison['p']:.3g} is at or near that floor.")
            print(f"     Treat this as a hint to chase, not a finding. The "
                  f"usual cause is that few patients spend enough time in BOTH "
                  f"bands; widening --min-minutes to take in more "
                  f"pre-induction record is what adds patients to the light "
                  f"band.")
        else:
            direction = "steeper" if comparison["difference"] > 0 else "flatter"
            print(f"  The interval excludes zero: the slope is {direction} in "
                  f"the {last} band. {variable} DOES modify the MAP-StO2 "
                  f"relationship, which is the interaction worth modelling.")
            if "p" in comparison and comparison["p"] > 0.05:
                print(f"     Note: the interval and the p-value "
                      f"({comparison['p']:.3g}) disagree. The weaker of the "
                      f"two is the honest reading — call it suggestive.")

    make_stratified_figure(bands, variable, unit, args, output_path)
    print(f"\n  Figure: {output_path}")


def main() -> int:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    for path in (args.filepaths, args.sto2_filepaths, args.redcap):
        if not path.is_file():
            sys.stderr.write(f"Input file does not exist: {path}\n")
            return 1
    map_list = resolve_map_list(args.map_filepaths, args.filepaths.parent)
    if map_list is None:
        sys.stderr.write(
            "No beat-to-beat MAP list found. This analysis is built on MAP, so "
            "it cannot run without one.\nPass --map-filepaths <file>. Looked "
            f"in {args.filepaths.parent} for: {', '.join(MAP_LIST_FALLBACKS)}\n")
        return 1
    hemo_list = (args.hemosphere_filepaths
                 if args.hemosphere_filepaths.is_file() else None)

    events, notes = load_event_times(args.redcap)
    if notes:
        banner("REDCap repairs")
        for note in notes:
            print(f"  {note}")
    print(f"\nREDCap: {len(events)} patients with an induction time for "
          f"{int(events['induction'].notna().sum())}")
    print(f"Window: {args.min_minutes:g} to {args.max_minutes:g} minutes from "
          f"induction, resampled every {args.step_seconds:g} s, never "
          f"interpolating across gaps wider than {args.max_gap_seconds:g} s.")

    aligned, hemo, skipped = build_aligned(args, events, map_list, hemo_list)
    if aligned.empty:
        sys.stderr.write("\nNo patient had MAP, StO2 and PSi overlapping.\n")
        for line in skipped[:20]:
            sys.stderr.write(f"  {line}\n")
        return 1

    banner("Aligned data")
    print(f"  {aligned['subject_id'].nunique()} patients with MAP, StO2 and "
          f"PSi on the same clock")
    print(f"  {len(aligned):,} time points "
          f"(median {len(aligned) // max(aligned['subject_id'].nunique(), 1):,} "
          f"per patient)")
    print(f"  MAP  median {aligned['map'].median():.0f} mmHg "
          f"(IQR {aligned['map'].quantile(.25):.0f}-{aligned['map'].quantile(.75):.0f})")
    print(f"  StO2 median {aligned['sto2'].median():.0f} % "
          f"(IQR {aligned['sto2'].quantile(.25):.0f}-{aligned['sto2'].quantile(.75):.0f})")
    print(f"  PSi  median {aligned['psi'].median():.0f} "
          f"(IQR {aligned['psi'].quantile(.25):.0f}-{aligned['psi'].quantile(.75):.0f})")
    if skipped:
        print(f"\n  {len(skipped)} patient(s) could not be aligned:")
        for line in skipped:
            print(f"    {line}")

    # ---- action item 1: PSi bands ------------------------------------------
    width = float(args.psi_bin_width)
    edges = np.arange(0.0, 100.0 + width / 2.0, width)
    run_stratification(aligned, "psi", edges, "PSi", "",
                       args, args.outdir / "map_sto2_by_psi.png")

    # ---- action item 2: cardiac-output bands -------------------------------
    banner("Cardiac output (Hemosphere)")
    if hemo_list is None:
        print(f"  No Hemosphere list at {args.hemosphere_filepaths}, so the "
              f"CO split is skipped. Pass --hemosphere-filepaths <file>.")
    elif "co" not in aligned or aligned["co"].notna().sum() == 0:
        print(f"  {hemo['listed']} Hemosphere file(s) listed, "
              f"{hemo['matched']} matched a patient who also has MAP, StO2 and "
              f"PSi, but no usable CO came out of them.")
        if hemo["errors"]:
            print(f"\n  {len(hemo['errors'])} file(s) could not be read at all. "
                  f"First few:")
            for line in hemo["errors"][:5]:
                print(f"    {line}")
        if hemo["columns"]:
            print("\n  Columns in the first Hemosphere file read:")
            for column in hemo["columns"]:
                print(f"    {column}")
        print(f"\n  CO column found: {hemo['co_column'] or 'NONE'}; "
              f"SQI column found: {hemo['sqi_column'] or 'NONE'}")
        print("  If the CO column is named differently, re-run with "
              "--co-column \"<exact name>\".")
    else:
        usable = aligned.dropna(subset=["co"])
        print(f"  {hemo['listed']} Hemosphere file(s) listed; "
              f"{hemo['matched']} matched a patient with MAP, StO2 and PSi; "
              f"{hemo['used']} produced usable CO.")
        print(f"  CO column '{hemo['co_column']}', quality column "
              f"'{hemo['sqi_column'] or 'NONE'}', "
              f"encoding {', '.join(hemo['encodings']) or 'unknown'}")
        if hemo["errors"]:
            print(f"  {len(hemo['errors'])} file(s) could not be read; first: "
                  f"{hemo['errors'][0]}")
        if hemo["no_co"]:
            print(f"  {len(hemo['no_co'])} file(s) read but held no CO passing "
                  f"the quality filter: {', '.join(hemo['no_co'][:8])}"
                  + (" ..." if len(hemo["no_co"]) > 8 else ""))
        if hemo["sqi_column"] is None:
            print(f"  !! No CO_SQI column was found, so the "
                  f"--min-co-sqi {args.min_co_sqi:g} filter could NOT be "
                  f"applied and every CO reading is being used. Check the "
                  f"column name before trusting the CO split.")
        else:
            dropped = hemo["rows_seen"] - hemo["rows_kept"]
            share = 100.0 * dropped / hemo["rows_seen"] if hemo["rows_seen"] else 0.0
            print(f"  SQI filter >= {args.min_co_sqi:g}: kept "
                  f"{hemo['rows_kept']:,} of {hemo['rows_seen']:,} CO readings "
                  f"({share:.0f}% dropped as low quality)")
        if hemo["technology"]:
            print(f"  Technology values seen: {', '.join(hemo['technology'])}")
        print(f"  After aligning: {usable['subject_id'].nunique()} patients, "
              f"{len(usable):,} time points")
        print(f"  CO median {usable['co'].median():.2f} L/min "
              f"(IQR {usable['co'].quantile(.25):.2f}-"
              f"{usable['co'].quantile(.75):.2f})")
        # Quantile edges so each band holds a comparable amount of data rather
        # than whatever an arbitrary round number happens to catch.
        quantiles = np.linspace(0, 1, args.co_bins + 1)
        edges = np.unique(np.round(usable["co"].quantile(quantiles).to_numpy(), 2))
        if len(edges) < 3:
            print("  Cardiac output barely varies; the split would be "
                  "meaningless, so it is skipped.")
        else:
            print(f"  Band edges (quantiles): "
                  f"{', '.join(f'{edge:g}' for edge in edges)}")
            run_stratification(usable, "co", edges, "CO", " L/min",
                               args, args.outdir / "map_sto2_by_co.png")

    banner("What these figures are for")
    print("  Flat slopes in every band  -> the brain is protected regardless "
          "of depth or output; no interaction to model.")
    print("  Slope steepens as PSi falls -> deep anesthesia weakens "
          "autoregulation, so deep AND hypotensive is the dangerous "
          "combination. That is the clinically actionable result.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
