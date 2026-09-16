#!/usr/bin/env python3
"""Raw scatter of a Sedline signal against time from induction, with a lag readout.

Every valid 2-second Sedline sample from every patient is drawn as one dot, on a
shared x-axis where TIME = 0 is anesthesia induction. Nothing is averaged,
binned or smoothed before plotting -- the dots are the raw file values.

On top of the dots an optional trend curve is drawn. It is fitted SEPARATELY
before and after induction so that no smoothing bleeds across TIME = 0, and its
only purpose is to read off the LAG: how long after the induction drug is pushed
the signal actually starts to fall, and how long it takes to reach its nadir.
Those numbers are printed to the terminal.

Data sources
------------
  * Sedline raw-cleaned files, one per patient, listed one path per line in
    --filepaths. Columns used (matched by prefix, so stray spaces are fine):
        Date, Time            -> timestamp   (Epoch Time used as a fallback)
        PSi (Sedline) Value   -> the signal  (or SEFL / SEFR / SR / EMG)
        ARTF % (Sedline) Value-> optional artifact filter
  * REDCap labeled export (--redcap) supplying, per patient:
        "What is the name of the file ([IU][HOSPITAL...]" -> research ID
        "Date of Surgery"                                 -> used to repair IDs
        "What time was INDUCTION (when primary induction med pushed) (TIME = 0)?"
        "What time did the patient enter the OR?"

Research-ID repair
------------------
  The research ID encodes the surgery date: IU + MH/UH + YYYYMMDD + patient of
  the day. Where the ID typed into REDCap disagrees with that row's Date of
  Surgery, the date wins and the ID is rebuilt, because the typo is in the ID
  field. This matters: IUMH2026010601 is BOTH a real patient (surgery 2026-01-06)
  and the typo sitting on IUMH2026010501 (surgery 2026-01-05), so a blanket
  find-and-replace would destroy the real one. Every repair is printed.
  Turn it off with --no-id-repair.

Window
------
  Each patient contributes samples from the first reading at or after OR entry
  through --max-minutes after induction, and the plot is clipped at
  --min-minutes on the left. x is minutes from induction (negative before it).

Validity
--------
  Sedline writes "-" when a value is unavailable; those rows are dropped, as are
  values outside the physiological range for the chosen signal. Pass --max-artf
  to additionally drop high-artifact samples.

Output
------
  One PNG. Everything else -- per-patient windows, exclusions, ID repairs, the
  lag readout and the summary statistics -- is printed to the terminal.

Usage
-----
    python psi_vs_time_regression.py
    python psi_vs_time_regression.py --signal SEFL --max-minutes 30
    python psi_vs_time_regression.py --fit none          # raw dots only
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
# Signals this script can plot. Adding a row here is all it takes to run the
# same figure for the next variable in the series.
# --------------------------------------------------------------------------- #
SIGNALS = {
    "PSi": {
        "column_prefix": "psi",
        "label": "PSi (Sedline)",
        "valid_range": (0.0, 100.0),
    },
    "SEFL": {
        "column_prefix": "sefl",
        "label": "SEF Left (Hz)",
        "valid_range": (0.0, 30.0),
    },
    "SEFR": {
        "column_prefix": "sefr",
        "label": "SEF Right (Hz)",
        "valid_range": (0.0, 30.0),
    },
    "SR": {
        "column_prefix": "sr %",
        "label": "Suppression Ratio (%)",
        "valid_range": (0.0, 100.0),
    },
    "EMG": {
        "column_prefix": "emg",
        "label": "EMG (%)",
        "valid_range": (0.0, 100.0),
    },
}

ARTF_PREFIX = "artf"
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
    parser.add_argument(
        "--filepaths", type=Path,
        default=Path("/N/project/Analgesia_BDproject/PR/scripts_PR/"
                     "9-16 Regression/sedline_filepaths.csv"),
        help="CSV listing one Sedline file path per line.")
    parser.add_argument(
        "--redcap", type=Path,
        default=Path("/N/project/Analgesia_BDproject/data/00_raw/"
                     "BDPostInductionHemod_DATA_LABELS_2026-08-21_1723.csv"),
        help="REDCap labeled export holding induction and OR-entry times.")
    parser.add_argument(
        "--outdir", type=Path,
        default=Path("/N/project/Analgesia_BDproject/PR/scripts_PR/"
                     "9-16 Regression/output"),
        help="Where the figure is written. No CSVs are produced.")
    parser.add_argument(
        "--signal", choices=sorted(SIGNALS), default="PSi",
        help="Which Sedline signal to put on the y-axis (default PSi).")
    parser.add_argument(
        "--max-minutes", type=float, default=20.0,
        help="Last minute after induction to plot (default 20).")
    parser.add_argument(
        "--min-minutes", type=float, default=-30.0,
        help="Earliest minute before induction to plot (default -30). Samples "
             "start at each patient's OR entry, so this only clips patients "
             "with unusually long pre-induction OR time.")
    parser.add_argument(
        "--max-artf", type=float, default=None,
        help="Optional: drop samples whose ARTF %% exceeds this. Off by "
             "default; the run reports what a threshold of 20 would cost.")
    parser.add_argument(
        "--fit", choices=["lowess", "linear", "spline", "none"],
        default="lowess",
        help="Trend curve drawn over the raw dots, fitted separately either "
             "side of induction. 'lowess' (default) is a local curve, which is "
             "what the lag readout needs; 'linear' is a straight least-squares "
             "line per side; 'none' plots the raw dots only. The dots "
             "themselves are always every raw valid sample.")
    parser.add_argument(
        "--lowess-frac", type=float, default=0.05,
        help="LOWESS smoothing span as a fraction of the segment (default "
             "0.05). Larger = smoother.")
    parser.add_argument(
        "--n-splines", type=int, default=80,
        help="Basis size when --fit spline is used.")
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
        help="Do not use Date of Surgery to repair mistyped REDCap research "
             "IDs (see the module docstring).")
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
    side instead, using that row's Date of Surgery.
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


def rebuild_id(hospital: str, surgery_date: pd.Timestamp, sequence: str) -> str:
    return f"IU{hospital}{surgery_date:%Y%m%d}{sequence}"


def repair_subject_ids(raw_ids: pd.Series,
                       surgery_dates: pd.Series) -> tuple[pd.Series, list[str]]:
    """Reconcile each REDCap research ID against that row's Date of Surgery.

    The ID carries the surgery date inside it, so the two fields are redundant
    and can check each other. When they disagree the date is believed and the ID
    is rebuilt, keeping the hospital and the patient-of-the-day number.

    A repair is refused if it would land on an ID that some other row already
    holds legitimately (its own date agrees), so a typo can never overwrite a
    real patient.
    """
    text = raw_ids.astype("string").str.strip().str.upper()

    # Rows whose ID already agrees with their own surgery date are trusted and
    # are never a valid repair target.
    trusted: set[str] = set()
    for value, date in zip(text, surgery_dates):
        if pd.isna(value) or pd.isna(date):
            continue
        match = ID_STRUCTURE.match(value)
        if match and match.group(2) == f"{date:%Y%m%d}":
            trusted.add(value)

    resolved: list[str | None] = []
    notes: list[str] = []
    for value, date in zip(text, surgery_dates):
        if pd.isna(value) or not value:
            resolved.append(None)
            continue

        structured = ID_STRUCTURE.match(value)
        if pd.isna(date):
            # Nothing to check against: keep a well-formed ID, drop a broken one.
            resolved.append(value if structured else None)
            if not structured:
                notes.append(f"{value}: malformed and no Date of Surgery — dropped")
            continue

        if structured:
            hospital, datepart, sequence = structured.groups()
            if datepart == f"{date:%Y%m%d}":
                resolved.append(value)
                continue
        else:
            loose = ID_LOOSE.match(value)
            if loose is None:
                resolved.append(None)
                notes.append(f"{value}: unrecognised ID format — dropped")
                continue
            hospital, sequence = loose.groups()

        candidate = rebuild_id(hospital, date, sequence)
        if candidate in trusted:
            resolved.append(value if structured else None)
            notes.append(
                f"{value}: Date of Surgery {date:%Y-%m-%d} implies {candidate}, "
                f"but that ID is already held by a row whose own date agrees — "
                f"left unchanged"
            )
            continue

        resolved.append(candidate)
        notes.append(f"{value} -> {candidate}  (Date of Surgery {date:%Y-%m-%d})")

    return pd.Series(resolved, index=raw_ids.index, dtype="object"), notes


def load_event_times(redcap_path: Path,
                     repair_ids: bool = True) -> tuple[pd.DataFrame, list[str]]:
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
    surgery_date = pd.to_datetime(redcap[date_column], errors="coerce")

    def resolve(values: pd.Series) -> pd.Series:
        """Attach Date of Surgery to bare HH:MM[:SS] entries.

        Clock-only strings are handled by an explicit vectorized path rather
        than left to dateutil, which would parse them one element at a time
        (slow, and it warns) and date them to today.
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
            parsed.loc[other] = pd.to_datetime(text.loc[other], errors="coerce")
        return parsed

    if repair_ids:
        subject_ids, notes = repair_subject_ids(redcap[id_column], surgery_date)
    else:
        subject_ids = redcap[id_column].astype("string").str.strip().str.upper()
        notes = []

    induction_at = resolve(redcap[induction_column])
    or_entry_at = resolve(redcap[or_entry_column])

    # A late-evening case crosses midnight: both clock times get stamped with the
    # same Date of Surgery, which puts induction apparently ~23 h BEFORE OR entry.
    # Roll induction onto the next day when that happens.
    crossed = (induction_at.notna() & or_entry_at.notna()
               & ((or_entry_at - induction_at) > pd.Timedelta(hours=12)))
    induction_at.loc[crossed] = induction_at.loc[crossed] + pd.Timedelta(days=1)
    if int(crossed.sum()):
        print(f"Note: {int(crossed.sum())} case(s) crossed midnight between OR "
              f"entry and induction; induction rolled to the next day.")

    events = pd.DataFrame({
        "subject_id": subject_ids,
        "induction": induction_at,
        "or_entry": or_entry_at,
    })

    events = events.loc[events["subject_id"].notna() & events["subject_id"].ne("")]
    # REDCap repeats a patient across form rows. Collapse to one row per patient
    # taking the first non-missing value of each time, so an induction time on
    # one row and an OR-entry time on another are not lost to each other.
    events = (
        events.sort_values("subject_id")
        .groupby("subject_id", as_index=True)
        .agg(induction=("induction", "first"), or_entry=("or_entry", "first"))
    )
    return events, notes


def load_sedline_file(path: str, signal: str) -> pd.DataFrame | None:
    """Timestamped, valid samples of one signal from one patient's file."""
    frame = pd.read_csv(path, low_memory=False)
    spec = SIGNALS[signal]

    value_column = find_column(frame, spec["column_prefix"])
    if value_column is None:
        print(f"WARNING: {Path(path).name} has no '{signal}' column; skipped")
        return None

    date_column = find_column(frame, "date")
    time_column = find_column(frame, "time")          # ' Time' has a leading space
    epoch_column = find_column(frame, "epoch")

    timestamp = pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns]")
    if date_column and time_column and date_column != time_column:
        timestamp = pd.to_datetime(
            frame[date_column].astype("string").str.strip() + " "
            + frame[time_column].astype("string").str.strip(),
            errors="coerce",
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


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #

def build_points(args: argparse.Namespace, events: pd.DataFrame):
    """Pool every valid sample from every patient onto minutes-from-induction."""
    spec = SIGNALS[args.signal]
    low, high = spec["valid_range"]

    chunks: list[pd.DataFrame] = []
    audit: list[dict] = []

    for path in read_filepath_list(args.filepaths):
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
            frame = load_sedline_file(path, args.signal)
        except Exception as exc:
            record["status"] = f"read error: {exc}"
            audit.append(record)
            continue
        if frame is None or frame.empty:
            record["status"] = "no usable rows"
            audit.append(record)
            continue

        record["rows_in_file"] = int(len(frame))
        record["rows_dash_or_blank"] = int(frame["value"].isna().sum())

        keep = frame["value"].between(low, high) & frame["timestamp"].notna()
        record["rows_out_of_range"] = int(
            (frame["value"].notna() & ~frame["value"].between(low, high)).sum()
        )

        # How much a conventional artifact threshold would cost, reported even
        # when the filter is off so the choice is visible rather than implicit.
        record["rows_artf_gt_20"] = int((frame["artf"] > 20).sum())
        if args.max_artf is not None:
            keep &= frame["artf"].isna() | frame["artf"].le(args.max_artf)

        frame = frame.loc[keep].copy()
        if frame.empty:
            record["status"] = "no valid samples"
            audit.append(record)
            continue

        # Left edge: the first sample at or after OR entry.
        if pd.notna(or_entry):
            frame = frame.loc[frame["timestamp"] >= or_entry]
            record["or_entry_used"] = "yes"
        else:
            record["or_entry_used"] = "NO — no REDCap OR-entry time"
            print(f"WARNING: {subject_id} has no OR-entry time; its "
                  f"pre-induction samples are bounded only by --min-minutes")
        if frame.empty:
            record["status"] = "no samples at/after OR entry"
            audit.append(record)
            continue

        frame["minutes"] = (
            (frame["timestamp"] - induction).dt.total_seconds() / 60.0
        )
        # Record where the patient's record really starts before clipping, so a
        # far-left outlier can be named rather than silently cropped.
        record["first_minute_raw"] = round(float(frame["minutes"].min()), 2)

        inside = frame["minutes"].between(args.min_minutes, args.max_minutes)
        record["points_clipped_left"] = int((frame["minutes"] < args.min_minutes).sum())
        frame = frame.loc[inside]
        if frame.empty:
            record["status"] = "no samples inside the plotted window"
            audit.append(record)
            continue

        frame["subject_id"] = subject_id
        chunks.append(frame[["subject_id", "minutes", "value", "artf"]])

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


def bootstrap_band(points: pd.DataFrame, grid: np.ndarray,
                   args: argparse.Namespace):
    """95% band from resampling PATIENTS, not rows.

    A patient contributes thousands of correlated 2-second samples, so a row
    bootstrap would produce an interval far narrower than the data supports.
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
        if (replicate + 1) % 25 == 0:
            print(f"  bootstrap {replicate + 1}/{args.n_boot} "
                  f"({len(curves)} succeeded)")

    if not curves:
        return None, None
    stacked = np.vstack(curves)
    return (np.nanpercentile(stacked, 2.5, axis=0),
            np.nanpercentile(stacked, 97.5, axis=0))


def lag_readout(grid: np.ndarray, fitted: np.ndarray,
                points: pd.DataFrame) -> list[str]:
    """When the signal actually starts moving after the drug is pushed.

    Baseline is the median of the raw pre-induction samples, not a fitted value,
    so the reference point does not depend on the smoother. Everything after
    that is read off the post-induction curve.
    """
    lines: list[str] = []
    pre = points.loc[points["minutes"] < 0, "value"]
    after = grid >= 0
    if pre.empty or not after.any() or np.isnan(fitted[after]).all():
        return ["Not enough data either side of induction for a lag readout."]

    baseline = float(pre.median())
    grid_after = grid[after]
    curve_after = fitted[after]
    nadir_index = int(np.nanargmin(curve_after))
    nadir_value = float(curve_after[nadir_index])
    total_drop = baseline - nadir_value

    lines.append(f"Pre-induction baseline (median of raw samples): {baseline:.1f}")
    lines.append(f"Curve nadir after induction: {nadir_value:.1f} "
                 f"at {grid_after[nadir_index]:.2f} min")

    if total_drop <= 0:
        lines.append("No net fall after induction — lag times not defined.")
        return lines

    def first_time_below(target: float) -> float | None:
        below = np.flatnonzero(curve_after <= target)
        return float(grid_after[below[0]]) if below.size else None

    for label, fraction in (("10%", 0.10), ("50%", 0.50), ("90%", 0.90)):
        moment = first_time_below(baseline - fraction * total_drop)
        if moment is None:
            lines.append(f"Time to {label} of the total fall: not reached in window")
        else:
            lines.append(f"Time to {label} of the total fall: {moment:.2f} min "
                         f"({moment * 60:.0f} s)")
    lines.append(f"Total fall from baseline to nadir: {total_drop:.1f} units")
    return lines


# --------------------------------------------------------------------------- #
# Figure
# --------------------------------------------------------------------------- #

def make_figure(points: pd.DataFrame, grid, fitted, band_low, band_high,
                args: argparse.Namespace, output_path: Path) -> None:
    spec = SIGNALS[args.signal]
    figure, axis = plt.subplots(figsize=(13, 7))

    n_points = len(points)
    axis.scatter(
        points["minutes"], points["value"],
        s=4, alpha=0.08 if n_points > 50_000 else 0.18,
        color="#4c78a8", edgecolors="none", rasterized=True, zorder=2,
        label=f"raw 2-second samples (n={n_points:,})",
    )

    if fitted is not None:
        if band_low is not None:
            axis.fill_between(
                grid, band_low, band_high, color="#d1495b", alpha=0.20, zorder=4,
                label="95% band: refit after resampling patients "
                      "(how much the curve moves if the cohort changed)",
            )
        fit_label = {
            "lowess": f"LOWESS local trend, span={args.lowess_frac}",
            "linear": "least-squares straight line",
            "spline": f"penalized spline, k={args.n_splines}",
        }.get(args.fit, args.fit)
        axis.plot(grid, fitted, color="#d1495b", lw=2.8, zorder=5,
                  label=f"trend through the dots — {fit_label}, "
                        f"fitted separately each side of induction")

    axis.axvline(0, color="black", ls="--", lw=1.5, zorder=3)
    axis.annotate(
        "INDUCTION", xy=(0, 1.0), xycoords=("data", "axes fraction"),
        xytext=(4, -10), textcoords="offset points",
        fontsize=10, fontweight="bold", va="top", color="black",
    )

    axis.set_xlim(args.min_minutes, args.max_minutes)
    axis.set_xlabel("Time from induction (minutes)   —   "
                    "0 = primary induction med pushed")
    axis.set_ylabel(spec["label"])
    axis.set_title(
        f"{spec['label']} versus time from induction\n"
        f"{points['subject_id'].nunique()} patients, {n_points:,} raw "
        f"2-second samples — every valid sample plotted, nothing averaged"
    )
    axis.grid(True, color="#d9d9d9", lw=0.6, alpha=0.8)
    axis.set_axisbelow(True)
    legend = axis.legend(loc="lower left", framealpha=0.95, fontsize=9,
                         markerscale=3)
    for handle in legend.legend_handles:
        try:
            handle.set_alpha(1.0)
        except Exception:
            pass

    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


# --------------------------------------------------------------------------- #
# Terminal reporting
# --------------------------------------------------------------------------- #

def banner(title: str) -> None:
    print("\n" + title)
    print("-" * max(len(title), 60))


def report_windows(audit: pd.DataFrame, args: argparse.Namespace) -> None:
    """Who starts earliest, and who got clipped by --min-minutes."""
    included = audit.loc[audit["status"] == "included"].copy()
    if included.empty:
        return

    banner("Earliest-starting patients (answers 'who is way out on the left?')")
    columns = ["subject_id", "first_minute_raw", "minutes_min", "minutes_max",
               "points_plotted", "points_clipped_left", "or_entry_used"]
    earliest = included.sort_values("first_minute_raw").head(10)
    print(earliest[columns].to_string(index=False))

    clipped = included.loc[included["points_clipped_left"] > 0]
    if len(clipped):
        banner(f"Patients clipped by --min-minutes {args.min_minutes:g}")
        print(clipped[["subject_id", "first_minute_raw",
                       "points_clipped_left", "or_entry_used"]]
              .sort_values("first_minute_raw").to_string(index=False))
        print("\nA record starting far to the left of OR entry usually means "
              "the OR-entry time is missing for that patient, or the Sedline "
              "clock and the REDCap clock disagree.")

    no_or_entry = included.loc[included["or_entry_used"] != "yes"]
    if len(no_or_entry):
        banner("Included patients with NO REDCap OR-entry time")
        print(", ".join(sorted(no_or_entry["subject_id"])))


def report_exclusions(audit: pd.DataFrame) -> None:
    excluded = audit.loc[audit["status"] != "included"]
    if excluded.empty:
        print("\nNo patient files were excluded.")
        return
    banner(f"{len(excluded)} patient file(s) excluded")
    for reason, group in excluded.groupby("status"):
        ids = sorted(str(value) for value in group["subject_id"].dropna())
        print(f"  {len(group):>3}  {reason}")
        if ids:
            print(f"       {', '.join(ids)}")


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #

def main() -> int:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    for required in (args.filepaths, args.redcap):
        if not required.is_file():
            sys.stderr.write(f"Input file does not exist: {required}\n")
            return 1
    if args.min_minutes >= args.max_minutes:
        sys.stderr.write("--min-minutes must be less than --max-minutes\n")
        return 1

    events, repairs = load_event_times(args.redcap,
                                       repair_ids=not args.no_id_repair)
    if repairs:
        banner(f"{len(repairs)} REDCap research ID(s) reconciled "
               f"against Date of Surgery")
        for note in repairs:
            print(f"  {note}")
    print(f"\nREDCap: {len(events)} patients; induction time for "
          f"{int(events['induction'].notna().sum())}, OR-entry for "
          f"{int(events['or_entry'].notna().sum())}")

    points, audit = build_points(args, events)
    if points.empty:
        sys.stderr.write("No plottable samples.\n")
        report_exclusions(audit)
        return 1

    banner("Cohort plotted")
    print(f"{points['subject_id'].nunique()} patients, {len(points):,} raw "
          f"{args.signal} samples")
    print(f"Window: {args.min_minutes:g} to {args.max_minutes:g} minutes from "
          f"induction (actual data spans {points['minutes'].min():.1f} to "
          f"{points['minutes'].max():.1f})")

    if args.max_artf is None and "rows_artf_gt_20" in audit:
        would_drop = int(audit["rows_artf_gt_20"].fillna(0).sum())
        if would_drop:
            print(f"Note: --max-artf 20 would drop {would_drop:,} more samples "
                  f"(artifact filter is currently OFF)")

    report_windows(audit, args)
    report_exclusions(audit)

    grid = np.linspace(args.min_minutes, args.max_minutes, 600)
    fitted = band_low = band_high = None
    if args.fit != "none":
        print(f"\nFitting the trend curve [{args.fit}], separately each side "
              f"of induction...")
        fitted = fit_curve(points["minutes"].to_numpy(float),
                           points["value"].to_numpy(float), grid, args)
        if fitted is None:
            print("  the curve could not be fitted; plotting raw dots only")
        elif args.band and args.n_boot > 0:
            print(f"Resampling patients for the 95% band "
                  f"({args.n_boot} replicates)...")
            band_low, band_high = bootstrap_band(points, grid, args)

    if fitted is not None:
        banner(f"Lag readout — how {args.signal} responds after the drug is pushed")
        for line in lag_readout(grid, fitted, points):
            print(f"  {line}")

    # Monotone-association summaries, split at induction, so this signal can be
    # compared against the next variable in the series.
    stats_rows = []
    for label, subset in (
        ("pre-induction", points.loc[points["minutes"] < 0]),
        ("post-induction", points.loc[points["minutes"] >= 0]),
        ("all", points),
    ):
        if len(subset) > 2:
            slope, _intercept = np.polyfit(subset["minutes"], subset["value"], 1)
            stats_rows.append({
                "signal": args.signal,
                "window": label,
                "n_points": len(subset),
                "n_patients": subset["subject_id"].nunique(),
                "slope_per_min": round(float(slope), 4),
                "pearson_vs_time": round(
                    float(subset["minutes"].corr(subset["value"])), 4
                ),
                "spearman_vs_time": round(
                    float(subset["minutes"].corr(subset["value"], method="spearman")), 4
                ),
                "median_value": round(float(subset["value"].median()), 2),
            })
    banner("Summary")
    print(pd.DataFrame(stats_rows).to_string(index=False))

    figure_path = args.outdir / f"{args.signal.lower()}_vs_time_scatter.png"
    make_figure(points, grid, fitted, band_low, band_high, args, figure_path)
    print(f"\nFigure: {figure_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
