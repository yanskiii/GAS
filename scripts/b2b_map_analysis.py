#!/usr/bin/env python3
"""
Beat-to-beat (B2B) Mean Arterial Pressure (MAP) cleaning & analysis pipeline.

Reads a MASTER csv that lists one file path per line (one line = one patient's
beat-to-beat MAP csv), then for every patient it:

  1. Loads that patient's MAP csv.
  2. Filters out "bad" rows (the ``databad`` column == 1).
  3. Filters out missing / physiologically-implausible MAP values.
  4. Converts the time column to *elapsed seconds from the start of the record*
     so that every patient is on a common, surgery-relative timeline.
  5. Resamples onto a steady 2-second grid (handles drift, duplicate stamps,
     and averages multiple readings that fall in the same 2 s bin).
  6. Writes a cleaned per-patient csv (kept organized in one output folder).
  7. Computes per-patient summary metrics (mean/median/spread, time spent
     below hypotension thresholds, and area-under-threshold "dose").

Finally it writes two group-level files:

  * ``patient_summary.csv``      -- one row per patient (the summary metrics).
  * ``all_patients_map_long.csv``-- every cleaned sample stacked in long format
                                    (patient_id, elapsed_sec, meanArterialPressure).
                                    This long/tidy file is what you will later
                                    merge against the PSI data on
                                    (patient_id, elapsed_sec).

Column layout of each patient csv (per the study description; 5 columns total):

    index 0 : <unused>
    index 1 : databad                 (1 == whole row is bad -> drop)
    index 2 : meanArterialPressure    (recorded every 2 s)
    index 3 : time
    index 4 : <unused>

If your files use different positions or names, edit the CONFIG block below
or pass the matching command-line flags.

Usage
-----
    python b2b_map_analysis.py
    python b2b_map_analysis.py --master /path/to/b2b_filepath.csv --outdir ./b2b_output
    python b2b_map_analysis.py --plots          # also save a per-patient PNG

Nothing here is specific to the cluster; run it wherever the data lives.
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# CONFIG -- sensible defaults; override any of these on the command line.
# --------------------------------------------------------------------------- #

DEFAULT_MASTER = "/N/project/Analgesia_BDproject/PR/scripts_PR/b2b_filepath.csv"

# Column positions inside each patient csv (0-based).
DATABAD_COL_INDEX = 1
MAP_COL_INDEX = 2
TIME_COL_INDEX = 3

# Preferred column *names* -- used only if the patient csv has a header row and
# the name is present. Falls back to the positional index above otherwise.
DATABAD_COL_NAME = "databad"
MAP_COL_NAME = "meanArterialPressure"
TIME_COL_NAME = "time"

# Sampling grid (seconds). The study records MAP every 2 seconds.
GRID_SECONDS = 2

# Physiologically plausible MAP window (mmHg). Values outside are treated as
# artefacts and dropped along with the databad rows. Widen/narrow as needed.
MAP_MIN_PLAUSIBLE = 20.0
MAP_MAX_PLAUSIBLE = 250.0

# Only bridge gaps this short by linear interpolation (in number of grid steps).
# A gap of <= this many missing 2 s samples gets filled; longer gaps stay NaN
# so real missing stretches are not invented. Set to 0 to never interpolate.
MAX_INTERP_STEPS = 2  # i.e. up to 4 s of gap

# Hypotension thresholds to score (mmHg). 65 is the common clinical cutoff.
MAP_THRESHOLDS = (65, 60, 55)


# --------------------------------------------------------------------------- #
# Data structures
# --------------------------------------------------------------------------- #

@dataclass
class PatientResult:
    """Summary row for a single patient (becomes one line of patient_summary.csv)."""
    patient_id: str
    source_path: str
    status: str                 # "ok" or "error: ..."
    n_rows_raw: int = 0
    n_rows_databad: int = 0     # rows removed because databad == 1
    n_rows_implausible: int = 0 # rows removed for MAP out of range / missing
    n_samples_clean: int = 0    # samples on the 2 s grid with a real value
    duration_min: float = np.nan
    coverage_pct: float = np.nan  # % of grid slots that have a real value
    map_mean: float = np.nan
    map_median: float = np.nan
    map_std: float = np.nan
    map_min: float = np.nan
    map_max: float = np.nan
    map_p05: float = np.nan
    map_p95: float = np.nan
    # Threshold metrics get added dynamically (time_below_65_min, auc_below_65, ...)
    extra: dict | None = None

    def to_row(self) -> dict:
        row = asdict(self)
        extra = row.pop("extra") or {}
        row.update(extra)
        return row


# --------------------------------------------------------------------------- #
# Core helpers
# --------------------------------------------------------------------------- #

def read_master_filelist(master_path: str) -> list[str]:
    """Return the list of patient csv paths from the master file.

    The master file is one path per line. We are tolerant of: an optional
    header, blank lines, quotes, surrounding whitespace, and a stray extra
    column (we take the first non-empty field of each line).
    """
    if not os.path.isfile(master_path):
        raise FileNotFoundError(f"Master file not found: {master_path}")

    paths: list[str] = []
    with open(master_path, "r", newline="") as fh:
        for raw in fh:
            line = raw.strip().strip('"').strip("'").strip()
            if not line:
                continue
            # If the line is comma/tab separated, keep the first field.
            first = line.replace("\t", ",").split(",")[0].strip().strip('"').strip("'")
            paths.append(first)

    # Drop a header line if it clearly isn't a path (no separator, looks like a name).
    if paths and not (os.sep in paths[0] or paths[0].lower().endswith(".csv")):
        header_like = paths[0].lower()
        if header_like in {"path", "filepath", "file", "filename", "files", "paths"}:
            paths = paths[1:]

    return paths


def patient_id_from_path(path: str) -> str:
    """Derive a stable patient id from the file path (the file's base name)."""
    base = os.path.basename(path.rstrip("/\\"))
    stem, _ext = os.path.splitext(base)
    return stem if stem else base


def _pick_column(df: pd.DataFrame, name: str, index: int, what: str) -> pd.Series:
    """Return the column for `what`, preferring the named column, else positional."""
    if name in df.columns:
        return df[name]
    if index < 0 or index >= df.shape[1]:
        raise ValueError(
            f"Cannot locate {what}: no column named '{name}' and index {index} "
            f"is out of range (file has {df.shape[1]} columns)."
        )
    return df.iloc[:, index]


def load_patient_csv(path: str) -> pd.DataFrame:
    """Load one patient csv, tolerating whitespace and optional header."""
    # header="infer" lets pandas use a real header if present; positional access
    # via _pick_column still works when there is no header (columns become 0..n).
    df = pd.read_csv(path, header="infer", skipinitialspace=True)

    # If pandas read a headerless file, columns are 0,1,2,... already, which is
    # fine. If it mistook the first data row for a header (all-numeric names),
    # re-read without a header so positions line up.
    looks_numeric_header = all(
        str(c).replace(".", "", 1).replace("-", "", 1).isdigit() for c in df.columns
    )
    if looks_numeric_header:
        df = pd.read_csv(path, header=None, skipinitialspace=True)

    return df


def time_to_elapsed_seconds(time_series: pd.Series) -> pd.Series:
    """Convert an arbitrary time column into *elapsed seconds from record start*.

    Handles three common encodings:
      * numeric seconds (elapsed or epoch)  -> subtract the minimum
      * clock/date strings (HH:MM:SS, ISO)  -> parse to datetime, subtract min
      * numeric that is really datetime-ns  -> handled by the datetime branch

    Returns a float Series (seconds). NaN where the time could not be parsed.
    """
    # First try a straight numeric interpretation.
    numeric = pd.to_numeric(time_series, errors="coerce")
    if numeric.notna().mean() >= 0.9:
        return numeric - numeric.min(skipna=True)

    # Otherwise parse as datetime / clock time.
    dt = pd.to_datetime(time_series, errors="coerce")
    if dt.notna().mean() >= 0.9:
        return (dt - dt.min()).dt.total_seconds()

    # Last resort: mixed / unparseable -> coerce numerically and warn upstream.
    return numeric - numeric.min(skipna=True)


def clean_and_resample(df: pd.DataFrame, result: PatientResult) -> pd.DataFrame:
    """Clean one patient's frame and return it on a steady 2 s grid.

    Output columns: ['elapsed_sec', 'meanArterialPressure'].
    Mutates `result` with the row-count bookkeeping.
    """
    result.n_rows_raw = len(df)

    databad = pd.to_numeric(
        _pick_column(df, DATABAD_COL_NAME, DATABAD_COL_INDEX, "databad"),
        errors="coerce",
    )
    map_vals = pd.to_numeric(
        _pick_column(df, MAP_COL_NAME, MAP_COL_INDEX, "meanArterialPressure"),
        errors="coerce",
    )
    elapsed = time_to_elapsed_seconds(
        _pick_column(df, TIME_COL_NAME, TIME_COL_INDEX, "time")
    )

    work = pd.DataFrame(
        {"elapsed_sec": elapsed, "meanArterialPressure": map_vals, "databad": databad}
    )

    # 1) Drop the explicitly-flagged bad rows (databad == 1).
    bad_mask = work["databad"] == 1
    result.n_rows_databad = int(bad_mask.sum())
    work = work[~bad_mask]

    # 2) Drop rows with unusable time or MAP, or implausible MAP.
    before = len(work)
    plausible = (
        work["meanArterialPressure"].between(MAP_MIN_PLAUSIBLE, MAP_MAX_PLAUSIBLE)
        & work["elapsed_sec"].notna()
    )
    work = work[plausible]
    result.n_rows_implausible = int(before - len(work))

    if work.empty:
        return pd.DataFrame(columns=["elapsed_sec", "meanArterialPressure"])

    # 3) Put onto a steady 2 s grid.
    #    Use a TimedeltaIndex + resample so drift, duplicate timestamps, and
    #    multiple readings per bin are all handled (bin -> mean).
    work = work.sort_values("elapsed_sec")
    work.index = pd.to_timedelta(work["elapsed_sec"], unit="s")
    grid = (
        work["meanArterialPressure"]
        .resample(f"{GRID_SECONDS}s")
        .mean()
    )

    # 4) Optionally bridge only *short* gaps so we don't invent long missing runs.
    if MAX_INTERP_STEPS > 0:
        grid = grid.interpolate(method="linear", limit=MAX_INTERP_STEPS,
                                limit_area="inside")

    out = pd.DataFrame({
        "elapsed_sec": grid.index.total_seconds(),
        "meanArterialPressure": grid.values,
    })
    return out


def compute_metrics(clean: pd.DataFrame, result: PatientResult) -> None:
    """Fill summary statistics + threshold metrics on `result` from cleaned data."""
    values = clean["meanArterialPressure"]
    real = values.dropna()

    result.n_samples_clean = int(real.shape[0])
    if clean.empty:
        return

    total_slots = len(clean)
    span_sec = clean["elapsed_sec"].iloc[-1] - clean["elapsed_sec"].iloc[0]
    result.duration_min = round(span_sec / 60.0, 3)
    result.coverage_pct = round(100.0 * real.shape[0] / total_slots, 2) if total_slots else np.nan

    if real.empty:
        return

    result.map_mean = round(float(real.mean()), 3)
    result.map_median = round(float(real.median()), 3)
    result.map_std = round(float(real.std(ddof=1)), 3) if real.shape[0] > 1 else 0.0
    result.map_min = round(float(real.min()), 3)
    result.map_max = round(float(real.max()), 3)
    result.map_p05 = round(float(real.quantile(0.05)), 3)
    result.map_p95 = round(float(real.quantile(0.95)), 3)

    # Threshold metrics. Each real sample represents GRID_SECONDS of time.
    minutes_per_sample = GRID_SECONDS / 60.0
    extra: dict = {}
    for thr in MAP_THRESHOLDS:
        below = real[real < thr]
        # Time spent below threshold (minutes).
        extra[f"time_below_{thr}_min"] = round(len(below) * minutes_per_sample, 3)
        # Area-under-threshold "dose": sum of (threshold - MAP) * time (mmHg*min).
        extra[f"auc_below_{thr}_mmHg_min"] = round(
            float((thr - below).sum()) * minutes_per_sample, 3
        )
    result.extra = extra


def process_patient(path: str, outdir: str, save_plot: bool) -> PatientResult:
    """Full pipeline for one patient. Never raises -- records errors in the result."""
    pid = patient_id_from_path(path)
    result = PatientResult(patient_id=pid, source_path=path, status="ok")

    try:
        if not os.path.isfile(path):
            result.status = "error: file not found"
            return result

        df = load_patient_csv(path)
        clean = clean_and_resample(df, result)
        compute_metrics(clean, result)

        # Persist the cleaned, grid-aligned data for later PSI merging.
        clean_dir = os.path.join(outdir, "cleaned")
        os.makedirs(clean_dir, exist_ok=True)
        clean_out = clean.copy()
        clean_out.insert(0, "patient_id", pid)
        clean_out.to_csv(os.path.join(clean_dir, f"{pid}_map_clean.csv"), index=False)

        if save_plot:
            _save_patient_plot(clean, pid, outdir)

    except Exception as exc:  # keep the batch going even if one file is broken
        result.status = f"error: {exc}"
        sys.stderr.write(f"[WARN] {pid}: {exc}\n{traceback.format_exc()}\n")

    return result


def _save_patient_plot(clean: pd.DataFrame, pid: str, outdir: str) -> None:
    """Save a simple MAP-vs-time PNG for eyeballing the cleaned trace."""
    try:
        import matplotlib
        matplotlib.use("Agg")  # headless / cluster safe
        import matplotlib.pyplot as plt
    except Exception:
        sys.stderr.write("[WARN] matplotlib not available; skipping plots.\n")
        return

    plot_dir = os.path.join(outdir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(clean["elapsed_sec"] / 60.0, clean["meanArterialPressure"],
            lw=0.8, color="#1f77b4")
    for thr in MAP_THRESHOLDS:
        ax.axhline(thr, ls="--", lw=0.7, color="#d62728", alpha=0.5)
    ax.set_xlabel("Elapsed time (min)")
    ax.set_ylabel("MAP (mmHg)")
    ax.set_title(f"Patient {pid} -- cleaned beat-to-beat MAP")
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, f"{pid}_map.png"), dpi=110)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #

def build_long_dataset(outdir: str) -> str:
    """Concatenate every cleaned per-patient csv into one long/tidy file."""
    clean_dir = os.path.join(outdir, "cleaned")
    long_path = os.path.join(outdir, "all_patients_map_long.csv")
    if not os.path.isdir(clean_dir):
        return long_path

    frames = []
    for name in sorted(os.listdir(clean_dir)):
        if name.endswith("_map_clean.csv"):
            frames.append(pd.read_csv(os.path.join(clean_dir, name)))
    if frames:
        pd.concat(frames, ignore_index=True).to_csv(long_path, index=False)
    return long_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Clean and analyze beat-to-beat MAP data for all patients."
    )
    parser.add_argument("--master", default=DEFAULT_MASTER,
                        help="Path to the master csv listing each patient's MAP csv "
                             "(one path per line).")
    parser.add_argument("--outdir", default="b2b_output",
                        help="Directory for cleaned files, summary, and plots.")
    parser.add_argument("--plots", action="store_true",
                        help="Also save a MAP-vs-time PNG per patient.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only the first N patients (for a quick test).")
    args = parser.parse_args(argv)

    os.makedirs(args.outdir, exist_ok=True)

    print(f"Reading master file list: {args.master}")
    paths = read_master_filelist(args.master)
    if args.limit:
        paths = paths[: args.limit]
    print(f"Found {len(paths)} patient file(s) to process.\n")

    results: list[PatientResult] = []
    for i, path in enumerate(paths, 1):
        pid = patient_id_from_path(path)
        print(f"[{i}/{len(paths)}] {pid} ...", end=" ", flush=True)
        res = process_patient(path, args.outdir, args.plots)
        print(res.status if res.status != "ok"
              else f"ok  (n={res.n_samples_clean}, mean MAP={res.map_mean})")
        results.append(res)

    # Group-level outputs.
    summary_df = pd.DataFrame([r.to_row() for r in results])
    summary_path = os.path.join(args.outdir, "patient_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    long_path = build_long_dataset(args.outdir)

    n_ok = sum(r.status == "ok" for r in results)
    n_err = len(results) - n_ok
    print("\n" + "=" * 60)
    print(f"Done. {n_ok} ok, {n_err} error(s).")
    print(f"  Per-patient cleaned csvs : {os.path.join(args.outdir, 'cleaned')}/")
    print(f"  Group summary            : {summary_path}")
    print(f"  Long (tidy) MAP dataset  : {long_path}")
    if args.plots:
        print(f"  Plots                    : {os.path.join(args.outdir, 'plots')}/")
    print("=" * 60)

    return 0 if n_err == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
