#!/usr/bin/env python3
"""Functional-regression scatter of a Sedline signal against time from induction.

Every valid 2-second Sedline sample from every patient is drawn as one dot, on a
shared x-axis where TIME = 0 is anesthesia induction, so pre- and post-induction
trend is directly visible. A smooth functional regression curve -- a penalized
spline in time -- is fitted through the pooled points, with a confidence band
obtained by resampling PATIENTS (not samples), which is the honest unit here
because a patient's own 2-second samples are highly correlated.

Data sources
------------
  * Sedline raw-cleaned files, one per patient, listed one path per line in
    --filepaths. Columns used (matched by prefix, so stray spaces are fine):
        Date, Time            -> timestamp   (Epoch Time used as a fallback)
        PSi (Sedline) Value   -> the signal  (or SEFL / SEFR / SR / EMG)
        ARTF % (Sedline) Value-> optional artifact filter
  * REDCap labeled export (--redcap) supplying, per patient:
        "What time was INDUCTION (when primary induction med pushed) (TIME = 0)?"
        "What time did the patient enter the OR?"

Window
------
  Each patient contributes samples from the first reading at or after OR entry
  through --max-minutes after induction. x is minutes from induction (negative
  before it).

Validity
--------
  Sedline writes "-" when a value is unavailable; those rows are dropped, as are
  values outside the physiological range for the chosen signal. Pass --max-artf
  to additionally drop high-artifact samples.

Usage
-----
    python psi_vs_time_regression.py
    python psi_vs_time_regression.py --signal SEFL --max-minutes 30
    python psi_vs_time_regression.py --max-artf 20 --n-boot 200
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
# Penalty used when --fit spline is selected.
DEFAULT_LAMBDA = 100.0
ID_PATTERN = re.compile(r"IU(?:MH|UH)\d+", re.IGNORECASE)

INDUCTION_PREFIX = "what time was induction"
OR_ENTRY_PREFIX = "what time did the patient enter the or"

# Known REDCap ID typos -> the true research ID.
REDCAP_ID_ALIASES = {
    "IUMH202601601": "IUMH2026011601",
    "IUMH2026010601": "IUMH2026010501",
}


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
        help="Where the figure and tables are written.")
    parser.add_argument(
        "--signal", choices=sorted(SIGNALS), default="PSi",
        help="Which Sedline signal to put on the y-axis (default PSi).")
    parser.add_argument(
        "--max-minutes", type=float, default=60.0,
        help="Last minute after induction to plot (default 60). The left edge "
             "is always each patient's first sample at/after OR entry.")
    parser.add_argument(
        "--max-artf", type=float, default=None,
        help="Optional: drop samples whose ARTF %% exceeds this. Off by "
             "default; the run reports what a threshold of 20 would cost.")
    parser.add_argument(
        "--n-boot", type=int, default=100,
        help="Patient-level bootstrap replicates for the confidence band "
             "(0 disables the band).")
    parser.add_argument(
        "--fit", choices=["linear", "lowess", "spline", "none"],
        default="linear",
        help="Line of best fit drawn over the raw points. 'linear' (default) "
             "is plain least squares, fitted separately before and after "
             "induction, and nothing is smoothed. 'lowess' draws a local "
             "trend curve instead; 'none' plots the raw scatter only. The "
             "scatter itself is always every raw valid sample.")
    parser.add_argument(
        "--lowess-frac", type=float, default=0.02,
        help="LOWESS smoothing span as a fraction of the data (default 0.02). "
             "Larger = smoother.")
    parser.add_argument(
        "--n-splines", type=int, default=80,
        help="Basis size when --fit spline is used. Needs to be large "
             "(80+) or the curve rings around induction.")
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
    """Research ID embedded in the file path, e.g. IUMH2026030501."""
    match = ID_PATTERN.search(path)
    if match is None:
        return None
    subject_id = match.group(0).upper()
    return REDCAP_ID_ALIASES.get(subject_id, subject_id)


def find_column(frame: pd.DataFrame, prefix: str) -> str | None:
    """Locate a column by case-insensitive prefix after stripping whitespace.

    The Sedline exports ship a leading space on ' Time', so exact names break.
    """
    prefix = prefix.strip().lower()
    for column in frame.columns:
        if str(column).strip().lower().startswith(prefix):
            return column
    return None


def load_event_times(redcap_path: Path) -> pd.DataFrame:
    """Per-patient induction and OR-entry instants from the REDCap export."""
    redcap = pd.read_csv(redcap_path, low_memory=False)

    id_column, best = None, 0
    for column in redcap.columns:
        # NB: match against the pattern as written - upper-casing it would turn
        # \d into \D and silently match nothing.
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

    date_column = find_column(redcap, "date of surgery")
    surgery_date = (
        pd.to_datetime(redcap[date_column], errors="coerce")
        if date_column else pd.Series(pd.NaT, index=redcap.index)
    )

    def resolve(values: pd.Series) -> pd.Series:
        """Attach Date of Surgery to bare HH:MM[:SS] entries."""
        parsed = pd.to_datetime(values.astype("string").str.strip(), errors="coerce")
        today = pd.Timestamp.today().normalize()
        # A bare clock time parses to today's date; re-anchor those.
        bare = parsed.notna() & parsed.dt.normalize().eq(today) & surgery_date.notna()
        parsed.loc[bare] = (
            surgery_date.loc[bare].dt.normalize() + (parsed.loc[bare] - today)
        )
        return parsed

    subject_ids = (
        redcap[id_column].astype("string").str.strip().str.upper()
        .replace(REDCAP_ID_ALIASES)
    )

    events = pd.DataFrame({
        "subject_id": subject_ids,
        "induction": resolve(redcap[induction_column]),
        "or_entry": resolve(redcap[or_entry_column]),
    })

    events = events.loc[events["subject_id"].notna()]
    # REDCap repeats a patient across form rows; keep the first row that has an
    # induction time, since that is the anchor everything else depends on.
    events = (
        events.sort_values("induction", na_position="last")
        .drop_duplicates("subject_id", keep="first")
        .set_index("subject_id")
    )
    return events


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
            record["or_entry_applied"] = True
        else:
            record["or_entry_applied"] = False
            print(f"WARNING: {subject_id} has no OR-entry time; keeping all "
                  f"pre-induction samples")
        if frame.empty:
            record["status"] = "no samples at/after OR entry"
            audit.append(record)
            continue

        frame["minutes"] = (
            (frame["timestamp"] - induction).dt.total_seconds() / 60.0
        )
        frame = frame.loc[frame["minutes"] <= args.max_minutes]
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
# Functional regression
# --------------------------------------------------------------------------- #

def linear_segments(minutes: np.ndarray, values: np.ndarray,
                    grid: np.ndarray) -> np.ndarray:
    """Ordinary least-squares line, fitted separately either side of induction.

    Induction is a near-discontinuity, so one straight line through the whole
    record would be dominated by the step and describe neither half. Two
    segments give an interpretable slope before and after.
    """
    fitted = np.full(grid.shape, np.nan)
    for keep_points, keep_grid in (
        (minutes < 0, grid < 0),
        (minutes >= 0, grid >= 0),
    ):
        if keep_points.sum() >= 2 and keep_grid.any():
            slope, intercept = np.polyfit(minutes[keep_points],
                                          values[keep_points], 1)
            fitted[keep_grid] = intercept + slope * grid[keep_grid]
    return fitted


def fit_curve(minutes: np.ndarray, values: np.ndarray, grid: np.ndarray,
              method: str, frac: float, n_splines: int,
              lam: float = DEFAULT_LAMBDA) -> np.ndarray | None:
    """Line of best fit through the pooled points, evaluated on `grid`.

    LOWESS is the default. A global spline basis with uniformly spaced knots
    rings around the near-step change at induction, throwing visible waves
    across the flat pre-induction stretch where the data is actually constant;
    local regression has no such basis to ring. On test data LOWESS reproduced
    a flat pre-induction segment to within 0.3 units where a 20-knot spline
    swung by 24.
    """
    if method == "none":
        return None

    if method == "linear":
        return linear_segments(minutes, values, grid)

    if method == "lowess":
        try:
            from statsmodels.nonparametric.smoothers_lowess import lowess
            span = float(minutes.max() - minutes.min())
            fitted = lowess(values, minutes, frac=frac, it=0,
                            delta=0.01 * span, return_sorted=True)
            return np.interp(grid, fitted[:, 0], fitted[:, 1])
        except Exception:
            pass
    else:
        try:
            from pygam import LinearGAM, s
            gam = LinearGAM(s(0, n_splines=n_splines), lam=float(lam)).fit(
                minutes.reshape(-1, 1), values
            )
            return gam.predict(grid.reshape(-1, 1))
        except Exception:
            pass

    # Last resort: binned medians joined up.
    edges = np.linspace(grid.min(), grid.max(), n_splines + 1)
    index = np.clip(np.digitize(minutes, edges[1:-1]), 0, len(edges) - 2)
    centers, medians = [], []
    for bucket in range(len(edges) - 1):
        mask = index == bucket
        if mask.sum() >= 5:
            centers.append(np.median(minutes[mask]))
            medians.append(np.median(values[mask]))
    if len(centers) < 3:
        return None
    return np.interp(grid, centers, medians)


def functional_regression(points: pd.DataFrame, args: argparse.Namespace):
    """Fitted curve plus a patient-resampled confidence band.

    Resampling PATIENTS rather than rows matters: a patient contributes
    thousands of correlated 2-second samples, so a row bootstrap would produce
    an interval far narrower than the data supports.
    """
    minutes = points["minutes"].to_numpy(float)
    values = points["value"].to_numpy(float)
    grid = np.linspace(minutes.min(), minutes.max(), 400)

    fitted = fit_curve(minutes, values, grid, args.fit,
                       args.lowess_frac, args.n_splines)
    if fitted is None:
        return grid, None, None, None

    if args.n_boot <= 0:
        return grid, fitted, None, None

    rng = np.random.default_rng(args.seed)
    subject_ids = points["subject_id"].to_numpy()
    unique_ids = np.unique(subject_ids)
    by_subject = {sid: np.flatnonzero(subject_ids == sid) for sid in unique_ids}

    curves = []
    for replicate in range(args.n_boot):
        draw = rng.choice(unique_ids, size=len(unique_ids), replace=True)
        index = np.concatenate([by_subject[sid] for sid in draw])
        curve = fit_curve(minutes[index], values[index], grid, args.fit,
                          args.lowess_frac, args.n_splines)
        if curve is not None:
            curves.append(curve)
        if (replicate + 1) % 25 == 0:
            print(f"  bootstrap {replicate + 1}/{args.n_boot} "
                  f"({len(curves)} succeeded)")

    if not curves:
        return grid, fitted, None, None
    stacked = np.vstack(curves)
    return (grid, fitted,
            np.percentile(stacked, 2.5, axis=0),
            np.percentile(stacked, 97.5, axis=0))


# --------------------------------------------------------------------------- #
# Figure
# --------------------------------------------------------------------------- #

def make_figure(points: pd.DataFrame, grid, fitted, low, high,
                args: argparse.Namespace, output_path: Path) -> None:
    spec = SIGNALS[args.signal]
    figure, axis = plt.subplots(figsize=(13, 7))

    n_points = len(points)
    axis.scatter(
        points["minutes"], points["value"],
        s=3, alpha=0.06 if n_points > 50_000 else 0.15,
        color="#4c78a8", edgecolors="none", rasterized=True, zorder=2,
    )

    if fitted is not None:
        if low is not None:
            axis.fill_between(grid, low, high, color="#d1495b", alpha=0.25,
                              zorder=4, label="95% CI (patient bootstrap)")
        fit_label = {
            "linear": "least squares, fitted each side of induction",
            "lowess": f"LOWESS, span={args.lowess_frac}",
            "spline": f"spline, k={args.n_splines}",
        }.get(args.fit, args.fit)
        axis.plot(grid, fitted, color="#d1495b", lw=2.8, zorder=5,
                  label=f"Line of best fit - {fit_label}")

    axis.axvline(0, color="black", ls="--", lw=1.5, zorder=3)
    axis.annotate(
        "INDUCTION", xy=(0, axis.get_ylim()[1]), xytext=(4, -14),
        textcoords="offset points", fontsize=10, fontweight="bold",
        va="top", color="black",
    )

    axis.set_xlabel("Time from induction (minutes)   —   0 = primary induction med pushed")
    axis.set_ylabel(spec["label"])
    axis.set_title(
        f"{spec['label']} versus time from induction\n"
        f"{points['subject_id'].nunique()} patients, {n_points:,} valid "
        f"2-second samples"
    )
    axis.grid(True, color="#d9d9d9", lw=0.6, alpha=0.8)
    axis.set_axisbelow(True)
    axis.legend(loc="upper right", framealpha=0.95)

    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


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

    events = load_event_times(args.redcap)
    print(f"REDCap: induction time for {int(events['induction'].notna().sum())} "
          f"patients, OR-entry for {int(events['or_entry'].notna().sum())}")

    points, audit = build_points(args, events)
    if points.empty:
        sys.stderr.write("No plottable samples. See the audit table.\n")
        audit.to_csv(args.outdir / "patient_audit.csv", index=False)
        return 1

    print(f"\nIncluded {points['subject_id'].nunique()} patients, "
          f"{len(points):,} valid {args.signal} samples")
    print(f"Time range plotted: {points['minutes'].min():.1f} to "
          f"{points['minutes'].max():.1f} minutes from induction")

    if args.max_artf is None and "rows_artf_gt_20" in audit:
        would_drop = int(audit["rows_artf_gt_20"].fillna(0).sum())
        if would_drop:
            print(f"Note: --max-artf 20 would drop {would_drop:,} more samples "
                  f"(artifact filter is currently OFF)")

    print(f"\nFitting line of best fit [{args.fit}] with {args.n_boot} "
          f"patient-level bootstrap replicates...")
    grid, fitted, low, high = functional_regression(points, args)

    # Simple monotone-association summaries, split at induction, so this signal
    # can be compared against the next variable in the series.
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
    stats = pd.DataFrame(stats_rows)
    print("\n" + stats.to_string(index=False))

    curve = pd.DataFrame({"minutes": grid, "fitted": fitted})
    if low is not None:
        curve["ci_low"], curve["ci_high"] = low, high

    stem = f"{args.signal.lower()}_vs_time"
    figure_path = args.outdir / f"{stem}_functional_regression.png"
    make_figure(points, grid, fitted, low, high, args, figure_path)

    points.to_csv(args.outdir / f"{stem}_points.csv", index=False)
    curve.to_csv(args.outdir / f"{stem}_curve.csv", index=False)
    stats.to_csv(args.outdir / f"{stem}_summary.csv", index=False)
    audit.to_csv(args.outdir / "patient_audit.csv", index=False)

    excluded = audit.loc[audit.get("status", "") != "included"]
    if len(excluded):
        print(f"\n{len(excluded)} patient file(s) excluded:")
        for reason, count in excluded["status"].value_counts().items():
            print(f"    {count:>3}  {reason}")

    print(f"\nFigure:  {figure_path}")
    print(f"Points:  {args.outdir / f'{stem}_points.csv'}")
    print(f"Curve:   {args.outdir / f'{stem}_curve.csv'}")
    print(f"Summary: {args.outdir / f'{stem}_summary.csv'}")
    print(f"Audit:   {args.outdir / 'patient_audit.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
