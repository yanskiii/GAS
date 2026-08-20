#!/usr/bin/env python3
"""
Induction-aligned MAP vs PSi analysis.

Implements the analysis plan:

  Clinical question: can we provide optimal sedation (induction med) without
  significantly altering hemodynamics (i.e., without hypotension)?

  * TIME = 0 is anesthesia INDUCTION (when the primary induction med was
    pushed). Every patient's timeline is re-aligned to that moment.
  * PREOP BASELINE MAP = median MAP between the "SQI > 3 bars" time (or preop
    monitoring start) and induction.
  * PRIMARY METRIC:  (lowest MAP in the 10 min after induction - baseline)
                     / baseline * 100   -> percent change from baseline.
  * PRIMARY PLOT:    sedation depth (PSi) on the x-axis vs the % MAP change
                     on the y-axis (one point per patient).
  * GROUP PLOT:      median + IQR at 20-second intervals, aligned to
                     induction, for MAP (as % of baseline) and PSi.
  * SENSITIVITY:     the Sedline PSi lags true brain state, so the analysis
                     can be re-run with PSi shifted earlier by 10 / 20 s
                     (--psi-shift).

Input format (long / tidy - e.g. Sample_1.xlsx or the combined csv):

    patientID , time , meanArterialPressure , PSi

  `time` may be HH:MM:SS or a full datetime. MAP and PSi do NOT need to share
  rows (in the export they are offset by ~1 s); each signal is resampled onto
  a common 2 s grid before they are compared.

Event times (induction etc.) come from an events csv (--events):

    patientID , induction_time , preop_start , sqi_time
    ABC123    , 07:40:32       , 07:07:00    , 07:08:00
    XYZ321    , 12:41:00       , 11:18:00    , 11:18:52

  Built-in defaults for the two sample patients are used if no file is given.

Usage
-----
    python induction_map_psi_analysis.py --data Sample_1.xlsx
    python induction_map_psi_analysis.py --data combined.csv --events events.csv
    python induction_map_psi_analysis.py --data Sample_1.xlsx --psi-shift 20
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------- #
# CONFIG
# --------------------------------------------------------------------------- #

GRID_SECONDS = 2          # base alignment grid (data is ~every 2 s per signal)
BIN_SECONDS = 20          # bin width for the median + IQR group plot
POST_WINDOW_MIN = 10      # "lowest MAP post-induction" search window (minutes)
PLOT_PRE_MIN = 10         # minutes before induction to show in plots
PLOT_POST_MIN = 30        # minutes after induction to show in plots
MAP_MIN_PLAUSIBLE = 20.0  # artefact filter (mmHg)
MAP_MAX_PLAUSIBLE = 250.0
HYPOTENSION_MAP = 65      # mmHg reference line / burden metric

MAP_COL = "meanArterialPressure"
PSI_COL = "PSi"

# Defaults for the sample patients (from the analysis plan). An --events csv
# with columns patientID,induction_time,preop_start,sqi_time overrides these.
DEFAULT_EVENTS = {
    "ABC123": {"induction_time": "07:40:32", "preop_start": "07:07:00", "sqi_time": "07:08:00"},
    "XYZ321": {"induction_time": "12:41:00", "preop_start": "11:18:00", "sqi_time": "11:18:52"},
}


# --------------------------------------------------------------------------- #
# Loading & alignment
# --------------------------------------------------------------------------- #

def to_seconds_of_day(series: pd.Series) -> pd.Series:
    """Parse HH:MM:SS strings / datetimes / times into seconds since midnight."""
    def one(v):
        if pd.isna(v):
            return np.nan
        ts = pd.to_datetime(str(v), errors="coerce")
        if pd.isna(ts):
            return np.nan
        return ts.hour * 3600 + ts.minute * 60 + ts.second
    return series.map(one)


def load_long_data(path: str) -> pd.DataFrame:
    """Load the long-format data file (xlsx or csv)."""
    if path.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)
    missing = {"patientID", "time", MAP_COL, PSI_COL} - set(df.columns)
    if missing:
        raise ValueError(f"Data file is missing columns: {missing}")
    return df


# --------------------------------------------------------------------------- #
# Direct loading from the master file-path lists (no combined csv needed).
# Mirrors the MASTER_combined build step, vectorized.
# --------------------------------------------------------------------------- #

def read_filepath_list(master: str) -> list[str]:
    """One file path per line; tolerate blanks/quotes/extra columns."""
    paths = []
    with open(master, "r") as fh:
        for raw in fh:
            first = raw.strip().split(",")[0].strip().strip('"').strip("'")
            if first:
                paths.append(first)
    return paths


def patient_id_from_path(path: str) -> str | None:
    """The 14 characters starting at 'IU' in the path."""
    i = path.find("IU")
    return None if i == -1 else path[i:i + 14]


def load_from_masters(btb_master: str, psi_master: str) -> pd.DataFrame:
    """Build the long patientID/time/MAP/PSi table straight from the two
    master lists of per-patient file paths (BTB MAP files + Sedline files)."""
    frames = []

    for p in read_filepath_list(btb_master):
        pid = patient_id_from_path(p)
        if pid is None:
            sys.stderr.write(f"[WARN] no 'IU' id in path, skipped: {p}\n")
            continue
        try:
            df = pd.read_csv(p, skipinitialspace=True)
            df.columns = df.columns.str.strip()
            good = df[pd.to_numeric(df["databad"], errors="coerce") == 0]
            frames.append(pd.DataFrame({
                "patientID": pid,
                "time": good["time"].astype(str).str[-8:],
                MAP_COL: pd.to_numeric(good["meanArterialPressure"], errors="coerce"),
                PSI_COL: np.nan,
            }))
        except Exception as exc:
            sys.stderr.write(f"[WARN] {pid} (BTB): {exc} — skipped\n")

    for p in read_filepath_list(psi_master):
        pid = patient_id_from_path(p)
        if pid is None:
            sys.stderr.write(f"[WARN] no 'IU' id in path, skipped: {p}\n")
            continue
        try:
            df = pd.read_csv(p, skipinitialspace=True)
            df.columns = df.columns.str.strip()  # fixes the ' Time' leading space
            frames.append(pd.DataFrame({
                "patientID": pid,
                "time": df["Time"].astype(str).str[-8:],
                MAP_COL: np.nan,
                PSI_COL: pd.to_numeric(df["PSi (Sedline) Value"], errors="coerce"),
            }))
        except Exception as exc:
            sys.stderr.write(f"[WARN] {pid} (Sedline): {exc} — skipped\n")

    if not frames:
        raise ValueError("Nothing loaded from the master file lists.")
    combined = pd.concat(frames, ignore_index=True)
    return combined.sort_values(["patientID", "time"]).reset_index(drop=True)


# The REDCap export's column labels (matched loosely: lowercase substring, so
# minor wording/typo fixes in future exports still match).
REDCAP_ID_COL = "screening id"
REDCAP_COLS = {
    "induction_time": "what time was induction",
    "preop_start": "time monitoring started in preop",
    "sqi_time": "sqi>3 bars in preop",
}


def load_events(path: str | None) -> dict:
    """Events per patient: induction_time, preop_start, sqi_time.

    Accepts EITHER format and auto-detects which one it got:
      1. simple csv:  patientID,induction_time,preop_start,sqi_time
      2. the REDCap DATA_LABELS export: patient id in 'Screening ID', times in
         the long question-label columns (see REDCAP_COLS above).
    """
    if path is None:
        return dict(DEFAULT_EVENTS)
    ev = pd.read_csv(path)

    # Format 1: the simple hand-made events csv.
    if "induction_time" in ev.columns:
        out = {}
        for _, r in ev.iterrows():
            out[str(r["patientID"]).strip()] = {
                "induction_time": str(r["induction_time"]).strip(),
                "preop_start": str(r.get("preop_start", "")).strip(),
                "sqi_time": str(r.get("sqi_time", "")).strip(),
            }
        return out

    # Format 2: REDCap export — find columns by (case-insensitive) substring.
    lower = {str(c).strip().lower(): c for c in ev.columns}

    def find_col(fragment: str) -> str | None:
        for lc, orig in lower.items():
            if fragment in lc:
                return orig
        return None

    id_col = find_col(REDCAP_ID_COL)
    time_cols = {key: find_col(frag) for key, frag in REDCAP_COLS.items()}
    if id_col is None or time_cols["induction_time"] is None:
        raise ValueError(
            f"Could not recognize the events file '{path}': it has neither an "
            f"'induction_time' column (simple format) nor REDCap columns like "
            f"'Screening ID' / 'What time was INDUCTION...'."
        )

    out = {}
    for _, r in ev.iterrows():
        pid = str(r[id_col]).strip()
        if not pid or pid.lower() == "nan":
            continue
        entry = {}
        for key, col in time_cols.items():
            val = str(r[col]).strip() if col is not None and pd.notna(r[col]) else ""
            entry[key] = "" if val.lower() == "nan" else val
        # REDCap can export a patient across several rows; keep the first row
        # that actually has an induction time.
        if pid not in out or (not out[pid]["induction_time"] and entry["induction_time"]):
            out[pid] = entry
    # Drop patients with no induction time at all (alignment impossible).
    return {pid: e for pid, e in out.items() if e["induction_time"]}


def align_patient(g: pd.DataFrame, induction_sec: float,
                  psi_shift_sec: float) -> pd.DataFrame:
    """Return one patient's MAP + PSi on a common 2 s grid, time-zeroed at induction.

    MAP and PSi live on different rows (offset ~1 s), so each signal is
    resampled independently (mean within each 2 s bin), then joined.
    `psi_shift_sec` > 0 moves PSi EARLIER to compensate monitor lag.
    """
    sec = to_seconds_of_day(g["time"])
    out = {}
    for col, shift in ((MAP_COL, 0.0), (PSI_COL, -psi_shift_sec)):
        vals = pd.to_numeric(g[col], errors="coerce")
        s = pd.Series(vals.values, index=pd.to_timedelta(sec + shift, unit="s")).dropna()
        if col == MAP_COL:
            s = s[(s >= MAP_MIN_PLAUSIBLE) & (s <= MAP_MAX_PLAUSIBLE)]
        out[col] = s.resample(f"{GRID_SECONDS}s").mean()
    df = pd.concat(out, axis=1)
    df["t_sec"] = df.index.total_seconds() - induction_sec  # 0 = induction
    return df.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Per-patient metrics (the plan's primary numbers)
# --------------------------------------------------------------------------- #

def patient_metrics(df: pd.DataFrame, events: dict, induction_sec: float) -> dict:
    """Baseline, post-induction nadir, % change, PSi depth, hypotension burden."""
    base_start = to_seconds_of_day(pd.Series([events.get("sqi_time") or
                                              events.get("preop_start")])).iloc[0]
    base_rel = (base_start - induction_sec) if not pd.isna(base_start) else -POST_WINDOW_MIN * 60

    pre = df[(df["t_sec"] >= base_rel) & (df["t_sec"] < 0)]
    post = df[(df["t_sec"] >= 0) & (df["t_sec"] <= POST_WINDOW_MIN * 60)].copy()

    baseline = pre[MAP_COL].median()
    # Nadir from a ~10 s rolling median, so a single artefact sample (art-line
    # flush / blood draw) can't masquerade as the true lowest MAP.
    smooth_n = max(1, int(10 / GRID_SECONDS))
    post["map_smooth"] = post[MAP_COL].rolling(smooth_n, center=True,
                                               min_periods=1).median()
    nadir = post["map_smooth"].min()
    nadir_t = (post.loc[post["map_smooth"].idxmin(), "t_sec"]
               if post["map_smooth"].notna().any() else np.nan)
    pct_change = (nadir - baseline) / baseline * 100 if baseline and not np.isnan(baseline) else np.nan

    psi_at_nadir = np.nan
    if not np.isnan(nadir_t):
        near = post[(post["t_sec"] >= nadir_t - 30) & (post["t_sec"] <= nadir_t + 30)]
        psi_at_nadir = near[PSI_COL].median()

    below = post[post[MAP_COL] < HYPOTENSION_MAP]
    return {
        "baseline_MAP": round(float(baseline), 1) if not np.isnan(baseline) else np.nan,
        "nadir_MAP_post": round(float(nadir), 1) if not np.isnan(nadir) else np.nan,
        "nadir_time_min": round(float(nadir_t) / 60, 2) if not np.isnan(nadir_t) else np.nan,
        "MAP_pct_change": round(float(pct_change), 1) if not np.isnan(pct_change) else np.nan,
        "PSi_baseline": round(float(pre[PSI_COL].median()), 1) if pre[PSI_COL].notna().any() else np.nan,
        "PSi_min_post": round(float(post[PSI_COL].min()), 1) if post[PSI_COL].notna().any() else np.nan,
        "PSi_at_MAP_nadir": round(float(psi_at_nadir), 1) if not np.isnan(psi_at_nadir) else np.nan,
        f"min_below_MAP{HYPOTENSION_MAP}": round(len(below) * GRID_SECONDS / 60, 2),
    }


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #

def fig_group_median_iqr(aligned: dict, summaries: pd.DataFrame, outdir: str,
                         psi_shift: float) -> str:
    """Median + IQR at BIN_SECONDS intervals across patients, aligned to induction.

    Top: MAP as % of each patient's preop baseline (so patients are comparable).
    Bottom: PSi.
    """
    frames = []
    for pid, df in aligned.items():
        d = df[(df["t_sec"] >= -PLOT_PRE_MIN * 60) & (df["t_sec"] <= PLOT_POST_MIN * 60)].copy()
        base = summaries.loc[summaries["patientID"] == pid, "baseline_MAP"]
        base = float(base.iloc[0]) if len(base) and not np.isnan(base.iloc[0]) else np.nan
        d["map_pct"] = d[MAP_COL] / base * 100 if base and not np.isnan(base) else np.nan
        d["bin"] = (d["t_sec"] // BIN_SECONDS) * BIN_SECONDS
        frames.append(d)
    allp = pd.concat(frames)

    stats = allp.groupby("bin").agg(
        map_med=("map_pct", "median"), map_q1=("map_pct", lambda x: x.quantile(.25)),
        map_q3=("map_pct", lambda x: x.quantile(.75)),
        psi_med=(PSI_COL, "median"), psi_q1=(PSI_COL, lambda x: x.quantile(.25)),
        psi_q3=(PSI_COL, lambda x: x.quantile(.75)),
    ).reset_index()
    tmin = stats["bin"] / 60

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    ax1.fill_between(tmin, stats["map_q1"], stats["map_q3"], alpha=.3, color="#d62728", label="IQR")
    ax1.plot(tmin, stats["map_med"], color="#d62728", lw=1.6, label="Median")
    ax1.axhline(100, color="gray", ls=":", lw=1)
    ax1.axvline(0, color="black", ls="--", lw=1.2)
    ax1.set_ylabel("MAP (% of preop baseline)")
    ax1.set_title(f"MAP and PSi aligned to induction — median + IQR per {BIN_SECONDS}s bin"
                  + (f"  [PSi shifted {psi_shift:.0f}s earlier]" if psi_shift else ""))
    ax1.legend(loc="upper right")

    ax2.fill_between(tmin, stats["psi_q1"], stats["psi_q3"], alpha=.3, color="#1f77b4", label="IQR")
    ax2.plot(tmin, stats["psi_med"], color="#1f77b4", lw=1.6, label="Median")
    ax2.axhspan(25, 50, alpha=.08, color="green")  # typical target sedation band
    ax2.axvline(0, color="black", ls="--", lw=1.2)
    ax2.set_ylabel("PSi")
    ax2.set_xlabel("Time from induction (min)")
    ax2.legend(loc="upper right")

    fig.tight_layout()
    path = os.path.join(outdir, "fig1_group_median_iqr.png")
    fig.savefig(path, dpi=140); plt.close(fig)
    return path


def fig_per_patient(aligned: dict, summaries: pd.DataFrame, outdir: str) -> list[str]:
    """One figure per patient: MAP + PSi vs time from induction, baseline band,
    nadir marker, hypotension line."""
    paths = []
    for pid, df in aligned.items():
        d = df[(df["t_sec"] >= -PLOT_PRE_MIN * 60) & (df["t_sec"] <= PLOT_POST_MIN * 60)]
        row = summaries[summaries["patientID"] == pid].iloc[0]

        fig, ax = plt.subplots(figsize=(12, 5))
        axb = ax.twinx()
        ax.plot(d["t_sec"] / 60, d[MAP_COL], color="#d62728", lw=1, label="MAP")
        axb.plot(d["t_sec"] / 60, d[PSI_COL], color="#1f77b4", lw=1, label="PSi")
        ax.axvline(0, color="black", ls="--", lw=1.2)
        ax.axhline(HYPOTENSION_MAP, color="#d62728", ls=":", lw=1)
        if not np.isnan(row["baseline_MAP"]):
            ax.axhline(row["baseline_MAP"], color="#d62728", ls="-.", lw=.8, alpha=.6)
        if not np.isnan(row["nadir_time_min"]):
            ax.plot(row["nadir_time_min"], row["nadir_MAP_post"], "v", color="black", ms=9,
                    label=f"nadir {row['nadir_MAP_post']:.0f} ({row['MAP_pct_change']:+.0f}%)")
        ax.axvspan(0, POST_WINDOW_MIN, alpha=.05, color="orange")
        ax.set_xlabel("Time from induction (min)")
        ax.set_ylabel("MAP (mmHg)", color="#d62728")
        axb.set_ylabel("PSi", color="#1f77b4")
        ax.set_title(f"{pid} — baseline {row['baseline_MAP']:.0f} mmHg, "
                     f"nadir {row['nadir_MAP_post']:.0f} mmHg ({row['MAP_pct_change']:+.0f}%), "
                     f"PSi at nadir {row['PSi_at_MAP_nadir']}")
        h1, l1 = ax.get_legend_handles_labels(); h2, l2 = axb.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=8)
        fig.tight_layout()
        p = os.path.join(outdir, f"fig2_patient_{pid}.png")
        fig.savefig(p, dpi=140); plt.close(fig)
        paths.append(p)
    return paths


def fig_depth_vs_map_drop(summaries: pd.DataFrame, outdir: str) -> str:
    """THE plan's primary plot: sedation depth (PSi, x) vs % MAP change (y).

    One point per patient. Two depth definitions shown side by side:
    minimum PSi post-induction, and PSi around the MAP nadir.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, xcol, label in (
        (axes[0], "PSi_min_post", f"Min PSi in first {POST_WINDOW_MIN} min"),
        (axes[1], "PSi_at_MAP_nadir", "PSi at MAP nadir (±30 s)"),
    ):
        d = summaries.dropna(subset=[xcol, "MAP_pct_change"])
        ax.scatter(d[xcol], d["MAP_pct_change"], s=60, color="#2ca02c", zorder=3)
        for _, r in d.iterrows():
            ax.annotate(r["patientID"], (r[xcol], r["MAP_pct_change"]),
                        textcoords="offset points", xytext=(6, 4), fontsize=8)
        if len(d) >= 3:  # fit only when there are enough points to mean anything
            coef = np.polyfit(d[xcol], d["MAP_pct_change"], 1)
            xs = np.linspace(d[xcol].min(), d[xcol].max(), 50)
            ax.plot(xs, np.polyval(coef, xs), "k--", lw=1)
        ax.axhline(0, color="gray", lw=.8)
        ax.axhline(-20, color="#d62728", ls=":", lw=1)  # common "significant drop" ref
        ax.set_xlabel(label)
        ax.set_ylabel("MAP change from baseline (%)")
    fig.suptitle("Sedation depth vs post-induction MAP drop (one point per patient)")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    path = os.path.join(outdir, "fig3_depth_vs_map_drop.png")
    fig.savefig(path, dpi=140); plt.close(fig)
    return path


def fig_lag_sensitivity(raw: pd.DataFrame, events: dict, outdir: str) -> str:
    """Within-patient MAP~PSi correlation (first POST_WINDOW_MIN after induction)
    with PSi shifted earlier by 0 / 10 / 20 s — the plan's lag check."""
    shifts = [0, 10, 20]
    rows = []
    for pid, g in raw.groupby("patientID"):
        ev = events.get(pid)
        if not ev:
            continue
        ind = to_seconds_of_day(pd.Series([ev["induction_time"]])).iloc[0]
        for sh in shifts:
            df = align_patient(g, ind, sh)
            w = df[(df["t_sec"] >= 0) & (df["t_sec"] <= POST_WINDOW_MIN * 60)].dropna(
                subset=[MAP_COL, PSI_COL])
            r = w[MAP_COL].corr(w[PSI_COL]) if len(w) > 5 else np.nan
            rows.append({"patientID": pid, "shift_s": sh, "pearson_r": r, "n": len(w)})
    sens = pd.DataFrame(rows)
    sens.to_csv(os.path.join(outdir, "lag_sensitivity.csv"), index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    for pid, g in sens.groupby("patientID"):
        ax.plot(g["shift_s"], g["pearson_r"], "o-", label=pid)
    ax.axhline(0, color="gray", lw=.8)
    ax.set_xticks(shifts)
    ax.set_xlabel("PSi shifted earlier by (s)")
    ax.set_ylabel(f"Pearson r, MAP vs PSi (0–{POST_WINDOW_MIN} min post-induction)")
    ax.set_title("Sensitivity of MAP–PSi correlation to Sedline lag correction")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(outdir, "fig4_lag_sensitivity.png")
    fig.savefig(path, dpi=140); plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Induction-aligned MAP vs PSi analysis.")
    ap.add_argument("--data", default=None,
                    help="Long-format data file (xlsx or csv): patientID,time,"
                         f"{MAP_COL},{PSI_COL}. Alternative to --btb/--psi.")
    ap.add_argument("--btb", default=None,
                    help="Master csv listing each patient's BTB MAP file path "
                         "(one per line). Use together with --psi.")
    ap.add_argument("--psi", default=None,
                    help="Master csv listing each patient's Sedline file path "
                         "(one per line). Use together with --btb.")
    ap.add_argument("--events", default=None,
                    help="csv with patientID,induction_time,preop_start,sqi_time "
                         "(defaults to the two sample patients).")
    ap.add_argument("--outdir", default="induction_output")
    ap.add_argument("--psi-shift", type=float, default=0.0,
                    help="Shift PSi this many seconds EARLIER (lag correction) "
                         "for the main figures. Fig 4 sweeps 0/10/20 regardless.")
    args = ap.parse_args(argv)

    os.makedirs(args.outdir, exist_ok=True)
    if args.btb and args.psi:
        raw = load_from_masters(args.btb, args.psi)
        print(f"Loaded {raw['patientID'].nunique()} patients from master lists "
              f"({raw[MAP_COL].notna().sum()} MAP, {raw[PSI_COL].notna().sum()} PSi values)")
    elif args.data:
        raw = load_long_data(args.data)
    else:
        ap.error("provide either --data, or both --btb and --psi")
    events = load_events(args.events)

    aligned, met_rows = {}, []
    for pid, g in raw.groupby("patientID"):
        ev = events.get(pid)
        if not ev:
            sys.stderr.write(f"[WARN] no event times for {pid} — skipped. "
                             f"Add them to the --events csv.\n")
            continue
        ind = to_seconds_of_day(pd.Series([ev["induction_time"]])).iloc[0]
        df = align_patient(g, ind, args.psi_shift)
        aligned[pid] = df
        met_rows.append({"patientID": pid, **patient_metrics(df, ev, ind)})

    if not aligned:
        sys.stderr.write("No patients processed — check --events.\n")
        return 1

    summaries = pd.DataFrame(met_rows)
    spath = os.path.join(args.outdir, "induction_summary.csv")
    summaries.to_csv(spath, index=False)
    print(summaries.to_string(index=False))

    p1 = fig_group_median_iqr(aligned, summaries, args.outdir, args.psi_shift)
    p2 = fig_per_patient(aligned, summaries, args.outdir)
    p3 = fig_depth_vs_map_drop(summaries, args.outdir)
    p4 = fig_lag_sensitivity(raw, events, args.outdir)

    print("\nOutputs:")
    for p in [spath, p1, *p2, p3, p4]:
        print(" ", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
