#!/usr/bin/env python3
"""
Induction-aligned MAP vs PSi vs cerebral oximetry (SctO2) analysis.

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
  * EPOCH SCATTERS:  cerebral oximetry (SctO2, y) vs MAP (x), one panel per
                     surgical epoch, with a LOWESS smoother -- the five
                     scatter plots requested by the PI (see EPOCHS below).

Input format (long / tidy - e.g. Sample_1.xlsx or the combined csv):

    patientID , time , meanArterialPressure , PSi

  `time` may be HH:MM:SS or a full datetime. MAP and PSi do NOT need to share
  rows (in the export they are offset by ~1 s); each signal is resampled onto
  a common 2 s grid before they are compared.

Cerebral oximetry is loaded separately with --sto2, a master csv listing one
StO2 file path per line (columns: Time, valid_CH1..4, StO2_CH1..4).

Event times (induction, OR entry, etc.) come from the hard-coded REDCap export.

Usage
-----
    python induction_map_psi_analysis.py --data Sample_1.xlsx
    python induction_map_psi_analysis.py --data combined.csv
    python induction_map_psi_analysis.py --data combined.csv --sto2 sto2_filepaths.csv
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

# ---- Cerebral oximetry (SctO2) -------------------------------------------- #
# Which StO2 channels are the LEFT and RIGHT cerebral (forehead) sensors.
# !! VERIFY against the montage used in this study before trusting the labels;
#    CH3/CH4 are often somatic rather than cerebral.
STO2_LEFT_CH = "StO2_CH1"
STO2_RIGHT_CH = "StO2_CH2"
STO2_L, STO2_R, STO2_MEAN = "SctO2_L", "SctO2_R", "SctO2_mean"
STO2_MIN_PLAUSIBLE = 10.0   # % -- the exports use -1 for "no reading"
STO2_MAX_PLAUSIBLE = 100.0

# Signals resampled onto the common grid (order matters only for readability).
VALUE_COLS = [MAP_COL, PSI_COL, STO2_L, STO2_R, STO2_MEAN]

# ---- The five epochs requested by the PI ---------------------------------- #
# "anchor" says what the window is measured from:
#   preop      -> preop baseline start .. end of preop monitoring / OR entry
#   or_pre     -> OR entry .. induction
#   induction  -> start/end minutes relative to induction (TIME = 0)
EPOCHS = [
    {"key": "preop",       "label": "1. Pre-op (no PSi)",         "anchor": "preop"},
    {"key": "or_preind",   "label": "2. OR pre-induction",        "anchor": "or_pre"},
    {"key": "post_0_5",    "label": "3. Post-induction 0-5 min",  "anchor": "induction",
     "start": 0, "end": 5},
    {"key": "post_5_60",   "label": "4. Post-induction 5-60 min", "anchor": "induction",
     "start": 5, "end": 60},
    {"key": "post_10_20",  "label": "5. Post-induction 10-20 min", "anchor": "induction",
     "start": 10, "end": 20},
]
OR_PRE_FALLBACK_MIN = 10   # if OR-entry time is missing, use this many min pre-induction
LOWESS_FRAC = 0.65         # smoother span for the epoch scatters

EVENTS_PATH = "/N/project/Analgesia_BDproject/data/00_raw/BDFILES/REDCap/6.16.26.FIXED-TYPOS-BDPostInductionHemod_DATA_LABELS_2026-06-16_1657.csv"

REDCAP_ID_ALIASES = {
    "IUMH202601601": "IUMH2026011601",
    "IUMH2026010601-20260105": "IUMH2026010501",
}

# --------------------------------------------------------------------------- #
# Loading & alignment
# --------------------------------------------------------------------------- #

def to_seconds_of_day(series: pd.Series) -> pd.Series:
    """Parse HH:MM:SS strings / datetimes into seconds since midnight.

    Vectorized: a regex fast path handles the HH:MM:SS forms (including the
    time half of a full datetime), and only the leftovers go through
    pd.to_datetime. Parsing element-by-element is far too slow at cohort scale.
    """
    text = series.astype(str).str.strip()
    out = pd.Series(np.nan, index=series.index, dtype=float)

    parts = text.str.extract(r"(\d{1,2}):(\d{2}):(\d{2})")
    hit = parts.notna().all(axis=1)
    if hit.any():
        p = parts[hit].astype(float)
        out.loc[hit] = p[0] * 3600 + p[1] * 60 + p[2]

    rest = ~hit & series.notna()
    if rest.any():
        dt = pd.to_datetime(text[rest], errors="coerce")
        out.loc[rest] = (dt.dt.hour * 3600 + dt.dt.minute * 60
                         + dt.dt.second).astype(float)
    return out


def load_long_data(path: str) -> pd.DataFrame:
    """Load the long-format data file (xlsx or csv)."""
    if path.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)
    if "patientID" not in df.columns and "ID" in df.columns:
        df = df.rename(columns={"ID": "patientID"})
    missing = {"patientID", "time", MAP_COL, PSI_COL} - set(df.columns)
    if missing:
        raise ValueError(f"Data file is missing columns: {missing}")
    return df


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


def load_sto2_from_master(master: str) -> pd.DataFrame:
    """Build a long patientID/time/SctO2_L/SctO2_R/SctO2_mean table.

    Each StO2 export has Time, valid_CH1..4 and StO2_CH1..4. A channel is used
    only where its valid flag is 1 and the reading is physiologically possible
    (the exports write -1 when the sensor has no signal).
    """
    frames = []
    for p in read_filepath_list(master):
        pid = patient_id_from_path(p)
        if pid is None:
            sys.stderr.write(f"[WARN] no 'IU' id in path, skipped: {p}\n")
            continue
        try:
            df = pd.read_csv(p, skipinitialspace=True)
            df.columns = df.columns.str.strip()
            out = pd.DataFrame({"patientID": pid,
                                "time": df["Time"].astype(str).str[-8:]})
            for ch, dest in ((STO2_LEFT_CH, STO2_L), (STO2_RIGHT_CH, STO2_R)):
                if ch not in df.columns:
                    out[dest] = np.nan
                    continue
                vals = pd.to_numeric(df[ch], errors="coerce")
                flag_col = ch.replace("StO2_", "valid_")
                if flag_col in df.columns:
                    ok = pd.to_numeric(df[flag_col], errors="coerce") == 1
                    vals = vals.where(ok)
                out[dest] = vals.where(vals.between(STO2_MIN_PLAUSIBLE,
                                                    STO2_MAX_PLAUSIBLE))
            out[STO2_MEAN] = out[[STO2_L, STO2_R]].mean(axis=1)
            frames.append(out.dropna(subset=[STO2_L, STO2_R, STO2_MEAN],
                                     how="all"))
        except Exception as exc:
            sys.stderr.write(f"[WARN] {pid} (StO2): {exc} - skipped\n")

    if not frames:
        raise ValueError(f"Nothing loaded from the StO2 master list: {master}")
    return pd.concat(frames, ignore_index=True)


def load_events() -> dict:
    """Load event times from the hard-coded REDCap export."""
    ev = pd.read_csv(EVENTS_PATH)

    def find_column(*groups: tuple[str, ...], required: bool = False) -> str | None:
        normalized = {column: str(column).lower().replace("_", " ")
                      for column in ev.columns}
        for group in groups:
            for column, label in normalized.items():
                if all(term in label for term in group):
                    return column
        if required:
            raise ValueError(f"Could not find REDCap event column matching {groups}")
        return None

    patient_column = next(
        (column for column in ev.columns
         if ev[column].astype("string").str.strip()
         .str.match(r"^IU(?:MH|UH)\d+$", na=False).any()),
        None,
    )
    if patient_column is None:
        patient_column = find_column(("study", "id"), ("subject", "id"),
                                     ("patient", "id"), required=True)
    induction_column = find_column(("induction", "time"), required=True)
    preop_column = find_column(("preop", "time"), ("monitoring", "start"))
    sqi_column = find_column(("sqi", "time"), ("sqi",))
    # Epoch boundaries: OR entry separates "pre-op" from "OR pre-induction",
    # and preop monitoring end closes the pre-op epoch.
    or_entry_column = find_column(("enter", "or"), ("entered", "or"))
    preop_end_column = find_column(("monitoring ended in preop",),
                                   ("ended", "preop"))

    out = {}
    for _, r in ev.iterrows():
        patient_id = (str(r[patient_column]).strip().upper()
                      .replace("_", "").replace("-", ""))
        if not patient_id.startswith(("IUMH", "IUUH")):
            continue
        surgery_date = str(r.get("Date of Surgery", "")).strip().replace("-", "")
        alias_key = f"{patient_id}-{surgery_date}"
        patient_id = REDCAP_ID_ALIASES.get(alias_key,
                                           REDCAP_ID_ALIASES.get(patient_id,
                                                                 patient_id))

        def get(col):
            return str(r[col]).strip() if col else ""

        out[patient_id] = {
            "induction_time": str(r[induction_column]).strip(),
            "preop_start": get(preop_column),
            "sqi_time": get(sqi_column),
            "or_entry": get(or_entry_column),
            "preop_end": get(preop_end_column),
        }
    return out


def align_patient(g: pd.DataFrame, induction_sec: float,
                  psi_shift_sec: float) -> pd.DataFrame:
    """Return one patient's signals on a common 2 s grid, time-zeroed at induction.

    MAP, PSi and SctO2 live on different rows (different devices, offset by
    ~1 s), so each signal is resampled independently (mean within each 2 s bin)
    and then joined. `psi_shift_sec` > 0 moves PSi EARLIER to compensate for
    monitor lag.
    """
    sec = to_seconds_of_day(g["time"])
    out = {}
    for col in VALUE_COLS:
        if col not in g.columns:
            continue
        shift = -psi_shift_sec if col == PSI_COL else 0.0
        vals = pd.to_numeric(g[col], errors="coerce")
        valid = sec.notna() & vals.notna() & np.isfinite(vals)
        s = pd.Series(vals[valid].values,
                      index=pd.to_timedelta(sec[valid] + shift, unit="s"))
        if col == MAP_COL:
            s = s[(s >= MAP_MIN_PLAUSIBLE) & (s <= MAP_MAX_PLAUSIBLE)]
        if s.empty:
            continue
        out[col] = s.resample(f"{GRID_SECONDS}s").mean()
    if not out:
        return pd.DataFrame(columns=VALUE_COLS + ["t_sec"])
    df = pd.concat(out, axis=1)
    for col in VALUE_COLS:                      # keep the schema stable
        if col not in df.columns:
            df[col] = np.nan
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

    pre = df[(df["t_sec"] >= base_rel) & (df["t_sec"] < 0)].copy()
    post = df[(df["t_sec"] >= 0) & (df["t_sec"] <= POST_WINDOW_MIN * 60)].copy()
    pre_map = pre[MAP_COL].dropna()
    post_map = post[MAP_COL].dropna()

    baseline = pre_map.median()
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

    below = post_map[post_map < HYPOTENSION_MAP]
    return {
        "baseline_MAP": round(float(baseline), 1) if not np.isnan(baseline) else np.nan,
        "nadir_MAP_post": round(float(nadir), 1) if not np.isnan(nadir) else np.nan,
        "nadir_time_min": round(float(nadir_t) / 60, 2) if not np.isnan(nadir_t) else np.nan,
        "MAP_pct_change": round(float(pct_change), 1) if not np.isnan(pct_change) else np.nan,
        "PSi_baseline": round(float(pre[PSI_COL].dropna().median()), 1) if pre[PSI_COL].notna().any() else np.nan,
        "PSi_min_post": round(float(post[PSI_COL].dropna().min()), 1) if post[PSI_COL].notna().any() else np.nan,
        "PSi_at_MAP_nadir": round(float(psi_at_nadir), 1) if not np.isnan(psi_at_nadir) else np.nan,
        f"min_below_MAP{HYPOTENSION_MAP}": round(len(below) * GRID_SECONDS / 60, 2),
    }


# --------------------------------------------------------------------------- #
# Epoch windows + per-epoch table (for the SctO2 vs MAP scatters)
# --------------------------------------------------------------------------- #

def epoch_window(epoch: dict, events: dict, induction_sec: float) -> tuple[float, float]:
    """Return (start_sec, end_sec) for one epoch, relative to induction."""
    def rel(key):
        v = events.get(key)
        s = to_seconds_of_day(pd.Series([v])).iloc[0] if v else np.nan
        return np.nan if pd.isna(s) else s - induction_sec

    if epoch["anchor"] == "induction":
        return epoch["start"] * 60.0, epoch["end"] * 60.0

    if epoch["anchor"] == "preop":
        start = rel("sqi_time")
        if np.isnan(start):
            start = rel("preop_start")
        end = rel("preop_end")
        if np.isnan(end):
            end = rel("or_entry")
        if np.isnan(start) or np.isnan(end) or end <= start:
            return np.nan, np.nan
        return start, end

    # "or_pre": OR entry -> induction
    start = rel("or_entry")
    if np.isnan(start):
        start = -OR_PRE_FALLBACK_MIN * 60.0
    return start, 0.0


def build_epoch_table(aligned: dict, events_by_pid: dict, induction_by_pid: dict,
                      mode: str = "patient") -> pd.DataFrame:
    """One row per patient per epoch (mode='patient') or per grid sample
    (mode='sample'), carrying MAP, SctO2 and PSi for the epoch scatters."""
    rows = []
    for pid, df in aligned.items():
        ev, ind = events_by_pid[pid], induction_by_pid[pid]
        for epoch in EPOCHS:
            lo, hi = epoch_window(epoch, ev, ind)
            if np.isnan(lo) or np.isnan(hi):
                continue
            w = df[(df["t_sec"] >= lo) & (df["t_sec"] <= hi)]
            if w.empty:
                continue
            if mode == "sample":
                keep = w.dropna(subset=[MAP_COL, STO2_MEAN])[
                    ["t_sec", MAP_COL, STO2_L, STO2_R, STO2_MEAN, PSI_COL]].copy()
                if keep.empty:
                    continue
                keep.insert(0, "epoch_label", epoch["label"])
                keep.insert(0, "epoch", epoch["key"])
                keep.insert(0, "patientID", pid)
                keep["t_min"] = keep.pop("t_sec") / 60
                rows.append(keep)                      # frame, concatenated below
            else:
                rows.append({
                    "patientID": pid, "epoch": epoch["key"],
                    "epoch_label": epoch["label"],
                    "window_start_min": round(lo / 60, 2),
                    "window_end_min": round(hi / 60, 2),
                    MAP_COL: w[MAP_COL].mean(),
                    STO2_L: w[STO2_L].mean(),
                    STO2_R: w[STO2_R].mean(),
                    STO2_MEAN: w[STO2_MEAN].mean(),
                    PSI_COL: w[PSI_COL].mean(),
                    "n_samples": int(w[MAP_COL].notna().sum()),
                })
    if not rows:
        return pd.DataFrame()
    if mode == "sample":
        return pd.concat(rows, ignore_index=True)
    return pd.DataFrame(rows)


LOWESS_MAX_POINTS = 2000   # LOWESS is ~O(n^2); above this use binned medians


def _smooth_fit(x: np.ndarray, y: np.ndarray, frac: float = LOWESS_FRAC):
    """Trend line for a scatter.

    LOWESS for modest n; for the sample-level scatters (tens of thousands of
    points) that is far too slow, so bin by MAP decile-ish bins and join the
    medians -- which is also the more honest summary at that density.
    """
    order = np.argsort(x)
    x, y = np.asarray(x)[order], np.asarray(y)[order]
    if len(x) < 5:
        return None, None

    if len(x) > LOWESS_MAX_POINTS:
        nbins = 30
        edges = np.quantile(x, np.linspace(0, 1, nbins + 1))
        edges = np.unique(edges)
        if len(edges) < 3:
            return None, None
        idx = np.clip(np.digitize(x, edges[1:-1]), 0, len(edges) - 2)
        med_x = np.array([np.median(x[idx == b]) for b in range(len(edges) - 1)
                          if np.any(idx == b)])
        med_y = np.array([np.median(y[idx == b]) for b in range(len(edges) - 1)
                          if np.any(idx == b)])
        return med_x, med_y

    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess
        fit = lowess(y, x, frac=frac, return_sorted=True)
        return fit[:, 0], fit[:, 1]
    except Exception:
        win = max(5, len(x) // 4)
        ys = pd.Series(y).rolling(win, center=True, min_periods=2).median()
        return x, ys.to_numpy()


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
    ax1.fill_between(tmin, stats["map_q1"], stats["map_q3"], alpha=.3, color="gray", label="IQR")
    ax1.plot(tmin, stats["map_med"], color="#d62728", lw=1.6, label="Median")
    ax1.axhline(100, color="gray", ls=":", lw=1)
    ax1.axvline(0, color="black", ls="--", lw=1.2)
    ax1.set_ylabel("MAP (% of preop baseline)")
    ax1.set_title(f"MAP and PSi aligned to induction - median + IQR per {BIN_SECONDS}s bin"
                  + (f"  [PSi shifted {psi_shift:.0f}s earlier]" if psi_shift else ""))
    ax1.legend(loc="upper right")

    ax2.fill_between(tmin, stats["psi_q1"], stats["psi_q3"], alpha=.3, color="gray", label="IQR")
    ax2.plot(tmin, stats["psi_med"], color="#1f77b4", lw=1.6, label="Median")
    ax2.axvline(0, color="black", ls="--", lw=1.2)
    ax2.set_ylabel("PSi")
    ax2.set_xlabel("Time from induction (min)")
    ax2.legend(loc="upper right")

    fig.tight_layout()
    path = os.path.abspath(os.path.join(outdir, "fig1_group_median_iqr.png"))
    fig.savefig(path, format="png", dpi=140)
    plt.close(fig)
    if not os.path.isfile(path) or os.path.getsize(path) == 0:
        raise IOError(f"Figure was not written correctly: {path}")
    return path


def fig_depth_vs_map_drop(summaries: pd.DataFrame, outdir: str) -> str:
    """Plot sedation depth against percentage MAP change."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, xcol, label in (
        (axes[0], "PSi_min_post", f"Min PSi in first {POST_WINDOW_MIN} min"),
        (axes[1], "PSi_at_MAP_nadir", "PSi at MAP nadir (+/-30 s)"),
    ):
        data = summaries.dropna(subset=[xcol, "MAP_pct_change"])
        ax.scatter(data[xcol], data["MAP_pct_change"],
                   s=42, color="#2ca02c", alpha=0.72,
                   edgecolors="white", linewidths=0.6, zorder=3)
        if len(data) >= 3:
            coef = np.polyfit(data[xcol], data["MAP_pct_change"], 1)
            xs = np.linspace(data[xcol].min(), data[xcol].max(), 50)
            ax.plot(xs, np.polyval(coef, xs), "k--", lw=1)
        ax.axhline(0, color="gray", lw=.8)
        ax.axhline(-20, color="#d62728", ls=":", lw=1)
        ax.grid(True, color="#d9d9d9", linewidth=.6, alpha=.7)
        ax.set_axisbelow(True)
        ax.text(0.02, 0.04, f"n = {len(data)}", transform=ax.transAxes,
                fontsize=9, color="#555555")
        ax.set_xlabel(label)
        ax.set_ylabel("MAP change from baseline (%)")
    fig.suptitle("Sedation depth vs post-induction MAP drop")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    path = os.path.abspath(os.path.join(outdir, "fig3_depth_vs_map_drop.png"))
    fig.savefig(path, format="png", dpi=140)
    plt.close(fig)
    return path


def fig_lag_sensitivity(raw: pd.DataFrame, events: dict, outdir: str) -> str:
    """Plot MAP/PSi correlation after shifting PSi earlier by 0/10/20 s."""
    shifts = [0, 10, 20]
    rows = []
    for pid, group in raw.groupby("patientID"):
        event = events.get(pid)
        if not event:
            continue
        induction = to_seconds_of_day(
            pd.Series([event["induction_time"]])).iloc[0]
        if pd.isna(induction):
            continue
        for shift in shifts:
            df = align_patient(group, induction, shift)
            window = df[(df["t_sec"] >= 0) &
                        (df["t_sec"] <= POST_WINDOW_MIN * 60)].dropna(
                            subset=[MAP_COL, PSI_COL])
            correlation = (window[MAP_COL].corr(window[PSI_COL])
                           if len(window) > 5 else np.nan)
            rows.append({"patientID": pid, "shift_s": shift,
                         "pearson_r": correlation, "n": len(window)})

    sensitivity = pd.DataFrame(rows)
    sensitivity.to_csv(os.path.join(outdir, "lag_sensitivity.csv"), index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    for pid, group in sensitivity.groupby("patientID"):
        valid = group.dropna(subset=["pearson_r"])
        if not valid.empty:
            ax.plot(valid["shift_s"], valid["pearson_r"], "-",
                    color="gray", alpha=.35, lw=.8)
    med = sensitivity.groupby("shift_s")["pearson_r"].median()
    ax.plot(med.index, med.values, "o-", color="#d62728", lw=2.5,
            label="Median across patients", zorder=5)
    ax.axhline(0, color="gray", lw=.8)
    ax.set_xticks(shifts)
    ax.set_xlabel("PSi shifted earlier by (s)")
    ax.set_ylabel(f"Pearson r, MAP vs PSi (0-{POST_WINDOW_MIN} min post-induction)")
    ax.set_title("Sensitivity of MAP-PSi correlation to Sedline lag correction")
    ax.legend()
    fig.tight_layout()
    path = os.path.abspath(os.path.join(outdir, "fig4_lag_sensitivity.png"))
    fig.savefig(path, format="png", dpi=140)
    plt.close(fig)
    return path


def fig_sto2_vs_map_epochs(epochs_df: pd.DataFrame, outdir: str,
                           mode: str = "patient") -> str:
    """The five requested scatters: cerebral SctO2 (y) vs MAP (x), one panel
    per epoch, each with a LOWESS smoother."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.ravel()

    for ax, epoch in zip(axes, EPOCHS):
        d = epochs_df[epochs_df["epoch"] == epoch["key"]].dropna(
            subset=[MAP_COL, STO2_MEAN])
        ax.scatter(d[MAP_COL], d[STO2_MEAN], s=34 if mode == "patient" else 6,
                   color="#2ca02c", alpha=.7 if mode == "patient" else .15,
                   edgecolors="white" if mode == "patient" else "none",
                   linewidths=.5, zorder=3)
        xs, ys = _smooth_fit(d[MAP_COL].to_numpy(), d[STO2_MEAN].to_numpy())
        if xs is not None:
            ax.plot(xs, ys, color="#0b7a5d", lw=2.4, zorder=4)

        r = d[MAP_COL].corr(d[STO2_MEAN]) if len(d) > 2 else np.nan
        note = f"n = {len(d)}" + ("" if np.isnan(r) else f"\nr = {r:.2f}")
        ax.text(.02, .04, note, transform=ax.transAxes, fontsize=9,
                color="#555555", va="bottom")
        ax.axvline(HYPOTENSION_MAP, color="#d62728", ls=":", lw=1)
        ax.grid(True, color="#d9d9d9", lw=.6, alpha=.7)
        ax.set_axisbelow(True)
        ax.set_title(epoch["label"], fontsize=11)
        ax.set_xlabel("MAP (mmHg)")
        ax.set_ylabel("Cerebral SctO2, R+L mean (%)")

    axes[len(EPOCHS)].axis("off")
    unit = "one point per patient (epoch mean)" if mode == "patient" \
        else "one point per 2-s sample"
    fig.suptitle(f"Cerebral oximetry vs MAP by epoch - {unit}", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.abspath(os.path.join(outdir, "fig5_sto2_vs_map_epochs.png"))
    fig.savefig(path, format="png", dpi=140)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #

def main(argv=None) -> int:
    default_data_path = "/N/project/Analgesia_BDproject/PR/scripts_PR/MASTER_combined.csv"

    ap = argparse.ArgumentParser(description="Induction-aligned MAP vs PSi analysis.")
    ap.add_argument("--data", default=default_data_path,
                    help="Long-format data file (xlsx or csv): patientID,time,"
                         f"{MAP_COL},{PSI_COL}.")
    ap.add_argument("--sto2", default=None,
                    help="Master csv listing one StO2 file path per line. "
                         "Enables the 5 epoch scatters (figure 5).")
    ap.add_argument("--epoch-points", choices=["patient", "sample"],
                    default="patient",
                    help="Epoch scatters: one point per patient (epoch mean, "
                         "default) or one point per 2-s sample.")
    ap.add_argument("--outdir", default="induction_output")
    ap.add_argument("--psi-shift", type=float, default=0.0,
                    help="Shift PSi this many seconds EARLIER (lag correction) "
                         "for the main figures. Fig 4 sweeps 0/10/20 regardless.")
    args = ap.parse_args(argv)

    os.makedirs(args.outdir, exist_ok=True)
    raw = load_long_data(args.data)
    events = load_events()

    if args.sto2:
        sto2 = load_sto2_from_master(args.sto2)
        raw = pd.concat([raw, sto2], ignore_index=True)
        n_pts = sto2["patientID"].nunique()
        print(f"Loaded SctO2 for {n_pts} patient(s), "
              f"{sto2[STO2_MEAN].notna().sum()} valid readings")

    aligned, met_rows = {}, []
    events_by_pid, induction_by_pid = {}, {}
    for pid, g in raw.groupby("patientID"):
        ev = events.get(pid)
        if not ev:
            sys.stderr.write(f"[WARN] no event times for {pid} - skipped. "
                             "Check the hard-coded REDCap export.\n")
            continue
        ind = to_seconds_of_day(pd.Series([ev["induction_time"]])).iloc[0]
        if pd.isna(ind):
            sys.stderr.write(f"[WARN] no valid induction time for {pid} - skipped.\n")
            continue
        df = align_patient(g, ind, args.psi_shift)
        metrics = patient_metrics(df, ev, ind)
        if pd.isna(metrics["baseline_MAP"]) or pd.isna(metrics["nadir_MAP_post"]):
            sys.stderr.write(f"[WARN] no valid MAP baseline/post data for {pid} - skipped.\n")
            continue
        aligned[pid] = df
        events_by_pid[pid], induction_by_pid[pid] = ev, ind
        met_rows.append({"patientID": pid, **metrics})

    if not aligned:
        sys.stderr.write("No patients processed - check the REDCap export.\n")
        return 1

    summaries = pd.DataFrame(met_rows)
    spath = os.path.join(args.outdir, "induction_summary.csv")
    summaries.to_csv(spath, index=False)

    outputs = [os.path.abspath(spath)]
    outputs.append(fig_group_median_iqr(aligned, summaries, args.outdir, args.psi_shift))
    outputs.append(fig_depth_vs_map_drop(summaries, args.outdir))
    outputs.append(fig_lag_sensitivity(raw, events, args.outdir))

    # ---- the five epoch scatters (only possible with SctO2 loaded) --------- #
    has_sto2 = any(df[STO2_MEAN].notna().any() for df in aligned.values())
    if has_sto2:
        epochs_df = build_epoch_table(aligned, events_by_pid, induction_by_pid,
                                      mode=args.epoch_points)
        epath = os.path.join(args.outdir, "epoch_sto2_map.csv")
        epochs_df.to_csv(epath, index=False)
        outputs.append(os.path.abspath(epath))
        outputs.append(fig_sto2_vs_map_epochs(epochs_df, args.outdir,
                                              mode=args.epoch_points))
        counts = (epochs_df.dropna(subset=[MAP_COL, STO2_MEAN])
                  .groupby("epoch_label").size())
        print("\nPoints per epoch:")
        for label, n in counts.items():
            print(f"  {label}: {n}")
    else:
        sys.stderr.write("[INFO] no SctO2 data - skipping the epoch scatters "
                         "(pass --sto2 <master list> to enable them).\n")

    print("\nOutputs:")
    for p in outputs:
        print(" ", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
