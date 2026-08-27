#!/usr/bin/env python3
"""MAP-versus-cerebral-StO2 scatters for clinical OR time windows.

Pairs beat-to-beat MAP (/data/raw_cleaned/03_BeatToBeat_MAP/*_BTB.csv) with
multi-channel cerebral StO2 (/data/raw_cleaned/02_StO2/*_STO2.csv) per patient
and produces FOUR figures:

  Per-beat (unchanged behaviour, original filenames):
    * map_sto2_or_entry_to_sqi3.png        -- OR entry -> intra-op SQI>3,
                                              all-channel StO2 mean
    * map_sto2_window_panels.png           -- 2 x 5 grid: Right (CH1) /
                                              Left (CH2) across five windows
  One-point-per-patient summaries (NEW filenames):
    * map_sto2_or_entry_to_sqi3_patient_medians.png
    * map_sto2_window_panels_patient_medians.png

    Per patient the MAP is a MEAN and the StO2 is a MEDIAN.

Window anchors come from REDCap labeled-export columns:
    Pre-induction NOT in OR : [preop SQI>3 .. preop monitor end] incl/incl
    Pre-induction IN the OR : [intra-op SQI>3 .. INDUCTION) incl/excl
    Post-induction 0-5 min  : [INDUCTION .. INDUCTION+5 min)
    Post-induction 5-10 min : [+5 min .. +10 min)
    Post-induction 10-20 min: [+10 min .. +20 min]
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MAP_MIN, MAP_MAX = 20.0, 160.0
STO2_MIN, STO2_MAX = 0.0, 100.0
CHANNELS = (1, 2, 3, 4)

OR_ENTRY_LABEL = "what time did the patient enter the or?"
SQI_TIME_PREFIX = "what time did you record the sqi>3 bars"
PREOP_END_PREFIX = "time that monitoring ended in preop"
INDUCTION_PREFIX = "what time was induction (when primary induction med pushed)"

STEM_RE = re.compile(r"^ScreeningID(\d+)_(IU[MU]H\d+)_(BTB|STO2)\.csv$", re.IGNORECASE)

WINDOWS = [
    ("pre_ind_not_in_or", "Pre-induction\n(preop, outside OR)"),
    ("pre_ind_in_or", "Pre-induction\n(in the OR)"),
    ("post_0_5", "Post-induction\n0-5 min"),
    ("post_5_10", "Post-induction\n5-10 min"),
    ("post_10_20", "Post-induction\n10-20 min"),
]
SIDES = (("sto2_ch1", "Right (CH1)"), ("sto2_ch2", "Left (CH2)"))
PANEL_COLORS = ["#2f5597", "#7f6000", "#548235", "#c55a11", "#7030a0"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map-dir", type=Path,
                        default=Path("/N/project/Analgesia_BDproject//data/raw_cleaned/03_BeatToBeat_MAP"),
                        help="Directory of *_BTB.csv beat-to-beat MAP files.")
    parser.add_argument("--sto2-dir", type=Path,
                        default=Path("/N/project/Analgesia_BDproject//data/raw_cleaned/02_StO2"),
                        help="Directory of *_STO2.csv cerebral StO2 files.")
    parser.add_argument("--redcap", type=Path,
                        default=Path("/N/project/Analgesia_BDproject/data/00_raw/"
                                     "BDPostInductionHemod_DATA_LABELS_2026-08-21_1723.csv"),
                        help="REDCap labels export holding the window anchor times.")
    parser.add_argument("--outdir", type=Path,
                        default=Path("/N/project/Analgesia_BDproject/PR/scripts_STO2_analysis/or_entry_sqi_output"))
    parser.add_argument("--tolerance-seconds", type=float, default=2.0,
                        help="Max |MAP timestamp - StO2 timestamp| for nearest-neighbour pairing.")
    parser.add_argument("--min-valid-channels", type=int, default=1, choices=(1, 2),
                        help="Valid StO2 channels demanded for the all-channel MEAN "
                             "(standalone OR-entry plot only; CH1/CH2 panels ignore this).")
    return parser.parse_args()


# --------------------------------------------------------------------------- discovery

def scan_directory(directory: Path, kind: str) -> dict[str, tuple[str, Path]]:
    found: dict[str, tuple[str, Path]] = {}
    for path in sorted(directory.glob("*.csv")):
        match = STEM_RE.match(path.name)
        if match is None:
            print(f"WARNING: skipping unrecognised file {path.name}")
            continue
        screening, research_id, file_kind = match.groups()
        if file_kind.upper() != kind:
            continue
        research_id = research_id.upper().replace("IUUH", "IUMH")
        if screening in found:
            print(f"WARNING: duplicate {kind} file for {screening}: {path.name}")
            continue
        found[screening] = (research_id, path)
    return found


def pair_files(args: argparse.Namespace) -> list[tuple[str, str, Path, Path]]:
    maps, sto2s = scan_directory(args.map_dir, "BTB"), scan_directory(args.sto2_dir, "STO2")
    pairs: list[tuple[str, str, Path, Path]] = []
    for screening, (research_id, map_path) in sorted(maps.items()):
        if screening not in sto2s:
            print(f"WARNING: {screening} ({research_id}) has no STO2 file; skipped")
            continue
        sto2_id, sto2_path = sto2s[screening]
        if sto2_id != research_id:
            print(f"WARNING: {screening} ID mismatch BTB={research_id} vs STO2={sto2_id}; skipped")
            continue
        pairs.append((screening, research_id, map_path, sto2_path))
    if not pairs:
        raise ValueError(
            f"No paired BTB/STO2 files under {args.map_dir} and {args.sto2_dir}; "
            f"expected names like ScreeningID012_IUMH2026010901_BTB.csv")
    return pairs


# --------------------------------------------------------------------------- loaders

def match_column(lookup: dict[str, str], predicate, description: str, unique: bool = True) -> str | None:
    hits = [column for column, text in lookup.items() if predicate(text)]
    if unique and len(hits) != 1:
        raise ValueError(f"{description}: expected 1 matching REDCap column, found {hits}")
    return None if not hits else hits[0]


def load_redcap_anchors(redcap_path: Path) -> pd.DataFrame:
    """Return per-subject resolved instants for all window anchors."""
    redcap = pd.read_csv(redcap_path, low_memory=False)

    def lowered(columns: pd.Index) -> dict[str, str]:
        return {column: column.strip().lower() for column in columns}

    id_column = next(
        (c for c in redcap.columns
         if redcap[c].astype("string").str.strip().str.upper()
         .str.replace("_", "", regex=False).str.match(r"IU[MU]H\d+", na=False).any()),
        None)
    if id_column is None:
        raise ValueError("Could not find a research-ID column (e.g. IUMH2606101) in REDCap")
    redcap["subject_id"] = (
        redcap[id_column].astype("string").str.strip().str.upper()
        .str.replace("_", "", regex=False).str.replace("IUUH", "IUMH", regex=False))

    lookup = lowered(redcap.columns)
    or_entry_column = match_column(
        lookup, lambda t: t.startswith(OR_ENTRY_LABEL), "OR-entry column")
    preop_sqi_column = match_column(
        lookup, lambda t: t.startswith(SQI_TIME_PREFIX) and "preop" in t, "PREOP SQI>3 column")
    intra_sqi_column = match_column(
        lookup, lambda t: t.startswith(SQI_TIME_PREFIX) and "preop" not in t, "intra-op SQI>3 column")
    preop_end_column = match_column(
        lookup, lambda t: t.startswith(PREOP_END_PREFIX), "preop monitoring-end column",
        unique=False)  # long Note suffix must not collide with anything else
    induction_column = match_column(
        lookup, lambda t: t.startswith(INDUCTION_PREFIX), "INDUCTION TIME=0 column")

    date_columns = [c for c, text in lookup.items() if text == "date of surgery"]
    anchor_dates = pd.to_datetime(redcap[date_columns[0]], errors="coerce") if date_columns \
        else pd.Series(pd.NaT, index=redcap.index, dtype="datetime64[ns]")
    missing_dates = int(anchor_dates.isna().sum())
    if missing_dates:
        print(f"WARNING: {missing_dates} REDCap row(s) lack a usable Date of Surgery; "
              f"their bare clock times stay unanchored")

    def resolve(times: pd.Series) -> pd.Series:
        """Attach Date-of-Surgery to bare HH:MM[:SS] REDCap times."""
        raw = times.astype("string").str.strip()
        parsed = pd.to_datetime(raw, errors="coerce")
        today = pd.Timestamp.today().normalize()
        stubbed = parsed.notna() & parsed.dt.normalize().eq(today) & anchor_dates.notna()
        resolved = parsed.copy()
        resolved.loc[stubbed] = anchor_dates.loc[stubbed].dt.normalize() + (parsed.loc[stubbed] - today)
        return resolved

    anchors = pd.DataFrame({
        "preop_sqi_time": resolve(redcap[preop_sqi_column]),
        "preop_end_time": resolve(redcap[preop_end_column]) if preop_end_column else pd.NaT,
        "or_entry_time": resolve(redcap[or_entry_column]),
        "intra_sqi_time": resolve(redcap[intra_sqi_column]),
        "induction_time": resolve(redcap[induction_column]),
    })
    anchors.insert(0, "subject_id", redcap["subject_id"])
    # groupby.first() takes the first non-null value per column across form/event rows
    anchors = anchors.dropna(subset=["subject_id"]).groupby("subject_id", sort=False).first()
    return anchors


def channel_value(row: pd.Series, channel: int) -> float:
    if pd.to_numeric(row.get(f"valid_CH{channel}"), errors="coerce") != 1:
        return np.nan
    value = pd.to_numeric(row.get(f"StO2_CH{channel}"), errors="coerce")
    return value if STO2_MIN <= value <= STO2_MAX else np.nan


def load_map_file(path: Path) -> pd.DataFrame | None:
    frame = pd.read_csv(path)
    if "time" not in frame.columns or "meanArterialPressure" not in frame.columns:
        print(f"WARNING: {path.name} lacks expected MAP columns; skipped")
        return None
    frame["time"] = pd.to_datetime(frame["time"], format="%d-%b-%Y %H:%M:%S", errors="coerce")
    frame["meanArterialPressure"] = pd.to_numeric(frame["meanArterialPressure"], errors="coerce")
    frame = frame.dropna(subset=["time"])
    bad = pd.to_numeric(frame.get("databad"), errors="coerce").fillna(1)
    usable = frame.loc[(bad != 1) & frame["meanArterialPressure"].between(MAP_MIN, MAP_MAX)]
    return usable[["time", "meanArterialPressure"]].rename(
        columns={"meanArterialPressure": "map"}).sort_values("time")


def load_sto2_file(path: Path, min_valid_channels: int) -> pd.DataFrame | None:
    frame = pd.read_csv(path)
    if "Time" not in frame.columns:
        print(f"WARNING: {path.name} lacks a Time column; skipped")
        return None
    frame["time"] = pd.to_datetime(frame["Time"], format="%m/%d/%Y %H:%M:%S", errors="coerce")
    frame = frame.dropna(subset=["time"])

    means, ch1_values, ch2_values = [], [], []
    for _, row in frame.iterrows():
        values = [channel_value(row, ch) for ch in CHANNELS]
        valid = [v for v in values if not np.isnan(v)]
        means.append(np.mean(valid) if len(valid) >= min_valid_channels else np.nan)
        ch1_values.append(values[0])
        ch2_values.append(values[1])
    return (frame[["time"]]
            .assign(sto2_mean=means, sto2_ch1=ch1_values, sto2_ch2=ch2_values)
            .dropna(how="all", subset=["sto2_mean", "sto2_ch1", "sto2_ch2"])
            .sort_values("time"))


# --------------------------------------------------------------------------- assembly

def window_masks(times: pd.Series, anchors: pd.Series) -> tuple[dict[str, pd.Series], list[str]]:
    """Boolean mask per window plus the list of windows unavailable (bad/missing anchors)."""
    start_preop = anchors.get("preop_sqi_time", pd.NaT)
    end_preop = anchors.get("preop_end_time", pd.NaT)
    start_in_or = anchors.get("intra_sqi_time", pd.NaT)
    induction = anchors.get("induction_time", pd.NaT)

    masks: dict[str, pd.Series] = {}
    unavailable: list[str] = []

    def close_window(key: str, low: pd.Timestamp, high: pd.Timestamp,
                     low_closed: bool, high_closed: bool) -> bool:
        low_ok, high_ok = pd.notna(low), pd.notna(high)
        if not (low_ok and high_ok) or low > high:
            unavailable.append(key)
            return False
        lower = times.ge(low) if low_closed else times.gt(low)
        upper = times.le(high) if high_closed else times.lt(high)
        masks[key] = lower & upper
        return True

    close_window("pre_ind_not_in_or", start_preop, end_preop, True, True)
    close_window("pre_ind_in_or", start_in_or, induction, True, False)
    if pd.notna(induction):
        offsets = {"post_0_5": (0, 300), "post_5_10": (300, 600), "post_10_20": (600, 1200)}
        for key, (low_off, high_off) in offsets.items():
            high_closed = key == "post_10_20"          # nothing follows it to double-count
            close_window(key, induction + pd.Timedelta(seconds=low_off),
                         induction + pd.Timedelta(seconds=high_off), True, high_closed)
    else:
        unavailable.extend(["post_0_5", "post_5_10", "post_10_20"])
    return masks, unavailable


def build_dataset(pairs, anchors: pd.DataFrame, args: argparse.Namespace):
    tolerance = pd.Timedelta(seconds=args.tolerance_seconds)
    standalone_chunks: list[pd.DataFrame] = []
    panel_chunks: list[pd.DataFrame] = []
    for screening, subject_id, map_path, sto2_path in pairs:
        map_frame = load_map_file(map_path)
        sto2_frame = load_sto2_file(sto2_path, args.min_valid_channels)
        if map_frame is None or map_frame.empty or sto2_frame is None or sto2_frame.empty:
            print(f"WARNING: {screening} ({subject_id}) yielded no usable rows; skipped")
            continue
        merged = pd.merge_asof(map_frame.assign(subject_id=subject_id),
                               sto2_frame, on="time", direction="nearest", tolerance=tolerance)
        if subject_id not in anchors.index:
            print(f"WARNING: {subject_id} absent from REDCap export; no windows applied")
            continue
        subject_anchors = anchors.loc[subject_id]

        # ---- standalone plot: OR entry -> intra-op SQI>3, all-channel mean
        or_entry, sqi3 = subject_anchors["or_entry_time"], subject_anchors["intra_sqi_time"]
        if pd.notna(or_entry) and pd.notna(sqi3) and or_entry < sqi3:
            base = merged.dropna(subset=["sto2_mean"])
            in_first = base["time"].between(or_entry, sqi3)
            if in_first.any():
                standalone_chunks.append(base.loc[in_first].assign(screening_id=screening))
        else:
            print(f"WARNING: {subject_id} missing/misordered OR-entry or intra-op SQI>3 anchors")

        # ---- windowed panels: CH1/CH2 sides, half-open boundaries per header contract
        masks, unavailable = window_masks(merged["time"], subject_anchors)
        if unavailable:
            print(f"WARNING: {subject_id} missing/misordered anchors for: "
                  f"{', '.join(unavailable)} (those panels get nothing)")
        for key, _ in WINDOWS:
            if key not in masks:
                continue
            for column, side_label in SIDES:
                selected = merged.loc[masks[key] & merged[column].notna()]
                if len(selected):
                    panel_chunks.append(selected.assign(
                        screening_id=screening, window=key, side=side_label,
                        sto2=selected[column]))

        summary = " ".join(f"{key.replace('_', '')}={int(mask.sum())}" for key, mask in masks.items())
        print(f"{screening} {subject_id}: beats paired={len(merged)} {summary}")

    standalone = pd.concat(standalone_chunks, ignore_index=True) if standalone_chunks else pd.DataFrame()
    panels = pd.concat(panel_chunks, ignore_index=True) if panel_chunks else pd.DataFrame()

    # ---- one point per patient: MEAN MAP against MEDIAN StO2
    standalone_medians = pd.DataFrame()
    if not standalone.empty:
        standalone_medians = (standalone.groupby(["subject_id"], as_index=False)
                              .agg(map=("map", "mean"),
                                   sto2_mean=("sto2_mean", "median"),
                                   beats=("map", "size")))
    panel_medians = pd.DataFrame()
    if not panels.empty:
        panel_medians = (panels.groupby(["window", "side", "subject_id"], as_index=False)
                         .agg(map=("map", "mean"), sto2=("sto2", "median"),
                              beats=("sto2", "size")))
    return standalone, panels, standalone_medians, panel_medians


# --------------------------------------------------------------------------- plotting

def style_axis(axis: plt.Axes) -> None:
    axis.set_xlim(MAP_MIN, MAP_MAX)
    axis.set_ylim(STO2_MIN, STO2_MAX)
    axis.grid(True, color="#d9d9d9", linewidth=0.6, alpha=0.8)
    axis.set_axisbelow(True)


def draw_scatter(axis: plt.Axes, data: pd.DataFrame, y_column: str, color: str, per_patient: bool) -> None:
    if per_patient:
        axis.scatter(data["map"], data[y_column], s=50, alpha=0.8, color=color,
                     edgecolors="#444444", linewidths=0.4)
    else:
        axis.scatter(data["map"], data[y_column], s=10, alpha=0.25, color=color,
                     edgecolors="none", rasterized=True)


def render_panel_figure(data: pd.DataFrame, y_column: str, per_patient: bool, output_path: Path) -> None:
    figure, axes = plt.subplots(len(SIDES), len(WINDOWS), figsize=(22, 9),
                                sharex=True, sharey=True)
    y_label = "Cerebral StO2 (%)" + ("\n(median per patient)" if per_patient else "")
    for row, (side_column, side_label) in enumerate(SIDES):
        for column, ((key, title), color) in enumerate(zip(WINDOWS, PANEL_COLORS)):
            axis = axes[row][column]
            selected = data.loc[data["window"].eq(key) & data["side"].eq(side_label)]
            if len(selected):
                draw_scatter(axis, selected, y_column, color, per_patient)
                points = selected["subject_id"].nunique()
                if per_patient:
                    subtitle = f"n = {points:,} pts (mean MAP)"
                else:
                    total_beats = int(selected["beats"].sum()) if "beats" in selected \
                        else len(selected)
                    subtitle = f"n = {total_beats:,} beats; {points:,} pts"
            else:
                axis.text(0.5, 0.5, "no data", ha="center", va="center",
                          transform=axis.transAxes, color="#888888")
                subtitle = "n = 0"
            axis.set_title(f"{title}\n{subtitle}", fontsize=10)
            style_axis(axis)
            if row == len(SIDES) - 1:
                axis.set_xlabel("MAP (mmHg)" + ("\n(mean per patient)" if per_patient else ""))
        axes[row][0].set_ylabel(f"{y_label}\n{side_label}")
    suffix = "\n(one point per patient: mean MAP vs median StO2)" if per_patient else ""
    figure.suptitle("Cerebral StO2 versus MAP by hemisphere and clinical window"
                    + suffix, y=1.02)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def render_standalone_figure(data: pd.DataFrame, y_column: str, per_patient: bool, output_path: Path) -> None:
    figure, axis = plt.subplots(figsize=(7, 5.5))
    draw_scatter(axis, data, y_column, "#2f5597", per_patient)
    patients = data["subject_id"].nunique()
    if per_patient:
        detail = f"n = {patients:,} patients (mean MAP, median StO2)"
    else:
        detail = f"n = {len(data):,} beats, {patients:,} patients"
    axis.set_title(f"Cerebral StO2 vs MAP\nOR entry -> SQI>3   ({detail})")
    style_axis(axis)
    axis.set_xlabel("MAP (mmHg)" + ("\n(mean per patient)" if per_patient else ""))
    axis.set_ylabel("Cerebral StO2 (%)" + ("\n(median per patient)" if per_patient else ""))
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


# --------------------------------------------------------------------------- driver

def main() -> int:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    pairs = pair_files(args)
    print(f"Found {len(pairs)} paired patient file(s).")

    anchors = load_redcap_anchors(args.redcap)
    standalone, panels, standalone_medians, panel_medians = build_dataset(pairs, anchors, args)

    produced_any = False

    # --- per-beat originals (filenames unchanged)
    points_path = args.outdir / "or_entry_to_sqi3_points.csv"
    if standalone.empty:
        print("No OR-entry->SQI>3 observations produced; standalone plot skipped.")
    else:
        standalone.to_csv(points_path, index=False)
        render_standalone_figure(standalone, "sto2_mean", False,
                                 args.outdir / "map_sto2_or_entry_to_sqi3.png")
        print(f"Per-beat standalone: {len(standalone):,} beats across "
              f"{standalone['subject_id'].nunique()} patients.")
        produced_any = True

    panel_png = args.outdir / "map_sto2_window_panels.png"
    panel_points = args.outdir / "window_side_points.csv"
    if panels.empty:
        print("No windowed observations produced; per-beat panel figure skipped.")
    else:
        panels.to_csv(panel_points, index=False)
        render_panel_figure(panels, "sto2", False, panel_png)
        produced_any = True

    # --- one point per patient (new filenames; originals untouched)
    if standalone_medians.empty:
        print("No per-patient summaries for the standalone interval.")
    else:
        standalone_medians.to_csv(args.outdir / "or_entry_to_sqi3_patient_medians.csv", index=False)
        render_standalone_figure(standalone_medians, "sto2_mean", True,
                                 args.outdir / "map_sto2_or_entry_to_sqi3_patient_medians.png")
        print(f"Per-patient standalone: {len(standalone_medians)} patients "
              f"(mean MAP, median StO2).")

    if panel_medians.empty:
        print("No per-patient summaries for the window panels.")
    else:
        panel_medians.to_csv(args.outdir / "window_side_patient_medians.csv", index=False)
        render_panel_figure(panel_medians, "sto2", True,
                            args.outdir / "map_sto2_window_panels_patient_medians.png")
        empty_windows = [key for key, _ in WINDOWS
                         if key not in set(panel_medians["window"])]
        if empty_windows:
            print(f"Windows with zero contributing patients: {empty_windows}")
        produced_any = True

    print(f"Figures written to: {args.outdir}")
    return 0 if produced_any else 1


if __name__ == "__main__":
    raise SystemExit(main())
