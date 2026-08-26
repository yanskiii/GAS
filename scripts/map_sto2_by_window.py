#!/usr/bin/env python3
"""Create five MAP-versus-cerebral-StO2 scatter panels, one point per patient.

Each patient contributes a single summary point per panel (median by default),
so every panel shows one dot per patient rather than a cloud of 20-second
samples.

Because patients silently disappear at several filtering stages, this script
writes a full per-panel funnel audit:

  * patient_panel_audit.csv  - one row per patient per panel, with the reason
                               that patient is or is not in that panel.
  * panel_funnel.csv         - patient counts at every filtering stage.
  * missing_first_plot_patients.csv
  * map_sto2_patient_points.csv - the plotted per-patient points.

Run with --explain to print the funnel and the top exclusion reasons.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


WINDOWS = [
    ("pre_induction", "Pre-induction\n(preop, outside OR)"),
    ("or_pre_induction", "Pre-induction\n(in OR)"),
    ("post_0_5", "Post-induction\n0-5 min"),
    ("post_5_10", "Post-induction\n5-10 min"),
    ("post_10_20", "Post-induction\n10-20 min"),
]

MAP_MIN, MAP_MAX = 20.0, 160.0
STO2_MIN, STO2_MAX = 0.0, 100.0
EXPECTED_FIRST_PATIENTS = 93

# Known REDCap ID typos -> the true research ID. Without these the affected
# patients fail the REDCap merge and vanish from the preop panel.
REDCAP_ID_ALIASES = {
    "IUMH202601601": "IUMH2026011601",
    "IUMH2026010601-20260105": "IUMH2026010501",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--data",
        type=Path,
        default=Path(
            "/N/project/Analgesia_BDproject/PR/data/"
            "analysis_timeseries_20s.csv"
        ),
        help="Aligned 20-second multimodal CSV or CSV.GZ.",
    )

    parser.add_argument(
        "--redcap",
        type=Path,
        default=Path(
            "/N/project/Analgesia_BDproject/PR/data/"
            "6.16.26.FIXED-TYPOS-BDPostInductionHemod_"
            "DATA_LABELS_2026-06-16_1657.csv"
        ),
        help="REDCap export containing preop monitoring start/end times.",
    )

    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("/N/project/Analgesia_BDproject/PR/figures/4plots"),
    )

    parser.add_argument(
        "--agg",
        choices=["median", "mean"],
        default="median",
        help="How to collapse each patient's samples within a window. "
             "Median resists art-line flushes and dropouts (default).",
    )

    parser.add_argument(
        "--min-obs",
        type=int,
        default=3,
        help="Minimum in-window samples for a patient to contribute a point.",
    )

    parser.add_argument(
        "--explain",
        action="store_true",
        help="Print the per-panel funnel and top exclusion reasons.",
    )

    return parser.parse_args()


def normalize_subject_id(values: pd.Series) -> pd.Series:
    """
    Standardize formatting without changing the hospital prefix.

    IUUH and IUMH remain different IDs.
    """

    return (
        values.astype("string")
        .str.strip()
        .str.upper()
        .str.replace(r"[\s_-]+", "", regex=True)
    )


def apply_id_aliases(values: pd.Series) -> pd.Series:
    """Repair known REDCap ID typos after normalization."""

    aliases = {
        normalize_subject_id(pd.Series([bad])).iloc[0]: good
        for bad, good in REDCAP_ID_ALIASES.items()
    }

    return values.replace(aliases)


def find_redcap_column(
    redcap: pd.DataFrame,
    prefix: str,
) -> str:
    """Find a REDCap column even when it contains extra spaces."""

    matches = [
        column
        for column in redcap.columns
        if str(column).strip().startswith(prefix)
    ]

    if not matches:
        raise ValueError(
            f"Could not find REDCap column beginning with: {prefix}"
        )

    return matches[0]


def find_redcap_id_column(
    redcap: pd.DataFrame,
) -> str | None:
    """Find the REDCap column containing research IDs.

    Picks the column with the MOST ID-shaped values, not merely the first
    column containing one, so a stray free-text field cannot win.
    """

    best_column = None
    best_count = 0

    for column in redcap.columns:
        normalized = normalize_subject_id(redcap[column])

        count = int(
            normalized.str.fullmatch(
                r"IU(?:MH|UH)\d+",
                na=False,
            ).sum()
        )

        if count > best_count:
            best_column = column
            best_count = count

    return best_column


def choose_redcap_rows(
    lookup: pd.DataFrame,
    key: str,
) -> pd.DataFrame:
    """
    Keep one REDCap preop interval per patient.

    Rows with both start and end times are preferred.
    """

    lookup = lookup.loc[lookup[key].notna()].copy()

    lookup["_complete"] = (
        lookup["preop_start"].notna().astype(int)
        + lookup["preop_end"].notna().astype(int)
        + lookup["or_entry"].notna().astype(int)
    )

    complete_intervals = (
        lookup.loc[
            lookup["preop_start"].notna()
            & lookup["preop_end"].notna(),
            [key, "preop_start", "preop_end"],
        ]
        .drop_duplicates()
    )

    interval_counts = complete_intervals.groupby(key).size()
    conflicting_ids = interval_counts.loc[interval_counts.gt(1)].index

    if len(conflicting_ids) > 0:
        print(
            "\nWARNING: Multiple REDCap preop intervals were found for "
            f"{len(conflicting_ids)} patients."
        )
        print(
            "Check these IDs:",
            ", ".join(map(str, conflicting_ids)),
        )

    lookup = (
        lookup.sort_values(
            "_complete",
            ascending=False,
        )
        .drop_duplicates(
            key,
            keep="first",
        )
    )

    return lookup[
        [
            key,
            "preop_start",
            "preop_end",
            "or_entry",
        ]
    ]


def load_data(
    data_path: Path,
    redcap_path: Path,
) -> pd.DataFrame:
    data = pd.read_csv(
        data_path,
        low_memory=False,
    )

    # PSI is reported for context only - it never filters or categorizes.
    required = {
        "subject_id",
        "timestamp_local",
        "map",
        "sto2_mean",
        "seconds_from_induction",
    }

    missing = required - set(data.columns)

    if missing:
        raise ValueError(
            f"Input is missing required columns: {sorted(missing)}"
        )

    data["subject_id_raw"] = data["subject_id"].astype("string")
    data["subject_id"] = apply_id_aliases(
        normalize_subject_id(data["subject_id"])
    )

    data["timestamp_local"] = pd.to_datetime(
        data["timestamp_local"],
        errors="coerce",
    )

    if "psi_median" not in data.columns:
        data["psi_median"] = pd.NA

    for column in [
        "map",
        "sto2_mean",
        "psi_median",
        "seconds_from_induction",
    ]:
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce",
        )

    if "brain_bilateral_usable" not in data.columns:
        data["brain_bilateral_usable"] = 1

    data["brain_bilateral_usable"] = pd.to_numeric(
        data["brain_bilateral_usable"],
        errors="coerce",
    )

    redcap_raw = pd.read_csv(
        redcap_path,
        low_memory=False,
    )

    preop_start_column = find_redcap_column(
        redcap_raw,
        "Time monitoring started in preop:",
    )

    preop_end_column = find_redcap_column(
        redcap_raw,
        "Time that monitoring ended in preop:",
    )

    # Panel 2 boundary: when the patient entered the OR.
    try:
        or_entry_column = find_redcap_column(
            redcap_raw,
            "What time did the patient enter the OR?",
        )
    except ValueError:
        or_entry_column = None
        print(
            "\nWARNING: No 'What time did the patient enter the OR?' column "
            "found. The in-OR pre-induction panel will fall back to "
            "'pre-induction and not inside the preop window'."
        )

    id_column = find_redcap_id_column(redcap_raw)

    if id_column is not None:
        redcap_lookup = pd.DataFrame(
            {
                "subject_id": apply_id_aliases(
                    normalize_subject_id(redcap_raw[id_column])
                ),
                "preop_start": pd.to_datetime(
                    redcap_raw[preop_start_column],
                    errors="coerce",
                ),
                "preop_end": pd.to_datetime(
                    redcap_raw[preop_end_column],
                    errors="coerce",
                ),
                "or_entry": (
                    pd.to_datetime(
                        redcap_raw[or_entry_column],
                        errors="coerce",
                    )
                    if or_entry_column
                    else pd.NaT
                ),
            }
        )

        redcap_lookup = choose_redcap_rows(
            redcap_lookup,
            key="subject_id",
        )

        data = data.merge(
            redcap_lookup,
            on="subject_id",
            how="left",
            validate="many_to_one",
        )

    elif (
        "screening_id" in data.columns
        and "Screening ID" in redcap_raw.columns
    ):
        data["screening_id"] = pd.to_numeric(
            data["screening_id"],
            errors="coerce",
        )

        redcap_lookup = pd.DataFrame(
            {
                "screening_id": pd.to_numeric(
                    redcap_raw["Screening ID"],
                    errors="coerce",
                ),
                "preop_start": pd.to_datetime(
                    redcap_raw[preop_start_column],
                    errors="coerce",
                ),
                "preop_end": pd.to_datetime(
                    redcap_raw[preop_end_column],
                    errors="coerce",
                ),
                "or_entry": (
                    pd.to_datetime(
                        redcap_raw[or_entry_column],
                        errors="coerce",
                    )
                    if or_entry_column
                    else pd.NaT
                ),
            }
        )

        redcap_lookup = choose_redcap_rows(
            redcap_lookup,
            key="screening_id",
        )

        data = data.merge(
            redcap_lookup,
            on="screening_id",
            how="left",
            validate="many_to_one",
        )

    else:
        raise ValueError(
            "Could not match REDCap patients to the timeseries data."
        )

    data["valid_map_sto2"] = (
        data["map"].between(MAP_MIN, MAP_MAX)
        & data["sto2_mean"].between(STO2_MIN, STO2_MAX)
        & data["brain_bilateral_usable"].eq(1)
    )

    return data


def preop_time_mask(data: pd.DataFrame) -> pd.Series:
    """Identify rows occurring inside each patient's preop window."""

    timestamp_seconds = (
        data["timestamp_local"].dt.hour * 3600
        + data["timestamp_local"].dt.minute * 60
        + data["timestamp_local"].dt.second
    )

    start_seconds = (
        data["preop_start"].dt.hour * 3600
        + data["preop_start"].dt.minute * 60
        + data["preop_start"].dt.second
    )

    end_seconds = (
        data["preop_end"].dt.hour * 3600
        + data["preop_end"].dt.minute * 60
        + data["preop_end"].dt.second
    )

    has_times = (
        timestamp_seconds.notna()
        & start_seconds.notna()
        & end_seconds.notna()
    )

    normal_window = start_seconds.le(end_seconds)

    within_normal_window = (
        normal_window
        & timestamp_seconds.ge(start_seconds)
        & timestamp_seconds.le(end_seconds)
    )

    # Handles an interval that crosses midnight.
    within_midnight_window = (
        ~normal_window
        & (
            timestamp_seconds.ge(start_seconds)
            | timestamp_seconds.le(end_seconds)
        )
    )

    time_match = (
        has_times
        & (
            within_normal_window
            | within_midnight_window
        )
    )

    # Exclude clearly post-induction rows.
    pre_induction_or_unknown = (
        data["seconds_from_induction"].lt(0)
        | data["seconds_from_induction"].isna()
    )

    return time_match & pre_induction_or_unknown


def _clock_seconds(values: pd.Series) -> pd.Series:
    """Time-of-day, in seconds since midnight."""

    return (
        values.dt.hour * 3600
        + values.dt.minute * 60
        + values.dt.second
    )


def or_entry_seconds_from_induction(data: pd.DataFrame) -> pd.Series:
    """OR-entry time expressed as seconds from induction (negative).

    REDCap stores a clock time, so it is anchored against each patient's
    induction datetime, which is recovered from the timeseries itself
    (timestamp_local minus seconds_from_induction). Differences are wrapped
    into +/-12 h so a case spanning midnight still works.
    """

    if "or_entry" not in data.columns:
        return pd.Series(np.nan, index=data.index, dtype=float)

    induction_timestamp = (
        data["timestamp_local"]
        - pd.to_timedelta(data["seconds_from_induction"], unit="s")
    )

    induction_timestamp = induction_timestamp.groupby(
        data["subject_id"]
    ).transform("median")

    delta = (
        _clock_seconds(data["or_entry"])
        - _clock_seconds(induction_timestamp)
    )

    # Wrap into (-12 h, +12 h].
    delta = ((delta + 43200) % 86400) - 43200

    return delta.astype(float)


def assign_windows(data: pd.DataFrame) -> pd.DataFrame:
    """
    Assign observations to plots.

    The first plot is created before applying the strict bilateral filter.
    Later plots use the strict quality filter.
    """

    induction = data["seconds_from_induction"]

    preop_match = preop_time_mask(data)

    preop_plottable = (
        preop_match
        & data["map"].between(MAP_MIN, MAP_MAX)
        & data["sto2_mean"].between(STO2_MIN, STO2_MAX)
    )

    preop_data = data.loc[preop_plottable].copy()
    preop_data["panel"] = "pre_induction"

    # Panel 2: in the OR, before induction. Purely a time window - PSI is
    # never used to include, exclude or categorize an observation.
    or_entry_rel = or_entry_seconds_from_induction(data)

    in_or_pre_induction = induction.lt(0) & ~preop_match

    # Where OR-entry is recorded, start the window there; where it is missing,
    # keep every pre-induction row that is not inside the preop window.
    in_or_pre_induction &= or_entry_rel.isna() | induction.ge(or_entry_rel)

    or_pre_data = data.loc[
        in_or_pre_induction
        & data["map"].between(MAP_MIN, MAP_MAX)
        & data["sto2_mean"].between(STO2_MIN, STO2_MAX)
    ].copy()
    or_pre_data["panel"] = "or_pre_induction"

    filtered = data.loc[data["valid_map_sto2"]].copy()

    induction = filtered["seconds_from_induction"]

    panel_masks = {
        "post_0_5": induction.between(
            0,
            300,
            inclusive="left",
        ),
        "post_5_10": induction.between(
            300,
            600,
            inclusive="left",
        ),
        "post_10_20": induction.between(
            600,
            1200,
            inclusive="left",
        ),
    }

    panels = [preop_data, or_pre_data]

    for panel_name, mask in panel_masks.items():
        panel_data = filtered.loc[mask].copy()
        panel_data["panel"] = panel_name
        panels.append(panel_data)

    return pd.concat(
        panels,
        ignore_index=True,
    )


def collapse_to_patient_points(
    plotted_data: pd.DataFrame,
    how: str = "median",
    min_obs: int = 3,
) -> pd.DataFrame:
    """Collapse every patient's in-window samples to ONE point per panel."""

    if plotted_data.empty:
        return pd.DataFrame(
            columns=[
                "subject_id",
                "panel",
                "map",
                "sto2_mean",
                "psi_median",
                "n_obs",
            ]
        )

    grouped = plotted_data.groupby(
        ["subject_id", "panel"],
        as_index=False,
    ).agg(
        map=("map", how),
        sto2_mean=("sto2_mean", how),
        psi_median=("psi_median", how),
        n_obs=("map", "size"),
        map_n=("map", "count"),
        sto2_n=("sto2_mean", "count"),
    )

    enough = (
        grouped["map_n"].ge(min_obs)
        & grouped["sto2_n"].ge(min_obs)
    )

    dropped = int((~enough).sum())

    if dropped:
        print(
            f"\nNote: {dropped} patient-panel points dropped for having "
            f"fewer than {min_obs} usable samples."
        )

    return grouped.loc[enough].reset_index(drop=True)


def build_panel_audit(
    loaded_data: pd.DataFrame,
    plotted_data: pd.DataFrame,
    patient_points: pd.DataFrame,
    min_obs: int,
) -> pd.DataFrame:
    """One row per patient per panel explaining inclusion or exclusion.

    This is the answer to 'why are patients missing?' - every patient appears
    in every panel, with the first failing stage named.
    """

    audit_source = loaded_data.copy()
    audit_source["_preop_time_match"] = preop_time_mask(audit_source)
    audit_source["_or_entry_rel"] = or_entry_seconds_from_induction(
        audit_source
    )
    audit_source["_pre_induction_in_or"] = (
        audit_source["seconds_from_induction"].lt(0)
        & ~audit_source["_preop_time_match"]
    )

    induction = audit_source["seconds_from_induction"]

    per_patient = audit_source.groupby("subject_id").agg(
        input_rows=("subject_id", "size"),
        has_redcap_times=(
            "preop_start",
            lambda s: bool(s.notna().any()),
        ),
        has_redcap_end=(
            "preop_end",
            lambda s: bool(s.notna().any()),
        ),
        rows_in_preop_window=("_preop_time_match", "sum"),
        rows_bilateral_usable=(
            "brain_bilateral_usable",
            lambda s: int(s.eq(1).sum()),
        ),
        rows_valid_map_sto2=("valid_map_sto2", "sum"),
        rows_pre_induction_in_or=("_pre_induction_in_or", "sum"),
        has_or_entry=(
            "_or_entry_rel",
            lambda s: bool(s.notna().any()),
        ),
        rows_post_induction=(
            "seconds_from_induction",
            lambda s: int(s.ge(0).sum()),
        ),
        max_seconds_from_induction=(
            "seconds_from_induction",
            "max",
        ),
    )

    obs_counts = (
        plotted_data.groupby(["subject_id", "panel"])
        .size()
        .rename("window_rows")
    )

    point_counts = (
        patient_points.set_index(["subject_id", "panel"])["n_obs"]
        .rename("plotted_n_obs")
    )

    rows = []

    for subject_id, summary in per_patient.iterrows():
        for panel_key, _label in WINDOWS:
            window_rows = int(
                obs_counts.get((subject_id, panel_key), 0)
            )

            plotted = (subject_id, panel_key) in point_counts.index

            rows.append(
                {
                    "subject_id": subject_id,
                    "panel": panel_key,
                    "window_rows": window_rows,
                    "in_plot": bool(plotted),
                    "reason": _exclusion_reason(
                        summary,
                        panel_key,
                        window_rows,
                        plotted,
                        min_obs,
                    ),
                    **summary.to_dict(),
                }
            )

    return pd.DataFrame(rows)


def _exclusion_reason(
    summary: pd.Series,
    panel_key: str,
    window_rows: int,
    plotted: bool,
    min_obs: int,
) -> str:
    """Name the FIRST stage at which this patient fell out of this panel."""

    if plotted:
        return "included"

    if window_rows > 0:
        return f"fewer than {min_obs} usable samples in window"

    if panel_key == "pre_induction":
        if not summary["has_redcap_times"] or not summary["has_redcap_end"]:
            return "no REDCap preop start/end times (ID match or blank field)"
        if summary["rows_in_preop_window"] == 0:
            return "no timeseries rows inside the preop clock window"
        return "no in-range MAP+StO2 pair inside preop window"

    if panel_key != "or_pre_induction" and summary["rows_post_induction"] == 0:
        return "no post-induction rows (induction time missing/unaligned)"

    if summary["rows_bilateral_usable"] == 0:
        return "brain_bilateral_usable never equals 1"

    if summary["rows_valid_map_sto2"] == 0:
        return "no rows pass MAP+StO2 range AND bilateral-usable filter"

    if panel_key == "or_pre_induction":
        if summary["rows_pre_induction_in_or"] == 0:
            return (
                "no pre-induction rows outside the preop window "
                "(record starts at induction?)"
            )
        return "no in-range MAP+StO2 pair between OR entry and induction"

    return "no rows inside this time window (record too short?)"


def print_funnel(
    loaded_data: pd.DataFrame,
    audit: pd.DataFrame,
    patient_points: pd.DataFrame,
) -> pd.DataFrame:
    """Print and return the patient-count funnel per panel."""

    total = loaded_data["subject_id"].nunique()

    per_patient = audit.drop_duplicates("subject_id")

    print("\n" + "=" * 68)
    print("COHORT FUNNEL (unique patients)")
    print("=" * 68)
    print(f"  patients in timeseries input .................. {total}")
    print(
        "  ... with REDCap preop start AND end ........... "
        f"{int((per_patient['has_redcap_times'] & per_patient['has_redcap_end']).sum())}"
    )
    print(
        "  ... with >=1 row inside preop clock window .... "
        f"{int(per_patient['rows_in_preop_window'].gt(0).sum())}"
    )
    print(
        "  ... with >=1 bilateral-usable row ............. "
        f"{int(per_patient['rows_bilateral_usable'].gt(0).sum())}"
    )
    print(
        "  ... with >=1 row passing MAP+StO2+bilateral ... "
        f"{int(per_patient['rows_valid_map_sto2'].gt(0).sum())}"
    )
    print(
        "  ... with >=1 post-induction row ............... "
        f"{int(per_patient['rows_post_induction'].gt(0).sum())}"
    )
    print(
        "  ... with a REDCap OR-entry time ............... "
        f"{int(per_patient['has_or_entry'].sum())}"
    )
    print(
        "  ... with >=1 pre-induction row outside preop .. "
        f"{int(per_patient['rows_pre_induction_in_or'].gt(0).sum())}"
    )

    print("\n" + "=" * 68)
    print("PER-PANEL RESULT")
    print("=" * 68)

    funnel_rows = []

    for panel_key, label in WINDOWS:
        panel_audit = audit.loc[audit["panel"].eq(panel_key)]
        n_in = int(panel_audit["in_plot"].sum())

        flat_label = label.replace("\n", " ")
        print(f"\n  {flat_label}: {n_in}/{total} patients")

        excluded = panel_audit.loc[~panel_audit["in_plot"], "reason"]

        for reason, count in excluded.value_counts().items():
            print(f"      {count:>4}  {reason}")

        funnel_rows.append(
            {
                "panel": panel_key,
                "patients_in_plot": n_in,
                "patients_total": total,
            }
        )

    return pd.DataFrame(funnel_rows)


def make_plot(
    patient_points: pd.DataFrame,
    output_path: Path,
    how: str,
) -> None:
    figure, axes = plt.subplots(
        1,
        5,
        figsize=(22, 5),
        sharex=True,
        sharey=True,
    )

    colors = [
        "#2f5597",
        "#7f6000",
        "#548235",
        "#c55a11",
        "#7030a0",
    ]

    for axis, (key, title), color in zip(
        axes,
        WINDOWS,
        colors,
    ):
        panel = patient_points.loc[patient_points["panel"].eq(key)]

        axis.scatter(
            panel["map"],
            panel["sto2_mean"],
            s=46,
            alpha=0.78,
            color=color,
            edgecolors="white",
            linewidths=0.6,
            zorder=3,
        )

        correlation = (
            panel["map"].corr(panel["sto2_mean"])
            if len(panel) > 2
            else np.nan
        )

        subtitle = f"{len(panel)} patients"

        if not np.isnan(correlation):
            subtitle += f"; r = {correlation:.2f}"

        axis.set_title(f"{title}\n{subtitle}")

        axis.grid(
            True,
            color="#d9d9d9",
            linewidth=0.6,
            alpha=0.8,
        )

        axis.set_axisbelow(True)
        axis.set_xlabel("MAP (mmHg)")

    # Data-driven limits: per-patient summaries occupy a much narrower range
    # than raw samples, so the fixed 20-160 / 0-100 box would bunch every dot.
    if not patient_points.empty:
        for column, setter in (
            ("map", "set_xlim"),
            ("sto2_mean", "set_ylim"),
        ):
            low = patient_points[column].min()
            high = patient_points[column].max()
            pad = max((high - low) * 0.08, 1.0)
            getattr(axes[0], setter)(low - pad, high + pad)

    axes[0].set_ylabel(
        "Cerebral StO2 (%)\n(bilateral mean)"
    )

    figure.suptitle(
        "Cerebral StO2 versus MAP by clinical time window "
        f"- one point per patient ({how} within window)",
        y=1.03,
    )

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(figure)


def main() -> int:
    args = parse_args()

    args.outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    loaded_data = load_data(
        args.data,
        args.redcap,
    )

    plotted_data = assign_windows(loaded_data)

    patient_points = collapse_to_patient_points(
        plotted_data,
        how=args.agg,
        min_obs=args.min_obs,
    )

    audit = build_panel_audit(
        loaded_data,
        plotted_data,
        patient_points,
        args.min_obs,
    )

    points_path = args.outdir / "map_sto2_patient_points.csv"
    audit_path = args.outdir / "patient_panel_audit.csv"
    funnel_path = args.outdir / "panel_funnel.csv"
    missing_path = args.outdir / "missing_first_plot_patients.csv"
    figure_path = args.outdir / "map_sto2_by_window.png"

    patient_points.to_csv(points_path, index=False)
    audit.to_csv(audit_path, index=False)

    funnel = print_funnel(loaded_data, audit, patient_points)
    funnel.to_csv(funnel_path, index=False)

    missing_first_plot = audit.loc[
        audit["panel"].eq("pre_induction") & ~audit["in_plot"]
    ]

    missing_first_plot.to_csv(missing_path, index=False)

    make_plot(patient_points, figure_path, args.agg)

    patients_in_input = loaded_data["subject_id"].nunique()

    patients_in_first_plot = int(
        patient_points.loc[
            patient_points["panel"].eq("pre_induction"),
            "subject_id",
        ].nunique()
    )

    print(f"\nPatients in input file: {patients_in_input}")

    print(
        "Patients in first plot: "
        f"{patients_in_first_plot} "
        f"(expected approximately {EXPECTED_FIRST_PATIENTS})"
    )

    print(
        "Patients missing from first plot: "
        f"{len(missing_first_plot)}"
    )

    if patients_in_input < EXPECTED_FIRST_PATIENTS:
        print(
            "\nWARNING: The input file already contains fewer than "
            f"{EXPECTED_FIRST_PATIENTS} patients. Check the upstream "
            "compiled dataset or raw data."
        )

    elif patients_in_first_plot < EXPECTED_FIRST_PATIENTS:
        print(
            "\nWARNING: The input contains the expected cohort, but "
            "some patients were lost during REDCap matching or the "
            "preop time/MAP/StO2 checks. See the per-panel reasons above "
            f"and {audit_path.name}."
        )

    else:
        print(
            "\nSUCCESS: The first plot contains the expected number "
            "of patients."
        )

    print(f"\nFigure: {figure_path}")
    print(f"Per-patient points: {points_path}")
    print(f"Per-panel audit: {audit_path}")
    print(f"Funnel: {funnel_path}")
    print(f"Missing patients: {missing_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
