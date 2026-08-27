#!/usr/bin/env python3
"""Four cerebral-StO2 (y) versus PSI (x) scatter panels, one point per patient.

Panels, all anchored on induction (TIME = 0):

  1. Pre-induction IN THE OR : OR entry -> induction, EXCLUDING the induction
                               sample itself.
  2. Post-induction  0-5 min
  3. Post-induction  5-10 min
  4. Post-induction 10-20 min

Each patient contributes ONE point per panel (mean by default -- panel 2 in
particular contains acute PSI swings, so an average is the requested summary).

A patient needs BOTH PSI and StO2 to appear. Patients with PSI but no StO2
file are therefore excluded automatically, which is the intended behaviour.

Every run writes a full per-panel audit so no patient disappears silently:

  * patient_panel_audit.csv     - one row per patient per panel, with the
                                  reason that patient is or is not included.
  * panel_funnel.csv            - patient counts per panel.
  * sto2_psi_patient_points.csv - the plotted per-patient points.

Run with --explain to print the funnel and the exclusion reasons.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Panel key -> plot title. Bounds are seconds from induction; the first panel
# is anchored on REDCap OR-entry instead of a fixed offset.
WINDOWS = [
    ("or_pre_induction", "Pre-induction\n(in OR, before induction)"),
    ("post_0_5", "Post-induction\n0-5 min"),
    ("post_5_10", "Post-induction\n5-10 min"),
    ("post_10_20", "Post-induction\n10-20 min"),
]

POST_BOUNDS = {
    "post_0_5": (0, 300),
    "post_5_10": (300, 600),
    "post_10_20": (600, 1200),
}

PSI_MIN, PSI_MAX = 0.0, 100.0
STO2_MIN, STO2_MAX = 0.0, 100.0

# Research IDs excluded from the analysis by investigator decision.
EXCLUDED_IDS = frozenset({
    "IUMH2025121201",
    "IUMH2026020301",
    "IUMH2026031001",
})

# The study roster: every valid research ID. Anything in the timeseries that is
# not on this roster is a data-entry problem, not a patient, so it is reported
# rather than silently analysed. EXCLUDED_IDS are already removed here.
COHORT_IDS = frozenset({
    "IUMH2025120901",
    "IUMH2025120902",
    "IUMH2025121501",
    "IUMH2025121801",
    "IUMH2025122901",
    "IUMH2025123001",
    "IUMH2025123101",
    "IUMH2025123102",
    "IUMH2026010501",
    "IUMH2026010601",
    "IUMH2026010602",
    "IUMH2026010901",
    "IUMH2026011301",
    "IUMH2026011302",
    "IUMH2026011501",
    "IUMH2026011502",
    "IUMH2026011601",
    "IUMH2026011602",
    "IUMH2026012201",
    "IUMH2026012301",
    "IUMH2026012302",
    "IUMH2026012701",
    "IUMH2026012702",
    "IUMH2026012901",
    "IUMH2026012902",
    "IUMH2026013001",
    "IUMH2026013002",
    "IUMH2026020501",
    "IUMH2026021001",
    "IUMH2026021002",
    "IUMH2026021901",
    "IUMH2026021902",
    "IUMH2026022001",
    "IUMH2026022401",
    "IUMH2026030301",
    "IUMH2026030302",
    "IUMH2026030501",
    "IUMH2026030502",
    "IUMH2026030401",
    "IUMH2026030602",
    "IUMH2026030901",
    "IUMH2026031002",
    "IUMH2026031701",
    "IUMH2026031702",
    "IUMH2026031801",
    "IUMH2026031901",
    "IUMH2026031902",
    "IUMH2026032001",
    "IUMH2026032002",
    "IUMH2026032401",
    "IUMH2026040201",
    "IUUH2026030601",
    "IUUH2026030602",
    "IUUH2026030603",
    "IUUH2026030901",
    "IUUH2026030902",
    "IUMH2026041701",
    "IUMH2026041702",
    "IUMH2026042101",
    "IUMH2026042102",
    "IUMH2026042301",
    "IUMH2026042401",
    "IUMH2026042701",
    "IUMH2026042702",
    "IUMH2026042801",
    "IUMH2026042802",
    "IUMH2026050501",
    "IUUH2026031001",
    "IUMH2026050502",
    "IUMH2026051201",
    "IUMH2026051202",
    "IUMH2026051203",
    "IUMH2026051901",
    "IUMH2026052001",
    "IUMH2026052002",
    "IUMH2026052101",
    "IUUH2026042801",
    "IUUH2026050601",
    "IUUH2026050602",
    "IUMH2026052201",
    "IUMH2026052202",
    "IUMH2026060101",
    "IUMH2026060102",
    "IUMH2026061601",
    "IUMH2026061701",
    "IUMH2026061702",
    "IUMH2026061901",
    "IUMH2026062301",
    "IUMH2026062302",
    "IUMH2026062501",
    "IUMH2026062502",
    "IUMH2026063001",
    "IUMH2026063002",
})

# Known REDCap ID typos -> the true research ID. Without these the affected
# patients fail the REDCap merge and vanish from the preop panel. Every value
# must be a member of COHORT_IDS.
REDCAP_ID_ALIASES = {
    "IUMH202601601": "IUMH2026011601",
    "IUMH2026010601-20260105": "IUMH2026010501",
}

EXPECTED_PATIENTS = len(COHORT_IDS)

_BAD_ALIASES = set(REDCAP_ID_ALIASES.values()) - COHORT_IDS
if _BAD_ALIASES:
    raise ValueError(
        f"REDCAP_ID_ALIASES point at IDs that are not on the roster: "
        f"{sorted(_BAD_ALIASES)}"
    )


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
        help="REDCap export containing the OR-entry time.",
    )

    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("/N/project/Analgesia_BDproject/PR/figures/4plots_PSI"),
    )

    parser.add_argument(
        "--agg",
        choices=["mean", "median"],
        default="mean",
        help="How to collapse each patient's samples within a panel. "
             "Mean is the default, as requested for the acute 0-5 min window.",
    )

    parser.add_argument(
        "--min-obs",
        type=int,
        default=3,
        help="Minimum paired PSI+StO2 samples for a patient to contribute.",
    )

    parser.add_argument(
        "--require-bilateral",
        action="store_true",
        help="Also require brain_bilateral_usable == 1. OFF by default so "
             "patients with one usable cerebral channel are still included; "
             "the audit reports what this filter would cost.",
    )

    parser.add_argument(
        "--no-roster-filter",
        action="store_true",
        help="Report roster mismatches but analyse every ID found.",
    )

    parser.add_argument(
        "--explain",
        action="store_true",
        help="Print the per-panel funnel and exclusion reasons.",
    )

    return parser.parse_args()


def normalize_subject_id(values: pd.Series) -> pd.Series:
    """Standardize formatting without changing the hospital prefix.

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


def reconcile_with_roster(
    data: pd.DataFrame,
    apply_filter: bool = True,
) -> pd.DataFrame:
    """Cross-check timeseries IDs against the roster; report every mismatch."""
    present = set(data["subject_id"].dropna().unique())

    excluded_present = sorted(present & EXCLUDED_IDS)
    off_roster = sorted(present - COHORT_IDS - EXCLUDED_IDS)
    missing_from_data = sorted(COHORT_IDS - present)

    print("\n" + "=" * 68)
    print("ROSTER RECONCILIATION")
    print("=" * 68)
    print(f"  roster size (after exclusions) ................ {len(COHORT_IDS)}")
    print(f"  roster patients present in timeseries ........ {len(present & COHORT_IDS)}")
    print(f"  roster patients with NO timeseries data ...... {len(missing_from_data)}")
    print(f"  investigator-excluded IDs found .............. {len(excluded_present)}")
    print(f"  IDs in data but NOT on roster ................ {len(off_roster)}")

    for subject_id in excluded_present:
        print(f"      [excluded by investigator] {subject_id}")
    for subject_id in off_roster:
        rows = int(data["subject_id"].eq(subject_id).sum())
        print(f"      [not on roster - likely typo] {subject_id} ({rows} rows)")
    for subject_id in missing_from_data:
        print(f"      [no timeseries data] {subject_id}")

    if not apply_filter:
        return data

    return data.loc[data["subject_id"].isin(COHORT_IDS)].copy()


def find_redcap_column(redcap: pd.DataFrame, prefix: str) -> str | None:
    """Find a REDCap column by prefix, tolerating stray whitespace."""
    prefix = prefix.strip().lower()
    for column in redcap.columns:
        if str(column).strip().lower().startswith(prefix):
            return column
    return None


def find_redcap_id_column(redcap: pd.DataFrame) -> str | None:
    """The column with the MOST ID-shaped values, so stray text cannot win."""
    best_column, best_count = None, 0
    for column in redcap.columns:
        count = int(
            normalize_subject_id(redcap[column])
            .str.fullmatch(r"IU(?:MH|UH)\d+", na=False)
            .sum()
        )
        if count > best_count:
            best_column, best_count = column, count
    return best_column


def load_data(
    data_path: Path,
    redcap_path: Path,
    roster_filter: bool = True,
    require_bilateral: bool = False,
) -> pd.DataFrame:
    data = pd.read_csv(data_path, low_memory=False)

    required = {
        "subject_id",
        "timestamp_local",
        "psi_median",
        "sto2_mean",
        "seconds_from_induction",
    }
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Input is missing required columns: {sorted(missing)}")

    data["subject_id_raw"] = data["subject_id"].astype("string")
    data["subject_id"] = apply_id_aliases(
        normalize_subject_id(data["subject_id"])
    )

    data = reconcile_with_roster(data, apply_filter=roster_filter)

    data["timestamp_local"] = pd.to_datetime(
        data["timestamp_local"], errors="coerce"
    )

    # Sedline writes "-" where PSI is unavailable; to_numeric turns those into
    # proper missing values rather than dropping the row outright.
    for column in ["psi_median", "sto2_mean", "seconds_from_induction"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    if "brain_bilateral_usable" not in data.columns:
        data["brain_bilateral_usable"] = 1
    data["brain_bilateral_usable"] = pd.to_numeric(
        data["brain_bilateral_usable"], errors="coerce"
    )

    # A point needs BOTH signals. Patients with PSI but no StO2 file drop out
    # here, which is the intended mismatch exclusion.
    data["valid_pair"] = (
        data["psi_median"].between(PSI_MIN, PSI_MAX)
        & data["sto2_mean"].between(STO2_MIN, STO2_MAX)
    )
    if require_bilateral:
        data["valid_pair"] &= data["brain_bilateral_usable"].eq(1)

    # ---- REDCap OR-entry time (panel 1 boundary) ------------------------- #
    redcap_raw = pd.read_csv(redcap_path, low_memory=False)
    or_entry_column = find_redcap_column(
        redcap_raw, "What time did the patient enter the OR?"
    )
    if or_entry_column is None:
        print(
            "\nWARNING: no 'What time did the patient enter the OR?' column. "
            "Panel 1 falls back to every pre-induction sample."
        )

    id_column = find_redcap_id_column(redcap_raw)
    if id_column is None:
        raise ValueError("Could not find a research-ID column in the REDCap export.")

    lookup = pd.DataFrame({
        "subject_id": apply_id_aliases(
            normalize_subject_id(redcap_raw[id_column])
        ),
        "or_entry": (
            pd.to_datetime(redcap_raw[or_entry_column], errors="coerce")
            if or_entry_column else pd.NaT
        ),
    })
    lookup = (
        lookup.loc[lookup["subject_id"].notna()]
        .sort_values("or_entry")
        .drop_duplicates("subject_id", keep="first")
    )

    data = data.merge(lookup, on="subject_id", how="left", validate="many_to_one")

    matched = int(
        data.groupby("subject_id")["or_entry"].apply(
            lambda column: bool(column.notna().any())
        ).sum()
    )
    print(f"\n  REDCap OR-entry time matched for {matched} patients")

    return data


def _clock_seconds(values: pd.Series) -> pd.Series:
    """Time-of-day in seconds since midnight."""
    return (
        values.dt.hour * 3600 + values.dt.minute * 60 + values.dt.second
    )


def or_entry_seconds_from_induction(data: pd.DataFrame) -> pd.Series:
    """OR entry expressed as seconds from induction (a negative number).

    REDCap stores a clock time, so it is anchored against each patient's
    induction datetime, recovered from the timeseries itself
    (timestamp_local minus seconds_from_induction). Differences wrap into
    +/-12 h so a case spanning midnight still resolves correctly.
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

    delta = _clock_seconds(data["or_entry"]) - _clock_seconds(induction_timestamp)
    delta = ((delta + 43200) % 86400) - 43200
    return delta.astype(float)


def assign_windows(data: pd.DataFrame) -> pd.DataFrame:
    """Tag every valid sample with the panel it belongs to."""
    induction = data["seconds_from_induction"]
    valid = data.loc[data["valid_pair"]].copy()
    valid_induction = valid["seconds_from_induction"]

    # Panel 1: OR entry -> induction, strictly BEFORE the induction sample.
    or_entry_rel = or_entry_seconds_from_induction(valid)
    in_or_pre = valid_induction.lt(0)
    in_or_pre &= or_entry_rel.isna() | valid_induction.ge(or_entry_rel)

    panels = []
    first = valid.loc[in_or_pre].copy()
    first["panel"] = "or_pre_induction"
    panels.append(first)

    for panel_key, (low, high) in POST_BOUNDS.items():
        piece = valid.loc[
            valid_induction.ge(low) & valid_induction.lt(high)
        ].copy()
        piece["panel"] = panel_key
        panels.append(piece)

    return pd.concat(panels, ignore_index=True)


def collapse_to_patient_points(
    plotted_data: pd.DataFrame,
    how: str = "mean",
    min_obs: int = 3,
) -> pd.DataFrame:
    """Collapse each patient's in-window samples to ONE point per panel."""
    if plotted_data.empty:
        return pd.DataFrame(
            columns=["subject_id", "panel", "psi_median", "sto2_mean", "n_obs"]
        )

    grouped = plotted_data.groupby(
        ["subject_id", "panel"], as_index=False
    ).agg(
        psi_median=("psi_median", how),
        sto2_mean=("sto2_mean", how),
        n_obs=("psi_median", "size"),
        psi_n=("psi_median", "count"),
        sto2_n=("sto2_mean", "count"),
    )

    enough = grouped["psi_n"].ge(min_obs) & grouped["sto2_n"].ge(min_obs)
    dropped = int((~enough).sum())
    if dropped:
        print(
            f"\nNote: {dropped} patient-panel points dropped for having fewer "
            f"than {min_obs} paired samples."
        )

    return grouped.loc[enough].reset_index(drop=True)


def build_panel_audit(
    loaded_data: pd.DataFrame,
    plotted_data: pd.DataFrame,
    patient_points: pd.DataFrame,
    min_obs: int,
) -> pd.DataFrame:
    """One row per patient per panel, naming why they are in or out."""
    source = loaded_data.copy()
    source["_or_entry_rel"] = or_entry_seconds_from_induction(source)
    induction = source["seconds_from_induction"]

    per_patient = source.groupby("subject_id").agg(
        input_rows=("subject_id", "size"),
        rows_with_psi=("psi_median", "count"),
        rows_with_sto2=("sto2_mean", "count"),
        rows_valid_pair=("valid_pair", "sum"),
        rows_bilateral_usable=(
            "brain_bilateral_usable", lambda column: int(column.eq(1).sum())
        ),
        has_or_entry=("_or_entry_rel", lambda column: bool(column.notna().any())),
        rows_pre_induction=(
            "seconds_from_induction", lambda column: int(column.lt(0).sum())
        ),
        rows_post_induction=(
            "seconds_from_induction", lambda column: int(column.ge(0).sum())
        ),
        max_minutes_after_induction=(
            "seconds_from_induction", lambda column: round(column.max() / 60, 1)
        ),
    )

    obs_counts = plotted_data.groupby(["subject_id", "panel"]).size()
    plotted_keys = set(
        map(tuple, patient_points[["subject_id", "panel"]].to_numpy())
    ) if not patient_points.empty else set()

    rows = []
    for subject_id, summary in per_patient.iterrows():
        for panel_key, _label in WINDOWS:
            window_rows = int(obs_counts.get((subject_id, panel_key), 0))
            included = (subject_id, panel_key) in plotted_keys
            rows.append({
                "subject_id": subject_id,
                "panel": panel_key,
                "window_rows": window_rows,
                "in_plot": included,
                "reason": _exclusion_reason(
                    summary, panel_key, window_rows, included, min_obs
                ),
                **summary.to_dict(),
            })

    return pd.DataFrame(rows)


def _exclusion_reason(
    summary: pd.Series,
    panel_key: str,
    window_rows: int,
    included: bool,
    min_obs: int,
) -> str:
    """Name the FIRST stage at which this patient fell out of this panel."""
    if included:
        return "included"

    if window_rows > 0:
        return f"fewer than {min_obs} paired PSI+StO2 samples in window"

    if summary["rows_with_sto2"] == 0:
        return "no StO2 data at all (unmatched patient - expected exclusion)"

    if summary["rows_with_psi"] == 0:
        return "no PSI data at all"

    if summary["rows_valid_pair"] == 0:
        return "PSI and StO2 never valid on the same 20-second sample"

    if panel_key == "or_pre_induction":
        if summary["rows_pre_induction"] == 0:
            return "no pre-induction samples (record starts at induction)"
        return "no valid PSI+StO2 pair between OR entry and induction"

    if summary["rows_post_induction"] == 0:
        return "no post-induction samples"

    return (
        "record does not reach this window "
        f"(ends {summary['max_minutes_after_induction']} min after induction)"
    )


def print_funnel(loaded_data: pd.DataFrame, audit: pd.DataFrame) -> pd.DataFrame:
    """Print and return the per-panel patient counts."""
    total = loaded_data["subject_id"].nunique()
    per_patient = audit.drop_duplicates("subject_id")

    print("\n" + "=" * 68)
    print("COHORT FUNNEL (unique patients)")
    print("=" * 68)
    print(f"  patients in timeseries input .................. {total}")
    print(
        "  ... with any PSI ............................. "
        f"{int(per_patient['rows_with_psi'].gt(0).sum())}"
    )
    print(
        "  ... with any StO2 ............................ "
        f"{int(per_patient['rows_with_sto2'].gt(0).sum())}"
    )
    print(
        "  ... with a valid PSI+StO2 pair ............... "
        f"{int(per_patient['rows_valid_pair'].gt(0).sum())}"
    )
    print(
        "  ... with a REDCap OR-entry time .............. "
        f"{int(per_patient['has_or_entry'].sum())}"
    )
    print(
        "  ... with >=1 bilateral-usable row ............ "
        f"{int(per_patient['rows_bilateral_usable'].gt(0).sum())}"
    )

    print("\n" + "=" * 68)
    print("PER-PANEL RESULT")
    print("=" * 68)

    funnel_rows = []
    for panel_key, label in WINDOWS:
        panel_audit = audit.loc[audit["panel"].eq(panel_key)]
        n_in = int(panel_audit["in_plot"].sum())
        print(f"\n  {label.replace(chr(10), ' ')}: {n_in}/{total} patients")
        excluded = panel_audit.loc[~panel_audit["in_plot"], "reason"]
        for reason, count in excluded.value_counts().items():
            print(f"      {count:>4}  {reason}")
        funnel_rows.append({
            "panel": panel_key,
            "patients_in_plot": n_in,
            "patients_total": total,
        })

    return pd.DataFrame(funnel_rows)


def make_plot(patient_points: pd.DataFrame, output_path: Path, how: str) -> None:
    figure, axes = plt.subplots(
        1, 4, figsize=(19, 5), sharex=True, sharey=True
    )
    colors = ["#2f5597", "#548235", "#c55a11", "#7030a0"]

    for axis, (key, title), color in zip(axes, WINDOWS, colors):
        panel = patient_points.loc[patient_points["panel"].eq(key)]

        axis.scatter(
            panel["psi_median"], panel["sto2_mean"],
            s=48, alpha=0.78, color=color,
            edgecolors="white", linewidths=0.6, zorder=3,
        )

        correlation = (
            panel["psi_median"].corr(panel["sto2_mean"])
            if len(panel) > 2 else np.nan
        )
        subtitle = f"{len(panel)} patients"
        if not np.isnan(correlation):
            subtitle += f"; r = {correlation:.2f}"
        axis.set_title(f"{title}\n{subtitle}")

        axis.grid(True, color="#d9d9d9", linewidth=0.6, alpha=0.8)
        axis.set_axisbelow(True)
        axis.set_xlabel("PSI")

    # Per-patient summaries occupy a far narrower range than raw samples, so
    # data-driven limits keep the dots readable instead of bunched.
    if not patient_points.empty:
        for column, setter in (
            ("psi_median", "set_xlim"), ("sto2_mean", "set_ylim")
        ):
            low = patient_points[column].min()
            high = patient_points[column].max()
            pad = max((high - low) * 0.08, 1.0)
            getattr(axes[0], setter)(low - pad, high + pad)

    axes[0].set_ylabel("Cerebral StO2 (%)")
    figure.suptitle(
        "Cerebral StO2 versus PSI by clinical time window "
        f"- one point per patient ({how} within window)",
        y=1.03,
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> int:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    loaded_data = load_data(
        args.data,
        args.redcap,
        roster_filter=not args.no_roster_filter,
        require_bilateral=args.require_bilateral,
    )

    plotted_data = assign_windows(loaded_data)
    patient_points = collapse_to_patient_points(
        plotted_data, how=args.agg, min_obs=args.min_obs
    )
    audit = build_panel_audit(
        loaded_data, plotted_data, patient_points, args.min_obs
    )

    points_path = args.outdir / "sto2_psi_patient_points.csv"
    audit_path = args.outdir / "patient_panel_audit.csv"
    funnel_path = args.outdir / "panel_funnel.csv"
    figure_path = args.outdir / "sto2_vs_psi_by_window.png"

    patient_points.to_csv(points_path, index=False)
    audit.to_csv(audit_path, index=False)

    funnel = print_funnel(loaded_data, audit)
    funnel.to_csv(funnel_path, index=False)

    make_plot(patient_points, figure_path, args.agg)

    patients_in_input = loaded_data["subject_id"].nunique()
    print(f"\nPatients in input file: {patients_in_input} "
          f"(roster has {EXPECTED_PATIENTS})")

    if patients_in_input < EXPECTED_PATIENTS:
        print(
            f"\nWARNING: the timeseries contains only {patients_in_input} of "
            f"the {EXPECTED_PATIENTS} roster patients. That ceiling is upstream "
            "of this script - see the roster reconciliation above."
        )

    print(f"\nFigure: {figure_path}")
    print(f"Per-patient points: {points_path}")
    print(f"Per-panel audit: {audit_path}")
    print(f"Funnel: {funnel_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
