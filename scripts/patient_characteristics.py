#!/usr/bin/env python3
"""Table 1 — patient characteristics for the cerebral oxygenation analysis.

This is Table 1 only. It fits nothing, so it needs no statsmodels and no
model decisions: pandas, numpy and matplotlib are enough. It reads the same
analysis dataset the mixed models use and writes

    table1_patient_characteristics.csv   for the manuscript
    table1_patient_characteristics.png   for slides

RUNNING IT

    python patient_characteristics.py

With no arguments it reads the analysis parquet at the path in DEFAULT_DATA
below. Point it somewhere else with --data:

    python patient_characteristics.py --data /path/to/other_file.parquet
    python patient_characteristics.py --inspect       # resolve columns only

WHAT IT REPORTS

    Patients        n, epochs, epochs per patient, age, sex, BMI, ASA
    Comorbidities   every recorded history column it can find
    Baseline        cerebral O2 and physiology before induction, per patient
    Medications     how many patients were exposed, and the dose among those
    Completeness    how much of each signal survives inside the window

HOW IT HANDLES A COLUMN IT CANNOT FIND
    It carries on and says so. A missing time column costs the baseline
    section and the window restriction; a missing outcome costs the cerebral
    O2 rows; neither stops the rest of the table. Only the patient identifier
    is required, because without it there are no patients to count. Run
    --inspect first if you want to see what resolved to what before trusting
    any of it.

EVERY NUMBER IS PER PATIENT, NOT PER EPOCH
    The dataset holds one row per patient per epoch, so a patient with a long
    recording contributes hundreds of rows. Summarising those rows directly
    would describe the recordings rather than the cohort, and would weight
    whoever was monitored longest most heavily. Every characteristic here is
    collapsed to one value per patient first.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# The analysis dataset the mixed models are fitted on.
DEFAULT_DATA = Path(
    "/N/project/Analgesia_BDproject/JT/processed_data/"
    "lmm_model/lmm_analysis_df_processed.parquet")

# Where the two files land, unless --outdir says otherwise.
DEFAULT_OUTDIR = Path(
    "/N/project/Analgesia_BDproject/PR/scripts_PR/"
    "StO2 LMM with meds/Output")


# --------------------------------------------------------------------------- #
# What the table is made of
# --------------------------------------------------------------------------- #

# Continuous physiology, with the unit shown in the row label. Several
# spellings of the cerebral oximetry channels are tried so that a file with
# separate left and right channels reports both — the question of whether the
# two sides behave alike cannot be asked of a table that has already averaged
# them.
PHYSIOLOGY: dict[str, dict] = {
    "so2_mean":  {"label": "Cerebral O₂, both sides", "unit": "%", "digits": 1},
    "so2_left":  {"label": "Cerebral O₂, left", "unit": "%", "digits": 1},
    "so2_right": {"label": "Cerebral O₂, right", "unit": "%", "digits": 1},
    "map":       {"label": "MAP", "unit": "mmHg", "digits": 0},
    "ci":        {"label": "Cardiac index", "unit": "L/min/m²", "digits": 2},
    "pr":        {"label": "Pulse rate", "unit": "bpm", "digits": 0},
    "psi":       {"label": "PSi", "unit": "units", "digits": 0},
}

# Extra spellings to try for each, beyond the key itself.
PHYSIOLOGY_ALIASES: dict[str, tuple[str, ...]] = {
    "so2_mean": ("sto2_mean", "scto2_mean", "so2_mean_raw", "sto2", "so2"),
    "so2_left": ("sto2_left", "so2_l", "sto2_l", "scto2_left", "so2_ch1",
                 "sto2_ch1"),
    "so2_right": ("sto2_right", "so2_r", "sto2_r", "scto2_right", "so2_ch2",
                  "sto2_ch2"),
    "map": ("mean_arterial_pressure", "map_mmhg", "mbp"),
    "ci": ("cardiac_index", "ci_l_min_m2"),
    "pr": ("pulse_rate", "heart_rate", "hr"),
    "psi": ("psi_value", "patient_state_index", "psi_index"),
}

PATIENT_NUMERIC: dict[str, dict] = {
    "age": {"label": "Age, years", "digits": 1},
    "bmi": {"label": "BMI, kg/m²", "digits": 1},
    "weight_kg": {"label": "Weight, kg", "digits": 1},
    "height_cm": {"label": "Height, cm", "digits": 1},
}
PATIENT_CATEGORICAL: dict[str, str] = {
    "sex": "Sex",
    "asa": "ASA class",
    "surgery_type": "Surgery type",
}

# Exactly the medication columns Model C uses.
BOLUS_COLUMNS: dict[str, str] = {
    "prop_cumulative_bolus_mg": "Propofol, mg",
    "ket_cumulative_bolus_mg": "Ketamine, mg",
    "midaz_cumulative_bolus_mg": "Midazolam, mg",
    "fent_cumulative_bolus_mcg": "Fentanyl, mcg",
    "ephedrine_cumulative_bolus_mg": "Ephedrine, mg",
    "phenyl_cumulative_bolus_mcg": "Phenylephrine, mcg",
    "rocuronium_cumulative_bolus_mg": "Rocuronium, mg",
}
INFUSION_COLUMNS: dict[str, str] = {
    "prop_infusion_mcg_kg_min": "Propofol, mcg/kg/min",
    "ket_infusion_mg_kg_hr": "Ketamine, mg/kg/hr",
    "remi_infusion_mcg_kg_min": "Remifentanil, mcg/kg/min",
    "phenyl_infusion_mcg_min": "Phenylephrine, mcg/min",
    "norepi_infusion_mcg_min": "Norepinephrine, mcg/min",
}

PATIENT_ALIASES = ("screening_id", "patient_id", "subject_id", "patient",
                   "id", "record_id", "study_id")
# What the axis is measured from. Induction is the one the analysis window is
# defined against; the others are real columns in this dataset and are offered
# only if nothing better exists.
TIME_ORIGIN_HINTS: tuple[tuple[str, int], ...] = (
    ("induction", 100), ("intubation", 60), ("surgerystart", 50),
    ("surgery", 45), ("incision", 45), ("anesthesiastart", 45),
)

# Generic names for a relative axis with no origin word in them.
TIME_GENERIC_HINTS: tuple[tuple[str, int], ...] = (
    ("timemin", 70), ("elapsed", 55), ("reltime", 55), ("timerel", 55),
    ("tmin", 50), ("epochmin", 50), ("timeminutes", 60),
)

# Suffixes marking a working copy rather than the column to analyse. They lose
# to the plain name so that 'seconds_from_induction' beats
# 'seconds_from_induction_original'.
TIME_VARIANT_SUFFIXES = ("original", "raw", "alt", "backup", "old", "prev",
                         "med", "v1", "v2", "test", "tmp", "temp", "check")

# Names that mean a wall clock, not an offset. A column called
# 'timestamp_local' starts with 'time', which is exactly how a prefix match
# picks a microsecond epoch clock and then reports that no epoch lies between
# 0 and 15 minutes. These are refused outright.
CLOCK_NAME_MARKERS = ("timestamp", "unix", "epoch_time", "utc", "datetime",
                      "date", "clock", "wall", "_local", "localtime")

# A relative-minute column has to fall in a believable range. Induction
# windows are minutes, not days; anything wider than this is a clock.
TIME_PLAUSIBLE_LOW, TIME_PLAUSIBLE_HIGH = -720.0, 2880.0

# Prefixes marking a recorded history item, stripped for display.
HISTORY_PREFIXES = ("pmh_", "hx_", "comorbid_", "history_", "cc_")
KNOWN_TERMS = {
    "cad": "CAD", "chf": "CHF", "copd": "COPD", "ckd": "CKD", "cva": "CVA",
    "tia": "TIA", "dm": "DM", "htn": "HTN", "afib": "AF", "mi": "MI",
    "osa": "OSA", "pvd": "PVD", "asa": "ASA", "bmi": "BMI", "psi": "PSi",
    "map": "MAP", "esrd": "ESRD", "pad": "PAD", "gerd": "GERD",
    "ckd3": "CKD 3", "hld": "HLD", "pud": "PUD", "ptsd": "PTSD",
}

TRUE_WORDS = {"yes", "y", "true", "t", "1", "checked", "present", "positive"}


# --------------------------------------------------------------------------- #
# Finding columns
# --------------------------------------------------------------------------- #

def normalize(name: object) -> str:
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def find_column(frame: pd.DataFrame, wanted: str,
                extra: tuple[str, ...] = ()) -> str | None:
    """Exact match first, then a normalized match, then a prefix match."""
    if wanted in frame.columns:
        return wanted
    candidates = (wanted,) + extra
    lookup = {normalize(column): column for column in frame.columns}
    for candidate in candidates:
        hit = lookup.get(normalize(candidate))
        if hit is not None:
            return hit
    for candidate in candidates:
        key = normalize(candidate)
        for normalized, column in lookup.items():
            if normalized.startswith(key):
                return column
    return None


def summarise_range(values: pd.Series) -> str:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return "no numeric values"
    return f"{numeric.min():,.1f} to {numeric.max():,.1f}"


def time_unit_in_minutes(key: str) -> tuple[float, str] | None:
    """How many minutes one unit of this column is, read off its name.

    A column called seconds_from_induction is the analysis axis; it is simply
    not in minutes. Checking the range before the unit is what made an earlier
    version reject it as 'too wide to be minutes', which was true and useless.
    Order matters here: 'millisecond' contains 'second'.
    """
    if "millisecond" in key or "msec" in key:
        return 1.0 / 60000.0, "milliseconds"
    if "microsecond" in key or "usec" in key:
        return 1.0 / 60000000.0, "microseconds"
    if "second" in key or "secs" in key or "_sec" in key or key.endswith("sec"):
        return 1.0 / 60.0, "seconds"
    if "hour" in key or key.endswith("hr"):
        return 60.0, "hours"
    if "minute" in key or "min" in key:
        return 1.0, "minutes"
    return None


def rank_time_columns(frame: pd.DataFrame) -> list[tuple]:
    """Score every column on how much it looks like time relative to an event.

    Three things have to line up, and an earlier version checked only the
    first two. The NAME has to name an origin, which is how 'timestamp_local'
    is ruled out. The name also has to give a UNIT, because the column may be
    in seconds. Only then can the VALUES be checked, in minutes, against what
    an induction window could plausibly span.

    Returns (score, column, minutes_per_unit, unit_name, description), best
    first.
    """
    scored: list[tuple] = []
    for column in frame.columns:
        key = normalize(column)
        if any(marker.replace("_", "") in key for marker in CLOCK_NAME_MARKERS):
            continue
        series = frame[column]
        if pd.api.types.is_datetime64_any_dtype(series):
            continue
        numeric = pd.to_numeric(series, errors="coerce").dropna()
        if numeric.empty or numeric.nunique() < 3:
            continue

        unit = time_unit_in_minutes(key)
        score = 0.0
        for hint, weight in TIME_ORIGIN_HINTS:
            if hint in key:
                score = max(score, float(weight))
        if score == 0.0:
            for hint, weight in TIME_GENERIC_HINTS:
                if key == hint:
                    score = max(score, weight + 20.0)
                elif hint in key:
                    score = max(score, float(weight))
        if score == 0.0:
            continue
        if unit is None:
            # An origin but no unit named: assume minutes, and prefer anything
            # that does say.
            unit, score = (1.0, "minutes (assumed)"), score - 15.0
        minutes_per_unit, unit_name = unit

        low = float(numeric.min()) * minutes_per_unit
        high = float(numeric.max()) * minutes_per_unit
        if not (TIME_PLAUSIBLE_LOW <= low and high <= TIME_PLAUSIBLE_HIGH):
            continue

        # A working copy loses to the plain name it was copied from.
        if any(key.endswith(suffix) for suffix in TIME_VARIANT_SUFFIXES):
            score -= 30.0
        # A real induction axis spans zero and reaches into the window.
        if low <= 0.0 <= high:
            score += 25.0
        if low <= 0.0 and high >= 10.0:
            score += 15.0

        description = (f"{low:,.1f} to {high:,.1f} min"
                       + (f", from {unit_name}" if minutes_per_unit != 1.0
                          else ""))
        scored.append((score, str(column), minutes_per_unit, unit_name,
                       description))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return scored


def numeric_column_report(frame: pd.DataFrame, patient: str | None = None,
                          limit: int = 12) -> list[str]:
    """Columns that could be a time axis, with why each one is or is not.

    Listing every numeric column is no help — a file has dozens, and MAP in
    mmHg has the same range as minutes. A time axis has a signature: it rises
    as the recording goes on, so within one patient it never goes backwards.
    That is what this looks for, and it reports the clock columns too, with
    their ranges, so it is clear why they were passed over.
    """
    lines: list[str] = []
    rising: list[tuple[str, str]] = []
    clocks: list[tuple[str, str]] = []

    seen: set[str] = set()
    for column in frame.columns:
        name = str(column)
        key = normalize(column)
        # The identifier is not the time axis, and asking for it alongside
        # itself would build a two-column frame with one label twice.
        if patient is not None and name == str(patient):
            continue
        if name in seen:                 # a file may repeat a column name
            continue
        seen.add(name)
        series = frame[column]
        if isinstance(series, pd.DataFrame):
            continue
        looks_timeish = any(word in key for word in
                            ("time", "min", "clock", "stamp", "elapsed",
                             "sec", "epoch"))
        if pd.api.types.is_datetime64_any_dtype(series):
            if looks_timeish:
                clocks.append((name, "datetime, not an offset in minutes"))
            continue
        numeric = pd.to_numeric(series, errors="coerce")
        if numeric.notna().sum() < 3 or numeric.nunique() < 3:
            continue
        low, high = float(numeric.min()), float(numeric.max())
        span = f"range {low:,.1f} to {high:,.1f}"

        if not (TIME_PLAUSIBLE_LOW <= low and high <= TIME_PLAUSIBLE_HIGH):
            if looks_timeish:
                clocks.append((name, f"{span} — too wide to be minutes"))
            continue
        # Known medication and covariate columns are not the time axis, and a
        # cumulative dose rises within a patient exactly like a clock does.
        if (key in {normalize(c) for c in BOLUS_COLUMNS}
                or key in {normalize(c) for c in INFUSION_COLUMNS}
                or key in {normalize(c) for c in PATIENT_NUMERIC}
                or key in {normalize(c) for c in PATIENT_CATEGORICAL}):
            continue

        # A time axis rises within every patient AND takes many values there.
        # That second half is what rules out age, which never moves, and a
        # cumulative bolus, which moves once and then holds.
        if patient is not None and patient in frame.columns:
            ids = frame[patient]
            if isinstance(ids, pd.DataFrame):      # repeated column label
                ids = ids.iloc[:, 0]
            keep = ids.isin(pd.Index(ids.dropna().unique()[:12]))
            grouped = numeric[keep.to_numpy()].groupby(
                ids[keep.to_numpy()].to_numpy())
            if grouped.ngroups == 0:
                continue
            rises = bool(grouped.apply(
                lambda s: s.dropna().is_monotonic_increasing).all())
            varies = float(grouped.nunique().median()) >= 10.0
            if not (rises and varies):
                continue
        elif not looks_timeish:
            continue
        rising.append((name, span))

    if rising:
        lines.append("  Rises steadily within each patient, so it could be "
                     "the time axis:")
        for name, span in rising[:limit]:
            lines.append(f"    {name:<38} {span}")
    if clocks:
        lines.append("  Time-like but not an offset in minutes, so not used:")
        for name, why in clocks[:limit]:
            lines.append(f"    {name:<38} {why}")
    return lines


def clock_to_seconds(series: pd.Series) -> tuple[pd.Series, str]:
    """Read a wall clock as seconds, whatever resolution it was stored in.

    A timestamp arrives as a datetime, or as an integer count since 1970 in
    seconds, milliseconds, microseconds or nanoseconds. Which one is decided
    by magnitude, because a count of seconds in this decade is about 1.8e9 and
    each finer unit multiplies that by a thousand.
    """
    if pd.api.types.is_datetime64_any_dtype(series):
        return series.astype("int64") / 1e9, "datetime"
    numeric = pd.to_numeric(series, errors="coerce")
    typical = float(numeric.abs().median()) if numeric.notna().any() else 0.0
    if typical > 1e17:
        return numeric / 1e9, "nanoseconds since 1970"
    if typical > 1e14:
        return numeric / 1e6, "microseconds since 1970"
    if typical > 1e11:
        return numeric / 1e3, "milliseconds since 1970"
    return numeric.astype(float), "seconds"


def derive_minutes(frame: pd.DataFrame, clock_column: str, patient: str,
                   induction_column: str | None) -> tuple[pd.Series, str]:
    """Minutes from an origin, built from a wall clock.

    The origin is the induction timestamp when the file carries one. When it
    does not, it is each patient's own first epoch, and the result is minutes
    from the start of that patient's recording — which is the same thing only
    if the recording starts at induction. The caller is told which it got,
    because a baseline computed against the wrong origin is not a baseline.
    """
    seconds, unit = clock_to_seconds(frame[clock_column])
    if induction_column is not None:
        origin, _ = clock_to_seconds(frame[induction_column])
        origin = origin.groupby(frame[patient]).transform("min")
        note = (f"minutes from the induction time in '{induction_column}' "
                f"(clock '{clock_column}' read as {unit})")
    else:
        origin = seconds.groupby(frame[patient]).transform("min")
        note = (f"minutes from each patient's FIRST EPOCH in '{clock_column}' "
                f"(read as {unit}) — not from induction unless the recording "
                f"starts there")
    return (seconds - origin) / 60.0, note


def pretty_label(column: object) -> str:
    """Turn a column name like 'pmh_cad' into something printable: 'CAD'."""
    text = str(column).strip()
    lowered = text.lower()
    for prefix in HISTORY_PREFIXES:
        if lowered.startswith(prefix):
            text = text[len(prefix):]
            break
    words = [w for w in text.replace("-", " ").replace("_", " ").split() if w]
    if not words:
        return str(column)
    shown = [KNOWN_TERMS.get(w.lower(), w) for w in words]
    if shown[0] not in KNOWN_TERMS.values():
        shown[0] = shown[0].capitalize()
    return " ".join(shown)


# --------------------------------------------------------------------------- #
# Summarising
# --------------------------------------------------------------------------- #

def describe_numeric(values: pd.Series, digits: int = 1) -> str:
    """Median (IQR), or a plain note when nothing is recorded."""
    values = pd.to_numeric(values, errors="coerce").dropna()
    if values.empty:
        return "not recorded"
    return (f"{values.median():.{digits}f} "
            f"({values.quantile(.25):.{digits}f}–"
            f"{values.quantile(.75):.{digits}f})")


def count_positive(values: pd.Series) -> int:
    """How many patients have this history item recorded as present.

    A checkbox arrives as 1/0, as yes/no, or as a date that Excel has helpfully
    turned a 1 into. Anything numeric and above zero counts; otherwise the
    recognised affirmative words count. A date-looking column is read as
    present when it holds anything at all, since a date recorded against a
    condition means the condition was recorded.
    """
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().any():
        return int((numeric > 0).sum())
    text = values.astype("string").str.strip().str.lower()
    hits = int(text.isin(TRUE_WORDS).sum())
    if hits:
        return hits
    # Neither numeric nor a known word: count anything non-empty.
    return int((text.notna() & text.ne("") & text.ne("nan")
                & text.ne("no") & text.ne("none")).sum())


# --------------------------------------------------------------------------- #
# Resolving what the file has
# --------------------------------------------------------------------------- #

class Schema:
    """Which columns this file actually has."""

    def __init__(self) -> None:
        self.patient: str = ""
        self.time: str | None = None
        self.physiology: dict[str, str] = {}
        self.numeric_patient: dict[str, str] = {}
        self.categorical_patient: dict[str, str] = {}
        self.boluses: dict[str, str] = {}
        self.infusions: dict[str, str] = {}
        self.comorbidities: list[str] = []
        self.absent: list[str] = []
        self.time_candidates: list[tuple] = []
        # Minutes per unit of the time column, and what that unit is called.
        self.time_scale: float = 1.0
        self.time_unit: str = "minutes"


def resolve_schema(frame: pd.DataFrame, args: argparse.Namespace) -> Schema:
    schema = Schema()

    patient = find_column(frame, args.patient_col, PATIENT_ALIASES)
    if patient is None:
        sys.stderr.write(
            "\nCould not find a patient identifier column, and without one "
            "there is nothing to count.\n"
            f"Looked for: {args.patient_col}, "
            f"{', '.join(PATIENT_ALIASES)}\n"
            "Pass the right one with --patient-col. Columns in the file:\n  "
            + "\n  ".join(str(c) for c in frame.columns[:80]) + "\n")
        raise SystemExit(2)
    schema.patient = patient

    if args.time_col:
        # The user named one. Use it, or say plainly that it is not there.
        found = find_column(frame, args.time_col)
        if found is None:
            sys.stderr.write(
                f"\n--time-col '{args.time_col}' is not a column in this "
                f"file.\n\nColumns whose values could be minutes from "
                f"induction:\n"
                + ("\n".join(numeric_column_report(frame, patient))
                   or "  (none found)") + "\n")
            raise SystemExit(2)
        schema.time = found
        unit = time_unit_in_minutes(normalize(found))
        if args.time_units:
            named = {"minutes": (1.0, "minutes"), "seconds": (1.0 / 60.0, "seconds"),
                     "milliseconds": (1.0 / 60000.0, "milliseconds"),
                     "hours": (60.0, "hours")}
            unit = named[args.time_units]
        if unit is not None:
            schema.time_scale, schema.time_unit = unit
    else:
        ranked = rank_time_columns(frame)
        schema.time_candidates = ranked
        if ranked:
            _, schema.time, schema.time_scale, schema.time_unit, _ = ranked[0]

    for key, aliases in PHYSIOLOGY_ALIASES.items():
        found = find_column(frame, key, aliases)
        if found is not None and found not in schema.physiology.values():
            schema.physiology[key] = found
        elif found is None:
            schema.absent.append(f"physiology {key}")
    for key in PATIENT_NUMERIC:
        found = find_column(frame, key)
        if found is not None:
            schema.numeric_patient[key] = found
        else:
            schema.absent.append(f"patient covariate {key}")
    for key in PATIENT_CATEGORICAL:
        found = find_column(frame, key)
        if found is not None:
            schema.categorical_patient[key] = found
        else:
            schema.absent.append(f"patient covariate {key}")
    for key in BOLUS_COLUMNS:
        found = find_column(frame, key)
        if found is not None:
            schema.boluses[key] = found
    for key in INFUSION_COLUMNS:
        found = find_column(frame, key)
        if found is not None:
            schema.infusions[key] = found

    taken = set(schema.physiology.values()) | set(schema.boluses.values()) \
        | set(schema.infusions.values()) | set(schema.numeric_patient.values()) \
        | set(schema.categorical_patient.values())
    schema.comorbidities = [
        str(c) for c in frame.columns
        if normalize(c).startswith(tuple(normalize(p) for p in HISTORY_PREFIXES))
        and c not in taken]
    return schema


def describe_schema(frame: pd.DataFrame, schema: Schema, path: Path) -> None:
    print(f"\nFile: {path}")
    print(f"  {len(frame):,} rows x {frame.shape[1]} columns")
    print(f"  patient column : {schema.patient} "
          f"({frame[schema.patient].nunique()} patients)")
    if schema.time is None:
        print("  time column    : NOT FOUND — the whole file is treated as "
              "one window and there is no baseline section")
        print("\n  Columns whose values could be minutes from induction:")
        for line in (numeric_column_report(frame, schema.patient)
                     or ["  (none found)"]):
            print(f"  {line}")
        print("  Pass the right one with --time-col.")
    else:
        times = pd.to_numeric(frame[schema.time], errors="coerce")
        print(f"  time column    : {schema.time} "
              f"({times.min():.1f} to {times.max():.1f})")
        if schema.time_scale != 1.0:
            print(f"                   stored in {schema.time_unit}, "
                  f"converted to minutes")
        for score, column, _scale, _unit, note in schema.time_candidates[1:5]:
            print(f"                   also considered: {column} "
                  f"({note}, score {score:.0f})")

    def show(title: str, found: dict[str, str]) -> None:
        print(f"\n  {title}:")
        if not found:
            print("    none found")
        for key, column in found.items():
            present = float(frame[column].notna().mean())
            print(f"    {key:<32} -> {column:<34} {present:.0%} non-missing")

    show("Physiology", schema.physiology)
    show("Patient covariates (numeric)", schema.numeric_patient)
    show("Patient covariates (categorical)", schema.categorical_patient)
    show("Cumulative bolus columns", schema.boluses)
    show("Infusion columns", schema.infusions)
    print(f"\n  History / comorbidity columns ({len(schema.comorbidities)}): "
          f"{', '.join(schema.comorbidities) or 'none'}")
    if schema.absent:
        print(f"\n  NOT FOUND ({len(schema.absent)}) — their rows are left out:")
        for line in schema.absent:
            print(f"    {line}")


# --------------------------------------------------------------------------- #
# Table 1
# --------------------------------------------------------------------------- #

def build_table1(frame: pd.DataFrame, window: pd.DataFrame, schema: Schema,
                 args: argparse.Namespace) -> tuple[list[tuple[str, str]], str]:
    """Every row of Table 1, as (label, value) pairs."""
    patients = window[schema.patient].nunique()
    rows: list[tuple[str, str]] = []

    def section(title: str) -> None:
        rows.append((f"__SECTION__{title}", ""))

    # One row per patient, for anything that does not change over the case.
    per_patient = window.groupby(schema.patient, sort=True).first()

    # ---- who the patients are ---------------------------------------------
    section("Patients")
    rows.append(("Patients analysed", f"{patients}"))
    rows.append(("Epochs analysed", f"{len(window):,}"))
    rows.append(("Epochs per patient",
                 describe_numeric(window.groupby(schema.patient).size(), 0)))

    for key, spec in PATIENT_NUMERIC.items():
        if key in schema.numeric_patient:
            rows.append((spec["label"], describe_numeric(
                per_patient[schema.numeric_patient[key]], spec["digits"])))

    for key, label in PATIENT_CATEGORICAL.items():
        if key not in schema.categorical_patient:
            continue
        values = per_patient[schema.categorical_patient[key]]
        text = values.astype("string").str.strip()
        text = text[text.notna() & text.ne("") & text.str.lower().ne("nan")]
        if text.empty:
            rows.append((label, "not recorded"))
            continue
        counts = text.value_counts()
        numeric_levels = pd.to_numeric(pd.Series(counts.index), errors="coerce")
        if numeric_levels.notna().all():
            counts = counts.iloc[np.argsort(numeric_levels.to_numpy())]
        rows.append((label, f"{len(text)} of {patients} recorded"))
        for level, count in counts.items():
            shown = str(level)
            if shown.endswith(".0"):
                shown = shown[:-2]
            rows.append((f"    {shown}",
                         f"{count} ({100 * count / patients:.0f}%)"))

    # ---- comorbidities ------------------------------------------------------
    if schema.comorbidities:
        section("Comorbidities")
        shown = 0
        for column in schema.comorbidities:
            present = count_positive(per_patient[column])
            if present == 0 and not args.show_empty_comorbidities:
                continue
            rows.append((pretty_label(column),
                         f"{present} ({100 * present / patients:.0f}%)"))
            shown += 1
        if shown == 0:
            rows.append(("None recorded as present", ""))

    # ---- baseline physiology ------------------------------------------------
    baseline_note = "not available: the file has no usable time column"
    if schema.time is not None:
        before = frame.loc[(frame[schema.time] >= args.baseline_start)
                           & (frame[schema.time] < args.window_start)]
        if not before.empty and before[schema.patient].nunique() >= patients * 0.5:
            source = before
            baseline_note = (f"mean of each patient's epochs from "
                             f"{args.baseline_start:g} to {args.window_start:g} "
                             f"min, before induction")
        else:
            source = (window.sort_values(schema.time)
                      .groupby(schema.patient).head(3))
            baseline_note = ("mean of each patient's first three epochs after "
                             "induction; the file carries no pre-induction "
                             "window, so this is NOT a true baseline")
        section(f"Baseline physiology ({baseline_note})")
        if not schema.physiology:
            rows.append(("No physiology columns found", ""))
        for key, column in schema.physiology.items():
            spec = PHYSIOLOGY[key]
            rows.append((f"{spec['label']}, {spec['unit']}", describe_numeric(
                source.groupby(schema.patient)[column].mean(), spec["digits"])))

    # ---- medication exposure -------------------------------------------------
    if schema.boluses:
        section("Cumulative bolus dose by end of window, among exposed")
        for key, column in schema.boluses.items():
            # The column is cumulative, so a patient's maximum is their total.
            # Summing it would add the same dose again at every epoch.
            totals = window.groupby(schema.patient)[column].max()
            exposed = totals[pd.to_numeric(totals, errors="coerce").fillna(0) > 0]
            if exposed.empty:
                rows.append((BOLUS_COLUMNS[key], "0 patients exposed"))
                continue
            rows.append((BOLUS_COLUMNS[key],
                         f"{len(exposed)} "
                         f"({100 * len(exposed) / patients:.0f}%) · "
                         f"{describe_numeric(exposed, 1)}"))

    if schema.infusions:
        section("Infusion rate while running, among exposed")
        for key, column in schema.infusions.items():
            rate = pd.to_numeric(window[column], errors="coerce")
            running = window.loc[rate > 0]
            if running.empty:
                rows.append((INFUSION_COLUMNS[key], "0 patients exposed"))
                continue
            exposed = running[schema.patient].nunique()
            rows.append((INFUSION_COLUMNS[key],
                         f"{exposed} ({100 * exposed / patients:.0f}%) · "
                         f"{describe_numeric(running[column], 2)} · "
                         f"running {100 * len(running) / len(window):.0f}% "
                         f"of epochs"))

    # ---- completeness --------------------------------------------------------
    if schema.physiology:
        section("Signal completeness over the analysis window")
        for key, column in schema.physiology.items():
            present = float(window[column].notna().mean())
            with_any = int(
                window.loc[window[column].notna(), schema.patient].nunique())
            rows.append((PHYSIOLOGY[key]["label"],
                         f"{present:.0%} of epochs · {with_any} patients with "
                         f"any value"))
        complete = window.dropna(subset=list(schema.physiology.values()))
        rows.append(("Epochs complete on every signal above",
                     f"{len(complete):,} "
                     f"({100 * len(complete) / max(len(window), 1):.0f}%)"))
        rows.append(("Patients contributing a complete epoch",
                     f"{complete[schema.patient].nunique()} of {patients}"))

    return rows, baseline_note


# --------------------------------------------------------------------------- #
# Writing it out
# --------------------------------------------------------------------------- #

def render_table(rows: list[list[str]], widths: list[int], title: str,
                 footnote: str, path: Path,
                 section_rows: frozenset[int] = frozenset()) -> None:
    """Draw the table as an image, in a monospace grid that survives any cell."""
    body_size, note_size = 9.2, 8.2
    line = body_size * 1.55 / 72.0
    note_lines = footnote.count("\n") + 1 if footnote else 0
    height = (0.55 + line * (len(rows) + 1)
              + note_size * 1.5 / 72.0 * note_lines + 0.45)
    character = body_size * 0.62 / 72.0
    width = max(character * (sum(widths) + 2 * (len(widths) - 1)),
                max((len(part) for part in footnote.split("\n")), default=0)
                * note_size * 0.52 / 72.0, 6.0) + 0.6

    figure = plt.figure(figsize=(width, max(height, 2.5)))
    axis = figure.add_axes([0.0, 0.0, 1.0, 1.0])
    axis.axis("off")
    top = 1.0 - 0.18 / height
    axis.text(0.025, top, title, fontsize=12.5, fontweight="bold",
              va="top", ha="left", transform=axis.transAxes)

    def render(cells: list[str]) -> str:
        return "  ".join(str(cell).ljust(w)[:max(w, 1)]
                         for cell, w in zip(cells, widths))

    y = top - 0.42 / height
    for index, cells in enumerate(rows):
        weight, colour = ("bold", "#1A1A1A") if index == 0 else ("normal",
                                                                 "#1A1A1A")
        if index in section_rows:
            weight, colour = "bold", "#20496B"
        axis.text(0.025, y, render(cells), fontsize=body_size,
                  family="monospace", va="top", ha="left", color=colour,
                  fontweight=weight, transform=axis.transAxes)
        y -= line / height
        if index == 0:
            axis.text(0.025, y, "-" * (sum(widths) + 2 * (len(widths) - 1)),
                      fontsize=body_size, family="monospace", va="top",
                      ha="left", transform=axis.transAxes)
            y -= line / height
    if footnote:
        axis.text(0.025, 0.16 / height, footnote, fontsize=note_size,
                  va="bottom", ha="left", color="#444444",
                  transform=axis.transAxes, linespacing=1.45)
    figure.savefig(path, dpi=200)
    plt.close(figure)
    print(f"  wrote {path}")


def load_frame(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    return pd.read_csv(path, low_memory=False)


def report_missing_data(path: Path) -> None:
    """Say what is wrong and where to look, rather than an argparse usage dump."""
    sys.stderr.write(f"\nNo data file at:\n  {path}\n")
    folder = path.parent
    if folder.is_dir():
        candidates = sorted(
            p for p in folder.iterdir()
            if p.suffix.lower() in {".parquet", ".pq", ".csv", ".xlsx"})
        if candidates:
            sys.stderr.write(f"\nData files in {folder}:\n")
            for candidate in candidates[:25]:
                size = candidate.stat().st_size / 1e6
                sys.stderr.write(f"  {candidate.name}  ({size:.1f} MB)\n")
            sys.stderr.write("\nRe-run with one of them:\n"
                             f"  python {Path(__file__).name} --data "
                             f"{folder / candidates[0].name}\n")
            return
        sys.stderr.write(f"\n{folder} holds no .parquet, .csv or .xlsx file.\n")
    else:
        sys.stderr.write(f"\nThe folder {folder} does not exist either.\n")
    sys.stderr.write(f"\nPass the dataset explicitly:\n"
                     f"  python {Path(__file__).name} --data "
                     f"/path/to/analysis_file.parquet\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Table 1 — patient characteristics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA,
                        help="analysis dataset (.parquet, .csv or .xlsx)")
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR,
                        help="where the CSV and PNG are written")
    parser.add_argument("--patient-col", default="screening_id")
    parser.add_argument("--time-col", default=None,
                        help="minutes from induction; auto-detected by name "
                             "and value range when not given")
    parser.add_argument("--window-start", type=float, default=0.0,
                        help="start of the analysis window, minutes from "
                             "induction")
    parser.add_argument("--window-end", type=float, default=15.0,
                        help="end of the analysis window")
    parser.add_argument("--baseline-start", type=float, default=-5.0,
                        help="start of the pre-induction baseline window")
    parser.add_argument("--time-units", default=None,
                        choices=("minutes", "seconds", "milliseconds",
                                 "hours"),
                        help="units of --time-col; read from its name when "
                             "not given")
    parser.add_argument("--derive-minutes-from", default=None,
                        metavar="CLOCK_COLUMN",
                        help="build a minutes axis from a wall-clock column "
                             "such as timestamp_local")
    parser.add_argument("--induction-col", default=None,
                        help="timestamp of induction, used as the origin for "
                             "--derive-minutes-from; without it each "
                             "patient's first epoch is the origin")
    parser.add_argument("--no-window", action="store_true",
                        help="summarise every row, ignoring the time window")
    parser.add_argument("--show-empty-comorbidities", action="store_true",
                        help="keep history rows that no patient has")
    parser.add_argument("--inspect", action="store_true",
                        help="report which column resolved to what, and stop")
    parser.add_argument("--no-png", action="store_true",
                        help="write only the CSV")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.data.is_file():
        report_missing_data(args.data)
        return 1

    frame = load_frame(args.data)
    schema = resolve_schema(frame, args)

    if args.inspect:
        describe_schema(frame, schema, args.data)
        print("\n--inspect: nothing was written. Re-run without it when the "
              "columns above look right.")
        return 0

    args.outdir.mkdir(parents=True, exist_ok=True)
    print(f"\nTABLE 1 WILL BE WRITTEN TO:\n  {args.outdir.resolve()}")
    print(f"Reading: {args.data}")

    if args.derive_minutes_from:
        clock = find_column(frame, args.derive_minutes_from)
        if clock is None:
            sys.stderr.write(
                f"\n--derive-minutes-from '{args.derive_minutes_from}' is not "
                f"a column in this file.\n")
            return 1
        induction = None
        if args.induction_col:
            induction = find_column(frame, args.induction_col)
            if induction is None:
                sys.stderr.write(
                    f"\n--induction-col '{args.induction_col}' is not a "
                    f"column in this file.\n")
                return 1
        minutes, origin_note = derive_minutes(frame, clock, schema.patient,
                                              induction)
        frame["minutes_from_origin"] = minutes
        schema.time = "minutes_from_origin"
        print(f"\n  Derived a time axis: {origin_note}")
        print(f"  It runs {minutes.min():.1f} to {minutes.max():.1f} minutes.")
        if induction is None:
            print("  !! Every patient therefore starts at 0, so the "
                  "pre-induction baseline window is empty\n"
                  "     and Table 1 falls back to the first epochs after the "
                  "window start. Pass\n"
                  "     --induction-col if the file carries a real induction "
                  "timestamp.")

    if args.no_window and schema.time is not None:
        print(f"\n  --no-window: every row is summarised and the time column "
              f"'{schema.time}' is ignored.")
        schema.time = None

    if schema.time is None:
        if not args.no_window:
            print("\n  No column in this file looks like minutes from "
                  "induction, so the whole file\n  is treated as the analysis "
                  "window and there is no baseline section.")
            for line in numeric_column_report(frame, schema.patient):
                print(f"  {line}")
            clocks = [str(c) for c in frame.columns
                      if any(w in normalize(c) for w in
                             ("timestamp", "datetime", "clock", "utc"))]
            print("\n  If one of those is the right column, name it with "
                  "--time-col.")
            if clocks:
                print(f"  To build a minutes axis from the wall clock "
                      f"instead:\n"
                      f"    python {Path(__file__).name} "
                      f"--derive-minutes-from {clocks[0]}")
            print(f"  Or, to summarise every row deliberately and silence "
                  f"this notice:\n    python {Path(__file__).name} "
                  f"--no-window")
        window = frame
        window_note = ("every row in the file; no analysis window was applied"
                       if args.no_window else
                       "the whole file; no relative time column was found")
    else:
        values = pd.to_numeric(frame[schema.time], errors="coerce")
        if schema.time_scale != 1.0:
            # The column is in seconds, or some other unit. Convert once, into
            # a new column, so the window and the baseline are both in minutes
            # and the original stays untouched.
            source = schema.time
            frame["time_min"] = values * schema.time_scale
            schema.time = "time_min"
            print(f"\n  Time axis: '{source}' is in {schema.time_unit}, "
                  f"converted to minutes\n"
                  f"  ({frame['time_min'].min():.1f} to "
                  f"{frame['time_min'].max():.1f} min).")
        else:
            frame[schema.time] = values
        window = frame.loc[frame[schema.time].between(args.window_start,
                                                      args.window_end)].copy()
        window_note = (f"{args.window_start:g} to {args.window_end:g} minutes "
                       f"from induction")
        if window.empty:
            sys.stderr.write(
                f"\nNo epochs fall between {args.window_start:g} and "
                f"{args.window_end:g} minutes on column '{schema.time}', "
                f"whose range is {frame[schema.time].min():,.1f} to "
                f"{frame[schema.time].max():,.1f}.\n"
                "That range is not minutes around an induction, so this is "
                "the wrong column.\n\n"
                "Columns whose values could be minutes from induction:\n"
                + ("\n".join(numeric_column_report(frame, schema.patient))
                   or "  (none in this file)")
                + f"\n\nRe-run naming the right one:\n"
                  f"  python {Path(__file__).name} --time-col <column>\n"
                  f"\nOr, if the file truly has no relative time axis, "
                  f"summarise all of it:\n"
                  f"  python {Path(__file__).name} --no-window\n")
            return 1

    print(f"\n  Window: {window_note}")
    print(f"  {window[schema.patient].nunique()} patients, "
          f"{len(window):,} epochs")
    if schema.absent:
        print(f"  Columns not found, so their rows are left out: "
              f"{', '.join(schema.absent)}")

    table, baseline_note = build_table1(frame, window, schema, args)

    print()
    for key, value in table:
        if key.startswith("__SECTION__"):
            print(f"\n  {key.replace('__SECTION__', '').upper()}")
        else:
            print(f"    {key:<54} {value}")

    patients = window[schema.patient].nunique()
    flat = [{"Characteristic": k.replace("__SECTION__", ""), "Value": v}
            for k, v in table]
    csv_path = args.outdir / "table1_patient_characteristics.csv"
    pd.DataFrame(flat).to_csv(csv_path, index=False)
    print(f"\n  wrote {csv_path}")

    if not args.no_png:
        grid = [["Characteristic", f"Value (n = {patients})"]]
        sections: set[int] = set()
        for key, value in table:
            if key.startswith("__SECTION__"):
                sections.add(len(grid))
                grid.append([key.replace("__SECTION__", "").upper(), ""])
            else:
                grid.append([key, value])
        widths = [max(len(r[0]) for r in grid), max(len(r[1]) for r in grid)]
        render_table(
            grid, widths,
            "Table 1. Patient characteristics, baseline physiology, "
            "medication exposure and signal completeness",
            "Numeric values are median (IQR), computed once per patient.\n"
            "Medication rows are patients exposed (%) followed by the dose "
            "distribution among those exposed only.\n"
            f"Analysis window: {window_note}. Baseline is the {baseline_note}.",
            args.outdir / "table1_patient_characteristics.png",
            section_rows=frozenset(sections))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
