#!/usr/bin/env python3
"""Table 1 and Table 2 for the cerebral-oxygenation induction analysis.

WHAT THIS PRODUCES
------------------
  table1_patient_characteristics   age, sex, BMI, ASA, comorbidities, baseline
                                   physiology, medication exposure counts and
                                   doses, and signal completeness
  table2_primary                   the within-patient MAP / CI / PR / PSi
                                   coefficients from the primary model, scaled
                                   to clinical denominators, with 95% CIs, raw
                                   P, Benjamini-Hochberg q, and the adjustment
                                   set
  table2_across_models             the same four exposures across Models A-D,
                                   so the stability of each coefficient is
                                   visible in one place
  table_model_comparison           AIC, BIC and likelihood-ratio tests

  Each is written as a CSV (for the manuscript) and a PNG (for slides).

TWO CORRECTIONS TO THE CURRENT ANALYSIS
---------------------------------------
  1. WITHIN-PATIENT ESTIMATES. The analysis plan is explicit: "For each
     time-varying physiologic exposure X, calculate the patient's mean over
     the analysis window and the deviation of each epoch from that mean. Enter
     these two terms separately ... Do not interpret a pooled slope as a purely
     within-patient association."

     A model written as `sto2 ~ map + ci + pr + psi` does not do this. Its
     coefficient blends two different questions: does THIS patient's brain
     oxygenation move when THIS patient's pressure moves, and do patients who
     run at a higher pressure sit at a higher saturation? Those can even have
     opposite signs. Every exposure here is therefore split into a
     within-patient deviation and a patient mean, and Table 2 reports the
     deviation coefficient, which is the one the aim asks for.

  2. ML, NOT REML, FOR MODEL COMPARISON. AIC, BIC and likelihood-ratio tests
     are only comparable across models that differ in their FIXED effects when
     both are fitted by maximum likelihood. REML likelihoods for different
     fixed-effect structures are not on the same scale, and comparing them
     silently gives wrong answers. The comparison table is fitted with
     reml=False and says so; the reported coefficients use REML, which is the
     better choice for the variance components.

EXPECTED INPUT
--------------
  One analysis-ready long file (CSV or Parquet): one row per patient-epoch
  over the analysis window, patient-level variables repeated on every row of
  that patient. The column names default to the ones already in use (sto2_mean,
  map, ci, pr, psi, age, bmi, sex, asa, *_cumulative_bolus_*, *_infusion_*).

  Run with --inspect first. It prints which columns were found, which were not,
  and what it would fit, and then stops without fitting anything.

USAGE
-----
    python cerebral_so2_tables.py --data analysis_ready.csv --inspect
    python cerebral_so2_tables.py --data analysis_ready.csv
    python cerebral_so2_tables.py --data analysis_ready.csv --outcome sto2_left
    python cerebral_so2_tables.py --data analysis_ready.csv --bootstrap 1000
"""

from __future__ import annotations

import argparse
import sys
import textwrap
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

try:
    import statsmodels.formula.api as smf
    HAVE_STATSMODELS = True
except Exception:                                    # pragma: no cover
    smf = None
    HAVE_STATSMODELS = False


# --------------------------------------------------------------------------- #
# What the analysis is made of
# --------------------------------------------------------------------------- #

# The four primary exposures, with the denominator each coefficient is scaled
# to. These denominators are what make the numbers comparable with each other
# and with the published literature; a raw per-unit coefficient for PSi is not
# comparable with a raw per-unit coefficient for cardiac index.
PRIMARY_EXPOSURES: dict[str, dict] = {
    "map": {"label": "MAP", "unit": "mmHg", "scale": 10.0,
            "denominator": "per 10 mmHg"},
    "ci":  {"label": "Cardiac index", "unit": "L/min/m²", "scale": 1.0,
            "denominator": "per 1 L/min/m²"},
    "pr":  {"label": "Pulse rate", "unit": "bpm", "scale": 10.0,
            "denominator": "per 10 bpm"},
    "psi": {"label": "PSi", "unit": "units", "scale": 10.0,
            "denominator": "per 10 PSi units"},
}

# Prefixes a dataset uses to mark a recorded history item, stripped for display.
HISTORY_PREFIXES = ("pmh_", "hx_", "comorbid_", "history_")
# Words that should keep their conventional capitalisation in a label.
KNOWN_TERMS = {
    "cad": "CAD", "chf": "CHF", "copd": "COPD", "ckd": "CKD", "cva": "CVA",
    "tia": "TIA", "dm": "DM", "htn": "HTN", "afib": "AF", "mi": "MI",
    "osa": "OSA", "pvd": "PVD", "asa": "ASA", "bmi": "BMI", "psi": "PSi",
    "map": "MAP", "esrd": "ESRD", "pad": "PAD", "gerd": "GERD",
}


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
    shown[0] = shown[0] if shown[0] in KNOWN_TERMS.values() else shown[0].capitalize()
    return " ".join(shown)

PATIENT_NUMERIC = {"age": "Age, years", "bmi": "BMI, kg/m²"}
PATIENT_CATEGORICAL = {"sex": "Sex", "asa": "ASA class"}

# Cumulative bolus columns, with the unit carried in the name as the current
# dataset spells it. Any that the file does not have are skipped.
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

# Model sequence, matching the slide deck. Each adds to the one before it, so
# a coefficient that moves between them moves because of what was added.
MODEL_BLOCKS: dict[str, dict] = {
    "A": {"label": "Physiology + time",
          "blocks": ("time", "physiology")},
    "B": {"label": "+ patient covariates",
          "blocks": ("time", "physiology", "patient")},
    "C": {"label": "+ medications",
          "blocks": ("time", "physiology", "patient", "medication")},
    "D": {"label": "+ history and surgery",
          "blocks": ("time", "physiology", "patient", "medication", "context")},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    base = Path(__file__).resolve().parent

    group = parser.add_argument_group("input")
    group.add_argument("--data", type=Path, required=True,
                       help="Analysis-ready long file, one row per "
                            "patient-epoch (.csv or .parquet).")
    group.add_argument("--patient-col", default="screening_id",
                       help="Patient identifier column (default screening_id). "
                            "Tried against a few common spellings if absent.")
    group.add_argument("--time-col", default="time_min",
                       help="Minutes from induction (default time_min). A "
                            "centered column is derived from it.")
    group.add_argument("--outcome", default="sto2_mean",
                       help="Outcome column (default sto2_mean). Use the "
                            "left or right column to run a side separately.")
    group.add_argument("--outdir", type=Path, default=base / "output",
                       help="Where the tables are written (default: an "
                            "'output' folder beside this script).")

    group = parser.add_argument_group("window and model")
    group.add_argument("--window-start", type=float, default=0.0,
                       help="First minute of the analysis window (default 0).")
    group.add_argument("--window-end", type=float, default=15.0,
                       help="Last minute of the analysis window (default 15).")
    group.add_argument("--baseline-start", type=float, default=-5.0,
                       help="Start of the pre-induction baseline window "
                            "(default -5 min). Used only if the file carries "
                            "epochs before induction.")
    group.add_argument("--time-basis", choices=("quadratic", "spline"),
                       default="quadratic",
                       help="How time enters the model. 'quadratic' matches "
                            "the current analysis; 'spline' is the 3-df "
                            "natural cubic spline the plan prefers.")
    group.add_argument("--random-slope", action="store_true",
                       help="Add a patient random slope on time. Off by "
                            "default: it often fails to converge at this "
                            "cohort size, and a failure is easy to miss.")
    group.add_argument("--primary-model", default="C", choices=tuple("ABCD"),
                       help="Which model Table 2's primary panel reports "
                            "(default C, the best fit-complexity balance).")
    group.add_argument("--context-columns", default="",
                       help="Comma-separated history and surgery columns for "
                            "Model D. Empty means Model D is skipped.")

    group = parser.add_argument_group("inference")
    group.add_argument("--bootstrap", type=int, default=0,
                       help="Patient-level bootstrap resamples for the primary "
                            "coefficients (the plan asks for >= 1000). 0 skips "
                            "it. Each resample refits the model, so 1000 takes "
                            "a while; the cluster-robust column is always "
                            "computed and needs no resampling.")
    group.add_argument("--seed", type=int, default=20261104)

    group = parser.add_argument_group("other")
    group.add_argument("--inspect", action="store_true",
                       help="Report which columns were found and what would be "
                            "fitted, then stop. Run this first.")
    group.add_argument("--no-png", action="store_true",
                       help="Write only the CSVs.")
    return parser.parse_args()


# --------------------------------------------------------------------------- #
# Finding the columns
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


PATIENT_ALIASES = ("screening_id", "patient_id", "subject_id", "patient",
                   "id", "record_id", "study_id")
TIME_ALIASES = ("time_min", "minutes", "time_minutes", "min_from_induction",
                "time_from_induction", "t_min", "time")


@dataclass
class Schema:
    """Which columns this file actually has, resolved once and reused."""
    patient: str
    time: str
    outcome: str
    exposures: dict[str, str] = field(default_factory=dict)
    numeric_patient: dict[str, str] = field(default_factory=dict)
    categorical_patient: dict[str, str] = field(default_factory=dict)
    boluses: dict[str, str] = field(default_factory=dict)
    infusions: dict[str, str] = field(default_factory=dict)
    context: list[str] = field(default_factory=list)
    comorbidities: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)


def resolve_schema(frame: pd.DataFrame, args: argparse.Namespace) -> Schema:
    missing: list[str] = []

    patient = find_column(frame, args.patient_col, PATIENT_ALIASES)
    time = find_column(frame, args.time_col, TIME_ALIASES)
    outcome = find_column(frame, args.outcome)
    for name, found in (("patient id", patient), ("time", time),
                        ("outcome", outcome)):
        if found is None:
            raise SystemExit(
                f"Could not find the {name} column. Columns in the file:\n  "
                + "\n  ".join(str(c) for c in frame.columns[:60]))

    schema = Schema(patient=patient, time=time, outcome=outcome)

    for key in PRIMARY_EXPOSURES:
        found = find_column(frame, key)
        if found is None:
            missing.append(f"exposure {key}")
        else:
            schema.exposures[key] = found
    for key, _label in PATIENT_NUMERIC.items():
        found = find_column(frame, key)
        if found is None:
            missing.append(f"patient covariate {key}")
        else:
            schema.numeric_patient[key] = found
    for key, _label in PATIENT_CATEGORICAL.items():
        found = find_column(frame, key)
        if found is None:
            missing.append(f"patient covariate {key}")
        else:
            schema.categorical_patient[key] = found
    for key, _label in BOLUS_COLUMNS.items():
        found = find_column(frame, key)
        if found is not None:
            schema.boluses[key] = found
    for key, _label in INFUSION_COLUMNS.items():
        found = find_column(frame, key)
        if found is not None:
            schema.infusions[key] = found

    schema.context = [c.strip() for c in str(args.context_columns).split(",")
                      if c.strip() and c.strip() in frame.columns]
    # Anything that looks like a recorded comorbidity, for Table 1 only.
    schema.comorbidities = [
        str(c) for c in frame.columns
        if normalize(c).startswith(("pmh", "hx", "comorbid", "history"))]
    schema.missing = missing
    return schema


def describe_schema(frame: pd.DataFrame, schema: Schema,
                    args: argparse.Namespace) -> None:
    print(f"\nFile: {args.data}")
    print(f"  {len(frame):,} rows x {frame.shape[1]} columns")
    print(f"  patient column : {schema.patient} "
          f"({frame[schema.patient].nunique()} patients)")
    print(f"  time column    : {schema.time} "
          f"({frame[schema.time].min():.1f} to {frame[schema.time].max():.1f})")
    print(f"  outcome        : {schema.outcome} "
          f"({int(frame[schema.outcome].notna().sum()):,} non-missing)")

    def show(title: str, found: dict[str, str]) -> None:
        print(f"\n  {title}:")
        if not found:
            print("    none found")
        for key, column in found.items():
            present = float(frame[column].notna().mean())
            print(f"    {key:<34} -> {column:<34} {present:.0%} non-missing")

    show("Primary exposures", schema.exposures)
    show("Patient covariates (numeric)", schema.numeric_patient)
    show("Patient covariates (categorical)", schema.categorical_patient)
    show("Cumulative bolus columns", schema.boluses)
    show("Infusion columns", schema.infusions)
    print(f"\n  Comorbidity-looking columns: "
          f"{', '.join(schema.comorbidities) or 'none'}")
    print(f"  Model D context columns    : "
          f"{', '.join(schema.context) or 'none (Model D will be skipped)'}")
    if schema.missing:
        print(f"\n  NOT FOUND ({len(schema.missing)}):")
        for line in schema.missing:
            print(f"    {line}")
        print("  Pass the right name, or accept that the term is dropped.")


# --------------------------------------------------------------------------- #
# Building the design
# --------------------------------------------------------------------------- #

def add_decomposition(frame: pd.DataFrame, schema: Schema) -> pd.DataFrame:
    """Split every time-varying exposure into a within part and a patient mean.

    This is the step that makes the coefficient answer the aim's question. The
    within part is the epoch's deviation from that patient's own mean over the
    window; the between part is the patient's mean, centered on the cohort so
    the intercept stays interpretable. With both in the model the within
    coefficient is a pure within-patient estimate, and the between coefficient
    is reported separately rather than averaged into it.
    """
    out = frame.copy()
    grouped = out.groupby(schema.patient, sort=False)
    for key, column in schema.exposures.items():
        patient_mean = grouped[column].transform("mean")
        out[f"{key}_within"] = out[column] - patient_mean
        out[f"{key}_between"] = patient_mean - patient_mean.mean()
    return out


def time_terms(frame: pd.DataFrame, schema: Schema,
               args: argparse.Namespace) -> tuple[pd.DataFrame, list[str]]:
    """Time from induction, centered, as a quadratic or a 3-df spline."""
    out = frame.copy()
    midpoint = (args.window_start + args.window_end) / 2.0
    out["time_c"] = out[schema.time] - midpoint
    if args.time_basis == "quadratic":
        out["time_c_sq"] = out["time_c"] ** 2
        return out, ["time_c", "time_c_sq"]

    # Natural cubic spline on 3 degrees of freedom, built by hand so the basis
    # is reproducible and the knots are reported rather than chosen silently.
    inner = np.quantile(out["time_c"].dropna(), [0.5])
    boundary = np.quantile(out["time_c"].dropna(), [0.05, 0.95])
    knots = np.sort(np.concatenate([boundary[:1], inner, boundary[1:]]))

    def truncated(x: np.ndarray, knot: float) -> np.ndarray:
        return np.clip(x - knot, 0.0, None) ** 3

    x = out["time_c"].to_numpy(float)
    first, middle, last = knots
    denominator = last - first
    basis_1 = (truncated(x, first) - truncated(x, last)) / denominator
    basis_2 = (truncated(x, middle) - truncated(x, last)) / (last - middle)
    out["time_s1"] = x
    out["time_s2"] = basis_1 - basis_2
    return out, ["time_s1", "time_s2"]


def model_terms(schema: Schema, time_columns: list[str],
                blocks: tuple[str, ...]) -> list[str]:
    terms: list[str] = []
    if "time" in blocks:
        terms += time_columns
    if "physiology" in blocks:
        for key in schema.exposures:
            terms += [f"{key}_within", f"{key}_between"]
    if "patient" in blocks:
        terms += list(schema.numeric_patient)
        terms += [f"C({key})" for key in schema.categorical_patient]
    if "medication" in blocks:
        terms += list(schema.boluses) + list(schema.infusions)
    if "context" in blocks:
        terms += [f"Q('{column}')" for column in schema.context]
    return terms


def term_column(term: str) -> str:
    """The underlying column a formula term refers to."""
    if term.startswith("C(") and term.endswith(")"):
        return term[2:-1]
    if term.startswith("Q('") and term.endswith("')"):
        return term[3:-2]
    return term


def screen_terms(frame: pd.DataFrame, plan: dict[str, list[str]],
                 schema: Schema) -> tuple[dict[str, list[str]], dict[str, str]]:
    """Drop terms this sample cannot estimate, and say which and why.

    A drug nobody received inside the window is an all-zero column; a bolus
    given once before induction is constant within every patient; two
    vasopressor columns can be proportional to each other. Each of those makes
    the design matrix singular, and statsmodels reports only "Singular matrix"
    without naming the culprit. Finding it by hand costs an afternoon, so the
    screen runs first and prints the reason for every term it removes.

    Nothing here is a judgement about importance. A term is removed only when
    the data carry no information with which to estimate it, which is why the
    removals are printed rather than applied quietly.
    """
    dropped: dict[str, str] = {}

    def keep(term: str) -> bool:
        column = term_column(term)
        if column not in frame.columns:
            dropped[term] = "column is not in the analysis sample"
            return False
        series = frame[column]
        if series.notna().sum() == 0:
            dropped[term] = "no non-missing values inside the window"
            return False
        if not pd.api.types.is_numeric_dtype(series) or term != column:
            levels = series.dropna().unique()
            if len(levels) < 2:
                dropped[term] = f"only one level in this sample ({levels[0]})"
                return False
            return True
        values = pd.to_numeric(series, errors="coerce").to_numpy(float)
        if float(np.nanstd(values)) <= 1e-12:
            dropped[term] = (f"constant at {np.nanmin(values):g} over the "
                             f"analysis sample")
            return False
        return True

    screened = {letter: [t for t in terms if keep(t)]
                for letter, terms in plan.items()}

    # Perfect collinearity among the numeric terms, checked by whether each
    # column raises the rank of what is already in. Terms are offered in plan
    # order, so time and the physiologic exposures are kept in preference to
    # anything added later.
    ordered = [t for t in dict.fromkeys(
        term for terms in screened.values() for term in terms)
        if t == term_column(t) and pd.api.types.is_numeric_dtype(frame[t])]
    design = np.ones((len(frame), 1))
    rank = 1
    for term in ordered:
        values = pd.to_numeric(frame[term], errors="coerce").to_numpy(float)
        values = np.nan_to_num(values, nan=float(np.nanmean(values)))
        trial = np.hstack([design, values.reshape(-1, 1)])
        trial_rank = int(np.linalg.matrix_rank(trial, tol=1e-9))
        if trial_rank > rank:
            design, rank = trial, trial_rank
        else:
            dropped[term] = ("perfectly collinear with terms already in the "
                             "model")

    if dropped:
        screened = {letter: [t for t in terms if t not in dropped]
                    for letter, terms in screened.items()}
        print(f"\n  Terms the sample cannot estimate, so they are dropped "
              f"({len(dropped)}):")
        for term, reason in dropped.items():
            print(f"    {term:<34} {reason}")
        primary = {f"{key}_within" for key in schema.exposures} | {
            f"{key}_between" for key in schema.exposures}
        hit = sorted(primary & set(dropped))
        if hit:
            print(f"\n  !! A PRIMARY EXPOSURE TERM WAS DROPPED: "
                  f"{', '.join(hit)}")
            print("     Table 2 cannot report that exposure. Check the data "
                  "before reading anything else below.")
    return screened, dropped


def rename_for_formula(frame: pd.DataFrame, schema: Schema) -> pd.DataFrame:
    """Give every modelled column a patsy-safe name matching the term list."""
    mapping: dict[str, str] = {}
    for key, column in schema.exposures.items():
        mapping[column] = key
    for key, column in {**schema.numeric_patient,
                        **schema.categorical_patient}.items():
        mapping[column] = key
    for key, column in {**schema.boluses, **schema.infusions}.items():
        mapping[column] = key
    mapping[schema.outcome] = "outcome"
    return frame.rename(columns={k: v for k, v in mapping.items() if k != v})


# --------------------------------------------------------------------------- #
# Fitting
# --------------------------------------------------------------------------- #

@dataclass
class Fit:
    label: str
    letter: str
    terms: list[str]
    params: pd.Series
    bse: pd.Series
    pvalues: pd.Series
    conf: pd.DataFrame
    n_obs: int
    n_patients: int
    aic: float
    bic: float
    llf: float
    k_fe: int
    converged: bool
    random_structure: str = "intercept"
    note: str = ""


def fit_mixed(frame: pd.DataFrame, terms: list[str], patient: str,
              label: str, letter: str, reml: bool,
              random_slope_on: str | None = None,
              quiet: bool = False) -> Fit | None:
    """One linear mixed model with a patient random intercept.

    A failure returns None rather than falling back to a simpler random-effects
    structure. Falling back per model is tempting but wrong: AIC, BIC and the
    likelihood-ratio test are only comparable across models whose RANDOM part
    is identical, so a model that quietly lost its random slope would be
    compared against models that kept theirs. The caller decides on one
    structure for the whole sequence instead.
    """
    formula = "outcome ~ " + " + ".join(terms) if terms else "outcome ~ 1"
    usable = frame.dropna(subset=["outcome"]).copy()
    kwargs = {}
    if random_slope_on:
        kwargs["re_formula"] = f"~{random_slope_on}"
    structure = (f"intercept + slope on {random_slope_on}" if random_slope_on
                 else "intercept")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            model = smf.mixedlm(formula, usable, groups=usable[patient], **kwargs)
            result = model.fit(reml=reml, method=["lbfgs", "bfgs"])
        except Exception as exc:
            if not quiet:
                print(f"  !! Model {letter} did not fit "
                      f"({'REML' if reml else 'ML'}): {exc}")
            return None
    note = ("" if getattr(result, "converged", True)
            else "the likelihood did not converge: do not quote this fit")
    if note and not quiet:
        print(f"  !! Model {letter} did not converge "
              f"({'REML' if reml else 'ML'})")

    conf = result.conf_int()
    conf.columns = ["low", "high"]
    aic, bic = getattr(result, "aic", np.nan), getattr(result, "bic", np.nan)
    return Fit(label=label, letter=letter, terms=terms,
               params=result.fe_params, bse=result.bse[result.fe_params.index],
               pvalues=result.pvalues[result.fe_params.index],
               conf=conf.loc[result.fe_params.index],
               n_obs=int(result.nobs),
               n_patients=int(usable[patient].nunique()),
               aic=float(aic), bic=float(bic), llf=float(result.llf),
               k_fe=int(result.k_fe),
               converged=bool(getattr(result, "converged", True)),
               random_structure=structure, note=note)


def fit_all(frame: pd.DataFrame, plan: dict[str, list[str]], patient: str,
            random_slope_on: str | None) -> tuple[dict[str, Fit],
                                                  dict[str, Fit], bool]:
    """Fit the whole model sequence twice, by REML and by ML.

    Returns the two sets of fits and whether every model in the sequence fitted
    and converged. The flag is what lets the caller keep one random-effects
    structure across the sequence rather than letting individual models fall
    back, which would make the comparison table compare two things at once.
    """
    reml_fits: dict[str, Fit] = {}
    ml_fits: dict[str, Fit] = {}
    complete = True
    for letter, terms in plan.items():
        label = MODEL_BLOCKS[letter]["label"]
        quiet = random_slope_on is not None
        reml = fit_mixed(frame, terms, patient, label, letter, True,
                         random_slope_on, quiet=quiet)
        ml = fit_mixed(frame, terms, patient, label, letter, False,
                       random_slope_on, quiet=quiet)
        if reml is None or ml is None or reml.note or ml.note:
            complete = False
        if reml is not None:
            reml_fits[letter] = reml
        if ml is not None:
            ml_fits[letter] = ml
    return reml_fits, ml_fits, complete


def cluster_robust_within(frame: pd.DataFrame, schema: Schema,
                          terms: list[str]) -> dict[str, dict]:
    """Patient fixed effects with patient-clustered standard errors.

    A companion to the mixed model, not a replacement. The mixed model assumes
    residuals are independent once the random intercept is removed; at
    ten-second epochs they are nothing of the sort, and that assumption is paid
    for entirely in the standard errors. A sandwich clustered on patient is
    valid under any within-patient correlation structure, serial correlation
    included, so where the two intervals disagree the clustered one is the
    conservative and more defensible report.
    """
    within_terms = [t for t in terms if t.endswith("_within")]
    others = [t for t in terms
              if not t.endswith(("_within", "_between"))
              and t == term_column(t)
              and t in frame.columns
              and pd.api.types.is_numeric_dtype(frame[t])]
    columns = within_terms + others
    usable = frame.dropna(subset=["outcome"] + columns).copy()
    if usable.empty or usable[schema.patient].nunique() < 3:
        return {}

    codes, _ = pd.factorize(usable[schema.patient])
    counts = np.bincount(codes).astype(float)
    y = usable["outcome"].to_numpy(float)

    def demean(values: np.ndarray) -> np.ndarray:
        sums = np.zeros((len(counts),) + values.shape[1:])
        np.add.at(sums, codes, values)
        return values - (sums / counts.reshape((-1,) + (1,) * (values.ndim - 1)))[codes]

    centered = demean(usable[columns].to_numpy(float))
    target = demean(y)

    # Patient fixed effects absorb everything constant within a patient, so a
    # column that survives demeaning as zeros, or as a copy of another column,
    # carries no information here. That is not an error — the mixed model still
    # estimates such a term between patients — but the sandwich would return a
    # meaningless zero with a zero standard error, so those columns are left
    # out of this companion and simply do not appear in its column of Table 2.
    kept: list[str] = []
    indices: list[int] = []
    rank = 0
    for position, name in enumerate(columns):
        trial = centered[:, indices + [position]]
        trial_rank = int(np.linalg.matrix_rank(trial, tol=1e-9))
        if trial_rank > rank:
            rank = trial_rank
            indices.append(position)
            kept.append(name)
    if not kept:
        return {}
    columns, centered = kept, centered[:, indices]
    gram = centered.T @ centered
    beta = np.linalg.pinv(gram) @ (centered.T @ target)
    residual = target - centered @ beta

    width = len(columns)
    meat = np.zeros((width, width))
    for group in range(len(counts)):
        rows = codes == group
        score = centered[rows].T @ residual[rows]
        meat += np.outer(score, score)
    inverse = np.linalg.pinv(gram)
    groups = len(counts)
    n, k = len(y), width + groups
    scale = (groups / max(groups - 1, 1)) * ((n - 1) / max(n - k, 1))
    cov = inverse @ meat @ inverse * scale
    dof = groups - 1
    critical = float(stats.t.ppf(0.975, dof))

    out: dict[str, dict] = {}
    for index, name in enumerate(columns):
        se = float(np.sqrt(max(cov[index, index], 0.0)))
        estimate = float(beta[index])
        t = estimate / se if se else np.nan
        out[name] = {"estimate": estimate, "se": se,
                     "low": estimate - critical * se,
                     "high": estimate + critical * se,
                     "p": float(2 * stats.t.sf(abs(t), dof)) if se else np.nan}
    return out


def benjamini_hochberg(pvalues: list[float]) -> list[float]:
    """BH q-values within one family, as the plan asks for the primary four."""
    values = np.asarray(pvalues, float)
    finite = np.isfinite(values)
    out = np.full(values.shape, np.nan)
    if not finite.any():
        return out.tolist()
    subset = values[finite]
    order = np.argsort(subset)
    ranked = subset[order]
    n = len(ranked)
    adjusted = ranked * n / np.arange(1, n + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    restored = np.empty(n)
    restored[order] = np.clip(adjusted, 0, 1)
    out[finite] = restored
    return out.tolist()


def patient_bootstrap(frame: pd.DataFrame, schema: Schema, terms: list[str],
                      draws: int, seed: int) -> dict[str, dict]:
    """Resample whole patient trajectories, as the plan specifies.

    Resampling epochs would treat neighbouring ten-second rows as independent
    evidence and give intervals that are far too tight. The patient is the
    independent unit, so the patient is what gets resampled.
    """
    if draws <= 0:
        return {}
    patients = frame[schema.patient].dropna().unique()
    rng = np.random.default_rng(seed)
    wanted = [t for t in terms if t.endswith("_within")]
    collected: dict[str, list[float]] = {name: [] for name in wanted}
    blocks = {pid: part for pid, part in frame.groupby(schema.patient, sort=False)}

    for draw in range(draws):
        picked = rng.choice(patients, size=len(patients), replace=True)
        pieces = []
        for order, pid in enumerate(picked):
            piece = blocks[pid].copy()
            # A patient drawn twice must enter as two distinct clusters.
            piece[schema.patient] = f"{pid}__{order}"
            pieces.append(piece)
        resampled = pd.concat(pieces, ignore_index=True)
        fit = fit_mixed(resampled, terms, schema.patient,
                        "bootstrap", "boot", reml=False)
        if fit is None:
            continue
        for name in wanted:
            if name in fit.params.index:
                collected[name].append(float(fit.params[name]))
        if (draw + 1) % 50 == 0:
            print(f"    bootstrap {draw + 1}/{draws}", flush=True)

    out: dict[str, dict] = {}
    for name, values in collected.items():
        if len(values) >= 20:
            out[name] = {"low": float(np.percentile(values, 2.5)),
                         "high": float(np.percentile(values, 97.5)),
                         "median": float(np.median(values)),
                         "draws": len(values)}
    return out


# --------------------------------------------------------------------------- #
# Table 1
# --------------------------------------------------------------------------- #

def describe_numeric(values: pd.Series, digits: int = 1) -> str:
    values = pd.to_numeric(values, errors="coerce").dropna()
    if values.empty:
        return "not recorded"
    return (f"{values.median():.{digits}f} "
            f"({values.quantile(.25):.{digits}f}–"
            f"{values.quantile(.75):.{digits}f})")


def build_table1(frame: pd.DataFrame, window: pd.DataFrame, schema: Schema,
                 args: argparse.Namespace) -> tuple[list[tuple[str, str]], str]:
    """Patient characteristics, baseline physiology, drugs, completeness."""
    patients = window[schema.patient].nunique()
    rows: list[tuple[str, str]] = []

    def section(title: str) -> None:
        rows.append((f"__SECTION__{title}", ""))

    # ---- who the patients are ---------------------------------------------
    per_patient = window.groupby(schema.patient, sort=True).first()
    section("Patients")
    rows.append(("Patients analysed", f"{patients}"))
    rows.append(("Epochs analysed", f"{len(window):,}"))
    epochs = window.groupby(schema.patient).size()
    rows.append(("Epochs per patient", describe_numeric(epochs, 0)))

    for key, label in PATIENT_NUMERIC.items():
        if key in schema.numeric_patient:
            rows.append((label, describe_numeric(per_patient[schema.numeric_patient[key]])))
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
            rows.append((f"    {shown}", f"{count} ({100 * count / patients:.0f}%)"))

    if schema.comorbidities:
        section("Comorbidities")
        for column in schema.comorbidities[:15]:
            values = per_patient[column]
            numeric = pd.to_numeric(values, errors="coerce")
            if numeric.notna().any():
                present = int((numeric > 0).sum())
            else:
                text = values.astype("string").str.strip().str.lower()
                present = int(text.isin({"yes", "y", "true", "1", "checked"}).sum())
            rows.append((pretty_label(column),
                         f"{present} ({100 * present / patients:.0f}%)"))

    # ---- baseline physiology ----------------------------------------------
    before = frame.loc[(frame[schema.time] >= args.baseline_start)
                       & (frame[schema.time] < args.window_start)]
    if not before.empty and before[schema.patient].nunique() >= patients * 0.5:
        source = before
        baseline_note = (f"mean of valid epochs from {args.baseline_start:g} to "
                         f"0 min before induction")
    else:
        earliest = window.sort_values(schema.time).groupby(schema.patient).head(3)
        source = earliest
        baseline_note = ("first three epochs after induction; the file carries "
                         "no pre-induction window")
    section(f"Baseline physiology ({baseline_note})")
    rows.append((f"Cerebral O₂, % ({schema.outcome})",
                 describe_numeric(source.groupby(schema.patient)[schema.outcome].mean())))
    for key, column in schema.exposures.items():
        spec = PRIMARY_EXPOSURES[key]
        digits = 2 if key == "ci" else 0
        rows.append((f"{spec['label']}, {spec['unit']}", describe_numeric(
            source.groupby(schema.patient)[column].mean(), digits)))

    # ---- medication exposure ----------------------------------------------
    if schema.boluses:
        section("Cumulative bolus dose by end of window, among exposed")
        for key, column in schema.boluses.items():
            totals = window.groupby(schema.patient)[column].max()
            exposed = totals[totals.fillna(0) > 0]
            label = BOLUS_COLUMNS[key]
            if exposed.empty:
                rows.append((label, "0 patients exposed"))
                continue
            rows.append((label,
                         f"{len(exposed)} ({100 * len(exposed) / patients:.0f}%) "
                         f"· {describe_numeric(exposed, 1)}"))
    if schema.infusions:
        section("Infusion rate while running, among exposed")
        for key, column in schema.infusions.items():
            running = window.loc[pd.to_numeric(window[column], errors="coerce") > 0]
            label = INFUSION_COLUMNS[key]
            if running.empty:
                rows.append((label, "0 patients exposed"))
                continue
            exposed = running[schema.patient].nunique()
            share = 100 * len(running) / len(window)
            rows.append((label,
                         f"{exposed} ({100 * exposed / patients:.0f}%) · "
                         f"{describe_numeric(running[column], 2)} · "
                         f"running {share:.0f}% of epochs"))

    # ---- completeness ------------------------------------------------------
    section("Signal completeness over the analysis window")
    tracked = {schema.outcome: f"Cerebral O₂ ({schema.outcome})"}
    tracked.update({column: PRIMARY_EXPOSURES[key]["label"]
                    for key, column in schema.exposures.items()})
    for column, label in tracked.items():
        present = float(window[column].notna().mean())
        with_any = int(window.loc[window[column].notna(), schema.patient].nunique())
        rows.append((label, f"{present:.0%} of epochs · {with_any} patients "
                            f"with any value"))
    complete = window.dropna(subset=[schema.outcome]
                             + list(schema.exposures.values()))
    rows.append(("Epochs complete on outcome and all four exposures",
                 f"{len(complete):,} ({100 * len(complete) / len(window):.0f}%)"))
    rows.append(("Patients contributing a complete epoch",
                 f"{complete[schema.patient].nunique()} of {patients}"))
    return rows, baseline_note


# --------------------------------------------------------------------------- #
# Table 2
# --------------------------------------------------------------------------- #

def scaled(estimate: float, low: float, high: float, scale: float) -> tuple:
    return estimate * scale, low * scale, high * scale


def format_effect(estimate: float, low: float, high: float,
                  digits: int = 3) -> str:
    if not np.isfinite(estimate):
        return "not estimable"
    return (f"{estimate:+.{digits}f} ({low:+.{digits}f} to {high:+.{digits}f})")


def format_p(p: float) -> str:
    """Journal convention: three decimals, and '<0.001' below that.

    Not scientific notation. A P of 7e-65 is not 60 orders of magnitude more
    convincing than a P of 0.0009 — at ten-second epochs both mostly reflect
    how many rows the file happens to have, and the exponent invites a reader
    to treat sample size as strength of evidence. The effect size and its
    interval carry the finding; the P value only needs to clear a threshold.
    """
    if not np.isfinite(p):
        return "—"
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"


def build_table2_primary(fit: Fit, schema: Schema, robust: dict,
                         boot: dict) -> list[dict]:
    """The four primary exposures from one model, scaled and FDR-adjusted."""
    rows: list[dict] = []
    raw_p: list[float] = []
    for key in schema.exposures:
        name = f"{key}_within"
        if name not in fit.params.index:
            continue
        spec = PRIMARY_EXPOSURES[key]
        estimate, low, high = scaled(
            float(fit.params[name]), float(fit.conf.loc[name, "low"]),
            float(fit.conf.loc[name, "high"]), spec["scale"])
        p = float(fit.pvalues[name])
        raw_p.append(p)
        row = {"Exposure": spec["label"], "Denominator": spec["denominator"],
               "beta (95% CI)": format_effect(estimate, low, high),
               "P": format_p(p), "_p": p}
        if name in robust:
            r = robust[name]
            row["Cluster-robust beta (95% CI)"] = format_effect(
                *scaled(r["estimate"], r["low"], r["high"], spec["scale"]))
            row["Cluster-robust P"] = format_p(r["p"])
        if name in boot:
            b = boot[name]
            row["Bootstrap 95% CI"] = (
                f"{b['low'] * spec['scale']:+.3f} to "
                f"{b['high'] * spec['scale']:+.3f}")
        rows.append(row)

    for row, q in zip(rows, benjamini_hochberg(raw_p)):
        row["FDR q"] = format_p(q)
        row.pop("_p", None)
    # Keep P and q next to the estimate they belong to.
    order = ["Exposure", "Denominator", "beta (95% CI)", "P", "FDR q",
             "Cluster-robust beta (95% CI)", "Cluster-robust P",
             "Bootstrap 95% CI"]
    return [{k: row[k] for k in order if k in row} for row in rows]


def build_table2_across(fits: dict[str, Fit], schema: Schema) -> list[dict]:
    """The same four exposures across every model that was fitted."""
    rows: list[dict] = []
    for key in schema.exposures:
        name = f"{key}_within"
        spec = PRIMARY_EXPOSURES[key]
        row = {"Exposure": f"{spec['label']} ({spec['denominator']})"}
        for letter, fit in fits.items():
            if name not in fit.params.index:
                row[f"Model {letter}"] = "—"
                continue
            estimate, low, high = scaled(
                float(fit.params[name]), float(fit.conf.loc[name, "low"]),
                float(fit.conf.loc[name, "high"]), spec["scale"])
            row[f"Model {letter}"] = format_effect(estimate, low, high)
        rows.append(row)
    return rows


def build_between_rows(fit: Fit, schema: Schema) -> list[dict]:
    """The patient-mean coefficients, reported so they are not mistaken for the
    within-patient ones. They answer a question about who the patients are."""
    rows: list[dict] = []
    for key in schema.exposures:
        name = f"{key}_between"
        if name not in fit.params.index:
            continue
        spec = PRIMARY_EXPOSURES[key]
        estimate, low, high = scaled(
            float(fit.params[name]), float(fit.conf.loc[name, "low"]),
            float(fit.conf.loc[name, "high"]), spec["scale"])
        rows.append({"Exposure": f"Patient-mean {spec['label']}",
                     "Denominator": spec["denominator"],
                     "beta (95% CI)": format_effect(estimate, low, high),
                     "P": format_p(float(fit.pvalues[name]))})
    return rows


def build_model_comparison(ml_fits: dict[str, Fit]) -> list[dict]:
    """AIC, BIC and nested likelihood-ratio tests, all on ML fits."""
    rows: list[dict] = []
    previous: Fit | None = None
    for letter, fit in ml_fits.items():
        row = {"Model": letter, "Variables": MODEL_BLOCKS[letter]["label"],
               "Fixed terms": str(fit.k_fe),
               "AIC": f"{fit.aic:.1f}" if np.isfinite(fit.aic) else "—",
               "BIC": f"{fit.bic:.1f}" if np.isfinite(fit.bic) else "—"}
        if previous is None:
            row["LRT P vs previous"] = "—"
        elif fit.random_structure != previous.random_structure:
            # Comparing these would test the fixed and the random part at once.
            row["LRT P vs previous"] = "not comparable"
            row["AIC"] = row["AIC"] + " *"
            row["BIC"] = row["BIC"] + " *"
        else:
            statistic = 2.0 * (fit.llf - previous.llf)
            ddf = fit.k_fe - previous.k_fe
            if ddf > 0 and statistic > 0:
                row["LRT P vs previous"] = format_p(
                    float(stats.chi2.sf(statistic, ddf)))
            elif ddf <= 0:
                row["LRT P vs previous"] = "not nested"
            else:
                # The larger model fits no better; with nested fixed effects
                # that means the optimiser stopped short, not that the test is
                # significant in the other direction.
                row["LRT P vs previous"] = "check: no likelihood gain"
        rows.append(row)
        previous = fit
    return rows


# --------------------------------------------------------------------------- #
# Writing the tables out
# --------------------------------------------------------------------------- #

def write_csv(rows: list[dict] | list[tuple[str, str]], path: Path,
              columns: list[str] | None = None) -> None:
    if rows and isinstance(rows[0], tuple):
        frame = pd.DataFrame(
            [{"Characteristic": k.replace("__SECTION__", ""), "Value": v}
             for k, v in rows])
    else:
        frame = pd.DataFrame(rows, columns=columns)
    frame.to_csv(path, index=False)
    print(f"  wrote {path}")


def render_table(rows: list[list[str]], widths: list[int], title: str,
                 footnote: str, path: Path, bold_rows: set[int] = frozenset(),
                 section_rows: set[int] = frozenset()) -> None:
    """Draw a table as an image, in a monospace grid that survives any cell."""
    body_size, note_size = 9.2, 8.2
    line = body_size * 1.55 / 72.0
    note_lines = footnote.count("\n") + 1 if footnote else 0
    height = 0.55 + line * (len(rows) + 1) + note_size * 1.5 / 72.0 * note_lines + 0.45
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
        weight = "bold" if index == 0 or index in bold_rows else "normal"
        colour = "#1A1A1A"
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


def dict_rows_to_grid(rows: list[dict]) -> tuple[list[list[str]], list[int]]:
    if not rows:
        return [], []
    columns = list(rows[0])
    grid = [columns] + [[str(row.get(c, "")) for c in columns] for row in rows]
    widths = [max(len(line[i]) for line in grid) for i in range(len(columns))]
    return grid, widths


# --------------------------------------------------------------------------- #

def banner(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def main() -> int:
    args = parse_args()
    if not HAVE_STATSMODELS:
        sys.stderr.write("statsmodels is required: pip install statsmodels\n")
        return 1
    if not args.data.is_file():
        sys.stderr.write(f"No such file: {args.data}\n")
        return 1

    frame = (pd.read_parquet(args.data) if args.data.suffix == ".parquet"
             else pd.read_csv(args.data, low_memory=False))
    schema = resolve_schema(frame, args)

    if args.inspect:
        describe_schema(frame, schema, args)
        print("\n--inspect: nothing was fitted. Re-run without it when the "
              "columns above look right.")
        return 0

    args.outdir.mkdir(parents=True, exist_ok=True)
    print(f"\nTABLES WILL BE WRITTEN TO:\n  {args.outdir.resolve()}")

    banner("Cohort and window")
    frame[schema.time] = pd.to_numeric(frame[schema.time], errors="coerce")
    window = frame.loc[frame[schema.time].between(args.window_start,
                                                  args.window_end)].copy()
    print(f"  Window {args.window_start:g} to {args.window_end:g} min from "
          f"induction: {window[schema.patient].nunique()} patients, "
          f"{len(window):,} epochs")
    if schema.missing:
        print(f"  Columns not found, so their terms are dropped: "
              f"{', '.join(schema.missing)}")

    # ---- Table 1 ------------------------------------------------------------
    banner("Table 1 — patient characteristics")
    table1, baseline_note = build_table1(frame, window, schema, args)
    for key, value in table1:
        if key.startswith("__SECTION__"):
            print(f"\n  {key.replace('__SECTION__', '').upper()}")
        else:
            print(f"    {key:<52} {value}")

    write_csv(table1, args.outdir / "table1_patient_characteristics.csv")
    if not args.no_png:
        grid = [["Characteristic", f"Value (n = {window[schema.patient].nunique()})"]]
        sections: set[int] = set()
        for key, value in table1:
            if key.startswith("__SECTION__"):
                sections.add(len(grid))
                grid.append([key.replace("__SECTION__", "").upper(), ""])
            else:
                grid.append([key, value])
        widths = [max(len(r[0]) for r in grid), max(len(r[1]) for r in grid)]
        render_table(grid, widths, "Table 1. Patient characteristics, "
                     "baseline physiology, medication exposure and completeness",
                     "Numeric values are median (IQR). Medication rows are "
                     "patients exposed (%) followed by the dose distribution "
                     "among those exposed.\n"
                     f"Baseline is the {baseline_note}.",
                     args.outdir / "table1_patient_characteristics.png",
                     section_rows=sections)

    # ---- the models ---------------------------------------------------------
    banner("Models")
    prepared = add_decomposition(window, schema)
    prepared, time_columns = time_terms(prepared, schema, args)
    prepared = rename_for_formula(prepared, schema)
    print(f"  Time enters as: {', '.join(time_columns)} "
          f"({args.time_basis}, centered at "
          f"{(args.window_start + args.window_end) / 2:g} min)")
    print("  Every exposure is split into a within-patient deviation and a "
          "patient mean, so the reported coefficient is within-patient.")

    plan = {letter: model_terms(schema, time_columns, spec["blocks"])
            for letter, spec in MODEL_BLOCKS.items()}
    if not schema.context:
        plan.pop("D", None)
        print("  Model D skipped: no --context-columns given.")

    # Fit every model on the SAME rows, so a coefficient that moves between
    # models moves because of what was added and not because the sample did.
    needed = {term_column(t) for terms in plan.values() for t in terms}
    common = prepared.dropna(subset=["outcome"] + sorted(needed & set(prepared.columns)))
    print(f"  Common complete-case sample: "
          f"{common[schema.patient].nunique()} patients, {len(common):,} epochs")

    # Screen on the sample that will actually be fitted, not on the whole file.
    plan, _dropped = screen_terms(common, plan, schema)

    slope = time_columns[0] if args.random_slope else None
    reml_fits, ml_fits, complete = fit_all(common, plan, schema.patient, slope)
    if slope is not None and not complete:
        print(f"\n  A random slope on {slope} was asked for, but at least one "
              f"model in the sequence would not fit or converge with it.")
        print("  Refitting the WHOLE sequence with a random intercept only. "
              "Letting just that model fall back would leave AIC, BIC and the "
              "likelihood-ratio tests comparing different random structures.")
        slope = None
        reml_fits, ml_fits, complete = fit_all(common, plan, schema.patient,
                                               None)
    print(f"  Random effects: per patient, "
          f"{(f'intercept + slope on {slope}' if slope else 'intercept only')}")
    for letter, terms in plan.items():
        reml = reml_fits.get(letter)
        state = "ok" if reml is not None and not reml.note else "CHECK"
        print(f"    Model {letter} ({MODEL_BLOCKS[letter]['label']}): "
              f"{len(terms)} terms, {state}"
              + (f" — {reml.note}" if reml is not None and reml.note else ""))

    if args.primary_model not in reml_fits:
        available = ", ".join(reml_fits) or "none"
        sys.stderr.write(f"\nPrimary model {args.primary_model} did not fit. "
                         f"Available: {available}\n")
        return 1
    primary = reml_fits[args.primary_model]

    # ---- model comparison ---------------------------------------------------
    banner("Model comparison (maximum likelihood)")
    print("  AIC, BIC and the likelihood-ratio test compare models that differ "
          "in their FIXED effects, so all three require ML, not REML. The "
          "coefficients reported in Table 2 come from the REML fit.")
    comparison = build_model_comparison(ml_fits)
    grid, widths = dict_rows_to_grid(comparison)
    for line in grid:
        print("    " + "  ".join(str(c).ljust(w) for c, w in zip(line, widths)))
    write_csv(comparison, args.outdir / "table_model_comparison.csv")
    if not args.no_png:
        render_table(grid, widths, "Model comparison",
                     "Fitted by maximum likelihood on one common complete-case "
                     "sample. Each likelihood-ratio test compares the model "
                     "with the one above it.",
                     args.outdir / "table_model_comparison.png")

    # ---- Table 2 ------------------------------------------------------------
    banner(f"Table 2 — primary within-patient associations (Model "
           f"{args.primary_model})")
    robust = cluster_robust_within(common, schema, plan[args.primary_model])
    boot = {}
    if args.bootstrap > 0:
        print(f"  Patient-level bootstrap, {args.bootstrap} resamples. This "
              f"refits the model each time.")
        boot = patient_bootstrap(common, schema, plan[args.primary_model],
                                 args.bootstrap, args.seed)

    table2 = build_table2_primary(primary, schema, robust, boot)
    grid, widths = dict_rows_to_grid(table2)
    for line in grid:
        print("    " + "  ".join(str(c).ljust(w) for c, w in zip(line, widths)))

    adjustment = MODEL_BLOCKS[args.primary_model]["label"]
    footnote = (
        f"Model {args.primary_model}: {adjustment}. Linear mixed model, "
        f"patient random intercept, REML. {primary.n_patients} patients, "
        f"{primary.n_obs:,} epochs.\n"
        "Coefficients are within-patient: each exposure enters as the "
        "deviation from that patient's own mean over the window, with the "
        "patient mean carried separately.\n"
        "FDR q is Benjamini-Hochberg within the four primary exposures; "
        "medications are a separate family and are not included in it.\n"
        "The cluster-robust column refits with patient fixed effects and a "
        "sandwich clustered on patient, which stays valid under serial "
        "correlation.")
    write_csv(table2, args.outdir / "table2_primary.csv")
    if not args.no_png:
        render_table(grid, widths,
                     f"Table 2. Primary within-patient associations with "
                     f"cerebral O₂ (Model {args.primary_model})",
                     footnote, args.outdir / "table2_primary.png")

    between = build_between_rows(primary, schema)
    if between:
        print("\n  Patient-mean (between-patient) coefficients, reported so "
              "they are not read as the within-patient ones:")
        bgrid, bwidths = dict_rows_to_grid(between)
        for line in bgrid:
            print("    " + "  ".join(str(c).ljust(w) for c, w in zip(line, bwidths)))
        write_csv(between, args.outdir / "table2_between_patient.csv")

    banner("Table 2b — the same exposures across models")
    across = build_table2_across(reml_fits, schema)
    grid, widths = dict_rows_to_grid(across)
    for line in grid:
        print("    " + "  ".join(str(c).ljust(w) for c, w in zip(line, widths)))
    write_csv(across, args.outdir / "table2_across_models.csv")
    if not args.no_png:
        render_table(grid, widths,
                     "Table 2b. Stability of the primary associations across "
                     "the adjustment sequence",
                     "All models fitted on one common complete-case sample, so "
                     "a coefficient that moves does so because of what was "
                     "added rather than because the sample changed.\n"
                     + " · ".join(f"{k}: {MODEL_BLOCKS[k]['label']}"
                                       for k in reml_fits),
                     args.outdir / "table2_across_models.png")

    banner("What to check before these go in the manuscript")
    print(textwrap.dedent("""\
      1. Run --inspect and confirm every column resolved to what you expect,
         especially the medication columns and their units.
      2. Confirm the window. Table 1's baseline row says which window it used;
         if the file holds no pre-induction epochs it falls back to the first
         three epochs after induction, which is NOT a baseline.
      3. The within-patient and patient-mean coefficients answer different
         questions. Table 2 reports the first; table2_between_patient.csv has
         the second. Quote the within-patient numbers as the primary result.
      4. Where the model-based and cluster-robust intervals disagree, report
         the cluster-robust one. Ten-second epochs are strongly autocorrelated
         and the mixed model's interval does not account for that.
      5. The LRT and AIC/BIC here are ML fits; if the earlier numbers came from
         REML fits they are not comparable and should be replaced."""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
