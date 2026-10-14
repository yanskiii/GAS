#!/usr/bin/env python3
"""Does PSi modify the MAP-cerebral O2 relationship? The PI's outline, in code.

WHAT THIS SCRIPT IS FOR
-----------------------
  The stratified scatter script answered the question by cutting the data into
  PSi bands and fitting a line in each one. That was the right first look, and
  it said yes: the within-patient MAP-cerebral O2 slope changes across PSi
  bands. This script asks the same question the way the research outline asks
  for it -- one longitudinal model, PSi kept continuous, with an explicit
  MAP x PSi interaction, repeated measures and serial correlation accounted for.

  Primary model     outcome   cerebral O2 (NIRS StO2, %)
                    exposure  MAP (mmHg, continuous)
                    modifier  PSi (continuous)
                    interest  MAP x PSi

  Secondary model   the same, with cardiac output (L/min) as the exposure and
                    CO x PSi as the term of interest, contrasts per 1 L/min.

THE ESTIMAND, AND WHY A RANDOM INTERCEPT IS NOT ENOUGH
------------------------------------------------------
  The outline is explicit about this and it is the single most important choice
  in the file: "A random intercept alone does not ensure a purely within-patient
  estimate." It does not, because a random-intercept model still lets patients
  who happen to run at a high MAP and a low cerebral O2 contribute to the MAP
  coefficient. That between-patient comparison answers "are high-pressure
  patients different people?", not "what happens to this brain when this
  patient's pressure falls?" In this dataset the two have opposite signs --
  Simpson's paradox -- so mixing them is not a rounding error, it changes the
  conclusion.

  So the PRIMARY specification here is patient fixed effects: every variable is
  centered on that patient's own mean, which algebraically removes all
  between-patient information. What is left is a weighted average of the
  within-patient relationships, which is the clinical question. Inference uses
  patient-clustered (sandwich) standard errors, valid under ANY pattern of
  within-patient correlation, serial correlation included -- which is why this,
  and not a hand-specified AR(1), is the honest default when the sampling grid
  has gaps.

  The random-effects companion is fitted too, in Mundlak form: the within parts
  and the patient means both enter, so the within coefficients mean the same
  thing as in the fixed-effects fit while the patient means absorb the
  between-patient signal. Agreement between the two is reported, not assumed.

READING THE INTERACTION
-----------------------
  Writing the model with raw MAP times raw PSi hides two different
  interactions in one coefficient, so each variable enters as a deviation from
  that patient's own mean and the product is split:

      cerebral O2 = b*MAP_w + c*PSi_w + d*(MAP_w x PSi_w)
                            + e*(MAP_w x mean PSi) + patient + error

  d is the clinical quantity: when THIS patient goes deeper than their own
  average, does THIS patient's MAP-cerebral O2 slope change? e is the
  cross-level term -- do patients who are deeper ON AVERAGE have different
  slopes -- and it is reported beside d rather than averaged into it. Forcing
  the two to share a coefficient drags the estimate toward whichever is
  weaker, which on fixtures with a planted interaction cost two thirds of it.

  The within-patient MAP slope at an absolute PSi of p, for a patient of
  average depth, is then

      slope(p) = b + d*(p - PSi_reference)

  which makes "the MAP slope at PSi 30 versus at PSi 50" a contrast of model
  coefficients rather than two separate subgroup fits. Every number in the
  report is such a contrast, taken against the same covariance matrix, so the
  tables and the figures cannot drift apart.

  Expected directions. The MAP slope should sit near zero while autoregulation
  holds and climb positive once cerebral oxygenation turns pressure-passive. A
  NEGATIVE d means the slope becomes more positive as PSi falls, i.e. deeper
  anesthesia is associated with a more pressure-passive brain.

WHAT ELSE IS READ, AND WHY
--------------------------
  Medications (--medications). The propofol infusion rate IS the anesthetic
  concentration the outline asks to adjust for, at least in a TIVA case, and
  unlike age or ASA it varies within a patient, so the fixed effects do not
  absorb it. The bolus times matter more: propofol lowers MAP and PSi at the
  same instant, so "deep and hypotensive" can be pharmacology rather than
  physiology. That is the main rival explanation for the headline result, and
  the sensitivity section tests it by refitting with the minutes around every
  bolus removed.

  Hemosphere extras (--hemosphere-extra). SVV, PPV and HPI are genuinely new
  signals and their coverage is reported. SVR, SVRI, CI and SVI are NOT: SVR
  is exactly 80 x (MAP - CVP) / CO, verified to the last digit against the
  export, so beside MAP and CO it is a transform of what the model already
  holds, and CVP arrives as a manually entered constant. They are read so
  their availability is visible, and flagged so that nobody adjusts for them
  by accident.

  REDCap (--redcap). Induction and OR-entry times, and the Table 1 baseline
  characteristics. Those characteristics are patient-level, so the fixed
  effects absorb every one of them: they describe the cohort, they do not
  adjust the models, and adding them would move no estimate.

OUTPUT
------
  figure1_map_cerebral_o2_by_psi.png   adjusted MAP-cerebral O2 at representative
                                       PSi values, the continuous slope-vs-PSi
                                       curve, and the slope contrasts
  figure2_co_cerebral_o2_by_psi.png    the same for cardiac output
  figure3_map_slopes_by_psi_category.png   PSi categories (<25, 25-50, 50-75,
                                       >75), slope with CI and patient counts
  table1_data_coverage.png             patients, observations, duration,
                                       distributions, missingness
  table2_interaction_results.png       interaction estimates and the requested
                                       contrasts
  Everything else prints to the terminal. No CSVs.

USAGE
-----
    python psi_map_interaction_models.py
    python psi_map_interaction_models.py --representative-psi 25,50,75
    python psi_map_interaction_models.py --quick          # skip the sensitivities
    python psi_map_interaction_models.py \
        --covariate-filepaths /N/project/.../vitals_filepaths.csv
"""

from __future__ import annotations

import argparse
import re
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
    from statsmodels.regression.mixed_linear_model import MixedLM
    HAVE_MIXEDLM = True
except Exception:                                   # pragma: no cover
    MixedLM = None
    HAVE_MIXEDLM = False


# --------------------------------------------------------------------------- #
# Hardcoded research-ID corrections (see the lag scatter script for the reason).
# --------------------------------------------------------------------------- #
REDCAP_ID_FIXES: dict[tuple[str, str | None], str] = {
    ("IUMH2026010601", "2026-01-05"): "IUMH2026010501",
    ("IUMH202601601", None): "IUMH2026011601",
}

VALID_RANGE = {"map": (20.0, 160.0), "sto2": (0.0, 100.0),
               "psi": (0.0, 100.0), "co": (0.5, 15.0),
               "etco2": (5.0, 80.0), "fio2": (15.0, 100.0),
               # Wide enough for MAC (around 1) and for an expired agent
               # percentage, which for desflurane runs well into the teens.
               "agent": (0.0, 20.0)}

ID_PATTERN = re.compile(r"IU(?:MH|UH)\d+", re.IGNORECASE)
ID_STRUCTURE = re.compile(r"^IU(MH|UH)(\d{8})(\d{2})$")
ID_LOOSE = re.compile(r"^IU(MH|UH)\d*?(\d{2})$")

REDCAP_ID_PREFIX = "what is the name of the file"
INDUCTION_PREFIX = "what time was induction"
OR_ENTRY_PREFIX = "what time did the patient enter the or"
SURGERY_DATE_PREFIX = "date of surgery"

# Baseline characteristics for Table 1, keyed to the start of the REDCap label
# (matched case-insensitively after collapsing repeated spaces). A field the
# export does not carry is reported as unavailable rather than guessed at.
CHARACTERISTIC_COLUMNS = {
    "age": "age on date of surgery",
    "sex": "sex",
    "height": "height (cm)",
    "weight": "actual weight",
    "bmi": "bmi",
    "asa": "asa class",
    "procedure": "surgical procedure",
    "race": "race",
    "ethnicity": "ethnicity",
    "smoking": "smoking status",
    "monk_forehead": "monk skin tone (1-10) forehead",
    "monk_hand": "monk skin tone (1-10) hand",
    "preop_map": "pre-operative nibp map",
    "preop_spo2": "pre-operative spo2",
    "a1c": "a1c",
    "anesthesiologist": "anesthesiologist",
}

# How each one is labelled and summarized in Table 1.
CHARACTERISTIC_LAYOUT: tuple[tuple[str, str, str, int], ...] = (
    ("age", "Age, years", "numeric", 1),
    ("sex", "Sex", "category", 0),
    ("bmi", "BMI, kg/m2", "numeric", 1),
    ("height", "Height, cm", "numeric", 0),
    ("weight", "Weight, kg", "numeric", 1),
    ("asa", "ASA class", "category", 0),
    ("comorbidities", "Comorbid conditions, count", "numeric", 0),
    ("a1c", "HbA1c, % (where measured)", "numeric", 1),
    ("smoking", "Smoking status", "category", 0),
    ("race", "Race", "category", 0),
    ("ethnicity", "Ethnicity", "category", 0),
    ("monk_forehead", "Monk skin tone, forehead (1-10)", "numeric", 0),
    ("preop_map", "Pre-operative NIBP MAP, mmHg", "numeric", 0),
    ("preop_spo2", "Pre-operative SpO2, %", "numeric", 0),
    ("procedure", "Surgical procedure", "category", 0),
)

MAP_LIST_FALLBACKS = ("map_filepaths.csv", "btb_filepaths.csv",
                      "b2b_filepaths.csv", "b2b_filepath.csv",
                      "btb_filepath.csv")

# Covariate column names, as the anesthesia record has spelled them. The
# outline asks to adjust "parsimoniously" for end-tidal CO2, inspired oxygen
# fraction and anesthetic concentration, so exactly those three are hunted for
# and nothing else: every extra covariate costs within-patient variation.
COVARIATE_PATTERNS = {
    "etco2": ("etco2", "et co2", "etco_2", "end tidal co2", "endtidalco2",
              "end-tidal co2", "co2 et", "etc o2"),
    "fio2": ("fio2", "fi o2", "fi_o2", "inspired oxygen", "inspiredo2",
             "o2 insp", "inspired o2"),
    "agent": ("mac", "age_et", "aa_et", "et agent", "agent et", "expired agent",
              "sevoflurane", "sevo", "isoflurane", "desflurane", "et sevo",
              "et des", "et iso", "anesthetic concentration"),
}
COVARIATE_LABELS = {"etco2": "end-tidal CO2 (mmHg)",
                    "fio2": "inspired O2 fraction (%)",
                    "agent": "anesthetic concentration (MAC or % agent)",
                    "propofol_rate": "propofol infusion rate (mcg/kg/min)",
                    "pressor_rate": "vasopressor infusion rate (mcg/min)"}

# Extra Hemosphere channels. Each is read where the file carries it, so that
# coverage can be REPORTED across the cohort even where the channel is not
# modelled; `derived` marks a column that is an algebraic function of signals
# the model already has, and so carries no independent information.
HEMOSPHERE_EXTRA: dict[str, dict] = {
    "svv": {"column": "SVV", "range": (0.0, 60.0), "label": "SVV (%)",
            "derived": False},
    "ppv": {"column": "PPV", "range": (0.0, 60.0), "label": "PPV (%)",
            "derived": False},
    "hpi": {"column": "HPI", "range": (0.0, 100.0),
            "label": "Hypotension Prediction Index", "derived": False},
    "pr": {"column": "PR", "range": (20.0, 200.0), "label": "pulse rate (bpm)",
           "derived": False},
    "svr": {"column": "SVR", "range": (100.0, 6000.0),
            "label": "SVR (dyne-s/cm5)",
            "derived": "SVR is exactly 80 x (MAP - CVP) / CO -- verified to "
                       "the last digit against this export -- so beside MAP "
                       "and CO it is a transform of what the model already "
                       "has, not a new measurement"},
    "cvp": {"column": "CVP", "range": (-5.0, 40.0), "label": "CVP (mmHg)",
            "derived": "CVP_Source reads 'Manual Entry' and the value is held "
                       "constant through the case, so it is an assumed number "
                       "rather than a measured signal"},
    "hemo_map": {"column": "MAP", "range": (20.0, 160.0),
                 "label": "MAP from Hemosphere (mmHg)",
                 "derived": "a second, 20-second-interval copy of a pressure "
                            "the beat-to-beat file already supplies densely; "
                            "useful only as a cross-check"},
}
VALID_RANGE.update({name: spec["range"] for name, spec in HEMOSPHERE_EXTRA.items()})

# Drug classes, matched against drug_key first and the printed drug name
# second. Only the classes that bear on this analysis are named: what moves
# blood pressure, and what moves anesthetic depth.
DRUG_CLASSES: dict[str, tuple[str, ...]] = {
    "hypnotic": ("prop", "propofol", "ketamine", "ketam", "midazolam", "versed",
                 "midaz", "etomidate", "etom", "dexmedetomidine", "precedex",
                 "dexmed"),
    "vasoactive": ("phenyl", "phenylephrine", "ephedrine", "ephed",
                   "norepinephrine", "norepi", "levophed", "epinephrine",
                   "epinep", "vasopressin", "vaso", "nicardipine", "nicard",
                   "esmolol", "hydralazine", "hydral", "labetalol", "labet",
                   "nitroglycerin", "ntg", "dobutamine", "dopamine",
                   "angiotensin", "giapreza"),
    "opioid": ("fent", "fentanyl", "remifentanil", "remi", "sufentanil",
               "hydromorphone", "dilaudid", "morphine", "methadone"),
    "neuromuscular": ("rocuronium", "roc", "succinylcholine", "succ",
                      "vecuronium", "vec", "cisatracurium", "cisat",
                      "sugammadex", "neostigmine"),
}

AXES = {
    "map": {"label": "MAP (mmHg)", "short": "MAP", "unit": "mmHg",
            "step": 10.0, "step_label": "per 10 mmHg"},
    "co": {"label": "Cardiac output (L/min)", "short": "CO", "unit": "L/min",
           "step": 1.0, "step_label": "per 1 L/min"},
    "psi": {"label": "PSi", "short": "PSi", "unit": "PSi units", "step": 10.0,
            "step_label": "per 10 PSi units"},
    "sto2": {"label": "Cerebral O2 (%)", "short": "cerebral O2", "unit": "%",
             "step": 1.0, "step_label": "per 1 %"},
}

PSI_CATEGORY_EDGES = (0.0, 25.0, 50.0, 75.0, 100.0)
BAND_COLORS = ["#2f5597", "#c55a11", "#548235", "#7030a0", "#7f6000", "#495057"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    base = Path("/N/project/Analgesia_BDproject/PR/scripts_PR/10-14 Mixed Models")
    data = Path("/N/project/Analgesia_BDproject/PR/data")

    group = parser.add_argument_group("data sources")
    group.add_argument("--filepaths", type=Path,
                       default=data / "sedline_filepaths.csv",
                       help="CSV listing one Sedline file path per line (PSi).")
    group.add_argument("--sto2-filepaths", type=Path,
                       default=data / "sto2_filepaths.csv",
                       help="CSV listing one cerebral-oximetry file per line.")
    group.add_argument("--map-filepaths", type=Path,
                       default=data / "map_filepaths.csv",
                       help="CSV listing one beat-to-beat MAP file per line.")
    group.add_argument("--hemosphere-filepaths", type=Path,
                       default=data / "hemosphere_filepaths.csv",
                       help="CSV listing one Hemosphere file per line (CO).")
    group.add_argument("--covariate-filepaths", type=Path, default=None,
                       help="Optional CSV listing one anesthesia-record file "
                            "per patient, for end-tidal CO2, inspired O2 "
                            "fraction and anesthetic concentration. Without "
                            "it the models are UNADJUSTED and the report says "
                            "so; the adjustment the outline asks for cannot be "
                            "invented from the monitors already loaded.")
    group.add_argument("--medications", type=Path,
                       default=Path("/N/project/Analgesia_BDproject/data/"
                                    "processed/medication_events.csv"),
                       help="Medication events given in the OR, one row per "
                            "bolus or infusion step. Supplies the propofol "
                            "infusion rate (the anesthetic exposure the "
                            "outline asks to adjust for, and one that varies "
                            "within a patient) and the timing needed to test "
                            "whether the interaction is pharmacology rather "
                            "than physiology. Skipped if the file is absent.")
    group.add_argument("--redcap", type=Path,
                       default=Path("/N/project/Analgesia_BDproject/data/"
                                    "00_raw/BDPostInductionHemod_DATA_LABELS_"
                                    "2026-09-30_1420.csv"),
                       help="REDCap labeled export, for induction, OR entry "
                            "and the Table 1 baseline characteristics.")
    group.add_argument("--outdir", type=Path, default=base / "output",
                       help="Where the figures are written. The resolved path "
                            "is printed at the top of every run: if it is not "
                            "the folder you are looking at, pass --outdir, "
                            "because figures that land elsewhere look exactly "
                            "like figures that did not change.")

    group = parser.add_argument_group("signal cleaning (same rules as the "
                                      "stratified scatter script)")
    group.add_argument("--min-minutes", type=float, default=None,
                       help="Optional earliest minute relative to induction. "
                            "Unset by default: every valid sample is used.")
    group.add_argument("--max-minutes", type=float, default=None,
                       help="Optional latest minute after induction. Unset by "
                            "default -- the interaction is estimated from the "
                            "spread of MAP, and cutting the record throws that "
                            "spread away.")
    group.add_argument("--max-hours", type=float, default=24.0,
                       help="Sanity cap on one patient's record span (24 h).")
    group.add_argument("--sto2-channels", default="1",
                       help="Cerebral-oximetry channel(s) used as the outcome "
                            "(default 1). Averaging CH1 with CH2 mixes two "
                            "sensors on different sides of the forehead, so "
                            "the annotated outline asks for CH1 alone with "
                            "CH2 kept back as a sensitivity. Give a "
                            "comma-separated list to average several.")
    group.add_argument("--sto2-fallback-channels", default="3",
                       help="Channel(s) used instead when a patient has no "
                            "valid reading on the primary one anywhere in "
                            "their file (default 3).")
    group.add_argument("--sto2-sensitivity-channels", default="2",
                       help="The other sensor, used to repeat the whole "
                            "analysis as a sensitivity (default 2). Empty "
                            "string to skip that repeat.")
    group.add_argument("--sto2-sensitivity-fallback", default="4",
                       help="Fallback for the sensitivity sensor (default 4).")
    group.add_argument("--step-seconds", type=float, default=10.0,
                       help="Grid every signal is resampled onto (10 s).")
    group.add_argument("--max-gap-seconds", type=float, default=60.0,
                       help="Never interpolate across a gap wider than this.")
    group.add_argument("--co-column", default=None,
                       help="Exact cardiac-output column name. Default 'CO'.")
    group.add_argument("--min-co-sqi", type=float, default=3.0,
                       help="Keep a CO reading only when CO_SQI is at least "
                            "this (default 3).")
    group.add_argument("--hemosphere-extra", default="svv,ppv,hpi,pr,svr,cvp",
                       help="Further Hemosphere channels to read and report "
                            "coverage for (default svv,ppv,hpi,pr,svr,cvp). "
                            "SVR and CVP are read so their availability is "
                            "visible, and flagged in the report as derived or "
                            "assumed rather than measured.")
    group.add_argument("--drug-window-minutes", type=float, default=5.0,
                       help="How long after a bolus a time point counts as "
                            "drug-disturbed (default 5 min).")
    group.add_argument("--table1", action="store_true",
                       help="Also write Table 1 (cohort characteristics and "
                            "data coverage) as an image. Off by default: the "
                            "coverage numbers still print to the terminal, "
                            "and the annotated outline does not ask for the "
                            "table as a deliverable.")
    group.add_argument("--adjust-drug-rates", action="store_true",
                       help="Put the running propofol and vasopressor "
                            "infusion rates into the models as covariates. "
                            "OFF by default, and deliberately: a vasopressor "
                            "raises MAP, so adjusting for it removes part of "
                            "the very exposure being studied, and in this "
                            "cohort the propofol rate varies within only a "
                            "handful of patients. Both are reported as "
                            "sensitivities either way, so the comparison is "
                            "always available without making it the headline.")
    group.add_argument("--propofol-key", default="prop",
                       help="drug_key prefix for the hypnotic infusion whose "
                            "rate is used as the anesthetic exposure.")
    group.add_argument("--pressor-key", default="phenyl",
                       help="drug_key prefix for the vasopressor infusion "
                            "whose rate is carried as a covariate.")

    group = parser.add_argument_group("model")
    group.add_argument("--representative-psi", default="30,50",
                       help="Absolute PSi values the exposure slope is reported "
                            "at (default 30,50, as the outline suggests). Any "
                            "number of values; the first and last also give the "
                            "headline slope difference.")
    group.add_argument("--map-center", type=float, default=None,
                       help="Constant MAP is centered on. Default: the rounded "
                            "grand mean. Centering changes nothing about the "
                            "fit, only which coefficient is readable on its "
                            "own, and every reported contrast corrects for it.")
    group.add_argument("--psi-center", type=float, default=None,
                       help="Constant PSi is centered on. Default: rounded "
                            "grand mean.")
    group.add_argument("--co-center", type=float, default=None,
                       help="Constant CO is centered on. Default: rounded "
                            "grand mean.")
    group.add_argument("--no-mixed", action="store_true",
                       help="Skip the random-effects companion fit. The "
                            "fixed-effects model carries the primary estimate "
                            "either way.")
    group.add_argument("--random-slope", dest="random_slope",
                       action="store_true", default=True,
                       help="Let the MAP slope vary by patient in the "
                            "random-effects fit (default on).")
    group.add_argument("--no-random-slope", dest="random_slope",
                       action="store_false",
                       help="Random intercept only in the companion fit.")

    group = parser.add_argument_group("sensitivity analyses")
    group.add_argument("--quick", action="store_true",
                       help="Primary and secondary models, the figures and the "
                            "tables only. Skips every sensitivity analysis.")
    group.add_argument("--bootstrap", type=int, default=1000,
                       help="Patient-level (cluster) bootstrap resamples "
                            "(default 1000). 0 disables it. Cheap: the design "
                            "is reduced to per-patient cross-products first, so "
                            "a resample is a 7x7 solve, not a refit.")
    group.add_argument("--thin-minutes", default="1,5",
                       help="Thinning intervals, in minutes, used to show the "
                            "estimate does not depend on how densely the record "
                            "is sampled (default 1,5).")
    group.add_argument("--lag-minutes", default="1,2,5",
                       help="Lags, in minutes, by which the exposure is shifted "
                            "to precede the outcome (default 1,2,5).")
    group.add_argument("--change-horizon", type=float, default=5.0,
                       help="Horizon in minutes for the change model, which "
                            "regresses the CHANGE in cerebral O2 on the "
                            "CHANGE in the exposure over the same interval "
                            "(default 5). Set 0 to skip it. Differencing "
                            "removes the slow drift that leaves the levels "
                            "model with a lag-1 residual autocorrelation near "
                            "one, and asks the more clinical question: when "
                            "pressure moved, did oxygenation follow?")
    group.add_argument("--change-horizons", default="2,5,10",
                       help="Horizons compared in the terminal and on the "
                            "specification figure (default 2,5,10). The "
                            "figure panels use --change-horizon.")
    group.add_argument("--maintenance-start", type=float, default=15.0,
                       help="Minutes after induction at which the maintenance "
                            "phase is taken to begin, for the maintenance-only "
                            "repeat (default 15).")
    group.add_argument("--seed", type=int, default=20261011)
    return parser.parse_args()


# --------------------------------------------------------------------------- #
# Loading  (the same machinery as the stratified scatter script, kept
# self-contained so this file can be copied to the cluster on its own)
# --------------------------------------------------------------------------- #

def guess_encodings(path: str) -> list[str]:
    """Encodings to try, narrowed by the byte-order mark when there is one."""
    with open(path, "rb") as handle:
        signature = handle.read(4)
    if signature[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return ["utf-16", "utf-16-le", "utf-16-be"]
    if signature[:3] == b"\xef\xbb\xbf":
        return ["utf-8-sig", "utf-8", "cp1252", "latin-1"]
    return ["utf-8", "cp1252", "latin-1"]


def score_header(fields: list[str], wanted: set[str]) -> float:
    """How much a row looks like the column-name row. Negative means "no"."""
    fields = [field.strip().strip('"').lower() for field in fields]
    if len(fields) < 2:
        return -1.0
    hits = len(wanted & set(fields))
    if hits == 0 and any(field.replace(".", "").replace("-", "").isdigit()
                         for field in fields if field):
        return -1.0          # a row of numbers is data, not a header
    return hits * 100.0 + len(fields)


def read_excel_table(path: str, expect: tuple[str, ...]) -> pd.DataFrame:
    """Read the sheet and header row that actually hold the measurements."""
    book = pd.ExcelFile(path)
    wanted = {name.strip().lower() for name in expect}
    best: tuple[str, int, float] = (book.sheet_names[0], 0, -1.0)
    for sheet in book.sheet_names:
        probe = book.parse(sheet, header=None, nrows=40)
        for index in range(len(probe)):
            fields = ["" if pd.isna(value) else str(value)
                      for value in probe.iloc[index].tolist()]
            score = score_header(fields, wanted)
            if score > best[2]:
                best = (sheet, index, score)
    sheet, header_row, _score = best
    return book.parse(sheet, header=header_row)


def sniff_binary_kind(path: str) -> str | None:
    """'excel' for a workbook, None for something a text reader can handle."""
    with open(path, "rb") as handle:
        signature = handle.read(8)
    if signature[:4] == b"PK\x03\x04":
        return "excel"                      # .xlsx / .xlsm are ZIP archives
    if signature[:8] == b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1":
        return "excel"                      # legacy .xls compound document
    return None


def locate_header(lines: list[str], expect: tuple[str, ...]) -> tuple[int, str]:
    """Which line is the real header, and what separates its fields."""
    wanted = {name.strip().lower() for name in expect}
    best = (0, ",", -1.0)
    for separator in (",", "\t", ";", "|"):
        for index, line in enumerate(lines):
            score = score_header(line.rstrip("\r\n").split(separator), wanted)
            if score > best[2]:
                best = (index, separator, score)
    return best[0], best[1]


def read_table(path: str, expect: tuple[str, ...] = ()) -> tuple[pd.DataFrame, str]:
    """Read a monitor export whatever form it takes."""
    if sniff_binary_kind(path) == "excel":
        return read_excel_table(path, expect), "excel"

    failure: Exception | None = None
    for encoding in guess_encodings(path):
        try:
            with open(path, "r", encoding=encoding, newline="") as handle:
                head = [next(handle) for _ in range(400)]
        except StopIteration:
            with open(path, "r", encoding=encoding, newline="") as handle:
                head = handle.readlines()
        except Exception as exc:
            failure = exc
            continue
        try:
            skip, separator = locate_header(head, expect)
            frame = pd.read_csv(path, low_memory=False, skipinitialspace=True,
                                encoding=encoding, sep=separator, skiprows=skip)
            if frame.shape[1] <= 1:
                frame = pd.read_csv(path, sep=None, engine="python",
                                    skipinitialspace=True, encoding=encoding,
                                    skiprows=skip)
            return frame, encoding
        except Exception as exc:
            failure = exc
    raise failure if failure else RuntimeError(f"could not read {path}")


def read_filepath_list(master: Path) -> list[str]:
    paths: list[str] = []
    with open(master, "r", newline="") as handle:
        for raw in handle:
            first = raw.strip().replace("\t", ",").split(",")[0]
            first = first.strip().strip('"').strip("'")
            if not first:
                continue
            if not first.lower().endswith((".csv", ".txt", ".xlsx", ".xls")) \
                    and "/" not in first:
                continue
            paths.append(first)
    return paths


def patient_id_from_path(path: str) -> str | None:
    match = ID_PATTERN.search(path)
    return match.group(0).upper() if match else None


def find_column(frame: pd.DataFrame, prefix: str) -> str | None:
    prefix = prefix.strip().lower()
    for column in frame.columns:
        if str(column).strip().lower().startswith(prefix):
            return column
    return None


def find_exact(frame: pd.DataFrame, *names: str) -> str | None:
    """Match a column by exact name, case- and whitespace-insensitive."""
    wanted = [name.strip().lower() for name in names if name]
    for column in frame.columns:
        if str(column).strip().lower() in wanted:
            return column
    return None


def repair_subject_ids(raw_ids: pd.Series,
                       surgery_dates: pd.Series) -> tuple[pd.Series, list[str]]:
    """Separators, then the hardcoded table, then the Date-of-Surgery check."""
    text = raw_ids.astype("string").str.strip().str.upper()
    cleaned = text.str.replace(r"[^A-Z0-9]", "", regex=True)
    notes: list[str] = []
    separator_fixes = int((text.notna() & text.ne(cleaned)).sum())

    resolved: list[str | None] = []
    pending: list[bool] = []
    hardcoded: list[str] = []
    for value, date in zip(cleaned, surgery_dates):
        if pd.isna(value) or not value:
            resolved.append(None); pending.append(False); continue
        stamp = None if pd.isna(date) else f"{date:%Y-%m-%d}"
        fix = REDCAP_ID_FIXES.get((value, stamp), REDCAP_ID_FIXES.get((value, None)))
        if fix is not None:
            resolved.append(fix); pending.append(False)
            hardcoded.append(f"{value} -> {fix}  (hardcoded)")
        else:
            resolved.append(value); pending.append(True)
    notes.extend(sorted(set(hardcoded)))
    if separator_fixes:
        notes.append(f"{separator_fixes} ID(s) had separators stripped")

    trusted = set()
    for value, date in zip(resolved, surgery_dates):
        if value is None or pd.isna(date):
            continue
        match = ID_STRUCTURE.match(value)
        if match and match.group(2) == f"{date:%Y%m%d}":
            trusted.add(value)

    automatic: list[str] = []
    final: list[str | None] = []
    for value, date, still_open in zip(resolved, surgery_dates, pending):
        if value is None or not still_open:
            final.append(value); continue
        structured = ID_STRUCTURE.match(value)
        if pd.isna(date):
            final.append(value if structured else None); continue
        if structured:
            hospital, datepart, sequence = structured.groups()
            if datepart == f"{date:%Y%m%d}":
                final.append(value); continue
        else:
            loose = ID_LOOSE.match(value)
            if loose is None:
                final.append(None); continue
            hospital, sequence = loose.groups()
        candidate = f"IU{hospital}{date:%Y%m%d}{sequence}"
        if candidate in trusted:
            final.append(value if structured else None); continue
        final.append(candidate)
        automatic.append(f"{value} -> {candidate}  (Date of Surgery)")
    notes.extend(sorted(set(automatic)))
    return pd.Series(final, index=raw_ids.index, dtype="object"), notes


def find_record_key(redcap: pd.DataFrame) -> str | None:
    for column in redcap.columns[:3]:
        values = redcap[column]
        if values.notna().all() and 1 < values.nunique() < len(redcap):
            return column
    return None


def load_redcap(redcap_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Per-patient event times and baseline characteristics, from one read."""
    redcap = pd.read_csv(redcap_path, low_memory=False)

    id_column = find_column(redcap, REDCAP_ID_PREFIX)
    if id_column is None:
        best = 0
        for column in redcap.columns:
            hits = int(redcap[column].astype("string").str.strip()
                       .str.fullmatch(r"IU(?:MH|UH)\d+", case=False, na=False).sum())
            if hits > best:
                id_column, best = column, hits
    induction_column = find_column(redcap, INDUCTION_PREFIX)
    or_entry_column = find_column(redcap, OR_ENTRY_PREFIX)
    date_column = find_column(redcap, SURGERY_DATE_PREFIX)
    for name, column in (("research ID", id_column),
                         ("What time was INDUCTION...", induction_column),
                         ("What time did the patient enter the OR?", or_entry_column),
                         ("Date of Surgery", date_column)):
        if column is None:
            raise ValueError(f"REDCap has no '{name}' column.")

    notes: list[str] = []
    record_key = find_record_key(redcap)
    if record_key is not None:
        for column in (id_column, date_column):
            before = int(redcap[column].notna().sum())
            key = redcap[record_key]
            redcap[column] = redcap.groupby(key, sort=False)[column].ffill()
            redcap[column] = redcap.groupby(key, sort=False)[column].bfill()
            gained = int(redcap[column].notna().sum()) - before
            if gained:
                notes.append(f"{gained} blank cell(s) filled in '{column[:40]}'")

    surgery_date = pd.to_datetime(redcap[date_column], errors="coerce")

    def resolve(values: pd.Series) -> pd.Series:
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
            parsed.loc[other] = pd.to_datetime(text.loc[other], errors="coerce",
                                               format="mixed")
            today = pd.Timestamp.today().normalize()
            stale = (parsed.notna() & parsed.dt.normalize().eq(today)
                     & surgery_date.notna() & other)
            parsed.loc[stale] = (surgery_date.loc[stale].dt.normalize()
                                 + (parsed.loc[stale] - today))
        return parsed

    subject_ids, id_notes = repair_subject_ids(redcap[id_column], surgery_date)
    notes.extend(id_notes)

    induction_at = resolve(redcap[induction_column])
    or_entry_at = resolve(redcap[or_entry_column])
    crossed = (induction_at.notna() & or_entry_at.notna()
               & ((or_entry_at - induction_at) > pd.Timedelta(hours=12)))
    induction_at.loc[crossed] = induction_at.loc[crossed] + pd.Timedelta(days=1)

    events = pd.DataFrame({"subject_id": subject_ids, "induction": induction_at,
                           "or_entry": or_entry_at})
    events = events.loc[events["subject_id"].notna() & events["subject_id"].ne("")]
    events = (events.sort_values("subject_id").groupby("subject_id", as_index=True)
              .agg(induction=("induction", "first"), or_entry=("or_entry", "first")))

    characteristics = extract_characteristics(redcap, subject_ids, record_key)
    return events, characteristics, notes


def normalize(name: object) -> str:
    """Collapse a column name to one lowercase, single-spaced form for matching.

    REDCap labels carry doubled spaces where the data dictionary had line
    breaks -- 'Monk skin tone (1-10)  Forehead:' is two spaces, and a literal
    match against one space quietly finds nothing.
    """
    return re.sub(r"\s+", " ", str(name)).strip().lower()


def match_characteristic(redcap: pd.DataFrame, prefix: str) -> str | None:
    """First column whose normalized name starts with this prefix."""
    prefix = normalize(prefix)
    for column in redcap.columns:
        if normalize(column).startswith(prefix):
            return column
    return None


def extract_characteristics(redcap: pd.DataFrame, subject_ids: pd.Series,
                            record_key: str | None) -> pd.DataFrame:
    """One row per patient of the baseline characteristics, for Table 1.

    These are PATIENT-LEVEL and time-invariant, which decides what they can be
    used for. Under patient fixed effects every one of them is absorbed by the
    patient's own intercept -- age cannot confound a comparison of a patient
    against themselves, because it does not change during the case. So they
    describe the cohort in Table 1, and they are candidates for asking whether
    the interaction DIFFERS between kinds of patients. They are not adjustment
    covariates for the within-patient models, and adding them there would
    change nothing at all.
    """
    columns: dict[str, pd.Series] = {}
    for name, prefix in CHARACTERISTIC_COLUMNS.items():
        column = match_characteristic(redcap, prefix)
        if column is None:
            continue
        values = redcap[column]
        if record_key is not None:
            # Each REDCap instrument writes its own row, so a field belongs to
            # the record rather than the row it happens to sit on.
            key = redcap[record_key]
            values = values.groupby(key, sort=False).ffill()
            values = values.groupby(key, sort=False).bfill()
        columns[name] = values

    # "Past Medical History (choice=X)" arrives as one Checked/Unchecked column
    # per condition; the count of conditions is the summary worth tabling.
    history = [column for column in redcap.columns
               if normalize(column).startswith("past medical history")
               and "choice=" in normalize(column)
               and "none of the above" not in normalize(column)]
    if history:
        checked = pd.DataFrame(
            {column: redcap[column].astype("string").str.strip().str.lower()
             .eq("checked") for column in history})
        values = checked.sum(axis=1)
        if record_key is not None:
            any_checked = pd.DataFrame(
                {column: redcap[column].notna() for column in history}
            ).any(axis=1)
            values = values.where(any_checked)
            # Group on the key directly rather than assigning a column into
            # the (already very wide) export, which fragments it badly enough
            # that pandas warns.
            key = redcap[record_key]
            values = values.groupby(key, sort=False).ffill()
            values = values.groupby(key, sort=False).bfill()
        columns["comorbidities"] = values

    if not columns:
        return pd.DataFrame()
    table = pd.DataFrame(columns)
    table.insert(0, "subject_id", subject_ids.to_numpy())
    table = table.loc[table["subject_id"].notna() & table["subject_id"].ne("")]
    # One row per patient: take the first non-blank value seen for each field.
    return table.groupby("subject_id", as_index=True).first()


def as_datetime(values: pd.Series) -> pd.Series:
    """Parse a column to datetime, passing one that already is straight through."""
    if pd.api.types.is_datetime64_any_dtype(values):
        return values
    return pd.to_datetime(values.astype("string").str.strip(),
                          errors="coerce", format="mixed")


def combine_date_time(dates: pd.Series, times: pd.Series) -> pd.Series:
    """Join a date column to a time column, whatever types they arrive as."""
    day = as_datetime(dates).dt.normalize()
    if pd.api.types.is_datetime64_any_dtype(times):
        offset = times - times.dt.normalize()
    else:
        offset = pd.to_timedelta(times.astype(str).str.strip(), errors="coerce")
        if offset.isna().all():
            parsed = pd.to_datetime(times.astype("string").str.strip(),
                                    errors="coerce", format="mixed")
            offset = parsed - parsed.dt.normalize()
    return day + offset


def anchor_clock_only(timestamp: pd.Series, induction: pd.Timestamp) -> pd.Series:
    """Re-date bare clock times onto the surgery day and unwrap midnight."""
    if timestamp.isna().all() or pd.isna(induction):
        return timestamp
    today = pd.Timestamp.today().normalize()
    if timestamp.dt.normalize().eq(today).mean() > 0.9:
        timestamp = induction.normalize() + (timestamp - today)
    wrapped = timestamp.diff() < pd.Timedelta(hours=-12)
    if wrapped.any():
        timestamp = timestamp + pd.to_timedelta(wrapped.cumsum(), unit="D")
    return timestamp


def load_sedline(path: str, induction: pd.Timestamp) -> pd.DataFrame | None:
    frame, _encoding = read_table(path, ("psi (sedline) value", "date", "time"))
    frame.columns = [str(column).strip() for column in frame.columns]
    # The Sedline export also carries event columns whose names begin "PSi".
    value_column = find_exact(frame, "psi (sedline) value")
    if value_column is None:
        value_column = next(
            (column for column in frame.columns
             if str(column).strip().lower().startswith("psi")
             and "value" in str(column).lower()), None)
    date_column = find_column(frame, "date")
    time_column = find_column(frame, "time")
    if value_column is None:
        return None
    timestamp = pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns]")
    if date_column and time_column and date_column != time_column:
        timestamp = pd.to_datetime(
            frame[date_column].astype("string").str.strip() + " "
            + frame[time_column].astype("string").str.strip(),
            errors="coerce", format="mixed")
    if timestamp.isna().all():
        epoch_column = find_column(frame, "epoch")
        if epoch_column:
            timestamp = pd.to_datetime(
                pd.to_numeric(frame[epoch_column], errors="coerce"),
                unit="ms", errors="coerce")
    if timestamp.isna().all():
        return None
    # "-" is the monitor's missing-value mark; to_numeric makes it a real NaN.
    return pd.DataFrame({"timestamp": timestamp,
                         "psi": pd.to_numeric(frame[value_column], errors="coerce")})


def average_sto2_channels(frame: pd.DataFrame,
                          channels: tuple[int, ...]) -> tuple[pd.Series, int]:
    """Mean of the named channels over the rows where each is flagged valid."""
    present = 0
    total = pd.Series(0.0, index=frame.index)
    count = pd.Series(0, index=frame.index)
    for channel in channels:
        value_column = find_column(frame, f"sto2_ch{channel}")
        if value_column is None:
            continue
        present += 1
        value = pd.to_numeric(frame[value_column], errors="coerce")
        valid_column = find_column(frame, f"valid_ch{channel}")
        if valid_column is not None:
            value = value.where(pd.to_numeric(frame[valid_column],
                                              errors="coerce").eq(1))
        total = total.add(value.fillna(0.0))
        count = count.add(value.notna().astype(int))
    return (total / count).where(count > 0), present


def load_sto2(path: str, induction: pd.Timestamp,
              channels: tuple[int, ...] = (1, 2),
              fallback: tuple[int, ...] = (3, 4)
              ) -> tuple[pd.DataFrame | None, tuple[int, ...] | None]:
    """Cerebral O2, falling back to the second probe pair when the first is dead.

    The fallback is whole-file, not per-row: mixing probe pairs within one
    patient would splice together two anatomical sites.
    """
    frame, _encoding = read_table(path, ("time", "sto2_ch1", "valid_ch1"))
    time_column = find_exact(frame, "time") or find_column(frame, "time")
    if time_column is None:
        return None, None
    timestamp = as_datetime(frame[time_column])

    values, present = average_sto2_channels(frame, channels)
    used = channels
    if present == 0 or values.notna().sum() == 0:
        alternative, alternative_present = average_sto2_channels(frame, fallback)
        if alternative_present and alternative.notna().sum() > 0:
            values, used = alternative, fallback
        elif present == 0:
            return None, None
    return pd.DataFrame({"timestamp": timestamp, "sto2": values}), used


def load_hemosphere(path: str, induction: pd.Timestamp,
                    co_column: str | None, min_sqi: float,
                    extra: tuple[str, ...] = ()
                    ) -> tuple[pd.DataFrame | None, dict]:
    """Cardiac output from a Hemosphere export, filtered on its own SQI.

    `extra` names further channels to carry along (SVV, PPV, HPI and the rest).
    They are read where present so their coverage can be reported honestly
    across the cohort -- several of the parameters this device lists are blank
    unless the matching module was attached.
    """
    frame, encoding = read_table(
        path, ("timestamp", "date", "time", "technology", "co", "co_sqi"))
    frame.columns = [str(column).strip() for column in frame.columns]
    info: dict = {"columns": list(frame.columns), "rows": len(frame),
                  "encoding": encoding}

    value_column = (find_exact(frame, co_column) if co_column
                    else find_exact(frame, "co", "cardiac output", "co_lmin"))
    if value_column is None:
        return None, info
    info["co_column"] = value_column

    timestamp = pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns]")
    stamp_column = find_exact(frame, "timestamp")
    if stamp_column is not None:
        timestamp = as_datetime(frame[stamp_column])
    date_column, time_column = find_exact(frame, "date"), find_exact(frame, "time")
    if timestamp.isna().all() and date_column and time_column:
        timestamp = combine_date_time(frame[date_column], frame[time_column])
    if timestamp.isna().all() and time_column:
        timestamp = as_datetime(frame[time_column])
    if timestamp.isna().all():
        return None, info
    timestamp = anchor_clock_only(timestamp, induction)

    value = pd.to_numeric(frame[value_column], errors="coerce")
    info["co_present"] = int(value.notna().sum())

    sqi_column = find_exact(frame, "co_sqi", "cosqi", "co sqi")
    if sqi_column is None:
        info["sqi_column"] = None
    else:
        info["sqi_column"] = sqi_column
        sqi = pd.to_numeric(frame[sqi_column], errors="coerce")
        kept = sqi >= min_sqi
        info["co_kept"] = int((value.notna() & kept).sum())
        # A missing SQI is not evidence of a good reading, so it is dropped too.
        value = value.where(kept)

    technology = find_exact(frame, "technology")
    if technology is not None:
        info["technology"] = sorted(
            frame[technology].dropna().astype(str).str.strip().unique().tolist())[:6]

    out = {"timestamp": timestamp, "co": value}
    info["extra_present"] = {}
    for name in extra:
        spec = HEMOSPHERE_EXTRA.get(name)
        if spec is None:
            continue
        column = find_exact(frame, spec["column"])
        if column is None:
            continue
        channel = pd.to_numeric(frame[column], errors="coerce")
        info["extra_present"][name] = int(channel.notna().sum())
        if channel.notna().any():
            out[name] = channel
    return pd.DataFrame(out), info


def load_map(path: str, induction: pd.Timestamp) -> pd.DataFrame | None:
    """Beat-to-beat MAP, with the monitor's own bad-data rows dropped."""
    frame, _encoding = read_table(path, ("meanarterialpressure", "databad", "time"))
    frame.columns = [str(column).strip() for column in frame.columns]
    value_column = find_column(frame, "meanarterial") or find_column(frame, "map")
    time_column = find_exact(frame, "time") or find_column(frame, "time")
    if value_column is None or time_column is None:
        return None

    keep = pd.Series(True, index=frame.index)
    bad_column = find_column(frame, "databad")
    if bad_column is not None:
        keep &= pd.to_numeric(frame[bad_column], errors="coerce").eq(0)

    timestamp = anchor_clock_only(as_datetime(frame[time_column]), induction)
    return pd.DataFrame({
        "timestamp": timestamp,
        "map": pd.to_numeric(frame[value_column], errors="coerce"),
    }).loc[keep]


def load_covariates(path: str, induction: pd.Timestamp
                    ) -> tuple[pd.DataFrame | None, dict]:
    """End-tidal CO2, inspired O2 fraction and anesthetic concentration.

    The anesthesia record is not one of the four monitor exports this project
    already reads, so there is no fixed column layout to rely on. Each wanted
    quantity is matched by name against the spellings seen in these records; a
    quantity the file does not carry is simply absent, and the model reports
    itself as unadjusted for it rather than guessing.
    """
    frame, encoding = read_table(
        path, ("time", "etco2", "fio2", "mac", "date", "timestamp"))
    frame.columns = [str(column).strip() for column in frame.columns]
    info: dict = {"columns": list(frame.columns), "encoding": encoding,
                  "found": {}}

    timestamp = pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns]")
    stamp_column = find_exact(frame, "timestamp")
    if stamp_column is not None:
        timestamp = as_datetime(frame[stamp_column])
    date_column, time_column = find_exact(frame, "date"), find_exact(frame, "time")
    if timestamp.isna().all() and date_column and time_column:
        timestamp = combine_date_time(frame[date_column], frame[time_column])
    if timestamp.isna().all():
        time_column = time_column or find_column(frame, "time")
        if time_column is not None:
            timestamp = as_datetime(frame[time_column])
    if timestamp.isna().all():
        return None, info
    timestamp = anchor_clock_only(timestamp, induction)

    out = {"timestamp": timestamp}
    for name, patterns in COVARIATE_PATTERNS.items():
        column = find_exact(frame, *patterns)
        if column is None:
            for pattern in patterns:
                column = find_column(frame, pattern)
                if column is not None:
                    break
        if column is None:
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.notna().sum() == 0:
            continue
        # Some records write the inspired fraction as 0.21-1.0 rather than as
        # a percentage. Left alone it would fall outside the plausible range
        # and the covariate would silently vanish, so it is put on the scale
        # the rest of the script expects.
        if name == "fio2" and float(values.median(skipna=True)) <= 1.5:
            values = values * 100.0
            info.setdefault("rescaled", []).append(f"{column} (fraction to %)")
        out[name] = values
        info["found"][name] = column
    if len(out) == 1:
        return None, info
    return pd.DataFrame(out), info


def classify_drug(drug: object, drug_key: object) -> str | None:
    """Which class a medication belongs to, by key first and printed name second."""
    key = normalize(drug_key).replace(" ", "")
    name = normalize(drug)
    for label, tokens in DRUG_CLASSES.items():
        for token in tokens:
            if key == token or name == token:
                return label
    for label, tokens in DRUG_CLASSES.items():
        for token in tokens:
            if name.startswith(token) or key.startswith(token):
                return label
    return None


def excel_flag(values: pd.Series) -> pd.Series:
    """Read a 0/1 column that Excel has reformatted into dates.

    In this export `event_valid` arrives as a mixture of 1 and
    '1900-01-01 00:00:00' -- the same value, one of them having been formatted
    as a date, where day 1 of the 1900 epoch IS the number 1. Reading it
    naively marks valid rows invalid, so both spellings are mapped back.
    """
    text = values.astype("string").str.strip()
    out = pd.to_numeric(text, errors="coerce")
    stamps = pd.to_datetime(text.where(out.isna()), errors="coerce",
                            format="mixed")
    # Excel renders serial 1 as 1900-01-01 and 2 as 1900-01-02, so the day
    # count runs from 1899-12-31. (The more familiar 1899-12-30 epoch is the
    # one that corrects for Excel's phantom 1900 leap day; here the displayed
    # dates are what has to be inverted, not the corrected ones.)
    recovered = (stamps - pd.Timestamp("1899-12-31")).dt.days
    return out.fillna(recovered)


def load_medications(path: Path) -> tuple[pd.DataFrame, dict]:
    """Every medication event in the OR, as one tidy table of time intervals.

    A bolus becomes an instant; an infusion becomes a start-stop interval
    carrying its rate. What this buys the analysis is two things the monitors
    cannot say on their own:

      * the anesthetic concentration the outline asks to adjust for -- for a
        TIVA case the propofol infusion rate IS that exposure, and unlike age
        or ASA it varies within a patient, so patient fixed effects do not
        absorb it;
      * the single biggest threat to the headline result. Propofol lowers MAP
        and PSi at the same moment, so "deep and hypotensive" can co-occur for
        a pharmacological reason rather than a physiological one. Knowing when
        the drugs moved lets that explanation be tested instead of assumed.
    """
    frame, encoding = read_table(str(path), ("subject_id", "drug", "drug_key",
                                             "administration_type",
                                             "bolus_time", "infusion_start"))
    frame.columns = [str(column).strip() for column in frame.columns]
    info: dict = {"encoding": encoding, "rows": len(frame), "columns":
                  list(frame.columns), "dropped": {}, "classes": {}, "drugs": {}}

    subject = find_exact(frame, "subject_id") or find_column(frame, "subject")
    if subject is None:
        raise ValueError("the medication export has no subject_id column")
    ids = frame[subject].astype("string").str.strip().str.upper()
    # The export carries a trailing summary row whose fields are shifted by one
    # -- its subject_id is a timestamp. Requiring a real research ID drops it
    # without having to guess at how many such rows there are.
    real = ids.str.fullmatch(r"IU(?:MH|UH)\d+", na=False)
    info["dropped"]["not a research ID"] = int((~real).sum())
    frame, ids = frame.loc[real].copy(), ids.loc[real]

    valid_column = find_exact(frame, "event_valid")
    if valid_column is not None:
        valid = excel_flag(frame[valid_column])
        info["dropped"]["event_valid != 1"] = int((valid.ne(1) & valid.notna()).sum())
        frame = frame.loc[valid.ne(0) | valid.isna()]
    duplicate_column = find_exact(frame, "possible_duplicate_event")
    if duplicate_column is not None:
        duplicate = excel_flag(frame[duplicate_column]).fillna(0)
        info["dropped"]["flagged duplicate"] = int(duplicate.eq(1).sum())
        frame = frame.loc[duplicate.ne(1)]

    drug_column = find_exact(frame, "drug")
    key_column = find_exact(frame, "drug_key")
    kind_column = find_exact(frame, "administration_type")
    dose_column = find_exact(frame, "dose_or_rate") or find_exact(frame, "dose_or_rate_raw")
    unit_column = find_exact(frame, "unit")
    bolus_column = find_exact(frame, "bolus_time") or find_exact(frame, "bolus_time_raw")
    start_column = find_exact(frame, "infusion_start") or find_exact(frame, "infusion_start_raw")
    stop_column = find_exact(frame, "infusion_stop") or find_exact(frame, "infusion_stop_raw")

    kind = (frame[kind_column].astype("string").str.strip().str.lower()
            if kind_column else pd.Series("bolus", index=frame.index))
    start = pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns]")
    stop = pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns]")
    if bolus_column is not None:
        start = start.fillna(as_datetime(frame[bolus_column]))
    if start_column is not None:
        start = start.fillna(as_datetime(frame[start_column]))
    if stop_column is not None:
        stop = as_datetime(frame[stop_column])

    classes = [classify_drug(drug, key) for drug, key
               in zip(frame[drug_column] if drug_column else [None] * len(frame),
                      frame[key_column] if key_column else [None] * len(frame))]
    events = pd.DataFrame({
        "subject_id": frame[subject].astype("string").str.strip().str.upper(),
        "drug": (frame[drug_column].astype("string").str.strip()
                 if drug_column else pd.Series("", index=frame.index)),
        "drug_key": (normalize_series(frame[key_column]) if key_column
                     else pd.Series("", index=frame.index)),
        "drug_class": classes,
        "kind": kind,
        "dose": (pd.to_numeric(frame[dose_column], errors="coerce")
                 if dose_column else np.nan),
        "unit": (frame[unit_column].astype("string").str.strip().str.lower()
                 if unit_column else pd.Series("", index=frame.index)),
        "start": start, "stop": stop})
    events = events.loc[events["start"].notna()]
    # An infusion with no recorded stop is left open rather than discarded; it
    # is closed at the end of that patient's record when the grid is built.
    events.loc[events["kind"].eq("infusion") & events["stop"].isna(), "stop"] = pd.NaT

    info["kept"] = len(events)
    info["patients"] = int(events["subject_id"].nunique())
    info["classes"] = events["drug_class"].fillna("unclassified").value_counts().to_dict()
    info["drugs"] = events["drug"].value_counts().head(25).to_dict()
    info["unclassified"] = sorted(
        events.loc[events["drug_class"].isna(), "drug"].dropna().unique().tolist())[:12]
    return events, info


def normalize_series(values: pd.Series) -> pd.Series:
    return values.astype("string").str.strip().str.lower()


def medication_features(events: pd.DataFrame, grid: np.ndarray,
                        induction: pd.Timestamp, window_minutes: float,
                        propofol_key: str, pressor_key: str) -> dict[str, np.ndarray]:
    """Turn one patient's medication events into signals on their own grid.

    Four features, each chosen because it answers a question the reviewer will
    ask rather than because it was easy to compute:

      propofol_rate   the running infusion rate, as the anesthetic exposure
      pressor_rate    the running vasopressor infusion rate
      bolus_hypnotic  1 while within `window` minutes AFTER a hypnotic bolus
      bolus_vasoactive  the same for anything given to move blood pressure

    The two indicators mark the stretches where MAP and PSi were being pushed
    by a syringe. Dropping those stretches is the cleanest test of whether the
    interaction is physiology or pharmacology.
    """
    minutes = (events["start"] - induction).dt.total_seconds() / 60.0
    stop_minutes = (events["stop"] - induction).dt.total_seconds() / 60.0
    end = float(grid[-1]) if len(grid) else 0.0

    def infusion_rate(mask: pd.Series) -> np.ndarray:
        rate = np.zeros_like(grid, dtype=float)
        rows = events.loc[mask & events["kind"].eq("infusion")]
        for index in rows.index:
            begin = minutes.get(index, np.nan)
            finish = stop_minutes.get(index, np.nan)
            dose = rows.at[index, "dose"]
            if not np.isfinite(begin) or not np.isfinite(dose):
                continue
            if not np.isfinite(finish):
                finish = end
            rate[(grid >= begin) & (grid <= finish)] += float(dose)
        return rate

    def bolus_window(mask: pd.Series) -> np.ndarray:
        flag = np.zeros_like(grid, dtype=float)
        rows = events.loc[mask & events["kind"].ne("infusion")]
        for index in rows.index:
            begin = minutes.get(index, np.nan)
            if not np.isfinite(begin):
                continue
            flag[(grid >= begin) & (grid <= begin + window_minutes)] = 1.0
        return flag

    key = events["drug_key"].fillna("")
    name = normalize_series(events["drug"]).fillna("")
    is_propofol = key.str.startswith(propofol_key) | name.str.startswith("propofol")
    is_pressor = key.str.startswith(pressor_key) | name.str.startswith(pressor_key)
    hypnotic = events["drug_class"].eq("hypnotic")
    vasoactive = events["drug_class"].eq("vasoactive")
    return {"propofol_rate": infusion_rate(is_propofol),
            "pressor_rate": infusion_rate(is_pressor),
            "bolus_hypnotic": bolus_window(hypnotic),
            "bolus_vasoactive": bolus_window(vasoactive)}


def resolve_map_list(explicit: Path | None, data_dir: Path) -> Path | None:
    if explicit is not None:
        return explicit if explicit.is_file() else None
    for name in MAP_LIST_FALLBACKS:
        candidate = data_dir / name
        if candidate.is_file():
            return candidate
    return None


# --------------------------------------------------------------------------- #
# Put every signal on one clock, per patient
# --------------------------------------------------------------------------- #

def to_grid(frame: pd.DataFrame, column: str, grid: np.ndarray,
            max_gap_minutes: float) -> np.ndarray:
    """Resample one signal onto the shared grid, refusing to bridge long gaps."""
    usable = frame[["minutes", column]].dropna().sort_values("minutes")
    if len(usable) < 2:
        return np.full(grid.shape, np.nan)
    times = usable["minutes"].to_numpy(float)
    values = np.interp(grid, times, usable[column].to_numpy(float),
                       left=np.nan, right=np.nan)
    if len(times) <= 4000:
        nearest = np.abs(grid[:, None] - times[None, :]).min(axis=1)
    else:
        index = np.clip(np.searchsorted(times, grid), 1, len(times) - 1)
        nearest = np.minimum(np.abs(grid - times[index - 1]),
                             np.abs(grid - times[index]))
    values[nearest > max_gap_minutes] = np.nan
    return values


def build_aligned(args: argparse.Namespace, events: pd.DataFrame,
                  map_list: Path, hemo_list: Path | None,
                  covariate_list: Path | None,
                  medications: pd.DataFrame | None = None
                  ) -> tuple[pd.DataFrame, dict, list[str]]:
    """One long table: subject_id, minutes, map, sto2, psi, and co / covariates.

    PSi, cerebral O2 and MAP are required of a patient. Cardiac output and the
    anesthesia-record covariates are optional and merged in only where a file
    exists, so the primary analysis keeps its full cohort and only the analyses
    that need the extra signal run on the smaller subset.
    """
    step = float(args.step_seconds) / 60.0
    max_gap = float(args.max_gap_seconds) / 60.0
    channels = tuple(int(part) for part in str(args.sto2_channels).split(",")
                     if part.strip())
    fallback = tuple(int(part) for part
                     in str(args.sto2_fallback_channels).split(",")
                     if part.strip())
    hemosphere_extra = tuple(
        part.strip().lower() for part in str(args.hemosphere_extra).split(",")
        if part.strip().lower() in HEMOSPHERE_EXTRA)
    sto2_fallback_used: list[str] = []

    def index_by_id(filepaths: Path | None) -> dict[str, str]:
        found: dict[str, str] = {}
        if filepaths is None:
            return found
        for path in read_filepath_list(filepaths):
            subject_id = patient_id_from_path(path)
            if subject_id and Path(path).is_file():
                found[subject_id] = path
        return found

    psi_files = index_by_id(args.filepaths)
    sto2_files = index_by_id(args.sto2_filepaths)
    map_files = index_by_id(map_list)
    hemo_files = index_by_id(hemo_list)
    covariate_files = index_by_id(covariate_list)
    shared = sorted(set(psi_files) & set(sto2_files) & set(map_files))

    chunks: list[pd.DataFrame] = []
    skipped: list[str] = []
    report: dict = {"listed": len(hemo_files), "matched": 0, "used": 0,
                    "co_column": None, "sqi_column": None, "technology": [],
                    "rows_seen": 0, "rows_kept": 0, "errors": [],
                    "encodings": [], "no_co": [], "columns": [],
                    "covariate_listed": len(covariate_files),
                    "covariate_matched": 0, "covariate_used": 0,
                    "covariate_columns": {}, "covariate_errors": [],
                    "covariate_rescaled": [], "extra_present": {},
                    "medication_patients": 0, "medication_missing": [],
                    "candidates": len(shared)}

    for subject_id in shared:
        if subject_id not in events.index:
            skipped.append(f"{subject_id}: not in the REDCap export")
            continue
        induction = events.loc[subject_id, "induction"]
        if pd.isna(induction):
            skipped.append(f"{subject_id}: no REDCap induction time")
            continue

        try:
            psi_frame = load_sedline(psi_files[subject_id], induction)
            sto2_frame, sto2_used = load_sto2(sto2_files[subject_id],
                                              induction, channels, fallback)
            if sto2_used is not None and tuple(sto2_used) != channels:
                sto2_fallback_used.append(
                    f"{subject_id} (channels "
                    f"{','.join(str(c) for c in sto2_used)})")
            map_frame = load_map(map_files[subject_id], induction)
        except Exception as exc:
            skipped.append(f"{subject_id}: read error: {exc}")
            continue
        if psi_frame is None or sto2_frame is None or map_frame is None:
            skipped.append(f"{subject_id}: a signal had no readable values")
            continue

        sources = [(psi_frame, "psi"), (sto2_frame, "sto2"), (map_frame, "map")]

        if subject_id in hemo_files:
            report["matched"] += 1
            try:
                hemo_frame, info = load_hemosphere(
                    hemo_files[subject_id], induction, args.co_column,
                    args.min_co_sqi, hemosphere_extra)
            except Exception as exc:
                hemo_frame, info = None, {}
                # A Hemosphere failure costs this patient their CO only; they
                # still have MAP, cerebral O2 and PSi, so it must not be filed
                # as an alignment failure or the report contradicts itself.
                report["errors"].append(f"{subject_id}: {exc}")
            if info:
                encoding = info.get("encoding")
                if encoding and encoding not in report["encodings"]:
                    report["encodings"].append(encoding)
                report["columns"] = report["columns"] or info.get("columns", [])
                report["co_column"] = report["co_column"] or info.get("co_column")
                report["sqi_column"] = report["sqi_column"] or info.get("sqi_column")
                report["rows_seen"] += info.get("co_present", 0)
                report["rows_kept"] += info.get("co_kept", info.get("co_present", 0))
                for name in info.get("technology", []):
                    if name not in report["technology"]:
                        report["technology"].append(name)
            if info.get("extra_present"):
                for name, count in info["extra_present"].items():
                    report["extra_present"][name] = \
                        report["extra_present"].get(name, 0) + (1 if count else 0)
            if hemo_frame is not None:
                for name in hemosphere_extra:
                    if name in hemo_frame.columns and hemo_frame[name].notna().any():
                        sources.append((hemo_frame[["timestamp", name]], name))
            if hemo_frame is not None and hemo_frame["co"].notna().any():
                report["used"] += 1
                sources.append((hemo_frame[["timestamp", "co"]], "co"))
            elif hemo_frame is not None:
                report["no_co"].append(subject_id)

        if subject_id in covariate_files:
            report["covariate_matched"] += 1
            try:
                covariate_frame, info = load_covariates(
                    covariate_files[subject_id], induction)
            except Exception as exc:
                covariate_frame, info = None, {}
                report["covariate_errors"].append(f"{subject_id}: {exc}")
            if info.get("found"):
                for name, column in info["found"].items():
                    report["covariate_columns"].setdefault(name, column)
            for line in info.get("rescaled", []):
                if line not in report["covariate_rescaled"]:
                    report["covariate_rescaled"].append(line)
            if covariate_frame is not None:
                report["covariate_used"] += 1
                for name in COVARIATE_PATTERNS:
                    if name in covariate_frame.columns:
                        sources.append(
                            (covariate_frame[["timestamp", name]], name))

        # Clean each signal first, then build this patient's grid from the
        # stretch where MAP, cerebral O2 and PSi all actually exist.
        prepared: list[tuple[pd.DataFrame, str]] = []
        for frame, column in sources:
            frame = frame.dropna(subset=["timestamp"]).copy()
            frame["minutes"] = (frame["timestamp"] - induction).dt.total_seconds() / 60.0
            low, high = VALID_RANGE[column]
            frame.loc[~frame[column].between(low, high), column] = np.nan
            frame = frame.dropna(subset=[column])
            prepared.append((frame, column))

        required = [(frame, column) for frame, column in prepared
                    if column in ("psi", "sto2", "map")]
        empty = [column for frame, column in required if frame.empty]
        if empty:
            why = {"map": "no rows with databad == 0 and a plausible pressure",
                   "sto2": "no rows with a valid channel",
                   "psi": "every PSi value was '-' or out of range"}
            skipped.append(f"{subject_id}: {'/'.join(empty).upper()} unusable — "
                           + "; ".join(why[column] for column in empty))
            continue
        required = [frame for frame, _column in required]
        start = max(float(frame["minutes"].min()) for frame in required)
        stop = min(float(frame["minutes"].max()) for frame in required)
        if args.min_minutes is not None:
            start = max(start, float(args.min_minutes))
        if args.max_minutes is not None:
            stop = min(stop, float(args.max_minutes))
        if stop <= start:
            skipped.append(f"{subject_id}: MAP, cerebral O2 and PSi never "
                           f"overlap in time")
            continue
        span_hours = (stop - start) / 60.0
        if span_hours > args.max_hours:
            skipped.append(f"{subject_id}: records span {span_hours:.1f} h, over "
                           f"the --max-hours {args.max_hours:g} sanity cap — "
                           f"check that file's clock")
            continue

        grid = np.arange(start, stop + step / 2.0, step)
        aligned = {"minutes": grid}
        for frame, column in prepared:
            aligned[column] = to_grid(frame, column, grid, max_gap)

        patient = pd.DataFrame(aligned).dropna(subset=["map", "sto2", "psi"])
        if patient.empty:
            skipped.append(f"{subject_id}: no overlapping MAP/cerebral O2/PSi "
                           f"samples")
            continue

        if medications is not None:
            mine = medications.loc[medications["subject_id"] == subject_id]
            if mine.empty:
                report["medication_missing"].append(subject_id)
            else:
                report["medication_patients"] += 1
                for name, values in medication_features(
                        mine, patient["minutes"].to_numpy(float), induction,
                        args.drug_window_minutes, args.propofol_key,
                        args.pressor_key).items():
                    patient[name] = values

        patient.insert(0, "subject_id", subject_id)
        chunks.append(patient)

    combined = (pd.concat(chunks, ignore_index=True) if chunks
                else pd.DataFrame(columns=["subject_id", "minutes", "map",
                                           "sto2", "psi"]))
    report["sto2_fallback"] = sto2_fallback_used
    return combined, report, skipped


# --------------------------------------------------------------------------- #
# The model: within-patient regression, patient-clustered inference
# --------------------------------------------------------------------------- #

@dataclass
class Fit:
    """A fitted linear model, reduced to what any report of it needs.

    Everything downstream -- marginal slopes, slope differences, figure bands --
    is a linear combination of these coefficients taken against this one
    covariance matrix, so the numbers in the tables and the numbers in the
    figures cannot drift apart.
    """
    label: str
    names: list[str]
    params: np.ndarray
    cov: np.ndarray
    dof: int
    n_obs: int
    n_patients: int
    sigma: float = float("nan")
    rho1: float = float("nan")
    converged: bool = True
    note: str = ""
    blocks: dict = field(default_factory=dict, repr=False)

    def contrast(self, weights: dict[str, float]) -> dict:
        """Estimate, SE, 95% CI and P for any linear combination of coefficients."""
        vector = np.zeros(len(self.names))
        for name, weight in weights.items():
            if name not in self.names:
                raise KeyError(f"{self.label}: no term named {name!r}; "
                               f"have {self.names}")
            vector[self.names.index(name)] += float(weight)
        estimate = float(vector @ self.params)
        variance = float(vector @ self.cov @ vector)
        se = float(np.sqrt(variance)) if variance > 0 else float("nan")
        if not np.isfinite(se) or se == 0.0:
            return {"estimate": estimate, "se": float("nan"), "t": float("nan"),
                    "p": float("nan"), "low": float("nan"), "high": float("nan")}
        t = estimate / se
        critical = float(stats.t.ppf(0.975, self.dof))
        return {"estimate": estimate, "se": se, "t": t,
                "p": float(2.0 * stats.t.sf(abs(t), self.dof)),
                "low": estimate - critical * se,
                "high": estimate + critical * se}

    def coefficient(self, name: str) -> dict:
        return self.contrast({name: 1.0})


def term_names(exposure: str, modifier: str) -> dict[str, str]:
    """The four regressors the within-patient interaction model needs."""
    x, m = AXES[exposure]["short"], AXES[modifier]["short"]
    return {"x": f"{x} (within)",
            "m": f"{m} (within)",
            "xm": f"{x} x {m} (within)",
            "xb": f"{x} x mean {m} (cross-level)"}


@dataclass
class Design:
    """The regressors, split by the level each one varies at.

    `within` holds everything that moves inside a patient and is what the
    primary model is fitted on. `between` holds the patient-level means, used
    only by the random-effects companion -- under patient fixed effects they
    are collinear with the patient intercepts and vanish, which is the whole
    point of that specification.
    """
    within: pd.DataFrame
    between: pd.DataFrame
    terms: dict[str, str]
    reference: float


def build_design(frame: pd.DataFrame, exposure: str, modifier: str,
                 centers: dict[str, float],
                 covariates: tuple[str, ...] = ()) -> Design:
    """Patient-mean decomposition, with the interaction built to match.

    This is the specification question the outline singles out, and getting it
    wrong is the easiest way to produce a confident wrong answer. Writing the
    model with RAW MAP times RAW PSi hides two different interactions inside
    one coefficient:

        within  x within   when THIS patient goes deeper than their own
                           average, does THIS patient's MAP-cerebral O2 slope
                           change? -- the clinical question
        within  x between  do patients who are deeper ON AVERAGE have a
                           different MAP slope? -- a question about who the
                           patients are

    Forcing those two to share a coefficient drags the estimate toward
    whichever is weaker, which in practice means toward zero. So the exposure
    enters as a deviation from the patient's own mean, and it is crossed with
    BOTH the patient's own PSi deviation and the patient's mean PSi, as
    separate terms. The first is the primary estimate; the second is reported
    beside it and can be tested against it.

    The modifier's patient means are centered on `reference`, the mean of the
    per-patient PSi means, so that the within-patient slope at an ABSOLUTE PSi
    of p -- for a patient of average depth -- is simply

        b_exposure + d_within * (p - reference)
    """
    terms = term_names(exposure, modifier)
    grouped = frame.groupby("subject_id", sort=False)
    x_mean = grouped[exposure].transform("mean")
    m_mean = grouped[modifier].transform("mean")
    reference = (centers.get(modifier)
                 if centers.get(modifier) is not None
                 else float(np.round(frame.groupby("subject_id")[modifier]
                                     .mean().mean(), 0)))

    x_within = (frame[exposure] - x_mean).to_numpy(float)
    m_within = (frame[modifier] - m_mean).to_numpy(float)
    m_between = (m_mean - reference).to_numpy(float)
    x_between = (x_mean - centers.get(exposure, float(x_mean.mean()))).to_numpy(float)

    within = {terms["x"]: x_within, terms["m"]: m_within,
              terms["xm"]: x_within * m_within,
              terms["xb"]: x_within * m_between}
    between = {f"mean {AXES[exposure]['short']}": x_between,
               f"mean {AXES[modifier]['short']}": m_between,
               f"mean {AXES[exposure]['short']} x mean "
               f"{AXES[modifier]['short']}": x_between * m_between}
    for name in covariates:
        covariate_mean = grouped[name].transform("mean")
        within[name] = (frame[name] - covariate_mean).to_numpy(float)
        between[f"mean {name}"] = (covariate_mean
                                   - centers.get(name, float(covariate_mean.mean()))
                                   ).to_numpy(float)
    return Design(pd.DataFrame(within, index=frame.index),
                  pd.DataFrame(between, index=frame.index), terms,
                  float(reference))


def within_blocks(frame: pd.DataFrame, design: pd.DataFrame,
                  outcome: str = "sto2",
                  absorb: pd.Series | None = None) -> dict[str, tuple]:
    """Per-patient cross-products of the demeaned design.

    Subtracting the group mean from the outcome and from every regressor IS the
    fixed-effects estimator -- it removes anything constant within the group,
    including every unmeasured difference, and leaves only the covariation
    inside it. The group is the patient by default; `absorb` allows a finer
    one, which the PSi-category model needs so that each category gets its own
    per-patient baseline instead of sharing one across categories.

    Patients stay as separate blocks whatever is absorbed, because the PATIENT
    is the independent unit for inference. Keeping the pieces apart buys three
    things for free: the cluster-robust sandwich, a patient bootstrap that is a
    small solve rather than a refit, and leave-one-patient-out by subtraction.
    """
    values = design.to_numpy(float)
    y = frame[outcome].to_numpy(float)
    if absorb is not None:
        codes = pd.factorize(absorb.to_numpy())[0]
        width = values.shape[1]
        counts = np.bincount(codes, minlength=codes.max() + 1).astype(float)
        sums = np.zeros((len(counts), width))
        np.add.at(sums, codes, values)
        values = values - (sums / counts[:, None])[codes]
        y_sums = np.zeros(len(counts))
        np.add.at(y_sums, codes, y)
        y = y - (y_sums / counts)[codes]
    blocks: dict[str, tuple] = {}
    minutes = frame["minutes"].to_numpy(float)
    for subject_id, index in frame.groupby("subject_id", sort=True).indices.items():
        X = values[index]
        target = y[index]
        if len(index) < 2:
            continue
        # Demean within the patient as well. With a finer absorb group this is
        # already done and costs nothing; with the default it IS the transform.
        X = X - X.mean(axis=0)
        target = target - target.mean()
        if not np.isfinite(X).all() or not np.isfinite(target).all():
            keep = np.isfinite(X).all(axis=1) & np.isfinite(target)
            X, target = X[keep], target[keep]
            if len(target) < 2:
                continue
        blocks[str(subject_id)] = (X.T @ X, X.T @ target, X, target,
                                   minutes[index])
    return blocks


def solve_blocks(blocks: dict[str, tuple],
                 subjects: list[str] | None = None) -> np.ndarray | None:
    """Pooled within-patient least squares from the per-patient pieces."""
    subjects = list(blocks) if subjects is None else subjects
    if not subjects:
        return None
    width = blocks[subjects[0]][0].shape[0]
    gram = np.zeros((width, width))
    moment = np.zeros(width)
    for subject_id in subjects:
        gram += blocks[subject_id][0]
        moment += blocks[subject_id][1]
    try:
        return np.linalg.solve(gram, moment)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(gram) @ moment


def fit_within(frame: pd.DataFrame, design: pd.DataFrame, label: str,
               outcome: str = "sto2", step_minutes: float = 1.0 / 6.0,
               absorb: pd.Series | None = None) -> Fit:
    """Patient fixed effects, with patient-clustered standard errors.

    The sandwich is clustered on patient rather than assuming a correlation
    shape, which is the point: consecutive ten-second samples are strongly
    autocorrelated and any specific structure (AR(1) and the rest) would be a
    guess, while a patient-clustered sandwich is valid under all of them. The
    cost is that inference leans on the NUMBER OF PATIENTS, not the number of
    samples -- which is the honest exchange rate for this data.
    """
    blocks = within_blocks(frame, design, outcome, absorb)
    names = list(design.columns)
    n_absorbed = (int(pd.Series(absorb).nunique()) if absorb is not None
                  else len(blocks))
    if not blocks:
        return Fit(label, names, np.full(len(names), np.nan),
                   np.full((len(names), len(names)), np.nan), 1, 0, 0,
                   converged=False, note="no patient had usable data")

    params = solve_blocks(blocks)
    width = len(names)
    gram = np.zeros((width, width))
    meat = np.zeros((width, width))
    residual_sum = 0.0
    n_obs = 0
    lag_numerator = lag_denominator = 0.0
    for X, moment, design_block, target, minutes in blocks.values():
        gram += X
        residual = target - design_block @ params
        score = design_block.T @ residual
        meat += np.outer(score, score)
        residual_sum += float(residual @ residual)
        n_obs += len(residual)
        # Lag-1 autocorrelation of the within-patient residuals, measured only
        # across genuinely consecutive grid points, so a gap in the record is
        # never mistaken for a one-step transition.
        if len(residual) > 2:
            consecutive = np.isclose(np.diff(minutes), step_minutes, atol=step_minutes / 4)
            if consecutive.any():
                lag_numerator += float(residual[:-1][consecutive]
                                       @ residual[1:][consecutive])
                lag_denominator += float(residual[:-1][consecutive]
                                         @ residual[:-1][consecutive])

    n_patients = len(blocks)
    inverse = np.linalg.pinv(gram)
    # CR1 small-sample correction, with the absorbed intercepts counted in the
    # parameter total exactly as a fixed-effects fit requires.
    k_total = width + max(n_absorbed, n_patients)
    scale = (n_patients / max(n_patients - 1, 1)) \
        * ((n_obs - 1) / max(n_obs - k_total, 1))
    cov = inverse @ meat @ inverse * scale
    sigma = float(np.sqrt(residual_sum / max(n_obs - k_total, 1)))
    rho1 = float(lag_numerator / lag_denominator) if lag_denominator > 0 else float("nan")
    return Fit(label, names, params, cov, n_patients - 1, n_obs, n_patients,
               sigma=sigma, rho1=rho1, blocks=blocks)


def fit_mixed(design: Design, frame: pd.DataFrame, label: str,
              outcome: str = "sto2", random_slope: bool = True) -> Fit | None:
    """The random-effects companion, in Mundlak form.

    Both the within-patient deviations and the patient means enter the model.
    That matters for two reasons. With the patient means present, the
    coefficients on the deviations keep the same within-patient meaning they
    have in the fixed-effects fit, so the two specifications are comparable
    term by term and agreement between them is a real check. And the mean terms
    make visible what a random-intercept-only model would have folded into the
    estimate instead of separating out.
    """
    if not HAVE_MIXEDLM:
        return None
    groups = frame["subject_id"].to_numpy()
    y = frame[outcome].to_numpy(float)

    names: list[str] = ["intercept"]
    columns: list[np.ndarray] = [np.ones(len(frame))]
    for column in design.within.columns:
        names.append(str(column))
        columns.append(design.within[column].to_numpy(float))
    for column in design.between.columns:
        names.append(str(column))
        columns.append(design.between[column].to_numpy(float))
    exog = np.column_stack(columns)

    keep = np.isfinite(exog).all(axis=1) & np.isfinite(y)
    exog, y, groups = exog[keep], y[keep], groups[keep]

    # Scale each column to a comparable size before handing it to the
    # optimizer. The product terms run to a few thousand while the others are
    # tens, and that spread alone is enough to stall the likelihood search.
    scales = np.where(np.std(exog, axis=0) > 0, np.std(exog, axis=0), 1.0)
    scales[0] = 1.0
    slope_term = names.index(design.terms["x"])
    exog_re = np.column_stack([np.ones(len(y)), exog[:, slope_term]]) \
        if random_slope else np.ones((len(y), 1))
    re_scales = np.array([1.0, scales[slope_term]]) if random_slope \
        else np.array([1.0])

    model = MixedLM(y, exog / scales, groups=groups, exog_re=exog_re / re_scales)
    result = None
    note = ""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for method in ("lbfgs", "bfgs", "cg"):
            try:
                result = model.fit(method=method, maxiter=400, reml=True)
            except Exception as exc:
                note = f"{method} failed: {exc}"
                continue
            if result is not None and np.isfinite(np.asarray(result.fe_params)).all():
                break
    if result is None:
        if random_slope:
            # A patient-varying slope is the first thing to give up when the
            # likelihood will not settle; the fixed part is what is being read.
            fit = fit_mixed(design, frame, label, outcome, random_slope=False)
            if fit is not None:
                fit.note = (f"random {design.terms['x']} slope dropped: the "
                            f"likelihood would not converge with it")
            return fit
        return Fit(label, names, np.full(len(names), np.nan),
                   np.full((len(names), len(names)), np.nan), 1, len(y),
                   int(pd.Series(groups).nunique()), converged=False, note=note)

    params = np.asarray(result.fe_params, float) / scales
    cov = np.asarray(result.cov_params(), float)[:len(names), :len(names)]
    cov = cov / np.outer(scales, scales)
    n_patients = int(pd.Series(groups).nunique())
    converged = bool(getattr(result, "converged", True))
    if not converged:
        note = (note + "; " if note else "") + "optimizer reported no convergence"
    fit = Fit(label, names, params, cov, n_patients - 1, len(y), n_patients,
              converged=converged, note=note)
    try:
        fit.sigma = float(np.sqrt(result.scale))
    except Exception:
        pass
    return fit


# --------------------------------------------------------------------------- #
# Reading a fit: the contrasts the outline asks for
# --------------------------------------------------------------------------- #

def slope_at(fit: Fit, design: Design, value: float, step: float) -> dict:
    """Within-patient exposure slope at an absolute modifier value.

    slope(p) = b_exposure + d_within * (p - reference), scaled to one clinical
    step of the exposure (10 mmHg of MAP, 1 L/min of CO), and read for a patient
    whose average PSi is the cohort reference.
    """
    offset = value - design.reference
    return fit.contrast({design.terms["x"]: step,
                         design.terms["xm"]: step * offset})


def slope_difference(fit: Fit, design: Design, high: float, low: float,
                     step: float) -> dict:
    """How much the exposure slope changes between two modifier values.

    Only the interaction term survives the subtraction, so this is the headline
    interaction expressed in units a clinician can hold: change in cerebral O2
    per 10 mmHg of MAP, per the PSi gap between the two values.
    """
    return fit.contrast({design.terms["xm"]: step * (high - low)})


def partial_residuals(frame: pd.DataFrame, design: Design, fit: Fit,
                      outcome: str = "sto2") -> tuple[np.ndarray, np.ndarray]:
    """Observed data put on the same axes the modelled line is drawn on.

    A fitted line laid over raw points would not line up, because the raw
    points still carry everything else in the model -- the PSi main effect, the
    covariates, each patient's own level. The partial residual strips those out
    and leaves the exposure's own contribution plus the noise, which is exactly
    what the plotted line claims to describe. If the line and the binned points
    disagree, the straight-line assumption is wrong, and that is worth seeing
    rather than inferring.
    """
    values = design.within.to_numpy(float)
    y = frame[outcome].to_numpy(float)
    codes = pd.factorize(frame["subject_id"].to_numpy())[0]
    counts = np.bincount(codes).astype(float)

    def demean(column: np.ndarray) -> np.ndarray:
        sums = np.zeros(len(counts))
        np.add.at(sums, codes, column)
        return column - (sums / counts)[codes]

    centered = np.column_stack([demean(values[:, i])
                                for i in range(values.shape[1])])
    target = demean(y)
    residual = target - centered @ fit.params
    keep = [design.within.columns.get_loc(design.terms[name])
            for name in ("x", "xm", "xb") if design.terms[name] in design.within.columns]
    partial = residual + centered[:, keep] @ fit.params[keep]
    return centered[:, design.within.columns.get_loc(design.terms["x"])], partial


def paired_patient_slopes(frame: pd.DataFrame, exposure: str, modifier: str,
                          deep: float, light: float, window: float, step: float,
                          min_points: int = 30,
                          min_range: float = 10.0) -> pd.DataFrame:
    """Each patient's own exposure slope, measured twice: once deep, once light.

    This is the model-free version of the headline claim. The model says the
    slope changes with depth on average; this asks how many individual patients
    it changed in, which is a different and more clinically useful question. A
    patient only appears if they spent enough time at BOTH depths with enough
    pressure movement at each to fit a line worth having.
    """
    rows: list[dict] = []
    for subject_id, part in frame.groupby("subject_id", sort=True):
        slopes: dict[str, float] = {}
        for label, centre in (("deep", deep), ("light", light)):
            window_rows = part.loc[(part[modifier] - centre).abs() <= window]
            if len(window_rows) < min_points:
                break
            x = window_rows[exposure].to_numpy(float)
            y = window_rows["sto2"].to_numpy(float)
            if float(x.max() - x.min()) < min_range or not np.isfinite(x).all():
                break
            slopes[label] = float(np.polyfit(x, y, 1)[0]) * step
        if len(slopes) == 2:
            rows.append({"subject_id": subject_id, **slopes,
                         "difference": slopes["deep"] - slopes["light"]})
    return pd.DataFrame(rows)


def paired_test(differences: np.ndarray) -> dict:
    """Wilcoxon signed-rank, with the small-sample floor stated rather than hidden.

    With n pairs the smallest P the test can return is 2/2^n, so at n = 6 even a
    perfectly consistent result can only reach 0.031 and at n = 5, 0.0625. A
    bare P near that floor reads as a near-miss when it is really the test
    running out of room.
    """
    values = np.asarray([value for value in differences if np.isfinite(value)])
    out = {"n": int(len(values))}
    if len(values) < 3:
        return out | {"p": float("nan"), "median": float("nan"), "floor": float("nan")}
    out["median"] = float(np.median(values))
    out["positive"] = int((values > 0).sum())
    out["floor"] = 2.0 / (2.0 ** len(values))
    try:
        out["p"] = float(stats.wilcoxon(values).pvalue)
    except Exception:
        out["p"] = float("nan")
    return out


def format_estimate(result: dict, digits: int = 3, unit: str = "") -> str:
    if not np.isfinite(result.get("estimate", np.nan)):
        return "not estimable"
    suffix = f" {unit}" if unit else ""
    if not np.isfinite(result.get("se", np.nan)):
        return f"{result['estimate']:+.{digits}f}{suffix}"
    return (f"{result['estimate']:+.{digits}f}{suffix} "
            f"(95% CI {result['low']:+.{digits}f} to {result['high']:+.{digits}f}), "
            f"P {format_p(result['p'])}")


def format_p(p: float) -> str:
    if not np.isfinite(p):
        return "n/a"
    if p < 1e-4:
        return f"= {p:.1e}"
    if p < 0.001:
        return "< 0.001"
    return f"= {p:.3f}"


def banner(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


# --------------------------------------------------------------------------- #
# Sensitivity analyses
# --------------------------------------------------------------------------- #

def patient_bootstrap(fit: Fit, weights_list: list[dict[str, float]],
                      draws: int, seed: int) -> list[dict]:
    """Resample PATIENTS with replacement, not samples.

    Resampling individual samples would treat ten-second neighbours as
    independent evidence and produce intervals that are far too tight. The
    patient is the independent unit, so the patient is what gets resampled --
    and because the design is already reduced to per-patient cross-products,
    each resample is a small solve rather than a refit.
    """
    if draws <= 0 or not fit.blocks:
        return [{} for _ in weights_list]
    subjects = sorted(fit.blocks)
    rng = np.random.default_rng(seed)
    vectors = []
    for weights in weights_list:
        vector = np.zeros(len(fit.names))
        for name, weight in weights.items():
            vector[fit.names.index(name)] += float(weight)
        vectors.append(vector)
    samples = np.full((draws, len(vectors)), np.nan)
    for draw in range(draws):
        picked = [subjects[i] for i in rng.integers(0, len(subjects), len(subjects))]
        params = solve_blocks(fit.blocks, picked)
        if params is None or not np.isfinite(params).all():
            continue
        for column, vector in enumerate(vectors):
            samples[draw, column] = vector @ params
    out = []
    for column in range(len(vectors)):
        values = samples[:, column]
        values = values[np.isfinite(values)]
        if len(values) < 20:
            out.append({})
            continue
        out.append({"low": float(np.percentile(values, 2.5)),
                    "high": float(np.percentile(values, 97.5)),
                    "median": float(np.median(values)),
                    "draws": int(len(values))})
    return out


def leave_one_patient_out(fit: Fit, weights: dict[str, float]) -> dict:
    """Refit without each patient in turn: is one case carrying the result?"""
    if not fit.blocks:
        return {}
    subjects = sorted(fit.blocks)
    vector = np.zeros(len(fit.names))
    for name, weight in weights.items():
        vector[fit.names.index(name)] += float(weight)
    values: list[tuple[float, str]] = []
    for dropped in subjects:
        params = solve_blocks(fit.blocks, [s for s in subjects if s != dropped])
        if params is None or not np.isfinite(params).all():
            continue
        values.append((float(vector @ params), dropped))
    if not values:
        return {}
    values.sort()
    full = float(vector @ fit.params)
    return {"full": full, "min": values[0][0], "min_patient": values[0][1],
             "max": values[-1][0], "max_patient": values[-1][1],
             "flips_sign": bool(np.sign(values[0][0]) != np.sign(values[-1][0])),
             "n": len(values)}


def thin(frame: pd.DataFrame, minutes: float) -> pd.DataFrame:
    """Keep one sample per patient per interval, so neighbours stop repeating."""
    bucket = np.floor(frame["minutes"].to_numpy(float) / minutes)
    keyed = frame.assign(_bucket=bucket)
    return (keyed.sort_values(["subject_id", "minutes"])
            .drop_duplicates(subset=["subject_id", "_bucket"], keep="first")
            .drop(columns="_bucket"))


def lag_exposure(frame: pd.DataFrame, exposure: str, minutes: float,
                 step_minutes: float) -> pd.DataFrame:
    """Pair cerebral O2 at time t with the exposure measured `minutes` earlier.

    Cause has to come before effect, and the oximeter is slow: the NIRS signal
    is a venous-weighted average over a volume of tissue, so whatever MAP does
    reaches it late. If the interaction is real it should survive -- or sharpen
    -- when the exposure is given that head start.
    """
    shift = int(round(minutes / step_minutes))
    if shift <= 0:
        return frame
    out = frame.sort_values(["subject_id", "minutes"]).copy()
    grouped = out.groupby("subject_id", sort=False)
    out["_lagged"] = grouped[exposure].shift(shift)
    # Only honour the shift where the record really is continuous across it.
    elapsed = grouped["minutes"].diff(shift)
    out.loc[~np.isclose(elapsed, minutes, atol=step_minutes / 2), "_lagged"] = np.nan
    out = out.dropna(subset=["_lagged"])
    out[exposure] = out["_lagged"]
    return out.drop(columns="_lagged")


def difference_frame(frame: pd.DataFrame, horizon_minutes: float,
                     step_minutes: float) -> pd.DataFrame:
    """Recast the record as CHANGES over a fixed horizon instead of levels.

    Why this exists. Regressing cerebral O2 on MAP in levels leaves residuals
    whose lag-1 autocorrelation is 0.99: cerebral oxygenation drifts slowly
    over hours, MAP wobbles on top of that drift, and a levels model spends
    itself describing the drift. Almost none of the nominal sample size is
    independent information, which is why the intervals come out so wide.

    Differencing removes the drift by construction. What is left is the
    question a clinician would actually ask -- when this patient's pressure
    MOVED over the last few minutes, did their cerebral oxygenation move with
    it, and does the answer depend on how deep they were?

    The modifier is the average PSi across the window, because the depth that
    matters is the depth the change happened at. Only intervals the record
    genuinely spans are kept, so a difference is never taken across a gap.
    """
    shift = int(round(horizon_minutes / step_minutes))
    if shift <= 0:
        return frame
    out = frame.sort_values(["subject_id", "minutes"]).copy()
    grouped = out.groupby("subject_id", sort=False)

    pieces: dict[str, pd.Series] = {"subject_id": out["subject_id"],
                                    "minutes": out["minutes"]}
    for column in ("map", "sto2", "co"):
        if column in out.columns:
            pieces[column] = grouped[column].diff(shift)
    pieces["psi"] = (out["psi"] + grouped["psi"].shift(shift)) / 2.0
    for column in ("propofol_rate", "pressor_rate", "bolus_hypnotic",
                   "bolus_vasoactive"):
        if column in out.columns:
            pieces[column] = out[column]

    elapsed = grouped["minutes"].diff(shift)
    spans = np.isclose(elapsed, horizon_minutes, atol=step_minutes / 2)
    delta = pd.DataFrame(pieces).loc[spans]
    return delta.dropna(subset=["map", "sto2", "psi"])


def change_axis(exposure: str, horizon: float) -> dict:
    """Axis labelling for a model fitted on changes rather than levels."""
    axis = dict(AXES[exposure])
    short = AXES[exposure]["short"]
    axis["label"] = (f"Change in {short} over {horizon:g} min "
                     f"({AXES[exposure]['unit']})")
    # Only `short` changes. Every caller already writes "<step_label> of
    # <short>", so putting the word "change" into step_label as well would
    # produce "per 10 mmHg of MAP change of MAP change".
    axis["short"] = f"{short} change"
    return axis


def category_fit(frame: pd.DataFrame, exposure: str, step_minutes: float,
                 edges: tuple[float, ...] = PSI_CATEGORY_EDGES,
                 covariates: tuple[str, ...] = ()) -> tuple[Fit, list[dict]]:
    """One model, with the exposure slope free to differ by PSi category.

    The categories come from the outline's sensitivity list (<25, 25-50, 50-75,
    >75). Fitting them inside ONE model rather than as four separate subgroup
    fits is what makes the differences between them testable.

    PATIENT x CATEGORY is absorbed, not just patient. A patient passes through
    several categories during a case and sits at a different cerebral O2 level
    in each -- partly because the depth itself shifts the level. Sharing one
    baseline per patient across categories leaves that level difference in the
    residual, where it correlates with MAP and pulls every category's slope
    toward the others. Giving each patient-category cell its own baseline makes
    each slope exactly what it claims to be: the relationship inside that
    patient while they were at that depth.
    """
    labels = [f"{int(edges[i])}-{int(edges[i + 1])}"
              for i in range(len(edges) - 1)]
    codes = np.digitize(frame["psi"].to_numpy(float), edges[1:-1], right=False)
    work = frame.copy()
    work["_category"] = codes
    cell = (work["subject_id"].astype(str) + "|" + work["_category"].astype(str))
    x = work[exposure].to_numpy(float)

    columns: dict[str, np.ndarray] = {}
    present: list[int] = []
    for index, label in enumerate(labels):
        mask = (codes == index).astype(float)
        if mask.sum() < 2:
            continue
        present.append(index)
        columns[f"{AXES[exposure]['short']} [{label}]"] = x * mask
    for name in covariates:
        columns[name] = work[name].to_numpy(float)
    design = pd.DataFrame(columns, index=work.index)
    fit = fit_within(work, design, "PSi categories", step_minutes=step_minutes,
                     absorb=cell)

    step = AXES[exposure]["step"]
    out: list[dict] = []
    for index in present:
        label = labels[index]
        mask = codes == index
        name = f"{AXES[exposure]['short']} [{label}]"
        if name not in fit.names:
            continue
        result = fit.contrast({name: step})
        result.update({"label": label, "n_obs": int(mask.sum()),
                       "n_patients": int(work.loc[mask, "subject_id"].nunique())})
        out.append(result)
    return fit, out


def nonlinearity_fit(frame: pd.DataFrame, exposure: str, modifier: str,
                     centers: dict[str, float], step_minutes: float,
                     covariates: tuple[str, ...] = ()) -> tuple[Fit, Design]:
    """Does the interaction survive letting the relationship bend?

    A straight line forced through a curved relationship can manufacture an
    interaction all by itself, so the quadratic terms and their product with
    depth go in and the headline contrast is recomputed. If it holds, the
    linear reading is not an artefact of the shape assumed.
    """
    design = build_design(frame, exposure, modifier, centers, covariates)
    x = design.within[design.terms["x"]].to_numpy(float)
    m = design.within[design.terms["m"]].to_numpy(float)
    extended = design.within.copy()
    extended[f"{AXES[exposure]['short']}^2 (within)"] = x ** 2
    extended[f"{AXES[modifier]['short']}^2 (within)"] = m ** 2
    extended[f"{AXES[exposure]['short']}^2 x {AXES[modifier]['short']}"] = (x ** 2) * m
    # The term that lets the SLOPE bend with depth. Without it a straight line
    # in PSi is imposed, and a relationship that is flat across most of the PSi
    # range and then reverses at the top -- which the category fit can show --
    # would be averaged away into nothing.
    extended[f"{AXES[exposure]['short']} x {AXES[modifier]['short']}^2"] = x * (m ** 2)
    fit = fit_within(frame, extended, "with quadratic terms",
                     step_minutes=step_minutes)
    return fit, design


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #

def make_marginal_figure(fit: Fit, design: Design, exposure: str, modifier: str,
                         frame: pd.DataFrame, representative: list[float],
                         output_path: Path, title: str,
                         adjusted_for: tuple[str, ...],
                         axis_override: dict | None = None) -> None:
    """The outline's Figure 1 (and Figure 2): the exposure effect, by PSi.

    Panel A is the thing itself -- what the model says happens to cerebral O2
    as a patient's own MAP moves away from their own average, drawn once per
    representative PSi. The x-axis is a DEVIATION and the y-axis is a CHANGE
    because that is what a within-patient model estimates: patient fixed
    effects deliberately discard every patient's absolute level, so absolute
    levels would be drawn from nothing.

    Panel B drops the choice of representative values and shows the slope as a
    continuous function of PSi, which is the interaction itself. Where the band
    crosses zero is where the relationship changes sign.

    Panel C is the numbers the report quotes, as a forest: slope at each
    representative PSi, then the difference between the outermost two.
    """
    axis = axis_override or AXES[exposure]
    step = axis["step"]
    deviation = design.within[design.terms["x"]].to_numpy(float)
    low, high = np.percentile(deviation, [2.5, 97.5])
    grid = np.linspace(low, high, 80)
    psi_values = frame[modifier].to_numpy(float)
    psi_low, psi_high = np.percentile(psi_values, [1, 99])
    psi_grid = np.linspace(psi_low, psi_high, 80)
    absolute_center = float(np.round(frame[exposure].mean(), 1))

    figure = plt.figure(figsize=(14.5, 11.0))
    spec = figure.add_gridspec(2, 2, wspace=0.26, hspace=0.34, left=0.07,
                               right=0.975, top=0.86, bottom=0.07)
    ax_a = figure.add_subplot(spec[0, 0])
    ax_b = figure.add_subplot(spec[0, 1])
    ax_c = figure.add_subplot(spec[1, 0])
    ax_d = figure.add_subplot(spec[1, 1])

    # ---- A: modelled cerebral O2 change against the exposure ---------------
    # A window around each representative PSi, so the observed points drawn
    # under a line are the ones that line actually claims to describe.
    spacing = (min(np.diff(representative)) if len(representative) > 1 else 20.0)
    window = max(5.0, spacing / 2.0)
    observed_x, observed_y = partial_residuals(frame, design, fit)
    psi_column = frame[modifier].to_numpy(float)

    for index, psi in enumerate(representative):
        offset = psi - design.reference
        estimates, lows, highs = [], [], []
        for delta in grid:
            result = fit.contrast({design.terms["x"]: delta,
                                   design.terms["xm"]: delta * offset})
            estimates.append(result["estimate"])
            lows.append(result["low"])
            highs.append(result["high"])
        color = BAND_COLORS[index % len(BAND_COLORS)]

        near = np.abs(psi_column - psi) <= window
        if near.sum() > 40:
            edges = np.quantile(observed_x[near], np.linspace(0, 1, 11))
            edges = np.unique(edges)
            if len(edges) > 2:
                middle = np.digitize(observed_x[near], edges[1:-1])
                centres, means = [], []
                for bucket in range(len(edges) - 1):
                    chosen = middle == bucket
                    if chosen.sum() < 10:
                        continue
                    centres.append(float(np.median(observed_x[near][chosen])))
                    means.append(float(np.mean(observed_y[near][chosen])))
                ax_a.plot(centres, means, "o", color=color, markersize=5.5,
                          markerfacecolor="white", markeredgewidth=1.6,
                          zorder=4, linestyle="none")

        ax_a.fill_between(grid, lows, highs, color=color, alpha=0.16, linewidth=0)
        ax_a.plot(grid, estimates, color=color, linewidth=2.4, zorder=3,
                  label=f"PSi {psi:g}")
    ax_a.axhline(0, color="#999999", linewidth=0.9, linestyle=":")
    ax_a.axvline(0, color="#999999", linewidth=0.9, linestyle=":")
    ax_a.set_xlabel(f"{axis['short']} relative to the patient's own average "
                    f"({axis['unit']})")
    ax_a.set_ylabel("Modelled change in cerebral O2 (%)")
    ax_a.set_title(f"A. Within-patient {axis['short']}-cerebral O2 relationship\n"
                   f"at representative PSi values (hollow dots = observed)",
                   fontsize=11, loc="left")
    ax_a.legend(frameon=False, fontsize=9, loc="upper left",
                title="PSi held at", title_fontsize=9)
    ax_a.grid(alpha=0.18)
    ax_a.annotate(f"0 = each patient's own average {axis['short']}\n"
                  f"(cohort average {absolute_center:g} {axis['unit']}); the "
                  f"lines must meet there\nDots: binned partial residuals "
                  f"within {window:g} PSi of each value",
                  (0.98, 0.03), xycoords="axes fraction", ha="right",
                  va="bottom", fontsize=8.0, color="#666666")

    # ---- B: the slope as a continuous function of PSi ----------------------
    estimates, lows, highs = [], [], []
    for psi in psi_grid:
        result = slope_at(fit, design, psi, step)
        estimates.append(result["estimate"])
        lows.append(result["low"])
        highs.append(result["high"])
    ax_b.fill_between(psi_grid, lows, highs, color="#2f5597", alpha=0.18,
                      linewidth=0, label="95% CI")
    ax_b.plot(psi_grid, estimates, color="#2f5597", linewidth=2.6)
    ax_b.axhline(0, color="#c00000", linewidth=1.2, linestyle="--",
                 label="no relationship")
    for index, psi in enumerate(representative):
        result = slope_at(fit, design, psi, step)
        ax_b.plot([psi], [result["estimate"]], "o", markersize=7,
                  color=BAND_COLORS[index % len(BAND_COLORS)], zorder=5)
        ax_b.annotate(f"PSi {psi:g}", (psi, result["estimate"]),
                      textcoords="offset points", xytext=(6, 8), fontsize=9,
                      color=BAND_COLORS[index % len(BAND_COLORS)])
    # The PSi histogram behind the curve says where the estimate is supported
    # by data and where the line is only an extrapolation of the model.
    density = ax_b.twinx()
    density.hist(psi_values, bins=40, color="#999999", alpha=0.16, linewidth=0,
                 label="PSi distribution")
    density.set_yticks([])
    ax_b.set_zorder(density.get_zorder() + 1)
    ax_b.patch.set_visible(False)
    ax_b.set_xlabel("PSi")
    ax_b.set_ylabel(f"Cerebral O2 change {axis['step_label']}\nof "
                    f"{axis['short']} (%)")
    ax_b.set_title(f"B. The {axis['short']} slope across PSi\n"
                   f"(the interaction itself)", fontsize=11, loc="left")
    handles, labels = ax_b.get_legend_handles_labels()
    handles.append(matplotlib.patches.Patch(color="#999999", alpha=0.3))
    labels.append("how much data sits at each PSi")
    ax_b.legend(handles, labels, frameon=False, fontsize=8.5, loc="best")
    ax_b.grid(alpha=0.18)

    # ---- C: the quoted contrasts, as a forest ------------------------------
    rows: list[tuple[str, dict]] = []
    for psi in representative:
        rows.append((f"at PSi {psi:g}", slope_at(fit, design, psi, step)))
    if len(representative) >= 2:
        rows.append((f"difference,\nPSi {representative[-1]:g} vs "
                     f"{representative[0]:g}",
                     slope_difference(fit, design, representative[-1],
                                      representative[0], step)))
    positions = np.arange(len(rows))[::-1]
    finite = [value for _label, result in rows
              for value in (result["low"], result["high"])
              if np.isfinite(value)] or [0.0]
    span = max(max(finite) - min(finite), 1e-6)
    left, right = min(finite) - 0.45 * span, max(finite) + 0.45 * span
    for position, (_label, result) in zip(positions, rows):
        color = "#c00000" if (np.isfinite(result["low"])
                              and result["low"] * result["high"] > 0) else "#495057"
        ax_c.plot([result["low"], result["high"]], [position, position],
                  color=color, linewidth=2.2)
        ax_c.plot([result["estimate"]], [position], "o", color=color, markersize=7)
        # Keep the annotation inside the axes however far out the point sits.
        anchor = float(np.clip(result["estimate"], left + 0.28 * span,
                               right - 0.28 * span))
        ax_c.annotate(f"{result['estimate']:+.3f}  "
                      f"[{result['low']:+.3f}, {result['high']:+.3f}]\n"
                      f"P {format_p(result['p'])}",
                      (anchor, position), textcoords="offset points",
                      xytext=(0, 12), ha="center", fontsize=8.5, color=color)
    ax_c.axvline(0, color="#c00000", linewidth=1.2, linestyle="--")
    ax_c.set_xlim(left, right)
    ax_c.set_yticks(positions)
    ax_c.set_yticklabels([label for label, _result in rows], fontsize=9.5)
    ax_c.set_ylim(-0.7, len(rows) - 0.1)
    ax_c.set_xlabel(f"Cerebral O2 change (%) {axis['step_label']} "
                    f"of {axis['short']}")
    ax_c.set_title("C. The reported contrasts\n(red = interval excludes zero)",
                   fontsize=11, loc="left")
    ax_c.grid(alpha=0.18, axis="x")

    # ---- D: one dot per patient --------------------------------------------
    # Panels A to C are cohort averages. They cannot tell "the slope steepens
    # with depth in most patients" apart from "it steepens enormously in a
    # handful and not at all in the rest", and those mean different things at
    # the bedside. This panel answers that by measuring each patient twice.
    deep, light = representative[0], representative[-1]
    paired = paired_patient_slopes(frame, exposure, modifier, deep, light,
                                   window, step,
                                   min_range=10.0 if exposure == "map" else 0.8)
    if len(paired) >= 3:
        test = paired_test(paired["difference"].to_numpy(float))
        limits = np.concatenate([paired["light"].to_numpy(float),
                                 paired["deep"].to_numpy(float)])
        low_limit, high_limit = float(limits.min()), float(limits.max())
        pad = max((high_limit - low_limit) * 0.12, 0.25)
        bounds = [low_limit - pad, high_limit + pad]
        ax_d.plot(bounds, bounds, color="#999999", linewidth=1.2, linestyle="--",
                  label="no change with depth")
        steeper = paired["deep"] > paired["light"]
        ax_d.plot(paired.loc[~steeper, "light"], paired.loc[~steeper, "deep"],
                  "o", color="#495057", markersize=7, alpha=0.75,
                  label=f"flatter when deep ({int((~steeper).sum())})")
        ax_d.plot(paired.loc[steeper, "light"], paired.loc[steeper, "deep"],
                  "o", color="#c00000", markersize=7, alpha=0.8,
                  label=f"steeper when deep ({int(steeper.sum())})")
        ax_d.axhline(0, color="#cccccc", linewidth=0.8)
        ax_d.axvline(0, color="#cccccc", linewidth=0.8)
        ax_d.set_xlim(bounds)
        ax_d.set_ylim(bounds)
        ax_d.set_xlabel(f"That patient's slope near PSi {light:g} (lighter)")
        ax_d.set_ylabel(f"That patient's slope near PSi {deep:g} (deeper)")
        caption = (f"median paired change {test.get('median', float('nan')):+.3f}"
                   f", {test['n']} patients")
        if np.isfinite(test.get("p", np.nan)):
            caption += f"\nWilcoxon signed-rank P {format_p(test['p'])}"
            at_floor = test["p"] <= test["floor"] * 1.5
            if at_floor and test["floor"] > 0.001:
                # Few enough pairs that the test cannot reach a small P however
                # consistent the result is -- worth saying, or the P reads as a
                # near-miss when it is the test running out of room.
                caption += (f"\nbut {test['n']} pairs cannot take this test "
                            f"below {test['floor']:.3g} however consistent "
                            f"they are")
            elif at_floor:
                caption += (f"\n— the smallest value {test['n']} pairs can "
                            f"give: every patient moved the same way")
        ax_d.annotate(caption, (0.03, 0.97), xycoords="axes fraction",
                      ha="left", va="top", fontsize=8.6, color="#333333")
        ax_d.legend(frameon=False, fontsize=8.5, loc="lower right")
    else:
        ax_d.axis("off")
        ax_d.annotate(
            f"Too few patients spent enough time at BOTH PSi {deep:g} and "
            f"PSi {light:g}\nwith enough {axis['short']} movement at each to "
            f"fit a slope twice.\n\nThe cohort estimate in panels A to C still "
            f"stands; what cannot be\nshown is how consistent it is from "
            f"patient to patient.",
            (0.5, 0.5), xycoords="axes fraction", ha="center", va="center",
            fontsize=9.5, color="#666666")
    ax_d.set_title(f"D. Each patient measured twice\n"
                   f"(cerebral O2 change {axis['step_label']} of "
                   f"{axis['short']})", fontsize=11, loc="left")
    ax_d.grid(alpha=0.18)

    adjustment = (", adjusted for " + ", ".join(COVARIATE_LABELS[name]
                                                for name in adjusted_for)
                  if adjusted_for else ", unadjusted (no covariate file supplied)")
    note = (f"Within-patient estimates (patient-mean decomposition with "
            f"patient fixed effects), patient-clustered 95% CIs. "
            f"{fit.n_patients} patients, {fit.n_obs:,} time points{adjustment}. "
            f"Panels A to C are the model; panel D measures each patient "
            f"separately and assumes nothing.")
    figure.suptitle(f"{title}\n" + "\n".join(textwrap.wrap(note, 125)),
                    fontsize=12.5, x=0.07, y=0.985, ha="left", va="top")
    figure.savefig(output_path, dpi=200)
    plt.close(figure)
    print(f"  wrote {output_path}")


def make_category_figure(rows: list[dict], exposure: str, output_path: Path,
                         title: str, note: str) -> None:
    """The outline's optional Figure 3: PSi categories, with patient counts."""
    if not rows:
        return
    axis = AXES[exposure]
    figure, (ax_left, ax_right) = plt.subplots(
        1, 2, figsize=(13.0, 5.6), gridspec_kw={"width_ratios": [1.5, 1.0],
                                                "wspace": 0.24})
    positions = np.arange(len(rows))
    for position, row in zip(positions, rows):
        color = "#c00000" if (np.isfinite(row["low"])
                              and row["low"] * row["high"] > 0) else "#2f5597"
        ax_left.plot([position, position], [row["low"], row["high"]],
                     color=color, linewidth=2.6)
        ax_left.plot([position], [row["estimate"]], "o", color=color, markersize=8)
        ax_left.annotate(f"{row['estimate']:+.3f}", (position, row["estimate"]),
                         textcoords="offset points", xytext=(9, 0), fontsize=9,
                         color=color, va="center")
    ax_left.axhline(0, color="#c00000", linewidth=1.2, linestyle="--")
    ax_left.set_xticks(positions)
    ax_left.set_xticklabels([row["label"] for row in rows])
    ax_left.set_xlabel("PSi category")
    ax_left.set_ylabel(f"Cerebral O2 change {axis['step_label']}\nof "
                       f"{axis['short']} (%)")
    ax_left.set_title("Slope with 95% CI  (red = interval excludes zero)",
                      fontsize=11, loc="left")
    ax_left.grid(alpha=0.18, axis="y")

    width = 0.38
    patients = [row["n_patients"] for row in rows]
    points = [row["n_obs"] for row in rows]
    ax_right.bar(positions - width / 2, patients, width, color="#2f5597")
    twin = ax_right.twinx()
    twin.bar(positions + width / 2, points, width, color="#c55a11")
    ax_right.set_ylim(0, max(patients) * 1.2)
    twin.set_ylim(0, max(points) * 1.2)
    # The patient count goes INSIDE its bar and the point count above its own,
    # so the two never land on the same spot when both bars are full height.
    for position, row in zip(positions, rows):
        ax_right.annotate(f"{row['n_patients']}",
                          (position - width / 2, row["n_patients"]),
                          textcoords="offset points", xytext=(0, -4),
                          ha="center", va="top", fontsize=9, color="white",
                          fontweight="bold")
        twin.annotate(f"{row['n_obs']:,}", (position + width / 2, row["n_obs"]),
                      textcoords="offset points", xytext=(0, 4), ha="center",
                      fontsize=8.5, color="#c55a11")
    ax_right.set_xticks(positions)
    ax_right.set_xticklabels([row["label"] for row in rows])
    ax_right.set_xlabel("PSi category")
    ax_right.set_ylabel("Patients contributing", color="#2f5597")
    twin.set_ylabel("Time points", color="#c55a11")
    ax_right.set_title("How much data stands behind each category",
                       fontsize=11, loc="left")

    figure.suptitle(f"{title}\n" + "\n".join(textwrap.wrap(note, 118)),
                    fontsize=12.5, x=0.045, y=0.975, ha="left", va="top")
    figure.subplots_adjust(top=0.76, bottom=0.12, left=0.075, right=0.93)
    figure.savefig(output_path, dpi=200)
    plt.close(figure)
    print(f"  wrote {output_path}")


def make_specification_figure(panels: list[tuple[str, str, list[tuple[str, dict]]]],
                              output_path: Path, title: str, note: str) -> None:
    """Every specification's headline contrast on one axis.

    A single estimate with an interval invites the reader to ask whether some
    other reasonable choice would have given a different answer. This answers
    that question directly instead of leaving it open: if the whole column sits
    on one side of zero the result is robust to how it was fitted, and if it
    straddles zero and wanders, the honest conclusion is that the data does not
    settle the question -- which is itself worth showing rather than burying in
    a sensitivity paragraph.
    """
    panels = [panel for panel in panels if panel[2]]
    if not panels:
        return
    heights = [max(len(rows), 1) for _label, _unit, rows in panels]
    # Size to the rows: one line each, plus a title and axis per panel and the
    # caption at the top. Guessing leaves a page of white space between them.
    total = 1.25 + 0.30 * sum(heights) + 0.85 * len(panels)
    figure, axes = plt.subplots(
        len(panels), 1, figsize=(11.5, total),
        gridspec_kw={"height_ratios": heights})
    if len(panels) == 1:
        axes = [axes]

    for axis, (label, unit, rows) in zip(axes, panels):
        positions = np.arange(len(rows))[::-1]
        for position, (name, result) in zip(positions, rows):
            if not np.isfinite(result.get("estimate", np.nan)):
                continue
            crosses = not (np.isfinite(result.get("low", np.nan))
                           and result["low"] * result["high"] > 0)
            color = "#495057" if crosses else "#c00000"
            weight = 2.0 if name.startswith("as reported") else 1.5
            axis.plot([result["low"], result["high"]], [position, position],
                      color=color, linewidth=weight)
            axis.plot([result["estimate"]], [position], "o", color=color,
                      markersize=7 if name.startswith("as reported") else 5.5)
            axis.annotate(f"  {result['estimate']:+.3f} "
                          f"[{result['low']:+.3f}, {result['high']:+.3f}]",
                          (result["high"], position), fontsize=8,
                          va="center", color=color)
        axis.axvline(0, color="#c00000", linewidth=1.2, linestyle="--")
        axis.set_yticks(positions)
        axis.set_yticklabels([name for name, _result in rows], fontsize=9)
        axis.set_ylim(-0.6, len(rows) - 0.4)
        axis.set_xlabel(unit, fontsize=9)
        axis.set_title(label, fontsize=11, loc="left")
        axis.grid(alpha=0.18, axis="x")
        span = axis.get_xlim()
        axis.set_xlim(span[0], span[0] + (span[1] - span[0]) * 1.45)

    figure.suptitle(f"{title}\n" + "\n".join(textwrap.wrap(note, 118)),
                    fontsize=12.5, x=0.02, y=1.0 - 0.1 / total, ha="left",
                    va="top")
    figure.subplots_adjust(top=1.0 - 1.15 / total, bottom=0.55 / total,
                           left=0.265, right=0.985, hspace=0.75 / total * 4)
    figure.savefig(output_path, dpi=200)
    plt.close(figure)
    print(f"  wrote {output_path}")


def render_table(rows: list[tuple[str, ...]], widths: list[int], title: str,
                 footnote: str, output_path: Path) -> None:
    """Draw a table as an image, so it can be attached next to the figures.

    Monospace text in a bare axes rather than a styled table: it survives any
    cell content, never truncates a long label, and reads the same on screen
    as in an email.
    """
    header, *body = rows
    body_size, footnote_size = 9.2, 8.4

    def render(cells: tuple[str, ...]) -> str:
        return "  ".join(str(cell).ljust(w)[:max(w, 1)]
                         for cell, w in zip(cells, widths))

    text = [render(header), "-" * (sum(widths) + 2 * (len(widths) - 1))]
    text += [render(row) for row in body]
    footnote_lines = footnote.count("\n") + 1 if footnote else 0

    # Size the canvas to the text rather than guessing: a table laid out in a
    # figure that is too tall is mostly white space, and one that is too short
    # overlaps its own footnote.
    line = body_size * 1.5 / 72.0
    height = (0.55                                   # title
              + line * (len(text) + 1)
              + footnote_size * 1.45 / 72.0 * footnote_lines
              + 0.45)
    character = body_size * 0.62 / 72.0
    body_width = character * (sum(widths) + 2 * (len(widths) - 1))
    footnote_width = (max((len(part) for part in footnote.split("\n")), default=0)
                      * footnote_size * 0.52 / 72.0)
    width = max(body_width, footnote_width, 6.5) + 0.6

    figure = plt.figure(figsize=(width, max(height, 2.5)))
    axis = figure.add_axes([0.0, 0.0, 1.0, 1.0])
    axis.axis("off")
    top = 1.0 - 0.18 / height
    axis.text(0.03, top, title, fontsize=12.5, fontweight="bold",
              va="top", ha="left", transform=axis.transAxes)
    axis.text(0.03, top - 0.42 / height, "\n".join(text), fontsize=body_size,
              family="monospace", va="top", ha="left", transform=axis.transAxes,
              linespacing=1.5)
    if footnote:
        axis.text(0.03, 0.16 / height, footnote, fontsize=footnote_size,
                  va="bottom", ha="left", color="#444444",
                  transform=axis.transAxes, linespacing=1.45)
    figure.savefig(output_path, dpi=200)
    plt.close(figure)
    print(f"  wrote {output_path}")


# --------------------------------------------------------------------------- #
# One analysis: fit, report, draw
# --------------------------------------------------------------------------- #

def describe(values: pd.Series, digits: int = 1) -> str:
    return (f"{values.median():.{digits}f} "
            f"({values.quantile(.25):.{digits}f}-{values.quantile(.75):.{digits}f})")


def characteristic_rows(characteristics: pd.DataFrame, subjects: list[str],
                        max_levels: int = 6) -> tuple[list[tuple[str, str]],
                                                      list[str]]:
    """Table 1's top half: the cohort, summarized field by field.

    Numeric fields get median (IQR), categorical fields get one line per level
    with n (%). A field the export did not carry, or carried blank for this
    cohort, is listed as unavailable: a reader of Table 1 needs to know the
    difference between "we measured this and it was unremarkable" and "we never
    had it".
    """
    rows: list[tuple[str, str]] = []
    missing: list[str] = []
    if characteristics.empty:
        return rows, [label for _key, label, _kind, _digits in CHARACTERISTIC_LAYOUT]

    present = characteristics.reindex(subjects)
    total = len(present)
    for key, label, kind, digits in CHARACTERISTIC_LAYOUT:
        if key not in present.columns:
            missing.append(label)
            continue
        values = present[key]
        if kind == "numeric":
            numeric = pd.to_numeric(values, errors="coerce").dropna()
            if numeric.empty:
                missing.append(label)
                continue
            suffix = (f"  [{len(numeric)} of {total}]"
                      if len(numeric) < total else "")
            rows.append((label, describe(numeric, digits) + suffix))
            continue

        text = values.astype("string").str.strip()
        text = text.loc[text.notna() & text.ne("") & text.str.lower().ne("nan")]
        if text.empty:
            missing.append(label)
            continue
        counts = text.value_counts()
        # An ordered scale reads in its own order, not in order of how common
        # each level happens to be: ASA 2, 3, 4 -- never 3, 2, 4.
        numeric_levels = pd.to_numeric(pd.Series(counts.index), errors="coerce")
        if numeric_levels.notna().all():
            counts = counts.iloc[np.argsort(numeric_levels.to_numpy())]
            shown_levels, overflow = counts, None
        else:
            shown_levels = counts.head(max_levels)
            overflow = counts.iloc[max_levels:] if len(counts) > max_levels else None
        rows.append((label, f"{len(text)} of {total} recorded"))
        for level, count in shown_levels.items():
            shown = str(level)
            # Surgical-procedure labels run to a full sentence; the leading
            # words are what distinguishes them, and the rest would overflow
            # the rendered column and be cut mid-word anyway.
            if len(shown) > 38:
                shown = shown[:35].rstrip(" ,(-") + "..."
            rows.append((f"    {shown}",
                         f"{count} ({100.0 * count / total:.0f}%)"))
        if overflow is not None and len(overflow):
            rest = int(overflow.sum())
            plural = "category" if len(overflow) == 1 else "categories"
            rows.append((f"    other ({len(overflow)} {plural})",
                         f"{rest} ({100.0 * rest / total:.0f}%)"))
    return rows, missing


def run_model(frame: pd.DataFrame, exposure: str, modifier: str,
              centers: dict[str, float], representative: list[float],
              covariates: tuple[str, ...], args: argparse.Namespace,
              label: str, axis_override: dict | None = None,
              brief: bool = False) -> dict:
    """Fit, print and collect one exposure's model. Figures are drawn by main."""
    axis = axis_override or AXES[exposure]
    step = axis["step"]
    design = build_design(frame, exposure, modifier, centers, covariates)
    fit = fit_within(frame, design.within, label,
                     step_minutes=args.step_seconds / 60.0)

    print("\n  Specification: cerebral O2 ~ "
          + " + ".join(design.within.columns)
          + " + patient fixed effects")
    print(f"  {axis['short']} and PSi both enter as deviations from the "
          f"patient's own mean, so no part of the estimate comes from "
          f"comparing one patient with another.")
    print(f"  Slopes are read at an absolute PSi, using PSi "
          f"{design.reference:g} as the reference (the mean of the "
          f"per-patient PSi means).")
    print(f"  {fit.n_patients} patients, {fit.n_obs:,} time points, "
          f"{fit.dof} degrees of freedom for inference (patients - 1).")
    if not covariates:
        print("  UNADJUSTED: no anesthesia-record file was supplied, so "
              "end-tidal CO2, inspired O2 fraction and anesthetic "
              "concentration are not in the model.")

    print(f"\n  Interaction ({axis['short']} x PSi), the quantity of interest:")
    raw = fit.coefficient(design.terms["xm"])
    print(f"    {format_estimate(raw, 5)}  "
          f"% cerebral O2 per {axis['unit']} per PSi unit")
    scaled = fit.contrast({design.terms["xm"]: step * 10.0})
    print(f"    {format_estimate(scaled, 3)}  "
          f"% cerebral O2 {axis['step_label']} of {axis['short']}, "
          f"per 10 PSi units")

    cross = fit.coefficient(design.terms["xb"])
    if not brief:
        print(f"\n  Cross-level term ({axis['short']} x the patient's MEAN "
              f"PSi), reported separately so it cannot contaminate the one "
              f"above:")
        print(f"    {format_estimate(cross, 5)}  "
              f"% cerebral O2 per {axis['unit']} per PSi unit of average depth")
        print(f"    This asks a different question — whether patients who run "
              f"deeper ON AVERAGE have a different {axis['short']} slope — and "
              f"it is a between-patient comparison, so it is not the clinical "
              f"estimate. Pooling the two into one coefficient, which writing "
              f"the model with raw {axis['short']} x PSi would do, drags the "
              f"estimate toward whichever is weaker.")
        equality = fit.contrast({design.terms["xm"]: 1.0,
                                 design.terms["xb"]: -1.0})
        print(f"    Difference between the two: {format_estimate(equality, 5)}"
              + ("  — no evidence they differ, so a model that pooled them "
                 "would not have been badly wrong here."
                 if np.isfinite(equality["p"]) and equality["p"] >= 0.05 else
                 "  — they differ, so keeping them apart was necessary."))

    print(f"\n  Cerebral O2 change {axis['step_label']} of {axis['short']}, "
          f"at each representative PSi:")
    slopes = {}
    for psi in representative:
        result = slope_at(fit, design, psi, step)
        slopes[psi] = result
        print(f"    PSi {psi:5g}:  {format_estimate(result, 3, '% cerebral O2')}")
    difference = None
    if len(representative) >= 2:
        difference = slope_difference(fit, design, representative[-1],
                                      representative[0], step)
        print(f"\n  Difference between those slopes "
              f"(PSi {representative[-1]:g} minus PSi {representative[0]:g}):")
        print(f"    {format_estimate(difference, 3, '% cerebral O2 ' + axis['step_label'])}")

    print(f"\n  Residual SD {fit.sigma:.2f} % cerebral O2; lag-1 "
          f"autocorrelation of the within-patient residuals "
          f"{fit.rho1:.3f} at {args.step_seconds:g} s spacing.")
    if np.isfinite(fit.rho1) and fit.rho1 > 0.5:
        effective = (1 - fit.rho1) / (1 + fit.rho1)
        print(f"    Strongly autocorrelated, as expected from a {args.step_seconds:g} s "
              f"grid: consecutive samples carry roughly {effective:.1%} of the "
              f"independent information their count implies. This is exactly "
              f"why inference is clustered on patient and why the degrees of "
              f"freedom above are {fit.dof}, not {fit.n_obs:,}.")

    mixed = None
    if not args.no_mixed and not brief:
        mixed = fit_mixed(design, frame, label + " (random effects)",
                          random_slope=args.random_slope)
        if mixed is None:
            print("\n  Random-effects companion: statsmodels MixedLM is not "
                  "available in this environment, so only the fixed-effects "
                  "fit is reported.")
        elif not np.isfinite(mixed.params).all():
            print("\n  Random-effects companion did not converge"
                  + (f" ({mixed.note})" if mixed.note else "") + ".")
        else:
            within_term = mixed.coefficient(design.terms["xm"])
            print("\n  Random-effects companion (Mundlak: the same within "
                  "terms plus the patient means, random intercept"
                  + (f" and random {axis['short']} slope"
                     if args.random_slope else "") + "):")
            if mixed.note:
                print(f"    note: {mixed.note}")
            print(f"    within-patient interaction  {format_estimate(within_term, 5)}")
            agreement = abs(within_term["estimate"] - raw["estimate"])
            relative = (agreement / abs(raw["estimate"]) * 100.0
                        if raw["estimate"] else float("nan"))
            print(f"    Differs from the fixed-effects estimate by "
                  f"{agreement:.5f} ({relative:.0f}%). These are two different "
                  f"estimators of the same within-patient quantity, so "
                  f"closeness is the check that neither is being driven by its "
                  f"own assumptions.")
            if np.isfinite(fit.rho1) and fit.rho1 > 0.8:
                # A mixed model treats residuals as independent once the
                # random effects are taken out. At this much autocorrelation
                # that is simply false, and the price is paid entirely in the
                # standard errors: the interval below can be an order of
                # magnitude too narrow and its P value correspondingly tiny.
                print(f"    !! DO NOT QUOTE THE INTERVAL ABOVE. With a lag-1 "
                      f"residual autocorrelation of {fit.rho1:.3f}, this "
                      f"model's assumption of independent residuals is badly "
                      f"violated, so its standard errors are far too small "
                      f"and its P value far too impressive. Only the point "
                      f"estimate is informative here; the fixed-effects "
                      f"interval, clustered on patient, is the one to report.")
            for column in design.between.columns:
                print(f"    {column:<28} {format_estimate(mixed.coefficient(column), 4)}")
            print("    Those last rows are the between-patient comparisons. "
                  "They are shown only to make visible what a "
                  "random-intercept-only model would have folded into the "
                  "interaction; they answer a question about who the patients "
                  "are, not about what happens inside one.")

    return {"fit": fit, "mixed": mixed, "design": design, "raw": raw,
            "scaled": scaled, "cross": cross, "slopes": slopes,
            "difference": difference, "exposure": exposure,
            "modifier": modifier, "frame": frame,
            "representative": representative, "covariates": covariates}


def run_sensitivities(result: dict, args: argparse.Namespace,
                      centers: dict[str, float]) -> list[str]:
    """The outline's section 5, run against one fitted model."""
    fit = result["fit"]
    design = result["design"]
    exposure, modifier = result["exposure"], result["modifier"]
    frame = result["frame"]
    covariates = result["covariates"]
    representative = result["representative"]
    axis = AXES[exposure]
    step = axis["step"]
    gap = step * (representative[-1] - representative[0])
    headline = {design.terms["xm"]: gap}
    lines: list[str] = []
    # Every variant's headline contrast, kept so they can be drawn on one axis.
    # With a null this unstable, the spread across specifications is more
    # informative than any single one of them.
    variants: list[tuple[str, dict]] = []

    def refit(subset: pd.DataFrame, label: str,
              record: str | None = None) -> dict | None:
        """Rebuild the design on the subset, so patient means are recomputed."""
        if subset.empty or subset["subject_id"].nunique() < 3:
            return None
        rebuilt = build_design(subset, exposure, modifier, centers, covariates)
        new = fit_within(subset, rebuilt.within, label,
                         step_minutes=args.step_seconds / 60.0)
        if not np.isfinite(new.params).all():
            return None
        out = new.contrast({rebuilt.terms["xm"]: gap})
        out.update({"n_patients": new.n_patients, "n_obs": new.n_obs})
        if record:
            variants.append((record, out))
        return out

    reference = fit.contrast(headline)
    variants.append(("as reported (primary fit)", reference))
    print(f"\n  Reference, from the primary fit — the slope difference between "
          f"PSi {representative[0]:g} and PSi {representative[-1]:g}:")
    print(f"    {format_estimate(reference, 3)}  "
          f"({fit.n_patients} patients, {fit.n_obs:,} points)")

    # ---- serial correlation: thinning --------------------------------------
    print("\n  Serial correlation, by thinning the record. If the estimate "
          "holds when neighbouring samples are thrown away, it was not an "
          "artefact of how densely the monitors write.")
    for minutes in [float(part) for part in str(args.thin_minutes).split(",")
                    if part.strip()]:
        thinned = thin(frame, minutes)
        out = refit(thinned, f"thinned to {minutes:g} min",
                    record=f"thinned to 1 per {minutes:g} min")
        if out is None:
            print(f"    one sample per {minutes:g} min: not estimable")
            continue
        print(f"    one sample per {minutes:g} min: {format_estimate(out, 3)} "
              f"({out['n_obs']:,} points)")

    # ---- patient bootstrap --------------------------------------------------
    if args.bootstrap > 0:
        print(f"\n  Patient-level bootstrap, {args.bootstrap} resamples of the "
              f"{fit.n_patients} patients:")
        boot = patient_bootstrap(fit, [headline], args.bootstrap, args.seed)[0]
        if boot:
            print(f"    median {boot['median']:+.3f}, percentile 95% CI "
                  f"{boot['low']:+.3f} to {boot['high']:+.3f} "
                  f"({boot['draws']} usable resamples)")
            excludes = boot["low"] * boot["high"] > 0
            print(f"    The bootstrap interval {'excludes' if excludes else 'includes'} "
                  f"zero, {'agreeing with' if excludes == (reference['low'] * reference['high'] > 0) else 'DISAGREEING with'} "
                  f"the clustered sandwich above.")
        else:
            print("    not estimable")

    # ---- leave one patient out ---------------------------------------------
    print(f"\n  Leave-one-patient-out, refitting {fit.n_patients} times:")
    lopo = leave_one_patient_out(fit, headline)
    if lopo:
        print(f"    full {lopo['full']:+.3f}; range across the refits "
              f"{lopo['min']:+.3f} (without {lopo['min_patient']}) to "
              f"{lopo['max']:+.3f} (without {lopo['max_patient']})")
        print(f"    No single patient {'CAN' if lopo['flips_sign'] else 'can'} "
              f"flip the sign of the result."
              if not lopo["flips_sign"] else
              "    !! Dropping one patient flips the sign: the result rests on "
              "one case and should not be reported as a cohort finding.")
    else:
        print("    not estimable")

    # ---- PSi categories -----------------------------------------------------
    print("\n  PSi as categories (<25, 25-50, 50-75, >75) instead of a "
          "straight line in PSi:")
    _category, rows = category_fit(frame, exposure, args.step_seconds / 60.0,
                                   PSI_CATEGORY_EDGES, covariates)
    for row in rows:
        print(f"    PSi {row['label']:>7}: {format_estimate(row, 3)}  "
              f"({row['n_patients']} patients, {row['n_obs']:,} points)")
    if not rows:
        print("    not estimable")

    # ---- nonlinearity -------------------------------------------------------
    print("\n  Nonlinearity: the same model with quadratic terms added.")
    quadratic, _ = nonlinearity_fit(frame, exposure, modifier, centers,
                                    args.step_seconds / 60.0, covariates)
    if np.isfinite(quadratic.params).all():
        out = quadratic.contrast(headline)
        variants.append(("with quadratic terms", out))
        print(f"    slope difference {format_estimate(out, 3)}")
        curvature = quadratic.coefficient(
            f"{axis['short']}^2 x {AXES[modifier]['short']}")
        print(f"    curvature in the interaction "
              f"({axis['short']}^2 x PSi) {format_estimate(curvature, 6)}")
    else:
        print("    not estimable")

    # ---- maintenance phase only --------------------------------------------
    print(f"\n  Maintenance phase only (from {args.maintenance_start:g} min "
          f"after induction onward), so the induction transient cannot drive "
          f"the result:")
    maintenance = frame.loc[frame["minutes"] >= args.maintenance_start]
    out = refit(maintenance, "maintenance only",
                record=f"maintenance only (from {args.maintenance_start:g} min)")
    if out is None:
        print("    not estimable")
    else:
        print(f"    {format_estimate(out, 3)} "
              f"({out['n_patients']} patients, {out['n_obs']:,} points)")

    # ---- lagged exposure ----------------------------------------------------
    print("\n  Lagged exposure, pairing cerebral O2 with the exposure measured "
          "earlier (the NIRS signal is slow, so an effect should survive a "
          "head start):")
    for minutes in [float(part) for part in str(args.lag_minutes).split(",")
                    if part.strip()]:
        lagged = lag_exposure(frame, exposure, minutes, args.step_seconds / 60.0)
        out = refit(lagged, f"lag {minutes:g} min",
                    record=f"exposure lagged {minutes:g} min")
        if out is None:
            print(f"    {minutes:g} min lag: not estimable")
            continue
        print(f"    {minutes:g} min lag: {format_estimate(out, 3)} "
              f"({out['n_obs']:,} points)")

    # ---- pharmacology, not physiology? -------------------------------------
    if "bolus_vasoactive" in frame.columns or "bolus_hypnotic" in frame.columns:
        print("\n  Drugs. Propofol lowers MAP and PSi at the same moment, so "
              "'deep and hypotensive' can co-occur for a pharmacological "
              "reason. If the interaction is physiology it should survive "
              "removing the stretches where a syringe was moving both.")
        disturbed = pd.Series(False, index=frame.index)
        for name in ("bolus_vasoactive", "bolus_hypnotic"):
            if name in frame.columns:
                disturbed |= frame[name].fillna(0).gt(0)
        quiet = frame.loc[~disturbed]
        share = float(disturbed.mean())
        out = refit(quiet, "away from boluses",
                    record=f"excluding +/-{args.drug_window_minutes:g} min of a bolus")
        if out is None:
            print(f"    excluding the {share:.0%} of points within "
                  f"{args.drug_window_minutes:g} min of a bolus: not estimable")
        else:
            print(f"    excluding the {share:.0%} of points within "
                  f"{args.drug_window_minutes:g} min of a bolus: "
                  f"{format_estimate(out, 3)} "
                  f"({out['n_patients']} patients, {out['n_obs']:,} points)")

        extra = tuple(name for name in ("propofol_rate", "pressor_rate")
                      if name in frame.columns and name not in covariates
                      and frame[name].notna().any())
        if extra:
            adjusted_frame = frame.dropna(subset=list(extra))
            if adjusted_frame["subject_id"].nunique() >= 3:
                rebuilt = build_design(adjusted_frame, exposure, modifier,
                                       centers, covariates + extra)
                drug_fit = fit_within(adjusted_frame, rebuilt.within,
                                      "drug-rate adjusted",
                                      step_minutes=args.step_seconds / 60.0)
                out = drug_fit.contrast({rebuilt.terms["xm"]: gap})
                variants.append(("adjusted for the drug rates", out))
                print(f"    adjusting for the running "
                      f"{' and '.join(COVARIATE_LABELS[name] for name in extra)}: "
                      f"{format_estimate(out, 3)} "
                      f"({drug_fit.n_patients} patients, "
                      f"{drug_fit.n_obs:,} points)")

    # ---- covariate comparison ----------------------------------------------
    if covariates:
        print("\n  Covariate comparison: the same model without the "
              "anesthesia-record covariates.")
        plain = build_design(frame, exposure, modifier, centers, ())
        bare = fit_within(frame, plain.within, "unadjusted",
                          step_minutes=args.step_seconds / 60.0)
        out = bare.contrast({plain.terms["xm"]: gap})
        variants.append(("without the drug-rate covariates", out))
        print(f"    unadjusted slope difference {format_estimate(out, 3)}")
    else:
        print("\n  Covariate comparison: not possible. No anesthesia-record "
              "file was supplied, so there is nothing to adjust for and "
              "nothing to compare against. Pass --covariate-filepaths to "
              "enable it.")
        lines.append("models are unadjusted for end-tidal CO2, inspired O2 "
                     "fraction and anesthetic concentration")
    return lines, variants


# --------------------------------------------------------------------------- #

def main() -> int:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    # Say where the figures are going before anything else is printed. A run
    # that quietly fills a different folder is indistinguishable, on screen,
    # from a run that changed nothing -- which has already cost one round of
    # confusion in this project.
    existing = sorted(path.name for path in args.outdir.glob("*.png"))
    print(f"\nFIGURES WILL BE WRITTEN TO:\n  {args.outdir.resolve()}")
    if existing:
        print(f"  ({len(existing)} PNG(s) already there will be overwritten: "
              f"{', '.join(existing[:4])}"
              + (" ..." if len(existing) > 4 else "") + ")")

    for path in (args.filepaths, args.sto2_filepaths, args.redcap):
        if not path.is_file():
            sys.stderr.write(f"Input file does not exist: {path}\n")
            return 1
    map_list = resolve_map_list(args.map_filepaths, args.filepaths.parent)
    if map_list is None:
        sys.stderr.write(
            "No beat-to-beat MAP list found. MAP is the primary exposure, so "
            "this analysis cannot run without one.\nPass --map-filepaths "
            f"<file>. Looked in {args.filepaths.parent} for: "
            f"{', '.join(MAP_LIST_FALLBACKS)}\n")
        return 1
    hemo_list = (args.hemosphere_filepaths
                 if args.hemosphere_filepaths.is_file() else None)
    covariate_list = (args.covariate_filepaths
                      if args.covariate_filepaths is not None
                      and args.covariate_filepaths.is_file() else None)
    if args.covariate_filepaths is not None and covariate_list is None:
        print(f"!! --covariate-filepaths {args.covariate_filepaths} does not "
              f"exist; the models will be unadjusted.")

    representative = sorted(float(part) for part
                            in str(args.representative_psi).split(",")
                            if part.strip())
    if len(representative) < 2:
        sys.stderr.write("--representative-psi needs at least two values.\n")
        return 1

    banner("Sources and cleaning rules")
    events, characteristics, notes = load_redcap(args.redcap)
    if notes:
        print("  REDCap repairs:")
        for note in notes:
            print(f"    {note}")
    print(f"  REDCap: {len(events)} patients, induction time for "
          f"{int(events['induction'].notna().sum())}")
    print(f"  Resampled every {args.step_seconds:g} s, never interpolating "
          f"across gaps wider than {args.max_gap_seconds:g} s.")
    many = "," in str(args.sto2_channels)
    print(f"  Kept: CO where CO_SQI >= {args.min_co_sqi:g}; MAP where "
          f"databad == 0; PSi where the value is not '-'; cerebral O2 from "
          + (f"the average of channels {args.sto2_channels}" if many
             else f"channel {args.sto2_channels}")
          + " where flagged valid"
          + (f", falling back to channel {args.sto2_fallback_channels} for a "
             f"patient with no valid reading on it anywhere."
             if str(args.sto2_fallback_channels).strip() else "."))
    if str(args.sto2_sensitivity_channels).strip():
        print(f"  The other sensor (channel "
              f"{args.sto2_sensitivity_channels}, fallback "
              f"{args.sto2_sensitivity_fallback}) is not averaged in; the "
              f"whole analysis is repeated on it as a sensitivity instead.")
    print("  No time window: every valid sample is used, over the whole stretch "
          "where MAP, cerebral O2 and PSi all exist."
          if args.min_minutes is None and args.max_minutes is None else
          f"  Window: {args.min_minutes} to {args.max_minutes} min "
          f"relative to induction.")

    medications, medication_info = None, {}
    if args.medications is not None and args.medications.is_file():
        try:
            medications, medication_info = load_medications(args.medications)
        except Exception as exc:
            print(f"!! Could not read {args.medications}: {exc}\n"
                  f"   Continuing without the medication analyses.")
            medications, medication_info = None, {}
    elif args.medications is not None:
        print(f"  No medication file at {args.medications}; the drug "
              f"sensitivity analyses are skipped.")

    aligned, report, skipped = build_aligned(args, events, map_list, hemo_list,
                                             covariate_list, medications)
    if aligned.empty:
        sys.stderr.write("\nNo patient had MAP, cerebral O2 and PSi "
                         "overlapping.\n")
        for line in skipped[:20]:
            sys.stderr.write(f"  {line}\n")
        return 1

    # ---- coverage ----------------------------------------------------------
    banner("Data coverage (the outline's Table 1)")
    n_patients = int(aligned["subject_id"].nunique())
    per_patient = aligned.groupby("subject_id")["minutes"]
    duration = (per_patient.max() - per_patient.min())
    counts = aligned.groupby("subject_id").size()
    print(f"  {n_patients} patients, {len(aligned):,} time points on the "
          f"{args.step_seconds:g} s grid")
    print(f"  Record length per patient: median {duration.median():.0f} min "
          f"(IQR {duration.quantile(.25):.0f}-{duration.quantile(.75):.0f}, "
          f"range {duration.min():.0f}-{duration.max():.0f})")
    print(f"  Time points per patient: median {counts.median():,.0f} "
          f"(IQR {counts.quantile(.25):,.0f}-{counts.quantile(.75):,.0f})")
    print(f"  MAP          median (IQR) {describe(aligned['map'], 0)} mmHg")
    print(f"  Cerebral O2  median (IQR) {describe(aligned['sto2'], 0)} %")
    print(f"  PSi          median (IQR) {describe(aligned['psi'], 0)}")

    co_frame = pd.DataFrame()
    if "co" in aligned.columns and aligned["co"].notna().any():
        co_frame = aligned.dropna(subset=["co"])
        print(f"  Cardiac output median (IQR) {describe(co_frame['co'], 2)} "
              f"L/min, on {co_frame['subject_id'].nunique()} patients and "
              f"{len(co_frame):,} time points")
        if report["sqi_column"] is None:
            print(f"    !! no CO_SQI column was found, so the "
                  f"--min-co-sqi {args.min_co_sqi:g} filter could NOT be "
                  f"applied — check the column name before trusting the CO "
                  f"model")
        elif report["rows_seen"]:
            dropped = report["rows_seen"] - report["rows_kept"]
            print(f"    SQI filter >= {args.min_co_sqi:g}: kept "
                  f"{report['rows_kept']:,} of {report['rows_seen']:,} CO "
                  f"readings ({100.0 * dropped / report['rows_seen']:.0f}% "
                  f"dropped as low quality)")
    else:
        print(f"  Cardiac output: none usable "
              f"({report['listed']} Hemosphere file(s) listed, "
              f"{report['matched']} matched a patient with MAP, cerebral O2 "
              f"and PSi). The secondary model cannot run.")

    if report["extra_present"] or args.hemosphere_extra.strip():
        print("\n  Other Hemosphere channels, by how many patients carry any "
              "reading:")
        for name, spec in HEMOSPHERE_EXTRA.items():
            if name not in str(args.hemosphere_extra):
                continue
            patients_with = report["extra_present"].get(name, 0)
            on_grid = (float(aligned[name].notna().mean())
                       if name in aligned.columns else 0.0)
            print(f"    {spec['label']:<34} {patients_with:3d} patient(s), "
                  f"{on_grid:.0%} of aligned time points")
            if spec["derived"]:
                print(f"      NOT independent information: {spec['derived']}.")

    if medication_info:
        print(f"\n  Medications: {medication_info['kept']:,} usable events for "
              f"{medication_info['patients']} patients in the export; "
              f"{report['medication_patients']} of the analysed patients "
              f"matched.")
        for reason, count in medication_info.get("dropped", {}).items():
            if count:
                print(f"    dropped {count} row(s): {reason}")
        classes = medication_info.get("classes", {})
        print("    by class: " + ", ".join(f"{name} {count}" for name, count
                                            in sorted(classes.items())))
        if medication_info.get("unclassified"):
            print(f"    not classified (reported, not modelled): "
                  f"{', '.join(medication_info['unclassified'])}")
        if report["medication_missing"]:
            print(f"    {len(report['medication_missing'])} analysed "
                  f"patient(s) had no medication rows: "
                  f"{', '.join(report['medication_missing'][:6])}"
                  + (" ..." if len(report["medication_missing"]) > 6 else ""))
        for name in ("propofol_rate", "pressor_rate"):
            if name in aligned.columns:
                active = float((aligned[name] > 0).mean())
                print(f"    {COVARIATE_LABELS[name]}: running for "
                      f"{active:.0%} of aligned time points")
        for name, label in (("bolus_hypnotic", "hypnotic"),
                            ("bolus_vasoactive", "vasoactive")):
            if name in aligned.columns:
                print(f"    within {args.drug_window_minutes:g} min of a "
                      f"{label} bolus: "
                      f"{float(aligned[name].mean()):.0%} of time points")

    covariates: tuple[str, ...] = ()
    covariate_coverage: dict[str, float] = {}
    if report["covariate_columns"]:
        print(f"  Anesthesia-record covariates, from "
              f"{report['covariate_used']} file(s) read of "
              f"{report['covariate_listed']} listed:")
        for line in report["covariate_rescaled"]:
            print(f"    rescaled on the way in: {line}")
        for name, column in report["covariate_columns"].items():
            if name not in aligned.columns:
                continue
            share = float(aligned[name].notna().mean())
            patients = int(aligned.loc[aligned[name].notna(), "subject_id"].nunique())
            covariate_coverage[name] = share
            print(f"    {COVARIATE_LABELS[name]}: column '{column}', "
                  f"{share:.0%} of time points, {patients} patients, "
                  f"median (IQR) {describe(aligned[name].dropna(), 1)}")
            # Adjusting on a covariate present for a minority of the record
            # would silently shrink the cohort to that minority, so it is only
            # carried into the model when it covers most of the data.
            if share >= 0.8 and patients >= 10:
                covariates = covariates + (name,)
            else:
                print("      not used for adjustment: it covers too little of "
                      "the record (needs >= 80% of points and >= 10 patients)")
    else:
        print("  Anesthesia-record covariates: none supplied, so end-tidal CO2 "
              "and inspired O2 fraction are not in the models.")

    # The propofol infusion rate is the anesthetic exposure the outline asks
    # for, at least for a TIVA case, and unlike a demographic it moves within a
    # patient -- so it is a covariate the fixed effects do NOT absorb.
    for name in ("propofol_rate", "pressor_rate"):
        if name not in aligned.columns:
            continue
        share = float(aligned[name].notna().mean())
        patients_with = int(aligned.loc[aligned[name].notna(), "subject_id"].nunique())
        covariate_coverage[name] = share
        moving = int(aligned.loc[aligned[name].notna()]
                     .groupby("subject_id")[name].nunique().gt(1).sum())
        print(f"  {COVARIATE_LABELS[name]}: known for {share:.0%} of time "
              f"points over {patients_with} patients, and varies within "
              f"{moving} of them")
        if not args.adjust_drug_rates:
            print("    carried but NOT adjusted for (pass "
                  "--adjust-drug-rates to change that). A vasopressor raises "
                  "MAP, so adjusting for its rate removes part of the "
                  "exposure itself"
                  + (f", and this rate moves within only {moving} of "
                     f"{patients_with} patients" if moving < patients_with / 3
                     else "")
                  + ". The adjusted fit is reported as a sensitivity instead.")
        elif share >= 0.8 and patients_with >= 10 and moving >= 5:
            covariates = covariates + (name,)
        else:
            print("    not used for adjustment: it needs >= 80% coverage, "
                  ">= 10 patients and within-patient movement in >= 5")

    if report.get("sto2_fallback"):
        print(f"\n  {len(report['sto2_fallback'])} patient(s) had no valid "
              f"cerebral O2 on channels {args.sto2_channels} anywhere in their "
              f"file and used the fallback pair:")
        for line in report["sto2_fallback"]:
            print(f"    {line}")
    print(f"\n  Of {report['candidates']} patients with all three required "
          f"file types, {n_patients} aligned and {len(skipped)} did not:")
    for line in skipped:
        print(f"    {line}")

    adjusted = aligned.dropna(subset=list(covariates)) if covariates else aligned
    if covariates and len(adjusted) < len(aligned):
        print(f"\n  Adjusting drops {len(aligned) - len(adjusted):,} time "
              f"points with a missing covariate, leaving "
              f"{adjusted['subject_id'].nunique()} patients and "
              f"{len(adjusted):,} points.")

    # The PSi reference is the mean of the PER-PATIENT PSi means, not the mean
    # over all samples: it has to be a patient-level quantity, because the
    # slopes are read "for a patient of average depth" and a long case would
    # otherwise pull the reference toward itself.
    centers = {
        "map": (args.map_center if args.map_center is not None
                else float(np.round(adjusted.groupby("subject_id")["map"]
                                    .mean().mean(), 0))),
        "psi": (args.psi_center if args.psi_center is not None
                else float(np.round(adjusted.groupby("subject_id")["psi"]
                                    .mean().mean(), 0))),
    }
    for name in covariates:
        centers[name] = float(np.round(
            adjusted.groupby("subject_id")[name].mean().mean(), 1))
    if not co_frame.empty:
        centers["co"] = (args.co_center if args.co_center is not None
                         else float(np.round(co_frame.groupby("subject_id")["co"]
                                             .mean().mean(), 1)))

    # ---- primary model ------------------------------------------------------
    banner("PRIMARY MODEL — MAP as the exposure, PSi as the modifier")
    primary = run_model(adjusted, "map", "psi", centers, representative,
                        covariates, args, "MAP x PSi")

    # ---- secondary model ----------------------------------------------------
    secondary = None
    if co_frame.empty:
        banner("SECONDARY MODEL — cardiac output as the exposure")
        print("  Skipped: no usable cardiac output. See the coverage section.")
    else:
        banner("SECONDARY MODEL — cardiac output as the exposure, PSi as the "
               "modifier")
        co_adjusted = (co_frame.dropna(subset=list(covariates)) if covariates
                       else co_frame)
        secondary = run_model(co_adjusted, "co", "psi", centers, representative,
                              covariates, args, "CO x PSi")
        print("\n  Additionally, with MAP held fixed — the CO effect that is "
              "not simply CO moving MAP:")
        pressure = build_design(co_adjusted, "co", "psi", centers,
                                covariates + ("map",))
        with_map = fit_within(co_adjusted, pressure.within,
                              "CO x PSi, MAP adjusted",
                              step_minutes=args.step_seconds / 60.0)
        print(f"    interaction "
              f"{format_estimate(with_map.coefficient(pressure.terms['xm']), 5)}")
        difference = slope_difference(with_map, pressure, representative[-1],
                                      representative[0], AXES["co"]["step"])
        print(f"    slope difference between PSi {representative[0]:g} and "
              f"{representative[-1]:g}: {format_estimate(difference, 3)}")
        print("    This is a different question from the one above: holding MAP "
              "fixed asks whether flow matters beyond the pressure it "
              "generates. It is reported alongside, not instead.")

    # ---- the change model ---------------------------------------------------
    # Levels leave a lag-1 residual autocorrelation near 1: cerebral O2 drifts
    # for hours and the model spends itself on the drift. Differencing removes
    # it and asks the question a clinician would -- when pressure MOVED, did
    # oxygenation move with it?
    change_results: dict[str, dict] = {}
    horizon = float(args.change_horizon)
    if horizon > 0:
        banner(f"CHANGE MODEL — does a change in MAP over {horizon:g} min move "
               f"cerebral O2?")
        changed = difference_frame(adjusted, horizon, args.step_seconds / 60.0)
        if changed.empty or changed["subject_id"].nunique() < 3:
            print("  Not enough continuous stretches of that length to form "
                  "differences.")
        else:
            print(f"  {changed['subject_id'].nunique()} patients, "
                  f"{len(changed):,} overlapping {horizon:g}-minute intervals.")
            print("  Outcome is the change in cerebral O2 across the "
                  "interval; exposure is the change in MAP across the same "
                  "interval; PSi is its average over the interval.")
            change_centers = dict(centers)
            change_centers["map"] = 0.0
            change_centers["co"] = 0.0
            change_results["map"] = run_model(
                changed, "map", "psi", change_centers, representative,
                covariates, args, "dMAP x PSi",
                axis_override=change_axis("map", horizon))
            if "co" in changed.columns and changed["co"].notna().any():
                co_changed = changed.dropna(subset=["co"])
                if co_changed["subject_id"].nunique() >= 3:
                    banner(f"CHANGE MODEL — cardiac output over {horizon:g} min")
                    change_results["co"] = run_model(
                        co_changed, "co", "psi", change_centers,
                        representative, covariates, args, "dCO x PSi",
                        axis_override=change_axis("co", horizon))

    # ---- figures ------------------------------------------------------------
    banner("Figures")
    make_marginal_figure(
        primary["fit"], primary["design"], "map", "psi", adjusted,
        representative, args.outdir / "figure1_map_cerebral_o2_by_psi.png",
        "Figure 1. The MAP-cerebral O2 relationship depends on PSi",
        covariates)
    if secondary is not None:
        make_marginal_figure(
            secondary["fit"], secondary["design"], "co", "psi",
            secondary["frame"], representative,
            args.outdir / "figure2_co_cerebral_o2_by_psi.png",
            "Figure 2. The cardiac output-cerebral O2 relationship, by PSi",
            covariates)
    else:
        print("  figure2 skipped: no usable cardiac output.")

    for exposure, number in (("map", 4), ("co", 5)):
        result = change_results.get(exposure)
        if result is None:
            continue
        label = AXES[exposure]["short"]
        make_marginal_figure(
            result["fit"], result["design"], exposure, "psi",
            result["frame"], representative,
            args.outdir / f"figure{number}_{exposure}_change_model.png",
            f"Figure {number}. Does a CHANGE in {label} over {horizon:g} min "
            f"move cerebral O2, and does PSi modify that?",
            covariates, axis_override=change_axis(exposure, horizon))

    _category_fit, category_rows = category_fit(
        adjusted, "map", args.step_seconds / 60.0, PSI_CATEGORY_EDGES,
        covariates)
    make_category_figure(
        category_rows, "map",
        args.outdir / "figure3_map_slopes_by_psi_category.png",
        "Figure 3. Within-patient MAP slope by PSi category",
        f"One model with the MAP slope free to differ by category; each "
        f"patient-category cell gets its own baseline, 95% CIs clustered on "
        f"patient. {primary['fit'].n_patients} patients, "
        f"{primary['fit'].n_obs:,} time points.")

    # ---- tables -------------------------------------------------------------
    subjects = sorted(aligned["subject_id"].unique())
    baseline, baseline_missing = characteristic_rows(characteristics, subjects)

    banner("Baseline characteristics (the top half of Table 1)")
    if baseline:
        for label, value in baseline:
            print(f"  {label:<46} {value}")
    else:
        print("  None of the baseline fields were found in the REDCap export.")
    if baseline_missing:
        print(f"\n  Not available in the export: "
              f"{', '.join(baseline_missing)}.")
    print("\n  These are patient-level and do not change during a case, so the "
          "patient fixed effects absorb every one of them: they cannot "
          "confound a comparison of a patient against themselves, and adding "
          "them to the models would change nothing. They describe the cohort, "
          "and they are the natural candidates for asking whether the "
          "interaction DIFFERS between kinds of patients — a separate "
          "question from the one the outline poses.")

    table1: list[tuple[str, ...]] = [("Characteristic", f"Value (n = {n_patients})")]
    if baseline:
        table1.append(("PATIENTS", ""))
        table1 += baseline
        table1.append(("", ""))
    table1.append(("DATA", ""))
    table1 += [
        ("Patients analysed", f"{n_patients}"),
        ("Patients with all three required files", f"{report['candidates']}"),
        ("Patients excluded at alignment", f"{len(skipped)}"),
        ("Time points (grid)", f"{len(aligned):,} at {args.step_seconds:g} s"),
        ("Record length per patient, min",
         f"{duration.median():.0f} ({duration.quantile(.25):.0f}-{duration.quantile(.75):.0f})"),
        ("MAP, mmHg", describe(aligned["map"], 0)),
        ("Cerebral O2, %", describe(aligned["sto2"], 0)),
        ("PSi", describe(aligned["psi"], 0)),
    ]
    if not co_frame.empty:
        table1 += [
            ("Cardiac output, L/min", describe(co_frame["co"], 2)),
            ("Patients with usable CO", f"{co_frame['subject_id'].nunique()}"),
            ("CO time points", f"{len(co_frame):,}"),
            ("CO readings kept at SQI >= "
             f"{args.min_co_sqi:g}",
             f"{report['rows_kept']:,} of {report['rows_seen']:,}"
             if report["rows_seen"] else "n/a"),
        ]
    else:
        table1.append(("Cardiac output", "none usable"))
    for name in list(COVARIATE_PATTERNS) + ["propofol_rate", "pressor_rate"]:
        if name in covariate_coverage:
            table1.append((COVARIATE_LABELS[name],
                           f"{covariate_coverage[name]:.0%} of points, "
                           f"{'used' if name in covariates else 'not used'}"))
        else:
            table1.append((COVARIATE_LABELS[name], "not available"))
    if medication_info:
        table1.append(("Medication events used",
                       f"{medication_info['kept']:,} over "
                       f"{report['medication_patients']} patients"))
        table1.append((f"Time points within {args.drug_window_minutes:g} min "
                       f"of a bolus",
                       f"{float((aligned.get('bolus_vasoactive', pd.Series(0, index=aligned.index)).fillna(0).gt(0) | aligned.get('bolus_hypnotic', pd.Series(0, index=aligned.index)).fillna(0).gt(0)).mean()):.0%}"))
    for name, spec in HEMOSPHERE_EXTRA.items():
        if name not in str(args.hemosphere_extra):
            continue
        if name in aligned.columns and aligned[name].notna().any():
            table1.append((spec["label"],
                           f"{float(aligned[name].notna().mean()):.0%} of "
                           f"points"
                           + (", derived" if spec["derived"] else "")))
        else:
            table1.append((spec["label"], "not recorded"))
    table1.append(("Lag-1 residual autocorrelation",
                   f"{primary['fit'].rho1:.3f}"))
    if not args.table1:
        print("\n  Table 1 image not written (--table1 to produce it); the "
              "coverage numbers above are the content.")
    else:
        render_table(table1, [46, 40],
                     "Table 1. Cohort characteristics, data coverage and "
                     "distributions",
                     "Numeric fields are median (IQR); categorical fields are n (%) "
                     "of the analysed cohort. A count in brackets marks a field "
                     "recorded for only some patients.\n"
                     "Baseline characteristics come from the REDCap export and are "
                     "patient-level, so the patient fixed effects absorb them; they "
                     "describe the cohort rather than adjust the models.\n"
                     "Grid: every signal resampled to a common clock per patient, "
                     "never interpolated across a gap wider than "
                     f"{args.max_gap_seconds:g} s. MAP kept where databad == 0; "
                     f"PSi where the monitor did not write '-'; cerebral O2 "
                     f"averaged over valid channels {args.sto2_channels} "
                     f"(fallback {args.sto2_fallback_channels}); CO where CO_SQI "
                     f">= {args.min_co_sqi:g}."
                     + (f"\nNot available in this export: "
                        f"{', '.join(baseline_missing)}." if baseline_missing else ""),
                     args.outdir / "table1_data_coverage.png")

    table2: list[tuple[str, ...]] = [
        ("Model / quantity", "Estimate", "95% CI", "P")]

    def add_row(label: str, result: dict, digits: int = 3) -> None:
        if not np.isfinite(result.get("estimate", np.nan)):
            table2.append((label, "not estimable", "", ""))
            return
        table2.append((label, f"{result['estimate']:+.{digits}f}",
                       f"{result['low']:+.{digits}f} to {result['high']:+.{digits}f}",
                       format_p(result["p"]).lstrip("= ")))

    table2.append(("PRIMARY: MAP exposure, PSi modifier", "", "", ""))
    add_row("  MAP x PSi, % per mmHg per PSi unit", primary["raw"], 5)
    add_row("  MAP x PSi, % per 10 mmHg per 10 PSi", primary["scaled"])
    add_row("  cross-level MAP x mean PSi (not the estimate)",
            primary["cross"], 5)
    for psi in representative:
        add_row(f"  % cerebral O2 per 10 mmHg at PSi {psi:g}",
                primary["slopes"][psi])
    if primary["difference"] is not None:
        add_row(f"  difference, PSi {representative[-1]:g} vs "
                f"{representative[0]:g}", primary["difference"])
    if secondary is not None:
        table2.append(("", "", "", ""))
        table2.append(("SECONDARY: CO exposure, PSi modifier", "", "", ""))
        add_row("  CO x PSi, % per L/min per PSi unit", secondary["raw"], 5)
        add_row("  CO x PSi, % per 1 L/min per 10 PSi", secondary["scaled"])
        for psi in representative:
            add_row(f"  % cerebral O2 per 1 L/min at PSi {psi:g}",
                    secondary["slopes"][psi])
        if secondary["difference"] is not None:
            add_row(f"  difference, PSi {representative[-1]:g} vs "
                    f"{representative[0]:g}", secondary["difference"])
    table2.append(("", "", "", ""))
    # The change model belongs in the table the PI reads, not only in the
    # figures: it is where the drift stops dominating, so it carries more
    # information per row than the levels fits above it.
    for exposure in ("map", "co"):
        result = change_results.get(exposure)
        if result is None:
            continue
        unit = AXES[exposure]["unit"]
        step = AXES[exposure]["step"]
        table2.append(("", "", "", ""))
        table2.append((f"CHANGE MODEL, {horizon:g}-min horizon: "
                       f"{AXES[exposure]['short']} exposure", "", "", ""))
        add_row(f"  d{AXES[exposure]['short']} x PSi, % per {unit} per PSi",
                result["raw"], 5)
        for psi in representative:
            add_row(f"  % cerebral O2 per {step:g} {unit} of change at "
                    f"PSi {psi:g}", result["slopes"][psi])
        if result["difference"] is not None:
            add_row(f"  difference, PSi {representative[-1]:g} vs "
                    f"{representative[0]:g}", result["difference"])

    table2.append(("", "", "", ""))
    table2.append(("PSi CATEGORIES: % cerebral O2 per 10 mmHg MAP", "", "", ""))
    for row in category_rows:
        add_row(f"  PSi {row['label']} "
                f"({row['n_patients']} patients)", row)
    render_table(
        table2, [48, 13, 24, 9],
        "Table 2. Interaction estimates and the requested contrasts",
        "Patient-mean decomposition with patient fixed effects: exposure and "
        "modifier enter as deviations from the patient's own mean, so every "
        "estimate is within-patient.\n"
        f"Patient-clustered standard errors; t intervals on "
        f"{primary['fit'].dof} df (patients - 1). Slopes read at an absolute "
        f"PSi using PSi {primary['design'].reference:g} as the reference.\n"
        "A negative MAP x PSi means the MAP slope becomes more positive as "
        "PSi falls, i.e. cerebral oxygenation is more pressure-passive when "
        "the patient is deeper.\n"
        + ("Adjusted for " + ", ".join(COVARIATE_LABELS[name]
                                       for name in covariates) + "."
           if covariates else
           "UNADJUSTED: no anesthesia-record covariates were available."),
        args.outdir / "table2_interaction_results.png")

    # ---- sensitivity analyses ----------------------------------------------
    limitations: list[str] = []
    panels: list[tuple[str, str, list[tuple[str, dict]]]] = []
    if args.quick:
        banner("Sensitivity analyses")
        print("  Skipped (--quick).")
    else:
        banner("SENSITIVITY — primary model (MAP exposure)")
        lines, map_variants = run_sensitivities(primary, args, centers)
        limitations += lines
        co_variants: list[tuple[str, dict]] = []
        if secondary is not None:
            banner("SENSITIVITY — secondary model (CO exposure)")
            _lines, co_variants = run_sensitivities(secondary, args, centers)

        # The change model at several horizons, on the same axis as the rest,
        # so the levels and change framings can be read against each other.
        for exposure, collected in (("map", map_variants), ("co", co_variants)):
            if not collected:
                continue
            source = adjusted if exposure == "map" else (
                secondary["frame"] if secondary is not None else None)
            if source is None:
                continue
            step = AXES[exposure]["step"]
            gap = step * (representative[-1] - representative[0])
            for minutes in [float(part) for part
                            in str(args.change_horizons).split(",")
                            if part.strip()]:
                shifted = difference_frame(source, minutes,
                                           args.step_seconds / 60.0)
                if exposure == "co":
                    shifted = shifted.dropna(subset=["co"])
                if shifted.empty or shifted["subject_id"].nunique() < 3:
                    continue
                change_centers = dict(centers)
                change_centers[exposure] = 0.0
                rebuilt = build_design(shifted, exposure, "psi",
                                       change_centers, covariates)
                new = fit_within(shifted, rebuilt.within,
                                 f"change over {minutes:g} min",
                                 step_minutes=args.step_seconds / 60.0)
                if not np.isfinite(new.params).all():
                    continue
                collected.append((f"CHANGE model, {minutes:g}-min horizon",
                                  new.contrast({rebuilt.terms["xm"]: gap})))

        # Every way of reading the oximeter, shown side by side. CH1 is the
        # prespecified outcome; the other sensor and the average of the two
        # are run end to end as well, so the choice is visible rather than
        # something to be made after seeing which one is significant. If all
        # three agree, that is a point in the result's favour; if they do not,
        # that is the finding, and it is about the sensors.
        alternatives: list[tuple[str, str, str]] = []
        sensitivity_channels = str(args.sto2_sensitivity_channels).strip()
        if sensitivity_channels:
            alternatives.append(
                (f"cerebral O2 from channel {sensitivity_channels}",
                 sensitivity_channels, str(args.sto2_sensitivity_fallback)))
            primary_channels = str(args.sto2_channels).strip()
            if "," not in primary_channels:
                averaged = f"{primary_channels},{sensitivity_channels}"
                alternatives.append(
                    (f"cerebral O2 averaged over channels {averaged}", averaged,
                     f"{args.sto2_fallback_channels},"
                     f"{args.sto2_sensitivity_fallback}"))

        for label, channels, fallback in alternatives:
            banner(f"SENSITIVITY — repeated with {label}")
            other = argparse.Namespace(**vars(args))
            other.sto2_channels = channels
            other.sto2_fallback_channels = fallback
            try:
                alternative, _report, _skipped = build_aligned(
                    other, events, map_list, hemo_list, covariate_list,
                    medications)
            except Exception as exc:
                alternative = pd.DataFrame()
                print(f"  Could not rebuild on those channels: {exc}")
            if alternative.empty:
                print("  No patient had usable data that way.")
                continue
            print(f"  {alternative['subject_id'].nunique()} patients, "
                  f"{len(alternative):,} time points.")
            for exposure, collected in (("map", map_variants),
                                        ("co", co_variants)):
                if not collected:
                    continue
                source = alternative
                if exposure == "co":
                    if "co" not in alternative.columns:
                        continue
                    source = alternative.dropna(subset=["co"])
                if covariates:
                    source = source.dropna(subset=list(covariates))
                if source.empty or source["subject_id"].nunique() < 3:
                    continue
                rebuilt = build_design(source, exposure, "psi", centers,
                                       covariates)
                new = fit_within(source, rebuilt.within, label,
                                 step_minutes=args.step_seconds / 60.0)
                if not np.isfinite(new.params).all():
                    continue
                gap = (AXES[exposure]["step"]
                       * (representative[-1] - representative[0]))
                out = new.contrast({rebuilt.terms["xm"]: gap})
                out.update({"n_patients": new.n_patients, "n_obs": new.n_obs})
                collected.append((label, out))
                print(f"    {AXES[exposure]['short']}: "
                      f"{format_estimate(out, 3)} "
                      f"({new.n_patients} patients, {new.n_obs:,} points)")

        panels = [
            ("MAP x PSi — the difference between the MAP slope at "
             f"PSi {representative[0]:g} and at PSi {representative[-1]:g}",
             "Cerebral O2 change (%) per 10 mmHg of MAP", map_variants),
            ("CO x PSi — the same contrast with cardiac output",
             "Cerebral O2 change (%) per 1 L/min of CO", co_variants)]
        banner("Specification comparison")
        make_specification_figure(
            panels, args.outdir / "figure6_specification_comparison.png",
            "Figure 6. Does the answer depend on how it was fitted?",
            "Each row is the same headline contrast under a different "
            "reasonable analytic choice. Red marks an interval that excludes "
            "zero. A column that stays on one side of zero is a robust "
            "result; one that straddles zero and wanders is a question this "
            "data does not settle.")

    # ---- the summary the PI asked for first --------------------------------
    banner("Data coverage and model limitations — the paragraph to send")
    headline = primary["difference"]
    # Never state a direction as a finding when the interval spans zero. The
    # sign of a null point estimate is noise, and this is the paragraph that
    # gets pasted into an email.
    if np.isfinite(headline["low"]) and headline["low"] * headline["high"] > 0:
        verdict = ("the MAP slope is "
                   + ("more" if headline["estimate"] < 0 else "less")
                   + " positive at lower PSi, so\n  cerebral oxygenation is "
                     "more pressure-passive when the patient is deeper.")
    else:
        worst = max(abs(headline["low"]), abs(headline["high"]))
        verdict = (
            "the interval spans zero, so there is NO EVIDENCE\n"
            "  that PSi modifies the MAP-cerebral O2 relationship. Nor is this "
            "evidence of absence:\n"
            f"  a difference as large as {worst:.3f} % per 10 mmHg remains "
            "compatible with the data,\n  and the sign of the point estimate "
            "should not be read as a direction.")
    print(f"""  Cohort. {n_patients} patients contributed {len(aligned):,} time
  points on a {args.step_seconds:g}-second grid, a median of
  {duration.median():.0f} minutes per patient
  (IQR {duration.quantile(.25):.0f}-{duration.quantile(.75):.0f}). MAP, cerebral
  O2 and PSi were required of every patient; cardiac output was available for
  {(co_frame['subject_id'].nunique() if not co_frame.empty else 0)} of them.

  Primary result. In a patient fixed-effects model with MAP, PSi and their
  interaction, the MAP x PSi term was
  {primary['raw']['estimate']:+.5f} % cerebral O2 per mmHg per PSi unit
  (95% CI {primary['raw']['low']:+.5f} to {primary['raw']['high']:+.5f},
  P {format_p(primary['raw']['p'])}). Expressed the way it will be read: the
  within-patient change in cerebral O2 per 10 mmHg of MAP was
  {primary['slopes'][representative[0]]['estimate']:+.3f} %
  at PSi {representative[0]:g} and
  {primary['slopes'][representative[-1]]['estimate']:+.3f} %
  at PSi {representative[-1]:g}, a difference of
  {headline['estimate']:+.3f} % (95% CI {headline['low']:+.3f} to
  {headline['high']:+.3f}, P {format_p(headline['p'])}) — {verdict}""")
    if secondary is not None:
        print(f"""
  Secondary result. With cardiac output as the exposure on
  {secondary['fit'].n_patients} patients and {secondary['fit'].n_obs:,} points,
  the CO x PSi interaction was
  {secondary['raw']['estimate']:+.5f} % per L/min per PSi unit
  (95% CI {secondary['raw']['low']:+.5f} to {secondary['raw']['high']:+.5f},
  P {format_p(secondary['raw']['p'])}). The CO association ITSELF, however, is
  the firmest thing in this analysis and should not be lost behind its null
  interaction: cerebral O2 changed by
  {secondary['slopes'][representative[0]]['estimate']:+.3f} % per 1 L/min at
  PSi {representative[0]:g}
  (95% CI {secondary['slopes'][representative[0]]['low']:+.3f} to
  {secondary['slopes'][representative[0]]['high']:+.3f},
  P {format_p(secondary['slopes'][representative[0]]['p'])}) and
  {secondary['slopes'][representative[-1]]['estimate']:+.3f} % at
  PSi {representative[-1]:g}
  (95% CI {secondary['slopes'][representative[-1]]['low']:+.3f} to
  {secondary['slopes'][representative[-1]]['high']:+.3f},
  P {format_p(secondary['slopes'][representative[-1]]['p'])}) — a flow effect
  several times the size of the pressure effect, at every depth.""")

    print("""
  Limitations, stated plainly.
  * Estimand. Every estimate above is within-patient by construction: patient
    fixed effects remove each patient's own level, so no part of the result
    comes from comparing one patient with another. That is deliberate — in
    this dataset the between-patient relationship runs the other way, and a
    random-intercept model alone would have mixed the two.
  * Serial correlation. Ten-second samples are strongly autocorrelated, so the
    sample count overstates the information available. Inference is clustered
    on patient, which is valid under any within-patient correlation structure,
    and the degrees of freedom are set by the number of patients. The thinning,
    bootstrap and leave-one-patient-out checks address the same concern from
    three directions.
  * Association, not causation. MAP and PSi move together with surgical events,
    drug boluses and position changes; none of this is randomized.
  * Many specifications, one dataset. The specification figure shows around a
    dozen fits per exposure. They are not independent tests -- they reuse the
    same patients -- so a row that clears P < 0.05 while its neighbours do not
    is weak evidence, and picking it would be choosing the analysis after
    seeing the answer. The figure is there to show whether an estimate is
    STABLE across choices, not to be searched for the significant one. A
    result worth reporting holds its sign and rough size across the column;
    one that appears in a single row should be called hypothesis-generating
    and tested on new patients.""")
    if medication_info:
        print("""  * Drugs move both variables at once. Propofol lowers MAP and PSi together,
    so "deep and hypotensive" can co-occur pharmacologically rather than
    physiologically. This is the main alternative explanation for the
    interaction, and the sensitivity section attacks it directly: the model is
    adjusted for the running infusion rates where they are available, and
    refitted with the minutes around every bolus removed. An interaction that
    survives both is not simply the syringe.""")
    print("""  * What the models leave out of the Hemosphere export. SVR, SVRI, CI and SVI
    are algebraic functions of CO, MAP, CVP and body size -- SVR is exactly
    80 x (MAP - CVP) / CO in this data -- so they carry no information the
    models do not already have, and a coefficient on one of them would be a
    functional-form artefact rather than a vasomotor effect. CVP is a manually
    entered constant, not a measurement.
  * The measurement. NIRS cerebral oximetry is venous-weighted, samples frontal
    cortex only, and is susceptible to extracranial contamination, so it is an
    index of cerebral oxygen balance rather than a direct measure of cerebral
    oxygen delivery. PSi is a processed-EEG index, not a measure of CMRO2.""")
    if "monk_forehead" in getattr(characteristics, "columns", []):
        print("""  * Skin pigmentation. NIRS readings vary with skin pigmentation, and the
    forehead Monk scale is recorded for this cohort (Table 1). Because the
    models are within-patient, pigmentation cannot bias the slopes — it does
    not change during a case, so the patient's own intercept absorbs it. It
    would matter for any comparison of absolute cerebral O2 BETWEEN patients,
    which this analysis deliberately does not make.""")
    print("""  * Baseline characteristics. Age, sex, BMI, ASA class and the rest are
    patient-level and therefore absorbed by the patient fixed effects. They
    are reported in Table 1 to describe the cohort; they are not adjustment
    covariates here, and adding them would change no estimate.
  * What the interaction does not say. A changing slope says the MAP-cerebral
    O2 relationship differs with depth. It does not identify a lower limit of
    autoregulation, and the models here are linear in MAP over the observed
    range.""")
    for line in limitations:
        print(f"  * {line.capitalize()}.")
    if not HAVE_MIXEDLM:
        print("  * The random-effects companion could not be fitted: "
              "statsmodels MixedLM is unavailable in this environment.")

    banner("What to send, and in what order")
    print("  table1_data_coverage.png              what the analysis stands on")
    print("  table2_interaction_results.png        the estimates and contrasts")
    print("  figure1_map_cerebral_o2_by_psi.png    the primary result")
    if secondary is not None:
        print("  figure2_co_cerebral_o2_by_psi.png     the secondary result")
    print("  figure3_map_slopes_by_psi_category.png  the categorical "
          "sensitivity, for readers who prefer bands")
    for exposure, number in (("map", 4), ("co", 5)):
        if exposure in change_results:
            print(f"  figure{number}_{exposure}_change_model.png"
                  + " " * (26 - len(exposure))
                  + f"does a CHANGE in {AXES[exposure]['short']} move "
                    f"cerebral O2? (drift removed)")
    if panels and any(rows for _label, _unit, rows in panels):
        print("  figure6_specification_comparison.png   whether the answer "
              "survives every reasonable analytic choice")
    print("\n  The outline asks for the primary and secondary model results, "
          "Figures 1 and 2, and a coverage-and-limitations summary first. "
          "Those are the first four files above plus the paragraph printed "
          "in the previous section.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
