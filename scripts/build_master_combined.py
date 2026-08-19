#!/usr/bin/env python3
"""
Build MASTER_combined.csv (long format) from the per-patient BTB MAP files and
Sedline PSI files, then it is ready for induction_map_psi_analysis.py.

This is the vectorized version of the row-by-row build loop: same output,
but pandas filters whole columns at once instead of looping over every row
(~100-1000x faster on 69 patients).

Inputs (edit CONFIG or pass flags):
  * BTB master list  : csv, one file path per line -> each is a patient's MAP
                       csv with columns incl. 'databad', 'meanArterialPressure',
                       'time'.
  * PSI master list  : csv, one file path per line -> each is a patient's
                       Sedline csv with columns incl. ' Time' (note the leading
                       space), 'PSi (Sedline) Value', 'SEFL Hz (Sedline) Value',
                       'SEFR Hz (Sedline) Value'. Missing values are '-'.

Patient ID = the 14 characters starting at 'IU' in each file path.

Output: one long csv with columns
    patientID, time, meanArterialPressure, PSi, SEFL_Hz, SEFR_Hz
MAP rows and PSi rows are separate (as in the exports); the analysis script
aligns them onto a common time grid, so that's fine.

Usage:
    python build_master_combined.py
    python build_master_combined.py --out /path/MASTER_combined.csv
    # then:
    python induction_map_psi_analysis.py --data /path/MASTER_combined.csv --events events.csv
"""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

# --------------------------- CONFIG (defaults) ----------------------------- #
BTB_FILEPATHS = "/N/project/Analgesia_BDproject/PR/scripts_PR/btb_filepaths.csv"
PSI_FILEPATHS = "/N/project/Analgesia_BDproject/PR/scripts_PR/sedline_filepaths.csv"
OUT_PATH = "/N/project/Analgesia_BDproject/PR/scripts_PR/MASTER_combined.csv"

INCLUDE_SEF = True  # also carry SEFL/SEFR (the plan uses them as depth later)


def read_filepath_list(master: str) -> list[str]:
    """One path per line; tolerate blanks/quotes/extra columns."""
    paths = []
    with open(master, "r") as fh:
        for raw in fh:
            first = raw.strip().split(",")[0].strip().strip('"').strip("'")
            if first:
                paths.append(first)
    return paths


def patient_id_from_path(path: str) -> str | None:
    """The 14 characters starting at 'IU' in the path, e.g. IUxxxxxxxxxxxx."""
    i = path.find("IU")
    if i == -1:
        return None
    return path[i:i + 14]


def load_btb(path: str, pid: str) -> pd.DataFrame:
    """One patient's MAP rows: keep databad == 0, numeric MAP, HH:MM:SS time."""
    df = pd.read_csv(path, skipinitialspace=True)
    df.columns = df.columns.str.strip()
    good = df[pd.to_numeric(df["databad"], errors="coerce") == 0].copy()
    out = pd.DataFrame({
        "patientID": pid,
        # keep only the clock part (last 8 chars of '... HH:MM:SS')
        "time": good["time"].astype(str).str[-8:],
        "meanArterialPressure": pd.to_numeric(good["meanArterialPressure"],
                                              errors="coerce"),
    })
    return out.dropna(subset=["meanArterialPressure"])


def load_sedline(path: str, pid: str) -> pd.DataFrame:
    """One patient's Sedline rows: numeric PSi (and SEFL/SEFR), '-' -> missing."""
    df = pd.read_csv(path, skipinitialspace=True)
    df.columns = df.columns.str.strip()  # fixes the ' Time' leading space
    out = pd.DataFrame({"patientID": pid, "time": df["Time"].astype(str).str[-8:]})
    out["PSi"] = pd.to_numeric(df["PSi (Sedline) Value"], errors="coerce")
    if INCLUDE_SEF:
        for src, dst in (("SEFL Hz (Sedline) Value", "SEFL_Hz"),
                         ("SEFR Hz (Sedline) Value", "SEFR_Hz")):
            if src in df.columns:
                out[dst] = pd.to_numeric(df[src], errors="coerce")
    # keep rows that have at least one real Sedline value
    value_cols = [c for c in out.columns if c not in ("patientID", "time")]
    return out.dropna(subset=value_cols, how="all")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Build the combined long MAP+PSI csv.")
    ap.add_argument("--btb", default=BTB_FILEPATHS)
    ap.add_argument("--psi", default=PSI_FILEPATHS)
    ap.add_argument("--out", default=OUT_PATH)
    args = ap.parse_args(argv)

    frames = []
    for master, loader, label in ((args.btb, load_btb, "BTB/MAP"),
                                  (args.psi, load_sedline, "Sedline/PSI")):
        paths = read_filepath_list(master)
        print(f"{label}: {len(paths)} file(s) listed in {master}")
        for p in paths:
            pid = patient_id_from_path(p)
            if pid is None:
                sys.stderr.write(f"[WARN] no 'IU' id in path, skipped: {p}\n")
                continue
            try:
                frames.append(loader(p, pid))
            except Exception as exc:
                sys.stderr.write(f"[WARN] {pid} ({label}): {exc} — skipped\n")

    if not frames:
        sys.stderr.write("Nothing loaded — check the master file lists.\n")
        return 1

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(["patientID", "time"]).reset_index(drop=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    combined.to_csv(args.out, index=False)

    n_map = combined["meanArterialPressure"].notna().sum()
    n_psi = combined["PSi"].notna().sum() if "PSi" in combined else 0
    print(f"Wrote {args.out}: {len(combined)} rows "
          f"({combined['patientID'].nunique()} patients, "
          f"{n_map} MAP values, {n_psi} PSi values)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
