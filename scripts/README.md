# Beat-to-beat MAP analysis (`b2b_map_analysis.py`)

Cleans and summarizes beat-to-beat **Mean Arterial Pressure (MAP)** data for
all patients, and lays the data out so it's ready to be merged with **PSI**
later on.

## What it does

Given a **master file** that lists one patient MAP csv per line, for each patient it:

1. Loads the patient's MAP csv.
2. Drops rows flagged as bad data (`databad == 1`).
3. Drops missing / physiologically-implausible MAP values (default 20–250 mmHg).
4. Converts the time column into **elapsed seconds from the start of the record**,
   so every patient shares one surgery-relative timeline.
5. Resamples onto a steady **2-second grid** (handles clock drift, duplicate
   timestamps, and multiple readings in the same bin). Only *short* gaps
   (≤ 4 s by default) are interpolated; longer gaps stay missing.
6. Computes per-patient metrics: mean / median / std / min / max / p05 / p95,
   plus **time spent below** and **area-under-threshold "dose"** for MAP < 65,
   60, and 55 mmHg.

## Input format assumed

Each patient csv has 5 columns. By position (0-based):

| index | meaning |
|-------|---------|
| 0 | (unused) |
| 1 | `databad` (1 = drop the whole row) |
| 2 | `meanArterialPressure` (every 2 s) |
| 3 | `time` |
| 4 | (unused) |

Header row optional. Time may be numeric seconds, an epoch, or a clock/date
string (e.g. `HH:MM:SS`) — all are converted to elapsed seconds. If your files
differ, edit the `CONFIG` block at the top of the script.

## Usage

```bash
# default master path (the cluster location)
python b2b_map_analysis.py

# or point it anywhere, and save per-patient plots
python b2b_map_analysis.py \
    --master /N/project/Analgesia_BDproject/PR/scripts_PR/b2b_filepath.csv \
    --outdir ./b2b_output \
    --plots

# quick test on just the first 3 patients
python b2b_map_analysis.py --limit 3
```

Requires `pandas`, `numpy`, and (only for `--plots`) `matplotlib`.

## Outputs (under `--outdir`, default `b2b_output/`)

```
b2b_output/
├── cleaned/
│   ├── <patient>_map_clean.csv     # patient_id, elapsed_sec, meanArterialPressure
│   └── ...
├── plots/                          # only with --plots
│   └── <patient>_map.png
├── patient_summary.csv             # one row per patient (all the metrics)
└── all_patients_map_long.csv       # every cleaned sample, stacked (tidy/long)
```

## How this sets up the PSI comparison (next step)

Because every patient is aligned to `elapsed_sec` on the same 2-second grid,
comparing against PSI later is a straight join. Once the PSI data is cleaned
onto the same `(patient_id, elapsed_sec)` grid, merge them:

```python
import pandas as pd
map_df = pd.read_csv("b2b_output/all_patients_map_long.csv")
psi_df = pd.read_csv("psi_output/all_patients_psi_long.csv")   # same layout
merged = map_df.merge(psi_df, on=["patient_id", "elapsed_sec"], how="inner")
```

That merged, timestamp-aligned table is what the group comparisons
(per-patient correlations, mixed-effects MAP↔PSI models, mean±spread trend
plots) will run on.
