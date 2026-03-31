"""
00_parse_gmd.py
================
Parse GMAT output files and prepare Python-readable inputs.

Author : Giuliano Pennacchio
Version: 2.0  (Opzione A — integrated GMAT → Python pipeline)

Role in the Pipeline
--------------------
This script is the bridge between GMAT (Stage 1) and the Python BLS
estimator (Stage 2).  It reads two files produced by GMAT:

  Batch_OD_Tracking.gmd      → data/observations.csv
  Batch_OD_GroundTruth.csv   → data/ground_truth.csv  (reformatted)

and writes two CSV files consumed by the downstream scripts:

  data/observations.csv
      time_s, station, range_km, rangerate_kms, elevation_deg (NaN)
      One row per measurement epoch (range and range-rate are paired
      by timestamp and station then written as a single row).

  data/ground_truth.csv
      time_s, x_km, y_km, z_km, vx_kms, vy_kms, vz_kms
      Time in elapsed seconds from the estimation epoch.

GMD File Format (GMAT R2025a, measurement types Range and RangeRate)
---------------------------------------------------------------------
Each non-comment line has the format:

    TAIMJD  MeasType  GS_ID  SC_ID  Value  [extra_fields...]

  Field 1: TAIMJD   — TAI Modified Julian Date of the observation
                       (receive epoch for two-way measurements)
  Field 2: MeasType — 'Range' or 'RangeRate'
  Field 3: GS_ID    — integer ID of the ground station (string in file)
                       Mapped to station name via GS_ID_MAP below
  Field 4: SC_ID    — integer ID of the spacecraft (not used in Python)
  Field 5: Value    — observable: km for Range, km/s for RangeRate

  Comment lines start with '%'.
  Blank lines are ignored.

Ground Truth CSV Format (GMAT R2025a ReportFile)
-------------------------------------------------
Header line:  Sat.A1ModJulian  Sat.X  Sat.Y  Sat.Z  Sat.VX  Sat.VY  Sat.VZ  ...
Data lines:   space-separated values (column width 20, precision 12)

  A1ModJulian: Modified Julian Date in A1 atomic time scale.
               Converted to elapsed seconds: t_s = (MJD - MJD_epoch) * 86400
               A1 - TAI ≈ 0 (A1 is the predecessor of TAI, offset < 0.1 s,
               negligible for our purposes).

Run
---
    python 00_parse_gmd.py [--gmd PATH] [--gt PATH] [--outdir PATH]

    Defaults:
      --gmd    ../gmat_output/Batch_OD_Tracking.gmd
      --gt     ../gmat_output/Batch_OD_GroundTruth.csv
      --outdir data/

Outputs
-------
    data/observations.csv
    data/ground_truth.csv
    data/parse_summary.txt
"""

import os
import re
import argparse
import numpy as np
import pandas as pd
from collections import defaultdict

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Map GMAT ground station IDs (as written to the .gmd file) to names.
# Must match the Id field set in the GMAT script for each GroundStation.
GS_ID_MAP = {
    "1001": "Kiruna",
    "1002": "Svalbard",
}

# Spacecraft ID (for validation only — not used in BLS)
SC_ID_EXPECTED = "2001"

# Reference epoch: 01 Jan 2024 12:00:00.000 UTC
# A1 MJD = 60310.5  (days since MJD epoch 17 Nov 1858 00:00:00)
# Computed as: datetime(2024,1,1,12,0,0) → JD 2460311.0 → MJD 60310.5
EPOCH_MJD_A1 = 60310.5   # A1 Modified Julian Date of the estimation epoch

# Default file paths (relative to this script's location)
_HERE        = os.path.dirname(os.path.abspath(__file__))
DEFAULT_GMD  = os.path.join(_HERE, "..", "gmat_output", "Batch_OD_Tracking.gmd")
DEFAULT_GT   = os.path.join(_HERE, "..", "gmat_output", "Batch_OD_GroundTruth.csv")
DEFAULT_OUT  = os.path.join(_HERE, "data")


# ===========================================================================
# GMD Parser
# ===========================================================================

def parse_gmd(gmd_path: str) -> pd.DataFrame:
    """
    Parse a GMAT Measurement Data (.gmd) file into a pandas DataFrame.

    The function reads Range and RangeRate records and pairs them by
    (timestamp, station) to produce one row per observation epoch.

    Parameters
    ----------
    gmd_path : str  Path to the .gmd file.

    Returns
    -------
    DataFrame with columns:
        taimjd       float  TAI MJD of the observation
        time_s       float  Elapsed seconds from EPOCH_MJD_A1
        station      str    Station name (from GS_ID_MAP)
        range_km     float  Noisy range [km]
        rangerate_kms float  Noisy range-rate [km/s]
    """
    if not os.path.exists(gmd_path):
        raise FileNotFoundError(
            f"GMD file not found: {gmd_path}\n"
            "  → Run the GMAT script first to generate Batch_OD_Tracking.gmd"
        )

    # Storage: keyed by (taimjd_str, gs_id) → {'Range': value, 'RangeRate': value}
    records = defaultdict(dict)
    n_lines = 0; n_range = 0; n_rr = 0; n_unknown = 0; n_skipped = 0

    with open(gmd_path, "r") as fh:
        for line in fh:
            line = line.strip()
            n_lines += 1

            # Skip comment and blank lines
            if not line or line.startswith("%"):
                continue

            # Split on whitespace
            parts = line.split()
            if len(parts) < 5:
                n_skipped += 1
                continue

            taimjd_str = parts[0]
            meas_type  = parts[1]
            gs_id      = parts[2]
            # sc_id    = parts[3]  # not used
            try:
                value = float(parts[4])
            except ValueError:
                n_skipped += 1
                continue

            # Only process known measurement types
            if meas_type not in ("Range", "RangeRate"):
                n_unknown += 1
                continue

            # Only process known stations
            if gs_id not in GS_ID_MAP:
                n_skipped += 1
                continue

            key = (taimjd_str, gs_id)
            records[key][meas_type] = value

            if meas_type == "Range":
                n_range += 1
            else:
                n_rr += 1

    print(f"  GMD file read: {n_lines} lines total")
    print(f"    Range records     : {n_range}")
    print(f"    RangeRate records : {n_rr}")
    if n_unknown:  print(f"    Unknown types     : {n_unknown}  (ignored)")
    if n_skipped:  print(f"    Skipped/malformed : {n_skipped}")

    # ── Pair Range + RangeRate by (timestamp, station) ───────────────────
    rows = []
    n_paired = 0; n_unpaired = 0
    for (taimjd_str, gs_id), meas_dict in records.items():
        if "Range" not in meas_dict or "RangeRate" not in meas_dict:
            n_unpaired += 1
            continue
        taimjd = float(taimjd_str)
        t_s    = (taimjd - EPOCH_MJD_A1) * 86400.0   # MJD → elapsed seconds
        rows.append({
            "taimjd":       taimjd,
            "time_s":       t_s,
            "station":      GS_ID_MAP[gs_id],
            "range_km":     meas_dict["Range"],
            "rangerate_kms": meas_dict["RangeRate"],
        })
        n_paired += 1

    if n_unpaired:
        print(f"    Unpaired records  : {n_unpaired}  (Range without RangeRate or vice-versa)")

    df = pd.DataFrame(rows).sort_values("time_s").reset_index(drop=True)
    return df


# ===========================================================================
# Ground Truth Parser
# ===========================================================================

def parse_ground_truth(gt_path: str) -> pd.DataFrame:
    """
    Parse the GMAT ReportFile ground truth CSV.

    GMAT writes space-delimited columns with a header line.  Column names
    contain the GMAT dot-notation (e.g., 'Sat.A1ModJulian').

    Parameters
    ----------
    gt_path : str  Path to the ground truth .csv file from GMAT.

    Returns
    -------
    DataFrame with columns:
        time_s  float  Elapsed seconds from EPOCH_MJD_A1
        x_km    float  ECI X position [km]
        y_km    float  ECI Y position [km]
        z_km    float  ECI Z position [km]
        vx_kms  float  ECI VX velocity [km/s]
        vy_kms  float  ECI VY velocity [km/s]
        vz_kms  float  ECI VZ velocity [km/s]
    """
    if not os.path.exists(gt_path):
        raise FileNotFoundError(
            f"Ground truth file not found: {gt_path}\n"
            "  → Run the GMAT script first to generate Batch_OD_GroundTruth.csv"
        )

    # GMAT ReportFile: first non-blank, non-comment line is the header.
    # Delimiter can be comma or space depending on GMAT version / settings.
    # We try comma first, then whitespace.
    raw = pd.read_csv(gt_path, sep=r'\s+', comment='%',
                      header=0, engine='python')

    # Normalise column names: strip whitespace, lower-case suffix after last '.'
    cols = []
    for c in raw.columns:
        c = c.strip()
        # GMAT names like 'Sat.A1ModJulian', 'Sat.X', etc.
        suffix = c.split(".")[-1].lower()
        cols.append(suffix)
    raw.columns = cols

    # Convert A1 MJD → elapsed seconds
    raw["time_s"] = (raw["a1modJulian".lower()] - EPOCH_MJD_A1) * 86400.0

    # Build clean output dataframe
    df = pd.DataFrame({
        "time_s": raw["time_s"],
        "x_km":   raw["x"],
        "y_km":   raw["y"],
        "z_km":   raw["z"],
        "vx_kms": raw["vx"],
        "vy_kms": raw["vy"],
        "vz_kms": raw["vz"],
    }).sort_values("time_s").reset_index(drop=True)

    return df


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Parse GMAT outputs for the Python BLS pipeline."
    )
    parser.add_argument("--gmd",    default=DEFAULT_GMD,
                        help="Path to Batch_OD_Tracking.gmd")
    parser.add_argument("--gt",     default=DEFAULT_GT,
                        help="Path to Batch_OD_GroundTruth.csv")
    parser.add_argument("--outdir", default=DEFAULT_OUT,
                        help="Output directory for parsed CSV files")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print("=" * 60)
    print("Project 2 — GMAT → Python: GMD + Ground Truth Parser")
    print("=" * 60)

    # ── Parse GMD ──────────────────────────────────────────────────────
    print(f"\n[1/2] Parsing GMD file: {args.gmd}")
    obs_df = parse_gmd(args.gmd)

    # Validation checks
    if len(obs_df) == 0:
        raise ValueError("No paired Range+RangeRate observations found in GMD file.")

    t_min = obs_df["time_s"].min() / 3600
    t_max = obs_df["time_s"].max() / 3600
    n_per_gs = obs_df.groupby("station").size()

    print(f"\n  Paired observations : {len(obs_df)}")
    print(f"  Time span           : {t_min:.2f} h — {t_max:.2f} h "
          f"({t_max/24:.2f} days)")
    for gs, n in n_per_gs.items():
        print(f"    {gs:<12}: {n} obs")

    # Range statistics
    rng_mean = obs_df["range_km"].mean()
    rng_min  = obs_df["range_km"].min()
    rng_max  = obs_df["range_km"].max()
    print(f"\n  Range stats: min={rng_min*1000:.0f} m  "
          f"mean={rng_mean*1000:.0f} m  max={rng_max*1000:.0f} m")

    obs_out = os.path.join(args.outdir, "observations.csv")
    obs_df.to_csv(obs_out, index=False, float_format="%.12f")
    print(f"\n  Saved → {obs_out}")

    # ── Parse Ground Truth ─────────────────────────────────────────────
    print(f"\n[2/2] Parsing ground truth: {args.gt}")
    gt_df = parse_ground_truth(args.gt)

    x0 = gt_df[["x_km","y_km","z_km","vx_kms","vy_kms","vz_kms"]].iloc[0].values
    r0 = np.linalg.norm(x0[:3])
    v0 = np.linalg.norm(x0[3:])

    print(f"  Epochs              : {len(gt_df)}")
    print(f"  Initial |r|         : {r0:.4f} km  (expected ~7078 km)")
    print(f"  Initial |v|         : {v0:.6f} km/s  (expected ~7.50 km/s)")
    print(f"  Initial state       : {x0}")

    gt_out = os.path.join(args.outdir, "ground_truth.csv")
    gt_df.to_csv(gt_out, index=False, float_format="%.12f")
    print(f"\n  Saved → {gt_out}")

    # ── Summary file ───────────────────────────────────────────────────
    summary_path = os.path.join(args.outdir, "parse_summary.txt")
    with open(summary_path, "w") as fh:
        fh.write("Project 2 — GMD Parse Summary\n")
        fh.write("=" * 50 + "\n\n")
        fh.write(f"GMD file   : {args.gmd}\n")
        fh.write(f"GT file    : {args.gt}\n\n")
        fh.write(f"Epoch (A1 MJD)    : {EPOCH_MJD_A1}\n")
        fh.write(f"Epoch (UTC)       : 2024-01-01 12:00:00\n\n")
        fh.write("Ground Station ID Map\n")
        for k, v in GS_ID_MAP.items():
            fh.write(f"  ID {k} → {v}\n")
        fh.write(f"\nObservations\n")
        fh.write(f"  Total paired    : {len(obs_df)}\n")
        for gs, n in n_per_gs.items():
            fh.write(f"  {gs:<12}    : {n}\n")
        fh.write(f"\nArc\n")
        fh.write(f"  Start  : {t_min:.4f} h  from epoch\n")
        fh.write(f"  End    : {t_max:.4f} h  from epoch\n")
        fh.write(f"  Length : {t_max/24:.4f} days\n")
        fh.write(f"\nGround Truth\n")
        fh.write(f"  Epochs          : {len(gt_df)}\n")
        fh.write(f"  Initial |r|     : {r0:.6f} km\n")
        fh.write(f"  Initial |v|     : {v0:.8f} km/s\n")
        fh.write(f"  x0              : {x0[:3]}\n")
        fh.write(f"  v0              : {x0[3:]}\n")

    print(f"\n  Summary saved → {summary_path}")
    print("\nDone. Next: run 01_batch_least_squares.py\n")


if __name__ == "__main__":
    main()