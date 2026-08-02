### Bin C_calculate_R_cells.py's per-cell response by brightness and plot mean R
### vs. magnitude - a combination of E_calculate_R_flux_bins.py (binning) and
### F_visualize_R_vs_brightness.py (plotting).
###
### Unlike E, which bins RAW per-row measurements by their own (shear-type-
### specific) flux before computing R, this bins the per-cell R that
### C_calculate_R_cells.py already computed from all 5 shear-type rows in that
### cell. Since a cell's noshear/1p/1m/2p/2m rows stay together through R's
### calculation, binning happens only after R already exists as one scalar
### per object - so an object can no longer contribute its shear-type
### measurements to different bins (see G_diagnose_shear_bin_migration.py for
### why that mattered for E's approach).
###
### Takes only C's output feather (R_per_cell.feather) as input.

import sys
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import bootstrap

## ++++++++++++++ I/O and general setups

parser = argparse.ArgumentParser(
    description="Bin C_calculate_R_cells.py's per-cell response by brightness "
                 "and plot mean R vs. magnitude.")
parser.add_argument('input_feather',
                     help="Path to C_calculate_R_cells.py's output feather file "
                          "(R_per_cell.feather)")
parser.add_argument('--n-bins', type=int, default=10,
                     help="Number of brightness bins (equal-count/quantile), default 10")
parser.add_argument('--n-resamples', type=int, default=9999,
                     help="Bootstrap resamples used for the standard-error columns, default 9999")
parser.add_argument('--mag-min', type=float, default=None,
                     help="Drop cells brighter than this before binning (mag_avg < mag-min)")
parser.add_argument('--mag-max', type=float, default=None,
                     help="Drop cells dimmer than this before binning (mag_avg > mag-max), "
                          "e.g. to cut out the noisy faint end")
args = parser.parse_args()

## ++++++++++++++ Load C's per-cell results

cata = pd.read_feather(args.input_feather)
print(f">>> Loaded {len(cata)} cell responses from {args.input_feather}")

required_cols = {'mag_avg', 'n_objects', 'R11', 'R22', 'R'}
missing = required_cols - set(cata.columns)
if missing:
    print(f"Error: missing required column(s) {sorted(missing)} - "
          f"rerun C_calculate_R_cells.py to regenerate this feather file.")
    sys.exit(1)

## Drop cells with NaN response/magnitude (e.g. cells C already flagged as
## invalid - shouldn't be present, but guard anyway since a single NaN would
## silently poison the weighted mean/bootstrap for its whole bin)
n_before = len(cata)
cata = cata.dropna(subset=['mag_avg', 'R11', 'R22', 'R']).reset_index(drop=True)
n_dropped = n_before - len(cata)
if n_dropped > 0:
    print(f">>> Dropped {n_dropped} cell(s) with NaN response/magnitude "
          f"({100*n_dropped/n_before:.3f}% of cells)")

## Optional brightness cut, applied before binning
if args.mag_min is not None:
    n_before = len(cata)
    cata = cata[cata['mag_avg'] >= args.mag_min]
    print(f">>> --mag-min {args.mag_min}: dropped {n_before - len(cata)} cell(s) "
          f"brighter than mag {args.mag_min}")
if args.mag_max is not None:
    n_before = len(cata)
    cata = cata[cata['mag_avg'] <= args.mag_max]
    print(f">>> --mag-max {args.mag_max}: dropped {n_before - len(cata)} cell(s) "
          f"dimmer than mag {args.mag_max}")

if len(cata) == 0:
    print("Error: no cells remain. Aborting...")
    sys.exit(1)

## ++++++++++++++ Bin cells by brightness (quantile bins on the already-computed
## per-cell mag_avg)

bin_edges = np.quantile(cata['mag_avg'], np.linspace(0, 1, args.n_bins + 1))
## Guard against duplicate edges (e.g. many cells with identical magnitude)
bin_edges = np.unique(bin_edges)
n_bins_actual = len(bin_edges) - 1
if n_bins_actual < args.n_bins:
    print(f">>> Warning: only {n_bins_actual} unique bin edges found "
          f"(requested {args.n_bins}); some bins were merged.")

cata['mag_bin'] = pd.cut(cata['mag_avg'], bins=bin_edges,
                          labels=False, include_lowest=True)


def weighted_mean(r, w, axis=-1):
    return np.average(r, weights=w, axis=axis)


## ++++++++++++++ Per-bin weighted mean + bootstrap SE (weighted by n_objects,
## so the rare multi-object cell doesn't silently get the same weight as a
## normal single-star cell)

R = []
invalid_bins = 0

for idx in range(n_bins_actual):
    b = cata[cata['mag_bin'] == idx]

    if len(b) == 0:
        invalid_bins += 1
        continue

    weights = b['n_objects'].to_numpy()
    R11_arr = b['R11'].to_numpy()
    R22_arr = b['R22'].to_numpy()
    R_arr = b['R'].to_numpy()

    R11_mean = weighted_mean(R11_arr, weights)
    R22_mean = weighted_mean(R22_arr, weights)
    R_mean = weighted_mean(R_arr, weights)

    if len(b) > 1:
        R11_se = bootstrap((R11_arr, weights), weighted_mean,
                            paired=True, vectorized=True,
                            n_resamples=args.n_resamples, method='basic',
                            random_state=0).standard_error
        R22_se = bootstrap((R22_arr, weights), weighted_mean,
                            paired=True, vectorized=True,
                            n_resamples=args.n_resamples, method='basic',
                            random_state=0).standard_error
        R_se = bootstrap((R_arr, weights), weighted_mean,
                          paired=True, vectorized=True,
                          n_resamples=args.n_resamples, method='basic',
                          random_state=0).standard_error
    else:
        # A single cell has no resampling variance to estimate - report the
        # point estimate with an undefined (NaN) SE rather than a fake 0.
        R11_se = R22_se = R_se = np.nan

    mag_lo, mag_hi = bin_edges[idx], bin_edges[idx + 1]

    R.append({
        'mag_bin': idx,
        'n_cells': len(b),
        'n_objects': weights.sum(),
        'mag_lo': mag_lo,
        'mag_hi': mag_hi,
        'mag_median': b['mag_avg'].median(),
        'R11': R11_mean,
        'R11_se': R11_se,
        'R22': R22_mean,
        'R22_se': R22_se,
        'R': R_mean,
        'R_se': R_se,
    })

R_df = pd.DataFrame(R)

print(f"\nResponse statistics:")
print(f"  Mean R: {R_df['R'].mean():.6f}")
print(f"  Std R:  {R_df['R'].std():.6f}")
print(f"  Min R:  {R_df['R'].min():.6f}")
print(f"  Max R:  {R_df['R'].max():.6f}")
print(f"Bins omitted due to insufficient data: {invalid_bins}")
print(f"\nBrightness bin summary (faint -> bright):")
print(R_df[['mag_bin', 'n_cells', 'n_objects', 'mag_median',
            'R11', 'R11_se', 'R22', 'R22_se', 'R', 'R_se']]
      .to_string(index=False))

## ++++++++++++++ Plot

plt.figure(figsize=(9, 6))

plt.errorbar(R_df['mag_median'], R_df['R'], yerr=R_df['R_se'],
             fmt='o-', color='tab:blue', ecolor='tab:blue',
             capsize=3, linewidth=1, markersize=6,
             label='R (bootstrap SE)')

plt.axhline(0, color='green', linestyle='--', linewidth=1,
            label='R = 0 (expected for stars)')

plt.gca().invert_xaxis()  # brighter (smaller mag) to the right
plt.xlabel('Magnitude (brighter →)', fontsize=12)
plt.ylabel('Shear Response (R)', fontsize=12)
plt.title('Stellar Shear Response vs. Brightness (binned by cell)', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)

plt.show()
