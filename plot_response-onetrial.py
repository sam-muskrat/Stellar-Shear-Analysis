import argparse
import sys

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm, laplace, bootstrap

parser = argparse.ArgumentParser(
    description="Plot the stellar shear-response histogram from "
                 "C_calculate_R_cells.py's output.")
parser.add_argument('input_file',
                     help="Path to C_calculate_R_cells.py's output feather file "
                          "(R_per_cell.feather)")
parser.add_argument('--mag-min', type=float, default=None,
                     help="Drop cells brighter than this (mag_avg < mag-min)")
parser.add_argument('--mag-max', type=float, default=None,
                     help="Drop cells dimmer than this (mag_avg > mag-max), "
                          "e.g. to cut out the noisy faint end")
parser.add_argument('--title', type=str,
                     default='Stellar Shear Response using METADETECT ',
                     help="Title for the histogram plot")
parser.add_argument('--bins', type=int, default=100,
                     help="Number of bins for the histogram (default: 100)")
args = parser.parse_args()

input_file = args.input_file

# Load the data
print(f"Loading data from: {input_file}")
df = pd.read_feather(input_file)
print(f"Loaded {len(df)} rows and {len(df.columns)} columns")

R_df = pd.read_feather(input_file)

print(f"Loaded {len(R_df)} cell responses")

## Drop cells with NaN R or n_objects (e.g. cells missing one of the four
## shear types upstream) - a single NaN silently poisons np.average() and
## turns the whole weighted mean/std/bootstrap into NaN.
n_before = len(R_df)
R_df = R_df.dropna(subset=['R', 'n_objects']).reset_index(drop=True)
n_dropped = n_before - len(R_df)
if n_dropped > 0:
    print(f">>> Dropped {n_dropped} cell(s) with NaN R or n_objects "
          f"({100*n_dropped/n_before:.3f}% of cells)")

## Optional brightness cut (applied to the per-cell mag_avg)
if args.mag_min is not None or args.mag_max is not None:
    if 'mag_avg' not in R_df.columns:
        print("Error: 'mag_avg' column not found in this feather file - "
              "rerun C_calculate_R_cells.py to regenerate R_per_cell.feather "
              "with per-cell magnitudes.")
        sys.exit(1)
    if args.mag_min is not None:
        n_before = len(R_df)
        R_df = R_df[R_df['mag_avg'] >= args.mag_min]
        print(f">>> --mag-min {args.mag_min}: dropped {n_before - len(R_df)} cell(s) "
              f"brighter than mag {args.mag_min}")
    if args.mag_max is not None:
        n_before = len(R_df)
        R_df = R_df[R_df['mag_avg'] <= args.mag_max]
        print(f">>> --mag-max {args.mag_max}: dropped {n_before - len(R_df)} cell(s) "
              f"dimmer than mag {args.mag_max}")
    print(f">>> {len(R_df)} cell(s) remaining after brightness cut")
    if len(R_df) == 0:
        print("Error: no cells remain after the brightness cut. Aborting...")
        sys.exit(1)


def weighted_mean(r, w, axis=-1):
    return np.average(r, weights=w, axis=axis)


# Weighted statistics
weighted_mean_R = weighted_mean(R_df['R'].to_numpy(), R_df['n_objects'].to_numpy())
weighted_var = np.average((R_df['R'] - weighted_mean_R)**2, weights=R_df['n_objects'])
weighted_std = np.sqrt(weighted_var)

res = bootstrap(
    (R_df['R'].to_numpy(), R_df['n_objects'].to_numpy()),
    weighted_mean,
    paired=True,       # keep each R value tied to its own weight when resampling
    vectorized=True,   # np.average supports axis, so this is fast
    n_resamples=9999,
    method='basic',
    random_state=0,
)
se = res.standard_error

# sanity check: does BCa agree with mean ± 1.96*SE?
naive_lo, naive_hi = weighted_mean_R - 1.96 * se, weighted_mean_R + 1.96 * se
print(f"  Weighted Mean R: {weighted_mean_R:.6f}")
print(f"  Naive normal 95% CI: ({naive_lo:.6f}, {naive_hi:.6f})")

print(f"\nResponse statistics:")
print(f"  Mean R (weighted): {weighted_mean_R:.6f}")
print(f"  Std R (weighted):  {weighted_std:.6f}")
print(f"  Min R:  {R_df['R'].min():.6f}")
print(f"  Max R:  {R_df['R'].max():.6f}")
print(f"  Total Objects:  {sum(R_df['n_objects']):.6f}")
print("Std. Err of Mean:", se)
print("BCa 95% CI:", res.confidence_interval)


# Create histogram
plt.figure(figsize=(10, 6))

# Plot histogram with density=True for proper scaling with Gaussian
n, bins, patches = plt.hist(R_df['R'], bins=args.bins, edgecolor='black', alpha=0.7,
                            weights=R_df['n_objects'], label='Observed')

# Calculate mean and std (use weighted mean/std)
mean_R = weighted_mean_R
std_R = weighted_std

# Generate Gaussian curve
#x = np.linspace(R_df['R'].min(), R_df['R'].max(), 100)
#gaussian = norm.pdf(x, mean_R, std_R) * sum(R_df['n_objects']) * (bins[1] - bins[0])
#plt.plot(x, gaussian, 'r-', linewidth=2, label=f'Gaussian fit\n(μ={mean_R:.6f}, σ={std_R:.6f})')

# Add Laplace distribution
#laplace_scale = std_R / np.sqrt(2)  # Convert std to Laplace scale parameter
#laplace_dist = laplace.pdf(x, loc=mean_R, scale=laplace_scale) * sum(R_df['n_objects']) * (bins[1] - bins[0])
#plt.plot(x, laplace_dist, 'b-', linewidth=2, label=f'Laplace fit\n(μ={mean_R:.6f}, b={laplace_scale:.6f})')


# Add vertical line at mean
plt.axvline(mean_R, color='red', linestyle='--', linewidth=2, alpha=0.5,
            label=f'Mean = {mean_R:.6f}')

# Add vertical line at zero
plt.axvline(0, color='green', linestyle='--', linewidth=1,
            label='R = 0 (expected for stars)')


# Add shaded region for ±1 sigma
plt.axvspan(mean_R - se, mean_R + se, alpha=0.2, color='red', 
            label=f'±SE = {se:.6f}')

# Add vertical lines at ±1 sigma
plt.axvline(mean_R - se, color='orange', linestyle=':', linewidth=1.5, alpha=0.7)
plt.axvline(mean_R + se, color='orange', linestyle=':', linewidth=1.5, alpha=0.7)



# Labels and title
plt.xlabel('Shear Response (R)', fontsize=12)
plt.ylabel('Number of objects', fontsize=12)
plt.title(args.title, fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)

# Save figure
output_plot = 'shear_response_histogram.png'
plt.savefig(output_plot, dpi=300, bbox_inches='tight')
print(f"\nSaved histogram to {output_plot}")

# Also show it
plt.show()

# Optional: Print outliers
#threshold = 3 * std_R
#outliers = R_df[np.abs(R_df['R'] - mean_R) > threshold]
#if len(outliers) > 0:
#    print(f"\nFound {len(outliers)} outlier cells (>3σ from mean):")
#    print(outliers)


    # Calculate kurtosis to quantify "peakedness"
#from scipy.stats import kurtosis
#kurt = kurtosis(R_df['R'])
#print(f"\nKurtosis: {kurt:.3f}")
#print("  > 0: Leptokurtic (more peaked than Gaussian)")
#print("  = 0: Mesokurtic (same as Gaussian)")
#print("  < 0: Platykurtic (flatter than Gaussian)")
