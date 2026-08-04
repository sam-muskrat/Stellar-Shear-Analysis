### Jointly bootstrap C_calculate_R_cells.py's per-trial output
### (R_per_cell*.feather) across a folder of PSF-size-error trials - the same
### underlying idea as I_visualize_R_vs_psf_error.py's --paired-bootstrap:
### resample the SAME common cell_id subset across all trials in a given
### iteration, so the shared noise realization these trials share (same
### stars/image, only the assumed PSF differs - see Run.py) moves coherently
### instead of being treated as independent per trial.
###
### Unlike I, this doesn't stop at a single point estimate + SE: for EVERY
### bootstrap iteration, it fits both a linear (R = a + b*x) and a quadratic
### (R = a + b*x + c*x^2) model relating shear response to PSF size error,
### and does this for x = fractional PSF size error (--PSF_size_error in
### Run.py), x = fractional PSF Moffat-beta error (--PSF_beta_error in
### Run.py, whenever more than one distinct beta value is present in
### input_dir - see calculate_R_from_cata.py for the combined naming
### convention, PSF_size_error_<v>_PSF_beta_error_<v>), and (optionally, via
### -T/--psf-dir) x = delta-T/T_true. Each iteration's per-trial resampled
### means and fit coefficients are written as one row of a wide-format
### .feather file (one column per parameter/statistic) - plot_R_vs_PSF_
### size_error.py / plot_R_vs_PSF_beta_error.py read this back to summarize
### and plot it.
###
### Since a folder can mix size-only, beta-only, and combined trials (all
### sharing the same baseline, plain R_per_cell.feather), the baseline row
### is identified by BOTH errors being 0 simultaneously - checking only one
### would also match every trial that varies the other error while holding
### this one at 0.
###
### Each axis is also fit using ONLY the trials that isolate it (one-factor-
### at-a-time): the size (and T) fits use trials where beta error = 0
### (size-only trials + baseline), and the beta fit uses trials where size
### error = 0 (beta-only trials + baseline). Combined trials (both errors
### nonzero, if present) isolate neither and are excluded from both. Fitting
### an axis against every trial regardless of the other error would let the
### other factor's effect masquerade as noise/misfit for this one - verified
### in practice: naively fitting vs. size using every trial in a folder that
### also has beta trials (all sitting at size error = 0) inflates chi^2 by
### ~6 orders of magnitude, and can bias the fitted coefficients too, since
### those points still pull on the least-squares solution.
###
### The per-trial weight used within EVERY iteration's fit is fixed up front
### as 1/se_R from an ordinary (non-paired) bootstrap of each trial's own
### data - i.e. how precisely we know that trial's own mean, independent of
### the shared-noise question the joint resampling below is about.
###
### --delta-fit replaces the traditional R = ... fit with one that fits
### delta R = R - R_baseline directly (through the origin: delta R = b*x for
### linear, delta R = a*x^2 + b*x for quadratic - the constant term is always
### exactly 0 at the baseline by construction, so it isn't fit). This differs
### subtly from just subtracting the constant out of the traditional fit
### afterward: here the baseline's own per-iteration noise never gets a say
### in the fit at all, whereas the traditional fit lets it help pin down the
### constant term (and, jointly, the other coefficients too).
###
### --hyperbolic-fit adds a third, physically-motivated model (vs. delta-T/
### T_true only - it needs -T/--psf-dir): R(delta) = 2*(A - delta*T_psf_eff) /
### (A - delta*T_psf_eff + T_kernel_eff), where delta = x_T,
### T_kernel_eff = B*(1+delta)^2, and T_psf_eff = C*(1+delta)^2 - i.e. B and C
### both scale by the same (1+delta)^2 dilation factor as the assumed PSF
### size (only the exponent is fixed at 2; B and C themselves are fit, not
### the T_kernel/T_psf values directly). A (~T_gal), B (~T_kernel), and C
### (~T_psf) are the three free parameters fit per iteration, seeded from
### --t-kernel and the baseline PSF's own fitted T as initial guesses for B
### and C respectively. Unlike the linear/quadratic models this is nonlinear
### in its free parameters, so each iteration solves for (A, B, C) with a
### bounded simplex minimizer instead of np.polyfit.

import os
import re
import sys
import glob
import argparse

import numpy as np
import pandas as pd
from scipy.stats import bootstrap
from scipy.optimize import minimize
from astropy.io import fits
import ngmix
from ngmix.metacal import MetacalFitGaussPSF

## Default T of the round kernel used to reconvolve the image after metacal's
## deconvolution step - a fixed constant for now (see --t-kernel); in reality
## it varies slightly per PSF size error, but that refinement isn't done yet.
T_KERNEL_DEFAULT = 0.360187

## ++++++++++++++ I/O and general setups

parser = argparse.ArgumentParser(
    description="Jointly bootstrap a folder of R_per_cell*.feather PSF-size-error trials, "
                 "fitting linear and quadratic models of shear response vs. PSF size error "
                 "(and, optionally, vs. delta-T/T_true) on every iteration. Saves one row "
                 "per iteration to a .feather file.")
parser.add_argument('input_dir',
                     help="Folder containing one R_per_cell*.feather file per PSF-error "
                          "trial (see sherlock/StarGrid_5sqDeg0 for the naming convention)")
parser.add_argument('output_feather',
                     help="Path to write the per-iteration bootstrap results to (.feather)")
parser.add_argument('--pattern', default='R_per_cell*.feather',
                     help="Glob pattern (relative to input_dir) used to find per-trial "
                          "feather files (default: R_per_cell*.feather)")
parser.add_argument('--n-resamples', type=int, default=9999,
                     help="Number of joint bootstrap iterations, default 9999")
parser.add_argument('--mag-min', type=float, default=None,
                     help="Drop cells brighter than this before averaging (mag_avg < mag-min)")
parser.add_argument('--mag-max', type=float, default=None,
                     help="Drop cells dimmer than this before averaging (mag_avg > mag-max), "
                          "e.g. to cut out the noisy faint end")
parser.add_argument('-T', '--psf-dir', type=str, default=None, metavar='PSF_DIR',
                     help="If given, also fit vs. delta-T/T_true (the PSF's Gaussian-fit "
                          "second moment T, relative to the baseline PSF's T). This folder "
                          "must contain one PSF fits image per trial, named to match each "
                          "trial's PSF size error - see sherlock/PSF1 for the naming "
                          "convention (psf_ima.fits for the baseline, "
                          "psf_ima_detect_err+0.3000.fits etc. for the rest). If omitted, "
                          "only the FWHM-size-error fit is done.")
parser.add_argument('--pixscale', type=float, default=0.2,
                     help="Pixel scale in arcsec/pixel, used only for the -T conversion "
                          "(default: 0.2)")
parser.add_argument('--delta-fit', action='store_true',
                     help="Fit delta R = R - R_baseline (baseline = PSF size error 0) instead "
                          "of the traditional R = ... fit. The model has no constant term "
                          "(delta R = b*x for linear, delta R = a*x^2 + b*x for quadratic, "
                          "since it's exactly 0 at the baseline by construction), and is fit "
                          "directly to the per-iteration baseline-differenced values rather "
                          "than derived afterward from a fit that included the baseline's own "
                          "noise. Replaces the traditional fit; requires a baseline trial.")
parser.add_argument('--hyperbolic-fit', action='store_true',
                     help="Also fit R(delta) = 2*(A - delta*C*(1+delta)^2)/(A - "
                          "delta*C*(1+delta)^2 + B*(1+delta)^2) vs. delta-T/T_true, with A "
                          "(~T_gal), B (~T_kernel), and C (~T_psf) as three free parameters "
                          "per iteration (only the (1+delta)^2 exponent is fixed). Optional "
                          "and additive - does not replace the linear/quadratic fits. "
                          "Requires -T/--psf-dir and at least 3 trials.")
parser.add_argument('--t-kernel', type=float, default=T_KERNEL_DEFAULT,
                     help=f"Initial guess for B (~T_kernel, arcsec^2), used only by "
                          f"--hyperbolic-fit (default: {T_KERNEL_DEFAULT})")
args = parser.parse_args()

if args.hyperbolic_fit and args.psf_dir is None:
    print("Error: --hyperbolic-fit requires -T/--psf-dir (it needs the baseline PSF's T). "
          "Aborting...")
    sys.exit(1)

## Matches "PSF_size_error_+0.0500" / "PSF_size_error_-0.0500" in a filename.
## A file with no such tag (e.g. plain R_per_cell.feather) is the baseline
## trial and is assigned PSF size error = 0.0.
PSF_ERROR_RE = re.compile(r'PSF_size_error_([+-]?\d+\.\d+)')


def psf_error_from_filename(path):
    match = PSF_ERROR_RE.search(os.path.basename(path))
    return float(match.group(1)) if match else 0.0


## Matches "PSF_beta_error_+0.0040" / "PSF_beta_error_-0.0040" in a filename -
## see Run.py's --PSF_beta_error and calculate_R_from_cata.py's naming
## convention. A file with no such tag is assigned PSF beta error = 0.0
## (true for both the plain baseline and any size-only trial).
PSF_BETA_RE = re.compile(r'PSF_beta_error_([+-]?\d+\.\d+)')


def psf_beta_error_from_filename(path):
    match = PSF_BETA_RE.search(os.path.basename(path))
    return float(match.group(1)) if match else 0.0


def psf_fits_path_for_error(psf_dir, psf_size_error):
    ## Matches Run.py's naming: baseline (no injected error) is saved as plain
    ## psf_ima.fits, other trials as psf_ima_detect_err{error:+.4f}.fits -
    ## see sherlock/PSF1.
    if psf_size_error == 0.0:
        return os.path.join(psf_dir, 'psf_ima.fits')
    return os.path.join(psf_dir, f'psf_ima_detect_err{psf_size_error:+.4f}.fits')


def weighted_mean(r, w, axis=-1):
    return np.average(r, weights=w, axis=axis)


def weighted_lstsq_through_origin(x, y, w, degree):
    ## Weighted least squares with NO constant term (matches np.polyfit's
    ## convention that a passed weight w multiplies the residual before
    ## squaring, i.e. minimizes sum((w*(pred-y))**2)). degree=1 fits y=b*x,
    ## degree=2 fits y=a*x^2+b*x.
    X = x[:, None] if degree == 1 else np.column_stack([x ** 2, x])
    Xw = X * w[:, None]
    yw = y * w
    beta, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
    return beta


def hyperbolic_model(A, delta, B, C):
    ## Both the reconvolution kernel's own T (B) and the PSF's own T (C)
    ## scale by the same (1+delta)^2 dilation factor as the assumed PSF size
    ## (delta = ΔT_psf/T_psf here, not a FWHM fraction, but this is the same
    ## scaling approximation used elsewhere for PSF size vs. T) - only the
    ## exponent is fixed; B and C themselves are fit, not the T_kernel/T_psf
    ## values directly.
    T_kernel_eff = B * (1 + delta) ** 2
    T_psf_eff = C * (1 + delta) ** 2
    z = A - delta * T_psf_eff
    return 2 * z / (z + T_kernel_eff)


def implied_A_estimate(delta, R, weight2, B, C):
    ## Analytic inversion of the model at fixed B, C: solve each trial's own R
    ## for the A it would imply if that trial's R were noiseless (R =
    ## 2z/(z+Tk_eff) => z = R*Tk_eff/(2-R) => A = z + delta*Tpsf_eff), then
    ## combine with a weighted average. Used only as a fast, stable starting
    ## guess for the per-iteration nonlinear solve below - not the fit itself.
    T_kernel_eff = B * (1 + delta) ** 2
    T_psf_eff = C * (1 + delta) ** 2
    z_implied = R * T_kernel_eff / (2 - R)
    A_implied = z_implied + delta * T_psf_eff
    return np.average(A_implied, weights=weight2)


def fit_hyperbolic_ABC(delta, R, weight2, A0, B0, C0):
    ## Nonlinear (rational in A, B, and C), so there's no polyfit-equivalent -
    ## minimize the same 1/se_R^2-weighted squared residual convention used
    ## for the linear/quadratic fits, via a bounded 3-parameter simplex
    ## solver (a large penalty additionally keeps it out of the unphysical
    ## z+Tk_eff <= 0 region within those bounds). B and C are both T-like
    ## (second moments), so they're bounded to stay positive - when delta is
    ## small across all trials, (1+delta)^2 is nearly identical for every
    ## point, so B and C become only weakly (and mutually) constrained, and
    ## an unbounded solver can wander to extreme, physically meaningless
    ## values chasing a nearly-flat direction (and take many iterations
    ## doing it).
    def sse(params):
        A, B, C = params
        T_kernel_eff = B * (1 + delta) ** 2
        T_psf_eff = C * (1 + delta) ** 2
        z = A - delta * T_psf_eff
        denom = z + T_kernel_eff
        if np.any(denom <= 0):
            return 1e10
        pred = 2 * z / denom
        return np.sum(weight2 * (pred - R) ** 2)

    A_margin = 10 * (abs(A0) + abs(B0)) + 1.0
    B_margin = 10 * abs(B0) + 1.0
    C_margin = 10 * abs(C0) + 1.0
    bounds = [(A0 - A_margin, A0 + A_margin),
              (max(1e-8, B0 - B_margin), B0 + B_margin),
              (max(1e-8, C0 - C_margin), C0 + C_margin)]
    # fatol=1e-14 turned out to be miscalibrated against this SSE's actual
    # scale (weight2 = 1/se_R^2 can be very large), so it almost never
    # satisfied the stopping criterion and burned through maxiter=3000 on
    # nearly every call (~100ms/call). xatol/fatol=1e-8/1e-12 with
    # maxiter=800 converges properly (confirmed against a known ground
    # truth) while resolving that - about 20x faster in the degenerate,
    # weakly-identified regime (small delta across all trials) where this
    # mattered most.
    result = minimize(sse, x0=[A0, B0, C0], method='Nelder-Mead', bounds=bounds,
                       options={'xatol': 1e-8, 'fatol': 1e-12, 'maxiter': 800})
    return result.x


def fitgauss_T(psf_filepath, pixscale=0.2):
    # Read into ngmix via astropy
    with fits.open(psf_filepath) as hdul:
        psf_data = hdul[0].data
    psf_obs = ngmix.Observation(image=psf_data)

    # Fit with Single Gaussian
    ## Need dummy observation...
    dummy_obs = ngmix.Observation(image=psf_data.copy(),
                                  psf=psf_obs)
    ## ...some rng...
    rng = np.random.RandomState(1234)
    ## ...and a fitter object
    mcal_fitter = MetacalFitGaussPSF(dummy_obs, rng=rng)
    ## This line returns fitted gaussian that's already dilated!
    psf_gauss_obs = mcal_fitter.get_all(step=0.01,
                                        types=["noshear"],)["noshear"].psf

    # Calculate moments -- Weighted moments with fixed round gaussian
    FWHM = 1.2/pixscale # fwhm of gaussian weight
    wmom = ngmix.gaussmom.GaussMom(fwhm=FWHM)
    moments= wmom.go(psf_gauss_obs)
    return moments["T"] * pixscale**2


## ++++++++++++++ Gather per-trial files

file_list = sorted(glob.glob(os.path.join(args.input_dir, args.pattern)))
if len(file_list) == 0:
    print(f"Error: no files matching '{args.pattern}' found in {args.input_dir}. Aborting...")
    sys.exit(1)

print(f">>> Found {len(file_list)} trial file(s) in {args.input_dir}")

## ++++++++++++++ Load + cut each trial, keeping per-cell data for the joint resampling below

results = []
trial_frames = {}
for path in file_list:
    psf_error = psf_error_from_filename(path)
    beta_error = psf_beta_error_from_filename(path)

    R_df = pd.read_feather(path)

    ## Drop cells with NaN R or n_objects - see D_visualize_cell_response.py
    n_before = len(R_df)
    R_df = R_df.dropna(subset=['R', 'n_objects']).reset_index(drop=True)
    n_dropped = n_before - len(R_df)
    if n_dropped > 0:
        print(f">>> {os.path.basename(path)}: dropped {n_dropped} cell(s) "
              f"with NaN R or n_objects")

    ## Optional brightness cut (applied to the per-cell mag_avg)
    if args.mag_min is not None or args.mag_max is not None:
        if 'mag_avg' not in R_df.columns:
            print(f"Error: 'mag_avg' column not found in {path} - rerun "
                  f"C_calculate_R_cells.py to regenerate it with per-cell magnitudes.")
            sys.exit(1)
        if args.mag_min is not None:
            R_df = R_df[R_df['mag_avg'] >= args.mag_min]
        if args.mag_max is not None:
            R_df = R_df[R_df['mag_avg'] <= args.mag_max]

    if len(R_df) == 0:
        print(f">>> Warning: {os.path.basename(path)} has no cells left "
              f"after cuts - skipping")
        continue

    if R_df['cell_id'].duplicated().any():
        print(f"Error: {path} has duplicate cell_id values - cannot align "
              f"cells across trials. Aborting...")
        sys.exit(1)

    r = R_df['R'].to_numpy()
    w = R_df['n_objects'].to_numpy()
    mean_R = weighted_mean(r, w)

    ## Ordinary (non-paired) bootstrap of this one trial in isolation - used
    ## only to set a fixed per-trial fit weight below, not as the final SE.
    res = bootstrap(
        (r, w), weighted_mean, paired=True, vectorized=True,
        n_resamples=args.n_resamples, method='basic', random_state=0,
    )
    se = res.standard_error

    results.append({
        'file': os.path.basename(path),
        'psf_size_error': psf_error,
        'psf_beta_error': beta_error,
        'mean_R': mean_R,
        'se_R': se,
        'n_cells': len(R_df),
        'n_objects': int(w.sum()),
    })
    trial_frames[os.path.basename(path)] = R_df.set_index('cell_id')[['R', 'n_objects']]

    print(f"  {os.path.basename(path)}: PSF size error = {psf_error:+.4f}, "
          f"PSF beta error = {beta_error:+.4f}, "
          f"mean R = {mean_R:.6f} +/- {se:.6f} ({len(R_df)} cells)")

if len(results) < 2:
    print("Error: need at least 2 usable trials to fit anything. Aborting...")
    sys.exit(1)

summary = pd.DataFrame(results).sort_values(
    ['psf_size_error', 'psf_beta_error']).reset_index(drop=True)
k = len(summary)
file_order = summary['file'].tolist()
x_fwhm = summary['psf_size_error'].to_numpy()
x_beta = summary['psf_beta_error'].to_numpy()
## Only fit vs. beta if the input folder actually contains more than one
## distinct beta value (e.g. a size-only folder has beta=0 for every trial,
## which would make a "fit" meaningless/degenerate).
have_beta = len(np.unique(x_beta)) > 1
## Fixed per-trial weight for every iteration's fit below (np.polyfit squares
## its w internally, so passing 1/se gives the correct 1/se^2 inverse-
## variance weighting).
fit_weight = 1 / summary['se_R'].to_numpy()

print("\nSummary (sorted by PSF size error, then beta error):")
print(summary.to_string(index=False))
if have_beta:
    print(f">>> {len(np.unique(x_beta))} distinct PSF beta error value(s) found - "
          f"will also fit vs. beta error")
else:
    print(">>> Only one distinct PSF beta error value found (probably 0.0) - "
          "skipping the beta-error fit")

## ++++++++++++++ Determine which trials isolate which axis (one-factor-at-a-time)
##
## Size (and T) fits use trials where beta = 0 (size-only trials + baseline);
## the beta fit uses trials where size = 0 (beta-only trials + baseline). A
## combined trial (both nonzero, if any exist) isolates neither and is used
## by neither fit - see the module docstring for why mixing them in is wrong,
## not just untidy.
beta_zero_mask = np.isclose(x_beta, 0.0)
size_zero_mask = np.isclose(x_fwhm, 0.0)

size_subset_idx = np.where(beta_zero_mask)[0]
beta_subset_idx = np.where(size_zero_mask)[0] if have_beta else np.array([], dtype=int)

have_lin_size = len(size_subset_idx) >= 2
have_quad_size = len(size_subset_idx) >= 3
if not have_quad_size:
    print(">>> Warning: fewer than 3 size-isolating trials (beta error = 0) - "
          "skipping the size-axis quadratic fit")

have_lin_beta = have_beta and len(beta_subset_idx) >= 2
have_quad_beta = have_beta and len(beta_subset_idx) >= 3
if have_beta and not have_quad_beta:
    print(">>> Warning: fewer than 3 beta-isolating trials (size error = 0) - "
          "skipping the beta-axis quadratic fit")

if args.hyperbolic_fit and len(size_subset_idx) < 3:
    print("Error: --hyperbolic-fit needs at least 3 size-isolating trials (beta "
          "error = 0) to fit its 3 free parameters (A, B, C). Aborting...")
    sys.exit(1)

# Both errors must be 0 to identify the TRUE baseline - a folder that mixes
# size-only and beta-only trials has multiple rows with size=0 (every beta
# trial) and multiple with beta=0 (every size trial), so checking only one
# would match the wrong row.
baseline_idx = None
baseline_rows = np.where(np.isclose(x_fwhm, 0.0) & np.isclose(x_beta, 0.0))[0]
if len(baseline_rows) > 0:
    baseline_idx = baseline_rows[0]

if args.delta_fit and baseline_idx is None:
    print("Error: --delta-fit requires a baseline (PSF size error = 0 AND "
          "PSF beta error = 0) trial. Aborting...")
    sys.exit(1)

size_other_idx = ([i for i in size_subset_idx if i != baseline_idx]
                   if baseline_idx is not None else list(size_subset_idx))
beta_other_idx = ([i for i in beta_subset_idx if i != baseline_idx]
                   if baseline_idx is not None else list(beta_subset_idx))

## ++++++++++++++ Optional: delta-T/T_true per trial

x_T = None
if args.psf_dir is not None:
    baseline_psf_path = os.path.join(args.psf_dir, 'psf_ima.fits')
    if not os.path.isfile(baseline_psf_path):
        print(f"Error: baseline PSF image not found at {baseline_psf_path}. Aborting...")
        sys.exit(1)

    print(f"\n>>> Fitting baseline PSF T from {baseline_psf_path}")
    T0 = fitgauss_T(baseline_psf_path, pixscale=args.pixscale)
    print(f">>> Baseline (PSF size error = 0) T_true = {T0:.6f} arcsec^2")

    delta_T_over_T = []
    for psf_error in summary['psf_size_error']:
        psf_path = psf_fits_path_for_error(args.psf_dir, psf_error)
        if not os.path.isfile(psf_path):
            print(f"Error: no PSF image found for PSF size error {psf_error:+.4f} "
                  f"(expected {psf_path}). Aborting...")
            sys.exit(1)
        T_i = fitgauss_T(psf_path, pixscale=args.pixscale)
        delta_T_over_T.append((T_i - T0) / T0)
        print(f"  PSF size error {psf_error:+.4f}: T = {T_i:.6f}, "
              f"deltaT/T_true = {(T_i - T0) / T0:.6f}")
    x_T = np.array(delta_T_over_T)

    if args.hyperbolic_fit:
        ## Fast, stable starting guess for (A, B, C), from the point-estimate
        ## (non-resampled) means - reused as the simplex starting point for
        ## every iteration's nonlinear solve below. B0 comes from --t-kernel,
        ## C0 from the baseline PSF's own fitted T; A is the analytic
        ## inversion of the model at those fixed B0, C0.
        hyp_B0 = args.t_kernel
        hyp_C0 = T0
        # T is tied to the size axis (beta trials have no beta-specific PSF
        # image, so their x_T is degenerate/~0) - restrict to size-isolating
        # trials for the same reason as the linear/quadratic size fit.
        hyp_A0 = implied_A_estimate(
            x_T[size_subset_idx], summary['mean_R'].to_numpy()[size_subset_idx],
            fit_weight[size_subset_idx] ** 2, hyp_B0, hyp_C0)
        print(f">>> Hyperbolic fit: initial A estimate = {hyp_A0:.6f} arcsec^2, "
              f"initial B (T_kernel) = {hyp_B0:.6f}, initial C (T_psf) = {hyp_C0:.6f}")

## ++++++++++++++ Align cells across trials (common cell_id subset)

frames = [trial_frames[f] for f in file_order]
common_idx = frames[0].index
for f in frames[1:]:
    common_idx = common_idx.intersection(f.index)
n_common = len(common_idx)
if n_common == 0:
    print("Error: no cell_id is common to all trials - cannot run the joint "
          "bootstrap. Aborting...")
    sys.exit(1)

per_trial_counts = ', '.join(str(len(f)) for f in frames)
print(f"\n>>> {n_common} cell(s) common to all {k} trials "
      f"(per-trial cell counts: {per_trial_counts})")

R_mat = np.column_stack([f.loc[common_idx, 'R'].to_numpy() for f in frames])
W_mat = np.column_stack([f.loc[common_idx, 'n_objects'].to_numpy() for f in frames])

## ++++++++++++++ Joint bootstrap: one row per iteration

print(f"\n>>> Running {args.n_resamples} joint bootstrap iterations "
      f"({n_common} cells resampled together per iteration)...")

rng = np.random.RandomState(0)
rows = []
for b in range(args.n_resamples):
    idx = rng.randint(0, n_common, size=n_common)
    Rb = R_mat[idx, :]
    Wb = W_mat[idx, :]
    y_b = (Rb * Wb).sum(axis=0) / Wb.sum(axis=0)

    row = {f'mean_R_{i}': y_b[i] for i in range(k)}

    if args.delta_fit:
        # Fit delta R = R - R_baseline directly (through the origin, no
        # constant term) to THIS iteration's own baseline-differenced values
        # - rather than deriving it afterward from a fit that also used the
        # baseline's own noise to help pin down the constant term. Computed
        # once for all k trials, then sliced per axis below (each axis only
        # uses the trials that isolate it - see the module docstring).
        delta_b_full = y_b - y_b[baseline_idx]

        if have_lin_size:
            (slope,) = weighted_lstsq_through_origin(
                x_fwhm[size_other_idx], delta_b_full[size_other_idx],
                fit_weight[size_other_idx], 1)
            row['lin_fwhm_slope'] = slope
        if have_quad_size:
            c2, c1 = weighted_lstsq_through_origin(
                x_fwhm[size_other_idx], delta_b_full[size_other_idx],
                fit_weight[size_other_idx], 2)
            row['quad_fwhm_c2'] = c2
            row['quad_fwhm_c1'] = c1

        if x_T is not None:
            if have_lin_size:
                (slope,) = weighted_lstsq_through_origin(
                    x_T[size_other_idx], delta_b_full[size_other_idx],
                    fit_weight[size_other_idx], 1)
                row['lin_T_slope'] = slope
            if have_quad_size:
                c2, c1 = weighted_lstsq_through_origin(
                    x_T[size_other_idx], delta_b_full[size_other_idx],
                    fit_weight[size_other_idx], 2)
                row['quad_T_c2'] = c2
                row['quad_T_c1'] = c1

        if have_lin_beta:
            (slope,) = weighted_lstsq_through_origin(
                x_beta[beta_other_idx], delta_b_full[beta_other_idx],
                fit_weight[beta_other_idx], 1)
            row['lin_beta_slope'] = slope
        if have_quad_beta:
            c2, c1 = weighted_lstsq_through_origin(
                x_beta[beta_other_idx], delta_b_full[beta_other_idx],
                fit_weight[beta_other_idx], 2)
            row['quad_beta_c2'] = c2
            row['quad_beta_c1'] = c1
    else:
        if have_lin_size:
            slope, intercept = np.polyfit(
                x_fwhm[size_subset_idx], y_b[size_subset_idx], 1,
                w=fit_weight[size_subset_idx])
            row['lin_fwhm_slope'] = slope
            row['lin_fwhm_intercept'] = intercept
        if have_quad_size:
            c2, c1, c0 = np.polyfit(
                x_fwhm[size_subset_idx], y_b[size_subset_idx], 2,
                w=fit_weight[size_subset_idx])
            row['quad_fwhm_c2'] = c2
            row['quad_fwhm_c1'] = c1
            row['quad_fwhm_c0'] = c0

        if x_T is not None:
            if have_lin_size:
                slope, intercept = np.polyfit(
                    x_T[size_subset_idx], y_b[size_subset_idx], 1,
                    w=fit_weight[size_subset_idx])
                row['lin_T_slope'] = slope
                row['lin_T_intercept'] = intercept
            if have_quad_size:
                c2, c1, c0 = np.polyfit(
                    x_T[size_subset_idx], y_b[size_subset_idx], 2,
                    w=fit_weight[size_subset_idx])
                row['quad_T_c2'] = c2
                row['quad_T_c1'] = c1
                row['quad_T_c0'] = c0

        if have_lin_beta:
            slope, intercept = np.polyfit(
                x_beta[beta_subset_idx], y_b[beta_subset_idx], 1,
                w=fit_weight[beta_subset_idx])
            row['lin_beta_slope'] = slope
            row['lin_beta_intercept'] = intercept
        if have_quad_beta:
            c2, c1, c0 = np.polyfit(
                x_beta[beta_subset_idx], y_b[beta_subset_idx], 2,
                w=fit_weight[beta_subset_idx])
            row['quad_beta_c2'] = c2
            row['quad_beta_c1'] = c1
            row['quad_beta_c0'] = c0

    if args.hyperbolic_fit:
        # Always fits R directly (not a delta-fit variant): A, B, and C all
        # appear in the baseline's own model value too, so none of them
        # cancels out of a difference the way an additive intercept/c0 would.
        # Restricted to size-isolating trials, same reasoning as the T fit.
        A_fit, B_fit, C_fit = fit_hyperbolic_ABC(
            x_T[size_subset_idx], y_b[size_subset_idx],
            fit_weight[size_subset_idx] ** 2, hyp_A0, hyp_B0, hyp_C0)
        row['hyp_T_A'] = A_fit
        row['hyp_T_B'] = B_fit
        row['hyp_T_C'] = C_fit

    rows.append(row)

boot_df = pd.DataFrame(rows)

## Trial-level constants, repeated on every row so this file is self-contained
## (L_plot_... doesn't need to re-derive x from the PSF images or refit anything).
for i in range(k):
    boot_df[f'x_fwhm_{i}'] = x_fwhm[i]
    if x_T is not None:
        boot_df[f'x_T_{i}'] = x_T[i]
    if have_beta:
        boot_df[f'x_beta_{i}'] = x_beta[i]
    boot_df[f'trial_file_{i}'] = file_order[i]

boot_df.to_feather(args.output_feather)
print(f"\nSaved {len(boot_df)} bootstrap iterations to {args.output_feather}")
print(f"Columns: {list(boot_df.columns)}")
