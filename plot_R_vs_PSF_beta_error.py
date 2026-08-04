### Read bootstrap_R_vs_PSF_error.py's per-iteration joint-bootstrap output
### and summarize + plot R vs. PSF Moffat-beta error: this shows one figure
### with:
###   - the per-trial means +/- bootstrap SE (top panel)
###   - the mean linear fit and mean quadratic fit, each with their
###     bootstrap-derived coefficient standard errors shown on the plot
###   - residuals from each fit (bottom panel)
###
### Mirrors plot_R_vs_PSF_size_error.py's design exactly (see that file for
### more detail on each piece) but targets the beta axis instead of size/T,
### which don't apply here (Moffat beta has no "T" second-moment analog and
### no hyperbolic model has been derived for it).
###
### The baseline trial (needed for --paired and delta-fit residuals) is
### identified by ALL PSF-error axes being 0 simultaneously (see
### GLOBAL_BASELINE_POS) - a folder that mixes size-only and beta-only
### trials has many rows with x_beta=0 (every size trial), so checking
### x_beta alone would pick the wrong one.
###
### This plot's fit and residuals only use trials that isolate the beta axis
### (size error = 0: beta-only trials + baseline) - one-factor-at-a-time,
### same reasoning as bootstrap_R_vs_PSF_error.py's per-axis subsetting (see
### its module docstring). Fitting vs. beta using every trial, including
### size-only ones that all sit at beta=0, would let size-driven variation
### masquerade as unexplained scatter in a model that only knows about beta.
###
### By default, the residual panel is the naive/standard one: each trial's
### bootstrap mean R minus the FIXED (mean-across-iterations) model
### prediction at that trial's x, with the bootstrap SE of that mean as the
### error bar. This is the common approach, but note that these trials share
### a lot of noise (same stars/image, only the assumed PSF differs - see
### Run.py), so that SE - and this residual panel - doesn't distinguish
### "genuine model misspecification" from "these points all wobbled together
### this way by chance."
###
### --paired switches to a matched-pairs design instead: for each trial, it
### differences that trial's resampled mean R against the baseline trial's
### resampled mean R in the SAME iteration (both already jointly resampled),
### which cancels the shared noise these trials share without needing to fit
### anything. That matched-pairs residual is then compared to the FIXED
### model's predicted difference from baseline, with an error bar
### (delta_se) that's much smaller than the naive SE because it's no longer
### inflated by the shared component.
###
### -q/--quadratic-only and -l/--linear-only (mutually exclusive) restrict
### the plot to just one of the two fits, skipping the other's line,
### residual, and chi^2 entirely.

import sys
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

## ++++++++++++++ I/O and general setups

parser = argparse.ArgumentParser(
    description="Summarize and plot bootstrap_R_vs_PSF_error.py's per-iteration bootstrap "
                 "output: per-trial means/SEs, mean linear+quadratic fits (vs. PSF beta "
                 "error) with coefficient SEs, and residuals from each fit.")
parser.add_argument('input_feather',
                     help="Path to bootstrap_R_vs_PSF_error.py's output feather file "
                          "(must have been run on a folder with more than one distinct "
                          "PSF beta error value)")
parser.add_argument('--title', type=str, default='Stellar Shear Response (R) vs. PSF Beta Error',
                     help="Title for the plot")
parser.add_argument('--paired', action='store_true',
                     help="Use a matched-pairs residual (differenced against the baseline "
                          "trial, canceling shared noise) instead of the default naive "
                          "residual (mean R minus the fixed model, with the raw bootstrap "
                          "SE of the mean as the error bar).")
fit_group = parser.add_mutually_exclusive_group()
fit_group.add_argument('-q', '--quadratic-only', action='store_true',
                        help="Show only the quadratic fit (skip linear entirely).")
fit_group.add_argument('-l', '--linear-only', action='store_true',
                        help="Show only the linear fit (skip quadratic entirely).")
args = parser.parse_args()

boot_df = pd.read_feather(args.input_feather)
print(f">>> Loaded {len(boot_df)} bootstrap iterations from {args.input_feather}")

## ++++++++++++++ Figure out how many trials + which fits are present

n_trials = 0
while f'mean_R_{n_trials}' in boot_df.columns:
    n_trials += 1
if n_trials == 0:
    print("Error: no mean_R_<i> columns found - is this a bootstrap_R_vs_PSF_error.py "
          "output file? Aborting...")
    sys.exit(1)

print(f">>> {n_trials} trial(s) found in this file")

has_beta = all(f'x_beta_{i}' in boot_df.columns for i in range(n_trials))
if not has_beta:
    print("Error: no x_beta_<i> columns found - was this file's bootstrap run on a "
          "folder with more than one distinct PSF beta error value? Aborting...")
    sys.exit(1)

x_fwhm = np.array([boot_df[f'x_fwhm_{i}'].iloc[0] for i in range(n_trials)])
x_beta = np.array([boot_df[f'x_beta_{i}'].iloc[0] for i in range(n_trials)])

y_boot_all = boot_df[[f'mean_R_{i}' for i in range(n_trials)]].to_numpy()  # (n_resamples, n_trials)

## Baseline = ALL available PSF-error axes are 0 simultaneously - checking
## only beta can misidentify a baseline in a folder that mixes size-only and
## beta-only trials (e.g. every size-only trial also has x_beta=0, so
## checking x_beta alone would match many rows, not just the true baseline).
_baseline_candidates = np.isclose(x_fwhm, 0.0) & np.isclose(x_beta, 0.0)
_baseline_rows = np.where(_baseline_candidates)[0]
GLOBAL_BASELINE_POS = _baseline_rows[0] if len(_baseline_rows) > 0 else None

## This plot's axis (beta error) only uses trials that isolate it: size held
## at 0 (beta-only trials + baseline). A folder that also has size-only
## trials would otherwise let size-driven variation masquerade as
## unexplained scatter in a model that only knows about beta.
beta_trial_idx = np.where(np.isclose(x_fwhm, 0.0))[0]
if len(beta_trial_idx) < 2:
    print("Error: fewer than 2 trials isolate the beta axis (size error = 0) - "
          "cannot fit anything. Aborting...")
    sys.exit(1)


## ++++++++++++++ Workhorse: summarize + plot the beta axis

def summarize_and_plot(trial_idx, x_full, x_label, prefix, title_suffix):
    x = x_full[trial_idx]
    y_boot = y_boot_all[:, trial_idx]
    n_trials = len(trial_idx)

    lin_slope_col = f'lin_{prefix}_slope'
    quad_c2_col = f'quad_{prefix}_c2'
    have_lin = lin_slope_col in boot_df.columns and not args.quadratic_only
    have_quad = quad_c2_col in boot_df.columns and not args.linear_only

    if not have_lin and not have_quad:
        print(f">>> No {prefix} fit columns found (or excluded by -q/-l) - "
              f"skipping the {title_suffix} plot")
        return

    # bootstrap_R_vs_PSF_error.py's --delta-fit stores no constant term (it's
    # always exactly 0 at the baseline by construction), so its absence is
    # how we detect that mode.
    is_delta_lin = have_lin and f'lin_{prefix}_intercept' not in boot_df.columns
    is_delta_quad = have_quad and f'quad_{prefix}_c0' not in boot_df.columns

    y_mean = y_boot.mean(axis=0)
    y_se = y_boot.std(axis=0, ddof=1)

    all_idx = list(range(n_trials))
    baseline_pos = None
    if args.paired or is_delta_lin or is_delta_quad:
        # Use the globally-identified baseline (all PSF-error axes at 0),
        # mapped into this call's (filtered) local index space - not just
        # this x-array's own zero - see GLOBAL_BASELINE_POS above.
        if GLOBAL_BASELINE_POS is not None and GLOBAL_BASELINE_POS in trial_idx:
            baseline_pos = int(np.where(trial_idx == GLOBAL_BASELINE_POS)[0][0])
        if baseline_pos is None:
            print(f">>> [{title_suffix}] No baseline (all PSF errors = 0) trial "
                  f"found - delta-fit/--paired residuals need one")
    other_idx = [i for i in all_idx if i != baseline_pos] if baseline_pos is not None else all_idx

    delta_mean = delta_se = None
    if baseline_pos is not None:
        delta_boot = y_boot - y_boot[:, [baseline_pos]]
        delta_mean = delta_boot.mean(axis=0)
        delta_se = delta_boot.std(axis=0, ddof=1)

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(8, 9), sharex=True, gridspec_kw={'height_ratios': [2, 1]})

    ax_top.errorbar(x, y_mean, yerr=y_se, fmt='o', capsize=4, color='C0', ecolor='C0',
                     label='Mean Shear Response (R)')

    x_fit = np.linspace(x.min(), x.max(), 200)

    lin_resid = quad_resid = None
    lin_dof = quad_dof = lin_err = quad_err = None

    if have_lin:
        slope_b = boot_df[lin_slope_col].to_numpy()
        slope_mean, slope_se = slope_b.mean(), slope_b.std(ddof=1)

        if is_delta_lin:
            print(f">>> [{title_suffix}] Linear (delta) fit: slope = {slope_mean:.6f} "
                  f"+/- {slope_se:.6f}")
            if baseline_pos is not None:
                # Plot on the same absolute-R axis as the data by anchoring
                # the delta model back onto the baseline's own measured mean.
                anchor = y_mean[baseline_pos]
                ax_top.plot(x_fit, anchor + slope_mean * x_fit, color='red', linewidth=2,
                            label=(r'Linear fit ($\Delta R$): $\Delta R$ = '
                                   f'({slope_mean:.4f}$\\pm${slope_se:.4f})$\\cdot$x'))
                model_delta = slope_mean * (x[other_idx] - x[baseline_pos])
                lin_resid = delta_mean[other_idx] - model_delta
                lin_dof = (n_trials - 1) - 1  # k-1 differenced points, 1 free parameter (slope)
                lin_err = (delta_se if args.paired else y_se)[other_idx]
        else:
            intercept_b = boot_df[f'lin_{prefix}_intercept'].to_numpy()
            intercept_mean, intercept_se = intercept_b.mean(), intercept_b.std(ddof=1)

            ax_top.plot(x_fit, slope_mean * x_fit + intercept_mean, color='red', linewidth=2,
                        label=(f'Linear fit: R = ({slope_mean:.4f}$\\pm${slope_se:.4f})$\\cdot$x '
                               f'+ ({intercept_mean:.6f}$\\pm${intercept_se:.6f})'))
            print(f">>> [{title_suffix}] Linear fit: slope = {slope_mean:.6f} +/- {slope_se:.6f}, "
                  f"intercept = {intercept_mean:.6f} +/- {intercept_se:.6f}")

            if args.paired and baseline_pos is not None:
                # Intercept cancels exactly in a difference against the
                # baseline - this fixed (mean-parameter) model is evaluated
                # on the same matched-pairs quantity the noise scale
                # (delta_se) was built from, so a nonzero residual reflects
                # genuine model misspecification, not the fit's own
                # per-draw flexibility.
                model_delta = slope_mean * (x[other_idx] - x[baseline_pos])
                lin_resid = delta_mean[other_idx] - model_delta
                lin_dof = (n_trials - 1) - 1
                lin_err = delta_se[other_idx]
            else:
                # Naive: the fixed model evaluated at every trial's own x, vs.
                # its own bootstrap mean - the common approach, but note the
                # error bar (y_se) doesn't account for the noise these trials
                # share.
                lin_resid = y_mean[other_idx] - (slope_mean * x[other_idx] + intercept_mean)
                lin_dof = n_trials - 2  # 2 fitted parameters: slope, intercept
                lin_err = y_se[other_idx]

    if have_quad:
        c2_b = boot_df[quad_c2_col].to_numpy()
        c1_b = boot_df[f'quad_{prefix}_c1'].to_numpy()
        c2_mean, c2_se = c2_b.mean(), c2_b.std(ddof=1)
        c1_mean, c1_se = c1_b.mean(), c1_b.std(ddof=1)

        if is_delta_quad:
            print(f">>> [{title_suffix}] Quadratic (delta) fit: c2 = {c2_mean:.6f} "
                  f"+/- {c2_se:.6f}, c1 = {c1_mean:.6f} +/- {c1_se:.6f}")
            if baseline_pos is not None:
                anchor = y_mean[baseline_pos]
                ax_top.plot(x_fit, anchor + c2_mean * x_fit ** 2 + c1_mean * x_fit,
                            color='purple', linewidth=2, linestyle='-',
                            label=(r'Quadratic fit of $\Delta R$ ($R_{fit}$ = $\langle R_0 \rangle$ + $\Delta R$):' + '\n' + r'$  \Delta R$ = '
                                   f'({c2_mean:.4f}$\\pm${c2_se:.4f})$\\cdot$x$^2$ + '
                                   f'({c1_mean:.4f}$\\pm${c1_se:.4f})$\\cdot$x'))
                model_delta = (c2_mean * (x[other_idx] ** 2 - x[baseline_pos] ** 2)
                               + c1_mean * (x[other_idx] - x[baseline_pos]))
                quad_resid = delta_mean[other_idx] - model_delta
                quad_dof = (n_trials - 1) - 2  # k-1 differenced points, 2 free parameters
                quad_err = (delta_se if args.paired else y_se)[other_idx]
        else:
            c0_b = boot_df[f'quad_{prefix}_c0'].to_numpy()
            c0_mean, c0_se = c0_b.mean(), c0_b.std(ddof=1)

            ax_top.plot(x_fit, c2_mean * x_fit ** 2 + c1_mean * x_fit + c0_mean,
                        color='purple', linewidth=2, linestyle='--',
                        label=(f'Quadratic fit: R = ({c2_mean:.4f}$\\pm${c2_se:.4f})$\\cdot$x$^2$ + '
                               f'({c1_mean:.4f}$\\pm${c1_se:.4f})$\\cdot$x + '
                               f'({c0_mean:.6f}$\\pm${c0_se:.6f})'))
            print(f">>> [{title_suffix}] Quadratic fit: c2 = {c2_mean:.6f} +/- {c2_se:.6f}, "
                  f"c1 = {c1_mean:.6f} +/- {c1_se:.6f}, c0 = {c0_mean:.6f} +/- {c0_se:.6f}")

            if args.paired and baseline_pos is not None:
                # c0 cancels in the difference the same way the linear
                # intercept does.
                model_delta = (c2_mean * (x[other_idx] ** 2 - x[baseline_pos] ** 2)
                               + c1_mean * (x[other_idx] - x[baseline_pos]))
                quad_resid = delta_mean[other_idx] - model_delta
                quad_dof = (n_trials - 1) - 2
                quad_err = delta_se[other_idx]
            else:
                quad_resid = y_mean[other_idx] - (
                    c2_mean * x[other_idx] ** 2 + c1_mean * x[other_idx] + c0_mean)
                quad_dof = n_trials - 3  # 3 fitted parameters: c2, c1, c0
                quad_err = y_se[other_idx]

    ax_top.set_ylabel(r'Mean Stellar Shear Response (R)', fontsize=12)
    ax_top.set_title(f'{args.title}', fontsize=14)
    ax_top.legend(fontsize=9)
    ax_top.grid(True, alpha=0.3)

    ax_bot.axhline(0, color='black', linewidth=1)
    if lin_resid is not None:
        suffix = ', matched-pairs vs. baseline' if (is_delta_lin or args.paired) else ''
        ax_bot.errorbar(x[other_idx], lin_resid, yerr=lin_err,
                         fmt='o', capsize=4, color='red', ecolor='red',
                         label=f'Residual (linear fit{suffix})')
        if lin_dof > 0:
            chi2_lin = np.sum((lin_resid / lin_err) ** 2)
            print(f">>> [{title_suffix}] Linear fit reduced chi^2 (dof={lin_dof}) = "
                  f"{chi2_lin / lin_dof:.4f}")
    if quad_resid is not None:
        suffix = ', matched-pairs vs. baseline' if (is_delta_quad or args.paired) else ''
        ax_bot.errorbar(x[other_idx], quad_resid, yerr=quad_err,
                         fmt='s', capsize=4, color='purple', ecolor='purple',
                         label=f'Residual (quadratic fit{suffix})')
        if quad_dof > 0:
            chi2_quad = np.sum((quad_resid / quad_err) ** 2)
            print(f">>> [{title_suffix}] Quadratic fit reduced chi^2 (dof={quad_dof}) = "
                  f"{chi2_quad / quad_dof:.4f}")

    ax_bot.set_xlabel(x_label, fontsize=12)
    ax_bot.set_ylabel(r'Residual ($\Delta R_{obs}$ - $\Delta R_{fit}$)', fontsize=12)
    ax_bot.legend(fontsize=9)
    ax_bot.grid(True, alpha=0.3)

    plt.tight_layout()
    fit_suffix = '_linear' if args.linear_only else ('_quadratic' if args.quadratic_only else '')
    output_plot = f'{prefix}_bootstrap_fits{fit_suffix}.png'
    plt.savefig(output_plot, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {output_plot}")

    plt.show()


## ++++++++++++++ Beta-error plot

summarize_and_plot(beta_trial_idx, x_beta, r'PSF Beta Error ($\Delta\beta$ / $\beta_{true}$)',
                    'beta', r'$\Delta\beta / \beta_{true}$')
