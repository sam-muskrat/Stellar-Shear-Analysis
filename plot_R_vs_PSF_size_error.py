### Read K_bootstrap_R_vs_psf_error_fits.py's per-iteration joint-bootstrap
### output and summarize + plot it: for each x-variant present in the file
### (PSF FWHM size error, and delta-T/T_true if K was run with -T), this
### shows one figure with:
###
### The baseline trial (needed for --paired and delta-fit residuals) is
### identified by ALL PSF-error axes being 0 simultaneously (see
### GLOBAL_BASELINE_POS), not just this plot's own x-axis - a folder that
### mixes size-only and beta-only trials has many rows with x_fwhm=0 (every
### beta trial), so checking x_fwhm alone would pick the wrong one.
###   - the per-trial means +/- bootstrap SE (top panel)
###   - the mean linear fit and mean quadratic fit, each with their
###     bootstrap-derived coefficient standard errors shown on the plot
###   - residuals from each fit (bottom panel)
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
### differences that trial's resampled mean R against the baseline (PSF size
### error = 0) trial's resampled mean R in the SAME iteration (both already
### jointly resampled by K), which cancels the shared noise these trials
### share without needing to fit anything. That matched-pairs residual is
### then compared to the FIXED model's predicted difference from baseline,
### with an error bar (delta_se) that's much smaller than the naive SE
### because it's no longer inflated by the shared component. See
### I_visualize_R_vs_psf_error.py's --paired-bootstrap for the same idea.
###
### -q/--quadratic-only and -l/--linear-only (mutually exclusive) restrict
### the plot to just one of the two fits, skipping the other's line,
### residual, and chi^2 entirely.
###
### --hyperbolic-fit additionally shows the physically-motivated model
### R(delta) = 2*(A - delta*C*(1+delta)^2)/(A - delta*C*(1+delta)^2 +
### B*(1+delta)^2) on the T-axis plot only (delta = x_T), if K was run with
### --hyperbolic-fit. A (~T_gal), B (~T_kernel), and C (~T_psf) are all free
### parameters. Unlike the linear/quadratic constant term, none of them
### cancels out of a --paired difference against baseline (all three enter
### the baseline's own model value too), so its --paired dof is three less
### than its naive dof.

import sys
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

## ++++++++++++++ I/O and general setups

parser = argparse.ArgumentParser(
    description="Summarize and plot K_bootstrap_R_vs_psf_error_fits.py's per-iteration "
                 "bootstrap output: per-trial means/SEs, mean linear+quadratic fits with "
                 "coefficient SEs, and residuals from each fit.")
parser.add_argument('input_feather',
                     help="Path to K_bootstrap_R_vs_psf_error_fits.py's output feather file")
parser.add_argument('--title', type=str, default='Stellar Shear Response (R) vs. PSF Size Error',
                     help="Base title for the plots (the x-variant name is appended)")
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
parser.add_argument('--hyperbolic-fit', action='store_true',
                     help="Also show the hyperbolic fit (R = 2*(A-delta*T_psf)/"
                          "(A-delta*T_psf+T_kernel)) on the T-axis plot, if K was run with "
                          "--hyperbolic-fit. Independent of -q/-l (T-axis only; ignored for "
                          "the FWHM-error plot).")
args = parser.parse_args()

boot_df = pd.read_feather(args.input_feather)
print(f">>> Loaded {len(boot_df)} bootstrap iterations from {args.input_feather}")

## ++++++++++++++ Figure out how many trials + which x-variants/fits are present

n_trials = 0
while f'mean_R_{n_trials}' in boot_df.columns:
    n_trials += 1
if n_trials == 0:
    print("Error: no mean_R_<i> columns found - is this a K_bootstrap_... "
          "output file? Aborting...")
    sys.exit(1)

print(f">>> {n_trials} trial(s) found in this file")

x_fwhm = np.array([boot_df[f'x_fwhm_{i}'].iloc[0] for i in range(n_trials)])
has_T = all(f'x_T_{i}' in boot_df.columns for i in range(n_trials))
x_T = np.array([boot_df[f'x_T_{i}'].iloc[0] for i in range(n_trials)]) if has_T else None
has_beta = all(f'x_beta_{i}' in boot_df.columns for i in range(n_trials))
x_beta = np.array([boot_df[f'x_beta_{i}'].iloc[0] for i in range(n_trials)]) if has_beta else None

y_boot_all = boot_df[[f'mean_R_{i}' for i in range(n_trials)]].to_numpy()  # (n_resamples, n_trials)

## Baseline = ALL available PSF-error axes are 0 simultaneously - checking
## only one axis can misidentify a baseline in a folder that mixes size-only
## and beta-only trials (e.g. every beta-only trial also has x_fwhm=0, so
## checking x_fwhm alone would match many rows, not just the true baseline).
_baseline_candidates = np.isclose(x_fwhm, 0.0)
if has_beta:
    _baseline_candidates &= np.isclose(x_beta, 0.0)
_baseline_rows = np.where(_baseline_candidates)[0]
GLOBAL_BASELINE_POS = _baseline_rows[0] if len(_baseline_rows) > 0 else None

## This plot's axis (FWHM size error, and T - both tied to the size axis)
## only uses trials that isolate it: beta held at 0 (size-only trials +
## baseline). A folder that also has beta-only trials would otherwise let
## beta-driven variation masquerade as unexplained scatter in a model that
## only knows about size - same reasoning as bootstrap_R_vs_PSF_error.py's
## per-axis subsetting (see its module docstring).
fwhm_trial_idx = np.where(np.isclose(x_beta, 0.0))[0] if has_beta else np.arange(n_trials)


## ++++++++++++++ Workhorse: summarize + plot one x-variant (fwhm or T)

def summarize_and_plot(trial_idx, x_full, x_label, prefix, title_suffix):
    x = x_full[trial_idx]
    y_boot = y_boot_all[:, trial_idx]
    n_trials = len(trial_idx)

    lin_slope_col = f'lin_{prefix}_slope'
    quad_c2_col = f'quad_{prefix}_c2'
    have_lin = lin_slope_col in boot_df.columns and not args.quadratic_only
    have_quad = quad_c2_col in boot_df.columns and not args.linear_only
    have_hyp = args.hyperbolic_fit and prefix == 'T' and 'hyp_T_A' in boot_df.columns

    if args.hyperbolic_fit and prefix == 'T' and not have_hyp:
        print(f">>> [{title_suffix}] --hyperbolic-fit requested but no hyp_T_A column "
              f"found - was K run with --hyperbolic-fit (and -T)? Skipping that fit.")

    if not have_lin and not have_quad and not have_hyp:
        print(f">>> No {prefix} fit columns found (or excluded by -q/-l) - "
              f"skipping the {title_suffix} plot")
        return

    # K's --delta-fit stores no constant term (it's always exactly 0 at the
    # baseline by construction), so its absence is how we detect that mode.
    is_delta_lin = have_lin and f'lin_{prefix}_intercept' not in boot_df.columns
    is_delta_quad = have_quad and f'quad_{prefix}_c0' not in boot_df.columns

    y_mean = y_boot.mean(axis=0)
    y_se = y_boot.std(axis=0, ddof=1)

    all_idx = list(range(n_trials))
    baseline_pos = None
    if args.paired or is_delta_lin or is_delta_quad:
        # Use the globally-identified baseline (all PSF-error axes at 0),
        # mapped into this call's (possibly filtered) local index space - not
        # just this x-array's own zero - see GLOBAL_BASELINE_POS above.
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

    lin_resid = quad_resid = hyp_resid = None
    lin_dof = quad_dof = hyp_dof = lin_err = quad_err = hyp_err = None
    hyp_idx = None

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

    if have_hyp:
        A_b = boot_df['hyp_T_A'].to_numpy()
        B_b = boot_df['hyp_T_B'].to_numpy()
        C_b = boot_df['hyp_T_C'].to_numpy()
        A_mean, A_se = A_b.mean(), A_b.std(ddof=1)
        B_mean, B_se = B_b.mean(), B_b.std(ddof=1)
        C_mean, C_se = C_b.mean(), C_b.std(ddof=1)

        def hyp_model(delta, A, B, C):
            # B (~T_kernel) and C (~T_psf) both scale by the same
            # (1+delta)^2 dilation factor as the assumed PSF size - see
            # K_bootstrap_...'s hyperbolic_model.
            T_kernel_eff = B * (1 + delta) ** 2
            T_psf_eff = C * (1 + delta) ** 2
            z = A - delta * T_psf_eff
            return 2 * z / (z + T_kernel_eff)

        ax_top.plot(x_fit, hyp_model(x_fit, A_mean, B_mean, C_mean), color='green',
                    linewidth=2, linestyle=':',
                    label=(r'Hyperbolic fit: $R=2(A-\delta C(1+\delta)^2)/'
                           r'(A-\delta C(1+\delta)^2+B(1+\delta)^2)$,' + '\n'
                           f'A = ({A_mean:.4f}$\\pm${A_se:.4f}), '
                           f'B = ({B_mean:.4f}$\\pm${B_se:.4f}), '
                           f'C = ({C_mean:.4f}$\\pm${C_se:.4f})'))
        print(f">>> [{title_suffix}] Hyperbolic fit: A = {A_mean:.6f} +/- {A_se:.6f}, "
              f"B = {B_mean:.6f} +/- {B_se:.6f}, C = {C_mean:.6f} +/- {C_se:.6f}")

        if args.paired and baseline_pos is not None:
            # Unlike the linear/quadratic constant term, none of A, B, C
            # cancels out of a difference against baseline (all three appear
            # nonlinearly in the baseline's own model value too) - so this
            # still costs a full 3 degrees of freedom relative to the naive
            # case.
            model_full = hyp_model(x, A_mean, B_mean, C_mean)
            model_delta = model_full[other_idx] - model_full[baseline_pos]
            hyp_resid = delta_mean[other_idx] - model_delta
            hyp_dof = (n_trials - 1) - 3  # k-1 differenced points, 3 free parameters (A, B, C)
            hyp_err = delta_se[other_idx]
            hyp_idx = other_idx
        else:
            # Naive: always uses ALL trials (there's no delta-fit variant of
            # this model), independent of whatever other_idx ended up being
            # for the linear/quadratic fits.
            hyp_resid = y_mean[all_idx] - hyp_model(x[all_idx], A_mean, B_mean, C_mean)
            hyp_dof = n_trials - 3  # 3 fitted parameters: A, B, C
            hyp_err = y_se[all_idx]
            hyp_idx = all_idx

    ax_top.set_ylabel(r'Mean Stellar Shear Response (R)', fontsize=12)
    ax_top.set_title(f'{args.title} ({title_suffix})', fontsize=14)
    ax_top.legend(fontsize=9)
    ax_top.grid(True, alpha=0.3)

    ax_bot.axhline(0, color='black', linewidth=1)
    if lin_resid is not None:
        suffix = '' if (is_delta_lin or args.paired) else ''
        ax_bot.errorbar(x[other_idx], lin_resid, yerr=lin_err,
                         fmt='o', capsize=4, color='red', ecolor='red',
                         label=f'Residual (bootstrapped{suffix})')
        if lin_dof > 0:
            chi2_lin = np.sum((lin_resid / lin_err) ** 2)
            print(f">>> [{title_suffix}] Linear fit reduced chi^2 (dof={lin_dof}) = "
                  f"{chi2_lin / lin_dof:.4f}")
    if quad_resid is not None:
        suffix = r'' if (is_delta_quad or args.paired) else ''
        ax_bot.errorbar(x[other_idx], quad_resid, yerr=quad_err,
                         fmt='s', capsize=4, color='purple', ecolor='purple',
                         label=f'Residual (bootstrapped{suffix})')
        if quad_dof > 0:
            chi2_quad = np.sum((quad_resid / quad_err) ** 2)
            print(f">>> [{title_suffix}] Quadratic fit reduced chi^2 (dof={quad_dof}) = "
                  f"{chi2_quad / quad_dof:.4f}")
    if hyp_resid is not None:
        suffix = ', matched-pairs vs. baseline' if args.paired else ''
        ax_bot.errorbar(x[hyp_idx], hyp_resid, yerr=hyp_err,
                         fmt='^', capsize=4, color='green', ecolor='green',
                         label=f'Residual (hyperbolic fit{suffix})')
        if hyp_dof > 0:
            chi2_hyp = np.sum((hyp_resid / hyp_err) ** 2)
            print(f">>> [{title_suffix}] Hyperbolic fit reduced chi^2 (dof={hyp_dof}) = "
                  f"{chi2_hyp / hyp_dof:.4f}")

    ax_bot.set_xlabel(x_label, fontsize=12)
    ax_bot.set_ylabel('Residual ($\Delta R_{obs}$ - $\Delta R_{fit}$)', fontsize=12)
    ax_bot.legend(fontsize=9)
    ax_bot.grid(True, alpha=0.3)

    plt.tight_layout()
    fit_suffix = '_linear' if args.linear_only else ('_quadratic' if args.quadratic_only else '')
    output_plot = f'{prefix}_bootstrap_fits{fit_suffix}.png'
    plt.savefig(output_plot, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {output_plot}")

    plt.show()


## ++++++++++++++ FWHM-size-error plot (always present), then T plot (if K was run with -T)

summarize_and_plot(fwhm_trial_idx, x_fwhm,
                    r'PSF FWHM Size Error ($\Delta$FWHM / FWHM$_{true}$)',
                    'fwhm', r'$\Delta FWHM / FWHM_{true}$')

if has_T:
    # T is tied to the size axis too (beta-only trials have no beta-specific
    # PSF image, so their x_T is degenerate/~0) - same trial subset as fwhm.
    summarize_and_plot(fwhm_trial_idx, x_T, r'$\Delta T / T_{true}$', 'T',
                        r'$\Delta T / T_{true}$')
else:
    print(">>> No delta-T/T_true columns found in this file (K was probably run "
          "without -T/--psf-dir) - skipping the T-based plot.")
