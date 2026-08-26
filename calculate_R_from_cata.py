import os
import sys
import glob

import numpy as np
import pandas as pd

## ++++++++++++++ I/O and general setups

# Gather runtag arg
if len(sys.argv) < 2:
    print("Error: No runTag provided!")
    print("Usage: python calculate_R_from_cata.py <runtag>")
    sys.exit(1)

RUNTAG = sys.argv[1]  # corresponds to folder name

## path to test directory
test_dir = '/scratch/users/samjc/output'

## What is the fitting model used in metadetect
fit_model = 'wmom'
flux_col = f'{fit_model}_band_flux'

## Zero point used only to report an approximate magnitude per cell (flux -> mag)
## Matches the dmag=30 flux convention used elsewhere in this pipeline.
ZERO_POINT = 30.0

## ++++++++++++++ Workhorse

def compute_R_per_cell(inpath_list):
    """
    Compute the per-cell shear response catalogue from a list of MetaDetect
    '.feather' catalogues (one per tile).
    """
    cata = []
    max_id = 0

    for inpath in inpath_list:
        cata_tmp = pd.read_feather(inpath)[
            ['shear_type',
             f'{fit_model}_s2n',
             f'{fit_model}_g_1',
             f'{fit_model}_g_2',
             f'{fit_model}_T',
             f'{fit_model}_T_ratio',
             flux_col,
             'cell_id']]

        cata_tmp['cell_id'] += max_id  # ensure unique cell id with multiple tiles
        max_id = cata_tmp['cell_id'].max() + 1

        cata.append(cata_tmp)
        del cata_tmp

    cata = pd.concat(cata, ignore_index=True)

    cells = cata['cell_id'].max() + 1
    R = []
    invalid_cells = 0
    multi_object_cells = 0

    for idx in range(cells):
        cell = cata[cata['cell_id'] == idx]

        # Skip if no data in this cell
        if len(cell) == 0:
            invalid_cells += 1
            continue

        noshear = cell[cell['shear_type']=='noshear']
        n_objects = len(noshear)

        if n_objects > 1:
            multi_object_cells += 1
            print(f">>> Warning: cell {idx} has {n_objects} 'noshear' rows "
                  f"(expected exactly 1 star per cell)")

        # Diagonal terms: response of g1 to an e1 shear, g2 to an e2 shear.
        g1_1p_s = cell.loc[cell['shear_type']=='1p', f'{fit_model}_g_1']
        g1_1m_s = cell.loc[cell['shear_type']=='1m', f'{fit_model}_g_1']
        g2_2p_s = cell.loc[cell['shear_type']=='2p', f'{fit_model}_g_2']
        g2_2m_s = cell.loc[cell['shear_type']=='2m', f'{fit_model}_g_2']
        # Off-diagonal (cross) terms: response of g1 to an e2 shear, g2 to an
        # e1 shear - same shear-type rows as above, just the OTHER g component.
        g1_2p_s = cell.loc[cell['shear_type']=='2p', f'{fit_model}_g_1']
        g1_2m_s = cell.loc[cell['shear_type']=='2m', f'{fit_model}_g_1']
        g2_1p_s = cell.loc[cell['shear_type']=='1p', f'{fit_model}_g_2']
        g2_1m_s = cell.loc[cell['shear_type']=='1m', f'{fit_model}_g_2']

        # Skip cells missing any of the four shear types (or noshear, needed for
        # mag_avg) - an empty-slice .mean() returns NaN silently (no exception),
        # which would otherwise poison R/mag_avg. The off-diagonal terms use the
        # same four shear-type row-groups as the diagonal ones, so this check
        # already covers them too.
        if min(len(g1_1p_s), len(g1_1m_s), len(g2_2p_s), len(g2_2m_s), n_objects) == 0:
            invalid_cells += 1
            continue

        try:
            # Calculate shear response for each cell
            R11 = (g1_1p_s.mean() - g1_1m_s.mean()) / 0.02
            R22 = (g2_2p_s.mean() - g2_2m_s.mean()) / 0.02
            R12 = (g1_2p_s.mean() - g1_2m_s.mean()) / 0.02
            R21 = (g2_1p_s.mean() - g2_1m_s.mean()) / 0.02

            mag_avg = (ZERO_POINT - 2.5 * np.log10(noshear[flux_col])).mean()

            resp = {
                'cell_id': idx,
                'n_objects': n_objects,
                'mag_avg': mag_avg,
                'R11': R11,
                'R22': R22,
                'R12': R12,
                'R21': R21,
                'R': (R11 + R22) / 2
            }
            R.append(resp)

        except (ZeroDivisionError, ValueError):
            # Skip cells with empty shear types
            invalid_cells += 1
            continue

    R_df = pd.DataFrame(R)
    return R_df, invalid_cells, multi_object_cells


def report(trial_label, R_df, invalid_cells, multi_object_cells, output_file):
    R_df.to_feather(output_file)
    print(f"\n>>> PSF error trial: {trial_label}")
    print(f"Saved {len(R_df)} cell responses to {output_file}")
    print(f"Response statistics:")
    print(f"  Mean R: {R_df['R'].mean():.6f}")
    print(f"  Std R:  {R_df['R'].std():.6f}")
    print(f"  Min R:  {R_df['R'].min():.6f}")
    print(f"  Max R:  {R_df['R'].max():.6f}")
    print(f"Cells ommitted due to insufficient data: {invalid_cells}")
    print(f"Cells with more than 1 'noshear' row: {multi_object_cells}")
    print(f"Object count statistics:")
    print(f"  Mean objects per cell: {R_df['n_objects'].mean():.1f}")
    print(f"  Median objects per cell: {R_df['n_objects'].median():.1f}")
    print(f"  Min objects per cell: {R_df['n_objects'].min():.0f}")
    print(f"  Max objects per cell: {R_df['n_objects'].max():.0f}")
    print(f"  Total objects across all cells: {R_df['n_objects'].sum():.0f}")


## Each PSF error trial (see Run.py --PSF_size_error / --PSF_beta_error) writes its
##  MetaDetect catalogues to its own subfolder under 'catalogues/shapes': the baseline
##  (both errors 0.0) lands directly in 'catalogues/shapes', a size-only trial goes to
##  'PSF_size_error_<value>', a beta-only trial to 'PSF_beta_error_<value>', and a
##  combined trial to 'PSF_size_error_<value>_PSF_beta_error_<value>' (matching the
##  tag Run.py builds).
shapes_dir = os.path.join(test_dir, RUNTAG, 'catalogues/shapes')
trials = [('baseline', shapes_dir)]
for psf_dir in sorted(glob.glob(os.path.join(shapes_dir, 'PSF_*error_*'))):
    if os.path.isdir(psf_dir):
        trials.append((os.path.basename(psf_dir), psf_dir))

n_done = 0
for label, cata_dir in trials:
    inpath_list = glob.glob(os.path.join(cata_dir, '*.feather'))
    print(f">>> Number of catalogues found in {cata_dir}: {len(inpath_list)} (should be one)")
    if len(inpath_list) < 1:
        print(f">>> No catalogues found for trial '{label}', skipping.")
        continue

    if label == 'baseline':
        out_file = 'R_per_cell.feather'
    else:
        out_file = f'R_per_cell_{label}.feather'
    output_file = os.path.join(test_dir, RUNTAG, out_file)

    R_df, invalid_cells, multi_object_cells = compute_R_per_cell(inpath_list)
    report(label, R_df, invalid_cells, multi_object_cells, output_file)
    n_done += 1

if n_done == 0:
    print('Error: no catalogs found for any PSF error trial. Aborting...')
    sys.exit(1)
