#!/usr/bin/env python
import glob
import os
from nilearn.glm.contrasts import compute_fixed_effects

tasks = ['nBack']
subjects = ['s03', 's10', 's19', 's29', 's43']

for sub in subjects:
    
    outdir = f'/oak/stanford/groups/russpold/data/network_grant/discovery_BIDS_21.0.1/derivatives/output/{task}_lev1_output'
    contrast_dir = f'{outdir}/task_{task}_rtmodel_rt_centered'

    fixed_effects_dir = f'{contrast_dir}/fixed_effects'

    contrast_names = [i.split('_rtmodel-rt_centered')[0].split(f'task-{task}_')[-1] for i in glob.glob(f'{contrast_dir}/contrast_estimates/*effect-size.nii.gz')]
    
    contrast_names = list(set(contrast_names))
    
    for contrast in contrast_names:
        effect_size_files = sorted(glob.glob(f'{contrast_dir}/contrast_estimates/))

#             for con_name, con in contrasts.items():
#                 effect_size_files = sorted(glob.glob(f'{contrast_dir}/contrast_estimates/sub-{subid}_*contrast-{con_name}*effect-size.nii.gz'))
#                 variance_files = sorted(glob.glob(f'{contrast_dir}/contrast_estimates/sub-{subid}_*contrast-{con_name}*variance.nii.gz'))
#                 fixed_fx_contrast, fixed_fx_variance, fixed_fx_stat = compute_fixed_effects(effect_size_files, variance_files, precision_weighted=True)
#                 fixed_effects_filename = (f'{contrast_dir}/contrast_estimates/sub-{subid}_task-{task}_contrast-{con_name}_rtmodel-{regress_rt}'
#                                         +'_stat-fixed-effects_t-test.nii.gz')
#                 fixed_fx_stat.to_filename(fixed_effects_filename)

contrast_names = list(set(contrast_names))

contrast_names


