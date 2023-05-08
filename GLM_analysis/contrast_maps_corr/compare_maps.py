import os
import glob
import numpy as np
from nilearn import image
from natsort import natsorted
from nilearn import plotting
from nilearn import maskers
import nibabel as nib
import matplotlib.pyplot as plt
import pandas as pd

#paths
bids = '/oak/stanford/groups/russpold/data/network_grant/discovery_BIDS_21.0.1/'
fmriprep = bids + 'derivatives/fmriprep_21.0.1_c/'
output_dir = bids+'derivatives/output/*/*/contrast_estimates/*'
zmap = '*stat-z_score.nii.gz'


def get_coeff(image1, image2):
    img1_arr = image.get_data(image1)
    img2_arr = image.get_data(image2)
    img1_flat = np.ndarray.flatten(img1_arr)
    img2_flat = np.ndarray.flatten(img2_arr)
    r = np.corrcoef(img1_flat, img2_flat)[0,1]
    return r


def get_key_contrast(task):
    lookup = {'cuedTS':'*cuedTS_*contrast-task_switch_cost',
              'DF':'*directedForgetting_*contrast-neg-con',
              'flanker':'*flanker_*contrast-incongruent - congruent',
              'nBack':'*nBack_*contrast-twoBack - oneBack',
              'spatialTS':'*spatialTS_*contrast-task_switch_cost',
              'shapeMatching':'*shapeMatching_*contrast-main_vars',
              'goNogo':'*goNogo_*contrast-nogo_success-go',
              'stopSignal':'*stopSignal_*contrast-stop_failure-go'
             }
    return lookup.get(task)


def get_task_baseline_contrast(task):
    lookup = {'cuedTS':'*cuedTS_*contrast-task-baseline',
              'DF':'*directedForgetting_*contrast-task-baseline',
              'flanker':'*flanker_*contrast-task-baseline',
              'nBack':'*nBack_*contrast-task-baseline',
              'spatialTS':'*spatialTS_*contrast-task-baseline',
              'shapeMatching':'*shapeMatching_*contrast-task-baseline',
              'goNogo':'*goNogo_*contrast-task-baseline',
              'stopSignal':'*stopSignal_*contrast-task-baseline'
             }
    return lookup.get(task)


subjects = ['s03', 's10', 's19', 's29', 's43']
tasks = ['cuedTS', 'DF', 'flanker', 'nBack', 'spatialTS', 'shapeMatching', 'goNogo', 'stopSignal']

# Tell us how similar maps are within subject across sessions, within subject across task within construct, within subjects across constructs, across subjects within task, etc.

#within subject across sessions diff contrasts
corr_dict = {}
for sub in subjects:
    for task in tasks:
        contrast = get_key_contrast(task)
        images = natsorted(glob.glob(output_dir+sub+'*'+contrast+zmap))
        arr_vals = []
        for i in range(len(images)):
            ses_1 = images[i].split('/')[-1].split('_')[1]
            for n in range(i+1, len(images)):
                ses_2 = images[n].split('/')[-1].split('_')[1]
                corr_dict[task+'_'+ses_1+'_'+ses_2] = get_coeff(images[i], images[n])
    df = pd.DataFrame(corr_dict.items(), columns = ['task_session', 'corr'])
    df.to_csv(f'{sub}_diff_contrast_maps_correlations.csv')

#within subject across sessions diff contrasts
corr_dict = {}
for sub in subjects:
    for task in tasks:
        contrast = get_task_baseline_contrast(task)
        images = natsorted(glob.glob(output_dir+sub+'*'+contrast+zmap))
        arr_vals = []
        for i in range(len(images)):
            ses_1 = images[i].split('/')[-1].split('_')[1]
            for n in range(i+1, len(images)):
                ses_2 = images[n].split('/')[-1].split('_')[1]
                corr_dict[task+'_'+ses_1+'_'+ses_2] = get_coeff(images[i], images[n])
    df = pd.DataFrame(corr_dict.items(), columns = ['task_session', 'corr'])
    df.to_csv(f'{sub}_task_baseline_contrast_maps_correlations.csv')


