import glob
from nilearn import plotting
from nilearn import image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import numpy as np


bids = '/oak/stanford/groups/russpold/data/uh2/aim1/BIDS/'
tfce = True
tstat = True
combine = True
tasks = ['ANT', 'CCTHot', 'DPX', 'WATT3', 'discountFix','motorSelectiveStop', 'stopSignal', 'twoByTwo', 'stroop']
# Define slices
slices = np.linspace(-30, 70, 5)

for task in tasks:
	contrasts = sorted(glob.glob(bids+f'derivatives/output/{task}_lev2_output/{task}*'))
	start = 'contrast_'
	end = '_rtmodel'
	contrast_names = [c[c.find(start)+len(start):c.rfind(end)] for c in contrasts]
	print(contrast_names)
	if tfce:
		f, axes = plt.subplots(len(contrast_names), 1, figsize = (20,len(contrast_names)*5), squeeze=False)
		plt.suptitle(task, fontsize=20)
		for idx, contrast in enumerate(contrast_names):
			contrast_map = glob.glob(bids+f'derivatives/output/{task}_*lev2*/{task}_lev1_contrast_{contrast}_rtmodel*/randomise_output_model_one_sampt_tfce_corrp_fstat1.nii.gz')[0]
			plotting.plot_stat_map(contrast_map, cmap='bwr', axes=axes[idx][0], display_mode='z', cut_coords=slices, title=contrast)
		visual_dir = bids+f'derivatives/output/{task}_lev2_output/visualizations'
		if not os.path.exists(visual_dir):
				os.makedirs(visual_dir)
		pdf = visual_dir+f'/task-{task}_visualization_tfce.pdf'
		plt.subplots_adjust(hspace=0)
		f.savefig(pdf)
	if tstat:
		f, axes = plt.subplots(len(contrast_names), 1, figsize = (20,len(contrast_names)*5), squeeze=False)
		plt.suptitle(task, fontsize=20)
		for idx, contrast in enumerate(contrast_names):
			contrast_map = glob.glob(bids+f'derivatives/output/{task}_*lev2*/{task}_lev1_contrast_{contrast}_rtmodel*/randomise_output_model_one_sampt_tstat1.nii.gz')[0]
			plotting.plot_stat_map(contrast_map, cmap='cold_hot', axes=axes[idx][0], display_mode='z', cut_coords=slices, title=contrast)
		visual_dir = bids+f'derivatives/output/{task}_lev2_output/visualizations'
		if not os.path.exists(visual_dir):
				os.makedirs(visual_dir)
		pdf = visual_dir+f'/task-{task}_visualization_tstat.pdf'
		plt.subplots_adjust(hspace=0)
		f.savefig(pdf)
	if combine:
		f, axes = plt.subplots(len(contrast_names), 1, figsize = (20,len(contrast_names)*5), squeeze=False)
		plt.suptitle(task, fontsize=20)
		for idx, contrast in enumerate(contrast_names):
			mask_file = glob.glob(bids+f'derivatives/output/{task}_*lev2*/{task}_lev1_contrast_{contrast}_rtmodel*/randomise_output_model_one_sampt_tfce_corrp_fstat1.nii.gz')[0]
			mask_img = image.load_img(mask_file)
			mask = image.binarize_img(mask_img, threshold=0.95)
			contrast_map = glob.glob(bids+f'derivatives/output/{task}_*lev2*/{task}_lev1_contrast_{contrast}_rtmodel*/randomise_output_model_one_sampt_tstat1.nii.gz')[0]
			contrast_img = image.load_img(contrast_map)
			combined_map = image.math_img('img1 * img2', img1=contrast_img, img2=mask)
			plotting.plot_stat_map(combined_map, cmap='cold_hot', axes=axes[idx][0], display_mode='z', cut_coords=slices, title=contrast)
		visual_dir = bids+f'derivatives/output/{task}_lev2_output/visualizations'
		if not os.path.exists(visual_dir):
				os.makedirs(visual_dir)
		pdf = visual_dir+f'/task-{task}_visualization_tstat-tfce-thresholded.pdf'
		plt.subplots_adjust(hspace=0)
		f.savefig(pdf)	