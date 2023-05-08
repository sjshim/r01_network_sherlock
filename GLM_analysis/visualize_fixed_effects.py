import glob
from nilearn import glm, plotting, image
import matplotlib.pyplot as plt
import os

subjects = ['s03', 's10', 's19', 's29', 's43']
tasks = ['nBack', 'stopSignal', 'spatialTS', 'shapeMatching']
                #'stopSignalWDirectedForgetting', 'stopSignalWFlanker',
                #'directedForgettingWFlanker']
bids = '/oak/stanford/groups/russpold/data/network_grant/discovery_BIDS_21.0.1/'
fixed_effects = True
sessions = True
threshold = True
threshold_val = 2

for sub in subjects:
    print(sub)
    for task in tasks:
        print(task)
        fitlins = sorted(glob.glob(bids+f'derivatives/fitlins_data/sub-{sub}/*/*/*{task}_*Denoised*nii.gz'))
        mean_image = image.mean_img(fitlins)
        if fixed_effects:
            fixed_effects_contrasts = sorted(glob.glob(bids+f'derivatives/output/{task}_*/*/contrast_estimates/sub-{sub}*t-test.nii.gz'))
            f,  axes = plt.subplots(len(fixed_effects_contrasts), 1, figsize = (20,len(fixed_effects_contrasts)*5), squeeze=False)
            plt.suptitle(sub+'_'+task, fontsize=20)
            if threshold:
                for idx, f_name in enumerate(fixed_effects_contrasts):
                    thresholded_map, threshold = glm.threshold_stats_img(f_name, threshold=threshold_val)
                    start=f'sub-{sub}_'
                    end='_rtmodel'
                    file = os.path.basename(f_name)
                    contrast_name = file[file.find(start)+len(start):file.rfind(end)]
                    plotting.plot_stat_map(thresholded_map, bg_img=mean_image, cmap='cold_hot', axes=axes[idx][0], display_mode='z', cut_coords=5, title=contrast_name+'_thresholded')
                visual_dir = os.path.dirname(f_name).replace('contrast_estimates','contrast_visualizations')
                if not os.path.exists(visual_dir):
                    os.makedirs(visual_dir)
                pdf = visual_dir+f'/sub-{sub}_task-{task}_fixed-effects_thresholded_{threshold_val}.pdf'
                plt.subplots_adjust(hspace=0)
                f.savefig(pdf)
            else:
                for idx, f_name in enumerate(fixed_effects_contrasts):
                    start=f'sub-{sub}_'
                    end='_rtmodel'
                    file = os.path.basename(f_name)
                    contrast_name = file[file.find(start)+len(start):file.rfind(end)]
                    plotting.plot_stat_map(f_name, bg_img=mean_image, cmap='cold_hot', axes=axes[idx][0], display_mode='z', cut_coords=5, title=contrast_name)
                visual_dir = os.path.dirname(f_name).replace('contrast_estimates','contrast_visualizations')
                if not os.path.exists(visual_dir):
                    os.makedirs(visual_dir)
                pdf = visual_dir+f'/sub-{sub}_task-{task}_fixed-effects.pdf'
                plt.subplots_adjust(hspace=0)
                f.savefig(pdf)
        
        if sessions:
            session_contrasts = sorted(glob.glob(bids+f'derivatives/output/{task}_*/*/contrast_estimates/sub-{sub}*ses-*z_score.nii.gz'))
            f,  axes = plt.subplots(len(session_contrasts), 1, figsize = (20,len(session_contrasts)*5), squeeze=False)
            plt.suptitle(sub+'_'+task, fontsize=20)
            if threshold:
                for idx, f_name in enumerate(session_contrasts):
                    thresholded_map, threshold = glm.threshold_stats_img(f_name, threshold=threshold_val)
                    start=f'sub-{sub}_'
                    end='_rtmodel'
                    file = os.path.basename(f_name)
                    contrast_name = file[file.find(start)+len(start):file.rfind(end)]
                    plotting.plot_stat_map(thresholded_map, bg_img=mean_image,cmap='cold_hot',axes=axes[idx][0], display_mode='z', cut_coords=5, title=contrast_name+'_thresholded')
                visual_dir = os.path.dirname(f_name).replace('contrast_estimates','contrast_visualizations')
                if not os.path.exists(visual_dir):
                    os.makedirs(visual_dir)
                pdf = visual_dir+f'/sub-{sub}_task-{task}_sessions_thresholded_{threshold_val}.pdf'
                plt.subplots_adjust(hspace=0)
                f.savefig(pdf)
                
            else:
                for idx, f_name in enumerate(session_contrasts):
                    start=f'sub-{sub}_'
                    end='_rtmodel'
                    file = os.path.basename(f_name)
                    contrast_name = file[file.find(start)+len(start):file.rfind(end)]
                    plotting.plot_stat_map(f_name, bg_img=mean_image,cmap='cold_hot',axes=axes[idx][0], display_mode='z', cut_coords=5, title=contrast_name)
                visual_dir = os.path.dirname(f_name).replace('contrast_estimates','contrast_visualizations')
                if not os.path.exists(visual_dir):
                    os.makedirs(visual_dir)
                pdf = visual_dir+f'/sub-{sub}_task-{task}_sessions.pdf'
                plt.subplots_adjust(hspace=0)
                f.savefig(pdf)



