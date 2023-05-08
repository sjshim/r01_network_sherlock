import glob
from nilearn import glm, plotting, image
import matplotlib.pyplot as plt
import os

subjects = ['s10', 's19', 's29', 's43', 's03']
tasks = ['cuedTS', 'directedForgetting', 'flanker', 'goNogo', 'nBack', 'stopSignal']
# tasks=['spatialTS', 'shapeMatching',
# 'stopSignalWDirectedForgetting', 'stopSignalWFlanker',
# 'directedForgettingWFlanker']
bids = '/oak/stanford/groups/russpold/data/network_grant/discovery_BIDS_21.0.1/'
threshold_val = 2

for sub in subjects:
    print(sub)
    for task in tasks:
        print(task)
        fitlins = sorted(glob.glob(bids+f'derivatives/fitlins_data/sub-{sub}/*/*/*{task}_*Denoised*nii.gz'))
        mean_image = image.mean_img(fitlins)
        yeo_contrasts = sorted(glob.glob(bids+f'derivatives/output/{task}_*/*/yeo_ROIs/sub-{sub}*.nii.gz'))
        print(len(yeo_contrasts))
        f,  axes = plt.subplots(len(yeo_contrasts), 1, figsize = (20,len(yeo_contrasts)*5), squeeze=False)
        plt.suptitle(sub+'_'+task, fontsize=20)
        
        # plt.clf()
        # for idx, f_name in enumerate(yeo_contrasts):
        #     thresholded_map, threshold = glm.threshold_stats_img(f_name, threshold=threshold_val)
        #     start=f'sub-{sub}_'
        #     end='_rtmodel'
        #     file = os.path.basename(f_name)
        #     contrast_name = file[file.find(start)+len(start):file.rfind(end)]
        #     plotting.plot_stat_map(thresholded_map, bg_img=mean_image, cmap='cold_hot', axes=axes[idx][0], display_mode='z', cut_coords=(-18, -1, 26, 43, 62), title=contrast_name+'_thresholded')
        # visual_dir = os.path.dirname(f_name).replace('yeo_ROIS','contrast_visualizations')
        # if not os.path.exists(visual_dir):
        #     os.makedirs(visual_dir)
        # pdf = visual_dir+f'/sub-{sub}_task-{task}_fixed-effects_yeo_thresholded_{threshold_val}.pdf'
        # print(pdf)
        # plt.subplots_adjust(hspace=0)
        # # f.savefig(pdf)

        #     plt.clf()
        #     f,  axes = plt.subplots(len(yeo_contrasts), 1, figsize = (20,len(yeo_contrasts)*5), squeeze=False)
        #     plt.suptitle(sub+'_'+task, fontsize=20)
        
        for idx, f_name in enumerate(yeo_contrasts):
            start=f'sub-{sub}_'
            end='_rtmodel'
            file = os.path.basename(f_name)
            contrast_name = file[file.find(start)+len(start):file.rfind(end)]
            plotting.plot_stat_map(f_name, bg_img=mean_image, cmap='cold_hot', axes=axes[idx][0], display_mode='z', cut_coords=(-18, -1, 26, 43, 62), title=contrast_name)
        visual_dir = os.path.dirname(f_name).replace('yeo_ROIS','contrast_visualizations')
        if not os.path.exists(visual_dir):
            os.makedirs(visual_dir)
        pdf = visual_dir+f'/sub-{sub}_task-{task}_fixed-effects_yeo.pdf'
        print(pdf)
        plt.subplots_adjust(hspace=0)
        f.savefig(pdf)


