import nibabel as nib
import glob
import matplotlib.pyplot as plt

def create_tr_dict(average=True):
    tr_dict = {}
    for task in ['cuedTS', 'directedForgetting', 'flanker', 'goNogo',
            'nBack', 'stopSignal', 'spatialTS', 'shapeMatching',
            'stopSignalWDirectedForgetting', 'stopSignalWFlanker',
            'directedForgettingWFlanker']:
        tr_list = []
        img_files = glob.glob(f'/oak/stanford/groups/russpold/data/network_grant/discovery_BIDS_21.0.1/derivatives/fitlins_data/*/*/func/*task-{task}_*_bold.nii.gz')
        for img in img_files:
            img = nib.load(img)
            tr_list.append(img.shape[-1])
        if average:
            tr_dict[task] = sum(tr_list)/len(tr_list)
        else:
            tr_dict[task] = tr_list
    return tr_dict

def create_tr_hist():
    tr_dict = create_tr_dict(average=False)
    
    for key, value in tr_dict.items():
        plt.clf()
        plt.hist(value, label=key, range=(0, 800))
        plt.legend()
        plt.savefig(f'tr_hist_{key}.png')

