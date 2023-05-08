import os
import glob
import shutil
import pydeface

oak = '/oak/stanford/groups/russpold/data/'
bids_dir = oak+'myconnectome_2022/BIDS/'
raw_dir = bids_dir+'scitran/russpold/myconnectome/s01/ses-3/'

#functional scans
for n in range(0,2):
    run = n+1
    funcs = sorted(glob.glob(raw_dir+f"*run-{run}*ssg/*nii.gz"))
    for file in funcs:
        print(file)
        #echo = file[-8]
        new_name = bids_dir+f'sub-s01/ses-3/func/sub-s01_ses-3_task-rest_run-{run}_bold.nii.gz'
        print(new_name)
        shutil.copyfile(file, new_name)


#fmap
files = glob.glob(raw_dir+'*fmap*/*fieldmap*')
for file in files:
    print(file)
    new_name = file.replace(raw_dir, bids_dir+'sub-s01/ses-3/fmap/')
    print(new_name)
    #shutil.copyfile(file, new_name)

#sbref
files = glob.glob(raw_dir+'*sbref*/*.json')
for file in files:
    print(file)
    new_name = bids_dir+f'sub-s01/ses-2/func/sub-s01_ses-2_task-rest_sbref.json'
    print(new_name)
    shutil.copyfile(file, new_name)

#dwi
dwi_scans = sorted(glob.glob(raw_dir+'*dwi/*'))
for scan in dwi_scans:
    print(scan)
    acq_details = scan.split('/')[-2]
    ftype = scan.split('.')[-1]
    new_file = bids_dir+f'/s01/ses-1/dwi/sub-s01_ses-1_{acq_details}.{ftype}'
    shutil.copyfile(scan, new_file)


