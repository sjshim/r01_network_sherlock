# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.1
#   kernelspec:
#     display_name: Python [conda env:multi-echo] *
#     language: python
#     name: conda-env-multi-echo-py
# ---

# %%
import os
import os.path as op
from glob import glob
import shutil
import json
import nibabel as nib 
import numpy as np 
import subprocess

# %% [markdown]
# to do: 
#
# accomodate ses-0 
#
# fix ses-010 bug 
#
# jupytext it! 
#
# sessions languishing in unsorted
#
# more helpful prints -> log files 
#
# better metadata
#

# %% [markdown]
# plan:
#
# check if subject/session folder already exists 
#
# ~~1. create session function~~
#     a. accomodate weird s43 session 
#
# ~~1. copy bold scans~~
# ~~s2. rename bold scans
#
# ~~1. copy anats 
# ~~2. rename anats~~
# ~~3. deface 
#
# ~~1. meta data - done for now (copied from heudiconv), but could be done here too. 
#
# ~~DOWNLOAD REST OF SCANS
#
# RUN FMRIPREP
#
#
# BIDS
#
#
#
# extras: 
# 1. copy dwi 
# 2. rename dwi
#
# 1. copy fmaps 
# 2. rename fmaps
#
# 1. tally "extras" 
# 2. report "missing" scans 
#
# 1. 

# %%
#OAK = '/oak/stanford/groups/russpold'
#mounted OAK 
OAK = '/Users/work/Analysis/network_fmri/OAK'

raw_dir = op.join(OAK, 'data/network_grant/nifti_raw_download')

BIDS_dir = op.join(OAK, 'data/network_grant/discovery_BIDS')
HOME = '/home/users/mphagen/network_fmri'

# %%
subjects = [i.split('/')[-1] for i in glob(op.join(raw_dir, 's*'))] 


# %%
#for testing
subject='s43'
sessions = [i.split('/')[-1] for i in sorted(glob.(op.join(raw_dir, f'{subject}/*')))]
session = sessions[-3]
raw_sub_path = op.join(raw_dir, subject, session)
sub_path = make_subject(subject)
sessions, session_dict = make_sessions(subject, sub_path)
func_files = glob.glob(op.join(raw_sub_path, f'*bold/*'))
file = func_files[0]

# %%
dataset_desc_json = {"Name": "Network Grant", 
                    "BIDSVersion": "1.5.0"} 

dataset_desc_path = op.join(BIDS_dir, 'dataset_description.json')
if not op.exists(dataset_desc_path): 
    print("making dataset description")
    with open(dataset_desc_path, 'w') as outfile:
        print("still making dataset description")
        json.dump(dataset_desc_json, outfile)
     

# %%
with open('echo_times.json', 'r') as e: 
    echo_dict = json.load(e)


# %%
def write_echo_metadata(echo, func_sub_path, subject, session_dict, task):
    echo_time = echo_dict[f'echo_{echo}']
    scan_echo_dict = {
        'EchoTime': float(echo_time),
        'EchoNumber': int(echo)
    }
    echo_json = op.join(func_sub_path, f'sub-{subject}_{session_dict[session]}_{task}_run-01_echo-{echo}_bold.json')
    with open(echo_json, 'w') as e: 
        json.dump(scan_echo_dict,e)


# %%
def make_subject_dirs(subject): 
    sub_path = op.join(BIDS_dir, f'sub-{subject}')
    os.makedirs(sub_path, exist_ok=True)
    return sub_path
        
def make_session_dirs(subject, sub_path): 
    """
    will currently put s43s out of order because of their weird session
    """
    
    sessions = [i.split('/')[-1] for i in sorted(glob(op.join(raw_dir, f'{subject}/*')))] #look at the source 
    session_dict = {}

    session_dict = {sessions[s]: f'ses-0{s+1}' for s in range(len(sessions))} #vpad 
    
    #create paths 
    for ses in session_dict.values(): 
        ses_path = op.join(sub_path, ses)
        os.makedirs(ses_path, exist_ok=True)
           
    with open(f'{subject}.json', 'w') as outfile:
        json.dump(session_dict, outfile)
    return sessions, session_dict

def move_func(subject, session, session_dict): 
    raw_sub_path = op.join(raw_dir, subject, session)
    func_files = glob(op.join(raw_sub_path, f'*bold/*_e*.nii.gz'))
    func_sub_path = op.join(BIDS_dir, f'sub-{subject}', session_dict[session], 'func')
    os.makedirs(func_sub_path, exist_ok=True)
    for file in func_files: 
        #can change to rename function with dictionary for dual tasks too 
        task = task_name_fix(file)
        echo = file.split('/')[-1].split('.')[0][-1]
        bids_func_path = op.join(func_sub_path, f'sub-{subject}_{session_dict[session]}_{task}_run-01_echo-{echo}_bold.nii.gz')
        if not op.exists(bids_func_path): 
            print(f'MOVING {task} {session}')
            shutil.copy2(file, bids_func_path)
            write_echo_metadata(echo, func_sub_path, subject, session_dict, task)
        else: print(f'EXISTS{task} {session}')



# %%
dual_dict = {
    'stopWDf': 'stopSignalWDirectedForgetting' ,
    'stopWFlanker': 'stopSignalWFlanker'
}

# %%
tasks = ['directedForgetting', 'stopSignal', 'goNogo', 'flanker', 
         'cuedTaskSwitching', 'spatialTaskSwitching', 'rest',
        'nback', 'shapeMatching']


def task_name_fix(file):
    """
    
    """

    # task = file.split('/')[-2].replace('task-', '').replace('shapeMaching', 'shapeMatching').replace('_bold', ''.replace('with', 'w')
    task = file.split('/')[-2]
    if 'task' in task: 
        task = task.split('-')[-1]
    task = task.replace('shapeMatching', 'shapeMaching')
    task = task.replace('_bold', '')
    task = task.replace('with', 'w')
    if 'w' in task: 
        task = ''.join([i.capitalize() for i in task.split('_')])
        task = task[0].lower() + task[1:]
    task = dual_dict.get(task, task)
    task =  f'task-{task}'

    return task



# %%
def t1w_check(t1w): 
    """
    checks for valid t1w
    """
    img = nib.load(t1w)
    dim = img.shape
    if len(dim) == 3: 
        return True

    #def t1w_check:
    #return len(nib.load(t1w).shape)==3
    

def fix_t1w_raw_path(t1w): 
    """fixes the stupid spaces"""
    no_spaces = t1w.replace(' ', '_')
    try: 
        os.rename(op.dirname(t1w), op.dirname(no_spaces)) #try shutil? 
    except OSError:  #be more specific 
        print('already moved')
    return no_spaces
 
    
def move_t1w(subject, session_dict):      
    t1w_files = glob(op.join(raw_dir, subject, '*', '*T1*', '*.nii.gz'))
    for t1w in t1w_files: 
        t1w = fix_t1w_raw_path(t1w)
        session = t1w.split('/')[-3]
        bids_anat_path = op.join(BIDS_dir, f'sub-{subject}', session_dict[session], 'anat')
        if t1w_check(t1w): #if it passes the check 
            os.makedirs(bids_anat_path, exist_ok=True)
            bids_t1_path  = op.join(bids_anat_path, f'sub-{subject}_{session_dict[session]}_run-01_T1w.nii.gz')
            if not op.exists(bids_t1_path): 
                cmd_string = f'bash network_pydeface.sh {t1w} {bids_t1_path}' 
                print(f'\tDefacing & moving...\n {subject}')
                p = subprocess.check_output(cmd_string, shell=True, env=os.environ.copy())         
         


# %%
for sub in subjects:
    sub_path = make_subject_dirs(sub)
    sessions, session_dict = make_session_dirs(sub, sub_path)
    move_t1w(sub, session_dict)
    for session in sessions: 
        move_func(sub, session, session_dict)
    

# %%
