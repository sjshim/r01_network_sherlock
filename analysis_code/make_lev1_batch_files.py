#!/usr/bin/env python

import glob
from pathlib import Path

def get_subids(root):
    subdirs = sorted(glob.glob(f'{root}/s*/'))
    subid = [val[-4:-1] for val in subdirs]
    return subid

tasks = ['cuedTS', 'directedForgetting', 'flanker', 'goNogo',
        'nBack', 'stopSignal', 'spatialTS', 'shapeMatching']
        # 'stopSignalWDirectedForgetting', 'stopSignalWFlanker',
        # 'directedForgettingWFlanker']

batch_stub = '/home/groups/russpold/network_fmri/analysis_code/run_stub.batch'
root = '/oak/stanford/groups/russpold/data/network_grant/validation_BIDS'

fixed_effects = True
qa = False
residuals = False

# For Jeanette's study no_rt is studied.  For other studies, use rt_centered unless
# modeling WATT3 and CCTHot as RT doesn't make sense in those paradigms
# When in doubt, ask Jeanette first before making changes!
rt_mapping = {
    'cuedTS':['rt_centered'], 
    'directedForgetting':['rt_centered'], 
    'flanker':['rt_centered'],
    'goNogo':['rt_centered'],
    'nBack':['rt_centered'],
    'stopSignal':['rt_centered'],
    'spatialTS':['rt_centered'],
    'shapeMatching':['rt_centered'],
    'stopSignalWDirectedForgetting':['rt_centered'],
    'stopSignalWFlanker':['rt_centered'],
    'directedForgettingWFlanker':['rt_centered']
}

#subids = get_subids(root)
subids = ['s286', 's295']
for task in tasks:
    batch_root = Path(f'{root}/derivatives/output/{task}_lev1_output/batch_files/')
    batch_root.mkdir(parents=True, exist_ok=True)
    rt_options = rt_mapping[task]
    for rt_inc in rt_options:
        if fixed_effects:
            batch_file = (f'{batch_root}/task_{task}_rtmodel_{rt_inc}_fixed-effects.batch')
            with open(batch_stub) as infile, open(batch_file, 'w') as outfile:
                for line in infile:
                    line = line.replace('JOBNAME', f"{task}_fixed-effects")
                    outfile.write(line)
                for sub in subids:
                    outfile.write(
                        f"echo /home/groups/russpold/network_fmri/analysis_code/analyze_lev1.py {task} {sub} {rt_inc} --fixed_effects --simplified_events \n"
                        f"/home/groups/russpold/network_fmri/analysis_code/analyze_lev1.py {task} {sub} {rt_inc} --fixed_effects  --simplified_events \n")
        elif residuals:
            batch_file = (f'{batch_root}/task_{task}_rtmodel_{rt_inc}_residuals.batch')
            with open(batch_stub) as infile, open(batch_file, 'w') as outfile:
                for line in infile:
                    line = line.replace('JOBNAME', f"{task}_residuals")
                    outfile.write(line)
                for sub in subids:
                    outfile.write(
                        f"echo /home/groups/russpold/network_fmri/analysis_code/analyze_lev1.py {task} {sub} {rt_inc} --residuals \n"
                        f"/home/groups/russpold/network_fmri/analysis_code/analyze_lev1.py {task} {sub} {rt_inc} --residuals \n") 
        else:
            batch_file = (f'{batch_root}/task_{task}_rtmodel_{rt_inc}.batch')   
            with open(batch_stub) as infile, open(batch_file, 'w') as outfile:
                for line in infile:
                    line = line.replace('JOBNAME', f"{task}_{rt_inc}")
                    outfile.write(line)
                for sub in subids:
                    outfile.write(
                        f"echo /home/groups/russpold/network_fmri/analysis_code/analyze_lev1.py {task} {sub} {rt_inc}\n"
                        f"/home/groups/russpold/network_fmri/analysis_code/analyze_lev1.py {task} {sub} {rt_inc}\n")                    
        if qa:
            batch_file = (f'{batch_root}/task_{task}_rtmodel_{rt_inc}_qa-only.batch')
            with open(batch_stub) as infile, open(batch_file, 'w') as outfile:
                for line in infile:
                    line = line.replace('JOBNAME', f"{task}_qa")
                    outfile.write(line)
                for sub in subids:
                    outfile.write(
                        f"echo /home/groups/russpold/network_fmri/analysis_code/analyze_lev1.py {task} {sub} {rt_inc} --qa_only \n"
                        f"/home/groups/russpold/network_fmri/analysis_code/analyze_lev1.py {task} {sub} {rt_inc} --qa_only \n")
