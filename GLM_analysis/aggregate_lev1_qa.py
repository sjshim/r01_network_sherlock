import glob
import pandas as pd
import numpy as np

lev1_qa_files = glob.glob("/oak/stanford/groups/russpold/data/uh2/aim1_mumford/output/*lev1*/*no_rt/*csv")

all_qa = []
for file in lev1_qa_files:
    if len(all_qa) == 0:
        all_qa = pd.read_csv(file)
    else:
        qa_new = pd.read_csv(file)
        all_qa = pd.concat([all_qa, qa_new], axis = 0)

#Check if anybody is in here multiple times

all_qa[['subid', 'task']] = all_qa.subid_task.str.split("_", expand=True)

unique_subid = np.unique(all_qa.subid)
extra_subs_omit = []
for sub in unique_subid:
    sub_loc = all_qa.subid.str.count(sub)
    num_times_excluded = np.sum(sub_loc)
    #This isn't the same value for all tasks??? Base it on nonzero sum, I guess
    missing_more_half_jaime = np.sum(all_qa.query(f'subid == "{sub}"')['missing_more_than_half_the_tasks'])
    if num_times_excluded >4 and missing_more_half_jaime == 0:
        extra_subs_omit.append(sub)

all_qa.to_csv('/oak/stanford/groups/russpold/data/uh2/aim1_mumford/output_explore/all_lev1_qa.csv')


# Check all events.tsv files for negative RTs
def get_subids(root):
    subdirs = glob.glob(f'{root}/s*/')
    subid = [val[-4:-1] for val in subdirs]
    return subid

tasks = ['stroop', 'ANT',  'stopSignal', 'twoByTwo', 'CCTHot', 'WATT3',
                 'discountFix', 'DPX', 'motorSelectiveStop']
root = '/oak/stanford/groups/russpold/data/uh2/aim1/BIDS'

subids = get_subids(root)
# Crude check of negative RT, since I'm only filtering the RT 
for sub in subids:
    for task in tasks:
        if not all_qa.subid_task.eq(f'{sub}_{task}').any():
            events_tsv_file = glob.glob(f'{root}/sub-s{sub}/ses*/func/*{task}*tsv')
            events_tsv = pd.read_csv(events_tsv_file[0], sep='\t')
            rts = events_tsv.loc[events_tsv.junk == 0, 'response_time']
            if np.min(rts)<0:
                print(f'sub {sub}, task {task} min RT = {np.min(rts)}')

# Crude block durations
for sub in subids:
    for task in tasks:
        if not all_qa.subid_task.eq(f'{sub}_{task}').any():
            events_tsv_file = glob.glob(f'{root}/sub-s{sub}/ses*/func/*{task}*tsv')
            events_tsv = pd.read_csv(events_tsv_file[0], sep='\t')
            min_block_dur = events_tsv.block_duration.min()
            if min_block_dur<0:
                print(f'sub {sub}, task {task} min block_duration = {min_block_dur}')

all_min_ssd = []
for sub in subids:
    for task in ['stopSignal']:
        if not all_qa.subid_task.eq(f'{sub}_{task}').any():
            events_tsv_file = glob.glob(f'{root}/sub-s{sub}/ses*/func/*{task}*tsv')
            events_tsv = pd.read_csv(events_tsv_file[0], sep='\t')
            min_ssd = events_tsv.SS_delay.min()
            all_min_ssd.append(min_ssd)
            if min_ssd==0:
                print(f'sub {sub}, task {task} min ssd = {min_ssd}')

all_min_rt = []
for sub in subids:
    for task in ['DPX']:
        if not all_qa.subid_task.eq(f'{sub}_{task}').any():
            events_tsv_file = glob.glob(f'{root}/sub-s{sub}/ses*/func/*{task}*tsv')
            events_tsv = pd.read_csv(events_tsv_file[0], sep='\t')
            rts = events_tsv.loc[events_tsv.junk == 0, 'response_time']
            min_rt = np.min(rts)
            all_min_rt.append(min_rt)
            if min_rt< .1:
                print(f'sub {sub}, task {task} min rt = {min_rt}')

all_min_rt = []
for sub in subids:
    for task in ['discountFix']:
        if not all_qa.subid_task.eq(f'{sub}_{task}').any():
            events_tsv_file = glob.glob(f'{root}/sub-s{sub}/ses*/func/*{task}*tsv')
            events_tsv = pd.read_csv(events_tsv_file[0], sep='\t')
            rts = events_tsv.loc[events_tsv.junk == 0, 'response_time']
            min_rt = np.min(rts)
            all_min_rt.append(min_rt)
            if min_rt< .1:
                print(f'sub {sub}, task {task} min rt = {min_rt}')


all_min_rt = []
for sub in subids:
    for task in ['ANT']:
        if not all_qa.subid_task.eq(f'{sub}_{task}').any():
            events_tsv_file = glob.glob(f'{root}/sub-s{sub}/ses*/func/*{task}*tsv')
            events_tsv = pd.read_csv(events_tsv_file[0], sep='\t')
            rts = events_tsv.loc[events_tsv.junk == 0, 'response_time']
            min_rt = np.min(rts)
            all_min_rt.append(min_rt)
            if min_rt< .1:
                print(f'sub {sub}, task {task} min rt = {min_rt}')