import glob
import pandas as pd
import os
import numpy as np

def calculate_mean_rt():
    base_dir = '/oak/stanford/groups/russpold/data/network_grant/discovery_BIDS_21.0.1/derivatives/fitlins_data/*/'
    tasks = ['cuedTS', 'directedForgetting', 'flanker', 'goNogo',
                    'nBack', 'stopSignal', 'spatialTS', 'shapeMatching',
                    'stopSignalWDirectedForgetting', 'stopSignalWFlanker',
                    'directedForgettingWFlanker']
    mean_rt_dict = {}
    for task in tasks:
        mean_rts = []
        event_files = glob.glob(base_dir+f'/*/func/*{task}_*events.tsv')
        if 'stopSignal' in task:
            for event_file in event_files:
                df = pd.read_csv(event_file, sep='\t')
                df['trial_type'].fillna('n/a', inplace=True)
                subset = df.query("(trial_type.str.contains('go') and response_time >= 0.2 and key_press == correct_response and junk == 0)" +
                                  "or (trial_type.str.contains('stop_failure') and response_time >= 0.2 and junk == 0)", engine='python')
                mean_rt = subset['response_time'].mean()
                mean_rts.append(mean_rt)
            mean_rt_dict[task] = sum(mean_rts)/len(mean_rts)
        else:
            for event_file in event_files:
                df = pd.read_csv(event_file, sep='\t')
                df['trial_type'].fillna('n/a', inplace=True)
                subset = df.query("key_press == correct_response and trial_type != 'n/a' and response_time >= 0.2 and junk == 0")
                mean_rt = subset['response_time'].mean()
                mean_rts.append(mean_rt)
            mean_rt_dict[task] = sum(mean_rts)/len(mean_rts)
    return mean_rt_dict


