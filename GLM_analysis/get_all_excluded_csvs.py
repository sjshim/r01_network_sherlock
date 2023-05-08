import pandas as pd
import os
import glob

main_df = pd.DataFrame()
for task in ['cuedTS', 'directedForgetting', 'flanker', 'goNogo',
        'nBack', 'stopSignal', 'spatialTS', 'shapeMatching',
        'stopSignalWDirectedForgetting', 'stopSignalWFlanker',
        'directedForgettingWFlanker']:
    exclusion_file = f'/oak/stanford/groups/russpold/data/network_grant/discovery_BIDS_21.0.1/derivatives/output/{task}_lev1_output/task_{task}_rtmodel_rt_centered/excluded_subject.csv'
    if os.path.exists(exclusion_file):
        df = pd.read_csv(exclusion_file, index_col=False)
        main_df = pd.concat([main_df, df])
        print(main_df)

main_df.to_csv('/home/groups/russpold/network_fmri/analysis_code/network_excluded_subjects.csv', index=False)