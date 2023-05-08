The following files are modified by SJ to be run on r01 network project from JM's code written for uh2 aim1 data

preprocessing done on data:
fmriprep -> remove first 7 TRs from nifti -> tedana -> denoised images saved in derivatives/GLM_data
event files from BIDS -> remove first 7 TRs from onset -> saved in derivatives/GLM_data
confounds_timeseries.tsv from fmriprep output -> remove first 7 timepoints -> saved in derivatives/GLM_data
copied over brain_mask

discovery data = '/oak/stanford/groups/russpold/data/network_grant/discovery_BIDS_21.0.1/'

validation data = '/oak/stanford/groups/russpold/data/network_grant/validation_BIDS/'

- analyze_lev1.py
: python script that gets image, confounds, events.tsv, and brainmask files from GLM_data directory and builds and fits design matrix into first level model

- utils_lev1/first_level_designs.py
: details of first level model design per task

- make_lev1_batch_files.py
: python script that creates batch files that can run analyze_lev1.py for each subject and task specified. The batch files will be saved in {data}/derivatives/output/{task}/batch_files

    - run_stub.batch
    : batch file template used by make_lev1_batch_files.py

- launch_all_lev1_sherlock.sh
: bash script that runs all batch files created by make_lev1_batch_files.py

- visualize_fixed_effects.py
: python script that creates visualization .pdf files of fixed effects and individual run nifti images

    - run_visualization.batch
    : batch script to run visualize_fixed_effects.py 

-----------------------------------------------------------------------------------------------------------------
- check_average_TR
: scripts and figures for investigating average TR of scans per task to use as cutoffs for incomplete scans
