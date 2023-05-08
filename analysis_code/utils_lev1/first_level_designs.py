from nilearn.glm.first_level import compute_regressor
import numpy as np
import pandas as pd
from calculate_mean_rt import calculate_mean_rt

mean_rt_dict = calculate_mean_rt()

#NOTE: having N/A trials as 'n/a' doesn't seem to work
#I'm not sure why but for now, I've put in 'na' for n/a trial_types
def make_regressor_and_derivative(n_scans, tr, events_df, add_deriv,
                   amplitude_column=None, duration_column=None,
                   onset_column=None, subset=None, demean_amp=False, 
                   cond_id = 'cond'):
    """ Creates regressor and derivative using spm + derivative option in
        nilearn's compute_regressor
        Input:
          n_scans: number of timepoints (TRs)
          tr: time resolution in seconds
          events_df: events data frame
          add_deriv: "yes"/"no", whether or not derivatives of regressors should
                     be included
          amplitude_column: Required.  Amplitude column from events_df
          duration_column: Required.  Duration column from events_df
          onset_column: optional.  if not specified "onset" is the default
          subset: optional.  Boolean for subsetting rows of events_df
          demean_amp: Whether amplitude should be mean centered
          cond_id: Name for regressor that is created.  Note "cond_derivative" will
            be assigned as name to the corresponding derivative
        Output:
          regressors: 2 column pandas data frame containing main regressor and derivative
    """
    if subset == None:
        events_df['temp_subset'] = True
        subset = 'temp_subset == True'
    if onset_column == None:
        onset_column = 'onset'
    if amplitude_column == None or duration_column == None:
        print('Must enter amplitude and duration columns')
        return
    if amplitude_column not in events_df.columns:
        print("must specify amplitude column that exists in events_df")
        return
    if duration_column not in events_df.columns:
        print("must specify duration column that exists in events_df")
        print(cond_id)
        return
    reg_3col = events_df.query(subset)[[onset_column, duration_column, amplitude_column]]
    reg_3col = reg_3col.rename(
        columns={duration_column: "duration",
        amplitude_column: "modulation"})
    if demean_amp:
        reg_3col['modulation'] = reg_3col['modulation'] - \
        reg_3col['modulation'].mean()
    if add_deriv == 'deriv_yes':
        hrf_model = 'spm + derivative'
    else:
        hrf_model= 'spm'
        
    regressor_array, regressor_names = compute_regressor(
        np.transpose(np.array(reg_3col)),
        hrf_model,
        #deals with slice timing issue with outputs from fMRIPrep
        np.arange(n_scans)*tr+tr/2,
        con_id=cond_id
    ) 
    regressors =  pd.DataFrame(regressor_array, columns=regressor_names)  
    return regressors

def define_nuisance_trials (events_df, task):
    
    if task in ['cuedTS', 'nBack', 'spatialTS', 'flanker', 'shapeMatching']:
        omission = events_df.key_press == -1
        commission = (events_df.key_press != events_df.correct_response) & (events_df.key_press != -1) & (events_df.response_time >= .2)
        rt_too_fast = (events_df.response_time < .2)
        bad_trials = omission | commission | rt_too_fast
        
    elif task in ['directedForgetting', 'directedForgettingWFlanker']:
        omission = (events_df.key_press == -1) & (events_df.trial_type != 'memory_cue')
        commission = (events_df.key_press != events_df.correct_response) & (events_df.key_press != -1) & (events_df.response_time >= .2) & (events_df.trial_type != 'memory_cue')
        rt_too_fast = (events_df.response_time < .2) & (events_df.trial_type != 'memory_cue')
        bad_trials = omission | commission | rt_too_fast
        
    elif task in ['stopSignal', 'goNogo']:
        omission = (events_df.key_press == -1) & (events_df.trial_type == 'go')
        commission = (events_df.key_press != events_df.correct_response) & (events_df.key_press != -1) & (events_df.trial_type == 'go') & (events_df.response_time >= .2)
        rt_too_fast = (events_df.response_time < .2) & (events_df.trial_type == 'go') 
        bad_trials = omission | commission| rt_too_fast
        
    elif task in ['stopSignalWDirectedForgetting']:
        omission = (events_df.key_press == -1) & (events_df.trial_type.isin(['go_pos', 'go_neg', 'go_con']))
        commission = (events_df.key_press != events_df.correct_response) & (events_df.key_press != -1) & (events_df.trial_type.isin(['go_pos', 'go_neg', 'go_con'])) & (events_df.response_time >= .2)
        rt_too_fast = (events_df.response_time < .2) & (events_df.trial_type.isin(['go_pos', 'go_neg', 'go_con']))
        bad_trials = omission | commission| rt_too_fast
        
    elif task in ['stopSignalWFlanker']:
        omission = (events_df.key_press == -1) & (events_df.trial_type.isin(['go_incongruent', 'go_congruent']))
        commission = (events_df.key_press != events_df.correct_response) & (events_df.key_press != -1) & (events_df.trial_type.isin(['go_incongruent', 'go_congruent'])) & (events_df.response_time >= .2)
        rt_too_fast = (events_df.response_time < .2) & (events_df.trial_type.isin(['go_incongruent', 'go_congruent']))
        bad_trials = omission | commission| rt_too_fast
        
    return 1*bad_trials, 1*omission, 1*commission, 1*rt_too_fast

def make_basic_cuedTS_desmat(events_file, duration_choice, add_deriv,
    regress_rt, n_scans, tr, confound_regressors
):
    """Creates basic cued task switching regressors (and derivatives), 
    defining error regressors and adding fmriprep confound regressors
       Input
         events_df: events data frame (events.tsv data)
         n_scans: Number of scans
         tr: time resolution
         confound_regressors: Confounds derived from fmriprep output
       Returns
          design_matrix: pd data frame including all regressors (and derivatives)
          contrasts: dictionary of contrasts in nilearn friendly format
          rt_subset: Boolean for extracting correct rows of events_df in the case that 
              rt regressors are requeset.
    """
    events_df = pd.read_csv(events_file, sep = '\t')
    events_df['junk_trials'], events_df['omission'], events_df['commission'], events_df['rt_fast'] = \
        define_nuisance_trials(events_df, 'cuedTS')
    percent_junk = np.mean(events_df['junk_trials'])
    events_df['constant_1_column'] = 1  

    omission_regressor = make_regressor_and_derivative(
            n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
            amplitude_column="omission", duration_column="constant_1_column",
            subset="trial_type != 'na'and junk == 0", demean_amp = False, cond_id = 'omission'
        )
    commission_regressor = make_regressor_and_derivative(
            n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
            amplitude_column="commission", duration_column="constant_1_column",
            subset="trial_type != 'na'  and junk == 0", demean_amp = False, cond_id = 'commission'
        )
    rt_fast = make_regressor_and_derivative(
            n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
            amplitude_column="rt_fast", duration_column="constant_1_column",
            subset="trial_type != 'na' and junk == 0", demean_amp = False, cond_id = 'rt_fast'
        )
    na_trials = make_regressor_and_derivative(
            n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
            amplitude_column="na_trials", duration_column="constant_1_column",
            subset=None, demean_amp = False, cond_id = 'na_trials'
        )
    junk_regressor = make_regressor_and_derivative(
            n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
            amplitude_column="junk", duration_column="constant_1_column",
            subset="trial_type != 'na'", demean_amp = False, cond_id = 'junk'
        )

    rt_subset = "key_press == correct_response and response_time >= 0.2 and na_trials == 0 and junk == 0"
    if duration_choice == 'mean_rt':
        mean_rt = events_df.query(rt_subset)['response_time'].mean()
        events_df['constant_column'] = events_df['constant_1_column'] * mean_rt
    else:
        events_df['constant_column'] = events_df['constant_1_column']
        
    task_stay_cue_switch = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
        amplitude_column="constant_1_column", duration_column="constant_column",
        subset="key_press == correct_response and response_time >= 0.2 and trial_type == 'tstay_cswitch' and junk == 0", demean_amp=False, 
        cond_id='task_stay_cue_switch'
    )
    task_stay_cue_stay = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
        amplitude_column="constant_1_column", duration_column="constant_column",
        subset="key_press == correct_response and response_time >= 0.2 and trial_type == 'tstay_cstay' and junk == 0", demean_amp=False, 
        cond_id='task_stay_cue_stay'
    )
    task_switch_cue_switch = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
        amplitude_column="constant_1_column", duration_column="constant_column",
        subset="key_press == correct_response and response_time >= 0.2 and trial_type == 'tswitch_cswitch' and junk == 0", 
        demean_amp=False, cond_id='task_switch_cue_switch'
    )
    print(task_stay_cue_switch)
    print(task_stay_cue_stay)
    print(task_switch_cue_switch)
    design_matrix = pd.concat([task_stay_cue_switch, task_stay_cue_stay, task_switch_cue_switch,
                               omission_regressor, commission_regressor, rt_fast, na_trials,
                               confound_regressors, junk_regressor], axis=1)
    contrasts = {'task_switch_cost': 'task_switch_cue_switch-task_stay_cue_switch',
                'cue_switch_cost': 'task_stay_cue_switch-task_stay_cue_stay',
                 'task-baseline': '1/3*(task_stay_cue_switch+task_stay_cue_stay+task_switch_cue_switch)'}

    if regress_rt == 'rt_centered':
        mn_rt = mean_rt_dict['cuedTS']
        events_df['response_time_centered'] = events_df.response_time - mn_rt
        rt = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
        amplitude_column="response_time_centered", duration_column="constant_column",
        subset=rt_subset, demean_amp=False, cond_id='response_time'
        ) 
        design_matrix = pd.concat([design_matrix, rt], axis=1)
        contrasts["response_time"] = "response_time"
    if regress_rt == 'rt_uncentered':
        rt = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
        amplitude_column="response_time", duration_column="constant_column",
        subset=rt_subset, demean_amp=False, cond_id='response_time'
        ) 
        design_matrix = pd.concat([design_matrix, rt], axis=1)
        contrasts["response_time"] = "response_time"
        
    return design_matrix, contrasts, percent_junk, events_df

def make_basic_directedForgetting_desmat(
    events_file, duration_choice, add_deriv, regress_rt, n_scans, tr, 
    confound_regressors
):
    """Creates basic directed forgetting regressors (and derivatives), 
    defining error regressors and adding fmriprep confound regressors
       Input
         events_df: events data frame (events.tsv data)
         n_scans: Number of scans
         tr: time resolution
         confound_regressors: Confounds derived from fmriprep output
       Returns
          design_matrix: pd data frame including all regressors (and derivatives)
          contrasts: dictionary of contrasts in nilearn friendly format
          rt_subset: Boolean for extracting correct rows of events_df in the case that 
              rt regressors are requeset.
    """
    events_df = pd.read_csv(events_file, sep = '\t')
    events_df['junk_trials'], events_df['omission'], events_df['commission'], events_df['rt_fast'] = \
        define_nuisance_trials(events_df, 'directedForgetting')
    percent_junk = np.mean(events_df['junk_trials'])
    events_df['constant_1_column'] = 1  
    omission_regressor = make_regressor_and_derivative(
            n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
            amplitude_column="omission", duration_column="constant_1_column",
            subset='trial_type != "memory_cue"', demean_amp = False, cond_id = 'omission'
        )
    commission_regressor = make_regressor_and_derivative(
            n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
            amplitude_column="commission", duration_column="constant_1_column",
            subset='trial_type != "memory_cue"', demean_amp = False, cond_id = 'commission'
        )
    rt_fast = make_regressor_and_derivative(
            n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
            amplitude_column="rt_fast", duration_column="constant_1_column",
            subset='trial_type != "memory_cue"', demean_amp = False, cond_id = 'rt_fast'
        )
    
    #added for new way of modeling
    memory_and_cue = make_regressor_and_derivative(
            n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
            amplitude_column="constant_1_column", duration_column="duration",
            subset='trial_type == "memory_cue"', demean_amp = False, cond_id = 'memory_and_cue'
    )

    rt_subset = 'key_press == correct_response and response_time >= 0.2'
    if duration_choice == 'mean_rt':
        mean_rt = events_df.query(rt_subset)['response_time'].mean()
        events_df['constant_column'] = events_df['constant_1_column'] * mean_rt
    else:
        events_df['constant_column'] = events_df['constant_1_column'] 

    con = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
        amplitude_column="constant_1_column", duration_column="constant_column",
        subset='key_press == correct_response and response_time >= 0.2 and trial_type == "con"', demean_amp=False, cond_id ='con'
    )
    pos = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
        amplitude_column="constant_1_column", duration_column="constant_column",
        subset='key_press == correct_response and response_time >= 0.2 and trial_type == "pos"', demean_amp=False, cond_id ='pos'
    )
    neg = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
        amplitude_column="constant_1_column", duration_column="constant_column",
        subset='key_press == correct_response and response_time >= 0.2 and trial_type == "neg"', demean_amp=False, cond_id ='neg'
    )
    design_matrix = pd.concat([con, pos, neg,
                               omission_regressor, commission_regressor, rt_fast,
                               confound_regressors, memory_and_cue], axis=1) #memory_and_cue
    
    contrasts = {
        "neg-con": "neg-con",
        "task-baseline": "1/4*(con+pos+neg+memory_and_cue)"#memory_and_cue
        }
    
    if regress_rt == 'rt_centered':
        mn_rt = mean_rt_dict['directedForgetting']
        events_df['response_time_centered'] = events_df.response_time - mn_rt
        rt = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
        amplitude_column="response_time_centered", duration_column="constant_column",
        subset=rt_subset, demean_amp=False, cond_id='response_time'
        ) 
        design_matrix = pd.concat([design_matrix, rt], axis=1)
        contrasts["response_time"] = "response_time"
    if regress_rt == 'rt_uncentered':
        rt = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
        amplitude_column="response_time", duration_column="constant_column",
        subset=rt_subset, demean_amp=False, cond_id='response_time'
        ) 
        design_matrix = pd.concat([design_matrix, rt], axis=1)
        contrasts["response_time"] = "response_time"
    return design_matrix, contrasts, percent_junk, events_df


def make_basic_flanker_desmat(events_file, duration_choice, add_deriv,
    regress_rt, n_scans, tr, confound_regressors
):
    """Creates basic flanker regressors (and derivatives), 
    defining error regressors and adding fmriprep confound regressors
       Input
         events_df: events data frame (events.tsv data)
         n_scans: Number of scans
         tr: time resolution
         confound_regressors: Confounds derived from fmriprep output
       Returns
          design_matrix: pd data frame including all regressors (and derivatives)
          contrasts: dictionary of contrasts in nilearn friendly format
          rt_subset: Boolean for extracting correct rows of events_df in the case that 
              rt regressors are requeset.
    """
    events_df = pd.read_csv(events_file, sep = '\t')
    events_df['junk_trials'], events_df['omission'], events_df['commission'], events_df['rt_fast'] = \
        define_nuisance_trials(events_df, 'flanker')
    percent_junk = np.mean(events_df['junk_trials'])
    events_df['constant_1_column'] = 1  
    omission_regressor = make_regressor_and_derivative(
            n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
            amplitude_column="omission", duration_column="constant_1_column",
            demean_amp = False, cond_id = 'omission'
        )
    commission_regressor = make_regressor_and_derivative(
            n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
            amplitude_column="commission", duration_column="constant_1_column",
            demean_amp = False, cond_id = 'commission'
        )
    rt_fast = make_regressor_and_derivative(
            n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
            amplitude_column="rt_fast", duration_column="constant_1_column",
            demean_amp = False, cond_id = 'rt_fast'
        )
    
    rt_subset = 'key_press == correct_response and response_time >= 0.2'
    if duration_choice == 'mean_rt':
        mean_rt = events_df.query(rt_subset)['response_time'].mean()
        events_df['constant_column'] = events_df['constant_1_column'] * mean_rt
    else:
        events_df['constant_column'] = events_df['constant_1_column'] 
    
    congruent = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
        amplitude_column="constant_1_column", duration_column="constant_column",
        subset="key_press == correct_response and response_time >= 0.2 and trial_type =='congruent'", demean_amp=False, cond_id='congruent'
    )
    
    incongruent = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
        amplitude_column="constant_1_column", duration_column="constant_column",
        subset="key_press == correct_response and response_time >= 0.2 and trial_type =='incongruent'", demean_amp=False, cond_id='incongruent'
    )
    
    design_matrix = pd.concat([congruent, incongruent,
                               omission_regressor, commission_regressor, rt_fast, 
                               confound_regressors], axis=1)
    contrasts = {'incongruent - congruent': 'incongruent - congruent',
                'task-baseline': '.5*congruent + .5*incongruent'#
                }
    if regress_rt == 'rt_centered':
        mn_rt = mean_rt_dict['flanker']
        events_df['response_time_centered'] = events_df.response_time - mn_rt
        rt = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
        amplitude_column="response_time_centered", duration_column="constant_column",
        subset=rt_subset, demean_amp=False, cond_id='response_time'
        ) 
        design_matrix = pd.concat([design_matrix, rt], axis=1)
        contrasts["response_time"] = "response_time"
    if regress_rt == 'rt_uncentered':
        rt = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
        amplitude_column="response_time", duration_column="constant_column",
        subset=rt_subset, demean_amp=False, cond_id='response_time'
        ) 
        design_matrix = pd.concat([design_matrix, rt], axis=1)
        contrasts["response_time"] = "response_time"
    return design_matrix, contrasts, percent_junk, events_df

def make_basic_goNogo_desmat(events_file, duration_choice, add_deriv, 
    regress_rt, n_scans, tr, confound_regressors
):
    """Creates basic go-nogo regressors (and derivatives), 
    defining error regressors and adding fmriprep confound regressors
       Input
         events_df: events data frame (events.tsv data)
         n_scans: Number of scans
         tr: time resolution
         confound_regressors: Confounds derived from fmriprep output
       Returns
          design_matrix: pd data frame including all regressors (and derivatives)
          contrasts: dictionary of contrasts in nilearn friendly format
          rt_subset: Boolean for extracting correct rows of events_df in the case that 
              rt regressors are requeset.
    """
    events_df = pd.read_csv(events_file, sep = '\t')
    events_df['junk_trials'], events_df['omission'], events_df['commission'], events_df['rt_fast'] = \
        define_nuisance_trials(events_df, 'goNogo')
    percent_junk = np.mean(events_df['junk_trials'])
    events_df['constant_1_column'] = 1
    
    go_omission_regressor = make_regressor_and_derivative(
            n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
            amplitude_column="omission", duration_column="constant_1_column",
            subset="junk == 0", demean_amp = False, cond_id = 'go_omission'
        )
    go_commission_regressor = make_regressor_and_derivative(
            n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
            amplitude_column="commission", duration_column="constant_1_column",
            subset="junk == 0", demean_amp = False, cond_id = 'go_commission'    
    )
    go_rt_fast = make_regressor_and_derivative(
            n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
            amplitude_column="rt_fast", duration_column="constant_1_column",
            subset="junk == 0", demean_amp = False, cond_id = 'go_rt_fast' 
    )
    nogo_failure = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
        amplitude_column="constant_1_column", duration_column="constant_1_column",
        subset="trial_type == 'nogo_failure' and junk == 0", demean_amp=False, 
        cond_id='nogo_failure'
    )
    junk_regressor = make_regressor_and_derivative(
            n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
            amplitude_column="junk", duration_column="constant_1_column",
            subset=None, demean_amp = False, cond_id = 'junk'
        )
    rt_subset = 'key_press == correct_response and trial_type == "go" and junk == 0'
    if duration_choice == 'mean_rt':
        mean_rt = events_df.query(rt_subset)['response_time'].mean()
        events_df['constant_column'] = events_df['constant_1_column'] * mean_rt
    else:
        events_df['constant_column'] = events_df['constant_1_column']
    go = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
        amplitude_column="constant_1_column", duration_column="constant_column",
        subset="key_press == correct_response and response_time >= 0.2 and trial_type == 'go' and junk == 0", demean_amp=False, 
        cond_id='go'
    )
    nogo_success = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
        amplitude_column="constant_1_column", duration_column="constant_column",
        subset="trial_type == 'nogo_success' and junk == 0", demean_amp=False, 
        cond_id='nogo_success'
    )
    
    design_matrix = pd.concat([go, nogo_success, 
                               nogo_failure, go_omission_regressor, go_commission_regressor, go_rt_fast,
                               confound_regressors, junk_regressor], axis=1)
    contrasts = {'go': 'go', #
                 'nogo_success': 'nogo_success',#
                 'nogo_success-go': 'nogo_success-go',
                 'task-baseline': '.5*go + .5*nogo_success'#
                  }
    if regress_rt == 'rt_centered':
        mn_rt = mean_rt_dict['goNogo']
        events_df['response_time_centered'] = events_df.response_time - mn_rt
        rt = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
        amplitude_column="response_time_centered", duration_column="constant_column",
        subset=rt_subset, demean_amp=False, cond_id='response_time'
        ) 
        design_matrix = pd.concat([design_matrix, rt], axis=1)
        contrasts["response_time"] = "response_time"
    if regress_rt == 'rt_uncentered':
        rt = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
        amplitude_column="response_time", duration_column="constant_column",
        subset=rt_subset, demean_amp=False, cond_id='response_time'
        ) 
        design_matrix = pd.concat([design_matrix, rt], axis=1)
        contrasts["response_time"] = "response_time"
    return design_matrix, contrasts, percent_junk, events_df

def make_basic_nBack_desmat(events_file, duration_choice, add_deriv, 
    regress_rt, n_scans, tr, confound_regressors
):
    """Creates basic nback regressors (and derivatives), 
    defining error regressors and adding fmriprep confound regressors
       Input
         events_df: events data frame (events.tsv data)
         n_scans: Number of scans
         tr: time resolution
         confound_regressors: Confounds derived from fmriprep output
       Returns
          design_matrix: pd data frame including all regressors (and derivatives)
          contrasts: dictionary of contrasts in nilearn friendly format
          rt_subset: Boolean for extracting correct rows of events_df in the case that 
              rt regressors are requeset.
    """
    events_df = pd.read_csv(events_file, sep = '\t')
    events_df['junk_trials'], events_df['omission'], events_df['commission'], events_df['rt_fast'] = \
        define_nuisance_trials(events_df, 'nBack')
    percent_junk = np.mean(events_df['junk_trials'])
    events_df['constant_1_column'] = 1
    
    #defined so that omission & commission during n/a trials are not included in nuissance regressors twice
    omission_regressor = make_regressor_and_derivative(
            n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
            amplitude_column="omission", duration_column="constant_1_column",
            subset="na_trials == 0", demean_amp = False, 
            cond_id = 'omission'
        )
    commission_regressor = make_regressor_and_derivative(
            n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
            amplitude_column="commission", duration_column="constant_1_column",
            subset="na_trials == 0", demean_amp = False, 
            cond_id = 'commission'    
    )
    rt_fast = make_regressor_and_derivative(
            n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
            amplitude_column="rt_fast", duration_column="constant_1_column",
            subset="na_trials == 0", demean_amp = False, 
            cond_id = 'rt_fast' 
    )
    na_trials = make_regressor_and_derivative(
            n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
            amplitude_column="na_trials", duration_column="constant_1_column",
            subset=None, demean_amp = False, 
            cond_id = 'na_trials' 
    )
    rt_subset = 'key_press == correct_response and na_trials == 0'
    if duration_choice == 'mean_rt':
        mean_rt = events_df.query(rt_subset)['response_time'].mean()
        events_df['constant_column'] = events_df['constant_1_column'] * mean_rt
    else:
        events_df['constant_column'] = events_df['constant_1_column']
        
    mismatch_1back = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
        amplitude_column="constant_1_column", duration_column="constant_column",
        subset="key_press == correct_response and response_time >= 0.2 and trial_type == 'mismatch' and delay == 1", demean_amp=False, 
        cond_id='mismatch_1back'
    )
    match_1back = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
        amplitude_column="constant_1_column", duration_column="constant_column",
        subset="key_press == correct_response and response_time >= 0.2 and trial_type == 'match' and delay == 1", demean_amp=False, 
        cond_id='match_1back'
    )
    mismatch_2back = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
        amplitude_column="constant_1_column", duration_column="constant_column",
        subset="key_press == correct_response and response_time >= 0.2 and trial_type == 'mismatch' and delay == 2", demean_amp=False, 
        cond_id='mismatch_2back'
    )
    match_2back = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
        amplitude_column="constant_1_column", duration_column="constant_column",
        subset="key_press == correct_response and response_time >= 0.2 and trial_type == 'match' and delay == 2", demean_amp=False, 
        cond_id='match_2back'
    )
    design_matrix = pd.concat([mismatch_1back, match_1back, mismatch_2back, match_2back,
                               omission_regressor, commission_regressor, rt_fast,  na_trials,
                               confound_regressors], axis=1)
    contrasts = {'twoBack-oneBack': 'mismatch_2back + match_2back - mismatch_1back - match_1back',
                 'match - mismatch': 'match_2back + match_1back - mismatch_2back - mismatch_1back',
                 'task-baseline': '1/4*(mismatch_1back + match_1back + mismatch_2back + match_2back)'#
                  }
    if regress_rt == 'rt_centered':
        mn_rt = mean_rt_dict['nBack']
        events_df['response_time_centered'] = events_df.response_time - mn_rt
        rt = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
        amplitude_column="response_time_centered", duration_column="constant_column",
        subset=rt_subset, demean_amp=False, cond_id='response_time'
        ) 
        design_matrix = pd.concat([design_matrix, rt], axis=1)
        contrasts["response_time"] = "response_time"
    if regress_rt == 'rt_uncentered':
        rt = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
        amplitude_column="response_time", duration_column="constant_column",
        subset=rt_subset, demean_amp=False, cond_id='response_time'
        ) 
        design_matrix = pd.concat([design_matrix, rt], axis=1)
        contrasts["response_time"] = "response_time"
    return design_matrix, contrasts, percent_junk, events_df

def make_basic_stopSignal_desmat(events_file, duration_choice, add_deriv, 
    regress_rt, n_scans, tr, confound_regressors
):
    """Creates basic stop signal regressors (and derivatives), 
    defining error regressors and adding fmriprep confound regressors
       Input
         events_df: events data frame (events.tsv data)
         n_scans: Number of scans
         tr: time resolution
         confound_regressors: Confounds derived from fmriprep output
       Returns
          design_matrix: pd data frame including all regressors (and derivatives)
          contrasts: dictionary of contrasts in nilearn friendly format
          rt_subset: Boolean for extracting correct rows of events_df in the case that 
              rt regressors are requeset.
    """
    events_df = pd.read_csv(events_file, sep = '\t')
    events_df['junk_trials'], events_df['omission'], events_df['commission'], events_df['rt_fast'] = \
        define_nuisance_trials(events_df, 'stopSignal')
    percent_junk = np.mean(events_df['junk_trials'])
    events_df['constant_1_column'] = 1
    go_omission_regressor = make_regressor_and_derivative(
            n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
            amplitude_column="omission", duration_column="constant_1_column",
            subset=None, demean_amp = False, 
            cond_id = 'go_omission'
        )
    go_commission_regressor = make_regressor_and_derivative(
            n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
            amplitude_column="commission", duration_column="constant_1_column",
            subset=None, demean_amp = False, 
            cond_id = 'go_commission'    
    )
    go_rt_fast = make_regressor_and_derivative(
            n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
            amplitude_column="rt_fast", duration_column="constant_1_column",
            subset=None, demean_amp = False, 
            cond_id = 'go_rt_fast' 
    )
    rt_subset = 'key_press == correct_response and trial_type == "go" and response_time >= 0.2'
    if duration_choice == 'mean_rt':
        mean_rt = events_df.query(rt_subset)['response_time'].mean()
        events_df['constant_column'] = events_df['constant_1_column'] * mean_rt
    else:
        events_df['constant_column'] = events_df['constant_1_column']
    go = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
        amplitude_column="constant_1_column", duration_column="constant_column",
        subset="key_press == correct_response and response_time >= 0.2 and trial_type == 'go'", demean_amp=False, 
        cond_id='go'
    )
    stop_success = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
        amplitude_column="constant_1_column", duration_column="constant_column",
        subset="trial_type == 'stop_success'", demean_amp=False, 
        cond_id='stop_success'
    )
    stop_failure = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
        amplitude_column="constant_1_column", duration_column="constant_column",
        subset="trial_type == 'stop_failure'", demean_amp=False, 
        cond_id='stop_failure'
    )
    design_matrix = pd.concat([go, stop_success, stop_failure, 
                               go_omission_regressor, go_commission_regressor, go_rt_fast,
                               confound_regressors], axis=1)
    contrasts = {'go': 'go', #
                 'stop_success': 'stop_success',#
                 'stop_failure': 'stop_failure',#
                 'stop_success-go': 'stop_success-go',
                 'stop_failure-go': 'stop_failure-go',
                 'stop_success-stop_failure': 'stop_success-stop_failure',
                 'stop_failure-stop_success': 'stop_failure-stop_success',
                 'task-baseline': '1/3*go + 1/3*stop_failure + 1/3*stop_success'#
                  }
    if regress_rt == 'rt_centered':
        mn_rt = mean_rt_dict['stopSignal']
        events_df['response_time_centered'] = events_df.response_time - mn_rt
        rt = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
        amplitude_column="response_time_centered", duration_column="constant_column",
        subset=rt_subset, demean_amp=False, cond_id='response_time'
        ) 
        design_matrix = pd.concat([design_matrix, rt], axis=1)
        contrasts["response_time"] = "response_time"
    if regress_rt == 'rt_uncentered':
        rt = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
        amplitude_column="response_time", duration_column="constant_column",
        subset=rt_subset, demean_amp=False, cond_id='response_time'
        ) 
        design_matrix = pd.concat([design_matrix, rt], axis=1)
        contrasts["response_time"] = "response_time"
    return design_matrix, contrasts, percent_junk, events_df

def make_basic_shapeMatching_desmat(events_file, duration_choice, add_deriv,
    regress_rt, n_scans, tr, confound_regressors
):
    """Creates basic shape matching regressors (and derivatives), 
    defining error regressors and adding fmriprep confound regressors
       Input
         events_df: events data frame (events.tsv data)
         n_scans: Number of scans
         tr: time resolution
         confound_regressors: Confounds derived from fmriprep output
       Returns
          design_matrix: pd data frame including all regressors (and derivatives)
          contrasts: dictionary of contrasts in nilearn friendly format
          rt_subset: Boolean for extracting correct rows of events_df in the case that 
              rt regressors are requeset.
    """
    events_df = pd.read_csv(events_file, sep = '\t')
    events_df['junk_trials'], events_df['omission'], events_df['commission'], events_df['rt_fast'] = \
        define_nuisance_trials(events_df, 'shapeMatching')
    percent_junk = np.mean(events_df['junk_trials'])
    events_df['constant_1_column'] = 1  
    omission_regressor = make_regressor_and_derivative(
            n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
            amplitude_column="omission", duration_column="constant_1_column",
            subset=None, demean_amp = False, cond_id = 'omission'
        )
    commission_regressor = make_regressor_and_derivative(
            n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
            amplitude_column="commission", duration_column="constant_1_column",
            subset=None, demean_amp = False, cond_id = 'commission'
        )
    rt_fast = make_regressor_and_derivative(
            n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
            amplitude_column="rt_fast", duration_column="constant_1_column",
            subset=None, demean_amp = False, cond_id = 'rt_fast'
        )

    rt_subset = "key_press == correct_response and response_time >= 0.2"
    if duration_choice == 'mean_rt':
        mean_rt = events_df.query(rt_subset)['response_time'].mean()
        events_df['constant_column'] = events_df['constant_1_column'] * mean_rt
    else:
        events_df['constant_column'] = events_df['constant_1_column']
        
    SSS = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
        amplitude_column="constant_1_column", duration_column="constant_column",
        subset="key_press == correct_response and response_time >= 0.2 and trial_type == 'SSS'", demean_amp=False, 
        cond_id='SSS'
    )        
    SDD = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
        amplitude_column="constant_1_column", duration_column="constant_column",
        subset="key_press == correct_response and response_time >= 0.2 and trial_type == 'SDD'", demean_amp=False, 
        cond_id='SDD'
    )
    SNN = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
        amplitude_column="constant_1_column", duration_column="constant_column",
        subset="key_press == correct_response and response_time >= 0.2 and trial_type == 'SNN'", demean_amp=False, 
        cond_id='SNN'
    )  
    DSD = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
        amplitude_column="constant_1_column", duration_column="constant_column",
        subset="key_press == correct_response and response_time >= 0.2 and trial_type == 'DSD'", demean_amp=False, 
        cond_id='DSD'
    )
    DNN = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
        amplitude_column="constant_1_column", duration_column="constant_column",
        subset="key_press == correct_response and response_time >= 0.2 and trial_type == 'DNN'", demean_amp=False, 
        cond_id='DNN'
    )
    DDD = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
        amplitude_column="constant_1_column", duration_column="constant_column",
        subset="key_press == correct_response and response_time >= 0.2 and trial_type == 'DDD'", 
        demean_amp=False, cond_id='DDD'
    )
    DDS = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
        amplitude_column="constant_1_column", duration_column="constant_column",
        subset="key_press == correct_response and response_time >= 0.2 and trial_type == 'DDS'", demean_amp=False, 
        cond_id='DDS'
    )
    design_matrix = pd.concat([SSS, SDD, SNN, DSD, DDD, DDS, DNN,
                               omission_regressor, commission_regressor, rt_fast,
                               confound_regressors], axis=1)
    contrasts = {"task-baseline":"1/7*(SSS+SDD+SNN+DSD+DDD+DDS+DNN)",
                "main_vars":"1/3*(SDD+DDD+DDS)-1/2*(SNN+DNN)",
                "SSS":"SSS",
                "SDD":"SDD",
                "SNN":"SNN",
                "DSD":"DSD",
                "DDD":"DDD",
                "DDS":"DDS",
                "DNN":"DNN"}
    
    if regress_rt == 'rt_centered':
        mn_rt = mean_rt_dict['shapeMatching']
        events_df['response_time_centered'] = events_df.response_time - mn_rt
        rt = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
        amplitude_column="response_time_centered", duration_column="constant_column",
        subset=rt_subset, demean_amp=False, cond_id='response_time'
        ) 
        design_matrix = pd.concat([design_matrix, rt], axis=1)
        contrasts["response_time"] = "response_time"
    if regress_rt == 'rt_uncentered':
        rt = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
        amplitude_column="response_time", duration_column="constant_column",
        subset=rt_subset, demean_amp=False, cond_id='response_time'
        ) 
        design_matrix = pd.concat([design_matrix, rt], axis=1)
        contrasts["response_time"] = "response_time"
    return design_matrix, contrasts, percent_junk, events_df


def make_basic_spatialTS_desmat(events_file, duration_choice, add_deriv,
    regress_rt, n_scans, tr, confound_regressors
):
    """Creates basic spatial task switching regressors (and derivatives), 
    defining error regressors and adding fmriprep confound regressors
       Input
         events_df: events data frame (events.tsv data)
         n_scans: Number of scans
         tr: time resolution
         confound_regressors: Confounds derived from fmriprep output
       Returns
          design_matrix: pd data frame including all regressors (and derivatives)
          contrasts: dictionary of contrasts in nilearn friendly format
          rt_subset: Boolean for extracting correct rows of events_df in the case that 
              rt regressors are requeset.
    """
    events_df = pd.read_csv(events_file, sep = '\t')
    events_df['junk_trials'], events_df['omission'], events_df['commission'], events_df['rt_fast'] = \
        define_nuisance_trials(events_df, 'spatialTS')
    percent_junk = np.mean(events_df['junk_trials'])
    events_df['constant_1_column'] = 1  
    omission_regressor = make_regressor_and_derivative(
            n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
            amplitude_column="omission", duration_column="constant_1_column",
            subset="trial_type != 'na'", demean_amp = False, cond_id = 'omission'
        )
    commission_regressor = make_regressor_and_derivative(
            n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
            amplitude_column="commission", duration_column="constant_1_column",
            subset="trial_type != 'na'", demean_amp = False, cond_id = 'commission'
        )
    rt_fast = make_regressor_and_derivative(
            n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
            amplitude_column="rt_fast", duration_column="constant_1_column",
            subset="trial_type != 'na'", demean_amp = False, cond_id = 'rt_fast'
        )
    na_trials = make_regressor_and_derivative(
            n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
            amplitude_column="constant_1_column", duration_column="constant_1_column",
            subset="trial_type == 'na'", demean_amp = False, cond_id = 'na_trials'
        )

    rt_subset = "key_press == correct_response and response_time >= 0.2 and trial_type != 'na'"
    if duration_choice == 'mean_rt':
        mean_rt = events_df.query(rt_subset)['response_time'].mean()
        events_df['constant_column'] = events_df['constant_1_column'] * mean_rt
    else:
        events_df['constant_column'] = events_df['constant_1_column']
        
    task_stay_cue_switch = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
        amplitude_column="constant_1_column", duration_column="constant_column",
        subset="key_press == correct_response and response_time >= 0.2 and trial_type == 'tstay_cswitch'", demean_amp=False, 
        cond_id='task_stay_cue_switch'
    )
    task_stay_cue_stay = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
        amplitude_column="constant_1_column", duration_column="constant_column",
        subset="key_press == correct_response and response_time >= 0.2 and trial_type == 'tstay_cstay'", demean_amp=False, 
        cond_id='task_stay_cue_stay'
    )
    task_switch_cue_switch = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
        amplitude_column="constant_1_column", duration_column="constant_column",
        subset="key_press == correct_response and response_time >= 0.2 and trial_type == 'tswitch_cswitch'", 
        demean_amp=False, cond_id='task_switch_cue_switch'
    )
    design_matrix = pd.concat([task_stay_cue_switch, task_stay_cue_stay, task_switch_cue_switch,
                               omission_regressor, commission_regressor, rt_fast, na_trials,
                               confound_regressors], axis=1)
    contrasts = {'task_switch_cost': 'task_switch_cue_switch-task_stay_cue_switch',
                'cue_switch_cost': 'task_stay_cue_switch-task_stay_cue_stay',
                 'task-baseline': '1/3*(task_stay_cue_switch+task_stay_cue_stay+task_switch_cue_switch)'}
    
    if regress_rt == 'rt_centered':
        mn_rt = mean_rt_dict['spatialTS']
        events_df['response_time_centered'] = events_df.response_time - mn_rt
        rt = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
        amplitude_column="response_time_centered", duration_column="constant_column",
        subset=rt_subset, demean_amp=False, cond_id='response_time'
        ) 
        design_matrix = pd.concat([design_matrix, rt], axis=1)
        contrasts["response_time"] = "response_time"
    if regress_rt == 'rt_uncentered':
        rt = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
        amplitude_column="response_time", duration_column="constant_column",
        subset=rt_subset, demean_amp=False, cond_id='response_time'
        ) 
        design_matrix = pd.concat([design_matrix, rt], axis=1)
        contrasts["response_time"] = "response_time"
    return design_matrix, contrasts, percent_junk, events_df

def make_basic_directedForgettingWFlanker_desmat(
    events_file, duration_choice, add_deriv, regress_rt, n_scans, tr, 
    confound_regressors
):
    """Creates basic directed forgetting + flanker regressors (and derivatives), 
    defining error regressors and adding fmriprep confound regressors
       Input
         events_df: events data frame (events.tsv data)
         n_scans: Number of scans
         tr: time resolution
         confound_regressors: Confounds derived from fmriprep output
       Returns
          design_matrix: pd data frame including all regressors (and derivatives)
          contrasts: dictionary of contrasts in nilearn friendly format
          rt_subset: Boolean for extracting correct rows of events_df in the case that 
              rt regressors are requeset.
    """
    events_df = pd.read_csv(events_file, sep = '\t')
    events_df['junk_trials'], events_df['omission'], events_df['commission'], events_df['rt_fast'] = \
        define_nuisance_trials(events_df, 'directedForgettingWFlanker')
    percent_junk = np.mean(events_df['junk_trials'])
    events_df['constant_1_column'] = 1  
    
    omission_regressor = make_regressor_and_derivative(
            n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
            amplitude_column="omission", duration_column="constant_1_column",
            subset='trial_type != "memory_cue"', demean_amp = False, cond_id = 'omission'
        )
    commission_regressor = make_regressor_and_derivative(
            n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
            amplitude_column="commission", duration_column="constant_1_column",
            subset='trial_type != "memory_cue"', demean_amp = False, cond_id = 'commission'
        )
    rt_fast = make_regressor_and_derivative(
            n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
            amplitude_column="rt_fast", duration_column="constant_1_column",
            subset='trial_type != "memory_cue"', demean_amp = False, cond_id = 'rt_fast'
        )
    #added for new way of modeling
    memory_and_cue = make_regressor_and_derivative(
            n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
            amplitude_column="constant_1_column", duration_column="duration",
            subset='trial_type == "memory_cue"', demean_amp = False, cond_id = 'memory_and_cue'
    )

    rt_subset = 'key_press == correct_response and response_time >= 0.2'
    if duration_choice == 'mean_rt':
        mean_rt = events_df.query(rt_subset)['response_time'].mean()
        events_df['constant_column'] = events_df['constant_1_column'] * mean_rt
    else:
        events_df['constant_column'] = events_df['constant_1_column'] 

    incongruent_con = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
        amplitude_column="constant_1_column", duration_column="constant_column",
        subset='key_press == correct_response and response_time >= 0.2 and trial_type == "incongruent_con"', demean_amp=False, cond_id ='incongruent_con'
    )
    incongruent_pos = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
        amplitude_column="constant_1_column", duration_column="constant_column",
        subset='key_press == correct_response and response_time >= 0.2 and trial_type == "incongruent_pos"', demean_amp=False, cond_id ='incongruent_pos'
    )
    incongruent_neg = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
        amplitude_column="constant_1_column", duration_column="constant_column",
        subset='key_press == correct_response and response_time >= 0.2 and trial_type == "incongruent_neg"', demean_amp=False, cond_id ='incongruent_neg'
    )
    congruent_con = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
        amplitude_column="constant_1_column", duration_column="constant_column",
        subset='key_press == correct_response and response_time >= 0.2 and trial_type == "congruent_con"', demean_amp=False, cond_id ='congruent_con'
    )
    congruent_pos = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
        amplitude_column="constant_1_column", duration_column="constant_column",
        subset='key_press == correct_response and response_time >= 0.2 and trial_type == "congruent_pos"', demean_amp=False, cond_id ='congruent_pos'
    )
    congruent_neg = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
        amplitude_column="constant_1_column", duration_column="constant_column",
        subset='key_press == correct_response and response_time >= 0.2 and trial_type == "congruent_neg"', demean_amp=False, cond_id ='congruent_neg'
    )
    design_matrix = pd.concat([incongruent_con, incongruent_pos, incongruent_neg,
                               congruent_con, congruent_pos, congruent_neg,
                               omission_regressor, commission_regressor, rt_fast,
                               confound_regressors, memory_and_cue], axis=1)#memory_and_cue
    
    contrasts = {
        "congruent_neg-congruent_con": "congruent_neg-congruent_con",
        "incongruent_con-congruent_con":"incongruent_con-congruent_con",
        "(incongruent_neg-incongruent_con)-(congruent_neg-congruent_con)":"(incongruent_neg+congruent_con) -(incongruent_con+congruent_neg)",
        "congruent_pos":"congruent_pos",
        "congruent_neg":"congruent_neg",
        "congruent_con":"congruent_con",
        "incongruent_pos":"incongruent_pos",
        "incongruent_neg":"incongruent_neg",
        "incongruent_con":"incongruent_con",
        "task-baseline": "1/7*(congruent_pos+congruent_neg+congruent_con+incongruent_pos+incongruent_neg+incongruent_con+memory_and_cue)"#memory_and_cue
        }
    
    if regress_rt == 'rt_centered':
        mn_rt = mean_rt_dict['directedForgettingWFlanker']
        events_df['response_time_centered'] = events_df.response_time - mn_rt
        rt = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
        amplitude_column="response_time_centered", duration_column="constant_column",
        subset=rt_subset, demean_amp=False, cond_id='response_time'
        ) 
        design_matrix = pd.concat([design_matrix, rt], axis=1)
        contrasts["response_time"] = "response_time"
    if regress_rt == 'rt_uncentered':
        rt = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
        amplitude_column="response_time", duration_column="constant_column",
        subset=rt_subset, demean_amp=False, cond_id='response_time'
        ) 
        design_matrix = pd.concat([design_matrix, rt], axis=1)
        contrasts["response_time"] = "response_time"
    return design_matrix, contrasts, percent_junk, events_df

def make_basic_stopSignalWDirectedForgetting_desmat(events_file, duration_choice, add_deriv, 
    regress_rt, n_scans, tr, confound_regressors
):
    """Creates basic stop signal + DF regressors (and derivatives), 
    defining error regressors and adding fmriprep confound regressors
       Input
         events_df: events data frame (events.tsv data)
         n_scans: Number of scans
         tr: time resolution
         confound_regressors: Confounds derived from fmriprep output
       Returns
          design_matrix: pd data frame including all regressors (and derivatives)
          contrasts: dictionary of contrasts in nilearn friendly format
          rt_subset: Boolean for extracting correct rows of events_df in the case that 
              rt regressors are requeset.
    """
    events_df = pd.read_csv(events_file, sep = '\t')
    events_df['junk_trials'], events_df['omission'], events_df['commission'], events_df['rt_fast'] = \
        define_nuisance_trials(events_df, 'stopSignalWDirectedForgetting')
    percent_junk = np.mean(events_df['junk_trials'])
    events_df['constant_1_column'] = 1

    go_omission_regressor = make_regressor_and_derivative(
            n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
            amplitude_column="omission", duration_column="constant_1_column",
            subset='trial_type != "memory_cue"', demean_amp = False, 
            cond_id = 'go_omission'
        )
    go_commission_regressor = make_regressor_and_derivative(
            n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
            amplitude_column="commission", duration_column="constant_1_column",
            subset='trial_type != "memory_cue"', demean_amp = False, 
            cond_id = 'go_commission'    
    )
    go_rt_fast = make_regressor_and_derivative(
            n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
            amplitude_column="rt_fast", duration_column="constant_1_column",
            subset='trial_type != "memory_cue"', demean_amp = False, 
            cond_id = 'go_rt_fast' 
    )
    
    #added for new way of modeling
    memory_and_cue = make_regressor_and_derivative(
            n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
            amplitude_column="constant_1_column", duration_column="duration",
            subset='trial_type == "memory_cue"', demean_amp = False, cond_id = 'memory_and_cue'
    )
    
    rt_subset = "key_press == correct_response and trial_type in ['go_pos', 'go_neg', 'go_con'] and response_time >= 0.2"
    if duration_choice == 'mean_rt':
        mean_rt = events_df.query(rt_subset)['response_time'].mean()
        events_df['constant_column'] = events_df['constant_1_column'] * mean_rt
    else:
        events_df['constant_column'] = events_df['constant_1_column']
    go_pos = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
        amplitude_column="constant_1_column", duration_column="constant_column",
        subset="key_press == correct_response and response_time >= 0.2 and trial_type == 'go_pos'", demean_amp=False, 
        cond_id='go_pos'
    )
    go_neg = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
        amplitude_column="constant_1_column", duration_column="constant_column",
        subset="key_press == correct_response and response_time >= 0.2 and trial_type == 'go_neg'", demean_amp=False, 
        cond_id='go_neg'
    )
    go_con = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
        amplitude_column="constant_1_column", duration_column="constant_column",
        subset="key_press == correct_response and response_time >= 0.2 and trial_type == 'go_con'", demean_amp=False, 
        cond_id='go_con'
    )
    stop_success_pos = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
        amplitude_column="constant_1_column", duration_column="constant_column",
        subset="trial_type == 'stop_success_pos'", demean_amp=False, 
        cond_id='stop_success_pos'
    )
    stop_success_neg = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
        amplitude_column="constant_1_column", duration_column="constant_column",
        subset="trial_type == 'stop_success_neg'", demean_amp=False, 
        cond_id='stop_success_neg'
    )
    stop_success_con = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
        amplitude_column="constant_1_column", duration_column="constant_column",
        subset="trial_type == 'stop_success_con'", demean_amp=False, 
        cond_id='stop_success_con'
    )
    stop_failure_pos = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
        amplitude_column="constant_1_column", duration_column="constant_column",
        subset="trial_type == 'stop_failure_pos'", demean_amp=False, 
        cond_id='stop_failure_pos'
    )
    stop_failure_neg = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
        amplitude_column="constant_1_column", duration_column="constant_column",
        subset="trial_type == 'stop_failure_neg'", demean_amp=False, 
        cond_id='stop_failure_neg'
    )
    stop_failure_con = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
        amplitude_column="constant_1_column", duration_column="constant_column",
        subset="trial_type == 'stop_failure_con'", demean_amp=False, 
        cond_id='stop_failure_con'
    )
    design_matrix = pd.concat([go_pos, go_neg, go_con,
                               stop_success_pos, stop_success_neg, stop_success_con, 
                               stop_failure_pos, stop_failure_neg, stop_failure_con, 
                               go_omission_regressor, go_commission_regressor, go_rt_fast,
                               confound_regressors, memory_and_cue], axis=1) #memory_and_cue
    contrasts = {"(stop_success_con+stop_success_pos+stop_success_neg)-(go_con+go_pos_+go_neg)":
                 "(stop_success_con+stop_success_pos+stop_success_neg) - (go_con+go_pos+go_neg)",
                 "(stop_failure_con+stop_failure_pos+stop_failure_neg)-(go_con+go_pos_+go_neg)":
                 "(stop_failure_con+stop_failure_pos+stop_failure_neg) - (go_con+go_pos+go_neg)",
                 "(stop_success_neg-go_neg)-(stop_success_con-go_con)":
                 "(stop_success_neg-go_neg)-(stop_success_con-go_con)",
                 "(stop_failure_neg-go_neg)-(stop_failure_con-go_con)":
                 "(stop_failure_neg-go_neg)-(stop_failure_con-go_con)",
                 "go_neg-go_con":"go_neg-go_con",
                 'go_pos':'go_pos',
                 'go_neg':'go_neg',
                 'go_con':'go_con',
                 'stop_success_pos':'stop_success_pos',
                 'stop_success_neg':'stop_success_neg',
                 'stop_success_con':'stop_success_con',
                 'stop_failure_pos':'stop_failure_pos',
                 'stop_failure_neg':'stop_failure_neg',
                 'stop_failure_con':'stop_failure_con',
                 'task-baseline': '1/10*(go_pos+go_neg+go_con+stop_success_pos+stop_success_neg+stop_success_con+stop_failure_pos+stop_failure_neg+stop_failure_con+memory_and_cue)'#memory_and_cue
                  }
    if regress_rt == 'rt_centered':
        mn_rt = mean_rt_dict['stopSignalWDirectedForgetting']
        events_df['response_time_centered'] = events_df.response_time - mn_rt
        rt = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
        amplitude_column="response_time_centered", duration_column="constant_column",
        subset=rt_subset, demean_amp=False, cond_id='response_time'
        ) 
        design_matrix = pd.concat([design_matrix, rt], axis=1)
        contrasts["response_time"] = "response_time"
    if regress_rt == 'rt_uncentered':
        rt = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
        amplitude_column="response_time", duration_column="constant_column",
        subset=rt_subset, demean_amp=False, cond_id='response_time'
        ) 
        design_matrix = pd.concat([design_matrix, rt], axis=1)
        contrasts["response_time"] = "response_time"
    return design_matrix, contrasts, percent_junk, events_df

def make_basic_stopSignalWFlanker_desmat(events_file, duration_choice, add_deriv, 
    regress_rt, n_scans, tr, confound_regressors
):
    """Creates basic stop + flanker regressors (and derivatives), 
    defining error regressors and adding fmriprep confound regressors
       Input
         events_df: events data frame (events.tsv data)
         n_scans: Number of scans
         tr: time resolution
         confound_regressors: Confounds derived from fmriprep output
       Returns
          design_matrix: pd data frame including all regressors (and derivatives)
          contrasts: dictionary of contrasts in nilearn friendly format
          rt_subset: Boolean for extracting correct rows of events_df in the case that 
              rt regressors are requeset.
    """
    events_df = pd.read_csv(events_file, sep = '\t')
    events_df['junk_trials'], events_df['omission'], events_df['commission'], events_df['rt_fast'] = \
        define_nuisance_trials(events_df, 'stopSignalWFlanker')
    percent_junk = np.mean(events_df['junk_trials'])
    events_df['constant_1_column'] = 1
    go_omission_regressor = make_regressor_and_derivative(
            n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
            amplitude_column="omission", duration_column="constant_1_column",
            subset=None, demean_amp = False, 
            cond_id = 'go_omission'
        )
    go_commission_regressor = make_regressor_and_derivative(
            n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
            amplitude_column="commission", duration_column="constant_1_column",
            subset=None, demean_amp = False, 
            cond_id = 'go_commission'    
    )
    go_rt_fast = make_regressor_and_derivative(
            n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
            amplitude_column="rt_fast", duration_column="constant_1_column",
            subset=None, demean_amp = False, 
            cond_id = 'go_rt_fast' 
    )
    rt_subset = "key_press == correct_response and trial_type in ['go_congruent', 'go_incongruent'] and response_time >= 0.2"
    if duration_choice == 'mean_rt':
        mean_rt = events_df.query(rt_subset)['response_time'].mean()
        events_df['constant_column'] = events_df['constant_1_column'] * mean_rt
    else:
        events_df['constant_column'] = events_df['constant_1_column']
    go_congruent = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
        amplitude_column="constant_1_column", duration_column="constant_column",
        subset="key_press == correct_response and response_time >= 0.2 and trial_type == 'go_congruent'", demean_amp=False, 
        cond_id='go_congruent'
    )
    go_incongruent = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
        amplitude_column="constant_1_column", duration_column="constant_column",
        subset="key_press == correct_response and response_time >= 0.2 and trial_type == 'go_incongruent'", demean_amp=False, 
        cond_id='go_incongruent'
    )
    stop_success_congruent = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
        amplitude_column="constant_1_column", duration_column="constant_column",
        subset="trial_type == 'stop_success_congruent'", demean_amp=False, 
        cond_id='stop_success_congruent'
    )
    stop_success_incongruent = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
        amplitude_column="constant_1_column", duration_column="constant_column",
        subset="trial_type == 'stop_success_incongruent'", demean_amp=False, 
        cond_id='stop_success_incongruent'
    )
    stop_failure_congruent = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
        amplitude_column="constant_1_column", duration_column="constant_column",
        subset="trial_type == 'stop_failure_congruent'", demean_amp=False, 
        cond_id='stop_failure_congruent'
    )
    stop_failure_incongruent = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
        amplitude_column="constant_1_column", duration_column="constant_column",
        subset="trial_type == 'stop_failure_incongruent'", demean_amp=False, 
        cond_id='stop_failure_incongruent'
    )

    design_matrix = pd.concat([go_incongruent, go_congruent, stop_success_incongruent, stop_success_congruent,
                               stop_failure_incongruent, stop_failure_congruent, 
                               go_omission_regressor, go_commission_regressor, go_rt_fast,
                               confound_regressors], axis=1)
    contrasts = {"(stop_success_congruent+stop_success_incongruent)-(go_congruent+go_incongruent)":
                 "(stop_success_congruent+stop_success_incongruent)-(go_congruent+go_incongruent)",
                 "(stop_failure_congruent+stop_failure_incongruent)-(go_congruent+go_incongruent)":
                 "(stop_failure_congruent+stop_failure_incongruent)-(go_congruent+go_incongruent)",
                 "(stop_success_incongruent-go_incongruent)-(stop_success_congruent-go_congruent)":
                 "(stop_success_incongruent-go_incongruent)-(stop_success_congruent-go_congruent)",
                 "(stop_failure_incongruent-go_incongruent)-(stop_failure_congruent-go_congruent)":
                 "(stop_failure_incongruent-go_incongruent)-(stop_failure_congruent-go_congruent)",
                 "go_incongruent-go_congruent":"go_incongruent-go_congruent",
                 'go_congruent':'go_congruent',
                 'go_incongruent':'go_incongruent',
                 'stop_success_congruent':'stop_success_congruent',
                 'stop_success_incongruent':'stop_success_incongruent',
                 'stop_failure_congruent':'stop_failure_incongruent',
                 'stop_failure_incongruent':'stop_failure_incongruent',
                 'task-baseline': '1/6*(go_congruent+go_incongruent+stop_success_congruent+stop_success_incongruent+stop_failure_congruent+stop_failure_incongruent)'
                  }
    if regress_rt == 'rt_centered':
        mn_rt = mean_rt_dict['stopSignalWFlanker']
        events_df['response_time_centered'] = events_df.response_time - mn_rt
        rt = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
        amplitude_column="response_time_centered", duration_column="constant_column",
        subset=rt_subset, demean_amp=False, cond_id='response_time'
        ) 
        design_matrix = pd.concat([design_matrix, rt], axis=1)
        contrasts["response_time"] = "response_time"
    if regress_rt == 'rt_uncentered':
        rt = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
        amplitude_column="response_time", duration_column="constant_column",
        subset=rt_subset, demean_amp=False, cond_id='response_time'
        ) 
        design_matrix = pd.concat([design_matrix, rt], axis=1)
        contrasts["response_time"] = "response_time"
    return design_matrix, contrasts, percent_junk, events_df


make_task_desmat_fcn_dict = {
    'cuedTS':make_basic_cuedTS_desmat,
    'directedForgetting':make_basic_directedForgetting_desmat,
    'flanker':make_basic_flanker_desmat,
    'goNogo':make_basic_goNogo_desmat,
    'nBack':make_basic_nBack_desmat,
    'shapeMatching':make_basic_shapeMatching_desmat,
    'spatialTS':make_basic_spatialTS_desmat,
    'stopSignal':make_basic_stopSignal_desmat,
    'directedForgettingWFlanker':make_basic_directedForgettingWFlanker_desmat,
    'stopSignalWDirectedForgetting':make_basic_stopSignalWDirectedForgetting_desmat,
    'stopSignalWFlanker':make_basic_stopSignalWFlanker_desmat
    }

