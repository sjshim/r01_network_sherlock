from nilearn.plotting import plot_design_matrix
from nilearn.glm.contrasts import expression_to_contrast_vector
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import base64
from io import BytesIO
from pathlib import Path
from check_average_TRs import create_tr_dict


def get_behav_exclusion(subid, task, ses):
    behav_exclusion = pd.DataFrame(columns=['subid_task_ses'])
    behav_exclusion_this_sub = behav_exclusion[behav_exclusion['subid_task_ses'].str.contains(f'{subid}_{task}_{ses}')]
    return behav_exclusion_this_sub


def qa_design_matrix(contrast_dir, contrasts, desmat, subid, task, ses, percent_junk=0):
    """
    Check design matrix for regressors that are included in contrasts that have 
    all zeros. >10% junk trials and unusually low number of TRs 
    input:
      contrast_dir: output contrast directory
      contrasts: contrasts to be estimated
      desmat:  design matrix
      subid: subject id number (without 's')
      task: task name 
      percent_junk: percent of junk trials (calculated when design matrix is made)
    return:
      any_fail: True=skip this run due to QA failures, False=design good to go
      error_message: Message explaining why subject was excluded (written to file as well)
      If errors are found, the excluded.csv file is updated
    """
    import functools as ft
    num_time_point_cutoff = create_tr_dict(average=True)
    #behav_exclusion_this_sub = get_behav_exclusion(subid, task)
    design_column_names = desmat.columns.tolist()
    contrast_matrix = []
    for i, (key, values) in enumerate(contrasts.items()):
        contrast_def = expression_to_contrast_vector(
            values, design_column_names)
        contrast_matrix.append(np.array(contrast_def))
    columns_to_check = np.where(np.sum(np.abs(contrast_matrix), 0)!=0)[0]
    checked_columns_fail = (desmat.iloc[:,columns_to_check] == 0).all()
    any_column_fail = checked_columns_fail.any()
    bad_columns = list(checked_columns_fail.index[checked_columns_fail.values])
    bad_columns = '_and_'.join(str(x) for x in bad_columns)

    num_trs = desmat.shape[0]

    failures = {'subid_task': f'{subid}_{task}_{ses}',
                'percent_junk_gt_30': [percent_junk if percent_junk > .30 else 0],
                f'num_trs_lt_{num_time_point_cutoff[task]}': [num_trs if num_trs < .5*num_time_point_cutoff[task] else 0],
                'task_related_regressor_all_zeros': bad_columns if any_column_fail else [0]}
    failures = pd.DataFrame(failures)
    all_exclusion = failures
    #all_exclusion = pd.merge(behav_exclusion_this_sub, failures)
    print(all_exclusion)
    any_fail = all_exclusion.loc[:, all_exclusion.columns != 'subid_task'].ne(0).any(1).bool()
    if any_fail:
        update_excluded_subject_csv(all_exclusion, subid, task, ses, contrast_dir)
    return all_exclusion, any_fail


def est_vif(desmat_vif):
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    desmat_with_intercept = desmat_vif.copy()
    desmat_with_intercept['intercept'] = 1
    vif_data = pd.DataFrame()
    vif_data["regressor"] = desmat_with_intercept.columns
    vif_data["VIF"] = [variance_inflation_factor(desmat_with_intercept.values, i)
                          for i in range(len(desmat_with_intercept.columns))]
    return vif_data


def get_eff_reg_vif(desmat, contrast):
    from scipy.linalg import null_space
    contrast_def = expression_to_contrast_vector(contrast, desmat.columns)
    des_nuisance_regs = desmat[desmat.columns[contrast_def == 0]]
    des_contrast_regs = desmat[desmat.columns[contrast_def != 0]]
    # Effective regressor as defined in Smith et al, 
    # Meaningful design and contrast estimability NeuroImage, 2007
    con = np.atleast_2d(contrast_def[contrast_def !=0])
    con2_t = null_space(con)
    con_t = np.transpose(con)
    x = des_contrast_regs.copy().values
    q = np.linalg.pinv(np.transpose(x)@ x)
    f1 = np.linalg.pinv(con @ q @ con_t)
    pc = con_t @ f1 @ con @ q
    con3_t = con2_t - pc @ con2_t
    f3 = np.linalg.pinv(np.transpose(con3_t) @ q @ con3_t)
    eff_reg = x @ q @ np.transpose(con) @ f1
    eff_reg = pd.DataFrame(eff_reg, columns = [contrast])
    other_reg = x @ q @ con3_t @ f3 
    other_reg_names = [f'orth_proj{val}' for val in range(other_reg.shape[1])]
    other_reg = pd.DataFrame(other_reg, columns = other_reg_names)
    des_for_vif = pd.concat([eff_reg, other_reg, des_nuisance_regs], axis = 1)
    vif_dat = est_vif(des_for_vif)
    vif_dat.rename(columns={'regressor': 'contrast'}, inplace=True)
    vif_of_interest = vif_dat[vif_dat.contrast == contrast]
    return vif_of_interest


def get_all_contrast_vif(desmat, contrasts):
    vif_contrasts = {'contrast': [],
                      'VIF': []}
    for key, item in contrasts.items():
        vif_out = get_eff_reg_vif(desmat, item)
        vif_contrasts['contrast'].append(vif_out['contrast'][0])
        vif_contrasts['VIF'].append(vif_out['VIF'][0]) 
    vif_contrasts = pd.DataFrame(vif_contrasts)
    return vif_contrasts      


def add_to_html_summary(subid, contrasts, desmat, outdir, regress_rt, duration_choice, task, any_fail, exclusion, session, percent_junk):
    desmat_fig = plot_design_matrix(desmat)
    desmat_tmpfile = BytesIO()
    desmat_fig.figure.savefig(desmat_tmpfile, format='png', dpi=60)
    desmat_encoded = base64.b64encode(desmat_tmpfile.getvalue()).decode('utf-8')
    if not any_fail:
        html_desmat = f'<h2>{task} design for subject {subid} {session}</h2>' + '<img src=\'data:image/png;base64,{}\'>'.format(desmat_encoded) + '<br>'
    if any_fail:
        print('ANY FAIL!')
        html_desmat = f'<h2>Check details <br> {exclusion.T.to_html()} <br> {task} design for subject {subid} {session}</h2>' + '<img src=\'data:image/png;base64,{}\'>'.format(desmat_encoded) + '<br>'

    design_column_names = desmat.columns.tolist()
    contrast_matrix = []
    for i, (key, values) in enumerate(contrasts.items()):
        contrast_def = expression_to_contrast_vector(
            values, design_column_names)
        contrast_matrix.append(np.array(contrast_def))
    contrast_matrix = np.asmatrix(np.asarray(contrast_matrix))
    #maxval = np.max(np.abs(contrast_def))
    maxval = 1
    max_len = np.max([len(str(name)) for name in design_column_names])

    plt.figure(figsize=(.4 * len(design_column_names),
                            1 + .5 * contrast_matrix.shape[0] + .1 * max_len))
    contrast_fig = plt.gca()
    mat = contrast_fig.matshow(contrast_matrix, aspect='equal',
                     cmap='gray', vmin=-maxval, vmax=maxval)
    contrast_fig.set_label('conditions')
    contrast_fig.set_ylabel('')
    contrast_fig.set_yticks(list(range(len(contrasts))), list(contrasts.keys()))
    contrast_fig.xaxis.set(ticks=np.arange(len(design_column_names)))
    contrast_fig.set_xticklabels(design_column_names, rotation=50, ha='left')
    plt.colorbar(mat, fraction=0.025, pad=0.08, shrink=.5)
    plt.tight_layout()
    contrast_tmpfile = BytesIO()
    contrast_fig.figure.savefig(contrast_tmpfile, format='png', dpi=75)
    contrast_encoded = base64.b64encode(contrast_tmpfile.getvalue()).decode('utf-8')
    html_contrast = f'<h2>{task} contrasts for subject {subid} {session} </h2>' + '<img src=\'data:image/png;base64,{}\'>'.format(contrast_encoded) + '<br>'
 
    desmat_vif = desmat
    #[desmat.columns.drop(list(desmat.filter(regex=r'(reject)')))]
    vif_data = est_vif(desmat_vif)
    vif_data_table = vif_data[~vif_data.regressor.str.contains(r'(?:reject|trans|rot|comp_cor|non_steady)')]
    vif_table = vif_data_table.to_html(index = False)

    vif_contrasts = get_all_contrast_vif(desmat, contrasts)
    vif_contrasts_table = vif_contrasts.to_html(index = False)

    corr_matrix = desmat.corr()
    f,  heatmap= plt.subplots(figsize=(20,20)) 
    heatmap = sns.heatmap(corr_matrix, 
                      square = True,
                      vmin=-1, vmax=1, center=0,
                      cmap="coolwarm")
    heatmap.set_xticklabels(
        heatmap.get_xticklabels(),
        rotation=45,
        horizontalalignment='right'
    )
    cormat_tmpfile = BytesIO()
    heatmap.figure.savefig(cormat_tmpfile, format='png', dpi=60)
    cormat_encoded = base64.b64encode(cormat_tmpfile.getvalue()).decode('utf-8')
    html_cormat = '<img src=\'data:image/png;base64,{}\'>'.format(cormat_encoded) + '<br>'
    html_file = (f'{outdir}/contrasts_task_{task}_rtmodel_{regress_rt}_'
                    f'duration_{duration_choice}_model_summary.html')
    with open(html_file,'a') as f:
        f.write('<hr>')
        f.write(f'<h2>Subject {subid} {session}</h2><br>')
        f.write(f'<h3>Percent Junk: {percent_junk}</h3>')
        f.write(html_desmat)
        f.write(html_contrast) 
        f.write(f'<h2>Variance inflation factors subject {subid} {session}</h2><br>')
        f.write(vif_table)
        f.write(vif_contrasts_table)
        f.write(html_cormat)
    plt.close('all')


def update_excluded_subject_csv(current_exclusion, subid, task, ses, contrast_dir):
    import functools as ft
    #behav_exclusion_this_sub = get_behav_exclusion(subid, task, ses)
    #full_sub_exclusion = pd.merge(behav_exclusion_this_sub, current_exclusion, how='inner')
    full_sub_exclusion = current_exclusion
    exclusion_out_path = Path(f'{contrast_dir}/excluded_subject.csv')
    if exclusion_out_path.exists():
        old_exclusion_csv = pd.read_csv(exclusion_out_path)
        full_sub_exclusion['subid_task'] = f'{subid}_{task}_{ses}'
        old_and_new_exclusion = pd.concat([old_exclusion_csv, full_sub_exclusion], axis = 0)
        old_and_new_exclusion.fillna(0, inplace=True)
    else:
        old_and_new_exclusion = current_exclusion
    old_and_new_exclusion.to_csv(exclusion_out_path, index=False)
