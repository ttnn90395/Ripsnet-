import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

data = 'modelnet'
mnc = 10
representation ='PI'

do_analysis = True
do_PI_analysis = False

cwd = os.getcwd()
results_dir = os.path.join(cwd, f'results/{representation}/{data}{mnc}')

analysis_csv = 'analysis_results.csv'

df_analysis = pd.read_csv(os.path.join(results_dir, analysis_csv))
df_analysis.drop('Unnamed: 0', inplace=True, axis=1)

num_folds = len(df_analysis['data gen fold'].unique())

df_analysis.drop([
    'num_classes',
    'N_test_sets',
    'N_train_sets',
    'normalization',
    'representation',
    'scl_fct',
    'normalize_pc',
    'sample_even',
    'PI_size',
    'bandwidth',
    'weight',
    'data gen fold',
    'eval_acc_xgb_RN_gudhi',
    'eval_acc_xgb_gudhi_RN',
    'eval_acc_NN_RN_gudhi',
    'eval_acc_NN_gudhi_RN',
], inplace=True, axis=1)

pct_noise_list = df_analysis['pct_noise'].unique()


df_mean_std = df_analysis.groupby(['pct_noise', 'homdim', 'batch_size', 'dropout', 'regularization']).agg(['mean', 'std'])


for col in df_mean_std.columns.get_level_values(0).unique():
    print(f'%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n{col} -- number of folds: {num_folds}')
    print(df_mean_std[col], '\n')

#### Get 'clean' means and standard deviations:

df_clean = df_analysis.drop(
    [ 'pct_noise',
    'GUDHI_time', 'RN_time', 'DTM_time', 'eval_acc_xgb_gudhi_gudhi',
    'eval_acc_xgb_RN_RN', 'eval_acc_xgb_gudhi_clean_gudhi',
    'eval_acc_xgb_DTM_DTM',
    'eval_acc_NN_gudhi_gudhi',
      'eval_acc_NN_RN_RN',
    ],
    inplace=False,
    axis=1,
)
df_clean_mean_std = df_clean.groupby(['homdim', 'batch_size', 'dropout', 'regularization']).agg(['mean', 'std'])
print('Count:\n', df_clean.groupby(['homdim', 'batch_size', 'dropout', 'regularization']).count())

for col in df_clean_mean_std.columns.get_level_values(0).unique():
    print(
        f'%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n{col} -- number of folds: {num_folds}')
    print(df_clean_mean_std[col], '\n')
