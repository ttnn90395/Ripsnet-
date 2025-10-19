import pandas as pd
import numpy as np
import os
import sys

path = "./" # Input the path to your data folder

expes = sys.argv[1] # 'ucr' or 'synth'
mode = sys.argv[2] # 'cluster' or 'laptop'
data = sys.argv[3] # 'generate' or 'precomputed'
train_nn = sys.argv[4] # 'train' or 'pretrained'
identifier = sys.argv[5] # run1

if expes == 'ucr':

    F = np.array(pd.read_csv(path + "DataSummary.csv", sep=",", header=0, index_col=0))
    datasets = ['ChlorineConcentration', 'ProximalPhalanxTW', 'Plane', 'GunPoint', 'PhalangesOutlinesCorrect', 'SonyAIBORobotSurface2', 'ProximalPhalanxOutlineAgeGroup', 'ECG5000', 'ECG200', 'MedicalImages', 'PowerCons', 'DistalPhalanxOutlineCorrect', 'ItalyPowerDemand', 'MiddlePhalanxOutlineAgeGroup', 'SonyAIBORobotSurface1', 'UMD', 'TwoLeadECG', 'MiddlePhalanxOutlineCorrect', 'GunPointOldVersusYoung', 'MiddlePhalanxTW', 'CBF']
    good_idxs = np.argwhere([F[:,1][d] in datasets for d in range(len(F[:,1]))]).ravel()

    for idx_data in good_idxs:

        if F[idx_data,5] != 'Vary':
            size_noise = 0.02 * int(F[idx_data,5])
        else:
            size_noise = 30

        command_line = "./expe_ucr.sh " + str(F[idx_data,1]) + " _train_TDE311LS_5" + identifier + " _train_TDE311LS_6" + identifier + " _test_TDE311LS_clean_3" + identifier + " _test_TDE311LS_noise_3" + identifier + "    3 1 1    01    0 "   + str(int(F[idx_data,2]/2)) + "     0 500    50    " + str(int(F[idx_data,2]/2)) + " " + str(F[idx_data,2]) + "    0 500    " + str(int(size_noise)) + " 0 1500    PL 5 - - " + mode + " " + data + " " + train_nn
        os.system(command_line)

elif expes == 'synth':

    command_line = "./expe_synth.sh synth _train_PI_1" + identifier + " _train_PI_2" + identifier + " _test_PI_clean_1" + identifier + " _test_PI_noise_1" + identifier + "    - - -    1    - -     - -    50      - -    - -    - 1 1000    PI 0 10 1 " + mode + " " + data + " " + train_nn
    os.system(command_line)
    command_line = "./expe_synth.sh synth _train_LS_1" + identifier + " _train_LS_2" + identifier + " _test_LS_clean_1" + identifier + " _test_LS_noise_1" + identifier + "    - - -    1    - -     - -    300     - -    - -    - 1 1000    PL 5 - - " + mode + " " + data + " " + train_nn
    os.system(command_line)
    
