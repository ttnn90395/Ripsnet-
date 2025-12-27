import matplotlib.pyplot as plt
import dill as pck
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from gudhi.representations import PersistenceImage, DiagramSelector, Landscape
from scipy.spatial import distance
from time import time
import sys
import os
import argparse
from pathlib import Path
import xgboost as xgb
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from helper_fctns.get_names import get_dirs_results, get_dir_model, get_dir_data, get_suffix_dataset
# These functions are assumed to be in global scope after executing AUqCs9B5pw2G


parser = argparse.ArgumentParser()
parser.add_argument(
    '-d',
    '--data',
    required=False,
    type=str,
    default='ucr_timeseries',
    help='Which data to use. Can be in ["ucr_timeseries", "modelnet"]'
)
parser.add_argument(
    '-md',
    '--model_dir',
    required=False,
    type=str,
    default='',
    help='Directory of the trained RipsNet.'
)
parser.add_argument(
    '-mn',
    '--model_name',
    required=False,
    type=str,
    help='Model name.',
    default=''
)
parser.add_argument(
    '-f',
    '--fold',
    required=False,
    type=int,
    default=0,
    help='Fold number when generating multiple data sets.'
)
parser.add_argument(
    '-tsn',
    '--time_series_name',
    required=False,
    type=str,
    default='',
    help='Name of the UCR time series.'
)
parser.add_argument(
    '-dPIp',
    '--data_PI_par',
    required=False,
    type=str,
    help='Dataset of PI parameters.',
    default=''
)
parser.add_argument(
    '-dtrn',
    '--data_train',
    required=False,
    type=str,
    help='Training dataset',
    default=''
)
parser.add_argument(
    '-dtst',
    '--data_test',
    required=False,
    type=str,
    help='Test dataset',
    default=''
)
parser.add_argument(
    '-n',
    '--normalize',
    required=False,
    action = 'store_true',
    help='Control whether to normalize or not.',
)
parser.add_argument(
    '-b',
    '--bulk',
    required=False,
    action = 'store_true',
    help='Control whether to analyze all models in the model directory "bulk"-wise.',
)
parser.add_argument(
    '-do',
    '--dropout',
    type=float,
    required=False,
    default=0,
    help='Dropout used to train the rips net.',
)
parser.add_argument(
    '-reg',
    '--regularization',
    type=float,
    required=False,
    default=0,
    help='l2 regularization used to train the rips net.',
)
parser.add_argument(
    '-bs',
    '--batch_size',
    type=int,
    required=False,
    default=32,
    help='Batch size used to train the rips net.',
)
# parser.add_argument(
#     '-cl',
#     '--num_classes',
#     required=False,
#     type=int,
#     default=3,
#     help='Number of classes of the classification task.',
# )
parser.add_argument(
    '-td',
    '--tde_dim',
    required=False,
    type=int,
    default=3,
    help='Time delay embedding dimension.'
)
parser.add_argument(
    '-hd',
    '--homdims',
    required=False,
    nargs='+',
    type=int,
    default=[1],
    help='List of homology dimensions.'
)

parser.add_argument(
    '-r',
    '--representation',
    required=False,
    type=str,
    default='PI',
    help='Choice of the persistence diagram representation; persistence image/landscape. Must be in: ["PI", "LS"].',
)
parser.add_argument(
    '-mnc',
    '--model_net_choice',
    required=False,
    type=int,
    default=10,
    help='Choice between modelnet40 and modelnet10. Must be in [10, 40]',
)

def create_model(num_classes=3, shape=2500, regularization=0, dropout=0):
    class ModelClassif(nn.Module):
        def __init__(self):
            super(ModelClassif, self).__init__()
            self.regularization = regularization
            self.dropout = dropout

            self.fc1 = nn.Linear(shape, 100)
            self.fc2 = nn.Linear(100, 50)
            self.fc3 = nn.Linear(50, num_classes)

            self.relu = nn.ReLU()
            self.dropout_layer = nn.Dropout(p=dropout)

        def forward(self, x):
            x = self.relu(self.fc1(x))
            if self.dropout > 0:
                x = self.dropout_layer(x)

            x = self.relu(self.fc2(x))
            if self.dropout > 0:
                x = self.dropout_layer(x)

            x = self.fc3(x)  # no activation → logits (from_logits=True)
            return x

    model_classif = ModelClassif()
    return model_classif

def DTW(a, b):
    an = a.size
    bn = b.size
    pointwise_distance = distance.cdist(a.reshape(-1,1),b.reshape(-1,1))
    cumdist = np.matrix(np.ones((an+1,bn+1)) * np.inf)
    cumdist[0,0] = 0
    for ai in range(an):
        for bi in range(bn):
            minimum_cost = np.min([cumdist[ai, bi+1], cumdist[ai+1, bi], cumdist[ai, bi]])
            cumdist[ai+1, bi+1] = pointwise_distance[ai,bi] + minimum_cost
    return cumdist[an, bn]

# Weight function dictionary (workaround for pickling error):
weight_dict = {'constant': lambda x: 1,
               'quadratic': lambda x: x[1] ** 2,
               'linear': lambda x: x[1],
               'tanh': lambda x: 10 * np.tanh(x[1]),
               'quintic': lambda x: x[1] ** 5,
               }

# Check if the execution is on the cluster or not:
on_cluster = False

# Parse arguments. In a Jupyter/Colab environment, sys.argv can contain the kernel file path.
# We check if explicit arguments are passed. If not, we provide an empty list to parse_args()
# to avoid parsing the kernel file as an argument.
if '__file__' not in globals() and len(sys.argv) > 1 and 'colab_kernel_launcher.py' in sys.argv[0]:
    # This indicates a Colab environment where the kernel launcher is the script
    # and there are no other actual command-line arguments meant for the parser.
    # Therefore, parse an empty list of arguments.
    args = parser.parse_args([])
else:
    args = parser.parse_args()

data_name = args.data
model_name = args.model_name
time_series_name = args.time_series_name
dataset_vectorization_params = args.data_PI_par
dataset_train_name = args.data_train
dataset_test_name = args.data_test
normalize = args.normalize
bulk = args.bulk
tdedim = args.tde_dim
homdims = args.homdims
representation = args.representation
model_net_choice = args.model_net_choice
fold = args.fold
dropout = args.dropout
batch_size = args.batch_size
regularization = args.regularization

## Make sure we always have a list of homology dimensions:
if isinstance(homdims, int):
    homdims = [homdims]

num_homdims = len(homdims)

model_net = False
assert(data_name in ['ucr_timeseries', 'modelnet'])
if data_name == 'modelnet':
    model_net = True
    data_name = data_name + str(model_net_choice)


assert(bulk or model_name != '')

# GPU support:
# PyTorch GPU setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


#Set the data directories:
cwd = os.getcwd()
data_base_dir = os.path.join(cwd, 'datasets/saved_datasets')
model_base_dir = os.path.join(cwd, f'models/{representation}')
model_dir = os.path.join(model_base_dir, data_name)
results_dir = os.path.join(cwd, f'results/{representation}/{data_name}')

vectorization_param_data_dir    = os.path.join(data_base_dir, 'rn_train')
ml_train_dataset_dir = os.path.join(data_base_dir, 'ml_train')
test_dataset_dir     = os.path.join(data_base_dir, 'test')
figures_output_dir   = os.path.join(results_dir, 'analysis', 'figures', time_series_name)
Path(vectorization_param_data_dir).mkdir(parents=True, exist_ok=True)
Path(ml_train_dataset_dir).mkdir(parents=True, exist_ok=True)
Path(test_dataset_dir).mkdir(parents=True, exist_ok=True)
Path(figures_output_dir).mkdir(parents=True, exist_ok=True)

# If the names of the datasets are not specified, automatically load the corresponding datasets:
if dataset_train_name == '' or dataset_test_name == '':
    assert(dataset_vectorization_params == '')

if dataset_vectorization_params != '':
    vectorization_param_ds_name = os.path.join(vectorization_param_data_dir, dataset_vectorization_params + '.pkl')
else:
    vectorization_param_ds_name = pck.load(open(os.path.join(model_dir, model_name, 'model_train_dataset_filename.pkl'), 'rb'))
if dataset_train_name != '':
    dataset_train_name = os.path.join(ml_train_dataset_dir, dataset_train_name + '.pkl')
else:
    dataset_train_name = vectorization_param_ds_name.replace('/rn_train/', '/ml_train/')
if dataset_test_name != '':
    dataset_test_name = os.path.join(test_dataset_dir, dataset_test_name + '.pkl')
else:
    dataset_test_name = vectorization_param_ds_name.replace('/rn_train/', '/test/')

vectorization_param_ds_name_name_simple = os.path.splitext(os.path.split(vectorization_param_ds_name)[1])[0]
dataset_train_name_simple = os.path.splitext(os.path.split(dataset_train_name)[1])[0]
dataset_test_name_simple = os.path.splitext(os.path.split(dataset_test_name)[1])[0]

results_file = os.path.join(results_dir, 'analysis_results.csv')

# Load the NN
if bulk:
    model_names = os.listdir(model_dir)
else:
    model_names = [model_name]

for model_name in model_names:
    # RipsNet model loading and prediction needs to be re-evaluated.
    # Placeholder for RipsNet model.
    # For now, we assume the RipsNet outputs (vect_RN_...) are available or simulated.
    # Original: model = tf.keras.models.load_model(os.path.join(model_dir, model_name))
    # Original: model.summary()
    # Original: SVG(tf.keras.utils.model_to_dot(model, show_shapes=True).create(prog='dot', format='svg'))
    # Original: tf.keras.utils.plot_model(model, to_file='dot_img_file.jpg', show_shapes=True, rankdir='LR')

    print('Dataset used to load the vectorization parameters (same as training of the ripsnet): ', vectorization_param_ds_name, '\n')

    ## Data used for the training of the ML model.
    print('Dataset used for the training of the ML model: ', dataset_train_name, '\n')
    data_trn = pck.load(open(dataset_train_name, 'rb'))
    label_train_ml_train          = data_trn["label_train"]
    label_test_ml_train           = data_trn["label_test"]
    pc_train_ml_train             = data_trn["data_train"]
    pc_test_ml_train              = data_trn['data_test']
    vect_gudhi_train_ml_train     = data_trn["vectorization_train"]
    vect_test_gudhi_ml_train      = data_trn['vectorization_test']
    vect_gudhi_train_ml_train_DTM = data_trn['vectorization_train_DTM']
    vect_test_gudhi_ml_train_DTM  = data_trn['vectorization_test_DTM']
    num_classes                   = data_trn['num_classes']
    scaling_factor                = data_trn['scaling_factor']

    if model_net:
        time_Gudhi_train_train = data_trn['time_Gudhi_train']
        time_DTM_train_train   = data_trn['time_DTM_train']
        normalize_pc           = data_trn['normalize_pc']
        sample_even            = data_trn['sample_even']
        num_pts                = data_trn['num_pts']

    if not model_net:
        len_TS = data_trn['TS length']
        skip = data_trn['skip']
        skipauto = data_trn['skipauto']
        delay = data_trn['delay']
        maxd_train = data_trn['RC_max_dist']
        ts_classif_train = np.vstack(data_trn["ts_train"])
    assert(representation == data_trn['representation'])
    assert(representation in ['PI', 'LS'])

    ## Noisy data used for the evaluation of the Ml models.
    print('(Noisy) Test dataset for evaluation of the models: ', dataset_test_name, '\n')
    data_test = pck.load(open(dataset_test_name, 'rb'))
    label_test_ml_eval                = data_test["label_test"]
    pc_test_ml_eval                   = data_test["data_test"]
    pc_test_ml_eval_clean             = data_test["data_test_clean"]
    vect_gudhi_test_ml_eval           = data_test['vectorization_test']
    vect_gudhi_test_ml_eval_clean     = data_test['vectorization_test_clean']
    vect_gudhi_test_ml_eval_DTM       = data_test['vectorization_test_DTM']
    vect_gudhi_test_ml_eval_clean_DTM = data_test['vectorization_test_clean_DTM']
    pct_noise                         = data_test['pct_noise']

    if model_net:
        time_Gudhi_test_ml_eval = data_test['time_Gudhi_test']
        time_DTM_test_ml_eval   = data_test['time_DTM_test']

    if not model_net:
        maxd_test = data_test['RC_max_dist']
        ts_classif_test = np.vstack(data_test["ts_test"])
        assert(len_TS       == data_test["TS length"])
        assert(skip         == data_test['skip'])
        assert (skipauto    == data_test['skipauto'])
        assert(delay        == data_test['delay'])

    assert (representation == data_test['representation'])
    assert (scaling_factor == data_test['scaling_factor'])
    assert (num_classes == data_test['num_classes'])

    if representation == 'PI':
        print(homdims)
        print(data_trn['vectorization_params'])
        vparams = data_trn['vectorization_params'][homdims[0]]
        PI_size = vparams['resolution'][0]
        vectorization_size = num_homdims * vparams['resolution'][0] * vparams['resolution'][1]
    else:
        vparams = data_trn['vectorization_params'][homdims[0]]
        num_landscapes = vparams['num_landscapes']
        resolution = vparams['resolution']
        vectorization_size = num_homdims * num_landscapes * resolution

    # PI_size   = int(np.sqrt(vect_train_gudhi_ml_train.shape[1]))

    # workaround for modelnet:
    if model_net:
        pc_train_ml_train = [pc_train_ml_train[i, :, :] for i in range(0, pc_train_ml_train.shape[0])]
        pc_test_ml_eval = [pc_test_ml_eval[i, :, :] for i in range(0, pc_test_ml_eval.shape[0])]

    data_sets_train_ml_train_test_ml_eval = pc_train_ml_train + pc_test_ml_eval
    N_sets    = len(pc_train_ml_train) + len(pc_test_ml_eval)


#######################################################################33
    # Plot the points clouds

    fig = plt.figure()
    fig.set_size_inches(6, 4.5)
    for i in range(9):
        ax = fig.add_subplot(3,3,i+1, projection='3d') #plt.subplot(3, 3, i + 1)
        x = data_sets_train_ml_train_test_ml_eval[len(pc_train_ml_train) + i][:, 0]
        y = data_sets_train_ml_train_test_ml_eval[len(pc_train_ml_train) + i][:, 1]
        z = data_sets_train_ml_train_test_ml_eval[len(pc_train_ml_train) + i][:, 2]
        ax.scatter(x, y, z, s=0.5)
        ax.axis('off')
        ax.title.set_text(label_test_ml_eval[i])
    plt.suptitle('Point clouds on (noisy) test data')
    # plt.savefig(os.path.join(figures_output_dir, dataset_test_name_simple + '_point_clouds_on_test.jpg'),
    #             dpi=200)
    plt.savefig(os.path.join(figures_output_dir, f'point_clouds_on_test_pct_noise_{pct_noise}.jpg'),
                dpi=200)

    fig = plt.figure()
    fig.set_size_inches(6, 4.5)
    for i in range(9):
        ax = fig.add_subplot(3,3,i+1, projection='3d') #plt.subplot(3, 3, i + 1)
        # x = data_sets_train_ml_train_test_ml_eval[i][:, 0]
        # y = data_sets_train_ml_train_test_ml_eval[i][:, 1]
        # z = data_sets_train_ml_train_test_ml_eval[i][:, 2]
        x = pc_test_ml_eval_clean[i][:, 0]
        y = pc_test_ml_eval_clean[i][:, 1]
        z = pc_test_ml_eval_clean[i][:, 2]
        ax.scatter(x, y, z, s=0.5)
        ax.axis('off')
        # ax.title.set_text(label_train_ml_train[i])
        ax.title.set_text(label_test_ml_eval[i])
    plt.suptitle('Point clouds on (clean) test data')
    # plt.savefig(os.path.join(figures_output_dir, dataset_test_name_simple + '_point_clouds_on_test_clean.jpg'),
    #             dpi=200)
    plt.savefig(os.path.join(figures_output_dir, f'point_clouds_on_test_clean.jpg'),
                dpi=200)

    dataset_rn_train = pck.load(open(vectorization_param_ds_name, 'rb'))
    vect_params_rn_train  = dataset_rn_train['vectorization_params'][homdims[0]]
    vect_test_rn_train = dataset_rn_train['vectorization_test'] #PIs generated from clean data
    label_test_rn_train         = dataset_rn_train['label_test']



    # Set the max distance parameter for the Rips complex:
    if not model_net:
        maxd_RC = dataset_rn_train['RC_max_dist']
        # maxd = np.min([maxd_PI, maxd_train, maxd_test])
        maxd = maxd_RC

    if representation == 'PI':
        wgt_name = vect_params_rn_train['weight']
        vect_params_rn_train['weight'] = weight_dict[vect_params_rn_train['weight']]  # weight_dict[weight_names[0]]
        vectorization = PersistenceImage(**vect_params_rn_train)
        vect_gudhi_ml_train_ml_eval = np.zeros((N_sets, vectorization_size))


    else:
        print('\nvectorization parameters: ', vect_params_rn_train)
        vectorization = Landscape(**vect_params_rn_train)

    if normalize:
        for hdidx in range(num_homdims):
            st_idx  = hdidx * int(vectorization_size / num_homdims)
            end_idx = (hdidx + 1) * int(vectorization_size / num_homdims)

            MV = np.max(vect_gudhi_train_ml_train[:, st_idx:end_idx])
            vect_gudhi_train_ml_train[:, st_idx:end_idx] /= MV
            MV = np.max(vect_gudhi_test_ml_eval_clean[:, st_idx:end_idx])
            vect_gudhi_test_ml_eval_clean[:, st_idx:end_idx] /= MV
            # MV = np.max(vect_gudhi_test_ml_eval[:, st_idx:end_idx])
            vect_gudhi_test_ml_eval[:, st_idx:end_idx] /= MV

            MV = np.max(vect_gudhi_train_ml_train_DTM[:, st_idx:end_idx])
            vect_gudhi_train_ml_train_DTM[:, st_idx:end_idx] /= MV
            MV = np.max(vect_gudhi_test_ml_eval_clean_DTM[:, st_idx:end_idx])
            vect_gudhi_test_ml_eval_clean_DTM[:, st_idx:end_idx] /= MV
            # MV = np.max(vect_gudhi_test_ml_eval[:, st_idx:end_idx])
            vect_gudhi_test_ml_eval_DTM[:, st_idx:end_idx] /= MV


    if model_net:
        vect_gudhi_ml_train_ml_eval = np.concatenate((vect_gudhi_train_ml_train, vect_gudhi_test_ml_eval), axis=0)
    else:
        vect_gudhi_ml_train_ml_eval = vect_gudhi_train_ml_train + vect_gudhi_test_ml_eval


    timeG = time_Gudhi_test_ml_eval + time_Gudhi_train_train
    print('Time taken by Gudhi = {} seconds\n'.format(timeG))
    timeDTM = time_DTM_test_ml_eval + time_DTM_train_train
    print('Time taken by DTM = {} seconds\n'.format(timeDTM))

    # Plot the true PIs
    if representation == 'PI':
        reshape_target_size = [num_homdims * PI_size, PI_size]
    else:
        reshape_target_size = [num_landscapes, num_homdims * resolution]

    plt.figure()
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        if representation == 'PI':
            plt.imshow(np.flip(np.reshape(vect_gudhi_ml_train_ml_eval[i], reshape_target_size), 0), vmin=0, vmax=1, cmap='jet')
            plt.suptitle('The true GUDHI PI on clean train')
        else:
            for landscape_id in vect_gudhi_ml_train_ml_eval[i].reshape(reshape_target_size):
                plt.plot(landscape_id)
            plt.suptitle('The true GUDHI LS on clean train')

    plt.savefig(os.path.join(figures_output_dir, dataset_test_name_simple + '_true_vect_on_train.jpg'))

    plt.figure()
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        if representation == 'PI':
            plt.imshow(np.flip(np.reshape(vect_gudhi_train_ml_train_DTM[i], reshape_target_size), 0), vmin=0, vmax=1, cmap='jet')
            plt.suptitle('The DTM PI on clean train')
        else:
            for landscape_id in vect_gudhi_train_ml_train_DTM[i].reshape(reshape_target_size):
                plt.plot(landscape_id)
            plt.suptitle('The DTM LS on clean train')

    plt.savefig(os.path.join(figures_output_dir, dataset_test_name_simple + '_DTM_vect_on_train.jpg'))


    plt.figure()
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        if representation == 'PI':
            plt.imshow(np.flip(np.reshape(vect_gudhi_test_ml_eval_clean[i], reshape_target_size), 0), vmin=0, vmax=1, cmap='jet')
            plt.suptitle('The corresponding true GUDHI PI on clean test')
        else:
            for landscape_id in vect_gudhi_test_ml_eval_clean[i].reshape(reshape_target_size):
                plt.plot(landscape_id)
            plt.suptitle('The corresponding true GUDHI LS on clean test')
    plt.savefig(os.path.join(figures_output_dir, dataset_test_name_simple + '_true_vect_on_clean_test.jpg'))

    plt.figure()
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        if representation == 'PI':
            plt.imshow(np.flip(np.reshape(vect_gudhi_test_ml_eval_clean_DTM[i], reshape_target_size), 0), vmin=0, vmax=1, cmap='jet')
            plt.suptitle('The corresponding DTM PI on clean test')
        else:
            for landscape_id in vect_gudhi_test_ml_eval_clean_DTM[i].reshape(reshape_target_size):
                plt.plot(landscape_id)
            plt.suptitle('The corresponding DTM LS on clean test')
    plt.savefig(os.path.join(figures_output_dir, dataset_test_name_simple + '_DTM_vect_on_clean_test.jpg'))


    plt.figure()
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        if representation == 'PI':
            plt.imshow(np.flip(np.reshape(vect_gudhi_ml_train_ml_eval[len(pc_train_ml_train) + i], reshape_target_size), 0), vmin=0, vmax=1, cmap='jet')
            plt.suptitle('The true GUDHI PI on noisy test')
        else:
            for landscape_id in vect_gudhi_ml_train_ml_eval[len(pc_train_ml_train) + i].reshape(reshape_target_size):
                plt.plot(landscape_id)
            plt.suptitle('The true GUDHI LS on noisy test')
    plt.savefig(os.path.join(figures_output_dir, dataset_test_name_simple + '_true_vect_on_noisy_test.jpg'))

    plt.figure()
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        if representation == 'PI':
            plt.imshow(np.flip(np.reshape(vect_gudhi_test_ml_eval_DTM[i], reshape_target_size), 0), vmin=0, vmax=1, cmap='jet')
            plt.suptitle('The DTM PI on noisy test')
        else:
            for landscape_id in vect_gudhi_test_ml_eval_DTM[i].reshape(reshape_target_size):
                plt.plot(landscape_id)
            plt.suptitle('The DTM LS on noisy test')
    plt.savefig(os.path.join(figures_output_dir, dataset_test_name_simple + '_DTM_vect_on_noisy_test.jpg'))

    # Compute their PIs with the NN and save computation time

    # The following lines related to TF ragged tensor and model.predict have been removed/adapted.
    # Placeholder for RipsNet outputs
    # data_sets_train_ml_train_test_ml_eval = tf.ragged.constant(...) # Removed
    # data_test_ml_eval_clean = tf.ragged.constant(...) # Removed
    # vect_RN_test_ml_eval_clean = model.predict(data_test_ml_eval_clean) # Removed
    # data_test_ml_train = tf.ragged.constant(...) # Removed
    # vect_RN_test_ml_train = model.predict(data_test_ml_train) # Removed
    starttimeRN = time()
    # vect_RN_train_ml_train_test_ml_eval = model.predict(data_sets_train_ml_train_test_ml_eval) # Removed
    timeRN = time() - starttimeRN
    print('Time taken by RN = {} seconds \n'.format(timeRN))

    # Placeholders for RN vectors after TF removal:
    # These now need to be actual PyTorch tensors, but for now we'll keep them as numpy array placeholders.
    # In a full conversion, a PyTorch equivalent of RipsNet would be needed to generate these.
    vect_RN_test_ml_eval_clean = np.zeros_like(vect_gudhi_test_ml_eval_clean)
    vect_RN_test_ml_train = np.zeros_like(vect_gudhi_test_ml_train)
    vect_RN_train_ml_train_test_ml_eval = np.zeros_like(np.concatenate((vect_gudhi_train_ml_train, vect_gudhi_test_ml_eval), axis=0))
    timeRN = 0 # Placeholder for time


    # Plot the predicted PIs

    plt.figure()
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        if representation == 'PI':
            plt.imshow(np.flip(np.reshape(vect_RN_train_ml_train_test_ml_eval[len(pc_train_ml_train) + i], reshape_target_size), 0), vmin=0, vmax=1, cmap='jet')
            plt.suptitle('The corresponding predicted PI by RN on (noisy) test')
        else:
            for landscape_id in vect_RN_train_ml_train_test_ml_eval[len(pc_train_ml_train) + i].reshape(reshape_target_size):  # TODO: uber dirty.
                plt.plot(landscape_id)
            plt.suptitle('The corresponding predicted LS by RN on (noisy) test')
    plt.savefig(os.path.join(figures_output_dir, dataset_test_name_simple + '_predicted_vect_on_test.jpg'))



    # Compute the MSE between the true and predicted PIs

    mse = (np.square(vect_RN_train_ml_train_test_ml_eval - vect_gudhi_ml_train_ml_eval)).mean(axis=None)

    # We test if the PIs computed with the NN and the PIs computed with Gudhi have the same distribution with Kolmogorov-Smirnov test, i.e., (H_0): D_NN = D_G

    vect_RN_train_ml_train, vect_RN_test_ml_eval = vect_RN_train_ml_train_test_ml_eval[:len(pc_train_ml_train)], vect_RN_train_ml_train_test_ml_eval[len(pc_train_ml_train):]


###################################################################

    N_sets_train = len(pc_train_ml_train)
    N_sets_test = len(pc_test_ml_eval)


    print("N_sets_train :       ", N_sets_train)
    print("N_sets_test :        ", N_sets_test)
    print("vectorization_size : ", vectorization_size)
    print("Number of classes:   ", num_classes)

    le = LabelEncoder().fit(np.concatenate([label_train_ml_train, label_test_ml_eval]))
    label_train_ml_train = le.transform(label_train_ml_train)
    label_test_ml_eval  = le.transform(label_test_ml_eval)
    label_test_ml_train = le.transform(label_test_ml_train)

    ############################ non-baseline models ##################################
    # Fit the classification model model_classif_gudhi and DTM
    model_classif_trained_on_gudhi = create_model(num_classes=num_classes, shape=vectorization_size).to(device)

    # Adapt training for PyTorch models
    optimizer_gudhi = torch.optim.Adam(model_classif_trained_on_gudhi.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Convert numpy arrays to torch tensors and move to device
    X_train_gudhi = torch.tensor(vect_gudhi_train_ml_train, dtype=torch.float32).to(device)
    y_train_gudhi = torch.tensor(label_train_ml_train, dtype=torch.long).to(device)
    X_test_gudhi = torch.tensor(vect_gudhi_test_ml_eval, dtype=torch.float32).to(device)
    y_test_gudhi = torch.tensor(label_test_ml_eval, dtype=torch.long).to(device)
    X_test_RN = torch.tensor(vect_RN_test_ml_eval, dtype=torch.float32).to(device)
    X_test_gudhi_clean = torch.tensor(vect_gudhi_test_ml_eval_clean, dtype=torch.float32).to(device)

    # Training loop for NN trained on Gudhi
    model_classif_trained_on_gudhi.train()
    for epoch in range(500):
        optimizer_gudhi.zero_grad()
        outputs = model_classif_trained_on_gudhi(X_train_gudhi)
        loss = criterion(outputs, y_train_gudhi)
        loss.backward()
        optimizer_gudhi.step()

    model_classif_trained_on_gudhi.eval()
    with torch.no_grad():
        outputs = model_classif_trained_on_gudhi(X_test_gudhi)
        _, predicted = torch.max(outputs.data, 1)
        eval_acc_NN_gudhi_gudhi = (predicted == y_test_gudhi).sum().item() / y_test_gudhi.size(0)

        outputs = model_classif_trained_on_gudhi(X_test_RN)
        _, predicted = torch.max(outputs.data, 1)
        eval_acc_NN_gudhi_RN = (predicted == y_test_gudhi).sum().item() / y_test_gudhi.size(0)

        outputs = model_classif_trained_on_gudhi(X_test_gudhi_clean)
        _, predicted = torch.max(outputs.data, 1)
        eval_acc_NN_gudhi_clean_gudhi = (predicted == y_test_gudhi).sum().item() / y_test_gudhi.size(0)

    print('\nTest accuracy of NN trained on clean GUDHI vectorization and evaluated on clean GUDHI vectorization: ',
          eval_acc_NN_gudhi_clean_gudhi)
    print('\nTest accuracy of NN trained on clean GUDHI vect evaluated on noisy GUDHI vect:', eval_acc_NN_gudhi_gudhi)
    print('\nTest accuracy of NN trained on clean GUDHI vect evaluated on noisy RN vect:', eval_acc_NN_gudhi_RN)

    model_classif_trained_on_DTM = create_model(num_classes=num_classes, shape=vectorization_size).to(device)

    optimizer_DTM = torch.optim.Adam(model_classif_trained_on_DTM.parameters(), lr=0.001)

    X_train_DTM = torch.tensor(vect_gudhi_train_ml_train_DTM, dtype=torch.float32).to(device)
    X_test_DTM = torch.tensor(vect_gudhi_test_ml_eval_DTM, dtype=torch.float32).to(device)
    X_test_DTM_clean = torch.tensor(vect_gudhi_test_ml_eval_clean_DTM, dtype=torch.float32).to(device)

    model_classif_trained_on_DTM.train()
    for epoch in range(500):
        optimizer_DTM.zero_grad()
        outputs = model_classif_trained_on_DTM(X_train_DTM)
        loss = criterion(outputs, y_train_gudhi)
        loss.backward()
        optimizer_DTM.step()

    model_classif_trained_on_DTM.eval()
    with torch.no_grad():
        outputs = model_classif_trained_on_DTM(X_test_DTM)
        _, predicted = torch.max(outputs.data, 1)
        eval_acc_NN_DTM_DTM = (predicted == y_test_gudhi).sum().item() / y_test_gudhi.size(0)

        outputs = model_classif_trained_on_DTM(X_test_DTM_clean)
        _, predicted = torch.max(outputs.data, 1)
        eval_acc_NN_DTM_clean_DTM = (predicted == y_test_gudhi).sum().item() / y_test_gudhi.size(0)

    print('\nTest accuracy of NN trained on clean DTM vectorization and evaluated on clean DTM vectorization: ',
          eval_acc_NN_DTM_clean_DTM)
    print('\nTest accuracy of NN trained on clean DTM vect evaluated on noisy DTM vect:', eval_acc_NN_DTM_DTM)


    # Fit the classification model model_classif_NN

    model_classif_trained_on_RN = create_model(num_classes=num_classes, shape=vectorization_size).to(device)

    optimizer_RN = torch.optim.Adam(model_classif_trained_on_RN.parameters(), lr=0.001)

    X_train_RN = torch.tensor(vect_RN_train_ml_train, dtype=torch.float32).to(device)
    X_test_RN_eval = torch.tensor(vect_RN_test_ml_eval, dtype=torch.float32).to(device)
    X_test_gudhi_eval = torch.tensor(vect_gudhi_test_ml_eval, dtype=torch.float32).to(device)
    X_test_gudhi_clean_eval = torch.tensor(vect_gudhi_test_ml_eval_clean, dtype=torch.float32).to(device)
    X_test_RN_clean_eval = torch.tensor(vect_RN_test_ml_eval_clean, dtype=torch.float32).to(device)

    model_classif_trained_on_RN.train()
    for epoch in range(500):
        optimizer_RN.zero_grad()
        outputs = model_classif_trained_on_RN(X_train_RN)
        loss = criterion(outputs, y_train_gudhi)
        loss.backward()
        optimizer_RN.step()

    model_classif_trained_on_RN.eval()
    with torch.no_grad():
        outputs = model_classif_trained_on_RN(X_test_RN_eval)
        _, predicted = torch.max(outputs.data, 1)
        eval_acc_NN_RN_RN = (predicted == y_test_gudhi).sum().item() / y_test_gudhi.size(0)

        outputs = model_classif_trained_on_RN(X_test_gudhi_eval)
        _, predicted = torch.max(outputs.data, 1)
        eval_acc_NN_RN_gudhi = (predicted == y_test_gudhi).sum().item() / y_test_gudhi.size(0)

        outputs = model_classif_trained_on_RN(X_test_gudhi_clean_eval)
        _, predicted = torch.max(outputs.data, 1)
        eval_acc_NN_RN_clean_gudhi = (predicted == y_test_gudhi).sum().item() / y_test_gudhi.size(0)

        outputs = model_classif_trained_on_RN(X_test_RN_clean_eval)
        _, predicted = torch.max(outputs.data, 1)
        eval_acc_NN_RN_clean_RN = (predicted == y_test_gudhi).sum().item() / y_test_gudhi.size(0)

    print('Test accuracy of NN trained on RipsNet vectorization, evaluated on clean Gudhi vectorization: ',
          eval_acc_NN_RN_clean_gudhi, '\n')
    print('Test accuracy of NN trained on RipsNet vectorization, evaluated on clean RN vectorization: ',
          eval_acc_NN_RN_clean_RN, '\n')
    print('\nTest accuracy of NN trained on RipsNet vectorization evaluated on noisy RN vectorization :',
          eval_acc_NN_RN_RN)
    print('\nTest accuracy of NN trained on RipsNet vectorization evaluated on noisy Gudhi vectorization :',
          eval_acc_NN_RN_gudhi, '\n')



    ### XGB classifier:
    # XGBoost classifiers remain largely the same, but inputs should be numpy arrays
    model_classif_xgb_trained_on_gudhi = xgb.XGBClassifier(eval_metric='logloss', use_label_encoder=False)
    model_classif_xgb_trained_on_gudhi.fit(vect_gudhi_train_ml_train, label_train_ml_train)
    eval_acc_xgb_gudhi_clean_gudhi = model_classif_xgb_trained_on_gudhi.score(vect_gudhi_test_ml_eval_clean,
                                                                              label_test_ml_eval)
    eval_acc_xgb_gudhi_gudhi = model_classif_xgb_trained_on_gudhi.score(vect_gudhi_test_ml_eval, label_test_ml_eval)
    eval_acc_xgb_gudhi_RN = model_classif_xgb_trained_on_gudhi.score(vect_RN_test_ml_eval, label_test_ml_eval)
    print('Test accuracy of XGB trained on Gudhi, evaluated on clean Gudhi vectorization: ',
          eval_acc_xgb_gudhi_clean_gudhi, '\n')
    print('Test accuracy XGB trained on Gudhi vect, evaluated on (noisy) Gudhi vect:', eval_acc_xgb_gudhi_gudhi, '\n')
    print('Test accuracy XGB trained on Gudhi vect, evaluated on (noisy) RN vect:', eval_acc_xgb_gudhi_RN, '\n')

    model_classif_xgb_trained_on_DTM = xgb.XGBClassifier(eval_metric='logloss', use_label_encoder=False)
    model_classif_xgb_trained_on_DTM.fit(vect_gudhi_train_ml_train_DTM, label_train_ml_train)
    eval_acc_xgb_DTM_clean_DTM = model_classif_xgb_trained_on_DTM.score(vect_gudhi_test_ml_eval_clean_DTM,
                                                                              label_test_ml_eval)
    eval_acc_xgb_DTM_DTM = model_classif_xgb_trained_on_DTM.score(vect_gudhi_test_ml_eval_DTM, label_test_ml_eval)
    eval_acc_xgb_DTM_RN = model_classif_xgb_trained_on_DTM.score(vect_RN_test_ml_eval, label_test_ml_eval)
    print('Test accuracy of XGB trained on DTM vect, evaluated on clean DTM vectorization: ',
          eval_acc_xgb_DTM_clean_DTM, '\n')
    print('Test accuracy XGB trained on DTM vect, evaluated on (noisy) DTM vect:', eval_acc_xgb_DTM_DTM, '\n')
    print('Test accuracy XGB trained on DTM vect, evaluated on (noisy) RN vect:', eval_acc_xgb_DTM_RN, '\n')

    model_classif_xgb_trained_on_RN = xgb.XGBClassifier(eval_metric='logloss', use_label_encoder=False)
    model_classif_xgb_trained_on_RN.fit(vect_RN_train_ml_train, label_train_ml_train)
    eval_acc_xgb_RN_clean_gudhi = model_classif_xgb_trained_on_RN.score(vect_gudhi_test_ml_eval_clean,
                                                                        label_test_ml_eval)
    eval_acc_xgb_RN_clean_RN = model_classif_xgb_trained_on_RN.score(vect_RN_test_ml_eval_clean,
                                                                        label_test_ml_eval)
    eval_acc_xgb_RN_RN = model_classif_xgb_trained_on_RN.score(vect_RN_test_ml_eval, label_test_ml_eval)
    eval_acc_xgb_RN_gudhi = model_classif_xgb_trained_on_RN.score(vect_gudhi_test_ml_eval, label_test_ml_eval)
    print('Test accuracy of XGB trained on RN vect, evaluated on clean Gudhi vect: ', eval_acc_xgb_RN_clean_gudhi, '\n')
    print('Test accuracy of XGB trained on RN vect, evaluated on clean RN vect: ', eval_acc_xgb_RN_clean_RN, '\n')
    print('Test accuracy of XGB trained on RipsNet vect, eval on noisy RN vect:', eval_acc_xgb_RN_RN, '\n')
    print('Test accuracy of XGB trained on RipsNet vect, eval on noisy Gudhi vect:', eval_acc_xgb_RN_gudhi, '\n')

##########################################################################

    results_dict = {
                    'representation': representation,
                    'normalization': int(normalize),
                    'num_classes': num_classes,
                    'data gen fold': fold,
                    'N_train_sets': N_sets_train,
                    'N_test_sets': N_sets_test,
                    'dropout': dropout,
                    'batch_size': batch_size,
                    'regularization': regularization,
                    'pct_noise': pct_noise,
                    'homdim': str(homdims).replace(',', " "),
                    'GUDHI_time': timeG,
                    'RN_time': timeRN,
                    'DTM_time': timeDTM,
                    'eval_acc_xgb_gudhi_gudhi': eval_acc_xgb_gudhi_gudhi,
                    'eval_acc_xgb_RN_RN': eval_acc_xgb_RN_RN,
                    'eval_acc_xgb_gudhi_clean_gudhi': eval_acc_xgb_gudhi_clean_gudhi,
                    'eval_acc_xgb_RN_clean_RN': eval_acc_xgb_RN_clean_RN,
                    'eval_acc_xgb_DTM_DTM': eval_acc_xgb_DTM_DTM,
                    'eval_acc_xgb_DTM_clean_DTM': eval_acc_xgb_DTM_clean_DTM,
                    'eval_acc_xgb_RN_gudhi': eval_acc_xgb_RN_gudhi,
                    'eval_acc_xgb_gudhi_RN': eval_acc_xgb_gudhi_RN,
                    'eval_acc_xgb_RN_clean_gudhi': eval_acc_xgb_RN_clean_gudhi,
                    'eval_acc_NN_gudhi_gudhi': eval_acc_NN_gudhi_gudhi,
                    'eval_acc_NN_RN_RN': eval_acc_NN_RN_RN,
                    'eval_acc_NN_RN_gudhi': eval_acc_NN_RN_gudhi,
                    'eval_acc_NN_gudhi_RN': eval_acc_NN_gudhi_RN,
                    'eval_acc_NN_gudhi_clean_gudhi': eval_acc_NN_gudhi_clean_gudhi,
                    'eval_acc_NN_RN_clean_RN': eval_acc_NN_RN_clean_RN,
                    'eval_acc_NN_DTM_DTM': eval_acc_NN_DTM_DTM,
                    'eval_acc_NN_DTM_clean_DTM': eval_acc_NN_DTM_clean_DTM
                    }
    if not model_net:
        results_dict.update({
            'TS_name': time_series_name,
            'TS length': len_TS,
            'skip': skip,
            'skipauto': skipauto,
            'delay': delay,
            'RC_max_dist': maxd,
            'tde dim': tdedim,
        })
    if model_net:
        results_dict.update({
            'normalize_pc': normalize_pc,
            'sample_even': sample_even,
        })

    if representation == 'PI':
        results_dict.update({
            'PI_size': PI_size,
            'bandwidth': vect_params_rn_train['bandwidth'],
            'scl_fct': scaling_factor,
            'weight': wgt_name,
        })
    if representation == 'LS':
        results_dict.update({
            'num_landscapes': num_landscapes,
            'resolution': resolution,
        })

    results_df = pd.DataFrame(results_dict, index=[0])
    add_header = False
    if not os.path.isfile(results_file):
        add_header = True
    results_df.to_csv(results_file, mode='a', header=add_header)

    plt.close('all')
