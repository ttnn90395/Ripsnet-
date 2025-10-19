## Creates pointclouds and vectorizations of persistence diagrams from the ModelNet dataset.
import sys

#import pandas as pd
from copy import deepcopy
import numpy as np
import gudhi as gd
import argparse
import os
import dill as pck
import warnings
from pathlib import Path
import matplotlib.pyplot as plt
from time import time
#from multiprocessing import Pool
import velour
from tqdm import tqdm
import trimesh as trm

from sklearn.metrics import pairwise_distances
#from sklearn.impute import SimpleImputer
#from gudhi.point_cloud.timedelay import TimeDelayEmbedding
from gudhi.representations import PersistenceImage, DiagramSelector, Landscape
path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))
from helper_fctns.get_names import get_dirs_results, get_dir_model, get_dir_data, get_suffix_dataset


def compute_PI_parameters(PDs):
    np.random.seed(0)
    # perm = np.random.permutation(len(all_PDs))
    perm = np.random.permutation(len(PDs))

    # sigma = np.quantile(pairwise_distances(all_PDs[perm][:int(0.1 * len(all_PDs))]).flatten(), .2)
    sigma = scaling_factor * np.quantile(
        pairwise_distances(PDs[perm][:int(0.01 * len(PDs))]).flatten(), .2)
    im_bnds = [np.quantile(PDs[:, 0], .1), np.quantile(PDs[:, 0], .9),
               np.quantile(PDs[:, 1], .1),
               np.quantile(PDs[:, 1], .9)]

    return sigma, im_bnds

def create_PDs(pc_array, homdims, use_DTM=False, m=0.005, p=2, max_dim=2):
    PDs = {}
    for hd in homdims:
        PDs.update({hd: []})
    for idx in tqdm(range(0, pc_array.shape[0]), file=sys.stdout, desc='create PDs'):
        if use_DTM:
            st = velour.AlphaDTMFiltration(pc_array[idx,:,:], m, p, max_dim)
        else:
            st = gd.AlphaComplex(points=pc_array[idx, :, :]).create_simplex_tree()  # (max_alpha_square=maxd)
        st.persistence()

        for hd in homdims:
            dg = st.persistence_intervals_in_dimension(hd)
            if len(dg) == 0:
                dg = np.empty([0,2])
            PDs[hd].append(dg)
    return PDs

def execute_PD_creation(pointclouds, homdims, use_DTM=False, m=0.005, p=2, max_dim=2):
    PDs = create_PDs(pointclouds, homdims, use_DTM, m, p, max_dim)
    for hidx in range(len(homdims)):
        curr_hd = homdims[hidx]
        PDs[curr_hd] = DiagramSelector(use=True).fit_transform(PDs[curr_hd])
    return PDs

def execute_vectorization(PDs, vectorization_fct):
    return vectorization_fct.fit_transform(DiagramSelector(use=True).fit_transform(PDs))


def create_trn_tst_point_clouds(data_dir, label, dataset_choice, clean_ratio=0.75, rn_train_ratio=0.6,
                      num_samples=1024,
                      pct_noise=0.15,
                      sample_even=False,
                      normalize=True,
                      global_bounding_box=False,
                      ):
    '''
    :param data_dir: directory containing 'train' and 'test' subdirectories containing the .off files
    :param label: label of the data (string), e.g. 'stairs'.
    :param dataset_choice: string in ['rn_train', 'ml_train', 'test']
    :param clean_ratio: ratio of data used for clean data, i.e. for 'rn_train' combined with 'ml_train'.
    :param rn_train_ratio: ratio of data used for 'rn_train' vs. 'ml_train'.
    :return: pcs_train, pcs_test: lists of point clouds sampled from mesh data.
             label_train, label_test: lists containing the labels with the same number of elements as the mesh arrays.
    '''
    N_noise = int(num_samples * pct_noise)

    test_dir  = os.path.join(data_dir, 'test')
    train_dir = os.path.join(data_dir, 'train')

    all_train_files = os.listdir(train_dir)
    all_test_files  = os.listdir(test_dir)
    N_train     = len(all_train_files)
    N_test      = len(all_test_files)

    if dataset_choice == 'rn_train':
        first_train, first_test = 0, 0
        last_train = int(N_train * rn_train_ratio)
        last_test  = int(N_test * clean_ratio * rn_train_ratio)
    elif dataset_choice == 'ml_train':
        first_train = int(N_train * rn_train_ratio)
        first_test  = int(N_test * clean_ratio * rn_train_ratio)
        last_train  = int(N_train * clean_ratio)
        last_test  = int(N_test * clean_ratio)

    elif dataset_choice == 'test': # we do not need any training data in this case
        first_train  = 0 #int(N_train * clean_ratio)
        first_test  = int(N_test * clean_ratio)
        last_train   = 0 #N_train
        last_test    = N_test
    else:
        raise ValueError(f'dataset choice {dataset_choice} is not a permitted value!\n Possible choices are: ["rn_train", "ml_train", "test"].')

    train_files = all_train_files[first_train:last_train]
    test_files   = all_test_files[first_test:last_test]

    print(f'\nLabel: {label}')
    # print(f'\nN_train, N_test: {N_train}, {N_test}\n')
    # print(f'First train, last train: {first_train}, {last_train}\nFirst test, last test: {first_test}, {last_test}\n')

    ## Defining the label arrays:
    label_test  = [label] * len(test_files)
    label_train = [label] * len(train_files)

    ## Loading the mesh data and creating the point clouds:
    def create_pointclouds(list_files, num_samples, dataset_choice,
                           normalize=normalize, sample_even=sample_even,
                           choice_train=True):
        sample_fct = trm.sample.sample_surface
        if sample_even:
            sample_fct = trm.sample.sample_surface_even

        pcs = []
        if choice_train:
            cur_dir = train_dir
        else:
            cur_dir = test_dir

        for file in tqdm(list_files,  file=sys.stdout, desc='creating PCs'):
            mesh = trm.load(os.path.join(cur_dir, file))
            pc = sample_fct(mesh, num_samples)[0]
            np.random.shuffle(pc)
            assert(pc.shape == (num_samples, 3) or sample_even)
            if normalize:
                pc -= np.mean(pc, axis=0, dtype=np.float64)
                MX = np.max(np.linalg.norm(pc, axis=1))
                pc /= MX
            pcs.append(pc)
        return pcs

    if dataset_choice != 'test':
        pc_train = create_pointclouds(list_files=train_files, num_samples=num_samples, dataset_choice=dataset_choice,
                                      sample_even=sample_even, choice_train=True)
    else:
        pc_train = []
    pc_test = create_pointclouds(list_files=test_files, num_samples=num_samples, dataset_choice=dataset_choice,
                                 sample_even=sample_even, choice_train=False)

    pc_test_clean = []
    ## Add noise if dataset_choice == 'test'
    if (dataset_choice == 'test' and N_noise > 0):
        pc_test_clean = deepcopy(pc_test)
        if not global_bounding_box:
            for pc in pc_test:
                rand_x = np.random.uniform(low=np.min(pc[:, 0]), high=np.max(pc[:, 0]), size=int(N_noise))
                rand_y = np.random.uniform(low=np.min(pc[:, 1]), high=np.max(pc[:, 1]), size=int(N_noise))
                rand_z = np.random.uniform(low=np.min(pc[:, 2]), high=np.max(pc[:, 2]), size=int(N_noise))
                rand = np.stack((rand_x, rand_y, rand_z), axis = 1)

                noise_idxs = np.random.choice(np.arange(num_samples), size=N_noise, replace=False)
                pc[noise_idxs] = rand

    return pc_train, pc_test, label_train, label_test, pc_test_clean

# argparse arguments:
parser = argparse.ArgumentParser()


parser.add_argument(
    '-dc',
    '--dataset_choice',
    required=False,
    type=str,
    default='rn_train',
    help='Dataset choice; possible options: "rn_train", "ml_train", "test"',
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
    '--modelnet_choice',
    required=False,
    type=int,
    default=10,
    help='Choice between modelnet40 and modelnet10. Must be in [10, 40]',
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
    '-hd',
    '--homdims',
    required=False,
    nargs='+',
    type=int,
    default=[1],
    help='List of homology dimensions.'
)
parser.add_argument(
    '-PIsz',
    '--PI_size',
    type=int,
    required=False,
    default=50,
    help='Size of the PIs generated from the "rn_train" dataset.',
)
parser.add_argument(
    '-sf',
    '--scaling_factor',
    type=float,
    required=False,
    default=1.0,
    help='Scaling factor for vectorization parameter tuning. (Particularly for the bandwith of PIs.)',
)
parser.add_argument(
    '-ns',
    '--num_samples',
    type=int,
    required=False,
    default=1024,
    help='Number of points sampled from the 3D objects surface, i.e. number of points in the generated pointcloud.',
)
parser.add_argument(
    '-pctn',
    '--pct_noise',
    type=float,
    required=False,
    default=0.15,
    help='Percentage of noise points added to the dataset.',
)
parser.add_argument(
    '-rs',
    '--resolution',
    type=int,
    required=False,
    default=150,
    help='Resolution of the persistence landscapes.',
)
parser.add_argument(
    '-nls',
    '--num_landscapes',
    type=int,
    required=False,
    default=5,
    help='Number of the persistence landscapes.',
)
parser.add_argument(
    '-wn',
    '--weight_name',
    required=False,
    type=str,
    default='linear',
    help="Name of weight function to be used. Must be one of: ['tanh', 'constant', 'linear', 'quadratic', 'quintic']"
)
parser.add_argument(
    '-n',
    '--normalize',
    required=False,
    action = 'store_true',
    help='Control whether to normalize or not.',
)
parser.add_argument(
    '-npc',
    '--normalize_pc',
    required=False,
    action = 'store_true',
    help='Control whether to normalize the point clouds or not.',
)
parser.add_argument(
    '-se',
    '--sample_even',
    required=False,
    action = 'store_true',
    help='Control whether to use "even sampling" for the pointcloud creation.',
)
# Set representation and model net choice parameters:
args = parser.parse_args()

representation      = args.representation
modelnet_choice    = args.modelnet_choice # must be 10 or 40.
assert(modelnet_choice in [10, 40])


# Parameters:
homdims              = args.homdims
num_classes         = args.modelnet_choice # model net choice corresponds to the number of classes
scaling_factor      = args.scaling_factor
num_samples         = args.num_samples
pct_noise           = args.pct_noise
clean_ratio         = 0.75
rn_train_ratio      = 0.6
dataset_choice      = args.dataset_choice
PI_size             = args.PI_size
wgt_name            = args.weight_name
normalize           = args.normalize
sample_even         = args.sample_even
normalize_pc        = args.normalize_pc
resolution          = args.resolution
num_landscapes      = args.num_landscapes
fold                = args.fold

save_output= True
global_bounding_box = False
overwrite = False
show_figure = False

# Set noise to 0 if dataset choice is not 'test':
if dataset_choice != 'test':
    pct_noise = 0

assert(representation in ['PI', 'LS'])
assert dataset_choice in ["rn_train", "ml_train", "test"], f'Choice of dataset must be one of: "rn_train", "ml_train", "test".\n Current choice is: {dataset_choice}'
assert (pct_noise >= 0 and pct_noise <= 1) , "Percentage of noise points must be between 0 and 1."
assert wgt_name in ['tanh', 'constant', 'linear', 'quadratic', 'quintic'], "Incompatible choice of weight function for PIs."


# Check if the execution is on the cluster or not:
on_cluster = False

########### Add full path to the ModelNet10 dataset here : #################
data_path     = ''

# Set the output directory where the generated dataset will be saved:
kwargs_data_dir = {
                   'representation': representation,
                   'data': f'modelnet{modelnet_choice}',
                   'dataset_choice': dataset_choice,
                   'hom_dim': homdims,
                   'PIsz': PI_size,
                   'scaling_factor': scaling_factor,
                   'wgt_name': wgt_name,
                   'normalize_vect': normalize,
                   'normalize_pc': normalize_pc,
                   'num_samples': num_samples,
                   'pct_noise': pct_noise,
                   'sample_even': sample_even,
                   'resolution': 100,
                   'num_landscapes': 5,
                   'on_cluster': on_cluster,
}
kwargs_suffix = {
                 'representation': representation,
                 'data': f'modelnet{modelnet_choice}',
                 'hom_dim': homdims,
                 'PIsz': PI_size,
                 'scaling_factor': scaling_factor,
                 'wgt_name': wgt_name,
                 'normalize_vect': normalize,
                 'normalize_pc': normalize_pc,
                 'num_samples': num_samples,
                 'pct_noise': pct_noise,
                 'sample_even': sample_even,
                 'resolution': 100,
                 'num_landscapes': 5,
                 'fold': fold,
                 }
data_output_dir = get_dir_data(**kwargs_data_dir)
# Refine the data output directory and files:
#data_output_dir = os.path.join(data_output_base, dataset_choice, f'homdim_{homdim}', f'num_pts_{num_samples}', f'pctnoise_{pct_noise}', f'wgtname_{wgt_name}', f'scl_fct_{scaling_factor}', f'samp_even{int(sample_even)}', f'norm_pc{int(normalize_pc)}')
Path(data_output_dir).mkdir(parents=True, exist_ok=True)

suffix = get_suffix_dataset(**kwargs_suffix)

data_dirs      = [os.path.join(data_path, d) for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]#[data_dir]#

possible_labels = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]


#suffix = f'modelnet{modelnet_choice}_homdim_{homdim}_num_pts_{num_samples}_pct_noise_{pct_noise}_wgt_name_{wgt_name}_scl_fct_{scaling_factor}_samp_even{int(sample_even)}_norm_pc{int(normalize_pc)}'

params_output_file = os.path.join(data_output_dir, 'parameters_' + suffix + '.pkl')
data_output_file = os.path.join(data_output_dir, suffix + '.pkl')

# Check if output dataset already exists:
if os.path.isfile(data_output_file) and not overwrite:
    raise SystemExit(f'The dataset {data_output_file} already exists.\n Refusing to overwrite. Aborting!\n')


# Creating the point clouds and labels from the mesh data:
pc_train_all, pc_test_all, label_train_all, label_test_all, pc_test_clean_all = [], [], [], [], []

print(f'\n\nStarting to generate data for fold number: {fold}\n\n')

num_samples_dict = {}

for data_dir in tqdm(data_dirs, file=sys.stdout, desc='Creating all PCs'):
    label = os.path.split(data_dir)[-1]
    assert(label in possible_labels)

    pc_train, pc_test, label_train, label_test, pc_test_clean = create_trn_tst_point_clouds(data_dir=data_dir,
                                                                             label=label,
                                                                             dataset_choice=dataset_choice,
                                                                             clean_ratio=clean_ratio,
                                                                             rn_train_ratio=rn_train_ratio,
                                                                             num_samples=num_samples,
                                                                             pct_noise=pct_noise,
                                                                             sample_even=sample_even,
                                                                             normalize=normalize_pc,
                                                                             global_bounding_box=global_bounding_box,
                                                                             )
    num_samples_dict[label] = {'num_train': len(pc_train), 'num_test': len(pc_test)}
    pc_train_all.extend(pc_train)
    pc_test_all.extend(pc_test)
    label_train_all.extend(label_train)
    label_test_all.extend(label_test)
    pc_test_clean_all.extend(pc_test_clean)
    # num_train = len(label_train_all)
    # num_test = len(label_test_all)

# Add noise if blobal_bounding_box is set to True:
if global_bounding_box:
    N_noise = int(num_samples * pct_noise)
    min_x = np.min(np.array(pc_test_all)[:, :, 0])
    max_x = np.max(np.array(pc_test_all)[:, :, 0])
    min_y = np.min(np.array(pc_test_all)[:, :, 1])
    max_y = np.max(np.array(pc_test_all)[:, :, 1])
    min_z = np.min(np.array(pc_test_all)[:, :, 2])
    max_z = np.max(np.array(pc_test_all)[:, :, 2])
    for pc in pc_test_all:
        rand_x = np.random.uniform(low=min_x, high=max_x, size=int(N_noise))
        rand_y = np.random.uniform(low=min_y, high=max_x, size=int(N_noise))
        rand_z = np.random.uniform(low=min_z, high=max_x, size=int(N_noise))
        rand = np.stack((rand_x, rand_y, rand_z), axis=1)

        noise_idxs = np.random.choice(np.arange(num_samples), size=N_noise, replace=False)
        pc[noise_idxs] = rand

# Shuffling the data:
assert(len(pc_test_all) == len(label_test_all))
assert(len(pc_train_all) == len(label_train_all))

np.random.seed(0)
if dataset_choice != 'test':
    perm_train = np.random.permutation(len(pc_train_all))
    pc_train_all = np.array(pc_train_all)[perm_train]
    #pc_train_all = pc_train_all[perm_train]
    label_train_all = np.array(label_train_all)[perm_train]
    pc_test_clean_all = np.array(pc_test_clean_all)
perm_test = np.random.permutation(len(pc_test_all))
pc_test_all = np.array(pc_test_all)[perm_test]
if dataset_choice == 'test':
    pc_test_clean_all = np.array(pc_test_clean_all)[perm_test]
label_test_all = np.array(label_test_all)[perm_test]

# # Setting the 'maxd' parameter for PI creation: # Remark: this might not be necessary in case alpha complex is used
# if dataset_choice == 'rn_train':
#     ds = []
#     for idx in range(0,int(0.2 * pc_train_all.shape[0])):
#         ds.append(pairwise_distances(pc_train_all[idx,:,:])).flatten()
#     allds = np.concatenate(ds)
#     maxd = np.max(allds)
#
# else:
#     #load 'maxd' parameter from 'rn_train' dataset.

# Create persistence diagrams and remove infinite points from the PDs:

### DTM parameters:
m, p, max_dim = 0.00075, 2, 2

if isinstance(homdims, int):
    homdims = [homdims]

time_PD_train = 0
time_PD_DTM_train = 0

if dataset_choice != 'test':
    ## non DTM:
    starttime_PD_train = time()
    PD_train = execute_PD_creation(pc_train_all, homdims, use_DTM=False)
    time_PD_train = time() - starttime_PD_train
    ## DTM:
    starttime_PD_DTM_train = time()
    PD_DTM_train = execute_PD_creation(pc_train_all, homdims, use_DTM=True, m=m, p=p, max_dim=max_dim)
    time_PD_DTM_train = time() - starttime_PD_DTM_train

else:
    PD_test_clean = execute_PD_creation(pc_test_clean_all, homdims, use_DTM=False)
    PD_DTM_test_clean = execute_PD_creation(pc_test_clean_all, homdims, use_DTM=True, m=m, p=p, max_dim=max_dim)

starttime_PD_test = time()
PD_test = execute_PD_creation(pc_test_all, homdims, use_DTM=False)
time_PD_test = time() - starttime_PD_test

starttime_PD_DTM_test = time()
PD_DTM_test = execute_PD_creation(pc_test_all, homdims, use_DTM=True, m=m, p=p, max_dim=max_dim)
time_PD_DTM_test = time() - starttime_PD_DTM_test


## Setting the vectorization parameters based on the training set only:
vectorization_params      = {}
vectorization             = {}
vectorization_train       = {}
vectorization_test_clean  = {}
vectorization_test        = {}
time_vect_train           = {}
time_vect_test            = {}
vectorization_params_DTM  = {}
vectorization_DTM         = {}
vectorization_train_DTM   = {}
vectorization_test_clean_DTM = {}
vectorization_test_DTM    = {}
time_vect_train_DTM       = {}
time_vect_test_DTM        = {}

if representation == 'PI':
    # Weight function dictionary (workaround for pickling error):
    weight_dict = {'constant': lambda x: 1,
                   'quadratic': lambda x: x[1] ** 2,
                   'linear': lambda x: x[1],
                   'tanh': lambda x: 10 * np.tanh(x[1]),
                   'quintic': lambda x: x[1] ** 5,
                   }
    weight_names = ['tanh', 'constant', 'linear', 'quadratic', 'quintic']

for hd in homdims:
    #vectorization_params.update({hd: {}})
    if dataset_choice == 'rn_train':
        PD_train_vs = np.vstack(PD_train[hd])
        PD_DTM_train_vs = np.vstack(PD_DTM_train[hd])

        if representation == 'PI':
            sigma, im_bnds = compute_PI_parameters(PD_train_vs)
            vectorization_params[hd] = {'bandwidth': sigma,
                                        'weight': weight_dict[wgt_name],
                                        'resolution': [PI_size, PI_size],
                                        'im_range': im_bnds}

            sigma, im_bnds = compute_PI_parameters(PD_DTM_train_vs)
            vectorization_params_DTM[hd] = {'bandwidth': sigma,
                                        'weight': weight_dict[wgt_name],
                                        'resolution': [PI_size, PI_size],
                                        'im_range': im_bnds}

        else: # if representation == 'LS'
             sample_range = [0, np.max(PD_train_vs[:, 1])]  # [np.min(PD_train_vs[:, 0]), np.max(PD_train_vs[:, 1])]
             # print('\nsample range:', sample_range)
             vectorization_params[hd] = {
                 'num_landscapes': num_landscapes,
                 'resolution': resolution,
                 'sample_range': sample_range,
                 }

             sample_range_DTM = [0, np.max(PD_train_vs[:, 1])]  # [np.min(PD_train_vs[:, 0]), np.max(PD_train_vs[:, 1])]
             # print('\nsample range:', sample_range)
             vectorization_params_DTM[hd] = {
                'num_landscapes': num_landscapes,
                'resolution': resolution,
                'sample_range': sample_range_DTM,
             }

    else:
        # Load parameters from saved dataset files.
        # The actual data and hyperparameters are saved in separate files to reduce RAM usage when loading the parameters.
        rn_train_dir                 = data_output_dir.replace(dataset_choice, 'rn_train').replace(f'pctnoise_{pct_noise}', 'pctnoise_0')
        #rn_train_dir                = rn_train_dir.replace(str(homdims).replace(" ", ""), str(hd))
        params_file                  = [f for f in os.listdir(rn_train_dir) if 'parameters' == f[:10]][0]
        params_file                  = os.path.join(rn_train_dir, params_file)
        parameters                   = pck.load((open(params_file, 'rb')))
        vectorization_params[hd]     = parameters['vectorization_params'][hd]
        vectorization_params_DTM[hd] = parameters['vectorization_params_DTM'][hd]

    if representation == 'PI':
        #print(PI_size, vectorization_params[hd])
        assert(vectorization_params[hd]['resolution'] == [PI_size, PI_size])
        vectorization_params[hd]['weight'] = weight_dict[wgt_name]
        vectorization_params_DTM[hd]['weight'] = weight_dict[wgt_name]
        wgt_name = [w for w in weight_dict.keys() if weight_dict[w] == vectorization_params[hd]['weight']][0]

        vectorization[hd]     = PersistenceImage(**vectorization_params[hd])
        vectorization_DTM[hd] = PersistenceImage(**vectorization_params_DTM[hd])
        vectorization_params[hd]['weight']     = wgt_name
        vectorization_params_DTM[hd]['weight'] = wgt_name

    else:
        vectorization[hd] = Landscape(**vectorization_params)
        vectorization_DTM[hd] = Landscape(**vectorization_params_DTM)

    print(f'\nCreating the vectorizations for homology dimension {hd}...please wait...\n')

    time_vect_train[hd] = 0
    if dataset_choice != 'test':
        start_time_vect_train   = time()
        vectorization_train[hd] = execute_vectorization(PD_train[hd], vectorization[hd])
        time_vect_train[hd]     = time() - start_time_vect_train

        start_time_vect_train_DTM   = time()
        vectorization_train_DTM[hd] = execute_vectorization(PD_DTM_train[hd], vectorization_DTM[hd])
        time_vect_train_DTM[hd]     = time() - start_time_vect_train_DTM
    else:
        #vectorization_train = []
        vectorization_test_clean[hd]     = execute_vectorization(PD_test_clean[hd], vectorization[hd])
        vectorization_test_clean_DTM[hd] = execute_vectorization(PD_DTM_test_clean[hd], vectorization_DTM[hd])
        time_vect_train_DTM[hd] = 0
    start_time_vect_test = time()
    vectorization_test[hd] = execute_vectorization(PD_test[hd], vectorization[hd])
    time_vect_test[hd] = time() - start_time_vect_test

    start_time_vect_test_DTM = time()
    vectorization_test_DTM[hd] = execute_vectorization(PD_DTM_test[hd], vectorization_DTM[hd])
    time_vect_test_DTM[hd] = time() - start_time_vect_test_DTM
    print('Vectorization complete!\n')

time_vect_train_total     = 0
time_vect_test_total      = 0
time_vect_train_total_DTM = 0
time_vect_test_total_DTM  = 0

for hd in homdims:
    time_vect_train_total     += time_vect_train[hd]
    time_vect_test_total      += time_vect_test[hd]
    time_vect_train_total_DTM += time_vect_train_DTM[hd]
    time_vect_test_total_DTM  += time_vect_test_DTM[hd]

## Concatenating the vectorizations of different homology dimensions:
vectorization_test_stack     = np.hstack(list(vectorization_test.values()))
vectorization_test_stack_DTM = np.hstack(list(vectorization_test_DTM.values()))

if vectorization_test_clean != {}:
    vectorization_test_clean_stack = np.hstack(list(vectorization_test_clean.values()))
if vectorization_test_clean_DTM != {}:
    vectorization_test_clean_stack_DTM = np.hstack(list(vectorization_test_clean_DTM.values()))

if vectorization_train != {}:
    vectorization_train_stack = np.hstack(list(vectorization_train.values()))
else:
    vectorization_train_stack = {}
if vectorization_train_DTM != {}:
    vectorization_train_stack_DTM = np.hstack(list(vectorization_train_DTM.values()))
else:
    vectorization_train_stack_DTM = {}


# Save data and parameters

output_params = {
    'representation': representation,
    'dataset_choice': dataset_choice,
    'num_pts': num_samples,
    'homdim': homdims,
    'pct_noise': pct_noise,
    'scaling_factor': scaling_factor,
    'vectorization_params': vectorization_params,
    'vectorization_params_DTM': vectorization_params_DTM,
#    'RC_max_dist': maxd,
    'normalize': int(normalize),
    'normalize_pc': int(normalize_pc),
    'sample_even': int(sample_even),
    'num_classes': num_classes,
    'time_Gudhi_train': time_PD_train + time_vect_train_total,
    'time_Gudhi_test': time_PD_test + time_vect_test_total,
    'time_DTM_train': time_PD_DTM_train + time_vect_train_total_DTM,
    'time_DTM_test': time_PD_DTM_test + time_vect_test_total_DTM,
    'p_DTM': p,
    'm_DTM': m,
    'max_dim_DTM': max_dim,
}

if dataset_choice == 'test':
    PD_train = []
    PD_DTM_train = []
    pc_train_all = []

output_data = {
    'data_train': pc_train_all,
    'data_test': pc_test_all,
    'PD_train': PD_train,
    'PD_test': PD_test,
    'PD_DTM_train': PD_DTM_train,
    'PD_DTM_test': PD_DTM_test,
    'vectorization_train': vectorization_train_stack,
    'vectorization_test': vectorization_test_stack,
    'vectorization_train_DTM': vectorization_train_stack_DTM,
    'vectorization_test_DTM': vectorization_test_stack_DTM,
    'label_train': label_train_all,
    'label_test': label_test_all,
}

if dataset_choice == 'test':
    output_data.update({
        'data_test_clean': pc_test_clean_all,
        'vectorization_test_clean': vectorization_test_clean_stack,
        'vectorization_test_clean_DTM': vectorization_test_clean_stack_DTM,
    })
# include the parameters also in the data output for the training of RipsNet
output_data.update(output_params)

if save_output:
    pck.dump(output_params, open(params_output_file, 'wb'))
    pck.dump(output_data, open(data_output_file, 'wb'))

assert (os.path.isfile(params_output_file))
assert (os.path.isfile(data_output_file))
print('\nCreated data set:\n', data_output_file, '\n')
print('\nwith parameters:\n', params_output_file, '\n')

print(f'\nFinished generating data for fold number: {fold}.')
