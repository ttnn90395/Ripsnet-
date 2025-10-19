import sys
import dill as pck
import pandas as pd
import numpy as np
import gudhi as gd
import matplotlib.pyplot as plt
import time
import utils
from sklearn.metrics import pairwise_distances
from sklearn.impute import SimpleImputer
from gudhi.point_cloud.timedelay import TimeDelayEmbedding
from gudhi.representations import DiagramSelector, PersistenceImage, Landscape, BirthPersistenceTransform

print(sys.argv)

# Synthetic data set name
dataset_name   = sys.argv[1]
suffix_name    = sys.argv[2]

# Time Delay Embedding parameters -- unused
tdedim         = sys.argv[3]
skip           = sys.argv[4]
delay          = sys.argv[5]

# Homology dimension
homdim         = [int(d) for d in sys.argv[6]]

# Indices of first and last train time series -- unused
first_train    = sys.argv[7]
last_train     = sys.argv[8]

# Indices of first and last test time series -- unused
first_test     = sys.argv[9]
last_test      = sys.argv[10]

# Whether to replace some values with noise and number of noisy replacements (unused)
noise          = int(sys.argv[11])
N_noise        = sys.argv[12]

# Whether to compute PVs and corresponding resolutions
comp_PV        = int(sys.argv[13]) #1
PV_type        = sys.argv[14]
PV_size        = int(sys.argv[15]) #50

if PV_type == 'PI':
    power_sigma = int(sys.argv[16])
    const_fct   = float(sys.argv[17])
    power_fct   = float(sys.argv[18])

elif PV_type == 'PL':
    num_ls = int(sys.argv[16])

mode = sys.argv[19]

tdedim       = 2
N_sets_train = 3000
N_sets_test  = 300
N_points     = 600
N_noise      = 200

data_train, label_train = utils.create_multiple_circles(N_sets_train, N_points, noisy=noise, N_noise=N_noise)
data_test,  label_test  = utils.create_multiple_circles(N_sets_test,  N_points, noisy=noise, N_noise=N_noise)

ds = []
for Xtde in data_train[:30]:
    ds.append(pairwise_distances(Xtde).flatten())
allds = np.concatenate(ds)
maxd = np.max(allds)

ts_train, PD_train = [], []
for Xtde in data_train:
    if tdedim <= 3:
        st = gd.AlphaComplex(points=Xtde).create_simplex_tree(max_alpha_square=maxd)
    else:
        st = gd.RipsComplex(points=Xtde, max_edge_length=maxd).create_simplex_tree(max_dimension=homdim+1)
    st.persistence()
    final_dg = []
    for hdim in homdim:
        dg = st.persistence_intervals_in_dimension(hdim)
        if len(dg) == 0:
            dg = np.empty([0,2])
        final_dg.append(dg)
    PD_train.append(final_dg)
    ts_train.append(np.ones(shape=[1,1]))

ts_test, PD_test = [], []
for Xtde in data_test:
    if tdedim <= 3:
        st = gd.AlphaComplex(points=Xtde).create_simplex_tree(max_alpha_square=maxd)
    else:
        st = gd.RipsComplex(points=Xtde, max_edge_length=maxd).create_simplex_tree(max_dimension=homdim+1)
    st.persistence()
    final_dg = []
    for hdim in homdim:
        dg = st.persistence_intervals_in_dimension(hdim)
        if len(dg) == 0:
            dg = np.empty([0,2])
        final_dg.append(dg)
    PD_test.append(final_dg)
    ts_test.append(np.ones(shape=[1,1]))

PVs_train, PVs_test, PVs_params = [], [], []
for hidx in range(len(homdim)):

    pds_train = DiagramSelector(use=True).fit_transform([PD_train[i][hidx] for i in range(len(PD_train))])
    pds_test  = DiagramSelector(use=True).fit_transform([PD_test[i][hidx] for i in range(len(PD_test))])
    vpdtr, vpdte = np.vstack(pds_train), np.vstack(pds_test)
    all_PDs = np.vstack([vpdtr, vpdte])
    
    if PV_type == 'PI':
        pers = pairwise_distances(np.hstack([vpdtr[:,0:1],vpdtr[:,1:2]-vpdtr[:,0:1]])[:200]).flatten()
        pers = pers[np.argwhere(pers > 1e-5).ravel()]
        sigma = 10**power_sigma * np.quantile(pers, .2)
        im_bnds = [np.quantile(vpdtr[:,0],0.), np.quantile(vpdtr[:,0],1.), np.quantile(vpdtr[:,1]-vpdtr[:,0],0.), np.quantile(vpdtr[:,1]-vpdtr[:,0],1.)]
        print(im_bnds)
        PV_params = {'bandwidth': sigma, 'weight': lambda x: const_fct * np.tanh(x[1])**power_fct, 'resolution': [PV_size, PV_size], 'im_range': im_bnds}
        PV = PersistenceImage(**PV_params)
        PV_params['weight'] = [const_fct, power_fct]
        PVs_params.append(PV_params)

    elif PV_type == 'PL':
        sp_bnds = [np.quantile(vpdtr[:,0],0.), np.quantile(vpdtr[:,1],1.)]
        PV_params = {'num_landscapes': num_ls, 'resolution': PV_size, 'sample_range': sp_bnds}
        PV = Landscape(**PV_params)
        PVs_params.append(PV_params)

    PV_train = PV.transform(pds_train)
    PV_test = PV.transform(pds_test)
    PVs_train.append(PV_train)
    PVs_test.append(PV_test)

# Save data

pck.dump({'hdims':homdim, 'ts_train':ts_train, 'ts_test':ts_test, 'data_train':data_train, 'data_test':data_test, 'PD_train':PD_train, 'PD_test':PD_test, 'PV_train':PVs_train, 'PV_test':PVs_test, 'label_train':label_train, 'label_test':label_test, 'PV_params': PVs_params, 'PV_type': PV_type}, open(dataset_name+suffix_name+".pkl", 'wb'))
