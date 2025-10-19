import sys
import dill as pck
import pandas as pd
import numpy as np
import gudhi as gd
import matplotlib.pyplot as plt
from time import time

from sklearn.metrics import pairwise_distances
from sklearn.impute import SimpleImputer
from gudhi.point_cloud.timedelay import TimeDelayEmbedding
from gudhi.representations import DiagramSelector, PersistenceImage, Landscape

print(sys.argv)

# UCR data set to use
dataset_name   = sys.argv[1]
suffix_name    = sys.argv[2]

# Time Delay Embedding parameters
tdedim         = int(sys.argv[3])
skip           = int(sys.argv[4])
delay          = int(sys.argv[5])

# Homology dimension
homdim         = [int(d) for d in sys.argv[6]]

# Indices of first and last train time series
first_train    = int(sys.argv[7])
last_train     = int(sys.argv[8])

# Indices of first and last test time series
first_test     = int(sys.argv[9])
last_test      = int(sys.argv[10])

# Whether to replace some values with noise and number of noisy replacements
noise          = int(sys.argv[11])
N_noise        = int(sys.argv[12])

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

path = "./" # Input the path to your data folder

X1 = np.array(pd.read_csv(path + dataset_name + "/" + dataset_name + "_TRAIN.tsv", sep="\t", header=None))
X2 = np.array(pd.read_csv(path + dataset_name + "/" + dataset_name + "_TEST.tsv",  sep="\t", header=None))
#np.random.seed(int(time()))
np.random.seed(0)
perm1 = np.random.permutation(len(X1))
perm2 = np.random.permutation(len(X2))
X1 = X1[perm1]
X2 = X2[perm2]
X = np.vstack([X1, X2])
L, TS = X[:,0], X[:,1:]
label_train, label_test = L[first_train:last_train], L[len(X1)+first_test:len(X1)+last_test]

imp = SimpleImputer(missing_values=np.nan, strategy="mean")
TS = imp.fit_transform(TS)
tde = TimeDelayEmbedding(dim=tdedim, delay=delay, skip=skip)

ds = []
for tsidx in range(0, min(30,len(X1))):
    ts = TS[tsidx,:]
    Xtde = tde(ts)
    ds.append(pairwise_distances(Xtde).flatten())
allds = np.concatenate(ds)
maxd = np.max(allds)
#allds = allds[np.argwhere(allds > 0)[:,0]]
#bdw = np.quantile(allds, .1)



ts_train, data_train, PD_train = [], [], []
for tsidx in range(first_train, last_train):
    ts = TS[tsidx,:]
    if noise:
        np.random.seed(int(time()))
        noise_idxs = np.random.choice(np.arange(len(ts)), size=N_noise, replace=False)
        np.random.seed(int(time()))
        ts[noise_idxs] = np.random.uniform(low=TS.min(), high=TS.max(), size=N_noise) 
    Xtde = tde(ts)
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
    data_train.append(Xtde)
    ts_train.append(ts[None,:])

ts_test, data_test, PD_test = [], [], []
for tsidx in range(first_test, min(last_test, len(X2))):
    ts = TS[len(X1)+tsidx,:]
    if noise:
        noise_idxs = np.random.choice(np.arange(len(ts)), size=N_noise, replace=False)
        ts[noise_idxs] = np.random.uniform(low=ts.min(), high=ts.max(), size=N_noise) 
    Xtde = tde(ts)
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
    data_test.append(Xtde)
    ts_test.append(ts[None,:])

PVs_train, PVs_test, PVs_params = [], [], []
for hidx in range(len(homdim)):

    pds_train = DiagramSelector(use=True).fit_transform([PD_train[i][hidx] for i in range(len(PD_train))])
    pds_test  = DiagramSelector(use=True).fit_transform([PD_test[i][hidx] for i in range(len(PD_test))])
    vpdtr, vpdte = np.vstack(pds_train), np.vstack(pds_test)
    
    if PV_type == 'PI':
        pers = pairwise_distances(np.hstack([vpdtr[:,0:1],vpdtr[:,1:2]-vpdtr[:,0:1]])[:200]).flatten()
        pers = pers[np.argwhere(pers > 1e-5).ravel()]
        sigma = 10**power_sigma * np.quantile(pers, .2)
        im_bnds = [np.quantile(vpdtr[:,0],0.), np.quantile(vpdtr[:,0],1.), np.quantile(vpdtr[:,1]-vpdtr[:,0],0.), np.quantile(vpdtr[:,1]-vpdtr[:,0],1.)]
        PV_params = {'bandwidth': sigma, 'weight': lambda x: const_fct * np.tanh(x[1])**power_fct, 'resolution': [PV_size, PV_size], 'im_range': im_bnds}
        PV = PersistenceImage(**PV_params)
        PV_params['weight'] = [const_fct, power_fct]
        PVs_params.append(PV_params)

    elif PV_type == 'PL':
        sp_bnds = [np.quantile(vpdtr[:,0],0.), np.quantile(vpdtr[:,1],1.)]
        PV_params = {'num_landscapes': num_ls, 'resolution': PV_size, 'sample_range': sp_bnds}
        PV = Landscape(**PV_params)
        PVs_params.append(PV_params)

    PV_train = PV.fit_transform(pds_train)
    PV_test = PV.fit_transform(pds_test)
    PVs_train.append(PV_train)
    PVs_test.append(PV_test)

# Save data

pck.dump({'hdims':homdim, 'ts_train':ts_train, 'ts_test':ts_test, 'data_train':data_train, 'data_test':data_test, 'PD_train':PD_train, 'PD_test':PD_test, 'PV_train':PVs_train, 'PV_test':PVs_test, 'label_train':label_train, 'label_test':label_test, 'PV_params': PVs_params, 'PV_type': PV_type}, open(dataset_name+suffix_name+".pkl", 'wb'))
