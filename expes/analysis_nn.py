# Analysis of the NN
#    1. comparison between calculation times of Gudhi and NN,
#    2. p-values of Kolmogorov-Smirnov tests on each PI pixel,
#    3. comparison between classification results on true and predicted PIs.

import matplotlib.pyplot as plt
import dill as pck
import numpy as np
import torch
import torch.nn as nn
from IPython.display import SVG
import gudhi as gd
from gudhi.representations import PersistenceImage, Landscape, DiagramSelector
from scipy.spatial import distance
import velour
from tqdm import tqdm
from time import time
from scipy.stats import ks_2samp
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import pairwise_distances
from xgboost import XGBClassifier
from expes.utils import DenseRagged, PermopRagged

model_name = sys.argv[1]
dataset_PV_params = sys.argv[2]
dataset_train_name = sys.argv[3]
dataset_test_name = sys.argv[4]
normalize = int(sys.argv[5])
PV_type = sys.argv[6]
mode = sys.argv[7]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def create_model_large():
    class LargeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.dense1 = DenseRagged(units=50, use_bias=True, activation='relu')
            self.dense2 = DenseRagged(units=30, use_bias=True, activation='relu')
            self.dense3 = DenseRagged(units=10, use_bias=True, activation='relu')
            self.permop = PermopRagged()
            self.fc = nn.Linear(10, 3)
        
        def forward(self, x):
            x = self.dense1(x)
            x = self.dense2(x)
            x = self.dense3(x)
            x = self.permop(x)
            x = self.fc(x)
            return x
    return LargeModel()

def create_model_small():
    class SmallModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.dense1 = DenseRagged(units=50, use_bias=True, activation='relu')
            self.permop = PermopRagged()
            self.fc = nn.Linear(50, 3)
        
        def forward(self, x):
            x = self.dense1(x)
            x = self.permop(x)
            x = self.fc(x)
            return x
    return SmallModel()

#custom metric
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

print(sys.argv)

# Load the NN
model_PV = torch.load('models/' + model_name + '.pt', map_location=device)
model_PV.to(device)
model_PV.eval()

PV_setting = pck.load(open('datasets/' + dataset_PV_params + '.pkl', 'rb'))
PV_params, homdim = PV_setting['PV_params'], PV_setting['hdims']
PV_size = PV_params[0]['resolution']

data = pck.load(open('datasets/' + dataset_train_name + ".pkl", 'rb'))
label_classif_train = data["label_train"]
data_classif_train  = data["data_train"]
PVs_train           = data["PV_train"]
ts_classif_train    = np.vstack(data["ts_train"])

data = pck.load(open('datasets/' + dataset_test_name + ".pkl", 'rb'))
label_classif_test  = data["label_test"]
data_classif_test   = data["data_test"]
ts_classif_test     = np.vstack(data["ts_test"])

PV_size = PV_params[0]['resolution'][0] if PV_type == 'PI' else PV_params[0]['resolution'] 
data_sets = data_classif_train + data_classif_test
N_sets = len(data_classif_train) + len(data_classif_test)

# Plot the points clouds

plt.figure()
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.scatter(data_sets[len(data_classif_train)+i][:,0], data_sets[len(data_classif_train)+i][:,1], s=3)
plt.savefig('results/' + dataset_test_name + '_point_clouds_on_test.png')

# Compute their PIs with Gudhi and save computation time

PD_gudhi = []
starttimeG = time()
for i in range(N_sets):

    rcX = gd.AlphaComplex(points=data_sets[i]).create_simplex_tree()
    rcX.persistence()
    final_dg = []
    for hdim in homdim:
        dg = rcX.persistence_intervals_in_dimension(hdim)
        if len(dg) == 0:
            dg = np.empty([0,2])
        final_dg.append(dg)
    PD_gudhi.append(final_dg)

PV_gudhi = []
for hidx in range(len(homdim)):
    if PV_type == 'PI':
        a, b = PV_params[hidx]['weight'][0], PV_params[hidx]['weight'][1]
        PV_params[hidx]['weight'] = lambda x: a * np.tanh(x[1])**b
    else:
        a, b = -1, -1
    PV = PersistenceImage(**PV_params[hidx]) if PV_type == 'PI' else Landscape(**PV_params[hidx])
    PV_gudhi_hidx= PV.fit_transform(DiagramSelector(use=True).fit_transform([PD_gudhi[i][hidx] for i in range(N_sets)]))
    if normalize:
        PV_gudhi_hidx /= np.max(PV_gudhi_hidx[:len(data_classif_train)])
    PV_gudhi.append(PV_gudhi_hidx)
    
timeG = time() - starttimeG

# Compute their PIs with Gudhi-noise and save computation time

noise_PD_gudhi = []
starttimeG1 = time()
for i in range(N_sets):
    m, p, dimension_max = 0.05, 2, 2
    if dataset_PV_params[:5] == 'synth':
        st_DTM = velour.AlphaDTMFiltration(data_sets[i], m, p, dimension_max)
    else:
        st_DTM = velour.DTMFiltration(data_sets[i], m, p, dimension_max)
    st_DTM.persistence()
    final_dg = []
    for hdim in homdim:
        dg = st_DTM.persistence_intervals_in_dimension(hdim)
        if len(dg) == 0:
            dg = np.empty([0,2])
        final_dg.append(dg)
    noise_PD_gudhi.append(final_dg)
starttimeG2 = time()

noise_PV_params = []
for hidx in range(len(homdim)):
    noise_pds_train = DiagramSelector(use=True).fit_transform([noise_PD_gudhi[i][hidx] for i in range(len(data_classif_train))])
    noise_pds_test  = DiagramSelector(use=True).fit_transform([noise_PD_gudhi[i][hidx] for i in range(len(data_classif_train), len(data_classif_train) + len(data_classif_test))])
    noise_vpdtr, noise_vpdte = np.vstack(noise_pds_train), np.vstack(noise_pds_test)    
    hdim = homdim[hidx]
    if PV_type == 'PI':
        pers = pairwise_distances(np.hstack([noise_vpdtr[:,0:1],noise_vpdtr[:,1:2]-noise_vpdtr[:,0:1]])[:200]).flatten()
        pers = pers[np.argwhere(pers > 1e-5).ravel()]
        sigma = np.quantile(pers, .2)
        im_bnds = [np.quantile(noise_vpdtr[:,0],0.), np.quantile(noise_vpdtr[:,0],1.), np.quantile(noise_vpdtr[:,1]-noise_vpdtr[:,0],0.), np.quantile(noise_vpdtr[:,1]-noise_vpdtr[:,0],1.)]
        print(im_bnds)
        noise_PV_params.append( {'bandwidth': sigma, 'weight': [1, 1], 'resolution': [PV_size, PV_size], 'im_range': im_bnds} )
    else:
        sp_bnds = [np.quantile(noise_vpdtr[:,0],0.), np.quantile(noise_vpdtr[:,1],1.)]
        noise_PV_params.append( {'num_landscapes': 5, 'resolution': PV_size, 'sample_range': sp_bnds} )

starttimeG3 = time()
noise_PV_gudhi = []
for hidx in range(len(homdim)):
    hdim = homdim[hidx]    
    if PV_type == 'PI':
        a, b = noise_PV_params[hidx]['weight'][0], noise_PV_params[hidx]['weight'][1]
        noise_PV_params[hidx]['weight'] = lambda x: a * x[1]**b
    else:
        a, b = -1, -1
    PV = PersistenceImage(**noise_PV_params[hidx]) if PV_type == 'PI' else Landscape(**noise_PV_params[hidx])
    PV_gudhi_hidx = PV.fit_transform(DiagramSelector(use=True).fit_transform([PD_gudhi[i][hidx] for i in range(N_sets)]))
    if normalize:
        PV_gudhi_hidx /= np.max(PV_gudhi_hidx[:len(data_classif_train)])
    noise_PV_gudhi.append(PV_gudhi_hidx)
starttimeG4 = time()

timeGn = (starttimeG4-starttimeG3) + (starttimeG2-starttimeG1)

# Plot the true PVs

for hidx in range(len(homdim)):

    hdim = homdim[hidx]
    plt.figure()
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        if PV_type == 'PI':
            plt.imshow(np.flip(np.reshape(PV_gudhi[hidx][len(data_classif_train)+i], [PV_size, PV_size]), 0), cmap='jet') #, vmin=0, vmax=1)
            plt.colorbar()
        elif PV_type == 'PL':
            nls = PV_params[hidx]['num_landscapes']
            for lidx in range(nls):
                plt.plot(PV_gudhi[hidx][len(data_classif_train)+i][lidx*PV_size:(lidx+1)*PV_size])
    plt.suptitle('Test true PV in hdim ' + str(hdim))
    plt.savefig('results/' + dataset_test_name + '_true_PVs_h' + str(hdim) + '_on_test.png')

# Plot the true PVs-noise

for hidx in range(len(homdim)):

    hdim = homdim[hidx]
    plt.figure()
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        if PV_type == 'PI':
            plt.imshow(np.flip(np.reshape(noise_PV_gudhi[hidx][len(data_classif_train)+i], [PV_size, PV_size]), 0), cmap='jet') #, vmin=0, vmax=1)
            plt.colorbar()
        elif PV_type == 'PL':
            nls = PV_params[hidx]['num_landscapes']
            for lidx in range(nls):
                plt.plot(noise_PV_gudhi[hidx][len(data_classif_train)+i][lidx*PV_size:(lidx+1)*PV_size])
    plt.suptitle('Test true PV-noise in hdim ' + str(hdim))
    plt.savefig('results/' + dataset_test_name + '_true_PVs-noise_h' + str(hdim) + '_on_test.png')

# Compute their PIs with the NN and save computation time

data_sets_torch = [torch.FloatTensor(data_sets[i]).to(device) for i in range(len(data_sets))]
starttimeNN = time()
with torch.no_grad():
    PV_NN = []
    for data_item in data_sets_torch:
        output = model_PV(data_item.unsqueeze(0))
        PV_NN.append(output.cpu().numpy())
PV_NN = np.vstack(PV_NN)
timeNN = time() - starttimeNN

print('Time taken by Gudhi = {:.2f} seconds'.format(timeG))
print('Time taken by Gudhi-noise = {:.2f} seconds'.format(timeGn))
print('Time taken by the NN = {:.2f} seconds'.format(timeNN))

# Plot the predicted PVs

for hidx in range(len(homdim)):

    hdim = homdim[hidx]

    plt.figure()
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        if PV_type == 'PI':
            plt.imshow(np.flip(np.reshape(PV_NN[len(data_classif_train)+i][(hidx)*(PV_size*PV_size):(hidx+1)*(PV_size*PV_size)], [PV_size, PV_size]), 0), cmap='jet') #, vmin=0, vmax=1)
            plt.colorbar()
        elif PV_type == 'PL':
            nls = PV_params[hidx]['num_landscapes']
            for lidx in range(nls):
                plt.plot(PV_NN[len(data_classif_train)+i][(hidx)*(PV_size*nls)+lidx*PV_size:(hidx)*(PV_size*nls)+(lidx+1)*PV_size])
    plt.suptitle('Test predicted PV in hdim ' + str(hdim))
    plt.savefig('results/' + dataset_test_name + '_predicted_PVs_h' + str(hdim) + '_on_test.png')

PV_train_NN, PV_test_NN = PV_NN[:len(data_classif_train)], PV_NN[len(data_classif_train):]
PV_train_gudhi, PV_test_gudhi = np.hstack(PV_gudhi)[:len(data_classif_train)], np.hstack(PV_gudhi)[len(data_classif_train):]
noise_PV_train_gudhi, noise_PV_test_gudhi = np.hstack(noise_PV_gudhi)[:len(data_classif_train)], np.hstack(noise_PV_gudhi)[len(data_classif_train):]

# We compare the classification results between the true PVs and the predicted PVs. For that, we create two models: model_classif_NN will be trained
# with the PVs computed by the NN and model_classif_gudhi will be trained with the PVs computed with Gudhi. We then compare their accuracies on new data.

N_sets_train = PV_train_gudhi.shape[0]
N_sets_test = PV_test_gudhi.shape[0]

le = LabelEncoder().fit(np.concatenate([label_classif_train, label_classif_test]))
label_classif_train = le.transform(label_classif_train)
label_classif_test  = le.transform(label_classif_test)

XG1 = time()
model_classif_gudhi = XGBClassifier(eval_metric='logloss', use_label_encoder=False)
model_classif_gudhi.fit(PV_train_gudhi, label_classif_train)
train_acc_gudhi = model_classif_gudhi.score(PV_train_gudhi, label_classif_train)
test_acc_gudhi =  model_classif_gudhi.score(PV_test_gudhi,  label_classif_test)
XG2 = time()
print('Train--Test accuracy Gudhi (XGB) of data set ' + str(dataset_test_name) + ': ' + str("{:.2f}".format(100*train_acc_gudhi)) + ' -- ' + str("{:.2f}".format(100*test_acc_gudhi)) + ', with parameters [' + ', '.join([str(k) + ':' + str(v) for k,v in PV_params[0].items()] + ['a:' + str(a), 'b:' + str(b)]) + '] and normalization ' + str(normalize))

XR1 = time()
model_classif_NN = XGBClassifier(eval_metric='logloss', use_label_encoder=False)
model_classif_NN.fit(PV_train_NN, label_classif_train)
train_acc = model_classif_NN.score(PV_train_NN, label_classif_train)
test_acc  = model_classif_NN.score(PV_test_NN,  label_classif_test)
XR2 = time()
print('Train--Test accuracy RipsNet (XGB) of data set ' + str(dataset_test_name) + ': ' + str("{:.2f}".format(100*train_acc)) + ' -- ' + str("{:.2f}".format(100*test_acc)) + ', with parameters [' + ', '.join([str(k) + ':' + str(v) for k,v in PV_params[0].items()] + ['a:' + str(a), 'b:' + str(b)]) + '] and normalization ' + str(normalize))

XGN1 = time()
model_classif_noise = XGBClassifier(eval_metric='logloss', use_label_encoder=False)
model_classif_noise.fit(noise_PV_train_gudhi, label_classif_train)
train_acc = model_classif_noise.score(noise_PV_train_gudhi, label_classif_train)
test_acc  = model_classif_noise.score(noise_PV_test_gudhi,  label_classif_test)
XGN2 = time()
print('Train--Test accuracy Gudhi-noise (XGB) of data set ' + str(dataset_test_name) + ': ' + str("{:.2f}".format(100*train_acc)) + ' -- ' + str("{:.2f}".format(100*test_acc)) + ', with parameters [' + ', '.join([str(k) + ':' + str(v) for k,v in noise_PV_params[0].items()] + ['a:' + str(a), 'b:' + str(b)]) + '] and normalization ' + str(normalize))

if dataset_PV_params[:5] == 'synth':

    data_classif_train_torch = [torch.FloatTensor(data_classif_train[i]).to(device) for i in range(len(data_classif_train))]
    data_classif_test_torch = [torch.FloatTensor(data_classif_test[i]).to(device) for i in range(len(data_classif_test))]
    label_train_tensor = torch.LongTensor(label_classif_train).to(device)
    label_test_tensor = torch.LongTensor(label_classif_test).to(device)

    criterion = nn.CrossEntropyLoss()

    XB11 = time()
    model_baseline = create_model_large().to(device)
    optimizer = torch.optim.Adam(model_baseline.parameters())
    
    num_epochs = 10000
    patience = 200
    best_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(num_epochs):
        model_baseline.train()
        total_loss = 0
        total_acc = 0
        for i in range(len(data_classif_train_torch)):
            x = data_classif_train_torch[i].unsqueeze(0)
            y = label_train_tensor[i:i+1]
            output = model_baseline(x)
            loss = criterion(output, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            preds = torch.argmax(output, dim=1)
            total_acc += (preds == y).float().mean().item()
        
        model_baseline.eval()
        val_loss = 0
        val_acc = 0
        with torch.no_grad():
            for i in range(len(data_classif_test_torch)):
                x = data_classif_test_torch[i].unsqueeze(0)
                y = label_test_tensor[i:i+1]
                output = model_baseline(x)
                loss = criterion(output, y)
                val_loss += loss.item()
                preds = torch.argmax(output, dim=1)
                val_acc += (preds == y).float().mean().item()
        
        val_loss /= len(data_classif_test_torch)
        val_acc /= len(data_classif_test_torch)
        
        if val_loss < best_loss - 1e-4:
            best_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            break
    
    train_acc = total_acc / len(data_classif_train_torch)
    test_acc = val_acc
    XB12 = time()
    print('Train--Test accuracy baseline (large NN DeepSet) of data set ' + str(dataset_test_name) + ': ' + str("{:.2f}".format(100*train_acc)) + ' -- ' + str("{:.2f}".format(100*test_acc)))

    XB21 = time()
    model_baseline = create_model_small().to(device)
    optimizer = torch.optim.Adam(model_baseline.parameters())
    
    best_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(num_epochs):
        model_baseline.train()
        total_loss = 0
        total_acc = 0
        for i in range(len(data_classif_train_torch)):
            x = data_classif_train_torch[i].unsqueeze(0)
            y = label_train_tensor[i:i+1]
            output = model_baseline(x)
            loss = criterion(output, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            preds = torch.argmax(output, dim=1)
            total_acc += (preds == y).float().mean().item()
        
        model_baseline.eval()
        val_loss = 0
        val_acc = 0
        with torch.no_grad():
            for i in range(len(data_classif_test_torch)):
                x = data_classif_test_torch[i].unsqueeze(0)
                y = label_test_tensor[i:i+1]
                output = model_baseline(x)
                loss = criterion(output, y)
                val_loss += loss.item()
                preds = torch.argmax(output, dim=1)
                val_acc += (preds == y).float().mean().item()
        
        val_loss /= len(data_classif_test_torch)
        val_acc /= len(data_classif_test_torch)
        
        if val_loss < best_loss - 1e-4:
            best_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            break
    
    train_acc = total_acc / len(data_classif_train_torch)
    test_acc = val_acc
    XB22 = time()
    print('Train--Test accuracy baseline (small NN DeepSet) of data set ' + str(dataset_test_name) + ': ' + str("{:.2f}".format(100*train_acc)) + ' -- ' + str("{:.2f}".format(100*test_acc)))


else:

    XB11 = time()
    model_baseline = KNeighborsClassifier(metric=DTW)
    model_baseline.fit(ts_classif_train, label_classif_train)
    train_acc = model_baseline.score(ts_classif_train, label_classif_train)
    test_acc = model_baseline.score(ts_classif_test, label_classif_test)
    XB12 = time()
    print('Train--Test accuracy baseline (DTW + k-NN) of data set ' + str(dataset_test_name) + ': ' + str("{:.2f}".format(100*train_acc)) + ' -- ' + str("{:.2f}".format(100*test_acc)))

    XB21 = time()
    model_baseline = KNeighborsClassifier(metric='euclidean')
    model_baseline.fit(ts_classif_train, label_classif_train)
    train_acc = model_baseline.score(ts_classif_train, label_classif_train)
    test_acc = model_baseline.score(ts_classif_test, label_classif_test)
    XB22 = time()
    print('Train--Test accuracy baseline (Euc + k-NN) of data set ' + str(dataset_test_name) + ': ' + str("{:.2f}".format(100*train_acc)) + ' -- ' + str("{:.2f}".format(100*test_acc)))

print('Time taken by XGB-Gudhi = {:.2f} seconds'.format(XG2-XG1))
print('Time taken by XGB-Gudhi-noise = {:.2f} seconds'.format(XGN2-XGN1))
print('Time taken by XGB-the NN = {:.2f} seconds'.format(XR2-XR1))
print('Time taken by Baseline 1 = {:.2f} seconds'.format(XB12-XB11))
print('Time taken by Baseline 2 = {:.2f} seconds'.format(XB22-XB21))
