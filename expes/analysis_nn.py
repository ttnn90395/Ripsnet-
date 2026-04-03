# Analysis of the NN
#    1. comparison between calculation times of Gudhi and NN,
#    2. p-values of Kolmogorov-Smirnov tests on each PI pixel,
#    3. comparison between classification results on true and predicted PIs.

import os
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

# ensure top repo is in path for models.py import from expes/
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# Ensure results directory exists
os.makedirs('results', exist_ok=True)

from models import (
    TensorFieldNetwork, GTTensorFieldNetwork, HierarchicalGTTFN,
    ScalarDistanceDeepSet, PointNetTutorial, ScalarInputMLP, MultiInputModel,
    DenseRagged, PermopRagged, RaggedPersistenceModel, DistanceMatrixRaggedModel,
)

MODEL_NAME = ['TensorFieldNetwork', 'GTTensorFieldNetwork', 'HierarchicalGTTFN', 'ScalarDistanceDeepSet', 'PointNetTutorial', 'ScalarInputMLP', 'MultiInputModel', 'DenseRagged', 'PermopRagged', 'RaggedPersistenceModel', 'DistanceMatrixRaggedModel']

# CLI args:
# if first arg is 'all', evaluate all models; else map to provided model name
requested_model = sys.argv[1]
dataset_PV_params = sys.argv[2]
dataset_train_name = sys.argv[3]
dataset_test_name = sys.argv[4]
normalize = int(sys.argv[5])
PV_type = sys.argv[6]
mode = sys.argv[7]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_paths = {}
script_dir = os.path.dirname(os.path.abspath(__file__))
candidate_dirs = [
    'models',
    os.path.join(script_dir, 'models'),
    os.path.join(script_dir, '..', 'models'),
    os.path.join(script_dir, '..', '..', 'models'),
]

if requested_model == 'all':
    models_to_test = MODEL_NAME
    for m in models_to_test:
        model_paths[m] = None
elif requested_model in MODEL_NAME:
    models_to_test = [requested_model]
    model_paths[requested_model] = None
else:
    models_to_test = []

    # direct file path support
    if os.path.isfile(requested_model):
        model_name0 = os.path.splitext(os.path.basename(requested_model))[0]
        models_to_test = [model_name0]
        model_paths[model_name0] = requested_model

    # exact file with extension in known dirs
    if not models_to_test:
        for d in candidate_dirs:
            for ext in ('.pt', '.pth'):
                candidate = os.path.join(d, requested_model + ext)
                if os.path.isfile(candidate):
                    model_name0 = requested_model
                    models_to_test = [model_name0]
                    model_paths[model_name0] = candidate
                    break
            if models_to_test:
                break

    # fuzzy wildcard in candidate dirs
    if not models_to_test:
        from glob import glob
        for d in candidate_dirs:
            candidates = glob(os.path.join(d, f'*{requested_model}*'))
            for c in candidates:
                if c.endswith('.pt') or c.endswith('.pth'):
                    model_name0 = os.path.splitext(os.path.basename(c))[0]
                    models_to_test.append(model_name0)
                    model_paths[model_name0] = c

    if not models_to_test:
        from glob import glob
        for d in candidate_dirs:
            candidates = glob(os.path.join(d, f'*{requested_model}*'))
            for c in candidates:
                if c.endswith('.pt') or c.endswith('.pth'):
                    model_name0 = os.path.splitext(os.path.basename(c))[0]
                    models_to_test.append(model_name0)
                    model_paths[model_name0] = c

    if not models_to_test:
        print(f"\nDEBUG: Could not find model '{requested_model}'")
        print(f"  Looked in candidate_dirs: {candidate_dirs}")
        print(f"  Tried exact file: {requested_model} (not a direct path)")
        for d in candidate_dirs:
            print(f"  Tried exact in dir: {os.path.join(d, requested_model)}.[pt|pth]")
            if os.path.isdir(d):
                print(f"    Directory exists, contents: {os.listdir(d)[:5]}")
            else:
                print(f"    Directory does not exist")
        print(f"  Wildcards: searched for *{requested_model}* in each dir\n")
        raise ValueError(
            f"Unknown model '{requested_model}', valid: {MODEL_NAME} or model checkpoint in models/*.pt/*.pth\n"
            f"Looked in: {candidate_dirs}\n"
            f"Hint: Train model first with train_nn.py, then use checkpoint name."
        )

print('models_to_test:', models_to_test)
print('model_paths:', model_paths)
print('dataset_PV_params:', dataset_PV_params, 'dataset_train_name:', dataset_train_name, 'dataset_test_name:', dataset_test_name, 'normalize:', normalize, 'PV_type:', PV_type, 'mode:', mode)


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

def build_analysis_model(name, output_dim, n=None):
    if name == 'TensorFieldNetwork':
        return TensorFieldNetwork(num_classes=output_dim)
    if name == 'GTTensorFieldNetwork':
        n_dim = dim if n is None else n
        return GTTensorFieldNetwork(n=n_dim, num_classes=output_dim)
    if name == 'HierarchicalGTTFN':
        n_dim = dim if n is None else n
        return HierarchicalGTTFN(n=n_dim, num_classes=output_dim)
    if name == 'ScalarDistanceDeepSet':
        return ScalarDistanceDeepSet(output_dim=output_dim)
    if name == 'PointNetTutorial':
        return PointNetTutorial(output_dim=output_dim)
    if name == 'ScalarInputMLP':
        return ScalarInputMLP(output_dim=output_dim)
    if name == 'MultiInputModel':
        return MultiInputModel(target_output_dim=output_dim, scalar_input_dim=1)
    if name == 'DenseRagged':
        return DenseRagged(in_features=None, out_features=output_dim)
    if name == 'PermopRagged':
        return PermopRagged()
    if name == 'RaggedPersistenceModel':
        return RaggedPersistenceModel(output_dim=output_dim)
    if name == 'DistanceMatrixRaggedModel':
        return DistanceMatrixRaggedModel(output_dim=output_dim, num_points=600)
    # fallback
    class RegressionModel(nn.Module):
        def __init__(self, output_dim):
            super().__init__()
            self.dense1 = DenseRagged(out_features=30, activation='relu')
            self.dense2 = DenseRagged(out_features=20, activation='relu')
            self.dense3 = DenseRagged(out_features=10, activation='relu')
            self.permop = PermopRagged()
            self.fc1 = nn.Linear(10, 50)
            self.fc2 = nn.Linear(50, 100)
            self.fc3 = nn.Linear(100, 200)
            self.fc_out = nn.Linear(200, output_dim)

        def forward(self, x):
            x = self.dense1(x)
            x = self.dense2(x)
            x = self.dense3(x)
            x = self.permop(x)
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = torch.relu(self.fc3(x))
            x = torch.sigmoid(self.fc_out(x))
            return x

    return RegressionModel(output_dim=output_dim)



PV_size = PV_params[0]['resolution'][0] if PV_type == 'PI' else PV_params[0]['resolution'] 
data_sets = data_classif_train + data_classif_test
N_sets = len(data_classif_train) + len(data_classif_test)
# support variable point cloud dimensionality
dim = data_sets[0].shape[1] if len(data_sets) > 0 else 2

# Plot the points clouds

plt.figure()
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.scatter(data_sets[len(data_classif_train)+i][:,0], data_sets[len(data_classif_train)+i][:,1], s=3)
plt.savefig('results/' + dataset_test_name + '_point_clouds_on_test.png')

# Compute their PIs with Gudhi and save computation time

PD_gudhi = []
starttimeG = time()
for i in tqdm(range(N_sets), desc="Computing Gudhi PIs"):
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
for hidx in tqdm(range(len(homdim)), desc="Processing Gudhi PVs"):
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
for i in tqdm(range(N_sets), desc="Computing Gudhi-noise PIs"):
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
for hidx in tqdm(range(len(homdim)), desc="Processing noise PV params"):
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
for hidx in tqdm(range(len(homdim)), desc="Computing noise PVs"):
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

def prepare_input_for_model(model_name, x):
    # x is a point cloud tensor; for TFNs, we only extend 2D data to 3D.
    arr = x.cpu().numpy()
    if model_name in ['TensorFieldNetwork', 'GTTensorFieldNetwork', 'HierarchicalGTTFN']:
        if arr.shape[1] == 2:
            arr = np.concatenate([arr, np.zeros((arr.shape[0], 1), dtype=arr.dtype)], axis=1)
        return torch.FloatTensor(arr).to(device)
    if model_name == 'PointNetTutorial':
        return x[:, :2]
    if model_name in ['ScalarDistanceDeepSet', 'DistanceMatrixRaggedModel', 'RaggedPersistenceModel']:
        arr = x.cpu().numpy()
        dmat = np.linalg.norm(arr[:, None, :] - arr[None, :, :], axis=-1)
        return torch.FloatTensor(dmat).to(device)
    # fallback: pass the point cloud directly for sparse ragged-like models
    return x

def model_predict(model, x, model_name):
    inp = prepare_input_for_model(model_name, x)
    if model_name in ['TensorFieldNetwork', 'GTTensorFieldNetwork', 'HierarchicalGTTFN', 'PointNetTutorial', 'DistanceMatrixRaggedModel', 'DenseRagged', 'PermopRagged', 'RaggedPersistenceModel', 'ScalarDistanceDeepSet']:
        out = model([inp])
    elif model_name == 'MultiInputModel':
        # Provide a placeholder scalar feature for MultiInputModel
        scalar = torch.zeros((1, 1), device=device)
        out = model([inp], scalar)
    elif model_name == 'ScalarInputMLP':
        # Scalar input model expects 1D input; use a zero placeholder
        out = model(torch.zeros((1, 1), device=device))
    else:
        out = model(inp.unsqueeze(0))
# loop over requested models and evaluate
for model_name in models_to_test:
    model_path = model_paths.get(model_name)
    if model_path is None:
        if os.path.exists(os.path.join('models', f'{model_name}.pt')):
            model_path = os.path.join('models', f'{model_name}.pt')
        elif os.path.exists(os.path.join('models', f'{model_name}.pth')):
            model_path = os.path.join('models', f'{model_name}.pth')
        else:
            raise FileNotFoundError(f"No checkpoint found for model '{model_name}'")
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model_state_dict = checkpoint['model_state_dict']
        output_dim = checkpoint.get('output_dim', None)
    else:
        model_state_dict = checkpoint
        output_dim = None

    # For generic checkpoint naming (not necessarily class names), support stored model_type
    if model_name in MODEL_NAME:
        model_PV = build_analysis_model(model_name, output_dim, n=dim)
    else:
        model_type = checkpoint.get('model_type', None)
        if model_type in MODEL_NAME:
            model_PV = build_analysis_model(model_type, output_dim, n=dim)
        else:
            raise ValueError(
                f"Checkpoint '{model_path}' has unknown model_type '{model_type}', cannot instantiate. "
                f"Use MODEL_NAME or include model_type in checkpoint."
            )
    model_PV.load_state_dict(model_state_dict)
    model_PV = model_PV.to(device)
    model_PV.eval()

    starttimeNN = time()
    with torch.no_grad():
        PV_NN = []
        for data_item in tqdm(data_sets_torch, desc=f"Predicting with {model_name}"):
            output = model_predict(model_PV, data_item, model_name)
            output_np = output.cpu().numpy() if isinstance(output, torch.Tensor) else np.asarray(output)
            if output_np.ndim > 1 and output_np.shape[0] == 1:
                output_np = output_np[0]
            PV_NN.append(output_np)

    PV_NN = np.vstack(PV_NN)
    timeNN = time() - starttimeNN
    print(f"Time taken by the NN ({model_name}) = {timeNN:.2f} seconds")

    for hidx in range(len(homdim)):
        hdim = homdim[hidx]
        plt.figure()
        for i in range(9):
            plt.subplot(3, 3, i + 1)
            if PV_type == 'PI':
                plt.imshow(np.flip(np.reshape(PV_NN[len(data_classif_train)+i][hidx*(PV_size*PV_size):(hidx+1)*(PV_size*PV_size)], [PV_size, PV_size]), 0), cmap='jet')
                plt.colorbar()
            else:
                nls = PV_params[hidx]['num_landscapes']
                for lidx in range(nls):
                    plt.plot(PV_NN[len(data_classif_train)+i][hidx*(PV_size*nls)+lidx*PV_size:(hidx)*(PV_size*nls)+(lidx+1)*PV_size])
        plt.suptitle(f"Test predicted PV in hdim {hdim} ({model_name})")
        plt.savefig(f"results/{dataset_test_name}_{model_name}_predicted_PVs_h{hdim}_on_test.png")

    PV_train_NN = PV_NN[:len(data_classif_train)]
    PV_test_NN = PV_NN[len(data_classif_train):]

    clf = XGBClassifier(eval_metric='logloss', use_label_encoder=False)
    clf.fit(PV_train_NN, label_classif_train)
    train_acc_nn = clf.score(PV_train_NN, label_classif_train)
    test_acc_nn = clf.score(PV_test_NN, label_classif_test)
    print(f"RipsNet ({model_name}) XGB train/test acc: {100*train_acc_nn:.2f}% / {100*test_acc_nn:.2f}%")

# followed by baseline/Gudhi classifiers
PV_train_gudhi, PV_test_gudhi = np.hstack(PV_gudhi)[:len(data_classif_train)], np.hstack(PV_gudhi)[len(data_classif_train):]
noise_PV_train_gudhi, noise_PV_test_gudhi = np.hstack(noise_PV_gudhi)[:len(data_classif_train)], np.hstack(noise_PV_gudhi)[len(data_classif_train):]

le = LabelEncoder().fit(np.concatenate([label_classif_train, label_classif_test]))
label_classif_train = le.transform(label_classif_train)
label_classif_test = le.transform(label_classif_test)

XG1 = time()
model_classif_gudhi = XGBClassifier(eval_metric='logloss', use_label_encoder=False)
model_classif_gudhi.fit(PV_train_gudhi, label_classif_train)
train_acc_gudhi = model_classif_gudhi.score(PV_train_gudhi, label_classif_train)
test_acc_gudhi = model_classif_gudhi.score(PV_test_gudhi, label_classif_test)
XG2 = time()
print(f"Gudhi (XGB) train/test: {100*train_acc_gudhi:.2f}% / {100*test_acc_gudhi:.2f}%")

XGN1 = time()
model_classif_noise = XGBClassifier(eval_metric='logloss', use_label_encoder=False)
model_classif_noise.fit(noise_PV_train_gudhi, label_classif_train)
train_acc_noise = model_classif_noise.score(noise_PV_train_gudhi, label_classif_train)
test_acc_noise = model_classif_noise.score(noise_PV_test_gudhi, label_classif_test)
XGN2 = time()
print(f"Gudhi-noise (XGB) train/test: {100*train_acc_noise:.2f}% / {100*test_acc_noise:.2f}%")

XB11 = time()
model_baseline_dtw = KNeighborsClassifier(metric=DTW)
model_baseline_dtw.fit(ts_classif_train, label_classif_train)
train_acc_dtw = model_baseline_dtw.score(ts_classif_train, label_classif_train)
test_acc_dtw = model_baseline_dtw.score(ts_classif_test, label_classif_test)
XB12 = time()
print(f"Baseline DTW k-NN train/test: {100*train_acc_dtw:.2f}% / {100*test_acc_dtw:.2f}%")

XB21 = time()
model_baseline_euc = KNeighborsClassifier(metric='euclidean')
model_baseline_euc.fit(ts_classif_train, label_classif_train)
train_acc_euc = model_baseline_euc.score(ts_classif_train, label_classif_train)
test_acc_euc = model_baseline_euc.score(ts_classif_test, label_classif_test)
XB22 = time()
print(f"Baseline Euclidean k-NN train/test: {100*train_acc_euc:.2f}% / {100*test_acc_euc:.2f}%")

print('Time taken summary: Gudhi XGB={:.2f}s Gudhi-noise XGB={:.2f}s DTW={:.2f}s Eucl={:.2f}s'.format(XG2-XG1, XGN2-XGN1, XB12-XB11, XB22-XB21))
