import os, sys, json, torch
import numpy as np
import torch.nn.functional as F
from collections import OrderedDict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from models import GTTensorFieldNetworkV2
from datasets.shapes3d import generate_dataset, DATASET_CONFIGS

IS_3D = True
N_POINTS = 600
N_PER_CLASS_TEST = 50
BASE = 'shapes3d_complex'

_, _, data_test, y_test_raw, class_names = generate_dataset(
    BASE, N_PER_CLASS_TEST, N_PER_CLASS_TEST, N_POINTS, noise_sigma=0.0, seed=42)
data_noisy, y_noisy_raw, _, _, _ = generate_dataset(
    BASE, N_PER_CLASS_TEST, 0, N_POINTS, noise_sigma=0.15, seed=123)

ckpts = {
    'baseline(mo0)': ('models/shape_shapes3d_complex_GTTensorFieldNetworkV2_t0-seed42.pt',
                      dict(max_order=0, hidden_channels=16, num_layers=2, num_rbf=64, k_neighbors=16, classifier_dims=[32], use_cov_features=False)),
    'max-order=1':   ('models/shape_shapes3d_complex_GTTensorFieldNetworkV2_t0_hp-mo1-seed42.pt',
                      dict(max_order=1, hidden_channels=16, num_layers=2, num_rbf=64, k_neighbors=16, classifier_dims=[32], use_cov_features=False)),
}

def run(model, clouds):
    model.eval()
    preds = []
    with torch.no_grad():
        for x in clouds:
            x = torch.FloatTensor(x).unsqueeze(0)
            logits = model(x)
            preds.append(logits.argmax(1).item())
    return np.array(preds)

for name, (path, hpc) in ckpts.items():
    if not os.path.exists(path):
        print(f'{name}: checkpoint not found ({path})'); continue
    ck = torch.load(path, map_location='cpu', weights_only=False)
    model = GTTensorFieldNetworkV2(n=3, num_classes=len(class_names), **hpc)
    model.load_state_dict(ck['model_state_dict'])
    y_c = np.array(y_test_raw)
    acc_c = (run(model, data_test) == y_c).mean()
    pred_n = run(model, data_noisy)
    y_n = np.array(y_noisy_raw)
    cm = np.zeros((len(class_names), len(class_names)), dtype=int)
    for p, t in zip(pred_n, y_n):
        cm[t, p] += 1
    print(f'\n=== {name} ===')
    print('  clean acc: %.4f' % acc_c)
    print('  noisy acc: %.4f' % ((pred_n == y_n).mean()))
    print('  rows=true cols=pred')
    print('  ' + '        '.join(class_names))
    for i, r in enumerate(class_names):
        print(f'  {r:<12}' + '  '.join(f'{v:7d}' for v in cm[i]))
