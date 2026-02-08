import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from itertools import product
from torch.utils.data import TensorDataset, DataLoader
import dill as pck
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
import sys
path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))
print(sys.path)
from helper_fctns.get_names import get_dir_data, get_suffix_dataset
from helper_fctns.create_pointnet import create_pointnet

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Check if the execution is on the cluster or not:
on_cluster = False


# Setting some parameters:
representation  = 'PI'
modelnet_choice = 10
data_name = 'modelnet' + str(modelnet_choice)
normalize_pc = True
homdims_list = [[0,1]]
PI_size = 25
wgt_name = 'quadratic'
pct_noise_list = [0.02, 0.05, 0.1, 0.15, 0.25, 0.5, 0.75, 0.9]
num_folds = 10
folds = range(num_folds)

NUM_CLASSES = modelnet_choice
BATCH_SIZE = 32
NUM_POINTS = 1024
NUM_EPOCHS = 200

#Set the data directories:

cwd = os.getcwd()
print('cwd: ', cwd)
data_base_dir = os.path.join(cwd, f'datasets/saved_datasets/{representation}/{data_name}')
model_save_dir = os.path.join(cwd, f'models/pointnet/{data_name}')
results_dir = os.path.join(cwd, f'results/pointnet/{data_name}')

results_file = os.path.join(results_dir, 'pointnet_results.csv')

figures_output_dir   = os.path.join(results_dir, 'figures')
Path(figures_output_dir).mkdir(parents=True, exist_ok=True)

## Load pretrained model:
model = create_pointnet(NUM_CLASSES=NUM_CLASSES, NUM_POINTS=NUM_POINTS, BATCH_SIZE=BATCH_SIZE)
model = model.to(device)

model_name = f'pointnet{modelnet_choice}_num_points_{NUM_POINTS}_epochs_{NUM_EPOCHS}_batch_size_{BATCH_SIZE}_weights'
model.load_state_dict(torch.load(os.path.join(model_save_dir, model_name + '.pt')))

## Evaluate pointnet on the relevant datasets:

results_list = []

grid = product(pct_noise_list, homdims_list, folds)
for pct_noise, homdims, fold in grid:

    kwargs_data_dir = {
        'representation': representation,
        'data': f'modelnet{modelnet_choice}',
        'dataset_choice': 'test',
        'hom_dim': homdims,
        'PIsz': PI_size,
        'wgt_name': wgt_name,
        'normalize_pc': normalize_pc,
        'num_samples': NUM_POINTS,
        'pct_noise': pct_noise,
        'on_cluster': on_cluster,
    }
    kwargs_suffix = {
        'representation': representation,
        'data': f'modelnet{modelnet_choice}',
        'hom_dim': homdims,
        'PIsz': PI_size,
        'wgt_name': wgt_name,
        'normalize_pc': normalize_pc,
        'num_samples': NUM_POINTS,
        'pct_noise': pct_noise,
        'fold': fold,
    }
    test_data_dir = get_dir_data(**kwargs_data_dir)
    suffix = get_suffix_dataset(**kwargs_suffix)
    test_data_file = os.path.join(test_data_dir, suffix + '.pkl')
    assert (os.path.isfile(test_data_file))
    print('test data file: ', test_data_file)

    ### Load noisy (and clean) pointclouds for evaluation:
    data_eval = pck.load(open(test_data_file, 'rb'))
    pc_test_eval = data_eval['data_test']
    pc_test_eval_clean = data_eval["data_test_clean"]
    label_test_eval = data_eval["label_test"]

    ### Encoding the labels:
    le = LabelEncoder().fit(np.concatenate([label_test_eval]))
    label_test_eval = le.transform(label_test_eval)

    eval_dataset = TensorDataset(
        torch.FloatTensor(pc_test_eval),
        torch.LongTensor(label_test_eval)
    )
    eval_dataset_clean = TensorDataset(
        torch.FloatTensor(pc_test_eval_clean),
        torch.LongTensor(label_test_eval)
    )

    eval_dataloader = DataLoader(eval_dataset, batch_size=BATCH_SIZE, shuffle=True)
    eval_dataloader_clean = DataLoader(eval_dataset_clean, batch_size=BATCH_SIZE, shuffle=True)

    ### Get the evaluation accuracies of the pointnet model:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    
    eval_loss_noisy = 0
    eval_acc_noisy = 0
    with torch.no_grad():
        for batch_x, batch_y in eval_dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs, _, _ = model(batch_x)
            loss = criterion(outputs, batch_y)
            eval_loss_noisy += loss.item()
            preds = torch.argmax(outputs, dim=1)
            eval_acc_noisy += (preds == batch_y).float().mean().item()
    eval_loss_noisy /= len(eval_dataloader)
    eval_acc_noisy /= len(eval_dataloader)
    
    eval_loss_clean = 0
    eval_acc_clean = 0
    with torch.no_grad():
        for batch_x, batch_y in eval_dataloader_clean:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs, _, _ = model(batch_x)
            loss = criterion(outputs, batch_y)
            eval_loss_clean += loss.item()
            preds = torch.argmax(outputs, dim=1)
            eval_acc_clean += (preds == batch_y).float().mean().item()
    eval_loss_clean /= len(eval_dataloader_clean)
    eval_acc_clean /= len(eval_dataloader_clean)
    
    print(f'\nFold {fold}, homdims {str(homdims)}:\n')
    print(f'Pointnet evaluation - clean:     acc: {eval_acc_clean}, loss: {eval_loss_clean}\n')
    print(f'Pointnet evaluation - noisy {pct_noise}: acc: {eval_acc_noisy}, loss: {eval_loss_noisy}')

    results_dict = {
        'pct_noise': pct_noise,
        'homdims': str(homdims).replace(',', ' '),
        'fold': fold,
        'noisy acc': eval_acc_noisy,
        'clean acc': eval_acc_clean,
        'num_classes': NUM_CLASSES,
        'num_epochs': NUM_EPOCHS,
        'num_points': NUM_POINTS,
    }
    results_list.append(results_dict)

results_df = pd.DataFrame(results_list)
add_header = False
if not os.path.isfile(results_file):
    add_header = True
results_df.to_csv(results_file, mode='a', header=add_header)

df_mean_std = results_df.drop(['fold'], axis=1).groupby(['pct_noise', 'homdims', 'num_epochs', 'num_points','num_classes']).agg(['mean', 'std'])

for col in df_mean_std.columns.get_level_values(0).unique():
    print(f'{col}\nnumber of folds: {num_folds}')
    print(df_mean_std[col], '\n')

