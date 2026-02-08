
"""
Title: Point cloud classification with PointNet
Author: [David Griffiths](https://dgriffiths3.github.io)
Date created: 2020/05/25
Last modified: 2020/05/26
Description: Implementation of PointNet for ModelNet10 classification.
"""
"""
# Point cloud classification
"""

"""
## Introduction
Classification, detection and segmentation of unordered 3D point sets i.e. point clouds
is a core problem in computer vision. This example implements the seminal point cloud
deep learning paper [PointNet (Qi et al., 2017)](https://arxiv.org/abs/1612.00593). For a
detailed introduction on PointNet see [this blog
post](https://medium.com/@luis_gonzales/an-in-depth-look-at-pointnet-111d7efdaa1a).
"""

"""
## Setup
If using colab first install trimesh with `!pip install trimesh`.
"""

import os
print(os.getcwd())
from pathlib import Path
import glob
import trimesh
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from matplotlib import pyplot as plt
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

NUM_POINTS = 1024
BATCH_SIZE = 32
NUM_EPOCHS = 200
suffix = f'_npts_{NUM_POINTS}_epochs_{NUM_EPOCHS}_bs_{BATCH_SIZE}_'


"""
Load the saved point clouds for train and test from files.
"""

# Check if the execution is on the cluster or not:
on_cluster = False


# Setting some parameters:
representation  = 'PI'
modelnet_choice = 10
data_name = 'modelnet' + str(modelnet_choice)
normalize_pc = True
homdims = [0,1]
PI_size = 25
wgt_name = 'quadratic'
pct_noise = 0.02
fold = 0

NUM_CLASSES = modelnet_choice

#Set the data directories:
cwd = os.getcwd()
data_base_dir = os.path.join(cwd, f'datasets/saved_datasets/{representation}/{data_name}')
model_save_dir = os.path.join(cwd, f'models/pointnet/{data_name}')
results_dir = os.path.join(cwd, f'results/pointnet/{data_name}')

figures_output_dir   = os.path.join(results_dir, 'figures')
Path(figures_output_dir).mkdir(parents=True, exist_ok=True)

kwargs_data_dir = {
                   'representation': representation,
                   'data': f'modelnet{modelnet_choice}',
                   'dataset_choice': 'ml_train',
                   'hom_dim': homdims,
                   'PIsz': PI_size,
                   'wgt_name': wgt_name,
                   'normalize_pc': normalize_pc,
                   'num_samples': NUM_POINTS,
                   'pct_noise': 0,
                   'on_cluster': on_cluster,
}

kwargs_suffix = {
                 'representation': representation,
                 'data': f'modelnet{modelnet_choice}',
                 'hom_dim': homdims,
                 'PIsz': PI_size,
                 'wgt_name': wgt_name,
                 'normalize_pc': normalize_pc,
                 'num_samples':NUM_POINTS,
                 'pct_noise': 0,
                 'fold': fold,
                 }

suffix = get_suffix_dataset(**kwargs_suffix)
ml_train_data_dir = get_dir_data(**kwargs_data_dir)
ml_train_data_file = os.path.join(ml_train_data_dir, suffix + '.pkl')
assert(os.path.isfile(ml_train_data_file))

kwargs_data_dir.update({'dataset_choice': 'test', 'pct_noise': pct_noise})
kwargs_suffix.update({'pct_noise': pct_noise})
test_data_dir = get_dir_data(**kwargs_data_dir)
suffix = get_suffix_dataset(**kwargs_suffix)
test_data_file = os.path.join(test_data_dir, suffix + '.pkl')
assert(os.path.isfile(test_data_file))

### Loading the data from files:
# Clean pointclouds for training:
data_trn = pck.load(open(ml_train_data_file, 'rb'))
pc_train_ml_train = data_trn["data_train"]
pc_test_ml_train = data_trn['data_test']
label_train_ml_train = data_trn["label_train"]
label_test_ml_train = data_trn["label_test"]

# noisy (and clean) pointclouds for evaluation:
data_eval = pck.load(open(test_data_file, 'rb'))
pc_test_eval = data_eval['data_test']
pc_test_eval_clean = data_eval["data_test_clean"]
label_test_eval = data_eval["label_test"]


### Encoding the labels:
le = LabelEncoder().fit(np.concatenate([label_train_ml_train, label_test_ml_train, label_test_eval]))
label_train_ml_train = le.transform(label_train_ml_train)
label_test_ml_train = le.transform(label_test_ml_train)
label_test_eval = le.transform(label_test_eval)


"""
Our data can now be converted to PyTorch tensors and created as datasets and dataloaders.
"""

train_dataset = TensorDataset(
    torch.FloatTensor(pc_train_ml_train),
    torch.LongTensor(label_train_ml_train)
)
test_dataset = TensorDataset(
    torch.FloatTensor(pc_test_ml_train),
    torch.LongTensor(label_test_ml_train)
)
eval_dataset = TensorDataset(
    torch.FloatTensor(pc_test_eval),
    torch.LongTensor(label_test_eval)
)
eval_dataset_clean = TensorDataset(
    torch.FloatTensor(pc_test_eval_clean),
    torch.LongTensor(label_test_eval)
)

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)
eval_dataloader = DataLoader(eval_dataset, batch_size=BATCH_SIZE, shuffle=True)
eval_dataloader_clean = DataLoader(eval_dataset_clean, batch_size=BATCH_SIZE, shuffle=True)

## Creating the pointnet model:
model = create_pointnet(NUM_CLASSES=NUM_CLASSES, NUM_POINTS=NUM_POINTS, BATCH_SIZE=BATCH_SIZE)
model = model.to(device)

"""
### Train model
Once the model is defined it can be trained with a standard training loop.
"""

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

history_loss = []
history_val_loss = []

for epoch in range(NUM_EPOCHS):
    # Training
    model.train()
    train_loss = 0
    for batch_x, batch_y in train_dataloader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer.zero_grad()
        outputs, _, _ = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_dataloader)
    history_loss.append(train_loss)
    
    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch_x, batch_y in test_dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs, _, _ = model(batch_x)
            loss = criterion(outputs, batch_y)
            val_loss += loss.item()
    val_loss /= len(test_dataloader)
    history_val_loss.append(val_loss)
    
    if (epoch + 1) % 20 == 0:
        print(f'Epoch {epoch+1}/{NUM_EPOCHS}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

# Save the model
model_name = f'pointnet{modelnet_choice}_num_points_{NUM_POINTS}_epochs_{NUM_EPOCHS}_batch_size_{BATCH_SIZE}_weights'
torch.save(model.state_dict(), os.path.join(model_save_dir, model_name + '.pt'))

"""
## Visualize predictions
We can use matplotlib to visualize our trained model performance.
"""

### Get predictions on noisy and clean evaluation data:
model.eval()
with torch.no_grad():
    points_eval, labels_eval = next(iter(eval_dataloader))
    points_eval_clean, labels_eval_clean = next(iter(eval_dataloader_clean))
    
    points_eval = points_eval[:8].to(device)
    labels_eval = labels_eval[:8].to(device)
    
    points_eval_clean = points_eval_clean[:8].to(device)
    labels_eval_clean = labels_eval_clean[:8].to(device)
    
    # run evaluation data through model
    preds, _, _ = model(points_eval)
    preds = torch.argmax(preds, dim=1)
    
    # run clean evaluation data through model
    preds_clean, _, _ = model(points_eval_clean)
    preds_clean = torch.argmax(preds_clean, dim=1)
    
    points_eval = points_eval.cpu().numpy()


# Get the evaluation accuracies of the pointnet model:
model.eval()
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

print(f'\nPointnet evaluation - clean:     acc: {eval_acc_clean}, loss: {eval_loss_clean}\n')
print(f'Pointnet evaluation - noisy {pct_noise}: acc: {eval_acc_noisy}, loss: {eval_loss_noisy}')

# plot points with predicted class and label
fig = plt.figure(figsize=(15, 10))
for i in range(8):
    ax = fig.add_subplot(2, 4, i + 1, projection="3d")
    ax.scatter(points_eval[i, :, 0], points_eval[i, :, 1], points_eval[i, :, 2], s=0.5)
    ax.set_title(
        "pred: {:}, label: {:}".format(
        le.inverse_transform(preds.cpu().numpy())[i], le.inverse_transform(labels_eval.cpu().numpy())[i]
        )
    )
    ax.set_axis_off()
    plt.suptitle(f'Predictions on noisy evaluation data. Noise: {pct_noise}')
    plt.savefig(os.path.join(figures_output_dir, f'predictions_on_noisy_eval_{suffix}.jpg'))

# plot points with predicted class and label
fig = plt.figure(figsize=(15, 10))
for i in range(8):
    ax = fig.add_subplot(2, 4, i + 1, projection="3d")
    ax.scatter(points_eval_clean[i, :, 0], points_eval_clean[i, :, 1], points_eval_clean[i, :, 2], s=0.5)
    ax.set_title(
        "pred: {:}, label: {:}".format(
            le.inverse_transform(preds_clean.cpu().numpy())[i], le.inverse_transform(labels_eval_clean.cpu().numpy())[i]
        )
    )
    ax.set_axis_off()
    plt.suptitle('Predictions on clean evaluation data.')
plt.savefig(os.path.join(figures_output_dir, f'predictions_on_clean_eval.jpg'))

### Plot loss curves:
loss = history_loss
val_loss = history_val_loss
epochs = range(1, len(loss) + 1)
plt.figure()
plt.plot(epochs[:], loss[:], 'g', label='Training loss')
plt.plot(epochs[:], val_loss[:], 'b', label='Validation loss')
plt.title('RN training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
Path(os.path.join(figures_output_dir, 'loss_curves')).mkdir(parents=True, exist_ok=True)
plt.savefig(os.path.join(figures_output_dir, 'loss_curves', f'pointnet{modelnet_choice}{suffix}loss_on_train.jpg'))

plt.close('all')

