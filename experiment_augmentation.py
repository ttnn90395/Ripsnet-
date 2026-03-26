import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import gridspec
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import gudhi as gd
from gudhi.representations import DiagramSelector, Landscape, PersistenceImage
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.optim import Adamax
from xgboost import XGBClassifier

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adamax

class DenseRagged(nn.Module):
    def __init__(self, in_features=None, out_features=30, activation='relu', use_bias=True):
        super(DenseRagged, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = use_bias
        self.activation = activation

        self.weight = None
        self.bias = None

    def forward(self, inputs):
        outputs = []
        for x in inputs:
            if self.weight is None:
                in_features = x.shape[-1]
                self.weight = nn.Parameter(torch.randn(in_features, self.out_features) * 0.01)
                if self.use_bias:
                    self.bias = nn.Parameter(torch.zeros(self.out_features))
            y = torch.matmul(x, self.weight)
            if self.use_bias:
                y = y + self.bias
            if self.activation == 'relu':
                y = F.relu(y)
            elif self.activation == 'sigmoid':
                y = torch.sigmoid(y)
            elif self.activation == 'tanh':
                y = torch.tanh(y)
            outputs.append(y)
        return outputs


class PermopRagged(nn.Module):
    def forward(self, inputs):
        return torch.stack([torch.sum(x, dim=0) for x in inputs])


class RaggedPersistenceModel(nn.Module):
    def __init__(self, output_dim):
        super(RaggedPersistenceModel, self).__init__()
        self.ragged_layers = nn.ModuleList([
            DenseRagged(out_features=30, activation='relu'),
            DenseRagged(out_features=20, activation='relu'),
            DenseRagged(out_features=10, activation='relu')
        ])
        self.perm = PermopRagged()

        self.fc = nn.Sequential(
            nn.Linear(10, 50),
            nn.ReLU(),
            nn.Linear(50, 100),
            nn.ReLU(),
            nn.Linear(100, 200),
            nn.ReLU(),
            nn.Linear(200, output_dim),
            nn.Sigmoid()
        )

    def forward(self, inputs):
        x = inputs
        for layer in self.ragged_layers:
            x = layer(x)
        x = self.perm(x)
        x = self.fc(x)
        return x

from typing import List

class DistanceMatrixRaggedModel(nn.Module):
    def __init__(self, output_dim, num_points=None, phi_dim=128, rho_hidden=(256,128)):
        """
        output_dim: final descriptor size (e.g., PI_train.shape[1])
        num_points: expected number of points (row length). If None, model infers size at first forward.
        phi_dim: per-row embedding size
        rho_hidden: sizes of hidden layers for global map
        """
        super().__init__()
        self.num_points = num_points
        inp = num_points if num_points is not None else 0
        self._phi_layers = None
        self.phi_dim = phi_dim
        self._build_phi(inp)
        layers = []
        prev = phi_dim
        for h in rho_hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.rho = nn.Sequential(*layers)

    def _build_phi(self, inp):
        if inp <= 0:
            self._phi_layers = None
            return
        hidden = max(64, self.phi_dim)
        self._phi_layers = nn.Sequential(
            nn.Linear(inp, hidden),
            nn.ReLU(),
            nn.Linear(hidden, self.phi_dim),
            nn.ReLU()
        )

    def forward(self, batch: List[torch.Tensor]):
        """
        batch: list of (Ni x Ni) distance matrices (torch.Tensor)
        returns: (B, output_dim)
        """
        if len(batch) == 0:
            return torch.empty(0, self.rho[-1].out_features, device=next(self.parameters()).device)

        sizes = [m.shape[0] for m in batch]
        max_n = max(sizes)
        device = next(self.parameters()).device

        if self._phi_layers is None or (self.num_points and self.num_points != max_n):
            self._build_phi(max_n)
            self.num_points = max_n
            self._phi_layers = self._phi_layers.to(device)

        B = len(batch)
        mats = torch.zeros((B, max_n, max_n), dtype=torch.float32, device=device)
        row_mask = torch.zeros((B, max_n), dtype=torch.bool, device=device)
        for i, m in enumerate(batch):
            n = m.shape[0]
            mats[i, :n, :n] = m.to(device).float()
            row_mask[i, :n] = True # Changed 1 to True for boolean tensor

        rows = mats.reshape(B * max_n, max_n)
        phi_out = self._phi_layers(rows)
        phi_out = phi_out.reshape(B, max_n, -1)

        lengths = row_mask.sum(dim=1).clamp(min=1).unsqueeze(-1).float()
        summed = (phi_out * row_mask.unsqueeze(-1).float()).sum(dim=1)
        aggregated = summed / lengths

        out = self.rho(aggregated)
        return out


import numpy as np
from tqdm import tqdm
import gudhi as gd


####################################
###  Creation of point clouds    ###
####################################

def create_circle(N_points, r, x_0, y_0):
    X = []
    for i in range(N_points):
        theta = np.random.uniform() * 2 * np.pi
        X.append([(r * np.cos(theta)) + x_0, (r * np.sin(theta) + y_0)])
    return np.array(X)


def create_1_circle_clean(N_points):
    r = 2
    x_0, y_0 = 10 * np.random.rand() - 5, 10 * np.random.rand() - 5
    return create_circle(N_points, r, x_0, y_0)


def create_2_circle_clean(N_points):
    r1 = 5
    r2 = 3
    x_0, y_0 = 30 * np.random.rand() - 15, 30 * np.random.rand() - 15
    x_1, y_1 = 30 * np.random.rand() - 15, 30 * np.random.rand() - 15
    while np.sqrt((x_0 - x_1) ** 2 + (y_0 - y_1) ** 2) <= r1 + r2:
        x_1, y_1 = 30 * np.random.rand() - 15, 30 * np.random.rand() - 15
    circle1 = create_circle(N_points // 2, r1, x_0, y_0)
    circle2 = create_circle(N_points - N_points // 2, r2, x_1, y_1)
    X = [0] * N_points
    X[:N_points // 2] = circle1
    X[N_points // 2:] = circle2
    np.random.shuffle(X)
    return np.array(X)


def create_3_circle_clean(N_points):
    r0 = 5
    r1 = 3
    r2 = 2
    x_0, y_0 = 30 * np.random.rand() - 15, 30 * np.random.rand() - 15
    x_1, y_1 = 30 * np.random.rand() - 15, 30 * np.random.rand() - 15
    while np.sqrt((x_0 - x_1) ** 2 + (y_0 - y_1) ** 2) <= r0 + r1:
        x_1, y_1 = 30 * np.random.rand() - 15, 30 * np.random.rand() - 15

    x_2, y_2 = 30 * np.random.rand() - 15, 30 * np.random.rand() - 15
    while(np.sqrt((x_0 - x_2)**2 + (y_0 - y_2)**2) <= r0 + r2) or (np.sqrt((x_1 - x_2)**2 + (y_1 - y_2)**2) <= r1 + r2):
        x_2, y_2 = 30 * np.random.rand() - 15, 30 * np.random.rand() - 15

    circle0 = create_circle(N_points // 3, r0, x_0, y_0)
    circle1 = create_circle(N_points // 3, r1, x_1, y_1)
    circle2 = create_circle(N_points // 3, r2, x_2, y_2)

    # Handler in case N_points mod 3 != 0.
    true_N_points = 3 * (N_points // 3)

    X = [[0,0]] * true_N_points
    X[:true_N_points // 3] = circle0
    X[true_N_points // 3:2 * true_N_points // 3] = circle1
    X[2 * true_N_points // 3:] = circle2
    np.random.shuffle(X)
    return np.array(X)


def create_1_circle_noisy(N_points, N_noise):
    r = 2
    x_0, y_0 = 10 * np.random.rand() - 5, 10 * np.random.rand() - 5
    X = create_circle(N_points, r, x_0, y_0)
    noise = []
    for i in range(N_noise):
        noise.append([np.random.uniform(x_0 - r, x_0 + r),
                      np.random.uniform(y_0 - r, y_0 + r)])
    X = np.array(X)
    X[np.random.choice(np.arange(len(X)), size=N_noise, replace=False, p=None)] = np.array(noise)
    return X


def create_2_circle_noisy(N_points, N_noise):
    r1 = 5
    r2 = 3
    x_0, y_0 = 30 * np.random.rand() - 15, 30 * np.random.rand() - 15
    x_1, y_1 = 30 * np.random.rand() - 15, 30 * np.random.rand() - 15
    while(np.sqrt((x_0 - x_1)**2 + (y_0 - y_1)**2) <= r1 + r2):
        x_1, y_1 = 30 * np.random.rand() - 15, 30 * np.random.rand() - 15
    circle1 = create_circle(N_points // 2, r1, x_0, y_0)
    circle2 = create_circle(N_points - N_points // 2, r2, x_1, y_1)
    X = [0] * N_points
    X[:N_points // 2] = circle1
    X[N_points // 2:] = circle2
    np.random.shuffle(X)
    noise = []
    for i in range(N_noise):
        noise.append([np.random.uniform(min(x_0 - r1, x_1 - r2), max(x_0 + r1, x_1 + r2)),
                      np.random.uniform(min(y_0 - r1, y_1 - r2), max(y_0 + r1, y_1 + r2))])
    X = np.array(X)
    X[np.random.choice(np.arange(len(X)), size=N_noise, replace=False, p=None)] = np.array(noise)
    return X


def create_3_circle_noisy(N_points, N_noise):
    r0 = 5
    r1 = 3
    r2 = 2
    x_0, y_0 = 30 * np.random.rand() - 15, 30 * np.random.rand() - 15
    x_1, y_1 = 30 * np.random.rand() - 15, 30 * np.random.rand() - 15
    while np.sqrt((x_0 - x_1) ** 2 + (y_0 - y_1) ** 2) <= r0 + r1:
        x_1, y_1 = 30 * np.random.rand() - 15, 30 * np.random.rand() - 15
    x_2, y_2 = 30 * np.random.rand() - 15, 30 * np.random.rand() - 15
    while(np.sqrt((x_0 - x_2)**2 + (y_0 - y_2)**2) <= r0 + r2) or (np.sqrt((x_1 - x_2)**2 + (y_1 - y_2)**2) <= r1 + r2):
        x_2, y_2 = 30 * np.random.rand() - 15, 30 * np.random.rand() - 15
    circle0 = create_circle(N_points // 3, r0, x_0, y_0)
    circle1 = create_circle(N_points // 3, r1, x_1, y_1)
    circle2 = create_circle(N_points // 3, r2, x_2, y_2)

    true_N_points = 3 * (N_points // 3)
    X = [[0,0]] * true_N_points
    X[:true_N_points // 3] = circle0
    X[true_N_points // 3:2 * true_N_points // 3] = circle1
    X[2 * true_N_points // 3:] = circle2

    np.random.shuffle(X)
    noise = []
    for i in range(N_noise):
        noise.append([np.random.uniform(np.min([x_0 - r0, x_1 - r1, x_2 - r2]), np.max([x_0 + r0, x_1 + r1, x_2 + r2])),
                      np.random.uniform(np.min([y_0 - r0, y_1 - r1, y_2 - r2]), np.max([y_0 + r0, y_1 + r1, y_2 + r2]))])
    X = np.array(X)
    X[np.random.choice(np.arange(len(X)), size=N_noise, replace=False, p=None)] = np.array(noise)
    return X

def augment_isometries(pc, n, rng, trans_frac=0.08):
    bbox = pc.max(axis=0) - pc.min(axis=0)
    t_max = trans_frac * np.linalg.norm(bbox)  # translation scale relative to cloud size
    augmented = []
    for _ in range(n):
        theta = rng.uniform(0, 2 * np.pi)
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        t = rng.uniform(-t_max, t_max, size=2)
        augmented.append((pc @ R.T) + t)
    return augmented

def data_augmentation_by_isometries(data_train, label_train, n_augment_per_sample, seed=42):
    rng = np.random.default_rng(seed)
    aug_data, aug_labels = [], []
    for pc, lbl in zip(data_train, label_train):
        aug_data.append(pc)
        aug_labels.append(lbl)
        for pc_aug in augment_isometries(pc, n_augment_per_sample, rng):
            aug_data.append(pc_aug)
            aug_labels.append(lbl)
    return aug_data, aug_labels

def augment_permutations(pc, n, rng):
    augmented = []
    for _ in range(n):
        shuffled_pc = rng.permutation(pc, axis=0) # Permute rows (points) of the point cloud
        augmented.append(shuffled_pc)
    return augmented

def create_multiple_circles(N_sets_train, N_points, noisy=False, N_noise=0, n_augment_per_sample = 0):

    data_train, PD_train = [[] for _ in range(N_sets_train)], []
    label_train = np.zeros((N_sets_train,))

    if not noisy:
        for i in tqdm(range(N_sets_train // 3)):
            data_train[i] = create_1_circle_clean(N_points)
            label_train[i] = 1
        for i in tqdm(range(N_sets_train // 3, 2 * N_sets_train // 3)):
            data_train[i] = create_2_circle_clean(N_points)
            label_train[i] = 2
        for i in tqdm(range(2 * N_sets_train // 3, N_sets_train)):
            data_train[i] = create_3_circle_clean(N_points)
            label_train[i] = 3
    else:
        for i in tqdm(range(N_sets_train // 3)):
            data_train[i] = create_1_circle_noisy(N_points, N_noise)
            label_train[i] = 1
        for i in tqdm(range(N_sets_train // 3, 2 * N_sets_train // 3)):
            data_train[i] = create_2_circle_noisy(N_points, N_noise)
            label_train[i] = 2
        for i in tqdm(range(2 * N_sets_train // 3, N_sets_train)):
            data_train[i] = create_3_circle_noisy(N_points, N_noise)
            label_train[i] = 3

    shuffler = np.random.permutation(len(data_train))
    label_train = label_train[shuffler]
    data_train = [data_train[p] for p in shuffler]
    if (n_augment_per_sample > 0):
        data_train,label_train = data_augmentation_by_isometries(data_train, label_train, n_augment_per_sample, seed=42)
    return data_train, label_train


############################################
### Computation of persistence diagrams  ###
############################################


def compute_PD(dataset, i):
    u = np.array(dataset[i])
    rcX = gd.AlphaComplex(points=u).create_simplex_tree()
    rcX.persistence()
    dgm = rcX.persistence_intervals_in_dimension(1)
    return dgm


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.array(x)

def train_model(model, optimizer, criterion, train_inputs, train_targets, val_inputs, val_targets, epochs=20, batch_size=32):
    """
    Train a ragged-input model. Returns (best_model, history, best_model_state).
    """
    model = model.to(device)
    patience = 20
    best_val_loss = float('inf')
    patience_counter = 0
    num_epochs = epochs if epochs is not None else 10000
    history = {'train_loss': [], 'val_loss': []}
    best_model_state = None

    # helper to move inputs to device
    def to_device_list(lst):
        out = []
        for x in lst:
            if isinstance(x, torch.Tensor):
                out.append(x.to(device).float())
            else:
                out.append(torch.tensor(x, dtype=torch.float32, device=device))
        return out

    train_inputs = to_device_list(train_inputs)
    val_inputs = to_device_list(val_inputs)

    if isinstance(train_targets, torch.Tensor):
        train_targets = train_targets.to(device).float()
    else:
        train_targets = torch.tensor(train_targets, dtype=torch.float32, device=device)

    if isinstance(val_targets, torch.Tensor):
        val_targets = val_targets.to(device).float()
    else:
        val_targets = torch.tensor(val_targets, dtype=torch.float32, device=device)

    for epoch in range(num_epochs):
        model.train()
        permutation = torch.randperm(len(train_inputs), device=device)
        epoch_loss = 0.0
        for i in range(0, len(train_inputs), batch_size):
            optimizer.zero_grad()
            indices = permutation[i:i + batch_size]
            batch_inputs = [train_inputs[int(idx)].to(device) for idx in indices]
            batch_targets = train_targets[indices]

            outputs = model(batch_inputs)
            if isinstance(outputs, (list, tuple)):
                outputs = outputs[0]
            if not isinstance(outputs, torch.Tensor):
                outputs = torch.tensor(outputs, dtype=batch_targets.dtype, device=device)
            else:
                outputs = outputs.to(device).type(batch_targets.dtype)

            loss = criterion(outputs, batch_targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(batch_inputs)

        epoch_loss /= len(train_inputs)
        history['train_loss'].append(epoch_loss)

        model.eval()
        with torch.no_grad():
            val_outputs = model(val_inputs)
            if isinstance(val_outputs, (list, tuple)):
                val_outputs = val_outputs[0]
            val_outputs = val_outputs.to(device).type(val_targets.dtype)
            val_loss = criterion(val_outputs, val_targets).item()
            history['val_loss'].append(val_loss)

        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss:.6f}, Val Loss: {val_loss:.6f}')

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            patience_counter = 0
            # save best model (on CPU to avoid holding GPU memory)
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    # if we saved a best state, load it back into the model on the current device
    if best_model_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})

    return model, history, best_model_state


def combine_augmentations(point_cloud, n_augment_per_sample, rng):
    """
    Combines isometric augmentations and permutation augmentations.
    For each isometry, a single permutation is applied.
    """
    # Step 2: Generate isometrically augmented point clouds
    isometry_augmented_pcs = augment_isometries(point_cloud, n_augment_per_sample, rng)

    # Step 3: Initialize list to store combined augmented point clouds
    combined_augmented_pcs = []

    # Step 4 & 5: Iterate through isometrically augmented PCs and apply permutation
    for iso_pc in isometry_augmented_pcs:
        # Generate a single permuted version (n=1) for each isometrically augmented PC
        permuted_iso_pc = augment_permutations(iso_pc, 1, rng)
        # Extend the list with the permuted point cloud (which is a list of one PC)
        combined_augmented_pcs.extend(permuted_iso_pc)

    # Step 7: Return the combined augmented point clouds
    return combined_augmented_pcs

print("Combined augmentation function `combine_augmentations` defined.")

def compute_smoothed_robustness_score(pc, model, augmentation_fn, n_augment_per_score, sigma, k, device, descriptor_type, seed=42):
    rng = np.random.default_rng(seed)
    model.eval()

    # 1. Smooth the original point cloud
    smoothed_pc_original = gaussian_smoothing(pc, sigma=sigma, k=k)

    # Determine input type based on model class
    is_distance_matrix_model = False
    if 'DistanceMatrixRaggedModel' in globals() and isinstance(model, globals()['DistanceMatrixRaggedModel']):
        is_distance_matrix_model = True

    # Prepare input for the original smoothed point cloud
    if is_distance_matrix_model:
        dm_smoothed_original = distance_matrix(smoothed_pc_original)
        input_original = [torch.tensor(dm_smoothed_original, dtype=torch.float32).to(device)]
    else:
        input_original = [torch.tensor(smoothed_pc_original, dtype=torch.float32).to(device)]

    # 2. Compute vector output for the original smoothed point cloud
    with torch.no_grad():
        pred_original = model(input_original)
        if isinstance(pred_original, (list, tuple)):
            pred_original = pred_original[0]
        pred_original = pred_original.cpu().numpy().flatten()

    # 3. Generate augmented point clouds (e.g., by isometry or permutation)
    augmented_pcs = augmentation_fn(pc, n_augment_per_score, rng)

    euclidean_distances = []
    for aug_pc in augmented_pcs:
        # 4. Smooth each augmented point cloud
        smoothed_aug_pc = gaussian_smoothing(aug_pc, sigma=sigma, k=k)

        # Prepare input for the smoothed augmented point cloud
        if is_distance_matrix_model:
            dm_smoothed_aug = distance_matrix(smoothed_aug_pc)
            input_augmented = [torch.tensor(dm_smoothed_aug, dtype=torch.float32).to(device)]
        else:
            input_augmented = [torch.tensor(smoothed_aug_pc, dtype=torch.float32).to(device)]

        # 5. Compute vector output for each smoothed augmented point cloud
        with torch.no_grad():
            pred_augmented = model(input_augmented)
            if isinstance(pred_augmented, (list, tuple)):
                pred_augmented = pred_augmented[0]
            pred_augmented = pred_augmented.cpu().numpy().flatten()

        # 6. Calculate L2 (Euclidean) distance between original smoothed and augmented smoothed predictions
        dist = np.linalg.norm(pred_original - pred_augmented)
        euclidean_distances.append(dist)

    # 7. Return the mean and standard deviation of these distances
    if len(euclidean_distances) > 0:
        return np.mean(euclidean_distances), np.std(euclidean_distances)
    else:
        return 0.0, 0.0

def compute_smoothed_ensemble_robustness(pc, ensemble_model, augmentation_fn, n_augment_per_score, sigma, k, device, descriptor_type, seed=42):
    rng = np.random.default_rng(seed)

    if isinstance(ensemble_model, torch.nn.Module):
        ensemble_model.eval()

    # 1. Smooth the original point cloud
    smoothed_pc_original = gaussian_smoothing(pc, sigma=sigma, k=k)

    # Get ensemble input for the original smoothed point cloud
    ensemble_input_original = get_ensemble_input_from_pc(smoothed_pc_original, device, descriptor_type)
    pred_original_ensemble = ensemble_model.predict_proba(ensemble_input_original.reshape(1, -1)).flatten()

    # 2. Generate augmented point clouds
    augmented_pcs = augmentation_fn(pc, n_augment_per_score, rng)

    euclidean_distances = []
    for aug_pc in augmented_pcs:
        # 3. Smooth each augmented point cloud
        smoothed_aug_pc = gaussian_smoothing(aug_pc, sigma=sigma, k=k)

        # Get ensemble input for the smoothed augmented point cloud
        ensemble_input_augmented = get_ensemble_input_from_pc(smoothed_aug_pc, device, descriptor_type)
        pred_augmented_ensemble = ensemble_model.predict_proba(ensemble_input_augmented.reshape(1, -1)).flatten()

        # 4. Calculate L2 (Euclidean) distance
        dist = np.linalg.norm(pred_original_ensemble - pred_augmented_ensemble)
        euclidean_distances.append(dist)

    if len(euclidean_distances) > 0:
        return np.mean(euclidean_distances), np.std(euclidean_distances)
    else:
        return 0.0, 0.0

print("Smoothed robustness functions defined.")


def get_ensemble_input_from_pc(pc, device, descriptor_type):
    # pc is a numpy array
    # descriptor_type: 'pi' or 'pl'

    # Ensure PC is a list of tensors for models expecting ragged inputs
    pc_tensor_list = [torch.tensor(pc, dtype=torch.float32).to(device)]

    # --- 1. Get raw descriptors/predictions from base models ---
    # Gudhi (requires re-computation of PD and then PI/PL transform)
    st = gd.AlphaComplex(points=pc).create_simplex_tree(max_alpha_square=maxd)
    st.persistence()
    dg = st.persistence_intervals_in_dimension(1)
    if len(dg) == 0:
        dg = np.empty([0,2])
    # DiagramSelector expects a list of diagrams
    pds_pc = DiagramSelector(use=True).fit_transform([dg])

    if descriptor_type == 'pi':
        gudhi_descriptor = PI.transform(pds_pc)
        rn_descriptor = model_PI(pc_tensor_list).detach().cpu().numpy()
        # Distance Matrix model
        dm = distance_matrix(pc)
        dm_descriptor = model_dm_pi([torch.tensor(dm, dtype=torch.float32).to(device)]).detach().cpu().numpy()
        pn_descriptor = model_PN_PI(pc_tensor_list).detach().cpu().numpy()

        # Ensure outputs are normalized consistently with training data for XGBoost
        gudhi_descriptor = gudhi_descriptor / MPI
        rn_descriptor = rn_descriptor / MPI
        dm_descriptor = dm_descriptor / MPI
        pn_descriptor = pn_descriptor / MPI

        # Ensure all descriptors are 2D arrays for predict_proba
        gudhi_descriptor = gudhi_descriptor.reshape(1, -1)
        rn_descriptor = rn_descriptor.reshape(1, -1)
        dm_descriptor = dm_descriptor.reshape(1, -1)
        pn_descriptor = pn_descriptor.reshape(1, -1)

        # --- 2. Get probabilities from base XGBoost classifiers ---
        proba_ripsnet = model_classif_RN_pi_base.predict_proba(rn_descriptor)
        proba_dm = model_classif_dm_pi_base.predict_proba(dm_descriptor)
        proba_pointnet = model_classif_PN_pi_base.predict_proba(pn_descriptor)

    elif descriptor_type == 'pl':
        gudhi_descriptor = PL.transform(pds_pc)
        rn_descriptor = model_PL(pc_tensor_list).detach().cpu().numpy()
        # Distance Matrix model
        dm = distance_matrix(pc)
        dm_descriptor = model_dm_pl([torch.tensor(dm, dtype=torch.float32).to(device)]).detach().cpu().numpy()
        pn_descriptor = model_PN_PL(pc_tensor_list).detach().cpu().numpy()

        # Ensure outputs are normalized consistently with training data for XGBoost
        gudhi_descriptor = gudhi_descriptor / MPL
        rn_descriptor = rn_descriptor / MPL
        dm_descriptor = dm_descriptor / MPL
        pn_descriptor = pn_descriptor / MPL

        # Ensure all descriptors are 2D for predict_proba
        gudhi_descriptor = gudhi_descriptor.reshape(1, -1)
        rn_descriptor = rn_descriptor.reshape(1, -1)
        dm_descriptor = dm_descriptor.reshape(1, -1)
        pn_descriptor = pn_descriptor.reshape(1, -1)

        # --- 2. Get probabilities from base XGBoost classifiers ---
        proba_ripsnet = model_classif_RN_pl_base.predict_proba(rn_descriptor)
        proba_dm = model_classif_dm_pl_base.predict_proba(dm_descriptor)
        proba_pointnet = model_classif_PN_pl_base.predict_proba(pn_descriptor)
    else:
        raise ValueError("descriptor_type must be 'pi' or 'pl'")

    # Concatenate probabilities to form input for ensemble model
    # Exclude proba_gudhi as the ensemble model was trained without it.
    ensemble_input = np.concatenate([proba_ripsnet, proba_dm, proba_pointnet], axis=1)
    return ensemble_input.flatten()

def compute_ensemble_robustness(pc, ensemble_model, augmentation_fn, n_augment_per_score, device, descriptor_type, seed=42):
    rng = np.random.default_rng(seed)
    # LogisticRegression does not have an 'eval' mode like torch.nn.Module, so skip if not applicable
    if isinstance(ensemble_model, torch.nn.Module):
        ensemble_model.eval()

    # Get ensemble input for the original point cloud
    ensemble_input_original = get_ensemble_input_from_pc(pc, device, descriptor_type)

    # Predict probabilities with the ensemble model
    # LogisticRegression expects 2D array, so reshape if needed
    pred_original_ensemble = ensemble_model.predict_proba(ensemble_input_original.reshape(1, -1)).flatten()

    augmented_pcs = augmentation_fn(pc, n_augment_per_score, rng)

    euclidean_distances = []
    for aug_pc in augmented_pcs:
        # Get ensemble input for the augmented point cloud
        ensemble_input_augmented = get_ensemble_input_from_pc(aug_pc, device, descriptor_type)
        # Predict probabilities with the ensemble model
        pred_augmented_ensemble = ensemble_model.predict_proba(ensemble_input_augmented.reshape(1, -1)).flatten()

        # 4. Calculate L2 (Euclidean) distance
        dist = np.linalg.norm(pred_original_ensemble - pred_augmented_ensemble)
        euclidean_distances.append(dist)

    if len(euclidean_distances) > 0:
        return np.mean(euclidean_distances), np.std(euclidean_distances)
    else:
        return 0.0, 0.0

print("Helper functions for ensemble robustness defined.")

def augment_permutations(pc, n, rng):
    augmented = []
    for _ in range(n):
        shuffled_pc = rng.permutation(pc, axis=0) # Permute rows (points) of the point cloud
        augmented.append(shuffled_pc)
    return augmented

def compute_permutation_robustness_score(pc, model, n_augment_per_score, device, seed=42):
    """
    Calculates the average Euclidean (L2) distance between the vector prediction of an original
    point cloud and those of its permuted versions.
    It adapts its input based on the model type (PointNet/RaggedPersistenceModel vs DistanceMatrixRaggedModel).
    """
    rng = np.random.default_rng(seed)
    model.eval()

    # Determine input type based on model class
    is_distance_matrix_model = False
    if 'DistanceMatrixRaggedModel' in globals() and isinstance(model, globals()['DistanceMatrixRaggedModel']):
        is_distance_matrix_model = True

    # Prepare input for the original point cloud
    if is_distance_matrix_model:
        # Convert point cloud to distance matrix. 'distance_matrix' function must be in scope.
        dm_original = distance_matrix(pc)
        input_original = [torch.tensor(dm_original, dtype=torch.float32).to(device)]
    else:
        # Use point cloud directly
        input_original = [torch.tensor(pc, dtype=torch.float32).to(device)]

    # 1. Compute vector output for the original point cloud
    with torch.no_grad():
        pred_original = model(input_original)
        if isinstance(pred_original, (list, tuple)):
            pred_original = pred_original[0]
        pred_original = pred_original.cpu().numpy().flatten()

    # 2. Generate augmented point clouds by permutation
    permuted_pcs = augment_permutations(pc, n_augment_per_score, rng)

    euclidean_distances = []
    for perm_pc in permuted_pcs:
        # Prepare input for permuted point cloud
        if is_distance_matrix_model:
            dm_permuted = distance_matrix(perm_pc)
            input_permuted = [torch.tensor(dm_permuted, dtype=torch.float32).to(device)]
        else:
            input_permuted = [torch.tensor(perm_pc, dtype=torch.float32).to(device)]

        # 3. Compute vector output for each permuted point cloud
        with torch.no_grad():
            pred_permuted = model(input_permuted)
            if isinstance(pred_permuted, (list, tuple)):
                pred_permuted = pred_permuted[0]
            pred_permuted = pred_permuted.cpu().numpy().flatten()

        # 4. Calculate L2 (Euclidean) distance
        dist = np.linalg.norm(pred_original - pred_permuted)
        euclidean_distances.append(dist)

    # 5. Return the mean of these distances
    if len(euclidean_distances) > 0:
        return np.mean(euclidean_distances)
    else:
        return 0.0

print("Functions `augment_permutations` and `compute_permutation_robustness_score` defined.")

def distance_matrix(point_cloud):
    """
    Compute pairwise Euclidean distance matrix for a point cloud.
    Accepts numpy array, list-of-lists, or torch.Tensor of shape (N, d).
    Returns a numpy array of shape (N, N).
    """
    # rely on existing imports: np, torch
    if isinstance(point_cloud, __import__("torch").Tensor):
        point_cloud = point_cloud.cpu().numpy()
    pc = np.asarray(point_cloud, dtype=float)
    if pc.ndim == 1:
        pc = pc.reshape(-1, 1)
    diff = pc[:, None, :] - pc[None, :, :]
    return np.sqrt((diff ** 2).sum(axis=-1))

N_sets_train = 900  # Number of train point clouds
N_sets_test  = 300  # Number of test  point clouds
N_points     = 600  # Point cloud cardinality
N_noise      = 200  # Number of corrupted points

data_train,      label_train       = create_multiple_circles(N_sets_train, N_points, noisy=0, N_noise=N_noise, n_augment_per_sample= 0)
clean_data_test, clean_label_test  = create_multiple_circles(N_sets_test,  N_points, noisy=0, N_noise=N_noise,n_augment_per_sample= 0)
noisy_data_test, noisy_label_test  = create_multiple_circles(N_sets_test,  N_points, noisy=1, N_noise=N_noise)

# --- Inserted Distance Matrix Computations ---
dm_train = []
for X in tqdm(data_train, desc='Computing DM for training data'):
    dm_train.append(distance_matrix(X))

dm_clean_test = []
for X in tqdm(clean_data_test, desc='Computing DM for clean test data'):
    dm_clean_test.append(distance_matrix(X))

dm_noisy_test = []
for X in tqdm(noisy_data_test, desc='Computing DM for noisy test data'):
    dm_noisy_test.append(distance_matrix(X))

print(f"Shape of first training distance matrix: {dm_train[0].shape}")
print(f"Shape of first clean test distance matrix: {dm_clean_test[0].shape}")
print(f"Shape of first noisy test distance matrix: {dm_noisy_test[0].shape}")
# --- End of Inserted Distance Matrix Computations ---
ds = [pairwise_distances(X).flatten() for X in data_train[:30]]
maxd = np.max(np.concatenate(ds))

n_augment_levels = list(range(2))

clean_accuracies = []
noisy_accuracies = []
clean_permutation_robustness_scores = []
noisy_permutation_robustness_scores = []

subset_size_for_robustness = 50

print(f"Defined n_augment_levels: {n_augment_levels}")
print(f"Initialized empty lists for accuracies and robustness scores.")
print(f"Set subset_size_for_robustness to: {subset_size_for_robustness}")


def data_augmentation_by_permutations(data_train, label_train, n_augment_per_sample, seed=42):
    rng = np.random.default_rng(seed)
    aug_data, aug_labels = [], []
    for pc, lbl in zip(data_train, label_train):
        aug_data.append(pc)
        aug_labels.append(lbl)
        for pc_aug in augment_permutations(pc, n_augment_per_sample, rng):
            aug_data.append(pc_aug)
            aug_labels.append(lbl)
    return aug_data, aug_labels

print("data_augmentation_by_permutations function defined.")


PD_train = []
for X in tqdm(data_train):
    st = gd.AlphaComplex(points=X).create_simplex_tree(max_alpha_square=maxd)
    st.persistence()
    dg = st.persistence_intervals_in_dimension(1)
    if len(dg) == 0:
        dg = np.empty([0,2])
    PD_train.append(dg)

clean_PD_test = []
for X in tqdm(clean_data_test):
    st = gd.AlphaComplex(points=X).create_simplex_tree(max_alpha_square=maxd)
    st.persistence()
    dg = st.persistence_intervals_in_dimension(1)
    if len(dg) == 0:
        dg = np.empty([0,2])
    clean_PD_test.append(dg)

noisy_PD_test = []
for X in tqdm(noisy_data_test):
    st = gd.AlphaComplex(points=X).create_simplex_tree(max_alpha_square=maxd)
    st.persistence()
    dg = st.persistence_intervals_in_dimension(1)
    if len(dg) == 0:
        dg = np.empty([0,2])
    noisy_PD_test.append(dg)

pds_train      = DiagramSelector(use=True).fit_transform(PD_train)
clean_pds_test = DiagramSelector(use=True).fit_transform(clean_PD_test)
noisy_pds_test = DiagramSelector(use=True).fit_transform(noisy_PD_test)

vpdtr = np.vstack(pds_train)
pers = vpdtr[:,1]-vpdtr[:,0]
bps_pairs = pairwise_distances(np.hstack([vpdtr[:,0:1],vpdtr[:,1:2]-vpdtr[:,0:1]])[:200]).flatten()
ppers = bps_pairs[np.argwhere(bps_pairs > 1e-5).ravel()]
sigma = np.quantile(ppers, .2)
im_bnds = [np.quantile(vpdtr[:,0],0.), np.quantile(vpdtr[:,0],1.), np.quantile(pers,0.), np.quantile(pers,1.)]
sp_bnds = [np.quantile(vpdtr[:,0],0.), np.quantile(vpdtr[:,1],1.)]

if 'im_bnds' not in locals() or im_bnds is None:
    all_points = np.concatenate(pds_train, axis=0)
    x_min, x_max = np.min(all_points[:, 0]), np.max(all_points[:, 0])
    y_min, y_max = np.min(all_points[:, 1]), np.max(all_points[:, 1])
    im_bnds = [x_min, x_max, y_min, y_max]

PI_params = {
    'bandwidth': sigma,
    'weight': lambda x: 10 * np.tanh(x[1]),
    'resolution': [50, 50],
    'im_range': im_bnds
}

PI = PersistenceImage(**PI_params)
PI.fit(pds_train)

PI_train = PI.transform(pds_train)
clean_PI_test = PI.transform(clean_pds_test)
noisy_PI_test = PI.transform(noisy_pds_test)

MPI = np.max(PI_train)
if MPI == 0 or np.isnan(MPI):
    raise ValueError("MPI (max value) is zero or NaN; check persistence diagrams.")
else:
    PI_train /= MPI
    clean_PI_test /= MPI
    noisy_PI_test /= MPI

PI_train = torch.tensor(PI_train, dtype=torch.float32)
clean_PI_test = torch.tensor(clean_PI_test, dtype=torch.float32)
noisy_PI_test = torch.tensor(noisy_PI_test, dtype=torch.float32)

print(f"PI_train shape: {PI_train.shape}")
print(f"clean_PI_test shape: {clean_PI_test.shape}")
print(f"noisy_PI_test shape: {noisy_PI_test.shape}")
print(f"Max pixel intensity (MPI): {MPI:.5f}")


if 'sp_bnds' not in locals() or sp_bnds is None:
    all_points = np.concatenate(pds_train, axis=0)
    x_min, x_max = np.min(all_points[:, 0]), np.max(all_points[:, 1])
    sp_bnds = [x_min, x_max]


PL_params = {
    'num_landscapes': 5,
    'resolution': 300,
    'sample_range': sp_bnds
}

PL = Landscape(**PL_params)
PL.fit(pds_train)

PL_train = PL.transform(pds_train)
clean_PL_test = PL.transform(clean_pds_test)
noisy_PL_test = PL.transform(noisy_pds_test)

MPL = np.max(PL_train)
if MPL == 0 or np.isnan(MPL):
    raise ValueError("MPL (max landscape value) is zero or NaN; check your persistence diagrams.")
else:
    PL_train /= MPL
    clean_PL_test /= MPL
    noisy_PL_test /= MPL

PL_train = torch.tensor(PL_train, dtype=torch.float32)
clean_PL_test = torch.tensor(clean_PL_test, dtype=torch.float32)
noisy_PL_test = torch.tensor(noisy_PL_test, dtype=torch.float32)

print(f"PL_train shape: {PL_train.shape}")
print(f"clean_PL_test shape: {clean_PL_test.shape}")
print(f"noisy_PL_test shape: {noisy_PL_test.shape}")
print(f"Max landscape value (MPL): {MPL:.5f}")


# Ensure 'device' is defined globally
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Lists to store results
clean_accuracies = []
noisy_accuracies = []
clean_permutation_robustness_scores_means = []
noisy_permutation_robustness_scores_means = []

# Store original training data/labels to reuse in each iteration
original_data_train = data_train
original_label_train = label_train

# Prepare original test data for robustness calculation (subset for efficiency)
# These subsets will be fixed for all augmentation levels
np.random.seed(42)  # for reproducibility of subset selection
subset_size_for_robustness = 50
clean_test_subset_indices = np.random.choice(len(clean_data_test), subset_size_for_robustness, replace=False)
noisy_test_subset_indices = np.random.choice(len(noisy_data_test), subset_size_for_robustness, replace=False)

clean_data_test_subset = [clean_data_test[i] for i in clean_test_subset_indices]
noisy_data_test_subset = [noisy_data_test[i] for i in noisy_test_subset_indices]

# Prepare original test data (full sets) for model evaluation. Convert to tensors once.
dm_clean_test_tensors = [torch.tensor(dm, dtype=torch.float32).to(device) for dm in dm_clean_test]
clean_PI_test_tensor = clean_PI_test.to(device)  # already a tensor, just move to device
dm_noisy_test_tensors = [torch.tensor(dm, dtype=torch.float32).to(device) for dm in dm_noisy_test]

n_augment_levels = list(range(2))

le = LabelEncoder().fit(label_train)
label_classif_train = le.transform(label_train)
clean_label_classif_test  = le.transform(clean_label_test)
noisy_label_classif_test  = le.transform(noisy_label_test)

for n_augment in tqdm(n_augment_levels, desc="Augmentation Levels"):
    print(f"\n--- Processing n_augment = {n_augment} ---")

    # 1. Memory cleanup
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 2. Generate augmented training data and labels
    data_train_augmented, label_train_augmented = data_augmentation_by_permutations(
        original_data_train, original_label_train, n_augment, seed=42
    )
    print(f"Generated {len(data_train_augmented)} augmented training samples for n_augment={n_augment}")

    # 3. Compute distance matrices for the augmented training data
    dm_train_augmented = []
    for X in tqdm(data_train_augmented, desc="Computing augmented DM"):
        dm_train_augmented.append(distance_matrix(X))
    print(f"Computed distance matrices for augmented training data.")

    # 4. Compute Persistence Images for the augmented training data (as targets for the regression model)
    PD_train_augmented = []
    for X in tqdm(data_train_augmented, desc="Computing augmented PDs"):
        st = gd.AlphaComplex(points=X).create_simplex_tree(max_alpha_square=maxd)
        st.persistence()
        dg = st.persistence_intervals_in_dimension(1)
        if len(dg) == 0:
            dg = np.empty([0, 2])
        PD_train_augmented.append(dg)

    pds_train_augmented = DiagramSelector(use=True).fit_transform(PD_train_augmented)
    PI_train_augmented = PI.transform(pds_train_augmented)  # Use the same PI transformer
    PI_train_augmented /= MPI  # Normalize with original MPI
    PI_train_augmented_tensor = torch.tensor(PI_train_augmented, dtype=torch.float32)

    print(f"Computed Persistence Images for augmented training data.")

    # 5. Re-instantiate and train model_dm_pi (DistanceMatrixRaggedModel)
    output_dim_pi = PI_train.shape[1]
    model_dm_pi = DistanceMatrixRaggedModel(output_dim=output_dim_pi).to(device)
    optimizer_dm_pi = Adamax(model_dm_pi.parameters(), lr=5e-4)
    criterion_dm_pi = nn.MSELoss()



    # Prepare inputs for training model_dm_pi
    dm_train_augmented_tensors = [torch.tensor(dm, dtype=torch.float32, device=device) for dm in dm_train_augmented]
    PI_train_augmented_tensor_on_device = PI_train_augmented_tensor.to(device)

    _, _, _ = train_model(
        model_dm_pi, optimizer_dm_pi, criterion_dm_pi,
        dm_train_augmented_tensors, PI_train_augmented_tensor_on_device,
        dm_clean_test_tensors, clean_PI_test_tensor,
        epochs=500,  # Reduced epochs for faster execution of the loop
        batch_size=32
    )
    print(f"Trained DistanceMatrixRaggedModel for n_augment={n_augment}.")

    # 6. Use the trained model_dm_pi to predict PIs for original clean and noisy test sets
    model_dm_pi.eval()
    with torch.no_grad():
        dm_clean_PI_prediction = model_dm_pi(dm_clean_test_tensors).cpu().numpy()
        dm_noisy_PI_prediction = model_dm_pi(dm_noisy_test_tensors).cpu().numpy()
        # Also get predictions for the augmented training data for the classifier
        dm_train_PI_prediction_for_classif = model_dm_pi(dm_train_augmented_tensors).cpu().numpy()
    print(f"Made PI predictions for test and augmented training sets with n_augment={n_augment}.")

    # 7. Train an XGBClassifier
    label_classif_train_augmented = le.transform(label_train_augmented)  # Use augmented labels
    model_classif_dm_pi = XGBClassifier(eval_metric='logloss', use_label_encoder=False, random_state=42)
    model_classif_dm_pi.fit(dm_train_PI_prediction_for_classif, label_classif_train_augmented)
    print(f"Trained XGBClassifier for n_augment={n_augment}.")

    # 8. Calculate classification accuracy
    current_clean_acc = model_classif_dm_pi.score(dm_clean_PI_prediction, clean_label_classif_test)
    current_noisy_acc = model_classif_dm_pi.score(dm_noisy_PI_prediction, noisy_label_classif_test)
    clean_accuracies.append(current_clean_acc)
    noisy_accuracies.append(current_noisy_acc)
    print(f"Accuracies for n_augment={n_augment}: Clean={current_clean_acc:.4f}, Noisy={current_noisy_acc:.4f}")

    # 9. Compute permutation robustness scores for a subset
    current_clean_pr_scores = []
    for pc in tqdm(clean_data_test_subset, desc=f"Clean PR n={n_augment}"):
        score = compute_permutation_robustness_score(pc, model_dm_pi, n_augment_per_score=5, device=device, seed=42)
        current_clean_pr_scores.append(score)
    clean_permutation_robustness_scores_means.append(np.mean(current_clean_pr_scores))

    current_noisy_pr_scores = []
    for pc in tqdm(noisy_data_test_subset, desc=f"Noisy PR n={n_augment}"):
        score = compute_permutation_robustness_score(pc, model_dm_pi, n_augment_per_score=5, device=device, seed=42)
        current_noisy_pr_scores.append(score)
    noisy_permutation_robustness_scores_means.append(np.mean(current_noisy_pr_scores))
    print(f"Computed permutation robustness scores for n_augment={n_augment}.")

    # 10. Explicitly delete large temporary variables to free up memory
    del data_train_augmented, label_train_augmented, dm_train_augmented, dm_train_augmented_tensors, \
        PD_train_augmented, pds_train_augmented, PI_train_augmented, PI_train_augmented_tensor, \
        PI_train_augmented_tensor_on_device, \
        dm_clean_PI_prediction, dm_noisy_PI_prediction, dm_train_PI_prediction_for_classif, \
        model_dm_pi, optimizer_dm_pi, criterion_dm_pi, model_classif_dm_pi, \
        current_clean_pr_scores, current_noisy_pr_scores
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

print("\n--- All augmentation levels processed ---")
print(f"Clean Accuracies: {clean_accuracies}")
print(f"Noisy Accuracies: {noisy_accuracies}")
print(f"Clean Permutation Robustness Means: {clean_permutation_robustness_scores_means}")
print(f"Noisy Permutation Robustness Means: {noisy_permutation_robustness_scores_means}")

# Plotting results
plt.figure(figsize=(14, 6))

# Plot Accuracy
plt.subplot(1, 2, 1)
plt.plot(n_augment_levels, clean_accuracies, label='Clean Accuracy', marker='o', linestyle='-')
plt.plot(n_augment_levels, noisy_accuracies, label='Noisy Accuracy', marker='x', linestyle='--')
plt.title('Classification Accuracy vs. Permutation Augmentation Levels')
plt.xlabel('Number of Permutation Augmentations per Sample (n_augment)')
plt.ylabel('Accuracy')
plt.xticks(n_augment_levels[::2])  # show fewer ticks for readability
plt.ylim(0, 1)  # Accuracy is between 0 and 1
plt.legend()
plt.grid(True)

# Plot Permutation Robustness
plt.subplot(1, 2, 2)
plt.plot(n_augment_levels, clean_permutation_robustness_scores_means, label='Clean Robustness Mean', marker='o', linestyle='-')
plt.plot(n_augment_levels, noisy_permutation_robustness_scores_means, label='Noisy Robustness Mean', marker='x', linestyle='--')
plt.title('Permutation Robustness vs. Permutation Augmentation Levels')
plt.xlabel('Number of Permutation Augmentations per Sample (n_augment)')
plt.ylabel('Mean L2 Distance (Robustness Score)')
plt.xticks(n_augment_levels[::2])  # show fewer ticks for readability
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()