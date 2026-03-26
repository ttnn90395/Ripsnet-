import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adamax
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import gudhi as gd
from typing import List, Tuple
import math

# --- Global Parameters ---
N_sets_train = 900  # Number of train point clouds
N_sets_test  = 300  # Number of test  point clouds
N_points     = 600  # Point cloud cardinality
N_noise      = 200  # Number of corrupted points

# --- Device Configuration ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Data Generation Functions ---
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
    return data_train, label_train

# --- TFN Specific Utility Functions ---
def convert_2d_to_3d(pc_list):
    """
    Converts a list of 2D point clouds to 3D by adding a zero z-coordinate.
    Input: List of numpy arrays, each (N, 2).
    Output: List of numpy arrays, each (N, 3).
    """
    converted_pcs = []
    for pc_2d in pc_list:
        if pc_2d.shape[1] == 2:
            pc_3d = np.concatenate((pc_2d, np.zeros((pc_2d.shape[0], 1))), axis=1)
        else:
            pc_3d = pc_2d # Assuming it's already 3D or similar
        converted_pcs.append(pc_3d)
    return converted_pcs

def train_model_classification(model, optimizer, criterion, train_inputs, train_targets, val_inputs, val_targets, epochs=20, batch_size=32):
    """
    Train a classification model (e.g., TFN).
    Returns (best_model, history, best_model_state).
    """
    model = model.to(device)
    patience = 10
    best_val_loss = float('inf')
    patience_counter = 0
    num_epochs = epochs if epochs is not None else 10000
    history = {'train_loss': [], 'val_loss': [], 'train_accuracy': [], 'val_accuracy': []}
    best_model_state = None

    train_targets_on_device = torch.tensor(train_targets, dtype=torch.long, device=device)
    val_targets_on_device = torch.tensor(val_targets, dtype=torch.long, device=device)

    train_inputs_on_device = train_inputs # Already list of tensors on device
    val_inputs_on_device = val_inputs # Already list of tensors on device

    for epoch in range(num_epochs):
        model.train()
        permutation = torch.randperm(len(train_inputs_on_device), device=device)
        epoch_loss = 0.0
        correct_train = 0
        total_train = 0

        for i in range(0, len(train_inputs_on_device), batch_size):
            optimizer.zero_grad()
            indices = permutation[i:i + batch_size]
            batch_inputs = [train_inputs_on_device[int(idx)] for idx in indices]
            batch_targets = train_targets_on_device[indices]

            outputs = model(batch_inputs)
            loss = criterion(outputs, batch_targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(batch_inputs)

            _, predicted = torch.max(outputs.data, 1)
            total_train += batch_targets.size(0)
            correct_train += (predicted == batch_targets).sum().item()

        epoch_loss /= len(train_inputs_on_device)
        train_accuracy = 100 * correct_train / total_train
        history['train_loss'].append(epoch_loss)
        history['train_accuracy'].append(train_accuracy)

        model.eval()
        correct_val = 0
        total_val = 0
        val_loss = 0.0
        with torch.no_grad():
            val_outputs = model(val_inputs_on_device)
            val_loss = criterion(val_outputs, val_targets_on_device).item()
            
            _, predicted = torch.max(val_outputs.data, 1)
            total_val += val_targets_on_device.size(0)
            correct_val += (predicted == val_targets_on_device).sum().item()

        val_accuracy = 100 * correct_val / total_val
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_accuracy)

        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss:.6f}, Train Acc: {train_accuracy:.2f}%, Val Loss: {val_loss:.6f}, Val Acc: {val_accuracy:.2f}%')

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    if best_model_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})

    return model, history, best_model_state

# --- TFN Model Definition (TensorFieldNetwork, RBFExpansion, TFNLayer) ---
class RBFExpansion(nn.Module):
    def __init__(self, num_rbf: int = 16, cutoff: float = 5.0):
        super().__init__()
        self.num_rbf = num_rbf
        self.cutoff = cutoff
        centers = torch.linspace(0.0, cutoff, num_rbf)
        self.register_buffer("centers", centers)
        self.width = (cutoff / num_rbf) ** 2

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        diff = distances.unsqueeze(-1) - self.centers
        return torch.exp(-diff ** 2 / self.width)

class TFNLayer(nn.Module):
    def __init__(
        self,
        in_f0: int,
        in_f1: int,
        out_f0: int,
        out_f1: int,
        num_rbf: int = 16,
    ):
        super().__init__()
        self.W00 = nn.Sequential(
            nn.Linear(num_rbf, 32), nn.SiLU(),
            nn.Linear(32, in_f0 * out_f0)
        )
        self.W10 = nn.Sequential(
            nn.Linear(num_rbf, 32), nn.SiLU(),
            nn.Linear(32, in_f1 * out_f0)
        )
        self.W01 = nn.Sequential(
            nn.Linear(num_rbf, 32), nn.SiLU(),
            nn.Linear(32, in_f0 * out_f1)
        )
        self.W11 = nn.Sequential(
            nn.Linear(num_rbf, 32), nn.SiLU(),
            nn.Linear(32, in_f1 * out_f1)
        )

        self.norm0 = nn.LayerNorm(out_f0)
        self.norm1 = nn.LayerNorm(out_f1)

        self.in_f0, self.in_f1 = in_f0, in_f1
        self.out_f0, self.out_f1 = out_f0, out_f1

    def forward(
        self,
        pos: torch.Tensor,
        f0: torch.Tensor,
        f1: torch.Tensor,
        rbf: torch.Tensor,
        r_hat: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        N = pos.shape[0]

        w00 = self.W00(rbf).view(N, N, self.in_f0, self.out_f0)
        msg00 = torch.einsum("ijd,ijdo->io", f0.unsqueeze(0).expand(N, -1, -1) * mask.unsqueeze(-1), w00)

        dot = torch.einsum("ijk,jlk->ijl", r_hat, f1)
        w10 = self.W10(rbf).view(N, N, self.in_f1, self.out_f0)
        msg10 = torch.einsum("ijc,ijco->io", dot * mask.unsqueeze(-1), w10)

        new_f0 = self.norm0(msg00 + msg10)

        outer = r_hat.unsqueeze(-1) * f0.unsqueeze(0).unsqueeze(2)
        w01 = self.W01(rbf).view(N, N, self.in_f0, self.out_f1)
        msg01 = torch.einsum("ijcf,ijfg->igc", outer * mask.view(N, N, 1, 1), w01)

        w11 = self.W11(rbf).view(N, N, self.in_f1, self.out_f1)
        msg11 = torch.einsum("jkx,ijkg->igx", f1, w11 * mask.view(N, N, 1, 1))

        new_f1_raw = msg01 + msg11
        norms = new_f1_raw.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        scale = self.norm1(norms.squeeze(-1)).unsqueeze(-1)
        new_f1 = new_f1_raw / norms * scale

        return new_f0, new_f1

def _pairwise_geometry(pos: torch.Tensor, rbf_encoder: RBFExpansion):
    diff = pos.unsqueeze(1) - pos.unsqueeze(0)
    dist = diff.norm(dim=-1)
    r_hat = diff / dist.unsqueeze(-1).clamp(min=1e-8)

    rbf = rbf_encoder(dist)

    N = pos.shape[0]
    mask = ~torch.eye(N, dtype=torch.bool, device=pos.device)
    return rbf, r_hat, mask

class TensorFieldNetwork(nn.Module):
    def __init__(
        self,
        num_classes: int,
        hidden_f0: int = 64,
        hidden_f1: int = 16,
        num_layers: int = 3,
        num_rbf: int = 16,
        cutoff: float = 5.0,
        classifier_dims: List[int] = [128, 64],
    ):
        super().__init__()
        self.rbf = RBFExpansion(num_rbf=num_rbf, cutoff=cutoff)

        init_f0 = 1
        init_f1 = 1

        layers = []
        in_f0, in_f1 = init_f0, init_f1
        for _ in range(num_layers):
            layers.append(TFNLayer(in_f0, in_f1, hidden_f0, hidden_f1, num_rbf))
            in_f0, in_f1 = hidden_f0, hidden_f1
        self.layers = nn.ModuleList(layers)

        inv_dim = hidden_f0 + hidden_f1

        rho = []
        in_d = inv_dim
        for d in classifier_dims:
            rho += [nn.Linear(in_d, d), nn.ReLU()]
            in_d = d
        rho.append(nn.Linear(in_d, num_classes))
        self.rho = nn.Sequential(*rho)

    def _encode_single(self, pos: torch.Tensor) -> torch.Tensor:
        N = pos.shape[0]

        f0 = pos.norm(dim=-1, keepdim=True)
        f1 = pos.unsqueeze(1)

        rbf, r_hat, mask = _pairwise_geometry(pos, self.rbf)

        for layer in self.layers:
            f0, f1 = layer(pos, f0, f1, rbf, r_hat, mask)

        f1_norm = f1.norm(dim=-1)
        node_inv = torch.cat([f0, f1_norm], dim=-1)

        global_feat = node_inv.max(dim=0).values
        return global_feat

    def forward(self, batch: List[torch.Tensor]) -> torch.Tensor:
        if len(batch) == 0:
            device = next(self.parameters()).device
            return torch.empty(0, self.rho[-1].out_features, device=device)

        global_feats = torch.stack([self._encode_single(pc) for pc in batch], dim=0)
        return self.rho(global_feats)

# --- Main Execution Block ---

# 1. Data Preparation
print("Generating data...")
data_train,      label_train       = create_multiple_circles(N_sets_train, N_points, noisy=0, N_noise=N_noise)
clean_data_test, clean_label_test  = create_multiple_circles(N_sets_test,  N_points, noisy=0, N_noise=N_noise)
noisy_data_test, noisy_label_test  = create_multiple_circles(N_sets_test,  N_points, noisy=1, N_noise=N_noise)

le = LabelEncoder().fit(label_train)
label_classif_train = le.transform(label_train)
clean_label_classif_test  = le.transform(clean_label_test)
noisy_label_classif_test  = le.transform(noisy_label_test)

print("Data generated and labels cleaned.")

tf_data_train_3d = convert_2d_to_3d(data_train)
tf_clean_data_test_3d = convert_2d_to_3d(clean_data_test)
tf_noisy_data_test_3d = convert_2d_to_3d(noisy_data_test)

tf_data_train_3d_tensor = [torch.tensor(pc, dtype=torch.float32).to(device) for pc in tf_data_train_3d]
tf_clean_data_test_3d_tensor = [torch.tensor(pc, dtype=torch.float32).to(device) for pc in tf_clean_data_test_3d]
tf_noisy_data_test_3d_tensor = [torch.tensor(pc, dtype=torch.float32).to(device) for pc in tf_noisy_data_test_3d]

print("Data prepared in 3D tensor format for TFN.")

# 2. Model Initialization
num_classes_tfn = len(np.unique(label_classif_train))
model_tfn = TensorFieldNetwork(num_classes=num_classes_tfn)
model_tfn.to(device)

criterion_tfn = nn.CrossEntropyLoss()
optimizer_tfn = Adamax(model_tfn.parameters(), lr=5e-4)

# Define num_epochs if not already defined (e.g., from a global context)
num_epochs = 10000 # Default for standalone execution

print(f"TensorFieldNetwork initialized with {num_classes_tfn} classes. Training for {num_epochs} epochs.")

# 3. Model Training
model_tfn, history_tfn, best_model_state_tfn = train_model_classification(
    model_tfn, optimizer_tfn, criterion_tfn,
    tf_data_train_3d_tensor, label_classif_train,
    tf_clean_data_test_3d_tensor, clean_label_classif_test,
    epochs=num_epochs, batch_size=32
)

best_val_accuracy_tfn = max(history_tfn['val_accuracy']) if history_tfn.get('val_accuracy') else 0.0
print(f"Training TensorFieldNetwork complete. Best val_accuracy = {best_val_accuracy_tfn:.2f}%")

# 4. Model Evaluation
model_tfn.eval()

with torch.no_grad():
    outputs_clean = model_tfn(tf_clean_data_test_3d_tensor)
    _, predicted_clean = torch.max(outputs_clean.data, 1)
    predicted_clean = predicted_clean.cpu().numpy()
    
    outputs_noisy = model_tfn(tf_noisy_data_test_3d_tensor)
    _, predicted_noisy = torch.max(outputs_noisy.data, 1)
    predicted_noisy = predicted_noisy.cpu().numpy()

clean_test_acc_tfn = accuracy_score(clean_label_classif_test, predicted_clean)
noisy_test_acc_tfn = accuracy_score(noisy_label_classif_test, predicted_noisy)

print(f"TensorFieldNetwork Accuracy on clean test set: {clean_test_acc_tfn:.4f}")
print(f"TensorFieldNetwork Accuracy on noisy test set: {noisy_test_acc_tfn:.4f}")